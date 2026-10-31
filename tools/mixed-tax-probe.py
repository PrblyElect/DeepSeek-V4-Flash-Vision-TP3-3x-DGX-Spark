#!/usr/bin/env python3
"""Mixed-tax probe: decode behavior beside a deep cold prefill (the
deferral-cadence instrument). Point it at your engine (run it on the
head node):

  MIXED_TAX_LANE=http://127.0.0.1:8000 MIXED_TAX_MODEL=ds4flash-vision \
      python3 tools/mixed-tax-probe.py

  A. solo decode TPOT distribution  (prose class, spec-decode on, x REPS)
  B. solo deep cold-prefill rate    (~440K fresh-nonce tokens)
  C. concurrent, x REPS: the SAME decode stream measured WHILE the
     same-depth cold prefill runs. Reports decode TPOT mean/p50/p95/max
     inside the prefill window, the worst single gap, post-window
     recovery, and the prefill-side mixed rate.

Read it against the deferral knob (DSPARK_PREFILL_DECODE_INTERVAL):
without it, this stack measures ~12x mean TPOT inflation with
multi-second worst gaps; with it, ~3x with worst gaps in the ~1 s class
and p50 near solo speed. Anti-contamination: every deep prefill carries
a fresh nonce (the prefix cache can never replay a measurement) and the
engine must be idle-gated before each stage (VALIDATION.md §2). Stdlib
only; loopback-only endpoints, no redirects; bounded runtime.

The probe is a GATE: it exits non-zero when the deferral knob is off (or
was not applied), when a stage failed, or when the measured inflation
crosses the bounds below. The bounds sit between the healthy class (~3x
mean, ~1 s worst gap) and the broken class (~12x mean, multi-second
gaps), so a healthy lane is not failed by noise.

Exit codes:
    0  PASS  the mixed tax stayed inside its bounds
    1  FAIL  the knob is off, a stage failed, or a bound was crossed
    2  USAGE bad arguments
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

ENGINE = os.environ.get("MIXED_TAX_LANE", "http://127.0.0.1:8000")
MODEL = os.environ.get("MIXED_TAX_MODEL", "ds4flash-vision")
ROWS_DEEP = 20000   # ~440K tokens (~22 tok/row with this filler)
ROWS_WARM = 6000    # ~130K sacrificial deep warmup (first-run JIT insurance)
DECODE_PROMPT = ("Write a clear, detailed essay about the history of "
                 "lighthouses and their keepers.")
DECODE_MAX = 640
REPS = 2


ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

def _validate(url):
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or p.hostname not in ALLOWED_HOSTS:
        raise SystemExit(f"refusing non-local URL: {url} (extend ALLOWED_HOSTS deliberately)")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


class Stream:
    """One streaming completion. Records content-chunk timestamps, first
    content token, and usage from the final chunk (include_usage)."""

    def __init__(self, content, max_tokens, temp=0.7):
        body = {"model": MODEL,
                "messages": [{"role": "user", "content": content}],
                "temperature": temp, "max_tokens": max_tokens,
                "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"thinking": False}}
        self._data = json.dumps(body).encode()
        self.timestamps = []
        self.usage = {}
        self.first = None
        self.err = None
        self.t0 = None
        self.wall = None

    def run(self):
        req = urllib.request.Request(_validate(ENGINE + "/v1/chat/completions"),
                                     data=self._data,
                                     headers={"Content-Type": "application/json"})
        self.t0 = time.time()
        try:
            with _opener.open(req, timeout=420) as r:
                for line in r:
                    if not line.startswith(b"data: ") or b"[DONE]" in line:
                        continue
                    try:
                        d = json.loads(line[6:])
                    except ValueError:
                        continue
                    if d.get("usage"):
                        self.usage = d["usage"]
                    if any(c.get("delta", {}).get("content") for c in d.get("choices", [])):
                        if self.first is None:
                            self.first = time.time()
                        self.timestamps.append(time.time())
            self.wall = time.time() - self.t0
        except Exception as e:  # noqa: BLE001 - probe must survive and report
            self.err = repr(e)
            self.wall = time.time() - self.t0
        return self


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def stats(ts):
    gaps = [(b - a) * 1000.0 for a, b in zip(ts, ts[1:])]  # ms
    if not gaps:
        return {"n": len(ts), "mean_ms": float("nan")}
    return {"n": len(ts), "mean_ms": sum(gaps) / len(gaps),
            "p50_ms": pct(gaps, 0.50), "p95_ms": pct(gaps, 0.95),
            "max_ms": max(gaps)}


def fmt(s):
    return ("{n} chunks mean={mean_ms:.0f}ms p50={p50_ms:.0f}ms "
            "p95={p95_ms:.0f}ms max={max_ms:.0f}ms").format(**s) \
        if "p95_ms" in s else "{n} chunks (no gaps)".format(**s)


def deep_prompt(tag, rows):
    nonce = f"mtax-{tag}-{int(time.time() * 1000)}-{rows}"
    ctx = " ".join(f"{nonce}-fact-{i} unique cold context row" for i in range(rows))
    return f"CONTEXT: {ctx}\n\nReply exactly: OK"


def solo_decode(rep):
    s = Stream(DECODE_PROMPT, 320).run()
    if s.err:
        print(f"A{rep} solo-decode ERROR {s.err}", flush=True)
        return None
    st = stats(s.timestamps)
    rate = (len(s.timestamps) - 1) / (s.timestamps[-1] - s.timestamps[0]) \
        if len(s.timestamps) > 1 else float("nan")
    print(f"A{rep} solo-decode : {fmt(st)}  rate={rate:.1f} tok/s "
          f"wall={s.wall:.1f}s", flush=True)
    return st


def solo_prefill(tag, rows):
    s = Stream(deep_prompt(tag, rows), 8).run()
    if s.err:
        print(f"B {tag} solo-prefill ERROR {s.err}", flush=True)
        return None
    pt = s.usage.get("prompt_tokens")
    ttft = s.first - s.t0 if s.first else s.wall
    rate = (pt / ttft) if pt and ttft else float("nan")
    print(f"B {tag} solo-prefill: ttft={ttft:.1f}s prompt={pt} "
          f"rate={rate:.0f} tok/s", flush=True)
    return {"ttft": ttft, "prompt": pt, "rate": rate}


def concurrent(rep):
    dec = Stream(DECODE_PROMPT, DECODE_MAX)

    def run_decode():
        dec.run()

    th = threading.Thread(target=run_decode)
    th.start()
    time.sleep(2.0)          # let the decode stream spin up
    pre = Stream(deep_prompt(f"c{rep}", ROWS_DEEP), 8)
    pre.run()                # deep cold prefill fires while decode streams
    th.join(timeout=420)

    if dec.err or pre.err:
        print(f"C{rep} ERROR decode={dec.err} prefill={pre.err}", flush=True)
        return None
    pttft = pre.first - pre.t0 if pre.first else pre.wall
    pt = pre.usage.get("prompt_tokens")
    # window = the prefill's own lifetime
    w0, w1 = pre.t0, (pre.first if pre.first else pre.t0 + pre.wall)
    ts = dec.timestamps
    dur = [(b - a) * 1000.0 for a, b in zip(ts, ts[1:]) if w0 <= b <= w1]
    aft = [(b - a) * 1000.0 for a, b in zip(ts, ts[1:]) if b > w1]
    st_all = stats(ts)
    st_dur = {"n": len(dur), "mean_ms": sum(dur) / len(dur) if dur else float("nan"),
              "p50_ms": pct(dur, 0.50), "p95_ms": pct(dur, 0.95),
              "max_ms": max(dur) if dur else float("nan")}
    print(f"C{rep} DURING-window : {fmt(st_dur)}  "
          f"window={w1 - w0:.1f}s", flush=True)
    print(f"C{rep} worst gap     : {st_dur['max_ms']:.0f} ms", flush=True)
    if aft:
        print(f"C{rep} after-window : n={len(aft)} mean={sum(aft) / len(aft):.0f}ms", flush=True)
    print(f"C{rep} prefill(mixed): ttft={pttft:.1f}s prompt={pt} "
          f"rate={(pt / pttft) if pt else float('nan'):.0f} tok/s  "
          f"decode_wall={dec.wall:.0f}s", flush=True)
    return {"during": st_dur, "all": st_all, "pttft": pttft, "pt": pt}


def main() -> int:
    global ENGINE, MODEL
    ap = argparse.ArgumentParser(
        prog="mixed-tax-probe.py",
        description="Measure decode behaviour beside a deep cold prefill (the "
                    "deferral-cadence instrument).",
    )
    ap.add_argument("--lane", default=os.environ.get("MIXED_TAX_LANE",
                                                     "http://127.0.0.1:8000"),
                    help="the engine's base URL (or MIXED_TAX_LANE)")
    ap.add_argument("--model", default=os.environ.get("MIXED_TAX_MODEL",
                                                      "ds4flash-vision"),
                    help="the served model name (or MIXED_TAX_MODEL)")
    ap.add_argument("--max-mean-ratio", type=float, default=8.0,
                    help="fail if the during-window mean TPOT is more than this "
                         "many times the solo mean (default: 8 — healthy is ~3x, "
                         "the knob-off state is ~12x)")
    ap.add_argument("--max-p50-ratio", type=float, default=6.0,
                    help="fail if the during-window median TPOT is more than this "
                         "many times the solo mean (default: 6 — healthy is near 1x)")
    ap.add_argument("--max-worst-gap-ms", type=float, default=5000.0,
                    help="fail if the worst during-window gap exceeds this many "
                         "milliseconds (default: 5000 — healthy is the ~1 s class)")
    args = ap.parse_args()

    try:
        ENGINE = _validate(args.lane)
    except SystemExit as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    MODEL = args.model

    print("=== mixed-tax probe (decode beside a deep cold prefill) ===", flush=True)
    print(f"warmup: sacrificial ~{ROWS_WARM * 22 / 1000:.0f}K cold prefill", flush=True)
    Stream(deep_prompt("warm", ROWS_WARM), 8).run()

    solo_dec = [s for s in (solo_decode(r + 1) for r in range(REPS)) if s]
    solo_pre = solo_prefill("solo", ROWS_DEEP)
    conc = [c for c in (concurrent(r + 1) for r in range(REPS)) if c]

    print("=== summary ===", flush=True)
    failures: list[str] = []
    if len(solo_dec) < REPS:
        failures.append(f"only {len(solo_dec)}/{REPS} solo decode runs completed")
    if solo_pre is None:
        failures.append("the solo deep prefill failed")
    if len(conc) < REPS:
        failures.append(f"only {len(conc)}/{REPS} concurrent runs completed")

    if solo_dec and conc:
        base_mean = min(s["mean_ms"] for s in solo_dec)
        for i, c in enumerate(conc, 1):
            d = c["during"]
            if not (d and d["n"]):
                failures.append(f"rep{i}: no decode chunks landed inside the "
                                "prefill window — nothing was measured")
                continue
            mean_ratio = d["mean_ms"] / base_mean
            p50_ratio = d["p50_ms"] / base_mean
            print(f"rep{i}: decode mean ratio = {mean_ratio:.1f}x "
                  f"p50 ratio = {p50_ratio:.1f}x "
                  f"(solo mean {base_mean:.0f}ms -> during {d['mean_ms']:.0f}ms; "
                  f"worst {d['max_ms']:.0f}ms)", flush=True)
            if mean_ratio > args.max_mean_ratio:
                failures.append(f"rep{i}: the during-window mean TPOT is "
                                f"{mean_ratio:.1f}x the solo mean (bound "
                                f"{args.max_mean_ratio:.1f}x) — the deferral knob "
                                "is off or was not applied")
            if p50_ratio > args.max_p50_ratio:
                failures.append(f"rep{i}: the during-window median TPOT is "
                                f"{p50_ratio:.1f}x the solo mean (bound "
                                f"{args.max_p50_ratio:.1f}x)")
            if d["max_ms"] > args.max_worst_gap_ms:
                failures.append(f"rep{i}: the worst during-window gap is "
                                f"{d['max_ms']:.0f}ms (bound "
                                f"{args.max_worst_gap_ms:.0f}ms)")
        if solo_pre and solo_pre["rate"] and conc:
            mixed = [c["pt"] / c["pttft"] for c in conc if c["pt"]]
            if mixed:
                print(f"prefill mixed tax: solo {solo_pre['rate']:.0f} -> "
                      f"mixed {sum(mixed) / len(mixed):.0f} tok/s "
                      f"({100 * (1 - sum(mixed) / len(mixed) / solo_pre['rate']):.0f}% tax)",
                      flush=True)

    print()
    if failures:
        print("MIXED-TAX PROBE FAIL:")
        for msg in failures:
            print(f"  - {msg}")
        return EXIT_FAIL
    print("MIXED-TAX PROBE PASS — the mixed tax stayed inside its bounds.")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
