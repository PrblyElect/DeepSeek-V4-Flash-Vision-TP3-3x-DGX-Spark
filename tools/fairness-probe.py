#!/usr/bin/env python3
"""fairness-probe.py — the admission-fairness instrument.

Measures what an interactive request experiences while deep prefills
hold the engine: the admission-starvation axis the chunk threshold and
the deferral-cadence patch do not cover (they protect streams that are
already running; this measures requests that are still WAITING).

Shape: idle chat TTFT (3 samples) -> two fresh-nonce bulk prefills fill
the partial-prefill cap -> mid-flight, a small chat request and a
parallel decode stream fire. Reports:

  - idle chat TTFT (the baseline)
  - chat TTFT mid-bulk        (THE number: the admission axis)
  - worst inter-token gap on the parallel decode stream
  - each bulk's TTFT + wall + needle-found (correctness: the needle
    must survive any chunk-boundary preemption + resume)

Expectations on the patched stack (both patches applied, priority
policy on): chat mid-bulk in the single-digit-seconds class — bounded
below by the chunk already in flight when the request arrives, not by
the queue — worst decode gap unchanged from the ~1 s class, needles
found. A minutes-class chat TTFT means admission is broken or the
admission patch is off / the policy flag is missing (check
DSPARK_INTERACTIVE_ADMISSION and --scheduling-policy priority).

Methodology (docs/VALIDATION.md): idle-gated start (the probe refuses
to run unless running == waiting == 0); fresh nonce at corpus position
~1 so repeat runs cannot prefix-cache-hit; the judged token is a short
dictionary-style codeword (echo-able at temperature 0), never the
nonce; `--selftest` checks the corpus construction offline before any
live call.

usage: fairness-probe.py [ENDPOINT] [LABEL] [BULK_TOKENS] [MODEL]
       e.g. fairness-probe.py http://127.0.0.1:8000/v1 admission 200000

Only talks to the endpoint you point it at (default loopback); redirects
refused. Extend ALLOWED_HOSTS if you probe a remote head node.
"""
import json
import random
import re
import sys
import threading
import time
import urllib.parse
import urllib.request

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

def validate(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or parts.hostname not in ALLOWED_HOSTS:
        raise SystemExit(f"refusing non-local URL: {url} (extend ALLOWED_HOSTS deliberately)")
    return url

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(newurl, code, "redirect refused", headers, fp)

_opener = urllib.request.build_opener(_NoRedirect)

_SELFTEST = "--selftest" in sys.argv[1:]
_args = [a for a in sys.argv[1:] if a != "--selftest"]
EP = validate(_args[0] if _args else "http://127.0.0.1:8000/v1")
LABEL = _args[1] if len(_args) > 1 else "run"
BULK_TOKENS = int(_args[2]) if len(_args) > 2 else 200000
MODEL = _args[3] if len(_args) > 3 else "ds4flash-vision"
RESULTS = {}

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu").split()
CODEWORDS = ["harbor", "lantern", "pebble", "orchard", "cinder", "juniper",
             "meadow", "thistle", "cobalt", "willow"]


def chat(messages, max_tokens=32, stream=False, timeout=900):
    body = json.dumps({
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.0, "stream": stream,
        # stacks that default thinking/reasoning ON eat small budgets with
        # reasoning content — disable it for the tiny probe requests
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(EP + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with _opener.open(req, timeout=timeout) as r:
        if not stream:
            out = json.load(r)
            return time.monotonic() - t0, out["choices"][0]["message"].get("content", "")
        first, gaps, last, text = None, [], None, []
        for line in r:
            if not line.startswith(b"data: "):
                continue
            now = time.monotonic()
            if first is None:
                first = now - t0
            elif last is not None:
                gaps.append(now - last)
            last = now
            payload = line[6:].strip()
            if payload and payload != b"[DONE]":
                try:
                    d = json.loads(payload)
                    delta = d["choices"][0].get("delta", {}).get("content")
                    if delta:
                        text.append(delta)
                except Exception:
                    pass
        return first, gaps, "".join(text)


def bulk_corpus(idx):
    """Nonce at position ~1 (cache-bust, never judged); a short
    dictionary-style codeword at ~72% depth is the judged needle."""
    rng = random.Random()
    nonce = f"ZQ{LABEL}{idx}{int(time.time() * 1000) % 10 ** 9:09d}"
    codeword = f"{rng.choice(CODEWORDS)}-{rng.choice(CODEWORDS)}-{rng.randint(10, 99)}"
    needle_pos = int(BULK_TOKENS * 0.72)
    parts = [f"corpus {nonce} begins."]
    n = 6
    while n < BULK_TOKENS:
        sent = " ".join(rng.choice(WORDS) for _ in range(14))
        if n <= needle_pos < n + 15:
            sent += f" the magic passport is {codeword} "
        parts.append(sent)
        n += 15
    filler = " ".join(parts)
    return [
        {"role": "system", "content": "You transcribe exactly. "
         "Reply with ONLY the magic passport value hidden in the text "
         "(it reads 'the magic passport is <value>' followed by nothing "
         "else you should output)."},
        {"role": "user", "content": filler},
    ], codeword


def selftest():
    corpus, codeword = bulk_corpus(0)
    text = corpus[1]["content"]
    assert codeword in text and text.count(f"the magic passport is {codeword}") == 1, \
        "codeword must appear exactly once"
    first_sent = text.split(".")[0]
    assert "ZQ" in first_sent and codeword not in first_sent, \
        "nonce belongs at corpus position ~1, the codeword at depth"
    assert len(codeword.split("-")[-1]) <= 2, "judged tail stays echo-able"
    print("selftest OK: nonce at position ~1, one codeword at ~72% depth, "
          f"judged token {codeword!r}")
    raise SystemExit(0)


def metrics_url():
    return EP.rsplit("/v1", 1)[0] + "/metrics"


def idle_gate():
    for _ in range(120):
        with _opener.open(metrics_url(), timeout=6) as r:
            m = r.read().decode()
        run = re.search(r'num_requests_running\{[^}]*\} (\d+)', m)
        wait = re.search(r'num_requests_waiting\{[^}]*\} (\d+)', m)
        if run and wait and run.group(1) == "0" and wait.group(1) == "0":
            return True
        time.sleep(5)
    return False


def main():
    print(f"[{LABEL}] idle gate...", flush=True)
    assert idle_gate(), "engine not idle — aborting (idle-gate rule, docs/VALIDATION.md §2)"
    ttfts = []
    for i in range(3):
        dt, _ = chat([{"role": "user", "content": f"Say the word 'ready' and nothing else. ({i})"}], 8)
        ttfts.append(dt)
    RESULTS["idle_chat_ttft_s"] = ttfts
    print(f"[{LABEL}] idle chat TTFT: {[round(t, 2) for t in ttfts]}s", flush=True)
    time.sleep(2)

    bulks = []
    def run_bulk(i):
        msgs, codeword = bulk_corpus(i)
        t0 = time.monotonic()
        ttft, text = chat(msgs, 64, stream=False)
        wall = time.monotonic() - t0
        bulks.append({"idx": i, "ttft_s": ttft, "wall_s": wall,
                      "found": codeword in text, "reply": text[:80]})
        print(f"[{LABEL}] bulk{i}: wall={wall:.1f}s reply={text[:60]!r}", flush=True)

    threads = [threading.Thread(target=run_bulk, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    time.sleep(12)  # both bulks admitted and chunking by now

    chat_res = {}
    def fire_chat():
        t0 = time.monotonic()
        dt, reply = chat([{"role": "user", "content": "Reply with exactly: ACK"}], 8, timeout=1800)
        chat_res["ttft"] = dt
        chat_res["reply"] = reply
        print(f"[{LABEL}] CHAT MID-BULK TTFT: {dt:.2f}s (reply={reply[:20]!r})", flush=True)
    chat_t = threading.Thread(target=fire_chat)
    chat_t.start()
    time.sleep(0.3)

    decode_first, gaps, _ = chat(
        [{"role": "user", "content": "Count slowly from one to sixty, one number per line."}], 220,
        stream=True, timeout=1200)
    RESULTS["decode_stream_first_token_s"] = decode_first
    RESULTS["decode_worst_gap_s"] = max(gaps) if gaps else None
    print(f"[{LABEL}] decode stream: first={decode_first:.2f}s worst_gap="
          f"{RESULTS['decode_worst_gap_s']:.2f}s", flush=True)
    chat_t.join()
    RESULTS["chat_midbulk_ttft_s"] = chat_res.get("ttft")

    for t in threads:
        t.join()
    RESULTS["bulks"] = bulks
    print(f"\n[{LABEL}] SUMMARY: idle={[round(t, 2) for t in ttfts]} "
          f"chat_midbulk={RESULTS['chat_midbulk_ttft_s']:.2f}s decode_worst_gap="
          f"{RESULTS['decode_worst_gap_s']:.2f}s "
          f"needles={'/'.join('OK' if b['found'] else 'MISS' for b in bulks)} "
          f"bulk_walls={[round(b['wall_s'], 1) for b in bulks]}", flush=True)


if __name__ == "__main__":
    if _SELFTEST:
        selftest()
    main()
