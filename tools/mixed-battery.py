#!/usr/bin/env python3
"""Mixed-surface acceptance battery (the lane-health gate). Stdlib-only.

A boot is not "accepted" because single-class probes pass. This battery
runs, AT THE SAME TIME: interactive coding-class streams, chat-class
requests, and image requests in flight — plus (when BATTERY_RERANK /
BATTERY_EMBED point at YOUR co-tenant services) a reranker and an embedder
hitting their production endpoints, with host MemAvailable sampled
throughout. A lane that only survives isolated benchmarks is not validated
for the shared-box operating mode this recipe targets.

Point the tenant endpoints at whatever small services share the boxes with
the lane (any OpenAI-style /v1/rerank and /v1/embeddings service); leaving
them unset skips those legs with a note — but the acceptance gate this
recipe publishes was run with the co-tenants LIVE, and a tenants-skipped
run is a diagnostic, not a gate.

Gates: 0 errors; interactive streams >= floor (BATTERY_FLOOR, tok/s);
chat healthy; vision answers name the image's two colors; reranker ranks;
embedder returns vectors.

Only talks to endpoints you configure (lane defaults to loopback);
redirects refused.
"""
import base64
import json
import os
import statistics
import struct
import threading
import time
import urllib.parse
import urllib.request
import zlib

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _validate(url: str) -> str:
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or parts.hostname not in ALLOWED_HOSTS:
        raise SystemExit(f"refusing non-local URL: {url} (extend ALLOWED_HOSTS for your own endpoints)")
    return url


LANE = _validate(os.environ.get("BATTERY_LANE", "http://127.0.0.1:8000")) + "/v1/chat/completions"
RERANK = os.environ.get("BATTERY_RERANK", "")   # your reranker's /v1/rerank endpoint URL
EMBED = os.environ.get("BATTERY_EMBED", "")     # your embedder's /v1/embeddings endpoint URL
EMBED_MODEL = os.environ.get("BATTERY_EMBED_MODEL", "")
MODEL = os.environ.get("BATTERY_MODEL", "ds4flash-vision")
INTERACTIVE_FLOOR = float(os.environ.get("BATTERY_FLOOR", "12.0"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def post(url, body, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=timeout) as r:
        return json.load(r)


def gen_stream(prompt, max_tokens):
    body = json.dumps({"model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False}}).encode()
    t0 = time.time(); ttft = None; ntok = 0
    req = urllib.request.Request(LANE, data=body, headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=240) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue  # keepalive/comments between SSE frames
            if d.get("usage"):
                ntok = d["usage"].get("completion_tokens", ntok)
            for ch in d.get("choices", []):
                if ch.get("delta", {}).get("content") and ttft is None:
                    ttft = time.time() - t0
    dt = time.time() - t0
    return (ntok / dt if dt > 0 else 0.0), ttft, 0


def split_png_b64(left_rgb, right_rgb, w=64, h=64):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(left_rgb * (w // 2) + right_rgb * (w // 2)) for _ in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


interactive_rates, chat_rates, vision_ok, errors = [], [], [], []
lock = threading.Lock()

def interactive(i):
    for r in range(4):
        try:
            rate, _, _ = gen_stream(f"Editing task {i}-{r}: refactor this loop to use itertools and explain the change in one paragraph. The loop iterates rows and accumulates totals for report {r}.", 256)
            with lock: interactive_rates.append(rate)
        except Exception as e:
            with lock: errors.append(f"interactive {i}-{r}: {e}")

def chat(i):
    for r in range(6):
        try:
            rate, _, _ = gen_stream(f"Chat message {i}-{r}: what are three quick tips for keeping houseplants alive?", 96)
            with lock: chat_rates.append(rate)
        except Exception as e:
            with lock: errors.append(f"chat {i}-{r}: {e}")

def vision(i):
    for r in range(3):
        try:
            d = post(LANE, {"model": MODEL, "temperature": 0.0, "max_tokens": 120,
                "chat_template_kwargs": {"thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"Round {r}: name the left color and the right color of this image, one short sentence."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + split_png_b64((255, 0, 0), (0, 0, 255))}}]}]},
                timeout=240)
            a = ((d.get("choices") or [{}])[0].get("message", {}).get("content") or "").lower()
            with lock: vision_ok.append("red" in a and "blue" in a)
        except Exception as e:
            with lock: errors.append(f"vision {i}-{r}: {e}")

mem_mins = [999999]
stop = threading.Event()
def mem_sampler():
    while not stop.is_set():
        try:
            with open("/proc/meminfo") as mi:
                for l in mi:
                    if l.startswith("MemAvailable"):
                        mem_mins[0] = min(mem_mins[0], int(l.split()[1]) // 1024)
                        break
        except OSError:
            pass  # not Linux / no procfs — sample nothing
        time.sleep(2)

m = threading.Thread(target=mem_sampler); m.start()
threads = [threading.Thread(target=interactive, args=(i,)) for i in range(2)] + \
          [threading.Thread(target=chat, args=(i,)) for i in range(8)] + \
          [threading.Thread(target=vision, args=(i,)) for i in range(2)]
t0 = time.time()
for t in threads: t.start()

time.sleep(3)
rr_ok = em_ok = True
rr_skipped = em_skipped = False
if RERANK:
    for r in range(3):
        try:
            d = post(_validate(RERANK), {"query": "statute of limitations for a contract breach claim",
                "documents": ["The limitations period for breach of contract is six years from accrual of the claim.",
                              "Bananas are rich in potassium and make a smooth snack."]}, timeout=60)
            if d["results"][0]["index"] != 0: rr_ok = False
        except Exception as e:
            rr_ok = False; errors.append(f"rerank {r}: {e}")
else:
    rr_skipped = True
if EMBED:
    for r in range(3):
        try:
            body = {"input": f"legal clause {r} about inventory"}
            if EMBED_MODEL:
                body["model"] = EMBED_MODEL
            d = post(_validate(EMBED), body, timeout=60)
            vec = ((d.get("data") or [{}])[0].get("embedding"))
            if not isinstance(vec, list) or len(vec) < 64: em_ok = False
        except Exception as e:
            em_ok = False; errors.append(f"embed {r}: {e}")
else:
    em_skipped = True

for t in threads: t.join()
stop.set(); m.join()
wall = time.time() - t0

results = [
    ("interactive-floor", all(r >= INTERACTIVE_FLOOR for r in interactive_rates) and len(interactive_rates) == 8,
     f"samples={len(interactive_rates)} min={min(interactive_rates):.1f} mean={statistics.mean(interactive_rates):.1f} (floor {INTERACTIVE_FLOOR})"),
    ("chat-healthy", len(chat_rates) == 48 and statistics.mean(chat_rates) >= 3.0,
     f"samples={len(chat_rates)} mean={statistics.mean(chat_rates):.1f}"),
    ("vision-in-flight", len(vision_ok) == 6 and all(vision_ok),
     f"samples={len(vision_ok)} correct={sum(vision_ok)} (image answers must name both colors)"),
    ("reranker-in-flight", rr_ok or rr_skipped,
     "3 calls ranked correctly" if not rr_skipped else "skipped (BATTERY_RERANK unset — diagnostic run, not the full gate)"),
    ("embed-in-flight", em_ok or em_skipped,
     "3 calls returned vectors" if not em_skipped else "skipped (BATTERY_EMBED unset — diagnostic run, not the full gate)"),
    ("zero-errors", not errors, f"{len(errors)} errors" + (f": {errors[:3]}" if errors else "")),
]
ok = all(r[1] for r in results)
for name, passed_, detail in results:
    print(f"{'PASS' if passed_ else 'FAIL'} {name}: {detail}")
if mem_mins[0] < 999999:
    print(f"memory: min MemAvailable on this box during battery = {mem_mins[0]} MiB")
print(f"wall {wall:.0f}s — MIXED BATTERY {'PASS' if ok else 'FAILURE'}")
