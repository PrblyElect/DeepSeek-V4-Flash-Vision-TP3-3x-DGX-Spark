#!/usr/bin/env python3
"""Instructed-recall needle ladder + vision fence probe. Stdlib-only.

The recall probe this recipe standardizes on. Three hard-won rules are
baked in (see docs/VALIDATION.md and the FIXLOG's instrument rungs):

1. INSTRUCTED, always, for correctness gates. An instruction-less needle
   (filler + "NOTE: the magic word is X." and nothing else) measures the
   model's temperature-0 continuation style, not recall: greedy decoding
   sits on a near-tie between echoing the filler and reciting the word,
   and which style wins flips with the nonce and with cache/kernel
   numerics. We measured uninstructed misses at depth coexisting with
   6/6 instructed passes on the same boot, at the same depths, minutes
   apart. The legacy uninstructed corpus is kept ONLY behind --legacy for
   reproducing historical runs, and is labeled as such.
2. Cold-corpus nonce: the nonce token at position ~1 makes every corpus
   differ from every other from the first block on, so repeat runs at the
   same depth cannot silently prefix-cache-hit.
3. Echo-able judged word. The judged token must be SHORT (a
   dictionary-style word plus a couple of digits). A judged "word" built
   from a long random nonce — the natural instinct for uniqueness, and
   exactly what a timestamp-nonce monitor produces — fails at the
   temperature-0 copy margin: the model attends the needle and recalls
   the stem (often the digits too) but truncates the random tail, and a
   strict word-exact judge then reports context loss that never happened.
   Days of phantom "deep-prefill corruption" here traced to precisely
   that; short words recalled word-exact at every depth tested to ~950K.
   The nonce busts the cache from corpus position 1; it never appears
   inside the judged token. --salad-word preserves the old construction
   for reproducing historical runs.

Also measures per-depth prefill tok/s (read usage.prompt_tokens — the
--depths target is a TOKEN ESTIMATE and the filler overshoots it by ~1.6x,
so never compare "actual" token counts across harnesses), and runs the
vision fence: single-image TTFT plus the default 8-image-per-request cap
(the engine's mm-limit flag, not a model limit — rung 37) at TP=3 (eight
color-split images, all eight left colors named in order).

--selftest runs the offline corpus assertions (plus one tiny live
instructed needle when the lane answers) — run it before every
measurement session; both harness bugs this recipe's history records
shipped before a selftest existed.

Only talks to the endpoint you point it at (default loopback); redirects
refused. Extend ALLOWED_HOSTS if you probe a remote head node.
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
import urllib.parse
import urllib.request
import zlib

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate(url: str) -> str:
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or parts.hostname not in ALLOWED_HOSTS:
        raise SystemExit(f"refusing non-local URL: {url} (extend ALLOWED_HOSTS deliberately)")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def chat(lane: str, model: str, content, max_tokens: int = 60, timeout: int = 2400):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "temperature": 0.0, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(f"{lane}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with _opener.open(req, timeout=timeout) as res:
        d = json.load(res)
    wall = time.time() - t0
    txt = d["choices"][0]["message"]["content"] or ""
    pt = d.get("usage", {}).get("prompt_tokens", 0)
    return txt, pt, wall


def needle(target_tokens: int, word: str, instruct: bool = True, nonce: str = "") -> str:
    filler = ("The committee reviewed the ledger and confirmed the previously "
              "recorded figures for the quarter, noting no further action. ")
    unit = len(filler.split())
    reps = max(1, target_tokens // max(1, int(unit * 0.75)))
    half = reps // 2
    corpus = (" ".join([filler] * half)
              + f" NOTE: the magic word is {word}. "
              + " ".join([filler] * (reps - half)))
    if nonce:
        # Cold-cache by construction: the nonce token at position ~1 makes
        # every corpus differ from every other from the first block on (a
        # mid-corpus nonce only differentiates the corpus at its midpoint,
        # so repeat runs at the same depth silently prefix-cache-hit).
        corpus = f"[{nonce}] " + corpus
    if instruct:
        corpus += ("\n\nINSTRUCTION: What is the magic word stated in the NOTE "
                   "above? Answer with ONLY the magic word, nothing else.")
    return corpus


def png_split(l, r, w=64, h=64) -> str:
    def ch(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(l * (w // 2) + r * (w // 2)) for _ in range(h))
    return base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="ds4flash-vision")
    ap.add_argument("--depths", default="131072,262144,524288",
                    help="token ESTIMATES; the corpus overshoots ~1.6x — read the reported prompt_tokens")
    ap.add_argument("--nonce", default="", help="distinguishes corpora so repeat runs miss the prefix cache")
    ap.add_argument("--salad-word", action="store_true",
                    help="legacy judged word: model-token + full nonce + doubled suffix "
                         "(up to ~60 random chars). Echo-marginal at temperature 0 — "
                         "fails as partial recall, not context loss. Only for "
                         "reproducing historical runs.")
    ap.add_argument("--legacy", action="store_true",
                    help="UNINSTRUCTED corpus (measures continuation style, NOT recall) — "
                         "historical reproduction only")
    ap.add_argument("--selftest", action="store_true",
                    help="harness integrity checks (offline corpus assertions +, if the lane "
                         "answers, one tiny live instructed needle). Run before every "
                         "measurement session.")
    ap.add_argument("--skip-needle", action="store_true")
    ap.add_argument("--skip-vision", action="store_true")
    args = ap.parse_args()
    lane = validate(args.lane)
    instruct = not args.legacy

    if args.selftest:
        ok = True
        corpus = needle(24000, "raptor123-selftest", instruct=True, nonce="selftest")
        checks = [
            ("cold-corpus nonce prefix", corpus.startswith("[selftest] ")),
            ("instructed suffix present", "INSTRUCTION:" in corpus),
            ("needle word embedded", "raptor123-selftest" in corpus),
            ("corpus non-trivial", len(corpus.split()) > 3000),
            ("legacy corpus has no instruction",
             "INSTRUCTION:" not in needle(1000, "w", instruct=False)),
            ("png payload plausible", png_split((1, 2, 3), (4, 5, 6)).startswith("iVBOR")),
        ]
        for name, passed in checks:
            print(f"  {'PASS' if passed else 'FAIL'} selftest: {name}")
            ok = ok and passed
        try:
            txt, pt, _ = chat(lane, args.model, needle(3000, "raptor7self-selftest",
                                                       instruct=True, nonce="stlive"),
                              max_tokens=30)
            passed = "raptor7self-selftest".split("-")[0] in txt and pt > 1000
            print(f"  {'PASS' if passed else 'FAIL'} selftest: live instructed micro-needle "
                  f"(pt={pt})")
            ok = ok and passed
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP selftest: live leg unreachable ({str(exc)[:60]})")
        print("SELFTEST:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if not args.skip_needle:
        print(f"== context ladder ({'INSTRUCTED' if instruct else 'UNINSTRUCTED/legacy — not a recall gate'}) ==")
        for depth in [int(x) for x in args.depths.split(",")]:
            if args.salad_word:
                word = f"raptor{depth % 977}{args.nonce}"
                embedded = word + (f'-{args.nonce}' if args.nonce else '')
            else:
                word = f"marigold{depth % 89}"
                embedded = word
            prompt = needle(depth, embedded, instruct=instruct, nonce=args.nonce)
            try:
                txt, pt, wall = chat(lane, args.model, prompt, max_tokens=40)
                ok = word in txt
                rate = int(pt / wall) if wall else 0
                print(f"  {'PASS' if ok else 'FAIL'} ~{depth // 1024}K: prompt_tokens={pt} "
                      f"prefill~{rate} tok/s wall={wall:.0f}s word={'found' if ok else 'MISSING'}")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL ~{depth // 1024}K: {str(exc)[:90]}")

    if not args.skip_vision:
        print("== vision fence ==")
        colors = ["red", "green", "blue", "yellow", "orange", "purple", "brown", "black"]
        rgbs = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
                "yellow": (255, 255, 0), "orange": (255, 165, 0), "purple": (128, 0, 128),
                "brown": (139, 69, 19), "black": (0, 0, 0)}
        t0 = time.time()
        try:
            txt, pt, _ = chat(lane, args.model, [
                {"type": "text", "text": "One word answer: what color is this image's left half?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_split((0, 0, 255), (255, 255, 255))}},
            ], max_tokens=20)
            print(f"  image-TTFT: {time.time() - t0:.1f}s (prompt_tokens={pt}) answer={txt.strip()[:40]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  image-TTFT FAIL: {str(exc)[:90]}")
        try:
            content = [{"type": "text", "text": "Eight images follow. List the LEFT color of each in order, comma-separated, colors only."}]
            for c in colors:
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_split(rgbs[c], (255, 255, 255))}})
            txt, pt, wall = chat(lane, args.model, content, max_tokens=80)
            low = txt.lower()
            hits = sum(1 for c in colors if c in low)
            print(f"  8-image default cap: {hits}/8 left-colors named in order "
                  f"(prompt_tokens={pt}, wall={wall:.0f}s) -> {txt.strip()[:80]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  8-image default cap FAIL: {str(exc)[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
