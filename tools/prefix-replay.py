#!/usr/bin/env python3
"""prefix-replay.py — the warm-prefix replay, as a gate that can actually fail.

One stable prefix, sent twice from the same client: the first call is cold,
the second should hit the prefix cache. The gate asserts that BOTH calls
succeed (HTTP 200 and non-empty content) and that the warm call did not
regress; it reports the time-to-first-token ratio, which is the replay win.

Why an executable gate and not a curl loop: `curl -s ... >/dev/null`
exits 0 on a 500 response, so a loop built from it cannot fail — it would
report a healthy replay for an engine that answered every call with an
error. This tool checks the status code and the content, and exits
non-zero when the replay did not work.

Payloads: by default the tool synthesises a stable prefix so the gate is
runnable as shipped. Point it at your own captured prefixes with
`--payload FILE` (repeatable) — one JSON file per prefix, each a full
chat-completions request body. This recipe ships no client payloads.

Exit codes:
    0  PASS  every replay answered 200 with content, and the warm call did
             not regress beyond --max-regression
    1  FAIL  a replay failed, or the warm call regressed
    2  USAGE bad arguments / unreachable endpoint
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

DEFAULT_PREFIX_WORDS = 1200


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def validate(url: str) -> str:
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or parts.hostname not in ALLOWED_HOSTS:
        raise SystemExit(
            f"refusing non-local URL: {url} (extend ALLOWED_HOSTS deliberately)"
        )
    return url


def synthetic_payload(model: str, words: int) -> dict:
    """A stable, long-ish prefix — long enough for the cache to have a block
    to hold. The same text every call: that is the point of a prefix."""
    filler = (
        "The quick brown fox jumps over the lazy dog while the engine serves a "
        "stable prefix for the replay comparison. "
    )
    body = (filler * (words // 14 + 1))[: words * 7]
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": body},
            {"role": "user", "content": "Reply with the single word: ready"},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": True,
        "chat_template_kwargs": {"thinking": False},
    }


def load_payload(path: str, model: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as exc:
        raise SystemExit(f"cannot read payload {path}: {exc}")
    except ValueError as exc:
        raise SystemExit(f"payload {path} is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        raise SystemExit(f"payload {path} must be a JSON object")
    payload.setdefault("model", model)
    payload.setdefault("stream", True)
    payload.setdefault("max_tokens", 16)
    payload.setdefault("temperature", 0.0)
    return payload


def replay(lane: str, payload: dict, timeout: int) -> tuple[float, str]:
    """Send the payload once; return (time-to-first-token, first content).

    Raises RuntimeError on anything that is not a clean 200 with content.
    """
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{lane}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with _opener.open(req, timeout=timeout) as res:
            status = getattr(res, "status", 200)
            if status != 200:
                raise RuntimeError(f"HTTP {status}")
            ttft = None
            content: list[str] = []
            for raw in res:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    obj = json.loads(chunk)
                except ValueError:
                    continue
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        if ttft is None:
                            ttft = time.time() - t0
                        content.append(text)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {lane}: {exc.reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot reach {lane}: {exc}") from exc

    if ttft is None:
        raise RuntimeError("no content token arrived (empty or non-streaming reply)")
    return ttft, "".join(content)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prefix-replay.py",
        description="Replay one stable prefix twice and report the warm/cold "
                    "time-to-first-token ratio.",
    )
    ap.add_argument("--lane", default="http://127.0.0.1:8000",
                    help="the engine's base URL (default: http://127.0.0.1:8000)")
    ap.add_argument("--model", default="ds4flash-vision",
                    help="the served model name")
    ap.add_argument("--payload", action="append", default=None,
                    help="a captured chat-completions request body to replay "
                         "(repeatable; default: a synthesised stable prefix)")
    ap.add_argument("--prefix-words", type=int, default=DEFAULT_PREFIX_WORDS,
                    help="size of the synthesised prefix, in words")
    ap.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    ap.add_argument("--max-regression", type=float, default=1.5,
                    help="fail if the warm call is more than this many times "
                         "slower than the cold one (default: 1.5)")
    ap.add_argument("--require-speedup", action="store_true",
                    help="also fail unless the warm call is FASTER than the cold "
                         "one (use only when the prefix is known to be cacheable)")
    args = ap.parse_args(argv)

    try:
        lane = validate(args.lane)
    except SystemExit as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.payload:
        payloads = [(p, load_payload(p, args.model)) for p in args.payload]
    else:
        payloads = [("synthesised", synthetic_payload(args.model, args.prefix_words))]

    failures: list[str] = []
    print(f"prefix-replay — {lane} — {len(payloads)} prefix(es)")
    for label, payload in payloads:
        try:
            cold_ttft, cold_text = replay(lane, payload, args.timeout)
        except RuntimeError as exc:
            failures.append(f"{label}: cold call failed — {exc}")
            print(f"  {label}: cold call FAILED — {exc}")
            continue
        try:
            warm_ttft, warm_text = replay(lane, payload, args.timeout)
        except RuntimeError as exc:
            failures.append(f"{label}: warm call failed — {exc}")
            print(f"  {label}: warm call FAILED — {exc}")
            continue

        ratio = warm_ttft / cold_ttft if cold_ttft > 0 else float("inf")
        print(f"  {label}: cold ttft={cold_ttft:.3f}s  warm ttft={warm_ttft:.3f}s  "
              f"ratio={ratio:.2f}x")
        if not cold_text.strip() or not warm_text.strip():
            failures.append(f"{label}: a reply was empty")
        if args.max_regression > 0 and ratio > args.max_regression:
            failures.append(
                f"{label}: the warm call regressed ({ratio:.2f}x the cold call, "
                f"limit {args.max_regression:.2f}x)"
            )
        if args.require_speedup and ratio >= 1.0:
            failures.append(
                f"{label}: the warm call was not faster ({ratio:.2f}x) — no replay win"
            )

    print()
    if failures:
        print("PREFIX REPLAY FAIL:")
        for msg in failures:
            print(f"  - {msg}")
        return EXIT_FAIL
    print("PREFIX REPLAY PASS — every replay answered 200 with content, and the "
          "warm call did not regress.")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
