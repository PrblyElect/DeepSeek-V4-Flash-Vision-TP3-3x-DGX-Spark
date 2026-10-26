#!/usr/bin/env python3
"""Aggregate-throughput sweep with the workload split. Stdlib-only.

Measures decode at each concurrency level: N parallel workers each issuing
sequential requests, in TWO prompt classes (count = high draftability,
prose = low) because spec-decode acceptance is workload-dependent —
publishing one headline number hides the split. Nonce'd prompts defeat
prefix-cache replay (a same-payload "new" prompt reports cache hits as
throughput). After the sweep: a temp-0 exact-repeat + strict-JSON
corruption spot-check.

Decode numbers from this tool are same-boot comparable only: this stack
measures ±10% boot-to-boot kernel-selection spread at identical config
(see docs/VALIDATION.md — cross-boot deltas under ~13% are noise).

Only talks to the endpoint you point it at (default loopback); redirects
refused. Extend ALLOWED_HOSTS if you sweep a remote head node. Tee all
output per your own recording discipline.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.parse
import urllib.request

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


def gen(lane: str, model: str, prompt: str, max_tokens: int) -> int:
    """Returns completion_tokens."""
    import uuid
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt} (nonce {uuid.uuid4().hex[:8]})"}],
        "temperature": 0.0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(f"{lane}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=300) as res:
        d = json.load(res)
    return d.get("usage", {}).get("completion_tokens", 0)


def sweep(lane: str, model: str, level: int, reqs: int, max_tokens: int, klass: str) -> dict:
    prompts = {
        "count": "Count from 1 to 300, separated by commas:",
        "prose": "Write a short product review paragraph about a generic kitchen blender:",
    }[klass]
    errors: list[str] = []
    tokens = [0] * level

    def worker(idx: int):
        got = 0
        for _ in range(reqs):
            try:
                got += gen(lane, model, prompts, max_tokens)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc)[:80])
        tokens[idx] = got

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(level)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    total = sum(tokens)
    return {
        "level": level, "class": klass, "reqs_per_worker": reqs,
        "agg_tok_s": round(total / wall, 1) if wall > 0 else 0.0,
        "total_tokens": total, "wall_s": round(wall, 1),
        "errors": len(errors),
    }


def corruption(lane: str, model: str) -> dict:
    import uuid
    prompt = f"Repeat this exact sentence back to me and nothing else: the sky is clear today (nonce {uuid.uuid4().hex[:8]})."
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 60,
                       "chat_template_kwargs": {"thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{lane}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    a = json.load(_opener.open(req, timeout=120))["choices"][0]["message"]["content"].lower()
    exact = "the sky is clear today" in a
    jbody = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 120,
                        "chat_template_kwargs": {"thinking": False},
                        "messages": [{"role": "user", "content": "Return ONLY this JSON object: {\"ok\": true, \"n\": 7}"}]}).encode()
    jreq = urllib.request.Request(f"{lane}/v1/chat/completions", data=jbody,
                                  headers={"Content-Type": "application/json"})
    ja = json.load(_opener.open(jreq, timeout=120))["choices"][0]["message"]["content"]
    json_ok = '"ok"' in ja and "7" in ja
    return {"exact_repeat": exact, "strict_json": json_ok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,16,32")
    ap.add_argument("--lane", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="ds4flash-vision")
    ap.add_argument("--reqs", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--classes", default="count,prose")
    ap.add_argument("--skip-corruption", action="store_true")
    args = ap.parse_args()
    lane = validate(args.lane)

    results = []
    for level in [int(x) for x in args.levels.split(",")]:
        for klass in args.classes.split(","):
            r = sweep(lane, args.model, level, args.reqs, args.max_tokens, klass)
            results.append(r)
            print(f"level={level:>2} {klass:>5}: agg={r['agg_tok_s']:>7.1f} tok/s "
                  f"({r['total_tokens']} tok / {r['wall_s']}s, errors={r['errors']})")
    if not args.skip_corruption:
        c = corruption(lane, args.model)
        results.append({"corruption": c})
        print(f"corruption: exact_repeat={c['exact_repeat']} strict_json={c['strict_json']}")
    bad = any(r.get("errors") for r in results if isinstance(r, dict) and "errors" in r)
    print("— AGG SWEEP " + ("CLEAN" if not bad else "WITH ERRORS"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
