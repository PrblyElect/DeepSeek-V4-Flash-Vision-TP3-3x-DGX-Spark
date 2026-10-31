#!/usr/bin/env python3
"""spec-decode-short-prompt-check.py — the short-prompt path under speculative decode.

Upstream vLLM tracks a prefill that can be misread as a uniform-decode batch.
The runner's shape check is pure arithmetic: when the number of scheduled
tokens for a request equals `uniform_decode_query_len * num_reqs`, the prefill
can be dispatched into the CUDA graph captured for speculative decode. For
hybrid models with recurrent-state layers the state write is then skipped and
the output is deterministic garbage (upstream issue #53051). The trigger is a
prompt of exactly `uniform_decode_query_len` tokens.

For this recipe's speed profile (DSpark k=5 draft tokens) that is 6 tokens, so
a 6-token prompt is enough to reach the shape.

This script probes exactly that shape and asks whether the answer is sane:

  1. it resolves `uniform_decode_query_len` — from the running engine's own
     command line when a container is named, from the speculative config you
     pass, from the shipped profile, or from an explicit `--query-len`;
  2. it builds a prompt of EXACTLY that many tokens, using the engine's own
     `/tokenize` endpoint for the token IDs and sending those IDs, so the
     count is exact by construction. It does NOT binary-search over
     characters: a byte-level BPE count is not prefix-monotonic, so "one more
     character" does not mean "one more token", and a character search lands
     on the wrong token count without saying so;
  3. it sends it at temperature 0 and checks the completion: the engine's own
     `usage.prompt_tokens` must equal the target, and the answer must be
     structurally sane and echo the marker the prompt carried.

The answer check is a marker echo plus structural checks. It does NOT reject
CJK output: this model legitimately emits CJK, and a script that bans it
would fail a healthy lane for a cosmetic reason.

It is a READ-ONLY probe: one request to a serving engine. Run it in a boot
window against a lane you are accepting, never against a lane carrying
production traffic, and never as a bring-up step.

Usage:
    python3 tools/spec-decode-short-prompt-check.py \\
        --base-url http://127.0.0.1:8000 --model ds4flash-vision

    # measure the trap length from the running engine itself (runs
    # `docker exec` on THIS host — run it on the box that owns the engine):
    python3 tools/spec-decode-short-prompt-check.py \\
        --base-url http://127.0.0.1:8000 --model ds4flash-vision \\
        --container ds4flash-vision-head

Exit codes: 0 = the short-prompt path answered sanely, 1 = it did not,
2 = the probe could not be set up (that is a setup error, not a verdict).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

MARKER = "ALPHA"

# Marker templates, shortest first: the first one that fits inside the target
# token budget is used, so a 6-token trap still carries an instruction.
MARKER_TEMPLATES = (
    "Echo {m}",
    "Say {m}",
    "Repeat {m}",
    "Print {m}",
    "Repeat exactly: {m}",
    "Please repeat exactly: {m}",
)

# Fallback templates with no marker — used only when no marker template fits
# inside the target budget. The answer check is then structural only.
PLAIN_TEMPLATES = (
    "Hi",
    "Hello",
    "Hello there",
    "Hello there friend",
)

# Padding. The marker template is placed LAST, so the instruction the model
# has to follow is the final thing it reads; the padding sits in front of it.
FILLER = " ok"


# --------------------------------------------------------------------------
# engine access


def post(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def engine_encode(base_url: str, model: str, text: str, timeout: int) -> list[int] | None:
    """The engine's own token IDs for a text, or None if it would not say."""
    try:
        out = post(f"{base_url}/tokenize", {"model": model, "prompt": text}, timeout)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    tokens = out.get("tokens")
    if isinstance(tokens, list) and tokens and all(isinstance(t, int) for t in tokens):
        return tokens
    if isinstance(tokens, list) and not tokens:
        return []
    return None


def engine_query_len_from_container(container: str) -> tuple[int | None, str]:
    """(uniform_decode_query_len, source) from the running engine's own state.

    Reads the engine process's own command line inside the container and
    applies the engine's own rule (a decode step submits one query for the
    sampled token plus one per draft token, so the uniform decode query
    length is 1 + num_speculative_tokens). No import of the engine package:
    the lane already owns the GPU.

    This runs `docker exec` on THIS host. Run the probe on the box that owns
    the engine container.
    """
    snippet = (
        "import json,sys\n"
        "argv=open('/proc/1/cmdline','rb').read().decode().split('\\x00')\n"
        "spec=None\n"
        "for i,a in enumerate(argv):\n"
        "    if a=='--speculative-config' and i+1<len(argv): spec=argv[i+1]\n"
        "    elif a.startswith('--speculative-config='): spec=a.split('=',1)[1]\n"
        "print(json.dumps({'spec':spec}))\n"
    )
    try:
        res = subprocess.run(
            ["docker", "exec", "-i", container, "python3", "-"],
            input=snippet, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"unavailable ({exc.__class__.__name__})"
    if res.returncode != 0:
        return None, f"unavailable (the probe failed in {container})"
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None, "unavailable (unreadable probe output)"

    raw = data.get("spec")
    if not raw:
        return None, "the engine runs WITHOUT speculative decode"
    try:
        cfg = json.loads(raw)
    except ValueError:
        return None, "unreadable --speculative-config on the engine"
    k = cfg.get("num_speculative_tokens")
    if not isinstance(k, int):
        return None, "the engine's speculative config has no num_speculative_tokens"
    return 1 + k, "the running engine's own command line"


def query_len_from_profile(profile: str) -> tuple[int | None, str]:
    """The trap length implied by the shipped profile's draft depth."""
    try:
        with open(profile, "r", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None, ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition("=")
        if name.strip() != "MTP_NUM_TOKENS":
            continue
        value = value.strip().strip('"').strip("'")
        if value.isdigit():
            return 1 + int(value), ("the shipped profile's MTP_NUM_TOKENS "
                                    "(INFERRED: 1 + the draft depth, not read "
                                    "from the running engine)")
        return None, ""
    return None, ""


# --------------------------------------------------------------------------
# the prompt plan


def build_prompt_ids(encode, target: int) -> tuple[list[int], str, bool] | None:
    """Build a prompt of EXACTLY `target` tokens.

    `encode(text)` returns the engine's token IDs for a text, or None.

    Returns (token_ids, template_used, carries_marker) or None when no
    template fits. The IDs are returned so the caller can SEND them: the
    engine then uses them as-is, which is what makes the count exact.
    """
    chosen: tuple[list[int], str, bool] | None = None
    for template in MARKER_TEMPLATES:
        ids = encode(template.format(m=MARKER))
        if ids is None or len(ids) > target:
            continue
        chosen = (ids, template.format(m=MARKER), True)
        break
    if chosen is None:
        for template in PLAIN_TEMPLATES:
            ids = encode(template)
            if ids is None or len(ids) > target:
                continue
            chosen = (ids, template, False)
            break
    if chosen is None:
        return None

    ids, template, carries_marker = chosen
    need = target - len(ids)
    if need:
        filler = _filler_ids(encode, need)
        if filler is None:
            return None
        ids = filler[:need] + ids
    return ids, template, carries_marker


def _filler_ids(encode, need: int) -> list[int] | None:
    """At least `need` padding token IDs."""
    text = FILLER * 16
    for _ in range(6):
        ids = encode(text)
        if ids is None:
            return None
        if len(ids) >= need:
            return ids
        text *= 2
    return None


# --------------------------------------------------------------------------
# the answer check


def looks_degenerate(text: str) -> str | None:
    """Return a reason string when the completion is not a sane answer.

    CJK output is NOT degenerate: this model legitimately emits it, so a
    script-level script ban would fail a healthy lane.
    """
    stripped = text.strip()
    if not stripped:
        return "the completion is empty"
    if any(ord(c) < 32 and c not in "\n\t\r" for c in stripped):
        return "the completion contains control characters"
    if "�" in stripped:
        return "the completion contains a replacement character"
    words = stripped.split()
    if len(words) >= 12:
        uniq = len(set(words))
        if uniq <= max(2, len(words) // 12):
            return "the completion repeats the same few words"
    if len(stripped) >= 24:
        for size in (4, 6, 8):
            block = stripped[:size]
            if block and stripped.count(block) >= max(3, len(stripped) // (size * 2)):
                return f"the completion repeats a {size}-character block"
    return None


def verdict(target: int, reported: int | None, text: str, carries_marker: bool,
            allow_missing_echo: bool = False) -> tuple[bool, list[str], list[str]]:
    """(passed, failures, warnings) for one probe answer."""
    failures: list[str] = []
    warnings: list[str] = []

    if reported is None:
        failures.append("the engine did not report usage.prompt_tokens — the probe "
                        "cannot prove it hit the intended shape")
    elif reported != target:
        failures.append(
            f"the engine scheduled {reported} prompt tokens, not {target} — "
            "the probe did not hit the intended shape"
        )

    reason = looks_degenerate(text)
    if reason:
        failures.append(reason)

    if carries_marker and MARKER.lower() not in text.lower():
        msg = (f"the completion does not echo the marker {MARKER} — a prefill that was "
               "misread as a decode batch answers with garbage")
        if allow_missing_echo:
            warnings.append(msg + " (allowed by --allow-missing-echo)")
        else:
            failures.append(msg)

    return (not failures), failures, warnings


# --------------------------------------------------------------------------
# selftest


def selftest() -> int:
    """Offline checks: the answer check and the token-plan builder."""
    print("spec-decode-short-prompt-check selftest — offline, no engine needed")
    ok = True

    cases = [
        ("a clean echo", "ALPHA", False),
        ("a plain sentence",
         "The quick brown fox jumps over the lazy dog near the river bank.", False),
        ("CJK output is legitimate for this model", "你好，世界。这是一次正常的回答。", False),
        ("Japanese output is legitimate for this model", "こんにちは。", False),
        ("an empty answer", "", True),
        ("control characters", "ab\x00\x01cd", True),
        ("a replacement character", "abc�def", True),
        ("degenerate word repetition", "yes yes yes yes yes yes yes yes yes yes yes yes",
         True),
        ("degenerate block repetition", "abcd" * 12, True),
    ]
    for label, text, want_degenerate in cases:
        reason = looks_degenerate(text)
        got = reason is not None
        passed = got == want_degenerate
        print(f"  {'PASS' if passed else 'FAIL'} selftest: {label} "
              f"({'flagged' if got else 'accepted'})")
        ok = ok and passed

    # the token plan: a fake tokenizer (one word per token)
    def fake_encode(text: str) -> list[int]:
        return [1000 + i for i, _ in enumerate(text.split())]

    for target in (1, 2, 6, 7, 32):
        plan = build_prompt_ids(fake_encode, target)
        if plan is None:
            print(f"  FAIL selftest: no prompt plan for a {target}-token target")
            ok = False
            continue
        ids, template, carries_marker = plan
        passed = len(ids) == target
        print(f"  {'PASS' if passed else 'FAIL'} selftest: a {target}-token prompt is built "
              f"exactly ({len(ids)} token(s), template {template!r})")
        ok = ok and passed

    # the marker template must sit at the END, so the instruction is the last
    # thing the model reads
    plan = build_prompt_ids(fake_encode, 6)
    if plan is None:
        print("  FAIL selftest: no prompt plan for the marker-tail check")
        ok = False
    else:
        ids, template, _marker = plan
        template_ids = fake_encode(template)
        passed = bool(template_ids) and ids[-len(template_ids):] == template_ids
        print(f"  {'PASS' if passed else 'FAIL'} selftest: the marker template is the tail "
              "of the prompt")
        ok = ok and passed

    # the verdict: a shape mismatch is a failure
    passed, failures, _warn = verdict(6, 5, "ALPHA", True)
    print(f"  {'PASS' if not passed else 'FAIL'} selftest: a prompt-token mismatch fails")
    ok = ok and not passed

    # the verdict: a degenerate answer fails
    passed, failures, _warn = verdict(6, 6, "��" * 20, True)
    print(f"  {'PASS' if not passed else 'FAIL'} selftest: a degenerate answer fails")
    ok = ok and not passed

    # the verdict: a sane CJK answer is not a failure on its own
    passed, failures, warnings = verdict(6, 6, "你好，世界。", False)
    print(f"  {'PASS' if passed else 'FAIL'} selftest: a sane CJK answer does not fail")
    ok = ok and passed

    # the verdict: a missing marker fails by default, and can be allowed
    passed, _failures, _warnings = verdict(6, 6, "Sure thing.", True)
    soft_passed, _f, _w = verdict(6, 6, "Sure thing.", True, allow_missing_echo=True)
    good = not passed and soft_passed
    print(f"  {'PASS' if good else 'FAIL'} selftest: a missing marker fails by default and "
          "can be downgraded with --allow-missing-echo")
    ok = ok and good

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# cli


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()

    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="the engine's OpenAI-compatible base URL")
    ap.add_argument("--model", default="ds4flash-vision", help="the served model name")
    ap.add_argument("--query-len", type=int, default=None,
                    help="the trap length to probe (skips the measurement)")
    ap.add_argument("--container", default=None,
                    help="a container running the engine, on THIS host")
    ap.add_argument("--spec-config", default=None,
                    help="the engine's --speculative-config JSON, if known")
    ap.add_argument("--profile", default=None,
                    help="the profile to read MTP_NUM_TOKENS from (default: config/cluster.env)")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--allow-missing-echo", action="store_true",
                    help="downgrade a missing marker echo to a warning")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on a usage error, and 2 is this tool's
        # INCONCLUSIVE code ("the engine could not be reached"). A malformed
        # command line must not be indistinguishable from an inconclusive
        # run -- the guard sibling uses 3 for exactly this.
        if exc.code not in (0, None):
            return 3
        return 0

    base = args.base_url.rstrip("/")

    # ---- 0. the engine must be up
    try:
        models = get(f"{base}/v1/models", args.timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"cannot reach {base}/v1/models ({exc.__class__.__name__}) — "
              "is the engine serving?", file=sys.stderr)
        return 2
    names = [m.get("id") for m in models.get("data", [])]
    print(f"engine up: {', '.join(str(n) for n in names) or 'no model listed'}")

    # ---- 1. the trap length
    query_len: int | None = None
    source = ""
    if args.query_len is not None:
        query_len = args.query_len
        source = "--query-len (given)"
    elif args.container:
        query_len, source = engine_query_len_from_container(args.container)
    elif args.spec_config:
        try:
            cfg = json.loads(args.spec_config)
        except ValueError:
            print("--spec-config is not valid JSON", file=sys.stderr)
            return 2
        k = cfg.get("num_speculative_tokens")
        if not isinstance(k, int):
            print("--spec-config has no num_speculative_tokens", file=sys.stderr)
            return 2
        query_len = 1 + k
        source = "the speculative config you passed"
    else:
        profile = args.profile
        if profile is None:
            profile = str(Path(__file__).resolve().parent.parent
                          / "config" / "cluster.env")
        query_len, source = query_len_from_profile(profile)
        if query_len is None:
            source = ""

    if query_len is None:
        print("could not resolve the trap length:", file=sys.stderr)
        print("  pass --query-len N, or --container <container> (the engine's own "
              "command line), or --spec-config '<json>'", file=sys.stderr)
        return 2
    print(f"uniform_decode_query_len = {query_len}  (source: {source})")

    # ---- 2. a prompt of exactly that many tokens, sent as token IDs
    def encode(text: str) -> list[int] | None:
        return engine_encode(base, args.model, text, args.timeout)

    plan = build_prompt_ids(encode, query_len)
    if plan is None:
        print(f"could not build a prompt of exactly {query_len} tokens — "
              "the /tokenize endpoint did not answer, or no template fits that budget",
              file=sys.stderr)
        return 2
    ids, template, carries_marker = plan
    print(f"probe prompt: {len(ids)} token(s) (target {query_len}), sent as token IDs")
    print(f"probe template: {template!r}"
          + ("" if carries_marker else "  (no marker fits this budget — structural check only)"))

    # ---- 3. ask, at temperature 0
    try:
        out = post(f"{base}/v1/completions", {
            "model": args.model,
            "prompt": ids,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }, args.timeout)
    except urllib.error.HTTPError as exc:
        print(f"the engine rejected the probe (HTTP {exc.code})", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"the probe request failed ({exc.__class__.__name__})", file=sys.stderr)
        return 2

    choices = out.get("choices") or []
    if not choices:
        print("the engine returned no choice", file=sys.stderr)
        return 1
    text = choices[0].get("text") or ""
    usage = out.get("usage") or {}
    reported = usage.get("prompt_tokens")

    print(f"engine-reported prompt tokens: {reported}")
    print(f"completion: {text.strip()[:200]!r}")

    passed, failures, warnings = verdict(
        query_len, reported if isinstance(reported, int) else None, text,
        carries_marker, args.allow_missing_echo,
    )
    for msg in warnings:
        print(f"WARNING: {msg}")

    if not passed:
        print("\nSPEC-SHORT-PROMPT FAIL")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("\nSPEC-SHORT-PROMPT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
