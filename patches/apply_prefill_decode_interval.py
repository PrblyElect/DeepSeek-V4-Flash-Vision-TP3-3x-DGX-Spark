#!/usr/bin/env python3
"""Hotfix: prefill-decode deferral cadence + race-proof chunk cap.
Env knob: ``DSPARK_PREFILL_DECODE_INTERVAL=N`` (N >= 1 enables; unset/0
= off, byte-equivalent behavior — gate the invocation so 0/unset skips
application entirely). Layers on top of the adaptive-chunk and issue-43
scheduler hotfixes (apply after both; the anchors refuse otherwise).

Why this exists
---------------
With a chunked-prefill threshold configured (1024 here), adaptive chunk
ON, and the bounded-decode-service (issue-43) floor all active, a decode
stream beside a deep cold prefill still measures mean TPOT ~12x solo
with worst gaps in the multi-second class. Two distinct causes:

  (a) a scheduler step that carries a prefill chunk takes the chunk's
      wall time no matter that decode is co-batched -- the decode floor
      guarantees decode is *scheduled* every step, not that the step is
      *short*;

  (b) the adaptive-chunk cap lift ("no decode-active request in
      ``self.running`` -> one full ``max_num_batched_tokens`` chunk")
      races with momentary decode absence (async scheduling and
      speculative-decode draft/verify structure both create windows
      where no decode-active request is observable), and the measured
      worst gaps match the full-chunk quantum (a full 8192-token chunk
      at ~1K tok/s prefill ≈ the multi-second stalls) exactly.

The engine-agnostic diagnostic: compare the worst decode inter-token
gap during a deep prefill to the chunk quantum,
``worst_gap ≈ max_chunk_tokens / prefill_rate``. A match at the FULL
chunk size means the configured threshold is not bounding step time --
something is lifting it.

What it changes (all in ``Scheduler.schedule``)
-----------------------------------------------
1. Deferral cadence (the main mechanism). While a decode stream is
   active now OR was scheduled within the last ``N + 2`` steps (the
   "recent-decode" signal, which survives the momentary-absence race),
   a running-loop prefill chunk may only be scheduled when at least
   ``N`` steps without prefill have elapsed since the last
   prefill-carrying step. Deferred chunks are dropped to 0 tokens and
   skipped by the existing zero-check -- the same safe skip path the
   issue-43 floor already uses -- so the intervening steps are
   decode-favorable (pure-decode steps at decode-step wall time, not
   chunk wall time).
2. Race-proof chunk cap. While that same recent-decode condition holds,
   the prefill chunk is additionally capped at the configured
   long-prefill token threshold regardless of the adaptive lift, in
   BOTH the running loop and the waiting-admission loop (the first
   chunk) -- the full-size escape is closed.

Solo prefill (no decode active or recent) is untouched: full chunks, no
deferral, so cold TTFT and any solo-prefill fast path are unchanged.
When decode goes idle mid-prefill, the recent-decode signal ages out
after ``N + 2`` steps and prefill resumes full speed -- no deadlock, no
livelock (prefill always progresses at worst once per ``N + 1`` steps
while decode exists, and immediately otherwise).

Off (0/unset): every injected line is behind ``_PDI_N > 0`` guards; the
patch text is present but behavior is byte-equivalent.

Idempotent (re-running is a no-op; ``--status`` reports without
changing). Anchors are asserted, so a diverged tree can never be
half-patched; requires the issue-43 and adaptive-chunk marks to be
present (refuses otherwise). Patches
/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
in-place inside the container before ``exec vllm serve``.
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
MARK = "# [dspark-pdi-hotfix]"
if "--help" in sys.argv or "-h" in sys.argv:
    print(__doc__.strip() if __doc__ else __file__)
    raise SystemExit(0)
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    status_src = P.read_text() if P.is_file() else ""
    print("dspark pdi (prefill-decode-interval):",
          "APPLIED" if MARK in status_src else "NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    print(f"[dspark-pdi] already applied to {P}")
    raise SystemExit(0)
assert "# [issue43-hotfix]" in src, "pdi: issue43 hotfix must be applied first"
assert "# [dspark-adaptive-chunk]" in src, "pdi: adaptive-chunk hotfix must be applied first"

# --- 0. module knob, after issue43's diag constant ---------------------------
A0_OLD = ('_ISSUE43_SCHED_DIAG = os.environ.get("DSPARK_ISSUE43_SCHED_DIAG", "0") '
          'not in ("0", "", "false", "False")\n')
assert src.count(A0_OLD) == 1, "pdi: issue43 diag-constant anchor not found"
src = src.replace(
    A0_OLD,
    A0_OLD
    + "# [dspark-pdi-hotfix] prefill-decode deferral cadence knob.\n"
    + "# DSPARK_PREFILL_DECODE_INTERVAL=N (N>=1): while a decode stream is (or\n"
    + "# was, within N+2 steps) active, force >=N decode-favorable steps between\n"
    + "# prefill chunks and hold the chunk cap at long_prefill_token_threshold.\n"
    + "# 0 or unset = off (byte-equivalent behavior).\n"
    + "try:\n"
    + "    _PDI_N = int(os.environ.get(\"DSPARK_PREFILL_DECODE_INTERVAL\", \"0\") or \"0\")\n"
    + "except ValueError:\n"
    + "    _PDI_N = 0\n"
    + "_PDI_N = max(0, _PDI_N)\n",
    1,
)

# --- 1. deferral + hard cap in the running loop: anchored on the issue-43
#         decode-floor block (unique) so the plain zero-check line's deeper-
#         indented siblings in the waiting loop cannot collide ---------------
A1_OLD = (
    "                if _dec_floor > 0:\n"
    "                    num_new_tokens = min(\n"
    "                        num_new_tokens,\n"
    "                        max(0, token_budget - _dec_floor))\n"
    "\n"
    "            if num_new_tokens == 0:\n"
)
assert src.count(A1_OLD) == 1, "pdi: issue43-floor/zero-check anchor not found/ambiguous"
A1_NEW = (
    "                if _dec_floor > 0:\n"
    "                    num_new_tokens = min(\n"
    "                        num_new_tokens,\n"
    "                        max(0, token_budget - _dec_floor))\n"
    "\n"
    "            # [dspark-pdi-hotfix] recent-decode signal: a decode-active\n"
    "            # request is in self.running now OR was scheduled within the\n"
    "            # last _PDI_N + 2 steps (survives the momentary-absence race\n"
    "            # that otherwise lifts the adaptive cap to a full-size chunk).\n"
    "            # Deferral cadence: drop this prefill chunk to 0 tokens (the\n"
    "            # zero-check below skips it -> decode-only step) unless\n"
    "            # >= _PDI_N prefill-free steps have passed since the last\n"
    "            # prefill-carrying step; when allowed, hold the configured\n"
    "            # threshold cap against the adaptive lift.\n"
    "            _pdi_recent_decode = False\n"
    "            if _PDI_N > 0 and getattr(request, \"is_prefill_chunk\", False):\n"
    "                _pdi_recent_decode = (\n"
    "                    any(not _r.is_prefill_chunk for _r in self.running)\n"
    "                    or (self.current_step\n"
    "                        - getattr(self, \"_pdi_last_decode_step\", -(1 << 30)))\n"
    "                    <= (_PDI_N + 2))\n"
    "                if (_pdi_recent_decode\n"
    "                        and getattr(self, \"_pdi_steps_since_prefill\", 1 << 30)\n"
    "                        < (_PDI_N + 1)):\n"
    "                    num_new_tokens = 0\n"
    "                elif _pdi_recent_decode:\n"
    "                    _pdi_cap = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < _pdi_cap < num_new_tokens:\n"
    "                        num_new_tokens = _pdi_cap\n"
    "\n"
    "            if num_new_tokens == 0:\n"
)
src = src.replace(A1_OLD, A1_NEW, 1)

# --- 2. waiting-loop first-chunk cap, after the adaptive waiting cap ---------
A2_OLD = (
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
)
assert src.count(A2_OLD) == 1, "pdi: waiting-loop cap anchor not found/ambiguous"
A2_NEW = A2_OLD + (
    "                    # [dspark-pdi-hotfix] first-chunk cap through the same\n"
    "                    # decode-detection race (recent-decode condition).\n"
    "                    if _PDI_N > 0 and (\n"
    "                            any(not _r.is_prefill_chunk for _r in self.running)\n"
    "                            or (self.current_step\n"
    "                                - getattr(self, \"_pdi_last_decode_step\", -(1 << 30)))\n"
    "                            <= (_PDI_N + 2)):\n"
    "                        _pdi_wcap = self.scheduler_config.long_prefill_token_threshold\n"
    "                        if 0 < _pdi_wcap < num_new_tokens:\n"
    "                            num_new_tokens = _pdi_wcap\n"
)
src = src.replace(A2_OLD, A2_NEW, 1)

# --- 3. cadence bookkeeping between the running loop and the waiting loop ----
A3_OLD = "        # Record the LoRAs in scheduled_running_reqs\n"
assert src.count(A3_OLD) == 1, "pdi: end-of-running-loop anchor not found/ambiguous"
A3_NEW = (
    "        # [dspark-pdi-hotfix] deferral-cadence bookkeeping (running loop\n"
    "        # only -- the deep-prefill chunks that matter; a waiting-admitted\n"
    "        # first chunk does not reset the cadence counter).\n"
    "        if _PDI_N > 0:\n"
    "            if prefill_scheduled:\n"
    "                self._pdi_steps_since_prefill = 0\n"
    "            else:\n"
    "                self._pdi_steps_since_prefill = (\n"
    "                    getattr(self, \"_pdi_steps_since_prefill\", 1 << 30) + 1)\n"
    "            if issue43_step_diag[\"decode\"]:\n"
    "                self._pdi_last_decode_step = self.current_step\n"
    "\n"
    "        # Record the LoRAs in scheduled_running_reqs\n"
)
src = src.replace(A3_OLD, A3_NEW, 1)

compile(src, "scheduler.py", "exec")
P.write_text(src)
print(f"[dspark-pdi] patched {P}")
