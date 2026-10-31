#!/usr/bin/env python3
"""Hotfix: interactive-first admission — size-inferred request classes +
chunk-boundary yield + waiting-interactive budget guard. Master knob:
``DSPARK_INTERACTIVE_ADMISSION=1`` together with the engine's
``--scheduling-policy priority`` flag (a mismatch logs a loud warning and
disables the feature — fail-visible, never silently). ``0``/unset = gate
the invocation so the patch is not even applied (and every injected line
would sit behind the master guard — byte-equivalent behavior).

Why this exists
---------------
The deferral-cadence applier (``apply_prefill_decode_interval.py``)
protects streams that are already decoding beside a chunking prefill.
It does nothing for requests that are still WAITING: admission is a
single FIFO queue plus the runtime's partial-prefill concurrency cap
(``DSPARK_MAX_INFLIGHT_PREFILLS``), so while deep prefills hold the cap,
every arriving interactive request queues behind them for the duration
of the deep work — a chat fired mid-bulk measured a minutes-class
completion wall
with the engine perfectly healthy (the number is the time for the first
bulk to finish, not a malfunction).

Two mechanisms close it, and the second one took a measurement to find:

  (a) the chunk-boundary yield frees a CAP SLOT but not TOKEN BUDGET.
      With no decode-active request in ``self.running``, the adaptive
      chunk lift holds chunks at full ``max_num_batched_tokens`` (the
      deferral cap only engages on recent decode), so after a yield
      leaves one solo bulk prefill running, it consumes the entire step
      budget in the running loop and the waiting loop can admit nothing
      until it finishes. Measured both arms on quiet idle-gated boots:
      the yield ALONE made mid-bulk chat wall slightly WORSE than no
      patch (still minutes-class) — a clean demonstration that freeing
      a slot is not freeing budget.

  (b) the waiting-interactive budget guard (the fix): while the
      most-important waiting request is interactive, chunks are held at
      the configured long-prefill token threshold in BOTH scheduler
      loops, so every step leaves the rest of its token budget for the
      admission the engine already owes. With (a) + (b) together the
      mid-bulk chat wall drops to the single-digit-seconds class —
      bounded below by the chunk already in flight when the request
      arrives (one full-chunk wall at this posture), not by the queue.

What it changes
---------------
1. **Size-inferred class** (``_enqueue_waiting_request``): prompts above
   ``DSPARK_INTERACTIVE_MAX_PROMPT_TOKENS`` (default 32768) are bulk —
   their ``priority`` is raised to ``DSPARK_BULK_PRIORITY`` (default
   100; lower schedules earlier, everything else defaults to 0) and they
   carry a marker. Under ``--scheduling-policy priority`` the waiting
   queue is a heap ordered by ``(priority, arrival_time)``, so
   interactive requests order ahead of bulk with zero client or
   protocol changes. Classification happens once per request object
   (first enqueue), so the class is stable across preempt/re-enqueue.
2. **Chunk-boundary yield** (top of ``Scheduler.schedule``, before any
   of the step's scheduling): if the most-important waiting request is
   interactive, the partial-prefill cap is full, and a running bulk
   prefill still has yield budget (``num_preemptions`` below
   ``DSPARK_BULK_YIELD_LIMIT``, default 2), preempt exactly ONE bulk
   (the largest) via the engine's own ``_preempt_request`` — the same
   transition the memory-pressure path uses: blocks freed, slot
   released, request re-queued in priority order. Decode-active
   requests are never candidates (still-prefilling bulk only, by
   construction). After the limit the bulk is immune and runs to
   completion: bulk cannot starve.
3. **Waiting-interactive budget guard**: a step-local signal computed
   right after the yield call — true while the waiting heap head is
   not bulk (under priority ordering an interactive request is always
   the head when one waits). While true, running-loop chunks AND
   waiting-loop admission chunks are capped at the long-prefill token
   threshold. The signal clears the moment no interactive request is
   waiting, so solo prefill returns to full-size chunks immediately
   (the adaptive lift's cold-TTFT purpose is untouched).

Cost accounting: each bulk yields at most twice, ever; a yield costs
the victim a recompute of its computed prefix (the freed blocks are
reconsumed by other work under pool pressure — do not assume a
prefix-cache walk resume on a fully loaded stack) plus a requeue wait
of seconds (re-admission needs one free slot beside short interactive
work). Measured wall penalty on deep prefills under interactive load:
upper-bounded ~20%, expected under ~10% from the bound math.

Off (0/unset): byte-equivalent. Idempotent per version — re-running is
a no-op (``--status`` reports without changing); a previous-version
mark on a reused container filesystem is a hard error (recreate the
containers — the documented boot path anyway). Anchors are asserted, so
a diverged tree can never be half-patched; requires the deferral-cadence,
issue-43 and adaptive-chunk marks to be present (refuses otherwise —
ordering is enforced by the anchors). Patches
/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
in-place inside the container before ``exec vllm serve``.
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
MARK = "# [dspark-iaa-hotfix]"
if "--help" in sys.argv or "-h" in sys.argv:
    print(__doc__.strip() if __doc__ else __file__)
    raise SystemExit(0)
V2_TAG = "v2 budget guard"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    status_src = P.read_text() if P.is_file() else ""
    if MARK in status_src and V2_TAG in status_src:
        print("dspark iaa (interactive-first admission): APPLIED (v2)")
    elif MARK in status_src:
        print("dspark iaa (interactive-first admission): APPLIED (v1 — STALE, "
              "recreate the containers for v2)")
    else:
        print("dspark iaa (interactive-first admission): NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    if V2_TAG in src:
        print(f"[dspark-iaa] v2 already applied to {P}")
        raise SystemExit(0)
    print("[dspark-iaa] FATAL: v1 mark present without v2 on this fs — the "
          "container fs was reused across a patch version bump; recreate "
          "the containers (the documented boot path) and relaunch", file=sys.stderr)
    raise SystemExit(1)
assert "# [dspark-pdi-hotfix]" in src, "iaa: pdi hotfix must be applied first"
assert "# [issue43-hotfix]" in src, "iaa: issue43 hotfix must be applied first"
assert "# [dspark-adaptive-chunk]" in src, "iaa: adaptive-chunk hotfix must be applied first"
assert "\nimport time\n" in src or "import time" in src.split("\n\n")[0], "iaa: time import missing"

# --- 0. module knobs, layered after the pdi knob block ------------------------
A0_OLD = "_PDI_N = max(0, _PDI_N)\n"
assert src.count(A0_OLD) == 1, "iaa: pdi knob anchor not found/ambiguous"
src = src.replace(
    A0_OLD,
    A0_OLD
    + "# [dspark-iaa-hotfix] Interactive-first admission knobs.\n"
    + "# DSPARK_INTERACTIVE_ADMISSION=1 + --scheduling-policy priority:\n"
    + "# prompts above the threshold are bulk (priority demoted + marker);\n"
    + "# a waiting interactive request makes one running bulk prefill yield\n"
    + "# per step at a chunk boundary (at most DSPARK_BULK_YIELD_LIMIT times\n"
    + "# per bulk request) and holds chunks at the mixed-load floor so the\n"
    + "# step keeps token budget for its admission (v2). 0/unset = off\n"
    + "# (byte-equivalent).\n"
    + "_IAA_MASTER = os.environ.get(\"DSPARK_INTERACTIVE_ADMISSION\", \"0\") "
    + "not in (\"0\", \"\", \"false\", \"False\")\n"
    + "try:\n"
    + "    _IAA_MAX_PROMPT_TOKENS = int(os.environ.get("
    + "\"DSPARK_INTERACTIVE_MAX_PROMPT_TOKENS\", \"32768\") or \"32768\")\n"
    + "except ValueError:\n"
    + "    _IAA_MAX_PROMPT_TOKENS = 32768\n"
    + "try:\n"
    + "    _IAA_BULK_PRIORITY = int(os.environ.get("
    + "\"DSPARK_BULK_PRIORITY\", \"100\") or \"100\")\n"
    + "except ValueError:\n"
    + "    _IAA_BULK_PRIORITY = 100\n"
    + "try:\n"
    + "    _IAA_YIELD_LIMIT = max(0, int(os.environ.get("
    + "\"DSPARK_BULK_YIELD_LIMIT\", \"2\") or \"2\"))\n"
    + "except ValueError:\n"
    + "    _IAA_YIELD_LIMIT = 2\n"
    + "# [dspark-iaa-hotfix] fail-visible: a non-positive bulk priority\n"
    + "# would silently disable demotion — everything would stay priority\n"
    + "# 0 = interactive and the mechanism would be a no-op.\n"
    + "assert _IAA_BULK_PRIORITY > 0, (\n"
    + "    \"dspark-iaa: DSPARK_BULK_PRIORITY must be > 0\")\n",
    1,
)

# --- 1. per-scheduler enable flag + policy validation --------------------------
A1_OLD = "        self.waiting = create_request_queue(self.policy)\n"
assert src.count(A1_OLD) == 1, "iaa: waiting-queue init anchor not found/ambiguous"
src = src.replace(
    A1_OLD,
    A1_OLD
    + "        # [dspark-iaa-hotfix] class-aware admission only means\n"
    + "        # something with the priority queue in force; fail visible,\n"
    + "        # never silently.\n"
    + "        self._dspark_iaa_enabled = (\n"
    + "            _IAA_MASTER and self.policy == SchedulingPolicy.PRIORITY)\n"
    + "        if _IAA_MASTER and not self._dspark_iaa_enabled:\n"
    + "            logger.warning(\n"
    + "                \"[dspark-iaa] DSPARK_INTERACTIVE_ADMISSION=1 requires \"\n"
    + "                \"--scheduling-policy priority; interactive admission is OFF\")\n",
    1,
)

# --- 2. size-inferred classification at enqueue -------------------------------
A2_OLD = (
    "    def _enqueue_waiting_request(self, request: Request) -> None:\n"
    "        if self._is_blocked_waiting_status(request.status):\n"
)
assert src.count(A2_OLD) == 1, "iaa: enqueue anchor not found/ambiguous"
src = src.replace(
    A2_OLD,
    (
        "    def _enqueue_waiting_request(self, request: Request) -> None:\n"
        "        # [dspark-iaa-hotfix] size-inferred admission class, applied\n"
        "        # ONCE per request object (first enqueue): prompts above the\n"
        "        # threshold are bulk — priority demoted (lower value schedules\n"
        "        # earlier; everything else defaults to 0) + marker. First-touch\n"
        "        # keeps the class stable across preempt/streaming re-enqueues\n"
        "        # (a session that grows past the threshold mid-conversation\n"
        "        # keeps its original class). Off = untouched.\n"
        "        if self._dspark_iaa_enabled \\\n"
        "                and getattr(request, \"_dspark_bulk\", None) is None:\n"
        "            request._dspark_bulk = (\n"
        "                request.num_prompt_tokens > _IAA_MAX_PROMPT_TOKENS)\n"
        "            if request._dspark_bulk:\n"
        "                request.priority = max(request.priority,\n"
        "                                        _IAA_BULK_PRIORITY)\n"
        "        if self._is_blocked_waiting_status(request.status):\n"
    ),
    1,
)

# --- 3. the yield helper, defined above _preempt_request ----------------------
A3_OLD = "    def _preempt_request(self, request: Request, timestamp: float) -> None:\n"
assert src.count(A3_OLD) == 1, "iaa: preempt def anchor not found/ambiguous"
src = src.replace(
    A3_OLD,
    (
        "    def _dspark_maybe_yield_for_interactive(\n"
        "            self, preempted_reqs: list) -> None:\n"
        "        \"\"\"[dspark-iaa-hotfix] Chunk-boundary yield: if the most-\n"
        "        important waiting request is interactive, the inflight-prefill\n"
        "        cap is full, and a running bulk prefill still has yield budget,\n"
        "        preempt exactly one bulk (the largest) via the engine's own\n"
        "        _preempt_request. Called at the top of schedule(), before this\n"
        "        step's running loop schedules anything, so the yielded request\n"
        "        was not scheduled this step and the memory-path's mid-step\n"
        "        scheduled-set surgery does not apply. NOTE: the engine's own\n"
        "        `if not preempted_reqs` gate skips the waiting loop for the\n"
        "        remainder of THIS step after any preemption — the interactive\n"
        "        request is admitted at step N+1 (verified, no livelock).\n"
        "        Decode-active requests are never candidates.\"\"\"\n"
        "        if not self._dspark_iaa_enabled or not self.waiting:\n"
        "            return\n"
        "        if self._pause_state != PauseState.UNPAUSED:\n"
        "            return\n"
        "        try:\n"
        "            head = self.waiting.peek_request()\n"
        "        except IndexError:\n"
        "            return\n"
        "        if getattr(head, \"_dspark_bulk\", False):\n"
        "            return  # the most-important waiter is itself bulk\n"
        "        cap = self._dspark_max_inflight_prefills\n"
        "        if cap <= 0:\n"
        "            return\n"
        "        partial = [r for r in self.running\n"
        "                   if r.num_computed_tokens < r.num_prompt_tokens]\n"
        "        if len(partial) < cap:\n"
        "            return  # admission slots free; the normal loop admits it\n"
        "        candidates = [\n"
        "            r for r in partial\n"
        "            if getattr(r, \"_dspark_bulk\", False)\n"
        "            and r.num_preemptions < _IAA_YIELD_LIMIT]\n"
        "        if not candidates:\n"
        "            return  # interactive prefills or exhausted yield budgets\n"
        "        victim = max(candidates, key=lambda r: r.num_prompt_tokens)\n"
        "        self.running.remove(victim)\n"
        "        self._preempt_request(victim, time.monotonic())\n"
        "        preempted_reqs.append(victim)\n"
        "        logger.info(\n"
        "            \"[dspark-iaa] chunk-boundary yield: req=%s prompt=%d \"\n"
        "            \"preemptions=%d, for waiting req=%s (%d tokens)\",\n"
        "            victim.request_id, victim.num_prompt_tokens,\n"
        "            victim.num_preemptions, head.request_id,\n"
        "            head.num_prompt_tokens)\n"
        "\n"
        "    def _preempt_request(self, request: Request, timestamp: float) -> None:\n"
    ),
    1,
)

# --- 4. the yield call + the v2 waiting-interactive signal, first thing in
#         schedule() -------------------------------------------------------------
A4_OLD = "        preempted_reqs: list[Request] = []\n"
assert src.count(A4_OLD) == 1, "iaa: schedule preempted-list anchor not found/ambiguous"
src = src.replace(
    A4_OLD,
    A4_OLD
    + "        # [dspark-iaa-hotfix] chunk-boundary yield check — before any\n"
    + "        # of this step's scheduling (see the helper above).\n"
    + "        if self._dspark_iaa_enabled:\n"
    + "            self._dspark_maybe_yield_for_interactive(preempted_reqs)\n"
    + "        # [dspark-iaa-hotfix] v2 budget guard signal: True while the\n"
    + "        # most-important waiting request is interactive. Under the\n"
    + "        # priority queue an interactive request is always the heap\n"
    + "        # head when one is waiting, so \"head not bulk\" == \"an\n"
    + "        # interactive request is waiting\". Step-local: computed after\n"
    + "        # the yield call (a just-yielded-for head is seen this step)\n"
    + "        # and read by both chunk-cap injections below so a step that\n"
    + "        # owes an admission keeps budget for it.\n"
    + "        _iaa_wait_interactive = False\n"
    + "        if self._dspark_iaa_enabled and self.waiting:\n"
    + "            try:\n"
    + "                _iaa_wait_interactive = not getattr(\n"
    + "                    self.waiting.peek_request(), \"_dspark_bulk\", False)\n"
    + "            except IndexError:\n"
    + "                pass\n",
    1,
)

# --- 5. v2 running-loop chunk cap, right after the pdi chunk cap ---------------
A5_OLD = (
    "                elif _pdi_recent_decode:\n"
    "                    _pdi_cap = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < _pdi_cap < num_new_tokens:\n"
    "                        num_new_tokens = _pdi_cap\n"
)
assert src.count(A5_OLD) == 1, "iaa: pdi running-cap tail anchor not found/ambiguous"
src = src.replace(
    A5_OLD,
    A5_OLD
    + "            # [dspark-iaa-hotfix] v2 budget guard: an interactive\n"
    + "            # request is waiting -> hold this chunk at the mixed-load\n"
    + "            # floor so the step leaves token budget for its admission.\n"
    + "            # Without this the adaptive lift hands a solo bulk the whole\n"
    + "            # step budget and the waiting head starves on BUDGET even\n"
    + "            # with cap slots free (measured: a minutes-class chat wall\n"
    + "            # with the yield alone).\n"
    + "            if _iaa_wait_interactive:\n"
    + "                _iaa_cap = self.scheduler_config.long_prefill_token_threshold\n"
    + "                if 0 < _iaa_cap < num_new_tokens:\n"
    + "                    num_new_tokens = _iaa_cap\n",
    1,
)

# --- 6. v2 waiting-loop admission-chunk cap, right after the pdi first-chunk
#         cap ----------------------------------------------------------------------
A6_OLD = (
    "                        _pdi_wcap = self.scheduler_config.long_prefill_token_threshold\n"
    "                        if 0 < _pdi_wcap < num_new_tokens:\n"
    "                            num_new_tokens = _pdi_wcap\n"
)
assert src.count(A6_OLD) == 1, "iaa: pdi waiting-cap tail anchor not found/ambiguous"
src = src.replace(
    A6_OLD,
    A6_OLD
    + "                    # [dspark-iaa-hotfix] v2 budget guard: same floor\n"
    + "                    # for any further admission chunk this step — a\n"
    + "                    # bulk admitted behind the interactive head must\n"
    + "                    # not re-monopolize the step's remaining budget.\n"
    + "                    if _iaa_wait_interactive:\n"
    + "                        _iaa_wcap = self.scheduler_config.long_prefill_token_threshold\n"
    + "                        if 0 < _iaa_wcap < num_new_tokens:\n"
    + "                            num_new_tokens = _iaa_wcap\n",
    1,
)

compile(src, "scheduler.py", "exec")
P.write_text(src)
print(f"[dspark-iaa] patched (v2) {P}")
