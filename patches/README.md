# patches/ — the SP-prefill, deferral-cadence, admission, and geospec appliers

Four appliers, all house-idiom appliers (not diffs), all:

- **Marker-guarded and idempotent** — every edit lands inside a marked
  block; re-running is a no-op (`--status` on the deferral, admission,
  and geospec appliers reports without changing; `--check`/`--revert`
  exist only on the SP-prefill applier).
- **Fail-closed anchors** — every edit aborts unless its anchor text
  occurs exactly once in the expected file, so a diverged image tree
  can never be half-patched.
- a syntax check runs inside the apply on every edited file
  (`py_compile` in the SP applier; `compile()` / `ast.parse()` in the
  others).

## 1. `apply_sp_prefill.py` — sequence-parallel prefill at TP=3

Ports community sequence-parallel **prefill** onto
this runtime's DeepSeek-V4.0 (Vision-Exp) tree, staged in correctness
gated stages. `--check` reports status; `--revert` restores the
`.spprebak` backups byte-exactly.

## The stages (one runtime env var: `DSV4V_PREFILL_SP_STAGES`)

| Stage | What it does | When it is active |
|---|---|---|
| 1 | Embedding reduce-scatter, gathered straight back — **bit-identical outputs** (each token has exactly one owning vocabulary shard) | scaffolding; exercises the row math and collectives with a correctness gate before any dataflow change |
| 2 | Sharded inter-layer state: the row-wise mHC/norm/residual work runs on local rows; the MoE late all-reduce and the attention-output all-reduce become reduce-scatters | long prefills only (see the threshold below) |
| 3 | **The narrow attention front** (what the serving profile ships): SWA-only layers run their row-wise front on local rows and all-gather the narrow q/kv product instead of the wide hidden state | adds onto stage 2 |
| 4 | The indexer query-side row split (documented variant — correctness-clean, no measured speed case on this stack; `--stages 4` emits it, `--stages <=3` emits **zero** stage-4 bytes so the tree stays byte-identical to the stage-3 output) | research |

`DSV4V_PREFILL_SP_MIN_ROWS` (the recipe ships 4097) is the row threshold:
below it everything is stock, and the resolver auto-raises it above the
decode ceiling (`max_num_seqs × (1 + k)` and the capture size), so decode
forwards never take the SP path. **Under mixed serving the scheduler
delivers deep prefills in 1024-token steps, so stages 2–3 stay stock
there by design** — SP engages when a solo deep prefill runs as full ~8K
chunks (the adaptive-chunk flag; the engine's split/stage log lines are
the engagement proof).

Stage-4 validation mode: `DSV4V_INDEXER_SPLIT_CHECK=1` runs the full-row
indexer twice beside the split and logs per-layer selection drift against
the full-row path's own run-to-run drift (radix top-k keeps arbitrary
subsets of tied positions — the comparison baseline must be drift-aware).

## Applying

```sh
# inside the container (or against an extracted tree), after the TP3 pad kit:
python3 apply_sp_prefill.py --check            # status only
python3 apply_sp_prefill.py --stages 3         # the serving profile
python3 apply_sp_prefill.py --revert           # byte-exact restore
```

Default `--vllm-root` is the image's dist-packages vLLM tree. If you stage
this beside the pad kit for automatic application, chain it **inside the
pad kit's `__main__` stanza, after its `main()` call** — a chained applier
that runs (and exits) before the pad kit leaves the ring crashing on raw
`64 % 3` head divisibility (FIXLOG rung 15: the `__main__` stanza IS the
execution order).

## Engineering constraints baked into the design (learned live)

- The layout crosses compiled regions as **plain ints** — a contextvar
  scope entered in compiled code is traced away by Dynamo; a module
  global is constant-folded and silently disables earlier stages; even
  inert per-call env reads cost 3–4× warm throughput in the eager
  attention path.
- The stage-4 publish is **one default-stream all-gather after the
  execute-in_parallel join** — issuing a collective from inside the
  indexer op (auxiliary CUDA stream) is unordered against the default-
  stream SP collectives and corrupts deep selection (the rev-1 defect).
- On this image the Vision-Exp hotfix monkey-patches the MoE forward
  with a positional-only signature — the MoE discovers the SP layout
  from a scoped context rather than a keyword argument.
- The boot-time mtp-buffer hotfix recognizes its own patched form on
  container re-entry; the stage-2 edit keeps that guard verbatim below
  the SP block so re-starts (vs re-creates) skip cleanly.

## 2. `apply_prefill_decode_interval.py` — decode fairness during deep prefills

A scheduler patch for the mixed-load hole the adaptive-chunk contract
leaves open: an interactive decode stream beside a deep cold prefill
collapses to ~12× TPOT inflation with multi-second freezes, even with
the chunk threshold configured. Two causes, both patched:

- the adaptive cap lift checks for a decode-active request
  *instantaneously* — async scheduling and spec-decode draft/verify
  both create momentary windows where none is observable, so full-size
  chunks slip past the threshold cap;
- any step that carries a prefill chunk costs the chunk's full wall
  time even with decode co-batched in the same forward.

The patch gates both on a **recent-decode window** (decode scheduled
within the last `N + 2` steps): while it holds, the chunk stays capped
at the configured threshold in both scheduler loops, and a prefill
chunk may only run after ≥ `N` decode-favorable steps. Idle/solo
prefill is untouched (full chunks — the SP engagement path); the signal
ages out ~2 steps after decode drains, so prefill resumes full speed.

**One env knob: `DSPARK_PREFILL_DECODE_INTERVAL`** — `2` ships (the
serving point); `0`/unset means byte-equivalent behavior (gate the
invocation so the patch is not even applied). Measured headline: mean
inflation ~12× → ~3× with worst gaps to the ~1 s class and the median
back at solo speed, outputs byte-identical — the full both-arms record
is FIXLOG rung 33.

**Applying** — it runs inside the container against the vLLM scheduler
tree, **after the adaptive-chunk and issue-43 hotfixes** (it asserts
both marks and refuses otherwise — ordering is enforced by the anchors,
not the docs):

```sh
python3 apply_prefill_decode_interval.py --status   # APPLIED / NOT APPLIED
python3 apply_prefill_decode_interval.py            # idempotent apply
```

For automatic application, stage it into the checkout's patch dir and
add it to the entrypoint chain after the issue-43 patch, gated on the
env knob (skip the invocation entirely at 0/unset). Recreating the
containers reverts everything — the patches re-apply at boot.

**Instrument:** `tools/mixed-tax-probe.py` (this repo) measures the
effect: a solo decode stream re-measured while a same-depth fresh-nonce
cold prefill runs. The engine-agnostic diagnostic for the underlying
scheduler race: if your worst decode gap during a deep prefill matches
the full-chunk quantum (`max_chunk_tokens / prefill_rate`) instead of
your configured threshold, something is lifting the cap.

## 3. `apply_interactive_admission.py` — interactive-first admission

The deferral applier protects streams that are already decoding. This
one protects requests that are still **waiting**: admission is a single
FIFO queue plus the runtime's partial-prefill cap
(`DSPARK_MAX_INFLIGHT_PREFILLS`), so while deep prefills hold the cap,
every arriving interactive request queues behind them — a chat fired
mid-bulk waits the minutes-class time for the first bulk to finish,
with the engine perfectly healthy. Three mechanisms, one master knob:

- **Size-inferred classes** (`DSPARK_INTERACTIVE_MAX_PROMPT_TOKENS`,
  default 32768): prompts above the threshold are bulk — priority
  demoted (`DSPARK_BULK_PRIORITY`, default 100) under
  `--scheduling-policy priority`, so the waiting heap orders every
  interactive request ahead of bulk. Zero client/protocol changes.
- **Chunk-boundary yield** (`DSPARK_BULK_YIELD_LIMIT`, default 2): a
  waiting interactive request makes one running bulk prefill yield per
  step, via the engine's own `_preempt_request` (the memory-pressure
  transition). After two yields the bulk is immune — bulk cannot
  starve. Decode-active requests are never victims by construction.
- **Waiting-interactive budget guard** (the v2 mechanism, and the one
  a measurement was needed to find): the yield alone frees a cap SLOT,
  not TOKEN BUDGET — with nothing decoding, the adaptive lift hands the
  surviving solo bulk full-size chunks, it consumes the whole step
  budget, and the waiting loop still admits nothing (measured both
  arms on quiet idle-gated boots: the yield alone was slightly WORSE
  than no patch). The guard holds chunks at the threshold in both
  scheduler loops while the waiting head is interactive, leaving the
  rest of every step's budget for the owed admission.

Net, measured on this stack: mid-bulk chat wall (full short
completion) minutes-class → **single-digit seconds** (bounded below by
the one chunk already in flight when the request arrives — that is the
physical floor, not a queue), worst decode gap unchanged, deep-prefill
wall penalty upper-bounded ~20%, needles word-exact through
preempt-recompute-resume — full both-arms record in FIXLOG rung 35,
including the refuted assumption worth keeping: a yielded bulk's
resume is a full recompute on a loaded stack (the freed blocks are
reconsumed) — do not budget a prefix-cache-walk resume here.

**One env knob + one flag**: `DSPARK_INTERACTIVE_ADMISSION=1` and
`--scheduling-policy priority` added to the serve line (a mismatch
logs a loud warning and disables the feature — fail-visible; with
all-default priorities the priority queue behaves exactly FCFS, so
the flag alone changes nothing else). `0`/unset = not applied.

**Applying** — it runs inside the container against the vLLM scheduler
tree, **after the deferral-cadence applier** (it asserts the
deferral, issue-43 and adaptive-chunk marks and refuses otherwise —
ordering is enforced by the anchors, not the docs):

```sh
python3 apply_interactive_admission.py --status   # APPLIED / NOT APPLIED
python3 apply_interactive_admission.py            # idempotent apply
```

Version note: a previous-version mark on a reused container filesystem
is a hard error by design — recreate the containers (the documented
boot path) rather than restarting them.

## 4. `hotfix-dsv4-prefill-chunk-geospec.py` — one compiled variant for every prefill chunk geometry

The permanent close of the TP-wedge class (FIXLOG rung 39; rung 38
recorded the wedge, this closes the root cause). The engine's prefill
chunk-metadata Triton kernel is launched with Python-int chunk bounds
whose values are arbitrary (they depend on packed context depth ×
concurrency), and Triton specializes integer arguments by residue
class — value 1 bakes a constexpr, multiples of 16 earn a divisibility
hint — plus a 16-byte-alignment key per pointer. Real traffic
therefore mints new kernel-cache keys mid-serve, and the first
compile that lands inside a TP collective stalls the whole lane with
`/health` still green. No request-side warmup can enumerate that
variant space; rung 38's warmup arms only papered over the classes
they happened to hit.

The applier adds `do_not_specialize` on the two geometry ints and
`do_not_specialize_on_alignment` on the sliced seq-lens pointer (the
same pattern this vLLM already uses on another kernel) — a
codegen-only change, outputs bit-identical. One variant then serves
every geometry and compiles during the existing boot warmup.

**Applying** — runs inside the container against the image's vLLM
tree (fail-closed: refuses unless the anchor decorator matches
exactly once and the pinned kwargs still name real parameters):

```sh
python3 hotfix-dsv4-prefill-chunk-geospec.py --status   # APPLIED / NOT APPLIED
python3 hotfix-dsv4-prefill-chunk-geospec.py            # idempotent apply
```

For automatic application, stage it with the other appliers and chain
it in the entrypoint after the image's own indexer-file hotfixes
(that file is sha-gated on the same source). **One env knob:**
`DSPARK_SKIP_PREFILL_CHUNK_GEOSPEC_HOTFIX=1` skips the invocation
entirely (rollback lever); unset = applied. Verified on a 3-box lane:
zero mid-serve JIT of the kernel across boot, the concurrent
deep-prefill trigger, and a full replay pass; a fixed set of
deterministic cache variants per box with zero growth under all
later gates; correctness battery and throughput band unchanged.

**Instrument:** `tools/fairness-probe.py` (this repo): two fresh-nonce
bulk prefills fill the cap, then a small chat and a parallel decode
stream fire mid-flight. The engine-agnostic tell for the budget hole:
an interactive request that still waits minutes despite the inflight
cap having free slots — the step's token budget, not the slot count,
is the binding constraint.

## Credits

The stage semantics follow the community V4.1 prefill-SP patch series
(SP-state / SP-dataflow / CED encoder / local attention front / transport
threshold / indexer split) documented in
[christopher_owen's DS4.1 TP=3 reference thread](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438);
this applier adapts those semantics onto the V4.0 tree of the pinned
runtime. The injected module carries its upstream
SPDX-License-Identifier: Apache-2.0 header; edited files in your tree
remain under their upstream licenses.
