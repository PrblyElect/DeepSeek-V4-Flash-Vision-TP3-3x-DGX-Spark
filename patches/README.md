# patches/ — the SP-prefill applier and the deferral-cadence applier

Two appliers, both house-idiom appliers (not diffs), both:

- **Marker-guarded and idempotent** — every edit lands inside a marked
  block; re-running is a no-op (`--status` on the deferral applier
  reports without changing).
- **Fail-closed anchors** — every edit aborts unless its anchor text
  occurs exactly once in the expected file, so a diverged image tree
  can never be half-patched.
- `py_compile` runs inside the apply on every edited file.

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
invocation so the patch is not even applied). Measured, same-night
both arms, idle-gated on this stack: mean inflation ~12× → ~3×, worst
gaps from the multi-second class to the ~1 s class, 4-stream ~6× → ~2×
per stream with the median back at solo speed, and the mixed prefill
rate itself *improves* (~14% → ~7–9% under solo). Outputs
byte-identical (temp-0 exact-repeat batteries).

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

## Credits

The stage semantics follow the community V4.1 prefill-SP patch series
(SP-state / SP-dataflow / CED encoder / local attention front / transport
threshold / indexer split) documented in
[christopher_owen's DS4.1 TP=3 reference thread](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438);
this applier adapts those semantics onto the V4.0 tree of the pinned
runtime. The injected module carries its upstream
SPDX-License-Identifier: Apache-2.0 header; edited files in your tree
remain under their upstream licenses.
