# patches/ — the SP-prefill applier (sequence-parallel prefill at TP=3)

`apply_sp_prefill.py` ports community sequence-parallel **prefill** onto
this runtime's DeepSeek-V4.0 (Vision-Exp) tree, staged in correctness
gated stages. It is a house-idiom applier, not a diff:

- **Marker-guarded and idempotent** — every edit lands inside a
  `# --- DSV4V_SP_PREFILL* ---` marker block; re-running is a no-op.
- **`--check`** reports status without changing anything;
  **`--revert`** restores the `.spprebak` backups byte-exactly (and
  removes the injected module).
- **Fail-closed anchors** — every edit aborts unless its anchor text
  occurs exactly once in the expected file, so a diverged image tree
  can never be half-patched. A pre-flight existence check runs before
  the first byte is written.
- `py_compile` runs inside the apply on every edited file.

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

## Credits

The stage semantics follow the community V4.1 prefill-SP patch series
(SP-state / SP-dataflow / CED encoder / local attention front / transport
threshold / indexer split) documented in
[christopher_owen's DS4.1 TP=3 reference thread](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438);
this applier adapts those semantics onto the V4.0 tree of the pinned
runtime. The injected module carries its upstream
SPDX-License-Identifier: Apache-2.0 header; edited files in your tree
remain under their upstream licenses.
