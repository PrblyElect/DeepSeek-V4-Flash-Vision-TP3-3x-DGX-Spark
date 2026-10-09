#!/usr/bin/env python3
"""Apply the DeepSeek-V4-Flash-Vision TP=3 prefill sequence-parallelism
patch (Stages 1+2+3+4) to an installed vLLM tree.

Stage 1 (this applier) makes ONLY the embedding sequence-parallel: the
vocab-parallel embedding's all-reduce becomes an exact reduce-scatter (each
token has exactly one owning vocabulary shard, so the sum needs no ordering)
that is immediately all-gathered back to full rows. Everything downstream
runs on full rows exactly as before, so outputs are bit-identical with the
patch enabled or disabled. The value of Stage 1 is correctness scaffolding:
it lands the sp_prefill module, the row-layout math, the RS collective path,
and the env gate that Stage 2 (inter-layer sharded state, MoE reduce-scatter,
then the attention split) builds on.

Stage 2 (same applier, marker DSV4V_SP_PREFILL_S2, applied but INERT unless
DSV4V_PREFILL_SP_STAGES=2) keeps the inter-layer state sharded for long
prefills: each rank owns L = Tp/TP rows of (x, residual, post_mix, res_mix)
between collectives; the row-wise mHC tilelang kernels and both norms run on
local rows; attention all-gathers its input and its wo_b projection returns
the TP partial (all-reduce skipped) which the decoder pads and
reduce-scatters; the routed-MoE runner reduce-scatters shared+routed from a
zero-padded buffer instead of its late all-reduce; the tail collapses, hc
head, final norm and aux hidden states run locally and are gathered back.
With DSV4V_PREFILL_SP_STAGES unset or 1 the runtime path is identical to
Stage 1.

Design and semantics follow the community V4.1 prefill-SP patch series
(SP-state, SP-dataflow, CED encoder, local attention front, transport
threshold, indexer split — credited in the repository README), adapted from
that V4.1 tree onto this image's V4.0 tree.
Enable with DSV4V_PREFILL_SP_MIN_ROWS=<rows> (unset or 0 keeps everything
off; even when set, the patch is a no-op below the threshold, so decode
forwards never change) and DSV4V_PREFILL_SP_STAGES=2 for the sharded
dataflow.

Adaptation note: on this image the Vision-Exp hotfix monkey-patches
``DeepseekV4MoE.forward`` with a positional-only signature, so the MoE
discovers the active SP layout via ``sp_prefill.current()`` (scoped by the
decoder layer around the ffn call) instead of a keyword argument; the
runner-facing ``reduce_scatter_rows`` kwarg stays explicit.

Stage 3 (marker DSV4V_SP_PREFILL_S3, DSV4V_PREFILL_SP_STAGES=3) adds the
local attention front for SWA-only layers (no indexer, no compressor - the
2 hash-MoE layers at the model entry on this checkpoint): the row-wise
front (fused_wqa_wkv; the fused q/kv norm after the gather) runs on this
rank's rows and the narrow qr_kv product (q_lora_rank + head_dim = 1536
columns) is all-gathered instead of the 5120-column hidden state. The
hidden-state gather moves from the decoder into the attention forward so
SWA-only layers can skip it; indexer/compressor layers gather exactly as
Stage 2 did. At stages <= 2 every Stage-3 branch is inert.

Stage 4 (marker DSV4V_SP_PREFILL_S4, DSV4V_PREFILL_SP_STAGES=4) splits the
replicated sparse-indexer's QUERY side over the SP rows (the reference
indexer-split patch, adapted). REV 3 architecture (supersedes rev 2): each rank runs wq_b + fused rope/quant and
scores/selects only its own SPRows of every prefill chunk; the layout is
transported as PLAIN INTS (sp_row_start, sp_real_rows) threaded
decoder-layer -> forward -> attention_impl -> indexer lambda ->
DeepseekV4Indexer.forward -> indexer_op -> op arguments - no ContextVar
scope entries in compiled regions (traced away in compiled regions), no module globals
(Dynamo constant-folding killed stages 2/3), no per-call env reads (even
inert stage-4 branches cost 3-4x warm throughput). The single runtime
stage gate lives in attention_impl; the publish is ONE all-gather on the
DEFAULT stream after the execute_in_parallel join (issuing collectives
inside the op = rev-1's aux-stream defect); decode stays stock on every
rank. This is NOT the in-file DCP top-k merge - that solves the candidate
axis; here rows are independent and the only collective is the row gather.
Numeric invariance: indexer wq_b is F8_E4M3 with per-row scales (row-count
invariant); weights_proj is BF16 and runs full-width upstream, only its
per-row output is sliced. The K-side compressor still writes all rows on
every rank. APPLY-TIME STAGE GATE (E1): a --stages <=3 apply emits ZERO
MARK4 bytes - the produced tree is byte-identical to the stage-3 output; the
embedded sp_prefill module is the stage-3 text and an
already-injected module is never rewritten. Validation mode
DSV4V_INDEXER_SPLIT_CHECK=1 passes full-row tensors and additionally runs
the full-row indexer twice per layer, logging per-layer selection drift
against the full-row path's own run-to-run drift (the flag is read ONCE
at module import, never per call).

Idempotent: every edit is guarded by its marker, so re-running is a no-op.
--check reports status without changing anything; --revert restores the
.bak copies. Every edit fails closed if its anchor does not occur exactly
once in the expected file.

Usage:
    python3 apply_sp_prefill.py [--vllm-root DIR] [--check] [--revert]
                                [--stages N] [--emit-debug]
"""

from __future__ import annotations

import argparse
import os

STAGES_ENV = "DSV4V_PREFILL_SP_STAGES"
import pathlib
import py_compile
import shutil
import sys

MARK = "# --- DSV4V_SP_PREFILL ---"
MARK2 = "# --- DSV4V_SP_PREFILL_S2 ---"
MARK3 = "# --- DSV4V_SP_PREFILL_S3 ---"
MARK4 = "# --- DSV4V_SP_PREFILL_S4 ---"
MODULE_V2_TAG = "# DSV4V sp_prefill module version: 2 (stage 2: local_slice, stages_from_env)"
MODULE_V3_TAG = "# DSV4V sp_prefill module version: 3 (stage 3: stages() accessor, narrow front)"
MODULE_V4_TAG = "# DSV4V sp_prefill module version: 4 (stage 4: indexer row split, indexer_split_check)"
MODULE_V5_TAG = "# DSV4V sp_prefill module version: 5 (stage 4 rev 2: stream-safe publish, no in-op collective)"
MODULE_V6_TAG = "# DSV4V sp_prefill module version: 6 (rev2 edits, module reverted to ContextVar after the global broke compiled SP)"

# ---------------------------------------------------------------------------
# Injected module: vllm/models/deepseek_v4/sp_prefill.py
# (adapted from the community V4.1 prefill-SP module; Apache-2.0 lineage)
# ---------------------------------------------------------------------------

SP_PREFILL_MODULE = '''\
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# --- DSV4V_SP_PREFILL ---
# DSV4V sp_prefill module version: 3 (stage 3: stages() accessor, narrow front)
"""Prefill sequence parallelism (SP) for DeepSeek V4.0 tensor parallelism.

At tensor parallel size N every rank repeats the row-wise work of a decoder
layer (mHC mixes, norms, the residual carry) on every token. For a long
prefill forward, SP turns per-layer all-reduces into dim-0 reduce-scatters
so each rank owns 1/N of the rows for that work, and all-gathers the rows
back before ops that need all of them. Weights stay exactly as in ordinary
TP; nothing is replicated.

Enable with ``DSV4V_PREFILL_SP_MIN_ROWS=<rows>``; unset or 0 keeps it off.
``DeepseekV4Model.forward`` decides per forward from the token count T,
which every rank shares, so no cross-rank agreement is needed. The
threshold is raised above decode token counts (CUDA graph sizes and
``max_num_seqs * (1 + num_speculative_tokens)``), so decode never uses SP.
SP is also off for pipeline parallelism and static sequence parallelism.

## Rows

- Collectives see ``Tp = N * ceil(T / N)`` rows; each rank owns ``L = Tp/N``
  rows, rank r the padded rows ``[r*L, (r+1)*L)``.
- Kernels never see more than T rows: gathered tensors are sliced to ``[:T]``.
- Reduce-scatter inputs are Tp-row buffers whose pad rows are zero.
- The last rank's local rows can include pad rows. They stay finite and are
  never read back into real outputs.

## Stage 1

Only the embedding is sequence-parallel: its all-reduce becomes an exact
reduce-scatter (each token has one owning vocabulary shard, so the reduction
picks up a single nonzero contribution - no summation-order sensitivity)
that is immediately all-gathered back to T rows. Everything else runs on
full rows as without SP, so outputs are bit-identical with SP on or off.

## Stage 2 (DSV4V_PREFILL_SP_STAGES=2)

The embedding reduce-scatter result stays local; the residual, mixes and
carried layer output are local, so the mHC pre / post / post_pre kernels,
both norms and the final hc head run on L rows. Attention all-gathers its
input to T rows (the layout is scoped to the call via ``current()`` /
``activate()``), runs every projection, indexer and sparse MLA on all rows
as before, and its wo_b projection returns the TP partial (all-reduce
skipped) which the decoder layer pads and reduce-scatters. The MoE input is
gathered to T rows with the full input_ids; gate, top-k, routed and shared
experts run as before, and the runner reduce-scatters shared + routed from
a zero-padded buffer instead of its final all-reduce. The final collapse,
hc head and norm run locally and are gathered to T rows, as is every aux
hidden state a drafter consumes. DSpark reads only those, so its
pre-collapse MTP buffer is left alone under SP; other drafters gather it.
"""

import os
from dataclasses import dataclass

import torch

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

MIN_ROWS_ENV = "DSV4V_PREFILL_SP_MIN_ROWS"
STAGES_ENV = "DSV4V_PREFILL_SP_STAGES"


def min_rows_from_env() -> int:
    """The requested SP threshold in tokens; unset, empty or 0 is off."""
    raw = os.environ.get(MIN_ROWS_ENV, "").strip()
    rows = int(raw) if raw else 0
    if rows < 0:
        raise ValueError(f"{MIN_ROWS_ENV} must be >= 0, got {rows}")
    return rows


def stages_from_env() -> int:
    """The SP stage selection; 1 unless DSV4V_PREFILL_SP_STAGES is set.

    Stage 1 gathers the embedding reduce-scatter back to full rows
    (bit-identical outputs). Stage 2 keeps the inter-layer state sharded
    and turns the MoE and attention-output all-reduces into
    reduce-scatters. Stage 3 adds the local attention front for SWA-only
    layers. Anything else is a configuration error.
    """
    raw = os.environ.get(STAGES_ENV, "").strip()
    stages = int(raw) if raw else 1
    if stages not in (1, 2, 3):
        raise ValueError(f"{STAGES_ENV} must be 1, 2 or 3, got {stages}")
    if stages >= 2:
        logger.info(
            "DeepSeek V4.0 prefill SP stage %d (%s)",
            stages,
            "sharded inter-layer state"
            if stages == 2
            else "sharded inter-layer state + local attention front",
        )
    return stages


_STAGES_CACHE: int | None = None


def stages() -> int:
    """The cached SP stage selection for this process (boot-time env)."""
    global _STAGES_CACHE
    if _STAGES_CACHE is None:
        _STAGES_CACHE = stages_from_env()
    return _STAGES_CACHE


def resolve_min_rows(vllm_config, *, static_sequence_parallel: bool = False) -> int:
    """The effective SP threshold for this model instance; 0 is off."""
    requested = min_rows_from_env()
    if requested <= 0:
        return 0
    parallel = vllm_config.parallel_config
    reason = None
    if parallel.tensor_parallel_size <= 1:
        reason = "tensor parallel size is 1"
    elif parallel.pipeline_parallel_size != 1:
        reason = "pipeline parallelism keeps full-row intermediate tensors"
    elif static_sequence_parallel:
        reason = "static sequence parallelism is enabled"
    elif getattr(vllm_config, "use_request_boundary_checkpoints", False):
        reason = "request boundary checkpoints are enabled"
    if reason is not None:
        logger.warning("%s=%d ignored: %s", MIN_ROWS_ENV, requested, reason)
        return 0
    spec = vllm_config.speculative_config
    decode_rows = max(
        vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        vllm_config.scheduler_config.max_num_seqs
        * (1 + (spec.num_speculative_tokens if spec is not None else 0)),
        parallel.tensor_parallel_size,
    )
    rows = max(requested, decode_rows + 1)
    if rows != requested:
        logger.warning(
            "%s=%d raised to %d: decode forwards of up to %d tokens never use it",
            MIN_ROWS_ENV,
            requested,
            rows,
            decode_rows,
        )
    logger.info("DeepSeek V4.0 prefill sequence parallelism from %d tokens", rows)
    return rows


@dataclass(frozen=True)
class SPRows:
    """Row layout of one sequence-parallel forward on one TP rank."""

    num_tokens: int
    world_size: int
    rank: int

    @property
    def padded_rows(self) -> int:
        """Tp: rows seen by collectives, T rounded up to a multiple of N."""
        return -(-self.num_tokens // self.world_size) * self.world_size

    @property
    def local_rows(self) -> int:
        """L: rows this rank owns between collectives."""
        return self.padded_rows // self.world_size

    @property
    def start(self) -> int:
        """First padded row owned by this rank."""
        return self.rank * self.local_rows

    @property
    def real_rows(self) -> int:
        """Leading local rows that are real tokens; the rest are padding."""
        return max(0, min(self.local_rows, self.num_tokens - self.start))

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather this rank's L rows into the first T rows."""
        if x.shape[0] != self.local_rows:
            raise ValueError(f"SP gather expects {self.local_rows} rows, got {x.shape}")
        return tensor_model_parallel_all_gather(x, dim=0)[: self.num_tokens]

    def reduce_scatter(self, padded: torch.Tensor) -> torch.Tensor:
        """Sum a zero-padded Tp-row buffer over the ranks into local rows."""
        if padded.shape[0] != self.padded_rows:
            raise ValueError(
                f"SP reduce-scatter expects {self.padded_rows} rows, got {padded.shape}"
            )
        return tensor_model_parallel_reduce_scatter(padded, dim=0)

    def local_slice(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's L rows of a T-row tensor; pad rows are zero."""
        if x.shape[0] != self.num_tokens:
            raise ValueError(f"SP slice expects {self.num_tokens} rows, got {x.shape}")
        real = self.real_rows
        rows = x[self.start : self.start + real]
        if real == self.local_rows:
            return rows
        local = x.new_zeros((self.local_rows, *x.shape[1:]))
        local[:real].copy_(rows)
        return local

    def empty_padded(
        self, row_shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """A Tp-row buffer whose pad rows (T and beyond) are zero."""
        padded = torch.empty((self.padded_rows, *row_shape), dtype=dtype, device=device)
        padded[self.num_tokens :].zero_()
        return padded


def plan_rows(
    num_tokens: int, min_rows: int, world_size: int, rank: int
) -> SPRows | None:
    """The SP row layout for a forward of ``num_tokens`` tokens, or None."""
    if min_rows <= 0 or world_size <= 1 or num_tokens < min_rows:
        return None
    if not 0 <= rank < world_size:
        raise ValueError(f"TP rank {rank} is outside world size {world_size}")
    return SPRows(num_tokens, world_size, rank)


# --- Stage 2/3 context: layers below read current() to find the active SP
# layout (attention gathers its input and reduce-scatters its output; the
# MoE receives reduce_scatter_rows from the model). ContextVar so each
# worker thread's forward is independent, matching vLLM's threading.
import contextvars  # noqa: E402

_sp_current: contextvars.ContextVar[SPRows | None] = contextvars.ContextVar(
    "dsv4_sp_rows", default=None
)


def current() -> SPRows | None:
    """The SP layout of the forward running on this thread, or None."""
    return _sp_current.get()


class activate:
    """Scope an SP layout for the duration of a block (reentrant-safe)."""

    def __init__(self, rows: SPRows):
        self._rows = rows
        self._token = None

    def __enter__(self) -> SPRows:
        self._token = _sp_current.set(self._rows)
        return self._rows

    def __exit__(self, *exc) -> None:
        _sp_current.reset(self._token)
'''


# ---------------------------------------------------------------------------
# Stage-1 edit definitions: (path, kind, anchor/old, new) — all fail-closed.
# ---------------------------------------------------------------------------

V4_MODEL_REL = pathlib.Path("models/deepseek_v4/nvidia/model.py")
VOCAB_REL = pathlib.Path("model_executor/layers/vocab_parallel_embedding.py")
SP_MODULE_REL = pathlib.Path("models/deepseek_v4/sp_prefill.py")
RUNNER_REL = pathlib.Path("model_executor/layers/fused_moe/runner/moe_runner.py")
OPROJ_REL = pathlib.Path("models/deepseek_v4/nvidia/ops/o_proj.py")
ATTENTION_REL = pathlib.Path("models/deepseek_v4/attention.py")
SAI_REL = pathlib.Path("model_executor/layers/sparse_attn_indexer.py")

MODEL_INIT_ANCHOR = "        self.parallel_config = vllm_config.parallel_config"

MODEL_INIT_INSERT = """\
        self.parallel_config = vllm_config.parallel_config
{mark}
        # Prefill SP (Stage 1): threshold + planner. Zero when the env gate
        # is unset, in which case the forward branch below never triggers.
        from vllm.models.deepseek_v4.sp_prefill import plan_rows as _sp_plan_rows
        from vllm.models.deepseek_v4.sp_prefill import (
            resolve_min_rows as _sp_resolve_min_rows,
        )

        self._sp_plan_rows = _sp_plan_rows
        self._sp_min_rows = _sp_resolve_min_rows(vllm_config)
# --- end DSV4V_SP_PREFILL ---""".format(mark=MARK)

MODEL_FORWARD_OLD = """\
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)"""

MODEL_FORWARD_NEW = """\
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
{mark}
                _sp = self._sp_plan_rows(
                    input_ids.shape[0],
                    self._sp_min_rows,
                    get_tensor_model_parallel_world_size(),
                    get_tensor_model_parallel_rank(),
                )
                if _sp is not None:
                    # Stage 1: exact one-owner embedding reduce-scatter,
                    # gathered straight back - bit-identical full rows.
                    hidden_states = _sp.gather(
                        self.embed_tokens.forward_reduce_scatter(
                            input_ids, _sp.padded_rows
                        )
                    )
                else:
                    hidden_states = self.embed_input_ids(input_ids)
# --- end DSV4V_SP_PREFILL ---
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)""".format(
    mark=MARK
)

VOCAB_IMPORT_OLD = """\
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)"""

VOCAB_IMPORT_NEW = """\
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)"""

VOCAB_METHOD_ANCHOR = """\
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output

    def extra_repr(self) -> str:"""

VOCAB_METHOD_NEW = """\
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output

__MARK__
    def forward_reduce_scatter(
        self, input_: torch.Tensor, padded_rows: int
    ) -> torch.Tensor:
        \"\"\"Embed ``input_`` and reduce-scatter the rows over the TP group.

        For sequence parallelism: the partial embeddings are written into a
        ``padded_rows`` buffer whose trailing rows are zero, and rank r
        receives rows ``[r * padded_rows // tp_size, ...)`` of the full
        embeddings. Each token has exactly one owning vocabulary shard, so
        the reduction picks up a single nonzero contribution and is exact.
        Mirrors :meth:`forward`'s masked (non-fused) path.
        \"\"\"
        rows = input_.shape[0]
        if (
            self.tp_size == 1
            or self.parallel_group is not None
            or input_.ndim != 1
            or padded_rows < rows
            or padded_rows % self.tp_size
        ):
            raise ValueError(
                "Embedding reduce-scatter needs 1-D ids, the default TP group "
                f"and padded rows divisible by it (rows={rows}, "
                f"padded_rows={padded_rows}, tp_size={self.tp_size})"
            )
        masked_input, input_mask = get_masked_input_and_mask(
            input_,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        local = self.quant_method.embedding(self, masked_input.long())
        if local.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise NotImplementedError("FP8 embedding reduce-scatter")
        output_parallel = local.new_empty((padded_rows, *local.shape[1:]))
        output_parallel[:rows].copy_(local.masked_fill_(input_mask[:, None], 0))
        output_parallel[rows:].zero_()
        return tensor_model_parallel_reduce_scatter(output_parallel, dim=0)
# --- end DSV4V_SP_PREFILL ---

    def extra_repr(self) -> str:""".replace("__MARK__", MARK)

# ---------------------------------------------------------------------------
# Stage-2 edit definitions (marker MARK2). Applied unconditionally but
# INERT unless DSV4V_PREFILL_SP_STAGES=2: every runtime branch they add is
# gated on an SP layout that only exists under stage 2.
# ---------------------------------------------------------------------------

# --- moe_runner.py ------------------------------------------------------------

RUNNER_IMPORT_OLD = """\
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)"""

RUNNER_IMPORT_NEW = """\
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)"""

RUNNER_METHOD_ANCHOR = """\
    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:"""

RUNNER_METHOD_NEW = """\
__M2__
    def _reduce_scatter_final_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
        padded_rows: int,
        trunc_size: int | None,
        output_is_reduced: bool,
    ) -> torch.Tensor:
        \"\"\"Sequence-parallel replacement for the late all-reduce.

        The combined output is written into a ``padded_rows`` buffer whose
        trailing rows are zero and reduce-scattered along dim 0, so TP rank
        r receives rows ``[r * padded_rows // tp, (r + 1) * padded_rows // tp)``.
        \"\"\"
        if (
            output_is_reduced
            or self.moe_config.is_sequence_parallel
            or self.moe_config.skip_final_all_reduce
            or isinstance(self.router, ZeroExpertRouter)
        ):
            raise NotImplementedError("MoE reduce-scatter needs an unreduced output")
        rows = fused_output.shape[0]
        dtype = (
            fused_output.dtype
            if shared_output is None
            else torch.result_type(shared_output, fused_output)
        )
        states = fused_output.new_empty(
            (padded_rows, *fused_output.shape[1:]), dtype=dtype
        )
        states[rows:].zero_()
        if shared_output is None:
            states[:rows].copy_(fused_output)
        else:
            torch.add(shared_output, fused_output, out=states[:rows])
        states = tensor_model_parallel_reduce_scatter(states, dim=0)
        return states[..., :trunc_size] if trunc_size is not None else states

__M2END__
    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:""".replace("__M2__", MARK2).replace("__M2END__", "# --- end DSV4V_SP_PREFILL_S2 ---")

RUNNER_FORWARD_OLD = """\
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        \"\"\"Invoke the fused moe layer."""

RUNNER_FORWARD_NEW = """\
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        *,
        reduce_scatter_rows: int | None = None,
    ) -> torch.Tensor:
        \"\"\"Invoke the fused moe layer."""

RUNNER_COMBINE_OLD = """\
        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:"""

RUNNER_COMBINE_NEW = """\
        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

__M2__
        if reduce_scatter_rows is not None:
            # Prefill sequence parallelism: reduce-scatter the combined
            # output instead of the late all-reduce (Stage 2).
            return self._reduce_scatter_final_output(
                shared_output,
                fused_output,
                reduce_scatter_rows,
                og_hidden_dim_post_xform,
                self._fused_output_is_reduced,
            )
# --- end DSV4V_SP_PREFILL_S2 ---
        if shared_output is not None:""".replace("__M2__", MARK2)

# --- ops/o_proj.py ------------------------------------------------------------

OPROJ_OLD = """\
    return wo_b(z.flatten(1))"""

OPROJ_NEW = """\
__M2__
    from vllm.models.deepseek_v4 import sp_prefill as _sp_prefill

    z = z.flatten(1)
    if _sp_prefill.current() is None:
        return wo_b(z)
    # Prefill SP (Stage 2): return this rank's wo_b partial without the
    # all-reduce; the caller pads the rows and reduce-scatters them.
    return wo_b.quant_method.apply(wo_b, z, None)
# --- end DSV4V_SP_PREFILL_S2 ---""".replace("__M2__", MARK2)

# --- nvidia/model.py ------------------------------------------------------------

MODEL2_INIT_OLD = """\
        self._sp_min_rows = _sp_resolve_min_rows(vllm_config)
# --- end DSV4V_SP_PREFILL ---"""

MODEL2_INIT_NEW = """\
        self._sp_min_rows = _sp_resolve_min_rows(vllm_config)
# --- end DSV4V_SP_PREFILL ---
__M2__
        # Stage selection: 1 = Stage 1 (gather back to full rows,
        # bit-identical); 2 = sharded inter-layer state with MoE and
        # attention-output reduce-scatter. Unset = 1 (inert edits).
        from vllm.models.deepseek_v4.sp_prefill import (
            stages_from_env as _sp_stages_env,
        )

        self._sp_stages = _sp_stages_env()
        # DSpark drafts from the (gathered) aux hidden states only; under
        # stage 2 the pre-hc_head MTP buffer stays stale rather than paying
        # a full-row gather.
        _spec_cfg = vllm_config.speculative_config
        self._sp_skips_mtp_buffer = _spec_cfg is not None and _spec_cfg.use_dspark()
# --- end DSV4V_SP_PREFILL_S2 ---""".replace("__M2__", MARK2)

MODEL2_SIG_OLD = """\
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:"""

MODEL2_SIG_NEW = """\
    ) -> torch.Tensor | IntermediateTensors:
        _sp2 = None  # the SP layout threaded through the layers (stage 2)
        if get_pp_group().is_first_rank:"""

MODEL2_EMBED_OLD = """\
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
__S1__
                _sp = self._sp_plan_rows(
                    input_ids.shape[0],
                    self._sp_min_rows,
                    get_tensor_model_parallel_world_size(),
                    get_tensor_model_parallel_rank(),
                )
                if _sp is not None:
                    # Stage 1: exact one-owner embedding reduce-scatter,
                    # gathered straight back - bit-identical full rows.
                    hidden_states = _sp.gather(
                        self.embed_tokens.forward_reduce_scatter(
                            input_ids, _sp.padded_rows
                        )
                    )
                else:
                    hidden_states = self.embed_input_ids(input_ids)
# --- end DSV4V_SP_PREFILL ---
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)""".replace(
    "__S1__", MARK
)

MODEL2_EMBED_NEW = """\
        if get_pp_group().is_first_rank:
__S1__
            _sp = self._sp_plan_rows(
                input_ids.shape[0],
                self._sp_min_rows,
                get_tensor_model_parallel_world_size(),
                get_tensor_model_parallel_rank(),
            )
# --- end DSV4V_SP_PREFILL ---
__M2__
            _sp2 = _sp if self._sp_stages >= 2 else None
            if inputs_embeds is not None:
                hidden_states = (
                    inputs_embeds if _sp2 is None else _sp2.local_slice(inputs_embeds)
                )
            elif _sp is not None:
                # One vocabulary shard owns each token, so the embedding
                # reduce-scatter is exact. Stage 1 gathers straight back
                # (bit-identical); stage 2 keeps this rank's rows.
                _embedded = self.embed_tokens.forward_reduce_scatter(
                    input_ids, _sp.padded_rows
                )
                hidden_states = _sp.gather(_embedded) if _sp2 is None else _embedded
            else:
                hidden_states = self.embed_input_ids(input_ids)
# --- end DSV4V_SP_PREFILL_S2 ---
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)""".replace(
    "__S1__", MARK
).replace(
    "__M2__", MARK2
)

MODEL2_LAYER_OLD = """\
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )"""

MODEL2_LAYER_NEW = """\
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
                sp_rows=_sp2,
            )"""

MODEL2_AUX_OLD = """\
                aux_hidden_states.append(aux_recon.mean(dim=1))"""

MODEL2_AUX_NEW = """\
__M2__
                _aux_h = aux_recon.mean(dim=1)
                if _sp2 is not None:
                    # Drafters consume full-row aux states.
                    _aux_h = _sp2.gather(_aux_h)
                aux_hidden_states.append(_aux_h)
# --- end DSV4V_SP_PREFILL_S2 ---""".replace("__M2__", MARK2)

MODEL2_INTERMED_OLD = """\
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})"""

MODEL2_INTERMED_NEW = """\
        if not get_pp_group().is_last_rank:
            if _sp2 is not None:
                hidden_states = _sp2.gather(hidden_states)
            return IntermediateTensors({"hidden_states": hidden_states})"""

MODEL2_MTP_OLD = """\
        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))"""

# The guarded copy_ block is kept VERBATIM below the SP gather block: the
# image's boot-time mtp-buffer hotfix recognizes its own patched form on
# container re-entry (restart, not recreate) and skips cleanly. Restruct-
# uring the guard itself makes that hotfix report "ambiguous state" and
# exit 1, crash-looping the container (found live during bring-up).
MODEL2_MTP_NEW = """\
        # Stash pre-hc_head residual for the MTP draft (captured copy_).
__M2__
        if _sp2 is not None:
            if self._sp_skips_mtp_buffer:
                # DSpark drafts from the gathered aux hidden states only;
                # skip the full-row gather this stash would otherwise force
                # (the stash below then writes this rank's rows into the
                # buffer, which no dspark drafter reads).
                pass
            else:
                hidden_states = _sp2.gather(hidden_states)
                _sp2 = None  # every row from here on
# --- end DSV4V_SP_PREFILL_S2 ---
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))""".replace(
    "__M2__", MARK2
)

MODEL2_FINAL_OLD = """\
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:"""

MODEL2_FINAL_NEW = """\
        hidden_states = self.norm(hidden_states)
__M2__
        if _sp2 is not None:
            # The final collapse, hc head and norm ran on local rows.
            hidden_states = _sp2.gather(hidden_states)
# --- end DSV4V_SP_PREFILL_S2 ---
        if len(aux_hidden_states) > 0:""".replace("__M2__", MARK2)

MODEL2_MOEFWD_OLD = """\
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None and input_ids is None:"""

MODEL2_MOEFWD_NEW = """\
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
__M2__
        if self.use_mega_moe:
            # Prefill SP does not cover the MegaMoE output path.
            from vllm.models.deepseek_v4 import sp_prefill as _sp_mod

            if _sp_mod.current() is not None:
                raise NotImplementedError(
                    "DeepSeek V4 MegaMoE output reduce-scatter"
                )
# --- end DSV4V_SP_PREFILL_S2 ---
        if self.gate.tid2eid is not None and input_ids is None:""".replace(
    "__M2__", MARK2
)

MODEL2_MOEFUSED_OLD = """\
    def _forward_fused_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        if self.experts.is_internal_router:"""

MODEL2_MOEFUSED_NEW = """\
    def _forward_fused_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
__M2__
        # Prefill SP (Stage 2): the layout arrives via sp_prefill.current()
        # rather than a kwarg because the Vision-Exp monkey-patch forwards
        # this method with positional args only. Under SP the input is this
        # rank's rows: gather every row for routing and the experts, and
        # the runner reduce-scatters the TP partial sum so the output lands
        # back on local rows (org_shape no longer describes it).
        from vllm.models.deepseek_v4 import sp_prefill as _sp_mod

        _sp = _sp_mod.current()
        if _sp is not None:
            _hidden = _sp.gather(hidden_states)
            _logits = (
                _hidden
                if self.experts.is_internal_router
                else self.gate(_hidden)[0]
            )
            return self.experts(
                hidden_states=_hidden,
                router_logits=_logits,
                input_ids=input_ids,
                reduce_scatter_rows=_sp.padded_rows,
            ).to(_hidden.dtype)
# --- end DSV4V_SP_PREFILL_S2 ---
        org_shape = hidden_states.shape
        if self.experts.is_internal_router:""".replace("__M2__", MARK2)

MODEL2_DECODER_SIG_OLD = """\
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:"""

MODEL2_DECODER_SIG_NEW = """\
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        sp_rows=None,  # prefill sequence-parallel layout (SPRows or None)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:"""

MODEL2_DECODER_ATTN_OLD = """\
        # attn_norm is fused into mhc_pre_tilelang / mhc_fused_post_pre above.
        x = self.attn(positions, x, None)"""

MODEL2_DECODER_ATTN_NEW = """\
        # attn_norm is fused into mhc_pre_tilelang / mhc_fused_post_pre above.
__M2__
        if sp_rows is None:
            x = self.attn(positions, x, None)
        else:
            # Prefill SP (Stage 2): attention gathers every row; _o_proj
            # leaves the wo_b partial (no all-reduce) and this layer pads
            # and reduce-scatters it back to local rows.
            from vllm.models.deepseek_v4 import sp_prefill as _sp_mod

            with _sp_mod.activate(sp_rows):
                _attn_out = self.attn(positions, sp_rows.gather(x), None)
            _padded = sp_rows.empty_padded(
                _attn_out.shape[1:], _attn_out.dtype, _attn_out.device
            )
            _padded[: sp_rows.num_tokens].copy_(_attn_out)
            x = sp_rows.reduce_scatter(_padded)
# --- end DSV4V_SP_PREFILL_S2 ---""".replace("__M2__", MARK2)

MODEL2_DECODER_FFN_OLD = """\
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix"""

MODEL2_DECODER_FFN_NEW = """\
__M2__
        if sp_rows is None:
            x = self.ffn(x, input_ids)
        else:
            # Prefill SP (Stage 2): the MoE reads the active layout from
            # sp_prefill.current() and gathers / reduce-scatters around
            # its own routing.
            from vllm.models.deepseek_v4 import sp_prefill as _sp_mod

            with _sp_mod.activate(sp_rows):
                x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix
# --- end DSV4V_SP_PREFILL_S2 ---""".replace("__M2__", MARK2)

# ---------------------------------------------------------------------------
# Stage-3 edit definitions (marker MARK3). SWA-only attention layers (no
# indexer, no compressor) run their row-wise front on local rows and gather
# the narrow qr_kv product; the hidden-state gather moves from the decoder
# into the attention forward. Inert unless DSV4V_PREFILL_SP_STAGES=3.
# ---------------------------------------------------------------------------

ATTN3_FORWARD_OLD = """\
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]"""

ATTN3_FORWARD_NEW = """\
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
__M3__
        from vllm.models.deepseek_v4 import sp_prefill as _sp_mod

        _sp = _sp_mod.current()
        if (
            _sp is not None
            and _sp_mod.stages() >= 3
            and self.indexer is None
            and self.compressor is None
        ):
            # Prefill SP (Stage 3), SWA-only layer: the row-wise front
            # (fused_wqa_wkv; the fused q/kv norm after the gather) runs
            # on this rank's rows and the narrow qr_kv product
            # (q_lora_rank + head_dim columns) is gathered instead of the
            # 5120-column hidden state. Nothing after the front in a
            # SWA-only forward reads hidden_states.
            num_tokens = _sp.num_tokens
            o_padded = torch.empty(
                (num_tokens, self.padded_heads, self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            qr_kv = _sp.gather(qr_kv)
            qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
            qr, kv = fused_q_kv_rmsnorm(
                qr,
                kv,
                self.q_norm.weight.data,
                self.kv_norm.weight.data,
                self.eps,
            )
            self.attention_impl(
                None, qr, kv, None, None, None, positions, o_padded
            )
            o = o_padded[:, : self.n_local_heads, :]
            return self._o_proj(o, positions)
        if _sp is not None and _sp_mod.stages() >= 3:
            # Prefill SP (Stage 3): indexer/compressor layers build global
            # KV from the full hidden state; the Stage-2 decoder-side
            # gather moved in here so SWA-only layers could skip it.
            hidden_states = _sp.gather(hidden_states)
# --- end DSV4V_SP_PREFILL_S3 ---
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]""".replace("__M3__", MARK3)

MODEL3_DECODER_OLD = """\
            from vllm.models.deepseek_v4 import sp_prefill as _sp_mod

            with _sp_mod.activate(sp_rows):
                _attn_out = self.attn(positions, sp_rows.gather(x), None)"""

MODEL3_DECODER_NEW = """\
            from vllm.models.deepseek_v4 import sp_prefill as _sp_mod

__M3__
            with _sp_mod.activate(sp_rows):
                if _sp_mod.stages() >= 3:
                    # Stage 3: attention owns the row gather - SWA-only
                    # layers run the front on these local rows and gather
                    # the narrow product; the rest gather the hidden state.
                    _attn_out = self.attn(positions, x, None)
                else:
                    _attn_out = self.attn(positions, sp_rows.gather(x), None)
# --- end DSV4V_SP_PREFILL_S3 ---""".replace("__M3__", MARK3)

# ---------------------------------------------------------------------------
# Stage-4 edit definitions (marker MARK4): the indexer query-side row split,
# REV 3 (supersedes rev 2).
#
# Layout transport: PLAIN INTS (sp_row_start, sp_real_rows) threaded
# decoder-layer -> attention.forward -> attention_impl -> indexer lambda ->
# DeepseekV4Indexer.forward -> indexer_op -> op arguments. Rationale
# (measured on this stack): a ContextVar scope entered in compiled code is traced
# away by Dynamo (traced away in compiled regions); a module global is constant-folded and
# silently kills stages 2/3; per-call env reads cost 3-4x warm throughput
# even when the branch is inert. The single runtime stage gate lives in
# attention_impl (stages() is env-cached after first call); everything
# downstream just follows the (None or ints) it was handed.
#
# Publish: ONE tensor_model_parallel_all_gather in attention_impl on the
# DEFAULT stream after the execute_in_parallel join, before forward_mqa
# reads the buffer (issuing the collective inside the op was the rev-1
# defect: the op runs on an aux CUDA stream).
#
# Decode stays stock on every rank (rev 2's rank-0 guard reverted): decode
# rows are tiny and replicated writes are identical everywhere.
#
# The runtime stage gate makes stages 1-3 boots on a stage-4 tree behave
# exactly like the v3 tree; and a --stages <=3 apply emits ZERO MARK4
# bytes, so that tree is byte-identical to the stage-3 output (E1).
# ---------------------------------------------------------------------------

# --- attention.py: hoisted hot-path imports (module level) -------------------

ATTN4_IMPORTS_OLD = """\
from vllm.models.deepseek_v4.common.ops import (
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
)"""

ATTN4_IMPORTS_NEW = """\
from vllm.models.deepseek_v4.common.ops import (
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
)
# --- DSV4V_SP_PREFILL_S4 ---
# Stage 4: hoisted hot-path imports - nothing below may import per call.
try:
    from vllm.distributed import (
        tensor_model_parallel_all_gather as _sp4_all_gather,
    )
    from vllm.models.deepseek_v4 import sp_prefill as _sp4_prefill
    import vllm.model_executor.layers.sparse_attn_indexer as _sp4_sai
except ImportError:  # pragma: no cover - the applier injects both together
    _sp4_all_gather = None
    _sp4_prefill = None
    _sp4_sai = None
# --- end DSV4V_SP_PREFILL_S4 ---"""

# --- attention.py: DeepseekV4Attention.forward gains sp_rows -----------------

ATTN4_FWD_OLD = """\
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
"""

ATTN4_FWD_NEW = """\
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        sp_rows=None,
    ) -> torch.Tensor:
"""

# --- attention.py: the SWA-only callsite threads the layout ------------------

ATTN4_SWACALL_OLD = """\
            self.attention_impl(
                None, qr, kv, None, None, None, positions, o_padded
            )
"""

ATTN4_SWACALL_NEW = """\
            self.attention_impl(
                None,
                qr,
                kv,
                None,
                None,
                None,
                positions,
                o_padded,
                sp_rows if sp_rows is not None else _sp,
            )
"""

# --- attention.py: the main callsite threads the layout ----------------------

ATTN4_CALL_OLD = """\
        self.attention_impl(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
"""

ATTN4_CALL_NEW = """\
        self.attention_impl(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
            sp_rows if sp_rows is not None else _sp,
        )
"""

# --- attention.py: attention_impl signature + the one stage gate -------------

ATTN4_IMPL_OLD = """\
    @eager_break_during_capture
    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
"""

ATTN4_IMPL_NEW = """\
    @eager_break_during_capture
    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
        sp_rows=None,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
# --- DSV4V_SP_PREFILL_S4 ---
        # Stage 4: the ONE runtime stage gate on this tree. A stage 1-3
        # boot leaves _sp4 None and every stage-4 branch below inert -
        # behavior identical to the v3 tree. sp_rows was threaded from the
        # decoder layer; no ContextVar, no module global, no env read.
        _sp4 = (
            sp_rows
            if (sp_rows is not None and _sp4_prefill.stages() >= 4)
            else None
        )
__SPDBG_IMPL__
# --- end DSV4V_SP_PREFILL_S4 ---
"""

# --- attention.py: the indexer lambda carries the layout ----------------------

ATTN4_LAMBDA_OLD = """\
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    ),
"""

ATTN4_LAMBDA_NEW = """\
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                        _sp4,
                    ),
"""

# --- attention.py: DeepseekV4Indexer.forward gains sp_rows --------------------

ATTN4_IDXSIG_OLD = """\
    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
"""

ATTN4_IDXSIG_NEW = """\
    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        sp_rows=None,
    ) -> torch.Tensor:
"""

# --- attention.py: the query-side slicing (rev-2 split point kept) ------------

ATTN4_OLD = """\
        def wq_b_and_q_quant():
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            q = q.view(-1, self.n_head, self.head_dim)
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
            )"""

ATTN4_NEW = """\
        def wq_b_and_q_quant():
            # ReplicatedLinear returns (output, bias); bias is None.
# --- DSV4V_SP_PREFILL_S4 ---
            # Prefill SP (Stage 4): this rank projects and quantizes only
            # its own rows. wq_b is FP8 with per-row scales (row-count
            # invariant) and fused_indexer_q_rope_quant is a per-token
            # kernel. The BF16 weights_proj ran upstream on the gathered
            # hidden state (full-width); only its per-row output is sliced
            # here. sp_rows is threaded from attention_impl. Split-check
            # mode keeps the full-row query so the op derives every slice
            # itself.
            _qr_in, _pos_in, _w_in = qr, positions, indexer_weights
            if (
                sp_rows is not None
                and _sp4_prefill.stages() >= 4
                and not _sp4_sai._SP4_SPLIT_CHECK
            ):
                _lo = sp_rows.start
                _hi = _lo + sp_rows.real_rows
                _qr_in = _qr_in[_lo:_hi]
                _pos_in = _pos_in[_lo:_hi]
                _w_in = _w_in[_lo:_hi]
# --- end DSV4V_SP_PREFILL_S4 ---
            q, _ = self.wq_b(_qr_in)
            q = q.view(-1, self.n_head, self.head_dim)
            return fused_indexer_q_rope_quant(
                _pos_in,
                q,
                rotary_emb.cos_sin_cache,
                _w_in,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
            )"""

# --- attention.py: the op call carries the ints -------------------------------

ATTN4_OPCALL_OLD = """\
        return self.indexer_op(hidden_states, q_quant, k, weights)"""

ATTN4_OPCALL_NEW = """\
# --- DSV4V_SP_PREFILL_S4 ---
__SPDBG_OPCALL__
        if sp_rows is not None and _sp4_prefill.stages() >= 4:
            # Runtime mode: q_quant/weights hold this rank's real rows and
            # the ints tell the op where they sit. Split-check mode still
            # passes the rank window; the op's own import-time constant
            # switches it to full-row tensors + derived slices.
            return self.indexer_op(
                hidden_states,
                q_quant,
                k,
                weights,
                sp_rows.start,
                sp_rows.real_rows,
            )
# --- end DSV4V_SP_PREFILL_S4 ---
        return self.indexer_op(hidden_states, q_quant, k, weights)"""

# --- attention.py: the publish (ONE default-stream all-gather) ----------------

ATTN4_PUBLISH_OLD = """\
        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)"""

ATTN4_PUBLISH_NEW = """\
# --- DSV4V_SP_PREFILL_S4 ---
        if _sp4 is not None and self.indexer is not None:
            # _sp4 publish: the indexer op (aux stream) wrote only this
            # rank's SP rows into the shared top-k buffer; gather the
            # rank-ordered slices so every rank covers all rows. Runs on
            # the DEFAULT stream, after the execute_in_parallel join,
            # before forward_mqa reads the buffer - issuing the collective
            # from inside the op was the rev-1 defect (aux-stream,
            # unordered against the default-stream SP collectives).
            _buf = self.indexer.topk_indices_buffer
            _sp4_end = _sp4.start + _sp4.local_rows
            if _sp4_end <= _buf.shape[0]:
                _loc = _buf[_sp4.start : _sp4_end]
            else:
                # Tp = ceil(T/N)*N can exceed the buffer's batch rows by
                # the world pad (e.g. 8192 % 3 != 0): pad the ragged tail
                # so the all-gather stays rectangular; pad rows are never
                # read downstream (consumers read [:num_decode] and
                # [num_decode : num_decode + num_prefill] only).
                _tail = _sp4_end - _buf.shape[0]
                _loc = torch.cat(
                    [
                        _buf[_sp4.start : _buf.shape[0]],
                        _buf.new_zeros((_tail,) + _buf.shape[1:]),
                    ],
                    dim=0,
                )
            _gathered = _sp4_all_gather(_loc, dim=0)
            # Shard padding: local_rows are equal-padded across ranks, so
            # the gathered stack can exceed the batch by up to world-size
            # pad rows - copy only the real rows. Exceeding by more means
            # a degenerate (dummy/profile) layout where every rank
            # already covered all rows.
            if _gathered.shape[0] <= _buf.shape[0] + 4:
                _n = min(_gathered.shape[0], _buf.shape[0])
                _buf[:_n].copy_(_gathered[:_n])
            else:
                logger.info_once(
                    "DSV4V indexer publish skipped: gathered %d rows vs "
                    "buffer %d (degenerate layout, rows already covered)",
                    _gathered.shape[0],
                    _buf.shape[0],
                )
# --- end DSV4V_SP_PREFILL_S4 ---
        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)"""

# --- sparse_attn_indexer.py: import-time check constant -----------------------

SAI4_CONST_OLD = """\
logger = init_logger(__name__)"""

SAI4_CONST_NEW = """\
logger = init_logger(__name__)
# --- DSV4V_SP_PREFILL_S4 ---
import os as _os

# Stage-4 split-check mode, read ONCE at import (never per call):
# DSV4V_INDEXER_SPLIT_CHECK=1 runs the full-row scorer twice beside the
# split (drift-aware validation). Optional debug builds print via
# DSV4V_SP_DEBUG - also read only here, never per call.
_SP4_SPLIT_CHECK = _os.environ.get("DSV4V_INDEXER_SPLIT_CHECK", "").strip() == "1"
_DSV4V_SPDBG = _os.environ.get("DSV4V_SP_DEBUG", "") not in ("", "0")
# --- end DSV4V_SP_PREFILL_S4 ---"""

# --- sparse_attn_indexer.py: op signature gains the ints ----------------------

SAI4_SIG_OLD = """\
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run"""

SAI4_SIG_NEW = """\
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    sp_row_start: int = -1,
    sp_real_rows: int = -1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run"""

# --- sparse_attn_indexer.py: fake impl signature ------------------------------

SAI4_FAKE_OLD = """\
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer"""

SAI4_FAKE_NEW = """\
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    sp_row_start: int = -1,
    sp_real_rows: int = -1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer"""

# --- sparse_attn_indexer.py: the resolve block (ints, fail-closed DCP) --------

SAI4_RESOLVE_OLD = """\
    # The buffer must be pre-filled with -1 (the \"no token\" sentinel) before the"""

SAI4_RESOLVE_NEW = """\
# --- DSV4V_SP_PREFILL_S4 ---
    # Prefill SP (Stage 4): each rank scores and selects only its own rows
    # of every prefill chunk. Rows are independent (cu_seqlen_ks/ke are
    # per-row KV bounds), so an offset-aligned row slice keeps every kernel
    # input valid - the metadata builder's qs_start query-slice precedent.
    # The layout arrives as plain ints from the attention layer: no
    # ContextVar (traced away in compiled regions), no module global
    # (constant-folded by Dynamo), no env reads in this hot path. Fails
    # closed under DCP. NOT the DCP top-k merge: that solves the candidate
    # axis (per-rank KV shards); this is the query axis, where the only
    # collective is the consumer-side row gather (attention.py, default
    # stream, after the execute_in_parallel join).
    _sp4_active = sp_row_start >= 0
    _sp4_check = _SP4_SPLIT_CHECK and _sp4_active
    if _sp4_active:
        if dcp_world_size > 1:
            raise NotImplementedError(
                "DSV4V SP stage 4 indexer row split does not compose with DCP."
            )
        logger.info_once(
            "DSV4V indexer split ACTIVE: rows [%d, %d) check=%s",
            sp_row_start,
            sp_row_start + sp_real_rows,
            _sp4_check,
        )
__SPDBG_SAI__
# --- end DSV4V_SP_PREFILL_S4 ---
    # The buffer must be pre-filled with -1 (the \"no token\" sentinel) before the"""

# --- sparse_attn_indexer.py: the drift-check helper (int-fed) ------------------

SAI4_HELPER_OLD = """\


@triton.jit
def _fused_indexer_q_rope_quant_kernel("""

SAI4_HELPER_NEW = """\


# --- DSV4V_SP_PREFILL_S4 ---
def _sp4_split_check(
    sp_row_start: int,
    sp_real_rows: int,
    topk_indices_buffer,
    q_quant,
    q_scale,
    weights,
    topk_tokens,
    hidden_states,
    meta,
    k_quant_full,
    k_scale_full,
    use_fp4_cache,
    kv_cache,
    layer_name,
):
    \"\"\"Drift-aware validation of the stage-4 indexer row split.

    Runs the FULL-row scoring twice beside the split and compares this
    rank's rows: the radix top-k keeps an arbitrary subset of the positions
    tied at its threshold, so two full-row runs differ too; the split
    passes when it differs from the first run no more than the second run
    does (the reference series' drift-aware pattern). Requires split-check
    mode, where the caller passed FULL-row q_quant/weights and the op
    derived every slice itself (offset base 0).
    \"\"\"
    device = hidden_states.device
    rows = hidden_states.shape[0]
    prefill = meta.prefill
    assert prefill is not None and q_quant.shape[0] == rows

    def full_row_pass(out):
        out.fill_(-1)
        for chunk in prefill.chunks:
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )
            if chunk.local_total_seq_lens == 0:
                continue
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            logits = fp8_fp4_mqa_logits(
                (q_slice_cast, q_scale_slice),
                (k_quant_cast, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                out[chunk.token_start : chunk.token_end, :topk_tokens],
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    ref1 = torch.full((rows, topk_tokens), -1, dtype=torch.int32, device=device)
    ref2 = torch.full((rows, topk_tokens), -1, dtype=torch.int32, device=device)
    full_row_pass(ref1)
    full_row_pass(ref2)

    # The split result lives in the shared buffer at this rank's global
    # rows (written by the loop above in check mode, offset base 0).
    actual = topk_indices_buffer[
        sp_row_start : sp_row_start + sp_real_rows, :topk_tokens
    ].clone()
    expected = ref1[sp_row_start : sp_row_start + sp_real_rows]
    repeat = ref2[sp_row_start : sp_row_start + sp_real_rows]

    def spread(a, e):
        a = a.sort(dim=-1).values
        e = e.sort(dim=-1).values
        where = torch.searchsorted(e, a).clamp_max(e.shape[-1] - 1)
        missing = ((e.gather(-1, where) != a) & (a >= 0)).sum(dim=-1)
        return (
            int((missing > 0).sum()),
            int(missing.sum()),
            int(missing.max()) if missing.numel() else 0,
        )

    split_rows, split_pos, split_max = spread(actual, expected)
    full_rows, full_pos, full_max = spread(repeat, expected)
    unwritten = int((actual == -1).all(dim=-1).sum())
    log = logger.warning if unwritten or split_rows > full_rows else logger.info
    log(
        "DSV4V indexer split check: layer %s rows %d-%d of %d: "
        "split vs full %d rows, %d positions (max %d); full vs full %d rows, "
        "%d positions (max %d); %d all-sentinel rows",
        layer_name,
        sp_row_start,
        sp_row_start + sp_real_rows,
        rows,
        split_rows,
        split_pos,
        split_max,
        full_rows,
        full_pos,
        full_max,
        unwritten,
    )
# --- end DSV4V_SP_PREFILL_S4 ---


@triton.jit
def _fused_indexer_q_rope_quant_kernel("""

# --- sparse_attn_indexer.py: the chunk-loop intersection split -----------------

SAI4_LOOP_OLD = """\
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError(\"XPU fp8_mqa_logits does not support FP4 Q\")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )"""

SAI4_LOOP_NEW = """\
# --- DSV4V_SP_PREFILL_S4 ---
            # Stage 4: this rank's intersection with the chunk, fed by the
            # threaded ints (runtime mode) or by full-row derivation
            # (check mode, offset base 0).
            _sp4_lo = _sp4_hi = -1
            if _sp4_active:
                _sp4_lo = max(chunk.token_start, sp_row_start)
                _sp4_hi = min(chunk.token_end, sp_row_start + sp_real_rows)
            if _sp4_active and _sp4_hi <= _sp4_lo:
                # Every row of this chunk belongs to another rank; the
                # consumer-side publish covers these rows.
                pass
            elif _sp4_active:
                _qb = 0 if _sp4_check else sp_row_start
                q_slice = q_quant[_sp4_lo - _qb : _sp4_hi - _qb]
                q_scale_slice = (
                    q_scale[_sp4_lo - _qb : _sp4_hi - _qb]
                    if q_scale is not None
                    else None
                )
                _ks = chunk.cu_seqlen_ks[
                    _sp4_lo - chunk.token_start : _sp4_hi - chunk.token_start
                ]
                _ke = chunk.cu_seqlen_ke[
                    _sp4_lo - chunk.token_start : _sp4_hi - chunk.token_start
                ]
                # Write directly into the SHARED buffer at this rank's
                # global row offset (the consumer-side publish gathers the
                # rank slices on the default stream).
                topk_indices = topk_indices_buffer[
                    _sp4_lo:_sp4_hi, :topk_tokens
                ]
                _weights_slice = weights[_sp4_lo - _qb : _sp4_hi - _qb]
                if chunk.local_total_seq_lens == 0:
                    topk_indices.fill_(-1)
                else:
                    if use_fp4_cache:
                        q_slice_cast = q_slice.view(torch.int8)
                        k_quant_cast = k_quant.view(torch.int8)
                        k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                    else:
                        q_slice_cast = q_slice
                        k_quant_cast = k_quant
                        k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        _weights_slice,
                        _ks,
                        _ke,
                        clean_logits=False,
                    )
                    num_rows = logits.shape[0]
                    ops.top_k_per_row_prefill(
                        logits,
                        _ks,
                        _ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )
            else:
# --- end DSV4V_SP_PREFILL_S4 ---
                q_slice = q_quant[chunk.token_start : chunk.token_end]
                q_scale_slice = (
                    q_scale[chunk.token_start : chunk.token_end]
                    if q_scale is not None
                    else None
                )
                topk_indices = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

                if chunk.local_total_seq_lens == 0:
                    logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                    topk_indices.fill_(-1)
                else:
                    # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                    # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                    if use_fp4_cache:
                        q_slice_cast = q_slice.view(torch.int8)
                        k_quant_cast = k_quant.view(torch.int8)
                        k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                    else:
                        q_slice_cast = q_slice
                        k_quant_cast = k_quant
                        k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                    if current_platform.is_xpu():
                        if q_scale_slice is not None:
                            raise RuntimeError(\"XPU fp8_mqa_logits does not support FP4 Q\")
                        logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                            q_slice_cast,
                            k_quant_cast,
                            k_scale_cast,
                            weights[chunk.token_start : chunk.token_end],
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                        )
                    else:
                        logits = fp8_fp4_mqa_logits(
                            (q_slice_cast, q_scale_slice),
                            (k_quant_cast, k_scale_cast),
                            weights[chunk.token_start : chunk.token_end],
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            clean_logits=False,
                        )
                    num_rows = logits.shape[0]
                    ops.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )

                _merge_dcp_topk_global(
                    logits,
                    topk_indices,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=chunk.cu_seqlen_ks,
                )"""

# --- sparse_attn_indexer.py: the tail (split-check invocation) -----------------

SAI4_TAIL_OLD = """\
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer"""

SAI4_TAIL_NEW = """\
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

# --- DSV4V_SP_PREFILL_S4 ---
    if _sp4_check and has_prefill:
        # Drift-aware validation: the full-row scorer ran twice beside the
        # split; compare per-layer selection drift against the full-row
        # path's own run-to-run drift.
        _sp4_split_check(
            sp_row_start,
            sp_real_rows,
            topk_indices_buffer,
            q_quant,
            q_scale,
            weights,
            topk_tokens,
            hidden_states,
            attn_metadata_narrowed,
            k_quant_full,
            k_scale_full,
            use_fp4_cache,
            kv_cache,
            k_cache_prefix,
        )
# --- end DSV4V_SP_PREFILL_S4 ---
    return topk_indices_buffer"""

# --- sparse_attn_indexer.py: forward_cuda passthrough --------------------------

SAI4_FWCUDA_OLD = """\
    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):"""

SAI4_FWCUDA_NEW = """\
    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        sp_row_start: int = -1,
        sp_real_rows: int = -1,
    ):
        if _DSV4V_SPDBG:
            print(
                f"[SPDBG fwcuda] rows={hidden_states.shape[0]} "
                f"ints=({sp_row_start},{sp_real_rows})",
                flush=True,
            )"""

SAI4_NATIVE_OLD = """\
    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):"""

SAI4_NATIVE_NEW = """\
    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        sp_row_start: int = -1,
        sp_real_rows: int = -1,
    ):"""

SAI4_NATCALL_OLD = """\
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)"""

SAI4_NATCALL_NEW = """\
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(
                hidden_states, q_quant, k, weights, sp_row_start, sp_real_rows
            )"""

SAI4_FWCALL_OLD = """\
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )"""

SAI4_FWCALL_NEW = """\
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            sp_row_start,
            sp_real_rows,
        )"""

# Debug-build fragments (spliced only when the applier runs --emit-debug;
# the performance build contains no prints and no env reads).
SPDBG_OPCALL = (
    "        import os as _dbg_os2, sys as _dbg_sys2\n"
    "\n"
    "        print(\n"
    "            f\"[SPDBG opcall] sp_rows={sp_rows is not None} \"\n"
    "            f\"stages={_sp4_prefill.stages()}\",\n"
    "            file=_dbg_sys2.stderr,\n"
    "            flush=True,\n"
    "        )\n"
)

SPDBG_SAI = (
    "    if _DSV4V_SPDBG:\n"
    "        import sys as _sys\n"
    "\n"
    "        print(\n"
    "            f\"[SPDBG sai] active={_sp4_active} \"\n"
    "            f\"rows={hidden_states.shape[0]} \"\n"
    "            f\"window=[{sp_row_start},{sp_row_start + sp_real_rows})\",\n"
    "            file=_sys.stderr,\n"
    "            flush=True,\n"
    "        )\n"
)
SPDBG_IMPL = (
    "        import os as _dbg_os, sys as _dbg_sys\n"
    "\n"
    "        if _dbg_os.environ.get(\"DSV4V_SP_DEBUG\", \"\") not in (\"\", \"0\"):\n"
    "            _dbg_rows = (\n"
    "                hidden_states.shape[0] if hidden_states is not None else -1\n"
    "            )\n"
    "            print(\n"
    "                f\"[SPDBG impl] rows={_dbg_rows} \"\n"
    "                f\"sp4={_sp4 is not None} idx={self.indexer is not None}\",\n"
    "                file=_dbg_sys.stderr,\n"
    "                flush=True,\n"
    "            )\n"
)


# --- model.py: the decoder threads the layout into attention (E2) -------------

MODEL4_SPROWS_OLD = """\
                if _sp_mod.stages() >= 3:
                    # Stage 3: attention owns the row gather - SWA-only
                    # layers run the front on these local rows and gather
                    # the narrow product; the rest gather the hidden state.
                    _attn_out = self.attn(positions, x, None)"""

MODEL4_SPROWS_NEW = """\
                if _sp_mod.stages() >= 3:
                    # Stage 3: attention owns the row gather - SWA-only
                    # layers run the front on these local rows and gather
                    # the narrow product; the rest gather the hidden state.
                    _attn_out = self.attn(positions, x, None, sp_rows=sp_rows)"""


def fail_closed(text: str, needle: str, what: str, expect: int = 1) -> None:
    n = text.count(needle)
    if n != expect:
        raise SystemExit(
            f"ANCHOR FAIL ({what}): expected {expect} occurrence(s), found {n}. "
            "Refusing to edit - the image tree has diverged from the tested "
            "revision; re-derive the anchors before applying."
        )


def read(root: pathlib.Path, rel: pathlib.Path) -> str:
    return (root / rel).read_text(encoding="utf-8")


def write(root: pathlib.Path, rel: pathlib.Path, text: str, suffix: str) -> None:
    target = root / rel
    backup = target.with_name(target.name + suffix)
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(text, encoding="utf-8")


def apply(
    root: pathlib.Path,
    check_only: bool,
    stages: int = 3,
    emit_debug: bool = False,
) -> list[str]:
    status: list[str] = []

    # Pre-flight: every file we edit must exist BEFORE anything is written
    # (a wrong path once applied Edits 1-6 and died on Edit 7 mid-boot).
    for rel in (
        V4_MODEL_REL,
        VOCAB_REL,
        RUNNER_REL,
        OPROJ_REL,
        ATTENTION_REL,
        SAI_REL,
    ):
        if not (root / rel).exists():
            raise SystemExit(
                f"PRE-FLIGHT FAIL: {rel} not found under {root} - wrong "
                "--vllm-root or the image layout changed. Nothing was written."
            )

    # Edit 1: inject (or upgrade to v5) the sp_prefill module.
    module_path = root / SP_MODULE_REL
    module_done = False
    if module_path.exists():
        text = module_path.read_text(encoding="utf-8")
        if MODULE_V6_TAG in text or MODULE_V3_TAG in text:
            # rev 3: the embedded module IS the stage-3 text; never
            # rewrite an already-injected module at stages<=3 or the apply
            # would diverge from the serving tree (byte-identity gate).
            if (
                stages >= 4
                and MODULE_V3_TAG in text
                and "stages not in (1, 2, 3)" in text
            ):
                # Stage-4 tree: widen the runtime stage range IN PLACE
                # (the serving tree's module is already injected, so the
                # fresh-injection widening below never runs on upgrades).
                if not check_only:
                    backup = module_path.with_name(module_path.name + ".spprebak")
                    if not backup.exists():
                        shutil.copy2(module_path, backup)
                    text = text.replace(
                        "    if stages not in (1, 2, 3):",
                        "    if stages not in (1, 2, 3, 4):",
                        1,
                    ).replace(
                        'raise ValueError(f"{STAGES_ENV} must be 1, 2 or 3, got {stages}")',
                        'raise ValueError('
                        'f"{STAGES_ENV} must be 1, 2, 3 or 4, got {stages}")',
                        1,
                    )
                    assert "1, 2, 3, 4" in text, "stage-4 widening failed"
                    module_path.write_text(text, encoding="utf-8")
                    py_compile.compile(str(module_path), doraise=True)
                status.append(
                    f"{'APPLY  ' if check_only else 'APPLIED'} {SP_MODULE_REL}: "
                    "stage-4 range widened in place"
                )
            else:
                status.append(f"OK      {SP_MODULE_REL}: already injected")
            module_done = True
        elif MARK not in text:
            status.append(f"FATAL   {SP_MODULE_REL}: foreign file in place")
            return status
        # else: an older OURS module with an unknown tag - upgrade below.
    if not module_done:
        if check_only:
            status.append(f"APPLY   {SP_MODULE_REL}: would inject")
        else:
            module_path.parent.mkdir(parents=True, exist_ok=True)
            backup = module_path.with_name(module_path.name + ".spprebak")
            if module_path.exists() and not backup.exists():
                shutil.copy2(module_path, backup)
            module_text = SP_PREFILL_MODULE
            if stages >= 4:
                # Stage-4 tree: the runtime module must ACCEPT stage 4.
                # Only the validation range widens - everything else stays
                # byte-identical to the stage-3 module, so a stages<=3
                # apply still emits the module untouched (byte-identity gate).
                module_text = module_text.replace(
                    "    if stages not in (1, 2, 3):",
                    "    if stages not in (1, 2, 3, 4):",
                    1,
                ).replace(
                    'raise ValueError(f"{STAGES_ENV} must be 1, 2 or 3, got {stages}")',
                    'raise ValueError('
                    'f"{STAGES_ENV} must be 1, 2, 3 or 4, got {stages}")',
                    1,
                )
                assert "1, 2, 3, 4" in module_text, "stage-4 widening failed"
            module_path.write_text(module_text, encoding="utf-8")
            py_compile.compile(str(module_path), doraise=True)
            status.append(
                f"APPLIED {SP_MODULE_REL}: injected (stage-3 module"
                + (", stage-4 range" if stages >= 4 else "")
                + ")"
            )

    # Edits 2+3: the V4 model Stage-1 wiring + the embedding method.
    # The Stage-1 model replacements are applied to the in-memory text even
    # in check mode so the Stage-2 model anchors below validate honestly.
    model = read(root, V4_MODEL_REL)
    model_s1_applied = MARK in model
    if model_s1_applied:
        status.append(f"OK      {V4_MODEL_REL}: stage 1 already patched")
    else:
        fail_closed(model, MODEL_INIT_ANCHOR, "model init anchor")
        fail_closed(model, MODEL_FORWARD_OLD, "model forward anchor")
        model = model.replace(MODEL_INIT_ANCHOR, MODEL_INIT_INSERT, 1)
        model = model.replace(MODEL_FORWARD_OLD, MODEL_FORWARD_NEW, 1)
        if check_only:
            status.append(f"APPLY   {V4_MODEL_REL}: would patch init+forward (stage 1)")
        else:
            write(root, V4_MODEL_REL, model, ".spprebak")
            py_compile.compile(str(root / V4_MODEL_REL), doraise=True)
            status.append(f"APPLIED {V4_MODEL_REL}: init+forward wired (stage 1)")

    vocab = read(root, VOCAB_REL)
    if MARK in vocab:
        status.append(f"OK      {VOCAB_REL}: already patched")
    else:
        fail_closed(vocab, VOCAB_IMPORT_OLD, "vocab import anchor")
        fail_closed(vocab, VOCAB_METHOD_ANCHOR, "vocab method anchor")
        if check_only:
            status.append(f"APPLY   {VOCAB_REL}: would add import+method")
        else:
            vocab = vocab.replace(VOCAB_IMPORT_OLD, VOCAB_IMPORT_NEW, 1)
            vocab = vocab.replace(VOCAB_METHOD_ANCHOR, VOCAB_METHOD_NEW, 1)
            write(root, VOCAB_REL, vocab, ".spprebak")
            py_compile.compile(str(root / VOCAB_REL), doraise=True)
            status.append(f"APPLIED {VOCAB_REL}: import+forward_reduce_scatter")

    # Edit 4: the MoE runner reduce-scatter (stage 2).
    runner = read(root, RUNNER_REL)
    if MARK2 in runner:
        status.append(f"OK      {RUNNER_REL}: stage 2 already patched")
    else:
        pairs = [
            (RUNNER_IMPORT_OLD, RUNNER_IMPORT_NEW, "runner import anchor"),
            (RUNNER_METHOD_ANCHOR, RUNNER_METHOD_NEW, "runner method anchor"),
            (RUNNER_FORWARD_OLD, RUNNER_FORWARD_NEW, "runner forward anchor"),
            (RUNNER_COMBINE_OLD, RUNNER_COMBINE_NEW, "runner combine anchor"),
        ]
        for old, _new, what in pairs:
            fail_closed(runner, old, what)
        if check_only:
            status.append(f"APPLY   {RUNNER_REL}: would patch stage 2 (RS)")
        else:
            for old, new, _what in pairs:
                runner = runner.replace(old, new, 1)
            write(root, RUNNER_REL, runner, ".spprebak")
            py_compile.compile(str(root / RUNNER_REL), doraise=True)
            status.append(f"APPLIED {RUNNER_REL}: stage 2 reduce-scatter")

    # Edit 5: the V4 model stage-2 threading (model + decoder + MoE wrapper).
    if MARK2 in model:
        status.append(f"OK      {V4_MODEL_REL}: stage 2 already patched")
    else:
        pairs = [
            (MODEL2_INIT_OLD, MODEL2_INIT_NEW, "model2 init anchor"),
            (MODEL2_SIG_OLD, MODEL2_SIG_NEW, "model2 forward-sig anchor"),
            (MODEL2_EMBED_OLD, MODEL2_EMBED_NEW, "model2 embedding anchor"),
            (MODEL2_LAYER_OLD, MODEL2_LAYER_NEW, "model2 layer-call anchor"),
            (MODEL2_AUX_OLD, MODEL2_AUX_NEW, "model2 aux anchor"),
            (MODEL2_INTERMED_OLD, MODEL2_INTERMED_NEW, "model2 intermediate anchor"),
            (MODEL2_MTP_OLD, MODEL2_MTP_NEW, "model2 mtp-buffer anchor"),
            (MODEL2_FINAL_OLD, MODEL2_FINAL_NEW, "model2 final-gather anchor"),
            (MODEL2_MOEFWD_OLD, MODEL2_MOEFWD_NEW, "model2 moe forward anchor"),
            (MODEL2_MOEFUSED_OLD, MODEL2_MOEFUSED_NEW, "model2 fused-moe anchor"),
            (MODEL2_DECODER_SIG_OLD, MODEL2_DECODER_SIG_NEW, "model2 decoder sig anchor"),
            (MODEL2_DECODER_ATTN_OLD, MODEL2_DECODER_ATTN_NEW, "model2 decoder attn anchor"),
            (MODEL2_DECODER_FFN_OLD, MODEL2_DECODER_FFN_NEW, "model2 decoder ffn anchor"),
        ]
        for old, _new, what in pairs:
            fail_closed(model, old, what)
        # Mutate in memory even in check mode so the stage-3 model anchor
        # below validates against the post-stage-2 text.
        for old, new, _what in pairs:
            model = model.replace(old, new, 1)
        if check_only:
            status.append(f"APPLY   {V4_MODEL_REL}: would patch stage 2 (threading)")
        else:
            write(root, V4_MODEL_REL, model, ".spprebak")
            py_compile.compile(str(root / V4_MODEL_REL), doraise=True)
            status.append(f"APPLIED {V4_MODEL_REL}: stage 2 threading")

    # Edit 6: the shared o_proj op (covers both FlashInfer impls).
    oproj = read(root, OPROJ_REL)
    if MARK2 in oproj:
        status.append(f"OK      {OPROJ_REL}: stage 2 already patched")
    else:
        fail_closed(oproj, OPROJ_OLD, "o_proj tail anchor")
        if check_only:
            status.append(f"APPLY   {OPROJ_REL}: would patch stage 2 (partial wo_b)")
        else:
            oproj = oproj.replace(OPROJ_OLD, OPROJ_NEW, 1)
            write(root, OPROJ_REL, oproj, ".spprebak")
            py_compile.compile(str(root / OPROJ_REL), doraise=True)
            status.append(f"APPLIED {OPROJ_REL}: stage 2 partial wo_b")

    # Edit 7: the attention narrow front (stage 3, SWA-only layers).
    attention = read(root, ATTENTION_REL)
    if MARK3 in attention:
        status.append(f"OK      {ATTENTION_REL}: stage 3 already patched")
    else:
        fail_closed(attention, ATTN3_FORWARD_OLD, "attention forward anchor")
        if check_only:
            status.append(f"APPLY   {ATTENTION_REL}: would patch stage 3 (narrow front)")
        else:
            attention = attention.replace(ATTN3_FORWARD_OLD, ATTN3_FORWARD_NEW, 1)
            write(root, ATTENTION_REL, attention, ".spprebak")
            py_compile.compile(str(root / ATTENTION_REL), doraise=True)
            status.append(f"APPLIED {ATTENTION_REL}: stage 3 narrow front")

    # Edit 8: the decoder passes local rows under stage 3 (model.py).
    if MARK3 in model:
        status.append(f"OK      {V4_MODEL_REL}: stage 3 already patched")
    else:
        fail_closed(model, MODEL3_DECODER_OLD, "model3 decoder anchor")
        model = model.replace(MODEL3_DECODER_OLD, MODEL3_DECODER_NEW, 1)
        if check_only:
            status.append(f"APPLY   {V4_MODEL_REL}: would patch stage 3 (decoder)")
        else:
            write(root, V4_MODEL_REL, model, ".spprebak")
            py_compile.compile(str(root / V4_MODEL_REL), doraise=True)
            status.append(f"APPLIED {V4_MODEL_REL}: stage 3 decoder")

    # ------------------------------------------------------------------
    # Edits 9-10 (stage 4): emitted ONLY when the target boot stage is 4.
    # A stages<=3 apply writes ZERO MARK4 bytes, so the resulting tree is
    # byte-identical to the stage-3 output (E1; even runtime-inert stage-4
    # branches in the eager attention path measured 3-4x slower warm
    # throughput). --emit-debug additionally splices the SPDBG
    # print blocks (debug builds only; the performance build has no
    # prints and no env reads anywhere in the emitted code).
    # ------------------------------------------------------------------
    if stages >= 4:
        # Edit 9: attention.py - hoisted imports, threading, publish,
        # indexer split closure + op call.
        if MARK4 in attention:
            status.append(f"OK      {ATTENTION_REL}: stage 4 already patched")
        else:
            attn_pairs = [
                (ATTN4_IMPORTS_OLD, ATTN4_IMPORTS_NEW, "attention import anchor"),
                (ATTN4_FWD_OLD, ATTN4_FWD_NEW, "attention forward signature anchor"),
                (ATTN4_SWACALL_OLD, ATTN4_SWACALL_NEW, "attention SWA callsite anchor"),
                (ATTN4_CALL_OLD, ATTN4_CALL_NEW, "attention callsite anchor"),
                (ATTN4_IMPL_OLD, ATTN4_IMPL_NEW, "attention_impl signature anchor"),
                (ATTN4_LAMBDA_OLD, ATTN4_LAMBDA_NEW, "indexer lambda anchor"),
                (ATTN4_IDXSIG_OLD, ATTN4_IDXSIG_NEW, "indexer forward signature anchor"),
                (ATTN4_OLD, ATTN4_NEW, "attention stage-4 closure anchor"),
                (ATTN4_OPCALL_OLD, ATTN4_OPCALL_NEW, "indexer_op callsite anchor"),
                (ATTN4_PUBLISH_OLD, ATTN4_PUBLISH_NEW, "attention publish anchor"),
            ]
            for old, _new, what in attn_pairs:
                fail_closed(attention, old, what)
            if check_only:
                status.append(
                    f"APPLY   {ATTENTION_REL}: would patch stage 4 (rev 3 threading+publish)"
                )
            else:
                for old, new, _what in attn_pairs:
                    attention = attention.replace(old, new, 1)
                attention = attention.replace(
                    "__SPDBG_IMPL__", SPDBG_IMPL if emit_debug else ""
                )
                attention = attention.replace(
                    "__SPDBG_OPCALL__", SPDBG_OPCALL if emit_debug else ""
                )
                assert "__SPDBG" not in attention, "debug placeholder leaked"
                write(root, ATTENTION_REL, attention, ".spprebak")
                py_compile.compile(str(root / ATTENTION_REL), doraise=True)
                status.append(f"APPLIED {ATTENTION_REL}: stage 4 (rev 3)")

        # Edit 10: the indexer op (constant, signatures, resolve, helper,
        # loop split, tail check, forward_cuda passthrough).
        sai = read(root, SAI_REL)
        if MARK4 in sai:
            status.append(f"OK      {SAI_REL}: stage 4 already patched")
        else:
            sai_pairs = [
                (SAI4_CONST_OLD, SAI4_CONST_NEW, "sai logger anchor"),
                (SAI4_SIG_OLD, SAI4_SIG_NEW, "sai signature anchor"),
                (SAI4_FAKE_OLD, SAI4_FAKE_NEW, "sai fake anchor"),
                (SAI4_RESOLVE_OLD, SAI4_RESOLVE_NEW, "sai resolve anchor"),
                (SAI4_HELPER_OLD, SAI4_HELPER_NEW, "sai helper anchor"),
                (SAI4_LOOP_OLD, SAI4_LOOP_NEW, "sai chunk-loop anchor"),
                (SAI4_TAIL_OLD, SAI4_TAIL_NEW, "sai tail anchor"),
                (SAI4_FWCUDA_OLD, SAI4_FWCUDA_NEW, "sai forward_cuda anchor"),
                (SAI4_NATIVE_OLD, SAI4_NATIVE_NEW, "sai forward_native anchor"),
                (SAI4_NATCALL_OLD, SAI4_NATCALL_NEW, "sai native-routing anchor"),
                (SAI4_FWCALL_OLD, SAI4_FWCALL_NEW, "sai op-call anchor"),
            ]
            for old, _new, what in sai_pairs:
                fail_closed(sai, old, what)
            # Edit 11: model.py threads sp_rows into the attention call (E2).
            if MODEL4_SPROWS_NEW in model:
                status.append(f"OK      {V4_MODEL_REL}: stage 4 already patched")
            else:
                fail_closed(model, MODEL4_SPROWS_OLD, "model stage-4 sp_rows anchor")
                if not check_only:
                    model = model.replace(MODEL4_SPROWS_OLD, MODEL4_SPROWS_NEW, 1)
                    write(root, V4_MODEL_REL, model, ".spprebak")
                    py_compile.compile(str(root / V4_MODEL_REL), doraise=True)
            if check_only:
                status.append(
                    f"APPLY   {SAI_REL}: would patch stage 4 (rev 3 int-threaded split)"
                )
                status.append(
                    f"APPLY   {V4_MODEL_REL}: would patch stage 4 (sp_rows threading)"
                )
            else:
                for old, new, _what in sai_pairs:
                    sai = sai.replace(old, new, 1)
                sai = sai.replace(
                    "__SPDBG_SAI__", SPDBG_SAI if emit_debug else ""
                )
                assert "__SPDBG" not in sai, "debug placeholder leaked"
                write(root, SAI_REL, sai, ".spprebak")
                py_compile.compile(str(root / SAI_REL), doraise=True)
                status.append(f"APPLIED {SAI_REL}: stage 4 split (rev 3)")
                status.append(f"APPLIED {V4_MODEL_REL}: stage 4 sp_rows threading")

    else:
        status.append(
            f"SKIP    stage-4 edits: target stage {stages} <= 3 - zero MARK4 "
            "emitted (byte-identical to the v3 tree)"
        )

    if stages <= 3:
        # E1 gate: no stage-4 bytes may exist in the produced tree.
        for rel in (V4_MODEL_REL, VOCAB_REL, RUNNER_REL, OPROJ_REL, ATTENTION_REL, SAI_REL):
            if MARK4 in read(root, rel):
                raise SystemExit(
                    f"STAGE GATE FAIL: MARK4 present in {rel} after a "
                    f"stages={stages} apply - the tree is not stage-3-identical."
                )
        status.append(
            "GATE     stages<=3: zero MARK4 in the produced tree "
            "(stage-3-identical; module range may be widened by an earlier "
            "stage-4 apply - functionally identical for stages<=3 boots)"
        )

    return status


def revert(root: pathlib.Path) -> list[str]:
    status: list[str] = []
    for rel in (
        V4_MODEL_REL,
        VOCAB_REL,
        RUNNER_REL,
        OPROJ_REL,
        ATTENTION_REL,
        SAI_REL,
    ):
        target = root / rel
        backup = target.with_name(target.name + ".spprebak")
        if backup.exists():
            shutil.copy2(backup, target)
            backup.unlink()
            status.append(f"REVERTED {rel}")
        elif any(
            m in target.read_text(encoding="utf-8")
            for m in (MARK, MARK2, MARK3, MARK4)
        ):
            status.append(f"DIRTY   {rel}: marker present but no backup")
        else:
            status.append(f"SKIP     {rel}: no backup")
    module_path = root / SP_MODULE_REL
    if module_path.exists():
        if MARK in module_path.read_text(encoding="utf-8"):
            module_path.unlink()
            status.append(f"REVERTED {SP_MODULE_REL} (removed injected module)")
        else:
            status.append(f"SKIP     {SP_MODULE_REL}: not ours, left in place")
    else:
        status.append(f"SKIP     {SP_MODULE_REL}: absent")
    module_backup = module_path.with_name(module_path.name + ".spprebak")
    if module_backup.exists():
        # A backup only exists when an OLDER version of our own module was
        # upgraded in place; pristine state is "no module", so drop it.
        if MARK in module_backup.read_text(encoding="utf-8"):
            module_backup.unlink()
            status.append(f"REVERTED {SP_MODULE_REL}.spprebak (stale upgrade copy)")
        else:
            status.append(f"SKIP     {SP_MODULE_REL}.spprebak: not ours")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-root",
        type=pathlib.Path,
        default=pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm"),
    )
    parser.add_argument("--check", action="store_true", help="report only")
    parser.add_argument("--revert", action="store_true", help="restore backups")
    parser.add_argument(
        "--stages",
        type=int,
        default=None,
        help="target boot stage 1-4 for THIS apply (default: "
        "DSV4V_PREFILL_SP_STAGES at apply time, else 3). A stages<=3 apply "
        "emits zero stage-4 bytes (stage-3-identical tree).",
    )
    parser.add_argument(
        "--emit-debug",
        action="store_true",
        help="include the SPDBG print blocks (debug builds only)",
    )
    args = parser.parse_args()

    if args.revert:
        for line in revert(args.vllm_root):
            print(line)
        return 0

    if args.stages is None:
        raw = os.environ.get(STAGES_ENV, "").strip()
        stages = int(raw) if raw else 3
    else:
        stages = args.stages
    if stages not in (1, 2, 3, 4):
        raise SystemExit(
            f"--stages must be 1-4 (or unset for {STAGES_ENV}), got {stages}"
        )

    emit_debug = args.emit_debug or (
        os.environ.get("DSV4V_SP_DEBUG", "").strip() not in ("", "0")
    )
    for line in apply(args.vllm_root, args.check, stages=stages, emit_debug=emit_debug):
        print(line)
    print(
        f"SP prefill apply target stage {stages}: "
        + ("dry-run complete" if args.check else "applied")
        + ". Gates: DSV4V_PREFILL_SP_MIN_ROWS (unset/0 = fully off);"
        " DSV4V_PREFILL_SP_STAGES (unset/1 = stage 1, 2 = sharded dataflow,"
        " 3 = + local SWA-only attention front, 4 = + indexer row split);"
        " DSV4V_INDEXER_SPLIT_CHECK=1 (stage 4 validation: full-row"
        " reference beside the split, drift-aware logging)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
