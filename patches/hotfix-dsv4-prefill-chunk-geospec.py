#!/usr/bin/env python3
"""Compile the prefill chunk metadata Triton kernel exactly once per boot.

`_build_prefill_chunk_metadata_kernel` (vllm/v1/attention/backends/mla/
indexer.py) is launched once per indexer prefill chunk with the Python ints
`query_slice_start` / `query_slice_stop` (sub-chunk bounds produced by
split_indexer_prefill_chunks: `max_q = max_logits_elems // chunk_n`, an
arbitrary integer that depends on packed context depth x concurrency) and the
sliced tensor `uncompressed_seq_lens[start_idx:end_idx]`. Triton 3.6 keys the
kernel cache on, per int arg: value==1 -> baked constexpr, %16==0 ->
tt.divisibility=16, else plain; per pointer arg: data_ptr()%16==0 -> "D".
The (depth x concurrency) traffic therefore generates new key combinations
mid-serve: on a persistent cache this single kernel accumulated dozens of
compiled variants spanning exactly three binary axes (query_slice_start
D/-/baked-1, query_slice_stop D/-, uncompressed_seq_lens_ptr D/-), and a
combination first minted by two concurrent ~52K-token prefills compiled on
all 3 ranks inside a TP collective and preceded a hard lane wedge. The
stock warmup (sparse_mla_triton_warmup._warm_prefill_chunk_metadata_
kernel) only enumerates a toy grid (query_len=8, slice offsets {0,1}x{0,-1}),
so it can never cover the residue classes real traffic produces.

Fix (same class as the issue #133 hotfix, and the same pattern this vLLM
already uses on _compute_swa_indices_and_lens_kernel: `@triton.jit(
do_not_specialize=["token_offset"])` in sparse_swa.py): stop specializing the
two geometry ints and the sliced-pointer alignment. All three are used as
scalar mask bounds / a scalar load base, so the lost hints cannot vectorize
anything; results are bit-identical (hints and constexpr-baking affect
codegen only). One variant then serves every geometry, compiled during the
existing boot warmup that already invokes this kernel.

Rollback: DSPARK_SKIP_PREFILL_CHUNK_GEOSPEC_HOTFIX=1 (patch not applied), or
remove the compose mount+invoke lines.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
)
MARK = "# [prefill-chunk-geospec-hotfix] do_not_specialize query slice bounds + sliced seq-lens ptr"

OLD = '''@triton.jit
def _build_prefill_chunk_metadata_kernel(
    # Inputs
    query_start_loc_ptr,
'''

NEW = '''@triton.jit(
    do_not_specialize=["query_slice_start", "query_slice_stop"],
    do_not_specialize_on_alignment=["uncompressed_seq_lens_ptr"],
)
''' + MARK + '''
def _build_prefill_chunk_metadata_kernel(
    # Inputs
    query_start_loc_ptr,
'''

DNS_ARGS = ("query_slice_start", "query_slice_stop")
DNS_ALIGN_ARGS = ("uncompressed_seq_lens_ptr",)


def _decorator_targets_valid(source: str) -> bool:
    """The kwargs we pin must still name real parameters of the kernel."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_build_prefill_chunk_metadata_kernel"
        ):
            params = {a.arg for a in node.args.args}
            return set(DNS_ARGS) <= params and set(DNS_ALIGN_ARGS) <= params
    return False


def patch_text(source: str) -> tuple[str, str]:
    if MARK in source and "do_not_specialize" in source:
        return source, "skipped"
    if source.count(OLD) != 1:
        return source, f"drift:old={source.count(OLD)}"
    updated = source.replace(OLD, NEW, 1)
    if not _decorator_targets_valid(updated):
        return source, "drift:kwargs-not-in-signature"
    try:
        compile(updated, "indexer.py", "exec")
    except SyntaxError as exc:
        return source, f"drift:syntax:{exc}"
    return updated, "applied"


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__.strip() if __doc__ else __file__)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        status_src = P.read_text() if P.is_file() else ""
        print(
            "prefill chunk geospec          :",
            "APPLIED" if MARK in status_src else "NOT APPLIED",
        )
        return 0
    if not P.is_file():
        print(f"[prefill-chunk-geospec-hotfix] missing {P}", file=sys.stderr)
        return 1
    src = P.read_text()
    updated, status = patch_text(src)
    if status == "skipped":
        print(f"[prefill-chunk-geospec-hotfix] already applied to {P}")
        return 0
    if status != "applied":
        print(
            f"[prefill-chunk-geospec-hotfix] refusing to patch ({status})",
            file=sys.stderr,
        )
        return 1
    P.write_text(updated)
    print(
        f"[prefill-chunk-geospec-hotfix] applied to {P} "
        "(one variant for all prefill chunk geometries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
