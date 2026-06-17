# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tensor.paged_gather (paged gather directly into L1 / UB).

The op lowers (in ConvertTensorToTileOps) into a fully-scalar per-row GM->on-chip
load loop on the Cube core:

    acc = tile.create([max_indices, size], target_memory=Mat)
    for i in range(rows):
        idx  = tensor.read(indices, [i])          # scalar GM read (pto.load_scalar)
        phys = block_table[idx // bs] * bs + idx % bs   # scalar
        row  = tile.load(src, [phys, col_off], [1, size], target_memory=Mat)  # GM->L1
        acc  = tile.assemble(acc, row, [i, 0])

Only the small index/page-table metadata is scalar-read from GM; the bulk KV
data goes straight GM->L1 (never UB).
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.backend import BackendType, is_backend_configured, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _build_program(
    *,
    space: pl.MemorySpace = pl.MemorySpace.Mat,
    is_trans: bool = False,
    rows: int = 16,
    max_indices: int = 16,
    src_dtype: DataType = pl.FP16,
):
    out_shape = [128, max_indices] if is_trans else [max_indices, 128]

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[256, 128], src_dtype],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[out_shape, src_dtype]:
            out = pl.paged_gather(
                src,
                idx,
                bt,
                block_size=128,
                size=128,
                max_indices=max_indices,
                space=space,
                is_trans=is_trans,
            )
            return out

        @pl.function
        def main(
            self,
            src: pl.Tensor[[256, 128], src_dtype],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[out_shape, src_dtype]:
            r = self.kernel(src, idx, bt)
            return r

    return Program


def _print_after_convert(program) -> str:
    after = passes.convert_tensor_to_tile_ops()(program)
    return ir.python_print(after)


def test_paged_gather_lowers_to_scalar_per_row_l1_loop():
    """space=Mat lowers to a ForStmt of scalar index math + GM->L1 tile.load + assemble."""
    text = _print_after_convert(_build_program())

    # Static L1 accumulator.
    assert "pl.tile.create([16, 128]" in text
    assert "target_memory=pl.Mem.Mat" in text
    # Per-row loop over the runtime index count.
    assert "for pg_i" in text
    # Scalar GM reads for the index and the page table (pto.load_scalar at codegen).
    assert "pl.tensor.read(idx, [pg_i])" in text
    assert "pl.tensor.read(bt, [pg_blk])" in text
    # Paged address math, all scalar.
    assert "// 128" in text
    assert "% 128" in text
    # Bulk KV goes GM->L1 via a per-row tile.gather_row (pto.subview + GM->Mat
    # pto.tload, no MAT->MAT pto.tmov): the row is written straight into the
    # accumulator sub-region [pg_i, 0].
    assert (
        "pl.tile.gather_row(paged_gather_iter, src, [pg_i, 0], [pg_phys, 0], [1, 128], transpose=False)"
        in text
    )
    # No tile.assemble (would lower to an unsupported MAT->MAT tmov) and the whole
    # src is NOT preloaded into a Vec tile.
    assert "pl.tile.assemble(" not in text
    assert "target_memory=pl.Mem.Vec" not in text


def test_paged_gather_transpose_swaps_output_and_load():
    """is_trans=True swaps the output dims and loads each row transposed into L1."""
    text = _print_after_convert(_build_program(is_trans=True))
    # Accumulator is [size, max_indices] when transposed.
    assert "pl.tile.create([128, 16]" in text
    assert "transpose=True" in text
    # Row written as a column at destination offset [0, i].
    assert (
        "pl.tile.gather_row(paged_gather_iter, src, [0, pg_i], [pg_phys, 0], [1, 128], transpose=True)"
        in text
    )


def test_paged_gather_space_vec():
    """space=Vec targets UB instead of L1 while keeping the same scalar per-row loop."""
    text = _print_after_convert(_build_program(space=pl.MemorySpace.Vec))
    assert "target_memory=pl.Mem.Vec" in text
    assert "for pg_i" in text


def test_paged_gather_dynamic_row_count():
    """A runtime (dynamic) row count drives the loop bound; the L1 tile stays static."""
    rows = pl.dynamic("ROWS")

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[256, 128], pl.FP16],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            out = pl.paged_gather(src, idx, bt, block_size=128, size=128, max_indices=64)
            return out

        @pl.function
        def main(
            self,
            src: pl.Tensor[[256, 128], pl.FP16],
            idx: pl.Tensor[[rows], pl.INT32],
            bt: pl.Tensor[[8], pl.INT32],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            r = self.kernel(src, idx, bt)
            return r

    text = _print_after_convert(Program)
    # Static buffer (max_indices), dynamic loop bound (ROWS).
    assert "pl.tile.create([64, 128]" in text
    assert "ROWS" in text


@pytest.mark.parametrize("is_trans", [False, True])
def test_paged_gather_survives_full_pipeline(is_trans):
    """The lowered loop survives the full Default pipeline through codegen lowering."""
    # A backend may already be configured by an earlier test in the session;
    # only set it when unconfigured so real set_backend_type failures still surface.
    if not is_backend_configured():
        set_backend_type(BackendType.Ascend910B)
    program = _build_program(is_trans=is_trans)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    result = pm.run_passes(program)
    assert result is not None


def test_paged_gather_rejects_non_2d_src():
    """src must be 2D."""
    with pytest.raises(Exception, match="2D src"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[256], pl.FP16],
                idx: pl.Tensor[[16], pl.INT32],
                bt: pl.Tensor[[8], pl.INT32],
            ) -> pl.Tensor[[16, 128], pl.FP16]:
                return pl.paged_gather(src, idx, bt, block_size=128, size=128, max_indices=16)


def test_paged_gather_rejects_non_int32_indices():
    """indices must be INT32."""
    with pytest.raises(Exception, match="indices dtype"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[256, 128], pl.FP16],
                idx: pl.Tensor[[16], pl.FP16],
                bt: pl.Tensor[[8], pl.INT32],
            ) -> pl.Tensor[[16, 128], pl.FP16]:
                return pl.paged_gather(src, idx, bt, block_size=128, size=128, max_indices=16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
