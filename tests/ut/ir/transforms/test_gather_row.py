# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tensor.create_l1 + tensor.gather_row (kernel-driven paged gather into L1).

These two tensor-level ops are the flexible counterpart to tensor.paged_gather: the
kernel itself computes the physical source row per slot (block-table lookups,
multi-source selection, invalid clamping) and fills an on-chip (L1/Mat) accumulator
row by row. They are deduced as TensorType so the gathered result composes with
tensor-level matmul / softmax, and lower (in ConvertTensorToTileOps) to:

    tensor.create_l1  -> tile.create(target_memory=Mat)   (transpose -> ZN Mat layout)
    tensor.gather_row -> tile.gather_row                   (per-row pto.subview + GM->Mat tload)

``transpose=True`` builds a matmul ``b_trans`` B-operand: create_l1 allocates the
transposed Mat (ZN) fractal and gather_row places each GM row [r, c] as an L1
column [c, r].
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType, is_backend_configured, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _build_program(
    *,
    transpose: bool = False,
    rows: int = 16,
    head_dim: int = 128,
    nsrc: int = 256,
):
    """A kernel that fills an L1 accumulator row by row via create_l1 + gather_row.

    The caller computes each physical source row itself (here a trivial ``r``);
    the gathered tile is returned so the lowering is observable.
    """
    acc_shape = [head_dim, rows] if transpose else [rows, head_dim]

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, src: pl.Tensor[[nsrc, head_dim], pl.BF16]) -> pl.Tensor[acc_shape, pl.BF16]:
            kv = pl.create_l1(acc_shape, pl.BF16, transpose=transpose)
            for r in pl.range(rows):
                if transpose:
                    # GM row [r, :] lands as the L1 column [:, r].
                    kv = pl.gather_row(kv, src, [0, r], [r, 0], [1, head_dim], transpose=True)
                else:
                    kv = pl.gather_row(kv, src, [r, 0], [r, 0], [1, head_dim])
            return kv

        @pl.function
        def main(self, src: pl.Tensor[[nsrc, head_dim], pl.BF16]) -> pl.Tensor[acc_shape, pl.BF16]:
            r = self.kernel(src)
            return r

    return Program


def _build_straightline(*, transpose: bool):
    """Already-SSA straight-line kernel: two literal-offset gathers into an L1 tile.

    Mirrors test_paged_gather's convert-only setup so ConvertTensorToTileOps runs
    standalone (no ConvertToSSA needed) and the literal offsets survive printing,
    making the row-vs-column write distinction assertable. The caller uses the
    ``r = self.kernel(...); return r`` form so the convert pass injects the
    DPS ``Out`` argument at the call site (a bare ``return self.kernel(...)`` is
    not rewritten, leaving an inconsistent call the roundtrip check would reject).
    """
    acc_shape = [128, 16] if transpose else [16, 128]
    # Branch in Python (not in the traced kernel source — an in-body `if` would
    # emit a both-branch IfStmt that violates SSA before ConvertToSSA runs). A
    # transposing gather writes the GM row as the L1 column [0, slot]; a plain
    # gather writes it as the L1 row [slot, 0].
    dst0 = [0, 0]
    dst1 = [0, 1] if transpose else [1, 0]

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, src: pl.Tensor[[256, 128], pl.BF16]) -> pl.Tensor[acc_shape, pl.BF16]:
            kv0 = pl.create_l1(acc_shape, pl.BF16, transpose=transpose)
            kv1 = pl.gather_row(kv0, src, dst0, [0, 0], [1, 128], transpose=transpose)
            kv2 = pl.gather_row(kv1, src, dst1, [1, 0], [1, 128], transpose=transpose)
            return kv2

        @pl.function
        def main(self, src: pl.Tensor[[256, 128], pl.BF16]) -> pl.Tensor[acc_shape, pl.BF16]:
            r = self.kernel(src)
            return r

    return Program


def _print_after_convert(program) -> str:
    after = passes.convert_tensor_to_tile_ops()(program)
    # format=False keeps each statement on one line, so substring assertions are
    # not split by the formatter's line-wrapping (the long ZN TileView annotation
    # would otherwise wrap the transpose-case create/gather_row calls).
    return ir.python_print(after, format=False)


def test_create_l1_lowers_to_mat_tile():
    """tensor.create_l1 lowers to a static L1 (Mat) tile.create."""
    text = _print_after_convert(_build_straightline(transpose=False))
    assert "pl.tile.create([16, 128]" in text
    assert "target_memory=pl.Mem.Mat" in text
    # Default (non-transpose) allocation requests no transposed layout.
    assert "transpose=False" in text


def test_gather_row_lowers_to_tile_gather_row():
    """tensor.gather_row lowers to the per-row tile.gather_row, no Vec round-trip."""
    text = _print_after_convert(_build_straightline(transpose=False))
    # GM row written straight into the accumulator row slot [0, 0] / [1, 0].
    assert "pl.tile.gather_row(" in text
    assert "[0, 0], [0, 0], [1, 128], transpose=False)" in text
    assert "[1, 0], [1, 0], [1, 128], transpose=False)" in text
    # No tile.assemble (an unsupported MAT->MAT tmov) and src is not preloaded into Vec.
    assert "pl.tile.assemble(" not in text
    assert "target_memory=pl.Mem.Vec" not in text


def test_create_l1_transpose_allocates_zn_layout():
    """transpose=True allocates the transposed Mat (ZN) fractal: blayout row / slayout col."""
    text = _print_after_convert(_build_straightline(transpose=True))
    # Accumulator shape is the transposed [head_dim, rows].
    assert "pl.tile.create([128, 16]" in text
    assert "transpose=True" in text
    assert "blayout=pl.TileLayout.row_major" in text
    assert "slayout=pl.TileLayout.col_major" in text


def test_gather_row_transpose_writes_column():
    """transpose=True forwards to tile.gather_row and writes the GM row as a column."""
    text = _print_after_convert(_build_straightline(transpose=True))
    assert "pl.tile.gather_row(" in text
    # Destination offset [0, slot]: the GM row lands as the L1 column at that slot.
    assert "[0, 0], [0, 0], [1, 128], transpose=True)" in text
    assert "[0, 1], [1, 0], [1, 128], transpose=True)" in text


def test_gather_row_rejects_dtype_mismatch():
    """acc and src must share dtype (matmul operand integrity)."""
    with pytest.raises(Exception, match="share dtype"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[256, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.BF16]:
                kv = pl.create_l1([16, 128], pl.BF16)
                kv = pl.gather_row(kv, src, [0, 0], [0, 0], [1, 128])
                return kv


def test_create_l1_rejects_non_positive_shape():
    """create_l1 shape dims must be positive compile-time ConstInt."""
    with pytest.raises(Exception, match="positive compile-time ConstInt"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[256, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                kv = pl.create_l1([16, 0], pl.BF16)
                kv = pl.gather_row(kv, src, [0, 0], [0, 0], [1, 128])
                return kv


@pytest.mark.parametrize("transpose", [False, True])
def test_gather_row_survives_full_pipeline(transpose):
    """The create_l1 + per-row gather_row loop survives the full Default pipeline."""
    # A backend may already be configured by an earlier test in the session;
    # only set it when unconfigured so real set_backend_type failures still surface.
    if not is_backend_configured():
        set_backend_type(BackendType.Ascend910B)
    program = _build_program(transpose=transpose)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    result = pm.run_passes(program)
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
