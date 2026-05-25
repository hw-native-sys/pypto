# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end regressions for column-load lowering paths.

Two scenarios live in this file:

1. Issue #1209 follow-up — ``pl.transpose(x, 0, 1)`` + ``pl.slice(xt, ...)``
   must access column ``h`` of ``x`` (not the first contiguous chunk in
   memory). The runtime ``Tensor::transpose`` is a metadata-only swap, so the
   IR result must record swapped physical strides for codegen to emit a
   correctly addressed ``make_tensor_view``. Restricted to a5/a5sim — the
   path produces a ``GlobalTensor<DN>`` source that a2a3 rejects at the
   ``TLOAD`` legality check.

2. Issue #1398 workaround — a direct
   ``pl.load(scale, [0, 0], [ROWS, 1], target_memory=Vec)`` is rejected on
   a2a3 by ``TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ``.
   ``ColumnLoadRowExpandMulCase`` produces the same ``y = x * scale[:, 0:1]``
   result via a c0-strip load + on-chip transpose + ISA ``TEXTRACT`` while
   the direct-load lowering gap is being addressed. Runs on all platforms —
   passing on a2a3 / a2a3sim provides the positive signal for the workaround.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec

# Slice variant constants (#1209 follow-up regression)
T = 8
PAD = 16
N = 4

# Extract variant constants (#1398 a2a3 [ROWS, 1] column-load workaround)
ROWS = 16
COLS = 64
SCALE_COLS = 8
C0_FP32 = 8  # 32-byte BLOCK / sizeof(FP32) — c0 strip width for Vec ND tiles


class TransposeSliceAssembleCase(PTOTestCase):
    """#1209 follow-up: orchestration ``pl.transpose`` + ``pl.slice`` + ``pl.assemble``.

    ``pl.transpose(x, 0, 1)`` is a metadata-only stride swap on the GM
    tensor view; ``pl.slice(xt, [1, T], [h, 0])`` must then address column
    ``h`` of the original ``x`` (not the first contiguous chunk in memory).
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"transpose_slice_assemble_{T}x{PAD}_to_{T}x{N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "x",
                [T, PAD],
                DataType.FP32,
                init_value=lambda: torch.arange(T * PAD, dtype=torch.float32).reshape(T, PAD),
            ),
            TensorSpec("out", [T, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TransposeSliceRepro:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[T, PAD], pl.FP32],
                out: pl.Out[pl.Tensor[[T, N], pl.FP32]],
            ):
                xt = pl.transpose(x, 0, 1)
                for h in pl.range(N):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="slice_transposed_row"):
                        col = pl.reshape(pl.slice(xt, [1, T], [h, 0]), [T, 1])
                        out = pl.assemble(out, col, [0, h])
                return out

        return TransposeSliceRepro

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["x"][:, :N]


class ColumnLoadRowExpandMulCase(PTOTestCase):
    """#1398 workaround: column-load via c0-strip + on-chip transpose + extract.

    Pipeline (UB-only after the GM loads):

    1. ``pl.load(scale, [0, 0], [ROWS, C0_FP32])`` — ND→ND strip load.
       ``C0_FP32 == 8`` is the minimum 32-byte-aligned row width for FP32 Vec ND tiles.
    2. ``pl.transpose(strip, 0, 1)`` — on-chip ``[ROWS, C0] -> [C0, ROWS]``.
    3. ``pl.tile.extract(strip_t, 0, 0, [1, ROWS], target_memory=Vec)`` —
       ISA ``TEXTRACT`` row 0.
    4. ``pl.reshape(row, [ROWS, 1])`` — metadata-only.

    The resulting ``[ROWS, 1]`` Vec column is what #1398's direct
    ``pl.load(..., [ROWS, 1], target_memory=Vec)`` would have produced,
    without the failing ``TLOAD(VecTile, GlobalTensor)`` lowering.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"column_load_row_expand_mul_{ROWS}x{COLS}_scale{ROWS}x{SCALE_COLS}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [ROWS, COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("scale", [ROWS, SCALE_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [ROWS, COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class ColumnLoadRowExpandMul:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scale: pl.Tensor[[ROWS, SCALE_COLS], pl.FP32],
                y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                # Produces the [ROWS, 1] Vec column that the direct Vec column-load
                # would have produced (but doesn't on a2a3 today).
                strip: pl.Tile[[ROWS, C0_FP32], pl.FP32] = pl.load(scale, [0, 0], [ROWS, C0_FP32])
                strip_t: pl.Tile[[C0_FP32, ROWS], pl.FP32] = pl.transpose(strip, axis1=0, axis2=1)
                row: pl.Tile[[1, ROWS], pl.FP32] = pl.tile.extract(
                    strip_t, 0, 0, [1, ROWS], target_memory=pl.MemorySpace.Vec
                )
                col: pl.Tile[[ROWS, 1], pl.FP32] = pl.reshape(row, [ROWS, 1])
                # Broadcast-multiply x by the per-row scale (exactly as in #1398).
                x_tile: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(x, [0, 0], [ROWS, COLS])
                y_tile: pl.Tile[[ROWS, COLS], pl.FP32] = pl.row_expand_mul(x_tile, col)
                return pl.store(y_tile, [0, 0], y)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scale: pl.Tensor[[ROWS, SCALE_COLS], pl.FP32],
                y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                y = self.kernel(x, scale, y)
                return y

        return ColumnLoadRowExpandMul

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["x"] * tensors["scale"][:, 0:1]


class TestTransposeColumnOperations:
    """Column-load lowering regressions."""

    @pytest.mark.platforms("a5", "a5sim")
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_transpose_slice_assemble(self, test_runner, platform):
        """Issue #1209 follow-up: column-h selection via orch transpose + slice.

        a5-only — the slice path produces a ``GlobalTensor<DN>`` view; a2a3's
        kernel-C++ ``TLOAD`` only accepts ``ND2ND`` / ``DN2DN`` / ``NZ2NZ``.
        """
        result = test_runner.run(TransposeSliceAssembleCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_column_load_row_expand_mul(self, test_runner, platform):
        """Issue #1398 workaround: c0-strip column-load fed into row_expand_mul.

        Parametrized on all platforms — every GM→UB load stays ND→ND, so
        a2a3 / a2a3sim passing here is the positive signal that the
        workaround sidesteps #1398's lowering gap.
        """
        result = test_runner.run(ColumnLoadRowExpandMulCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
