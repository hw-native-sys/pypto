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

# Standalone N-D transpose deinterleave constants (#1651 regression)
DEINT_T = 128
DEINT_RD = 64
DEINT_RH = DEINT_RD // 2

# Minimal 2D transpose constants (#1651 alignment-hypothesis probe)
# Input [16, 2] FP32 transposes to [2, 16]; the [16, 2] source row byte size
# is 2 * sizeof(FP32) = 8 bytes, which is not 32-byte aligned — used to check
# whether the ptoas alloc_tile 32-byte row-alignment error is triggered by the
# narrow column dimension alone.
SIMPLE_R = 16
SIMPLE_C = 2

# Standalone 3D transpose constants (#1651): [4, 24, 8] swap axes (1, 2) -> [4, 8, 24].
# Axes 1 and 2 are deliberately different (24 vs 8) so the swap actually changes
# the shape — a non-degenerate transpose, not the equal-dim case where input and
# output shapes coincide. Both inner dims (input 8, output 24) are multiples of 8,
# so each tile row (cols * sizeof(FP32)) is 32-byte aligned, yet neither is
# 16-aligned: ptoas infers an `nz` layout for a 3D ND tensor whose innermost dim
# is 16-aligned (e.g. [4, 8, 16]), which conflicts with the `nd` layout the codegen
# declares for `pto.make_tensor_view`. Keeping both inner dims non-16-aligned keeps
# the inferred layout `nd`, isolating the FlattenTileNdTo2D 3D-transpose lowering
# under test from that separate 3D-output codegen gap.
ND3_B = 4  # leading batch dim (untouched)
ND3_R = 24  # axis 1 (multiple of 8, not 16-aligned)
ND3_C = 8  # axis 2 (multiple of 8, not 16-aligned)


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


class NdTransposeDeinterleaveCase(PTOTestCase):
    """#1651: standalone N-D ``pl.transpose`` (last-two-axes) on a 3D tile.

    The kernel deinterleaves an ``[T, RD]`` row into its even/odd lanes via
    ``reshape([T, RH, 2]) -> transpose(1, 2) -> reshape([T, RD])``. The middle
    ``transpose`` is a standalone >2D ``tile.transpose`` (no ``batch_matmul``
    consumer), which previously failed to compile in ``FlattenTileNdTo2D``
    because the pass left the transpose input at rank 3 while its flattened
    scratch ``tmp`` was rank 2. The pass now lowers it to per-batch 2D
    transposes; the numeric result must equal the even/odd column split.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"nd_transpose_deinterleave_{DEINT_T}x{DEINT_RD}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [DEINT_T, DEINT_RD], DataType.FP32, init_value=torch.randn),
            TensorSpec("even_out", [DEINT_T, DEINT_RH], DataType.FP32, is_output=True),
            TensorSpec("odd_out", [DEINT_T, DEINT_RH], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class NdTransposeDeinterleave:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[DEINT_T, DEINT_RD], pl.FP32],
                even_out: pl.Out[pl.Tensor[[DEINT_T, DEINT_RH], pl.FP32]],
                odd_out: pl.Out[pl.Tensor[[DEINT_T, DEINT_RH], pl.FP32]],
            ) -> tuple[pl.Tensor[[DEINT_T, DEINT_RH], pl.FP32], pl.Tensor[[DEINT_T, DEINT_RH], pl.FP32]]:
                x_tile: pl.Tile[[DEINT_T, DEINT_RD], pl.FP32] = pl.load(x, [0, 0], [DEINT_T, DEINT_RD])
                x3d: pl.Tile[[DEINT_T, DEINT_RH, 2], pl.FP32] = pl.reshape(x_tile, [DEINT_T, DEINT_RH, 2])
                x3d_t: pl.Tile[[DEINT_T, 2, DEINT_RH], pl.FP32] = pl.transpose(x3d, axis1=1, axis2=2)
                x_deint: pl.Tile[[DEINT_T, DEINT_RD], pl.FP32] = pl.reshape(x3d_t, [DEINT_T, DEINT_RD])
                even: pl.Tile[[DEINT_T, DEINT_RH], pl.FP32] = pl.tile.extract(
                    x_deint, 0, 0, [DEINT_T, DEINT_RH], target_memory=pl.MemorySpace.Vec
                )
                odd: pl.Tile[[DEINT_T, DEINT_RH], pl.FP32] = pl.tile.extract(
                    x_deint, 0, DEINT_RH, [DEINT_T, DEINT_RH], target_memory=pl.MemorySpace.Vec
                )
                even_out = pl.store(even, [0, 0], even_out)
                odd_out = pl.store(odd, [0, 0], odd_out)
                return even_out, odd_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[DEINT_T, DEINT_RD], pl.FP32],
                even_out: pl.Out[pl.Tensor[[DEINT_T, DEINT_RH], pl.FP32]],
                odd_out: pl.Out[pl.Tensor[[DEINT_T, DEINT_RH], pl.FP32]],
            ) -> tuple[pl.Tensor[[DEINT_T, DEINT_RH], pl.FP32], pl.Tensor[[DEINT_T, DEINT_RH], pl.FP32]]:
                even_out, odd_out = self.kernel(x, even_out, odd_out)
                return even_out, odd_out

        return NdTransposeDeinterleave

    def compute_expected(self, tensors, params=None):
        x = tensors["x"]
        tensors["even_out"][:] = x[:, 0::2]
        tensors["odd_out"][:] = x[:, 1::2]


class SimpleTransposeCase(PTOTestCase):
    """#1651 probe: minimal 2D ``pl.transpose`` on a narrow ``[16, 2]`` tile.

    Loads ``[16, 2]`` into a Vec tile, transposes to ``[2, 16]`` and stores it.
    The source tile's row byte size is ``2 * sizeof(FP32) = 8`` bytes, which is
    not 32-byte aligned. This isolates whether the ptoas ``alloc_tile`` 32-byte
    row-alignment error reproduces with a plain 2D transpose (no reshape / N-D
    path), confirming the narrow column dimension is the trigger.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"simple_transpose_{SIMPLE_R}x{SIMPLE_C}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [SIMPLE_R, SIMPLE_C], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [SIMPLE_C, SIMPLE_R], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class SimpleTranspose:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[SIMPLE_R, SIMPLE_C], pl.FP32],
                out: pl.Out[pl.Tensor[[SIMPLE_C, SIMPLE_R], pl.FP32]],
            ) -> pl.Tensor[[SIMPLE_C, SIMPLE_R], pl.FP32]:
                x_tile: pl.Tile[[SIMPLE_R, SIMPLE_C], pl.FP32] = pl.load(x, [0, 0], [SIMPLE_R, SIMPLE_C])
                x_t: pl.Tile[[SIMPLE_C, SIMPLE_R], pl.FP32] = pl.transpose(x_tile, axis1=0, axis2=1)
                return pl.store(x_t, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[SIMPLE_R, SIMPLE_C], pl.FP32],
                out: pl.Out[pl.Tensor[[SIMPLE_C, SIMPLE_R], pl.FP32]],
            ) -> pl.Tensor[[SIMPLE_C, SIMPLE_R], pl.FP32]:
                out = self.kernel(x, out)
                return out

        return SimpleTranspose

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["x"].t()


class Nd3DTransposeCase(PTOTestCase):
    """#1651: standalone 3D ``pl.transpose`` (direct last-two-axes swap).

    Loads ``[ND3_B, ND3_R, ND3_C]`` into a tile, transposes axes ``(1, 2)`` to
    ``[ND3_B, ND3_C, ND3_R]`` and stores it. Unlike
    ``NdTransposeDeinterleaveCase`` there is no surrounding reshape — this is a
    direct >2D last-two-axes swap on a 3D tile with no ``batch_matmul``
    consumer, exercising the ``LowerNdTranspose`` path in ``FlattenTileNdTo2D``
    that lowers a standalone N-D transpose to per-batch 2D transposes.

    Axes 1 and 2 differ (``ND3_R = 24`` vs ``ND3_C = 8``) so the swap actually
    changes the shape. Both inner dims are multiples of 8 (each tile row is
    32-byte aligned) yet neither is 16-aligned — see the ``ND3_*`` constants — so
    the 3D ND tensor-view layout stays ``nd``. A 16-aligned inner dim would make
    ptoas infer an ``nz`` layout that conflicts with the ``nd`` the codegen
    declares, a separate 3D-output gap this case avoids.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"nd3d_transpose_{ND3_B}x{ND3_R}x{ND3_C}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [ND3_B, ND3_R, ND3_C], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [ND3_B, ND3_C, ND3_R], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Nd3DTranspose:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[ND3_B, ND3_R, ND3_C], pl.FP32],
                out: pl.Out[pl.Tensor[[ND3_B, ND3_C, ND3_R], pl.FP32]],
            ) -> pl.Tensor[[ND3_B, ND3_C, ND3_R], pl.FP32]:
                x_tile: pl.Tile[[ND3_B, ND3_R, ND3_C], pl.FP32] = pl.load(x, [0, 0, 0], [ND3_B, ND3_R, ND3_C])
                x_t: pl.Tile[[ND3_B, ND3_C, ND3_R], pl.FP32] = pl.transpose(x_tile, axis1=1, axis2=2)
                return pl.store(x_t, [0, 0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[ND3_B, ND3_R, ND3_C], pl.FP32],
                out: pl.Out[pl.Tensor[[ND3_B, ND3_C, ND3_R], pl.FP32]],
            ) -> pl.Tensor[[ND3_B, ND3_C, ND3_R], pl.FP32]:
                out = self.kernel(x, out)
                return out

        return Nd3DTranspose

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = tensors["x"].transpose(1, 2)


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

    @pytest.mark.skip(
        reason="Blocked by a separate transpose alloc_tile gap: ptoas requires the "
        "transpose source/scratch tile row (cols * sizeof(dtype)) to be 32-byte "
        "aligned. The deinterleave kernel transposes a [.., 2] page (8-byte row) "
        "which ptoas rejects. Tracked separately from #1651; to be filed as its "
        "own issue."
    )
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_nd_transpose_deinterleave(self, test_runner, platform):
        """Issue #1651: standalone N-D ``tile.transpose`` (last-two-axes swap).

        Parametrized on all platforms — the kernel uses only Vec ops, so a2a3 /
        a2a3sim passing confirms ``FlattenTileNdTo2D`` now lowers a standalone
        >2D transpose to per-batch 2D transposes with the correct numeric
        deinterleave (even/odd lane split).
        """
        result = test_runner.run(NdTransposeDeinterleaveCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(
        reason="Blocked by a separate transpose alloc_tile gap: the [16, 2] source "
        "has an 8-byte row (2 * sizeof(FP32)), which ptoas alloc_tile rejects for "
        "not being 32-byte aligned. This is the narrow-column alignment limitation, "
        "tracked separately from #1651; to be filed as its own issue."
    )
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_simple_transpose(self, test_runner, platform):
        """Issue #1651 probe: minimal 2D ``pl.transpose`` on a ``[16, 2]`` tile.

        Isolates the 32-byte row-alignment hypothesis — the ``[16, 2]`` source
        has an 8-byte row, so if ptoas ``alloc_tile`` still rejects it here (no
        reshape / N-D path involved), the narrow column dimension is confirmed
        as the trigger for the alignment error.
        """
        result = test_runner.run(SimpleTransposeCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_nd3d_transpose(self, test_runner, platform):
        """Issue #1651: standalone 3D ``pl.transpose`` (direct last-two-axes swap).

        Transposes a ``[4, 24, 8]`` tile over axes ``(1, 2)`` to ``[4, 8, 24]``
        with no surrounding reshape and no ``batch_matmul`` consumer — exercising
        the ``LowerNdTranspose`` path in ``FlattenTileNdTo2D`` that lowers a
        standalone >2D transpose to per-batch 2D transposes. Axes 1 and 2 differ
        (24 vs 8) so the swap changes the shape; both inner dims are multiples of
        8 (32-byte-aligned rows) but not 16-aligned, keeping the 3D output tensor
        view ``nd`` and avoiding the separate 3D-output ``nz`` layout gap.
        """
        result = test_runner.run(Nd3DTransposeCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
