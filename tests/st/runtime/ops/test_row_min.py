# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for ``pto.trowmin``.

The direct tile path exercises the mandatory ``(src, tmp)`` signature and the
tensor path verifies compiler-owned scratch allocation.  PTOAS supports
``i16/i32/f16/f32`` on both architecture families.  The matrix covers every
dtype, exact and oversized scratch buffers, full tiles, and independent row,
column, and combined ``valid_shape`` tails.

PTOAS treats ``tmp`` as A2/A3 workspace and an A5 ABI placeholder.  PyPTO keeps
the conservative hardware-safe rule that a user-provided scratch tile has the
same dtype/rank and is not smaller than ``src``; both legal size forms are
covered here.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec

_PL_DT = {
    DataType.FP32: pl.FP32,
    DataType.FP16: pl.FP16,
    DataType.INT32: pl.INT32,
    DataType.INT16: pl.INT16,
}


def _input_data(
    m: int,
    n: int,
    dtype: DataType,
    valid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Exercise first/middle/last and tied minima, with lower invalid poison."""
    valid_rows, valid_cols = valid_shape or (m, n)
    values = torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(37) + 10

    for row in range(valid_rows):
        min_col = (0, valid_cols // 2, valid_cols - 1)[row % 3]
        values[row, min_col] = -50
        if row % 4 == 3 and valid_cols > 1:
            values[row, (min_col + 1) % valid_cols] = -50

    # If the implementation accidentally reduces the physical width instead of
    # valid_cols, this value wins and the golden comparison fails.
    if valid_cols < n:
        values[:valid_rows, valid_cols:] = -1000
    if valid_rows < m:
        values[valid_rows:, :] = -1000

    if dtype in (DataType.FP16, DataType.FP32):
        values = values / 3
    return values.to(dtype.torch_dtype).contiguous()


class TileRowMinCase(PTOTestCase):
    """Direct ``pl.tile.row_min(src, tmp)`` path."""

    __test__ = False

    def __init__(
        self,
        *,
        m: int = 32,
        n: int = 64,
        dtype: DataType = DataType.FP32,
        valid_shape: tuple[int, int] | None = None,
        oversized_tmp: bool = False,
        platform: str,
    ):
        super().__init__(platform=platform)
        self.m = m
        self.n = n
        self.dtype = dtype
        self.valid_shape = valid_shape
        self.oversized_tmp = oversized_tmp

    def get_name(self) -> str:
        valid = self.valid_shape or (self.m, self.n)
        tmp_kind = "oversized_tmp" if self.oversized_tmp else "exact_tmp"
        return f"tile_row_min_{self.dtype.value}_{self.m}x{self.n}_v{valid[0]}x{valid[1]}_{tmp_kind}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _input_data(self.m, self.n, self.dtype, self.valid_shape),
            ),
            TensorSpec(
                "out",
                [self.m, 1],
                self.dtype,
                init_value=torch.zeros,
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        m, n = self.m, self.n
        dtype = _PL_DT[self.dtype]
        valid_shape = list(self.valid_shape or (m, n))
        tmp_m = m + 8 if self.oversized_tmp else m
        tmp_n = max(n + 32, 128) if self.oversized_tmp else n

        @pl.program
        class RowMinProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, 1], dtype]],
            ) -> pl.Tensor[[m, 1], dtype]:
                src_tile: pl.Tile[[m, n], dtype] = pl.load(src, [0, 0], [m, n], valid_shape=valid_shape)
                tmp: pl.Tile[[tmp_m, tmp_n], dtype] = pl.tile.create(
                    [tmp_m, tmp_n], dtype=dtype, target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[m, 1], dtype] = pl.tile.row_min(src_tile, tmp)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, 1], dtype]],
            ) -> pl.Tensor[[m, 1], dtype]:
                return self.kernel(src, out)

        return RowMinProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows, valid_cols = self.valid_shape or (self.m, self.n)
        expected = torch.zeros_like(tensors["out"])
        expected[:valid_rows, 0] = torch.amin(tensors["src"][:valid_rows, :valid_cols], dim=1)
        tensors["out"][:] = expected


class TensorRowMinCase(PTOTestCase):
    """Tensor frontend lowered through compiler-created ``tmp`` storage."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType,
        platform: str,
        m: int = 32,
        n: int = 64,
    ):
        super().__init__(platform=platform)
        self.m = m
        self.n = n
        self.dtype = dtype

    def get_name(self) -> str:
        return f"tensor_row_min_{self.dtype.value}_{self.m}x{self.n}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _input_data(self.m, self.n, self.dtype),
            ),
            TensorSpec("out", [self.m, 1], self.dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self.m, self.n
        dtype = _PL_DT[self.dtype]

        @pl.program
        class TensorRowMinProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.Out[pl.Tensor[[m, 1], dtype]],
            ) -> pl.Tensor[[m, 1], dtype]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[m, 1], dtype] = pl.tensor.row_min(src)
                    out = pl.assemble(out, result, [0, 0])
                return out

        return TensorRowMinProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.amin(tensors["src"], dim=1, keepdim=True)


_DTYPE_CASES = [
    pytest.param(DataType.FP32, 32, 64, id="f32-32x64"),
    pytest.param(DataType.FP16, 16, 128, id="f16-16x128"),
    pytest.param(DataType.INT32, 32, 128, id="i32-32x128"),
    pytest.param(DataType.INT16, 16, 192, id="i16-16x192"),
]

_VALID_SHAPE_CASES = [
    pytest.param((20, 64), id="row-tail"),
    pytest.param((32, 50), id="column-tail-unaligned"),
    pytest.param((20, 50), id="row-column-tail"),
]


class TestTileRowMin:
    """Direct PTO op coverage across both architectures and all legal dtypes."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("dtype,m,n", _DTYPE_CASES)
    def test_dtypes_and_boundary_shapes(self, test_runner, platform, dtype, m, n):
        result = test_runner.run(TileRowMinCase(m=m, n=n, dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("valid_shape", _VALID_SHAPE_CASES)
    def test_valid_shape_tails(self, test_runner, platform, valid_shape):
        result = test_runner.run(TileRowMinCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_minimum_aligned_physical_tile_with_single_valid_element(self, test_runner, platform):
        result = test_runner.run(
            TileRowMinCase(
                m=8,
                n=16,
                valid_shape=(1, 1),
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_oversized_tmp_with_combined_tail(self, test_runner, platform):
        result = test_runner.run(
            TileRowMinCase(
                valid_shape=(20, 50),
                oversized_tmp=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


class TestTensorRowMin:
    """Tensor frontend/conversion coverage, including integer reduction."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(DataType.FP32, id="f32"),
            pytest.param(DataType.INT16, id="i16"),
        ],
    )
    def test_tensor_lowering(self, test_runner, platform, dtype):
        result = test_runner.run(TensorRowMinCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
