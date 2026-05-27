# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile.cast (pto.tcvt) narrowing conversions.

Regression for #1549: pto.tcvt mis-orders elements when its source tile is
col_major. A [1, N] -> [N, 1] reshape yields a col_major [N, 1] tile; narrowing
that view i32 -> i16 used to reverse the elements (silent wrong output). The
tile.cast row_major layout spec lets ResolveBackendOpLayouts repair col_major
sources by reshaping [N, 1] -> [1, N] row_major around the cast.

  TileCastColMajorNarrowTestCase : narrow a col_major [N, 1] view i32 -> i16
                                   (the regression — must preserve element order).
  TileCastRowMajorNarrowTestCase : narrow a row_major [1, N] tile i32 -> i16
                                   (always-correct control / baseline).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

N = 16


def _arange_row() -> torch.Tensor:
    """Row-major [1, N] i32 with distinct, non-symmetric values [0 .. N-1].

    Distinct ascending values make element reordering observable: a reversed
    cast would produce [N-1 .. 0] instead of [0 .. N-1].
    """
    return torch.arange(0, N, dtype=torch.int32).reshape(1, N)


# ---------------------------------------------------------------------------
# Kernel programs
# ---------------------------------------------------------------------------


@pl.program
class TileCastColMajorNarrowProgram:
    """Narrow a col_major [N, 1] index-style view i32 -> i16.

    Reshaping [1, N] -> [N, 1] produces a col_major tile; the cast runs on that
    col_major source (the #1549 trigger). The result is reshaped back to a
    32-byte-aligned [1, N] row before the store so only the cast source is
    col_major.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT16]],
    ) -> pl.Tensor[[1, N], pl.INT16]:
        tile_a: pl.Tile[[1, N], pl.INT32] = pl.load(a, [0, 0], [1, N])
        col: pl.Tile[[N, 1], pl.INT32] = pl.tile.reshape(tile_a, [N, 1])
        narrowed: pl.Tile[[N, 1], pl.INT16] = pl.tile.cast(col, target_type=pl.INT16)
        row: pl.Tile[[1, N], pl.INT16] = pl.tile.reshape(narrowed, [1, N])
        out = pl.store(row, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT16]],
    ) -> pl.Tensor[[1, N], pl.INT16]:
        out = self.kernel(a, out)
        return out


@pl.program
class TileCastRowMajorNarrowProgram:
    """Narrow a row_major [1, N] tile i32 -> i16 (control, no col_major view)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT16]],
    ) -> pl.Tensor[[1, N], pl.INT16]:
        tile_a: pl.Tile[[1, N], pl.INT32] = pl.load(a, [0, 0], [1, N])
        narrowed: pl.Tile[[1, N], pl.INT16] = pl.tile.cast(tile_a, target_type=pl.INT16)
        out = pl.store(narrowed, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[1, N], pl.INT32],
        out: pl.Out[pl.Tensor[[1, N], pl.INT16]],
    ) -> pl.Tensor[[1, N], pl.INT16]:
        out = self.kernel(a, out)
        return out


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TileCastColMajorNarrowTestCase(PTOTestCase):
    """i32 -> i16 cast on a col_major [N, 1] view must preserve element order."""

    def get_name(self) -> str:
        return "tile_cast_col_major_narrow"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [1, N], DataType.INT32, init_value=_arange_row),
            TensorSpec("out", [1, N], DataType.INT16, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileCastColMajorNarrowProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None) -> None:
        # Reshape round-trip is a no-op on values; the cast only narrows the dtype,
        # so each element keeps its value and position: out[0, k] == a[0, k].
        tensors["out"][:] = tensors["a"].to(torch.int16)


class TileCastRowMajorNarrowTestCase(PTOTestCase):
    """i32 -> i16 cast on a row_major [1, N] tile (always-correct baseline)."""

    def get_name(self) -> str:
        return "tile_cast_row_major_narrow"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [1, N], DataType.INT32, init_value=_arange_row),
            TensorSpec("out", [1, N], DataType.INT16, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileCastRowMajorNarrowProgram

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def compute_expected(self, tensors, params=None) -> None:
        tensors["out"][:] = tensors["a"].to(torch.int16)


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------


class TestCast:
    """End-to-end tile.cast narrowing tests, covering col_major and row_major sources."""

    def test_tile_cast_col_major_narrow(self, test_runner):
        """Regression for #1549: i32 -> i16 on a col_major [N, 1] view keeps element order."""
        result = test_runner.run(TileCastColMajorNarrowTestCase())
        assert result.passed, f"col_major narrow cast failed: {result.error}"

    def test_tile_cast_row_major_narrow(self, test_runner):
        """Control: i32 -> i16 on a row_major [1, N] tile is correct."""
        result = test_runner.run(TileCastRowMajorNarrowTestCase())
        assert result.passed, f"row_major narrow cast failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
