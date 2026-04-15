# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile row/col broadcast operations using the PyPTO frontend.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType

MN_CASES: list[tuple[int, int]] = [(8, 16), (16, 16), (16, 8)]


class TestTileRowExpand(PTOTestCase):
    """Test case for tile.row_expand."""

    __test__ = False

    def __init__(self, m: int = 16, n: int = 16, *, backend_type: BackendType | None = None, config=None):
        super().__init__(config, backend_type=backend_type)
        self.M = m
        self.N = n

    def get_name(self) -> str:
        return f"tile_row_expand_{self.M}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("row_vec", [self.M, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, N = self.M, self.N

        @pl.program
        class TileRowExpandProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def row_expand_kernel(
                self,
                row_vec: pl.Tensor[[M, 1], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                row_tile: pl.Tile[[M, 1], pl.FP32] = pl.load(row_vec, [0, 0], [M, 1])
                target_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.create([M, N], dtype=pl.FP32)
                expanded: pl.Tile[[M, N], pl.FP32] = pl.tile.row_expand(target_tile, row_tile)
                out: pl.Tensor[[M, N], pl.FP32] = pl.store(expanded, [0, 0], y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                row_vec: pl.Tensor[[M, 1], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                y = self.row_expand_kernel(row_vec, y)
                return y

        return TileRowExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["row_vec"].repeat(1, self.N)


class TestTileColExpand(PTOTestCase):
    """Test case for tile.col_expand."""

    __test__ = False

    def __init__(self, m: int = 16, n: int = 16, *, backend_type: BackendType | None = None, config=None):
        super().__init__(config, backend_type=backend_type)
        self.M = m
        self.N = n

    def get_name(self) -> str:
        return f"tile_col_expand_{self.M}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("col_vec", [1, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, N = self.M, self.N

        @pl.program
        class TileColExpandProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def col_expand_kernel(
                self,
                col_vec: pl.Tensor[[1, N], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                col_tile: pl.Tile[[1, N], pl.FP32] = pl.load(col_vec, [0, 0], [1, N])
                target_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.create([M, N], dtype=pl.FP32)
                expanded: pl.Tile[[M, N], pl.FP32] = pl.tile.col_expand(target_tile, col_tile)
                out: pl.Tensor[[M, N], pl.FP32] = pl.store(expanded, [0, 0], y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                col_vec: pl.Tensor[[1, N], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                y = self.col_expand_kernel(col_vec, y)
                return y

        return TileColExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["col_vec"].repeat(self.M, 1)


class TestBroadcastOperations:
    """Test suite for tile broadcast operations."""

    @pytest.mark.parametrize("m, n", MN_CASES)
    @pytest.mark.parametrize("backend", PLATFORMS)
    def test_tile_row_expand(self, test_runner, backend, m, n):
        """Test tile.row_expand across platforms."""
        result = test_runner.run(TestTileRowExpand(m=m, n=n, backend_type=backend))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("m, n", MN_CASES)
    @pytest.mark.parametrize("backend", PLATFORMS)
    def test_tile_col_expand(self, test_runner, backend, m, n):
        """Test tile.col_expand across platforms."""
        result = test_runner.run(TestTileColExpand(m=m, n=n, backend_type=backend))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
