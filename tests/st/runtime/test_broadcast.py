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
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


class TestTileRowExpand(PTOTestCase):
    """Base test case for tile.row_expand."""

    __test__ = False

    def __init__(self, m: int = 16, n: int = 16, config=None):
        super().__init__(config)
        self.M = m
        self.N = n

    def get_name(self) -> str:
        return f"tile_row_expand_{self.M}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [self.M, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, N = self.M, self.N

        @pl.program
        class TileRowExpandProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def row_expand_kernel(
                self,
                x: pl.Tensor[[M, N], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile: pl.Tile[[M, N], pl.FP32] = pl.load(x, [0, 0], [M, N])
                expanded: pl.Tile[[M, N], pl.FP32] = pl.tile.row_expand(tile)
                out: pl.Tensor[[M, N], pl.FP32] = pl.store(expanded, [0, 0], y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[M, N], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                y = self.row_expand_kernel(x, y)
                return y

        return TileRowExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["x"][:, :1].repeat(1, self.N)


class TestTileRowExpandPTO(TestTileRowExpand):
    """Test tile.row_expand with PTO backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"tile_row_expand_pto_{self.M}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class TestTileRowExpandA5(TestTileRowExpand):
    """Test tile.row_expand with A5 (Ascend 950) backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"tile_row_expand_a5_{self.M}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950


class TestTileColExpand(PTOTestCase):
    """Base test case for tile.col_expand."""

    __test__ = False

    def __init__(self, m: int = 16, n: int = 16, config=None):
        super().__init__(config)
        self.M = m
        self.N = n

    def get_name(self) -> str:
        return f"tile_col_expand_{self.M}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("target", [self.M, self.N], DataType.FP32, init_value=torch.randn),
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
                target: pl.Tensor[[M, N], pl.FP32],
                col_vec: pl.Tensor[[1, N], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                target_tile: pl.Tile[[M, N], pl.FP32] = pl.load(target, [0, 0], [M, N])
                col_tile: pl.Tile[[1, N], pl.FP32] = pl.load(col_vec, [0, 0], [1, N])
                expanded: pl.Tile[[M, N], pl.FP32] = pl.tile.col_expand(target_tile, col_tile)
                out: pl.Tensor[[M, N], pl.FP32] = pl.store(expanded, [0, 0], y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                target: pl.Tensor[[M, N], pl.FP32],
                col_vec: pl.Tensor[[1, N], pl.FP32],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                y = self.col_expand_kernel(target, col_vec, y)
                return y

        return TileColExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["col_vec"].repeat(self.M, 1)


class TestTileColExpandPTO(TestTileColExpand):
    """Test tile.col_expand with PTO backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"tile_col_expand_pto_{self.M}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class TestTileColExpandA5(TestTileColExpand):
    """Test tile.col_expand with A5 (Ascend 950) backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"tile_col_expand_a5_{self.M}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950


class TestBroadcastOperations:
    """Test suite for tile broadcast operations."""

    def test_tile_row_expand_pto(self, test_runner):
        """Test tile.row_expand with PTO backend."""
        test_case = TestTileRowExpandPTO()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (row_expand): {result.error}"

    def test_tile_col_expand_pto(self, test_runner):
        """Test tile.col_expand with PTO backend."""
        test_case = TestTileColExpandPTO()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (col_expand): {result.error}"

    @pytest.mark.a5
    def test_tile_row_expand_a5(self, test_runner):
        """Test tile.row_expand with A5 (Ascend 950) backend."""
        test_case = TestTileRowExpandA5()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (A5 row_expand): {result.error}"

    @pytest.mark.a5
    def test_tile_col_expand_a5(self, test_runner):
        """Test tile.col_expand with A5 (Ascend 950) backend."""
        test_case = TestTileColExpandA5()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (A5 col_expand): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
