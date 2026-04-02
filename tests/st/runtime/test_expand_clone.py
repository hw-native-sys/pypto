# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile.expand_clone using the PyPTO frontend.

Validates broadcast expansion along a single axis in InCore code, and ensures
correct codegen and execution through the PTO test harness.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


class TestExpandClone(PTOTestCase):
    """Base test case for tile.expand_clone."""

    __test__ = False  # Not a pytest test class

    def __init__(self, b: int = 4, n: int = 8, k: int = 16, config=None):
        super().__init__(config)
        self.B = b
        self.N = n
        self.K = k

    def get_name(self) -> str:
        return f"expand_clone_{self.B}x{self.N}x{self.K}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [self.B, 1, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.B, self.N, self.K], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, N, K = self.B, self.N, self.K

        @pl.program
        class ExpandCloneProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def expand_clone_kernel(
                self,
                x: pl.Tensor[[B, 1, K], pl.FP32],
                y: pl.Out[pl.Tensor[[B, N, K], pl.FP32]],
            ) -> pl.Tensor[[B, N, K], pl.FP32]:
                tile_src: pl.Tile[[B, 1, K], pl.FP32] = pl.load(x, offsets=[0, 0, 0], shapes=[B, 1, K])
                tile_dst: pl.Tile[[B, N, K], pl.FP32] = pl.tile.expand_clone(tile_src, [B, N, K])
                out: pl.Tensor[[B, N, K], pl.FP32] = pl.store(tile_dst, offsets=[0, 0, 0], output_tensor=y)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[B, 1, K], pl.FP32],
                y: pl.Out[pl.Tensor[[B, N, K], pl.FP32]],
            ) -> pl.Tensor[[B, N, K], pl.FP32]:
                y = self.expand_clone_kernel(x, y)
                return y

        return ExpandCloneProgram

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = tensors["x"].repeat(1, self.N, 1)


class TestExpandClonePTO(TestExpandClone):
    """Test expand_clone with PTO backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"expand_clone_pto_{self.B}x{self.N}x{self.K}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class TestExpandCloneA5(TestExpandClone):
    """Test expand_clone with A5 (Ascend 950) backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"expand_clone_a5_{self.B}x{self.N}x{self.K}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950


class TestExpandCloneOperations:
    """Test suite for expand_clone runtime execution."""

    def test_expand_clone_pto(self, test_runner):
        """Test expand_clone with PTO backend."""
        test_case = TestExpandClonePTO()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (PTO): {result.error}"

    @pytest.mark.a5
    def test_expand_clone_a5(self, test_runner):
        """Test expand_clone with A5 (Ascend 950) backend."""
        test_case = TestExpandCloneA5()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (A5): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
