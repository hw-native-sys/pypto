# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tensor.expand_clone using the PyPTO frontend.

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
    """Base test case for tensor.expand_clone."""

    __test__ = False  # Not a pytest test class

    def __init__(self, b: int = 8, n: int = 8, k: int = 8, broadcast_dim: int = 1, config=None):
        super().__init__(config)
        self.B = b
        self.N = n
        self.K = k
        self.broadcast_dim = broadcast_dim

    def get_name(self) -> str:
        return f"expand_clone_d{self.broadcast_dim}_{self.B}x{self.N}x{self.K}"

    def _input_shape(self) -> list[int]:
        if self.broadcast_dim == 0:
            return [1, self.N, self.K]
        if self.broadcast_dim == 1:
            return [self.B, 1, self.K]
        if self.broadcast_dim == 2:
            return [self.B, self.N, 1]
        raise ValueError(f"Unsupported broadcast_dim: {self.broadcast_dim}")

    def define_tensors(self) -> list[TensorSpec]:
        input_shape = self._input_shape()
        return [
            TensorSpec("x", input_shape, DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [self.B, self.N, self.K], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, N, K = self.B, self.N, self.K
        input_shape = self._input_shape()

        @pl.program
        class ExpandCloneProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def expand_clone_kernel(
                self,
                x: pl.Tensor[input_shape, pl.FP32],
                y: pl.Out[pl.Tensor[[B, N, K], pl.FP32]],
            ) -> pl.Tensor[[B, N, K], pl.FP32]:
                out: pl.Tensor[[B, N, K], pl.FP32] = pl.expand_clone(x, y)
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
        if self.broadcast_dim == 0:
            tensors["y"][:] = tensors["x"].repeat(self.B, 1, 1)
        elif self.broadcast_dim == 1:
            tensors["y"][:] = tensors["x"].repeat(1, self.N, 1)
        elif self.broadcast_dim == 2:
            tensors["y"][:] = tensors["x"].repeat(1, 1, self.K)
        else:
            raise ValueError(f"Unsupported broadcast_dim: {self.broadcast_dim}")


class TestExpandClonePTO(TestExpandClone):
    """Test expand_clone with PTO backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"expand_clone_pto_d{self.broadcast_dim}_{self.B}x{self.N}x{self.K}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class TestExpandCloneA5(TestExpandClone):
    """Test expand_clone with A5 (Ascend 950) backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"expand_clone_a5_d{self.broadcast_dim}_{self.B}x{self.N}x{self.K}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950


class TestExpandCloneOperations:
    """Test suite for expand_clone runtime execution."""

    @pytest.mark.parametrize("broadcast_dim", [0, 1, 2])
    def test_expand_clone_pto(self, test_runner, broadcast_dim):
        """Test expand_clone with PTO backend."""
        test_case = TestExpandClonePTO(broadcast_dim=broadcast_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (PTO, dim={broadcast_dim}): {result.error}"

    @pytest.mark.a5
    @pytest.mark.parametrize("broadcast_dim", [0, 1, 2])
    def test_expand_clone_a5(self, test_runner, broadcast_dim):
        """Test expand_clone with A5 (Ascend 950) backend."""
        test_case = TestExpandCloneA5(broadcast_dim=broadcast_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (A5, dim={broadcast_dim}): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
