# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for matrix multiplication operation using PyPTO frontend.

This test validates the matmul operation implementation through the
pto-testing-framework, ensuring correct code generation and execution.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


class TestMatmul(PTOTestCase):
    __test__ = False  # Not a pytest test class

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, config=None):
        super().__init__(config)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                # store can support l0c -> GM directly
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                out_c = self.matmul(a, b, out_c)
                return out_c

        return MatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


class TestMatmulTranspose(PTOTestCase):
    """Matmul with B transposed: C = A @ B^T.

    B is stored as [N, K] in memory and transposed during the load to L1.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, config=None):
        super().__init__(config)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_transpose_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulTransposeProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_transpose(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(
                    b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                out_c = self.matmul_transpose(a, b, out_c)
                return out_c

        return MatmulTransposeProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].to(torch.float32), tensors["b"].to(torch.float32).T)


class TestMatmulATranspose(PTOTestCase):
    """Matmul with A transposed: C = A^T @ B.

    A is stored as [K, M] in memory and transposed during the load to L1.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, config=None):
        super().__init__(config)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_atranspose_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.K, self.M], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulATransposeProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_atranspose(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(
                    a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[K, M], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                out_c = self.matmul_atranspose(a, b, out_c)
                return out_c

        return MatmulATransposeProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].to(torch.float32).T, tensors["b"].to(torch.float32))


class TestMatmulABTranspose(PTOTestCase):
    """Matmul with both A and B transposed: C = A^T @ B^T.

    A is stored as [K, M] and B as [N, K] in memory, both transposed during load.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, config=None):
        super().__init__(config)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_abtranspose_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.K, self.M], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulABTransposeProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_abtranspose(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(
                    a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_b_l1 = pl.load(
                    b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[K, M], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                out_c = self.matmul_abtranspose(a, b, out_c)
                return out_c

        return MatmulABTransposeProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].to(torch.float32).T, tensors["b"].to(torch.float32).T)


class TestMatmulPTO(TestMatmul):
    """Test matmul with PTO backend and PTOAS optimization."""

    __test__ = False

    def get_name(self) -> str:
        return f"matmul_pto_{self.M}x{self.K}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO


class TestMatmulTransposePTO(TestMatmulTranspose):
    """Test matmul transpose with PTO backend and PTOAS optimization."""

    __test__ = False

    def get_name(self) -> str:
        return f"matmul_transpose_pto_{self.M}x{self.K}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO


class TestMatmulATransposePTO(TestMatmulATranspose):
    """Test matmul A transpose with PTO backend and PTOAS optimization."""

    __test__ = False

    def get_name(self) -> str:
        return f"matmul_atranspose_pto_{self.M}x{self.K}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO


class TestMatmulABTransposePTO(TestMatmulABTranspose):
    """Test matmul AB transpose with PTO backend and PTOAS optimization."""

    __test__ = False

    def get_name(self) -> str:
        return f"matmul_abtranspose_pto_{self.M}x{self.K}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO


class TestMatmulOperations:
    """Test suite for matrix multiplication (matmul) operations."""

    @pytest.mark.parametrize(
        "m,k,n",
        [
            (64, 64, 64),
            (128, 64, 128),
            (64, 128, 64),
        ],
    )
    def test_matmul(self, test_runner, m, k, n):
        """Test matmul with configurable matrix dimensions."""
        test_case = TestMatmul(m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "m,k,n",
        [(64, 64, 64), (128, 64, 128), (64, 128, 64), (32, 64, 32)],
    )
    def test_matmul_transpose(self, test_runner, m, k, n):
        """Test matmul with B transposed (C = A @ B^T)."""
        test_case = TestMatmulTranspose(m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "m,k,n",
        [
            (64, 64, 64),
            (128, 64, 128),
            (64, 128, 64),
        ],
    )
    def test_matmul_pto(self, test_runner, m, k, n):
        """Test matmul with PTO backend and PTOAS optimization."""
        test_case = TestMatmulPTO(m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (PTO): {result.error}"

    @pytest.mark.parametrize(
        "m,k,n",
        [(64, 64, 64), (128, 64, 128), (64, 128, 64), (32, 64, 32)],
    )
    def test_matmul_transpose_pto(self, test_runner, m, k, n):
        """Test matmul with B transposed (C = A @ B^T)."""
        test_case = TestMatmulTransposePTO(m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "m,k,n",
        [(64, 64, 64), (128, 64, 128), (64, 128, 64), (32, 64, 32)],
    )
    def test_matmul_atranspose(self, test_runner, m, k, n):
        """Test matmul with A transposed (C = A^T @ B)."""
        test_case = TestMatmulATranspose(m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "m,k,n",
        [(64, 64, 64), (128, 64, 128), (64, 128, 64), (32, 64, 32)],
    )
    def test_matmul_abtranspose(self, test_runner, m, k, n):
        """Test matmul with both A and B transposed (C = A^T @ B^T)."""
        test_case = TestMatmulABTranspose(m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "m,k,n",
        [(64, 64, 64), (128, 64, 128), (64, 128, 64), (32, 64, 32)],
    )
    def test_matmul_atranspose_pto(self, test_runner, m, k, n):
        """Test matmul A transpose with PTO backend."""
        test_case = TestMatmulATransposePTO(m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "m,k,n",
        [(64, 64, 64), (128, 64, 128), (64, 128, 64), (32, 64, 32)],
    )
    def test_matmul_abtranspose_pto(self, test_runner, m, k, n):
        """Test matmul AB transpose with PTO backend."""
        test_case = TestMatmulABTransposePTO(m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
