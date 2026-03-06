# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Matrix multiplication with transpose system tests for PyPTO.

Corresponds to examples/language/beginner/matmul_transpose.py.

Tests three transpose variants:
1. C = A @ B^T  (MatmulTransBProgram)
2. C = A^T @ B  (MatmulTransAProgram)
3. C = A^T @ B^T (MatmulTransABProgram)

Each uses pl.transpose in Vector kernel(s) + matmul in Cube kernel.
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.beginner.matmul_transpose import (
    MatmulTransABProgram,
    MatmulTransAProgram,
    MatmulTransBProgram,
)


# =============================================================================
# Test Cases
# =============================================================================


class MatmulTransB(PTOTestCase):
    """C = A @ B^T (64x64)."""

    __test__ = False

    def get_name(self) -> str:
        return "matmul_trans_b_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MatmulTransBProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"].T)


class MatmulTransA(PTOTestCase):
    """C = A^T @ B (64x64)."""

    __test__ = False

    def get_name(self) -> str:
        return "matmul_trans_a_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MatmulTransAProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].T, tensors["b"])


class MatmulTransAB(PTOTestCase):
    """C = A^T @ B^T (64x64)."""

    __test__ = False

    def get_name(self) -> str:
        return "matmul_trans_ab_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MatmulTransABProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].T, tensors["b"].T)


# =============================================================================
# Test Class
# =============================================================================


class TestMatmulTranspose:
    """System tests for matmul with transpose operations."""

    def test_matmul_trans_b(self, test_runner):
        """Test C = A @ B^T (64x64)"""
        result = test_runner.run(MatmulTransB())
        assert result.passed, f"Matmul A @ B^T failed: {result.error}"

    def test_matmul_trans_a(self, test_runner):
        """Test C = A^T @ B (64x64)"""
        result = test_runner.run(MatmulTransA())
        assert result.passed, f"Matmul A^T @ B failed: {result.error}"

    def test_matmul_trans_ab(self, test_runner):
        """Test C = A^T @ B^T (64x64)"""
        result = test_runner.run(MatmulTransAB())
        assert result.passed, f"Matmul A^T @ B^T failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
