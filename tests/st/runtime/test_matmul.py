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

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

from examples.language.beginner.matmul import MatmulProgram


class MatmulTestCase(PTOTestCase):
    """Test case for matrix multiplication (64x64)."""

    def get_name(self) -> str:
        return "matmul_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


class MatmulPTOTestCase(MatmulTestCase):
    """Test case for matmul with PTO backend and PTOAS optimization."""

    def get_name(self) -> str:
        return "matmul_pto_64x64"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.PTOAS

    def get_backend_type(self) -> BackendType:
        return BackendType.PTO


class TestMatmulOperations:
    """Test suite for matrix multiplication (matmul) operations."""

    def test_matmul_64x64(self, test_runner):
        """Test matmul with 64x64 matrices."""
        test_case = MatmulTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_matmul_pto_64x64(self, test_runner):
        """Test matmul with PTO backend and PTOAS optimization."""
        test_case = MatmulPTOTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (PTO): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
