# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Basic Fused Operations System Tests for PyPTO.

Corresponds to examples/language/beginner/basic/basic_ops.py but implemented
using the PyPTO language DSL (@pl.program / pl.block).

Four fused operation patterns are demonstrated:
  1. FusedAddScale     — vector: c = (a + b) * 2.0
  2. FusedAddRelu      — vector: c = relu(a + b)
  3. FusedMatmulBias   — cube + vector: c = matmul(a, b) + bias
  4. FusedLinearRelu   — cube + vector: y = relu(matmul(x, w) + bias)
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.beginner.basic_ops import (
    FusedAddReluProgram,
    FusedAddScaleProgram,
    FusedLinearReluProgram,
    FusedMatmulBiasProgram,
)


class FusedAddScale(PTOTestCase):
    """Fused element-wise add and scale: c = (a + b) * 2.0

    Corresponds to basic_ops.py Example 2: Element-wise Operations.
    Two vector ops (add, scalar mul) are fused in a single InCore kernel,
    avoiding an intermediate global memory write-back.
    """

    __test__ = False

    def get_name(self) -> str:
        return "fused_add_scale_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FusedAddScaleProgram

    def compute_expected(self, tensors, params=None):
        """Expected: c = (a + b) * 2.0"""
        tensors["c"][:] = (tensors["a"] + tensors["b"]) * 2.0


class FusedAddRelu(PTOTestCase):
    """Fused element-wise add and relu: c = relu(a + b)

    Corresponds to basic_ops.py Example 4: Activation Functions.
    Add and relu activation are fused in a single vector InCore kernel.
    """

    __test__ = False

    def get_name(self) -> str:
        return "fused_add_relu_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FusedAddReluProgram

    def compute_expected(self, tensors, params=None):
        """Expected: c = relu(a + b)"""
        tensors["c"][:] = torch.relu(tensors["a"] + tensors["b"])


class FusedMatmulBias(PTOTestCase):
    """Fused matmul and bias add: c = matmul(a, b) + bias

    Corresponds to part of basic_ops.py Example 6: Combined Operations.
    Orchestrates two InCore kernels — cube matmul followed by vector add_bias —
    without exposing the intermediate result as a program output.
    """

    __test__ = False

    def get_name(self) -> str:
        return "fused_matmul_bias_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("bias", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FusedMatmulBiasProgram

    def compute_expected(self, tensors, params=None):
        """Expected: c = matmul(a, b) + bias"""
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"]) + tensors["bias"]


class FusedLinearRelu(PTOTestCase):
    """Fused linear layer with relu: y = relu(matmul(x, w) + bias)

    Corresponds to basic_ops.py Example 6: Combined Operations.
    Orchestrates two InCore kernels:
      - matmul_kernel: cube unit computes x @ w
      - add_bias_relu_kernel: vector unit fuses bias add and relu in one pass
    """

    __test__ = False

    def get_name(self) -> str:
        return "fused_linear_relu_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("w", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("bias", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("y", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FusedLinearReluProgram

    def compute_expected(self, tensors, params=None):
        """Expected: y = relu(matmul(x, w) + bias)"""
        tensors["y"][:] = torch.relu(torch.matmul(tensors["x"], tensors["w"]) + tensors["bias"])


# =============================================================================
# pytest test functions
# =============================================================================


class TestBasicFusedOps:
    """System tests for basic fused operator kernels.

    Verifies that fused kernels produce results matching PyTorch reference
    implementations across three fusion patterns:
      - Vector-only fusion (add+scale, add+relu)
      - Cube+vector fusion (matmul+bias)
      - Full linear layer (matmul+bias+relu)
    """

    def test_fused_add_scale(self, test_runner):
        """Test fused add and scale: c = (a + b) * 2.0"""
        test_case = FusedAddScale()
        result = test_runner.run(test_case)
        assert result.passed, f"Fused add+scale failed: {result.error}"

    def test_fused_add_relu(self, test_runner):
        """Test fused add and relu: c = relu(a + b)"""
        test_case = FusedAddRelu()
        result = test_runner.run(test_case)
        assert result.passed, f"Fused add+relu failed: {result.error}"

    def test_fused_matmul_bias(self, test_runner):
        """Test fused matmul and bias add: c = matmul(a, b) + bias"""
        test_case = FusedMatmulBias()
        result = test_runner.run(test_case)
        assert result.passed, f"Fused matmul+bias failed: {result.error}"

    def test_fused_linear_relu(self, test_runner):
        """Test fused linear layer with relu: y = relu(matmul(x, w) + bias)"""
        test_case = FusedLinearRelu()
        result = test_runner.run(test_case)
        assert result.passed, f"Fused linear+relu failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
