# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Activation Function System Tests for PyPTO.

Four activation patterns are demonstrated:
  1. SiLU   — x * sigmoid(x)
  2. GELU   — x * sigmoid(1.702 * x)
  3. SwiGLU — gate * sigmoid(gate) * up
  4. GeGLU  — gate * sigmoid(1.702 * gate) * up
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.intermediate.activation import (
    GegluProgram,
    GeluProgram,
    SiluProgram,
    SwigluProgram,
)


class TestSiluActivation(PTOTestCase):
    """SiLU (Swish) activation with 32x128 input: output = x * sigmoid(x)"""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "silu_activation_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SiluProgram

    def compute_expected(self, tensors, params=None):
        x = tensors["x"]
        tensors["output"][:] = x * torch.sigmoid(x)


class TestGeluActivation(PTOTestCase):
    """GELU activation with 32x128 input: output = x * sigmoid(1.702 * x)"""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "gelu_activation_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GeluProgram

    def compute_expected(self, tensors, params=None):
        x = tensors["x"]
        tensors["output"][:] = x * torch.sigmoid(1.702 * x)


class TestSwigluActivation(PTOTestCase):
    """SwiGLU activation with 32x128 input: output = gate * sigmoid(gate) * up"""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "swiglu_activation_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("gate", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("up", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SwigluProgram

    def compute_expected(self, tensors, params=None):
        gate = tensors["gate"]
        up = tensors["up"]
        tensors["output"][:] = gate * torch.sigmoid(gate) * up


class TestGegluActivation(PTOTestCase):
    """GeGLU activation with 32x128 input: output = gate * sigmoid(1.702 * gate) * up"""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "geglu_activation_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("gate", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("up", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return GegluProgram

    def compute_expected(self, tensors, params=None):
        gate = tensors["gate"]
        up = tensors["up"]
        tensors["output"][:] = gate * torch.sigmoid(1.702 * gate) * up


class TestActivationOperations:
    """Test suite for activation operations."""

    def test_silu_activation_32x128(self, test_runner):
        """Test SiLU (Swish) activation with 32x128 input."""
        test_case = TestSiluActivation()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_gelu_activation_32x128(self, test_runner):
        """Test GELU activation with 32x128 input."""
        test_case = TestGeluActivation()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_swiglu_activation_32x128(self, test_runner):
        """Test SwiGLU activation with 32x128 input."""
        test_case = TestSwigluActivation()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_geglu_activation_32x128(self, test_runner):
        """Test GeGLU activation with 32x128 input."""
        test_case = TestGegluActivation()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
