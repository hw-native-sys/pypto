# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
FFN Module System Tests for PyPTO.

Three FFN patterns are demonstrated (all on 64x64 tiles):
  1. FFN + GELU   — GELU(hidden @ gate_proj) @ down_proj
  2. FFN + SwiGLU — SwiGLU(hidden @ gate_proj, hidden @ up_proj) @ down_proj
  3. FFN + ReLU   — ReLU(hidden @ gate_proj) @ down_proj
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec, TestConfig

from examples.language.intermediate.ffn_activations import (
    FFNGeluProgram,
    FFNReluProgram,
    FFNSwigluProgram,
)


class TestFFNGelu(PTOTestCase):
    """FFN with GELU activation on 64x64 tiles.

    Pipeline: output = GELU(hidden_states @ gate_proj_weight) @ down_proj_weight
    GELU approximation: x * sigmoid(1.702 * x)
    """

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "ffn_gelu_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("hidden_states", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("gate_proj_weight", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("down_proj_weight", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FFNGeluProgram

    def compute_expected(self, tensors, params=None):
        hidden_states = tensors["hidden_states"]
        gate_proj_weight = tensors["gate_proj_weight"]
        down_proj_weight = tensors["down_proj_weight"]
        gate = hidden_states @ gate_proj_weight
        activated = gate * torch.sigmoid(1.702 * gate)
        tensors["output"][:] = activated @ down_proj_weight


class TestFFNSwiglu(PTOTestCase):
    """FFN with SwiGLU activation on 64x64 tiles.

    Pipeline: output = SwiGLU(gate, up) @ down_proj_weight
    where gate = hidden_states @ gate_proj_weight
          up   = hidden_states @ up_proj_weight
    """

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "ffn_swiglu_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("hidden_states", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("gate_proj_weight", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("up_proj_weight", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("down_proj_weight", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FFNSwigluProgram

    def compute_expected(self, tensors, params=None):
        hidden_states = tensors["hidden_states"]
        gate_proj_weight = tensors["gate_proj_weight"]
        up_proj_weight = tensors["up_proj_weight"]
        down_proj_weight = tensors["down_proj_weight"]
        gate = hidden_states @ gate_proj_weight
        up = hidden_states @ up_proj_weight
        activated = gate * torch.sigmoid(gate) * up
        tensors["output"][:] = activated @ down_proj_weight


class TestFFNRelu(PTOTestCase):
    """FFN with ReLU activation on 64x64 tiles.

    Pipeline: output = ReLU(hidden_states @ gate_proj_weight) @ down_proj_weight
    """

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "ffn_relu_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("hidden_states", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("gate_proj_weight", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("down_proj_weight", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return FFNReluProgram

    def compute_expected(self, tensors, params=None):
        hidden_states = tensors["hidden_states"]
        gate_proj_weight = tensors["gate_proj_weight"]
        down_proj_weight = tensors["down_proj_weight"]
        gate = hidden_states @ gate_proj_weight
        activated = torch.relu(gate)
        tensors["output"][:] = activated @ down_proj_weight


class TestFFNActivationOperations:
    """Test suite for FFN module operations."""

    def test_ffn_gelu_64x64(self, test_runner):
        """Test FFN with GELU activation: GELU(hidden @ gate_proj) @ down_proj."""
        test_case = TestFFNGelu(TestConfig(atol=3e-3, rtol=3e-3))
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_ffn_swiglu_64x64(self, test_runner):
        """Test FFN with SwiGLU activation: SwiGLU(gate, up) @ down_proj."""
        test_case = TestFFNSwiglu(TestConfig(atol=3e-3, rtol=3e-3))
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_ffn_relu_64x64(self, test_runner):
        """Test FFN with ReLU activation: ReLU(hidden @ gate_proj) @ down_proj."""
        test_case = TestFFNRelu(TestConfig(atol=3e-3, rtol=3e-3))
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
