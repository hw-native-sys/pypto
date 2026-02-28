# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
LayerNorm System Tests for PyPTO.

One layer normalization pattern is demonstrated:
  1. LayerNorm  — (x - mean) / sqrt(var + eps) * gamma + beta
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.intermediate.layer_norm import LayerNormProgram


class TestLayerNormCore(PTOTestCase):
    """LayerNorm with 4x64 input: normalize across hidden dim, then scale and shift."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "layer_norm_core_4x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("gamma", [1, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("beta", [1, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return LayerNormProgram

    def compute_expected(self, tensors, _params=None):
        x = tensors["x"]
        gamma = tensors["gamma"]
        beta = tensors["beta"]
        hidden_size = 64
        eps = 1e-5

        mean = x.sum(dim=-1, keepdim=True) / hidden_size
        centered = x - mean
        var = (centered**2).sum(dim=-1, keepdim=True) / hidden_size
        std = torch.sqrt(var + eps)
        normalized = centered / std
        tensors["output"][:] = normalized * gamma + beta


class TestLayerNormOperations:
    """Test suite for LayerNorm operations."""

    def test_layer_norm_core_4x64(self, test_runner):
        """Test LayerNorm: normalize across hidden dim (64), scale by gamma, shift by beta."""
        test_case = TestLayerNormCore()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
