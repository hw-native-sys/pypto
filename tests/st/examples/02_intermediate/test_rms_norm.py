# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
RMSNorm System Tests for PyPTO.

One RMS normalization pattern is demonstrated:
  1. RMSNorm  — x / sqrt(mean(x^2) + eps) * gamma
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.intermediate.rms_norm import RMSNormProgram


class TestRMSNormCore(PTOTestCase):
    """RMSNorm with 32x64 input: normalize by RMS across hidden dim, then scale by gamma."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "rms_norm_core_32x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("gamma", [1, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RMSNormProgram

    def compute_expected(self, tensors, _params=None):
        x = tensors["x"]
        gamma = tensors["gamma"]
        hidden_size = 64
        eps = 1e-5

        mean_sq = (x**2).sum(dim=-1, keepdim=True) / hidden_size
        rms = torch.sqrt(mean_sq + eps)
        normalized = x / rms
        tensors["output"][:] = normalized * gamma


class TestRMSNormOperations:
    """Test suite for RMSNorm operations."""

    def test_rms_norm_core_32x64(self, test_runner):
        """Test RMSNorm: normalize by RMS across hidden dim (64), scale by gamma."""
        test_case = TestRMSNormCore()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
