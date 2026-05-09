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

Corresponds to examples.kernels.fused_ops (02_fused_ops.py), implemented using @pl.jit.

Four fused operation patterns are demonstrated:
  1. fused_add_scale     — vector: c = (a + b) * 2.0
  2. fused_add_relu      — vector: c = relu(a + b)
  3. fused_matmul_bias   — cube + vector: c = matmul(a, b) + bias
  4. fused_linear_relu   — cube + vector: y = relu(matmul(x, w) + bias)
"""

import pytest
import torch
from examples.kernels.fused_ops import (
    fused_add_relu,
    fused_add_scale,
    fused_linear_relu,
    fused_matmul_bias,
)


class TestBasicFusedOps:
    """System tests for basic fused operator kernels.

    Verifies that fused kernels produce results matching PyTorch reference
    implementations across three fusion patterns:
      - Vector-only fusion (add+scale, add+relu)
      - Cube+vector fusion (matmul+bias)
      - Full linear layer (matmul+bias+relu)
    """

    def test_fused_add_scale(self, test_config):
        """Test fused add and scale: c = (a + b) * 2.0"""
        fused_add_scale._cache.clear()
        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)
        fused_add_scale(a, b, c, config=test_config)
        expected = (a + b) * 2.0
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"Fused add+scale failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_fused_add_relu(self, test_config):
        """Test fused add and relu: c = relu(a + b)"""
        fused_add_relu._cache.clear()
        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)
        fused_add_relu(a, b, c, config=test_config)
        expected = torch.relu(a + b)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"Fused add+relu failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_fused_matmul_bias(self, test_config):
        """Test fused matmul and bias add: c = matmul(a, b) + bias"""
        fused_matmul_bias._cache.clear()
        torch.manual_seed(0)
        a = torch.full((64, 64), 2.0, dtype=torch.float32)
        b = torch.full((64, 64), 3.0, dtype=torch.float32)
        bias = torch.randn(64, 64, dtype=torch.float32)
        c = torch.zeros((64, 64), dtype=torch.float32)
        fused_matmul_bias(a, b, bias, c, config=test_config)
        expected = torch.matmul(a, b) + bias
        assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3), (
            f"Fused matmul+bias failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_fused_linear_relu(self, test_config):
        """Test fused linear layer with relu: y = relu(matmul(x, w) + bias)"""
        fused_linear_relu._cache.clear()
        torch.manual_seed(0)
        x = torch.full((64, 64), 2.0, dtype=torch.float32)
        w = torch.full((64, 64), 3.0, dtype=torch.float32)
        bias = torch.randn(64, 64, dtype=torch.float32)
        y = torch.zeros((64, 64), dtype=torch.float32)
        fused_linear_relu(x, w, bias, y, config=test_config)
        expected = torch.relu(torch.matmul(x, w) + bias)
        assert torch.allclose(y, expected, rtol=1e-3, atol=1e-3), (
            f"Fused linear+relu failed: max diff = {(y - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
