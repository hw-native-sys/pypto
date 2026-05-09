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

import pytest
import torch
from examples.kernels.activation import geglu, gelu, silu, swiglu


class TestSiluActivation:
    """SiLU (Swish) activation with 32x128 input: output = x * sigmoid(x)."""

    def test_silu_activation(self, test_config):
        silu._cache.clear()
        torch.manual_seed(0)
        x = torch.randn(32, 128, dtype=torch.float32)
        output = torch.zeros_like(x)
        silu(x, output, config=test_config)
        expected = x * torch.sigmoid(x)
        assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), (
            f"silu failed: max diff = {(output - expected).abs().max().item()}"
        )


class TestGeluActivation:
    """GELU activation with 32x128 input: output = x * sigmoid(1.702 * x)."""

    def test_gelu_activation(self, test_config):
        gelu._cache.clear()
        torch.manual_seed(0)
        x = torch.randn(32, 128, dtype=torch.float32)
        output = torch.zeros_like(x)
        gelu(x, output, config=test_config)
        expected = x * torch.sigmoid(1.702 * x)
        assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), (
            f"gelu failed: max diff = {(output - expected).abs().max().item()}"
        )


class TestSwigluActivation:
    """SwiGLU activation with 32x128 input: output = gate * sigmoid(gate) * up."""

    def test_swiglu_activation(self, test_config):
        swiglu._cache.clear()
        torch.manual_seed(0)
        gate = torch.randn(32, 128, dtype=torch.float32)
        up = torch.randn(32, 128, dtype=torch.float32)
        output = torch.zeros_like(gate)
        swiglu(gate, up, output, config=test_config)
        expected = gate * torch.sigmoid(gate) * up
        assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), (
            f"swiglu failed: max diff = {(output - expected).abs().max().item()}"
        )


class TestGegluActivation:
    """GeGLU activation with 32x128 input: output = gate * sigmoid(1.702 * gate) * up."""

    def test_geglu_activation(self, test_config):
        geglu._cache.clear()
        torch.manual_seed(0)
        gate = torch.randn(32, 128, dtype=torch.float32)
        up = torch.randn(32, 128, dtype=torch.float32)
        output = torch.zeros_like(gate)
        geglu(gate, up, output, config=test_config)
        expected = gate * torch.sigmoid(1.702 * gate) * up
        assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), (
            f"geglu failed: max diff = {(output - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
