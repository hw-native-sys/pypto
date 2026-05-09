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

import pytest
import torch
from examples.models.ffn import ffn_gelu, ffn_relu, ffn_swiglu


class TestFFNActivationOperations:
    """Test suite for FFN module operations."""

    def test_ffn_gelu_64x64(self, test_config):
        """Test FFN with GELU activation: GELU(hidden @ gate_proj) @ down_proj."""
        ffn_gelu._cache.clear()
        torch.manual_seed(0)
        hidden = torch.randn(64, 64, dtype=torch.float32)
        gate = torch.randn(64, 64, dtype=torch.float32)
        down = torch.randn(64, 64, dtype=torch.float32)
        output = torch.zeros(64, 64, dtype=torch.float32)

        ffn_gelu(hidden, gate, down, output, config=test_config)

        gate_out = hidden @ gate
        expected = (gate_out * torch.sigmoid(1.702 * gate_out)) @ down
        assert torch.allclose(output, expected, rtol=3e-3, atol=3e-3), (
            f"ffn_gelu failed: max diff = {(output - expected).abs().max().item()}"
        )

    def test_ffn_swiglu_64x64(self, test_config):
        """Test FFN with SwiGLU activation: SwiGLU(gate, up) @ down_proj."""
        ffn_swiglu._cache.clear()
        torch.manual_seed(0)
        hidden = torch.randn(64, 64, dtype=torch.float32)
        gate = torch.randn(64, 64, dtype=torch.float32)
        up = torch.randn(64, 64, dtype=torch.float32)
        down = torch.randn(64, 64, dtype=torch.float32)
        output = torch.zeros(64, 64, dtype=torch.float32)

        ffn_swiglu(hidden, gate, up, down, output, config=test_config)

        gate_out = hidden @ gate
        up_out = hidden @ up
        expected = (gate_out * torch.sigmoid(gate_out) * up_out) @ down
        assert torch.allclose(output, expected, rtol=3e-3, atol=3e-3), (
            f"ffn_swiglu failed: max diff = {(output - expected).abs().max().item()}"
        )

    def test_ffn_relu_64x64(self, test_config):
        """Test FFN with ReLU activation: ReLU(hidden @ gate_proj) @ down_proj."""
        ffn_relu._cache.clear()
        torch.manual_seed(0)
        hidden = torch.randn(64, 64, dtype=torch.float32)
        gate = torch.randn(64, 64, dtype=torch.float32)
        down = torch.randn(64, 64, dtype=torch.float32)
        output = torch.zeros(64, 64, dtype=torch.float32)

        ffn_relu(hidden, gate, down, output, config=test_config)

        gate_out = hidden @ gate
        expected = torch.relu(gate_out) @ down
        assert torch.allclose(output, expected, rtol=3e-3, atol=3e-3), (
            f"ffn_relu failed: max diff = {(output - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
