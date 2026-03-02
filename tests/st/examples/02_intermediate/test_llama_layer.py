# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
LLaMA Decoder Layer System Tests for PyPTO.

Tests the complete LLaMA decoder layer composite operator:
  1. LlamaLayer  — pre-norm → QKV attention → dense proj → residual →
                   pre-norm → SwiGLU MLP → residual

Reference: framework/tests/cmake/scripts/golden/net/llama/llamalayer_golden.py
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.intermediate.llama_layer import LlamaLayerProgram


class TestLlamaLayer(PTOTestCase):
    """LLaMA decoder layer with B=1, N=1, S=64, D=64 (all FP32).

    Tensor shapes (BS=64, ND=64):
      hidden  [64, 64] — input hidden states
      wq      [64, 64] — query projection weight
      wk      [64, 64] — key projection weight
      wv      [64, 64] — value projection weight
      w_dense [64, 64] — dense (output) projection weight
      w_gate  [64, 64] — FFN gate projection weight
      w_up    [64, 64] — FFN up   projection weight
      w_down  [64, 64] — FFN down projection weight
      output  [64, 64] — layer output
    """

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "llama_layer_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        # Weights are initialised with small values (uniform [-0.03, 0.03])
        # to match the golden reference and prevent numerical overflow.
        # Pre-compute tensors directly (GoldenGenerator does not support arbitrary callables).
        def small_rand() -> torch.Tensor:
            return torch.rand([64, 64]) * 0.06 - 0.03

        return [
            TensorSpec("hidden", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("wq", [64, 64], DataType.FP32, init_value=small_rand()),
            TensorSpec("wk", [64, 64], DataType.FP32, init_value=small_rand()),
            TensorSpec("wv", [64, 64], DataType.FP32, init_value=small_rand()),
            TensorSpec("w_dense", [64, 64], DataType.FP32, init_value=small_rand()),
            TensorSpec("w_gate", [64, 64], DataType.FP32, init_value=small_rand()),
            TensorSpec("w_up", [64, 64], DataType.FP32, init_value=small_rand()),
            TensorSpec("w_down", [64, 64], DataType.FP32, init_value=small_rand()),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return LlamaLayerProgram

    def compute_expected(self, tensors, params=None):
        """Reference implementation matching LlamaLayerProgram.llama_layer_orch."""
        hidden = tensors["hidden"]
        wq = tensors["wq"]
        wk = tensors["wk"]
        wv = tensors["wv"]
        w_dense = tensors["w_dense"]
        w_gate = tensors["w_gate"]
        w_up = tensors["w_up"]
        w_down = tensors["w_down"]

        eps = 1e-6
        hidden_size = 64

        def rms_norm(h: torch.Tensor) -> torch.Tensor:
            """RMSNorm: h / sqrt(mean(h^2) + eps)."""
            mean_sq = (h**2).sum(dim=-1, keepdim=True) / hidden_size
            rms = torch.sqrt(mean_sq + eps)
            return h / rms

        # ===== Attention block =====
        # Pre-attention RMSNorm
        normed = rms_norm(hidden)

        # QKV projections
        q = normed @ wq
        k = normed @ wk
        v = normed @ wv

        # Scaled dot-product attention: softmax(Q @ K^T) @ V
        scores = q @ k.T
        probs = torch.softmax(scores, dim=-1)
        attn_out = probs @ v

        # Dense projection + first residual add
        dense_out = attn_out @ w_dense
        attn_res = hidden + dense_out

        # ===== MLP block =====
        # Pre-MLP RMSNorm
        normed2 = rms_norm(attn_res)

        # Gate and up projections
        gate = normed2 @ w_gate
        up = normed2 @ w_up

        # SwiGLU: SiLU(gate) * up = gate / (1 + exp(-gate)) * up
        swish_up = gate * torch.sigmoid(gate) * up

        # Down projection + second residual add
        mlp_out = swish_up @ w_down
        tensors["output"][:] = attn_res + mlp_out


class TestLlamaLayerOperations:
    """Test suite for the LLaMA decoder layer composite operator."""

    def test_llama_layer_64x64(self, test_runner):
        """Test LLaMA decoder layer with B=1, N=1, S=64, D=64."""
        test_case = TestLlamaLayer()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
