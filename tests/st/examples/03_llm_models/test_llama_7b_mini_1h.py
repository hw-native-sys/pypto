# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
Minimal LLaMA 7B-style Complete Model System Tests for PyPTO.

Tests the minimal single-head LLaMA 7B-style model with RoPE, scaling, and LM head:
  1. Decoder Layer — pre-norm → 1-head RoPE-QKV scaled causal attention
                   → dense → residual → pre-norm → SwiGLU MLP → residual
  2. Final RMSNorm
  3. LM Head projection → logits [16, 64]

Minimal model dimensions (single-head):
  hidden_size = 64  (num_heads=1 × head_dim=64)
  num_heads   = 1
  head_dim    = 64
  head_dim/2  = 32  (for RoPE)
  seq_len     = 16
  vocab_size  = 64

Key simplification vs test_llama_7b_full.py:
  - Single head: no head splitting or concatenation
  - All square projection weights are [64, 64]
  - Score matrix is [16, 16] with single-head (vs 2×[16,16] in full)

Reference: examples/language/intermediate/llama_7b_mini.py
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.llm_models.llama_7b_mini import build_llama_mini_program


class TestLlamaMini(PTOTestCase):
    """Minimal LLaMA 7B-style model: 1 head, RoPE, scaling, LM head (S=16, D=64, FP32).

    Tensor shapes:
      hidden      [16, 64]  — input hidden states
      causal_mask [16, 16]  — additive causal mask: 0 on lower tri, -1e9 on upper tri
      cos_emb     [16, 32]  — precomputed cos values for RoPE (positions × head_dim/2)
      sin_emb     [16, 32]  — precomputed sin values for RoPE (positions × head_dim/2)

      wq          [64, 64]  — query weight
      wk          [64, 64]  — key weight
      wv          [64, 64]  — value weight
      w_dense     [64, 64]  — output projection weight
      w_gate      [64, 64]  — FFN gate weight
      w_up        [64, 64]  — FFN up weight
      w_down      [64, 64]  — FFN down weight

      w_lm        [64, 64]  — LM head weight [hidden, vocab]
      output      [16, 64]  — logits [S, vocab] (output)
    """

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "llama_7b_mini_1h_s16"

    def define_tensors(self) -> list[TensorSpec]:
        # Small weights to prevent overflow through deep computation chain.
        def small_rand(*shape: int) -> torch.Tensor:
            return torch.rand(shape) * 0.04 - 0.02

        # Causal mask: 0 on lower triangle, -1e9 on upper triangle
        causal_mask = torch.zeros(16, 16).masked_fill(torch.triu(torch.ones(16, 16), diagonal=1).bool(), -1e9)

        # Precompute RoPE cos/sin embeddings.
        # theta_i = 1 / (10000 ^ (2i / head_dim)) for i in [0, head_dim/2)
        head_dim = 64
        positions = torch.arange(16).float().unsqueeze(1)  # [16, 1]
        freqs = torch.arange(32).float()
        theta = 1.0 / (10000.0 ** (2 * freqs / head_dim))  # [32]
        angles = positions * theta.unsqueeze(0)  # [16, 32]
        cos_emb = torch.cos(angles)
        sin_emb = torch.sin(angles)

        return [
            TensorSpec("hidden", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("causal_mask", [16, 16], DataType.FP32, init_value=causal_mask),
            TensorSpec("cos_emb", [16, 32], DataType.FP32, init_value=cos_emb),
            TensorSpec("sin_emb", [16, 32], DataType.FP32, init_value=sin_emb),
            # Attention and projection weights [hidden, hidden] = [64, 64]
            TensorSpec("wq", [64, 64], DataType.FP32, init_value=small_rand(64, 64)),
            TensorSpec("wk", [64, 64], DataType.FP32, init_value=small_rand(64, 64)),
            TensorSpec("wv", [64, 64], DataType.FP32, init_value=small_rand(64, 64)),
            TensorSpec("w_dense", [64, 64], DataType.FP32, init_value=small_rand(64, 64)),
            TensorSpec("w_gate", [64, 64], DataType.FP32, init_value=small_rand(64, 64)),
            TensorSpec("w_up", [64, 64], DataType.FP32, init_value=small_rand(64, 64)),
            TensorSpec("w_down", [64, 64], DataType.FP32, init_value=small_rand(64, 64)),
            # LM head weight [hidden_size, vocab_size] = [64, 64]
            TensorSpec("w_lm", [64, 64], DataType.FP32, init_value=small_rand(64, 64)),
            # Output: logits [S, vocab] = [16, 64]
            TensorSpec("output", [16, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return build_llama_mini_program()

    def compute_expected(self, tensors, params=None):
        """Reference implementation matching llama_mini_orch (seq_len=16, head_dim=64)."""
        hidden = tensors["hidden"]  # [16, 64]
        causal_mask = tensors["causal_mask"]  # [16, 16]
        cos_emb = tensors["cos_emb"]  # [16, 32]
        sin_emb = tensors["sin_emb"]  # [16, 32]

        wq = tensors["wq"]  # [64, 64]
        wk = tensors["wk"]
        wv = tensors["wv"]
        w_dense = tensors["w_dense"]
        w_gate = tensors["w_gate"]
        w_up = tensors["w_up"]
        w_down = tensors["w_down"]

        w_lm = tensors["w_lm"]  # [64, 64]

        eps = 1e-6
        hidden_size = 64
        scale = 0.125  # 1 / sqrt(head_dim=64)

        def rms_norm(h: torch.Tensor) -> torch.Tensor:
            """RMSNorm with divisor=hidden_size=64."""
            mean_sq = (h**2).sum(dim=-1, keepdim=True) / hidden_size
            rms = torch.sqrt(mean_sq + eps)
            return h / rms

        def apply_rope(x_head: torch.Tensor) -> torch.Tensor:
            """Apply RoPE to a [S, head_dim] tensor using shared cos/sin tables.

            Rotate-half pattern:
              x_left  = x_head[:, :32]
              x_right = x_head[:, 32:]
              rotated_left  = x_left * cos - x_right * sin
              rotated_right = x_right * cos + x_left * sin
            """
            x_left = x_head[:, :32]
            x_right = x_head[:, 32:]
            rotated_left = x_left * cos_emb - x_right * sin_emb
            rotated_right = x_right * cos_emb + x_left * sin_emb
            return torch.cat([rotated_left, rotated_right], dim=-1)

        # ===== Attention block =====
        normed = rms_norm(hidden)  # [16, 64]

        # QKV projections
        q = normed @ wq  # [16, 64]
        k = normed @ wk
        v = normed @ wv

        # Apply RoPE to Q and K (single head, no splitting needed)
        q = apply_rope(q)
        k = apply_rope(k)

        # Scaled causal dot-product attention
        scores = q @ k.T * scale  # [16, 16], scaled by 1/sqrt(head_dim)
        masked = scores + causal_mask  # apply causal mask
        probs = torch.softmax(masked, dim=-1)
        attn_out = probs @ v  # [16, 64]

        # Dense projection + first residual
        dense_out = attn_out @ w_dense
        attn_res = hidden + dense_out

        # ===== MLP block =====
        normed2 = rms_norm(attn_res)

        gate = normed2 @ w_gate
        up = normed2 @ w_up
        swish_up = gate * torch.sigmoid(gate) * up  # SiLU(gate) * up
        mlp_out = swish_up @ w_down

        h1 = attn_res + mlp_out

        # ===== Final RMSNorm =====
        h_normed = rms_norm(h1)  # [16, 64]

        # ===== LM Head: [16,64] @ [64,64] → logits [16,64] =====
        logits = h_normed @ w_lm
        tensors["output"][:] = logits


class TestLlamaMiniOperations:
    """Test suite for the minimal LLaMA 7B-style model (1 head, RoPE, LM head)."""

    def test_llama_7b_mini_1h_s16(self, test_runner):
        """Test minimal LLaMA 7B model (1 layer, 1-head RoPE, scaling, LM head, S=16)."""
        test_case = TestLlamaMini()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
