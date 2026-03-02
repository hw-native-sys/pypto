# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
LLaMA decoder layer composite operator using PyPTO language DSL.

Implements a single LLaMA transformer decoder layer (B=1, N=1, S=64, D=64):

  LlamaLayerProgram — full LLaMA decoder layer:
    1. Pre-attention RMSNorm
    2. QKV projections (separate wq, wk, wv weights [64, 64])
    3. Scaled dot-product attention (Q @ K^T → softmax → @ V)
    4. Dense (output) projection + first residual add
    5. Pre-MLP RMSNorm
    6. SwiGLU MLP: gate_proj + up_proj → SiLU(gate) * up → down_proj
    7. Second residual add

Reference: framework/tests/cmake/scripts/golden/net/llama/llamalayer_golden.py
"""

import pypto.language as pl

# Shape constants (B=1, N=1, S=64, D=64 → BS=64, ND=64)
_S = 64  # sequence length (BS = B*S)
_D = 64  # hidden size (ND = N*D)


@pl.program
class LlamaLayerProgram:
    """Single LLaMA decoder layer: pre-norm → attention → MLP with residual connections."""

    # -------------------------------------------------------------------------
    # InCore kernel: RMSNorm [64, 64]
    # Formula: output = x / sqrt(mean(x^2) + eps)
    # -------------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_rms_norm(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """RMSNorm: x / sqrt(mean(x^2) + eps) across hidden dimension."""
        tile_x: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])

        # squared = x * x
        squared: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_x, tile_x)

        # mean_sq = row_sum(x^2) / hidden_size → [64, 1]
        tmp: pl.Tile[[64, 64], pl.FP32] = pl.create_tile(
            [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        mean_sq: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(squared, tmp)
        # [64, 1] is ColMajor; reshape to [1, 64] for scalar mul, then back
        mean_sq_T: pl.Tile[[1, 64], pl.FP32] = pl.reshape(mean_sq, [1, 64])
        mean_sq_T = pl.mul(mean_sq_T, 0.015625)  # 1.0 / 64  # type: ignore[reportArgumentType]
        mean_sq = pl.reshape(mean_sq_T, [64, 1])

        # rms = sqrt(mean_sq + eps) → [64, 1]
        mean_sq_T2: pl.Tile[[1, 64], pl.FP32] = pl.reshape(mean_sq, [1, 64])
        rms_T: pl.Tile[[1, 64], pl.FP32] = pl.add(mean_sq_T2, 1e-6)  # type: ignore[reportArgumentType]
        rms_T = pl.sqrt(rms_T)
        rms: pl.Tile[[64, 1], pl.FP32] = pl.reshape(rms_T, [64, 1])

        # normalized = x / rms (broadcast rms across hidden dim)
        normalized: pl.Tile[[64, 64], pl.FP32] = pl.row_expand_div(tile_x, rms)

        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(normalized, [0, 0], [64, 64], output)
        return out

    # -------------------------------------------------------------------------
    # InCore kernel: matmul [64, 64] @ [64, 64] → [64, 64]
    # -------------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_matmul(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """64x64 matrix multiplication: output = a @ b."""
        tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.l0c_store(tile_c_l0c, [0, 0], [64, 64], output)
        return out

    # -------------------------------------------------------------------------
    # InCore kernel: matmul with B transposed: [64, 64] @ [64, 64]^T
    # Used for Q @ K^T in attention
    # -------------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_matmul_trans_b(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """64x64 matrix multiplication with B transposed: output = a @ b^T."""
        tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right, transpose=True)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.l0c_store(tile_c_l0c, [0, 0], [64, 64], output)
        return out

    # -------------------------------------------------------------------------
    # InCore kernel: row-wise softmax [64, 64]
    # Formula: exp(x - max(x)) / sum(exp(x - max(x)))
    # -------------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_softmax(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Row-wise numerically stable softmax."""
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])

        # Row max for numerical stability
        max_tmp: pl.Tile[[64, 64], pl.FP32] = pl.create_tile(
            [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        row_max: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)

        # Subtract row max, exponentiate
        shifted: pl.Tile[[64, 64], pl.FP32] = pl.row_expand_sub(tile_a, row_max)
        exp_shifted: pl.Tile[[64, 64], pl.FP32] = pl.exp(shifted)

        # Divide by row sum
        sum_tmp: pl.Tile[[64, 64], pl.FP32] = pl.create_tile(
            [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        row_sum: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(exp_shifted, sum_tmp)
        result: pl.Tile[[64, 64], pl.FP32] = pl.row_expand_div(exp_shifted, row_sum)

        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    # -------------------------------------------------------------------------
    # InCore kernel: element-wise add [64, 64]
    # -------------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Element-wise addition: output = a + b."""
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64])
        result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    # -------------------------------------------------------------------------
    # InCore kernel: SwiGLU activation [64, 64]
    # Formula: SiLU(gate) * up = gate / (1 + exp(-gate)) * up
    # -------------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_swiglu(
        self,
        gate: pl.Tensor[[64, 64], pl.FP32],
        up: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """SwiGLU: SiLU(gate) * up = gate * sigmoid(gate) * up."""
        tile_gate: pl.Tile[[64, 64], pl.FP32] = pl.load(gate, [0, 0], [64, 64])
        tile_up: pl.Tile[[64, 64], pl.FP32] = pl.load(up, [0, 0], [64, 64])

        # SiLU(gate) = gate / (1 + exp(-gate))
        gate_neg: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_gate, -1.0)  # type: ignore[reportArgumentType]
        exp_neg: pl.Tile[[64, 64], pl.FP32] = pl.exp(gate_neg)
        denom: pl.Tile[[64, 64], pl.FP32] = pl.add(exp_neg, 1.0)  # type: ignore[reportArgumentType]
        sigmoid: pl.Tile[[64, 64], pl.FP32] = pl.recip(denom)
        swish: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_gate, sigmoid)

        # SwiGLU = SiLU(gate) * up
        result: pl.Tile[[64, 64], pl.FP32] = pl.mul(swish, tile_up)

        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    # -------------------------------------------------------------------------
    # Orchestration: full LLaMA decoder layer
    # -------------------------------------------------------------------------

    @pl.function(type=pl.FunctionType.Orchestration)
    def llama_layer_orch(
        self,
        hidden: pl.Tensor[[64, 64], pl.FP32],  # [BS, ND] = [64, 64]
        wq: pl.Tensor[[64, 64], pl.FP32],  # Query weight  [ND, ND]
        wk: pl.Tensor[[64, 64], pl.FP32],  # Key weight    [ND, ND]
        wv: pl.Tensor[[64, 64], pl.FP32],  # Value weight  [ND, ND]
        w_dense: pl.Tensor[[64, 64], pl.FP32],  # Dense (output projection) weight [ND, ND]
        w_gate: pl.Tensor[[64, 64], pl.FP32],  # FFN gate projection weight [ND, ND]
        w_up: pl.Tensor[[64, 64], pl.FP32],  # FFN up   projection weight [ND, ND]
        w_down: pl.Tensor[[64, 64], pl.FP32],  # FFN down projection weight [ND, ND]
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Full LLaMA decoder layer:
        residual → pre-norm → QKV → attention → dense → add →
        residual → pre-norm → gate/up proj → SwiGLU → down proj → add
        """
        # ===== Attention Block =====

        # Pre-attention RMSNorm
        normed: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        normed = self.kernel_rms_norm(hidden, normed)

        # QKV projections: normed @ w{q,k,v}
        q: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        q = self.kernel_matmul(normed, wq, q)
        k: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        k = self.kernel_matmul(normed, wk, k)
        v: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        v = self.kernel_matmul(normed, wv, v)

        # Scaled dot-product attention
        # scores = Q @ K^T  [64, 64]
        scores: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        scores = self.kernel_matmul_trans_b(q, k, scores)
        # probs = softmax(scores)
        probs: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        probs = self.kernel_softmax(scores, probs)
        # attn_out = probs @ V
        attn_out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        attn_out = self.kernel_matmul(probs, v, attn_out)

        # Dense (output) projection
        dense_out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        dense_out = self.kernel_matmul(attn_out, w_dense, dense_out)

        # First residual add: hidden + dense_out
        attn_res: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        attn_res = self.kernel_add(hidden, dense_out, attn_res)

        # ===== MLP Block =====

        # Pre-MLP RMSNorm
        normed2: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        normed2 = self.kernel_rms_norm(attn_res, normed2)

        # Gate and up projections
        gate: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        gate = self.kernel_matmul(normed2, w_gate, gate)
        up: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        up = self.kernel_matmul(normed2, w_up, up)

        # SwiGLU activation: SiLU(gate) * up
        swish_up: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        swish_up = self.kernel_swiglu(gate, up, swish_up)

        # Down projection
        mlp_out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        mlp_out = self.kernel_matmul(swish_up, w_down, mlp_out)

        # Second residual add: attn_res + mlp_out
        output: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        output = self.kernel_add(attn_res, mlp_out, output)

        return output
