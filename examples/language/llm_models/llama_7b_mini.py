# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Minimal complete LLaMA 7B-style language model, using PyPTO language DSL.

Default single-head configuration:
  hidden_size = 64  (num_heads=1 × head_dim=64)
  num_heads   = 1
  head_dim    = 64
  head_dim/2  = 32  (for RoPE half-rotation)
  seq_len     = 16
  vocab_size  = 64  (reduced for hardware compatibility)

Architecture (same as llama_7b_full.py, simplified for single head):
  hidden [S, D]
    → Decoder Layer: RMSNorm → QKV[S,D] → RoPE → scaled Q@K^T[S,S] → +mask
                    → softmax → @V → dense → residual →
                    → RMSNorm → SwiGLU MLP → residual
    → Final RMSNorm
    → LM Head: [S,D] @ [D,V] → logits [S,V]

Key simplification vs llama_7b_full.py:
  - Single head: no head splitting (kernel_extract_head) or concat needed
  - All square projection weights are [D, D]
  - Score matrix is [S, S] with single-head

Constraints:
  - head_dim must be divisible by 16 (K-tiling in kernel_matmul_trans_b)

Use build_llama_mini_program() to obtain the @pl.program class parameterised
by runtime tensor dimensions.

Reference: LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
"""

import pypto.language as pl


def build_llama_mini_program(
    seq_len: int = 16,
    head_dim: int = 64,
    vocab_size: int = 64,
):
    """Build a minimal single-head LLaMA @pl.program for the given shapes.

    Returns the decorated program class (not an instance).  The tensor type
    annotations in all kernels and the orchestration function are filled in
    from the arguments so that the PyPTO DSL can resolve static dimensions
    at compile time.

    Parameters
    ----------
    seq_len:    sequence length (default: 16)
    head_dim:   per-head feature dimension = hidden_size for single head (default: 64)
    vocab_size: vocabulary size for LM head output (default: 64)

    Note: head_dim must be divisible by 16 (K-tiling constraint).
    """
    half_dim = head_dim // 2
    k_tile_width = 16  # hardware TMOV constraint: src.shape == dst.shape
    k1 = k_tile_width  # K-tile column offset 1
    k2 = k_tile_width * 2  # K-tile column offset 2
    k3 = k_tile_width * 3  # K-tile column offset 3
    inv_head_dim = 1.0 / head_dim  # pre-computed Python float for RMSNorm divisor
    attn_scale = head_dim**-0.5  # pre-computed Python float for attention scaling

    @pl.program
    class LlamaMiniProgram:
        """Minimal LLaMA 7B-style model: 1 decoder layer, 1 head, RoPE, LM head.

        Dimensions: hidden_size=head_dim, num_heads=1, seq_len, vocab_size.

        Architecture:
          hidden [S,D]
            → Decoder Layer: RMSNorm→QKV→RoPE→scaled-attn→dense→add
                           → RMSNorm→SwiGLU MLP→add
            → Final RMSNorm
            → LM Head: [S,D] @ [D,V] → logits [S,V]
        """

        # =========================================================================
        # InCore kernel: RMSNorm [S, D]
        # Formula: x / sqrt(mean(x^2) + eps), divisor = head_dim
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_rms_norm(
            self,
            x: pl.Tensor[[seq_len, head_dim], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, head_dim], pl.FP32]],
        ) -> pl.Tensor[[seq_len, head_dim], pl.FP32]:
            """RMSNorm: x / sqrt(mean(x^2) + eps) across head_dim."""
            tile_x: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.load(
                x, [0, 0], [seq_len, head_dim], target_memory=pl.MemorySpace.Vec
            )

            squared: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.mul(tile_x, tile_x)

            tmp: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.create_tile(
                [seq_len, head_dim], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            mean_sq: pl.Tile[[seq_len, 1], pl.FP32] = pl.row_sum(squared, tmp)
            # [S, 1] is ColMajor; reshape to [1, S] for scalar mul, then back
            mean_sq_T: pl.Tile[[1, seq_len], pl.FP32] = pl.reshape(mean_sq, [1, seq_len])
            mean_sq_T = pl.mul(mean_sq_T, inv_head_dim)
            mean_sq = pl.reshape(mean_sq_T, [seq_len, 1])

            mean_sq_T2: pl.Tile[[1, seq_len], pl.FP32] = pl.reshape(mean_sq, [1, seq_len])
            rms_T: pl.Tile[[1, seq_len], pl.FP32] = pl.add(mean_sq_T2, 1e-6)
            rms_T = pl.sqrt(rms_T)
            rms: pl.Tile[[seq_len, 1], pl.FP32] = pl.reshape(rms_T, [seq_len, 1])

            normalized: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.row_expand_div(tile_x, rms)
            out: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.store(normalized, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: matmul [S, D] @ [D, D] → [S, D]
        # Used for QKV projections, dense projection, and MLP projections.
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_matmul(
            self,
            a: pl.Tensor[[seq_len, head_dim], pl.FP32],
            b: pl.Tensor[[head_dim, head_dim], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, head_dim], pl.FP32]],
        ) -> pl.Tensor[[seq_len, head_dim], pl.FP32]:
            """[S,D] @ [D,D] → [S,D] matrix multiplication."""
            tile_a_l1 = pl.load(a, [0, 0], [seq_len, head_dim], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(b, [0, 0], [head_dim, head_dim], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
            out = pl.store(tile_c_l0c, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: LM head matmul [S, D] @ [D, V] → [S, V]
        # Separate kernel because vocab_size may differ from head_dim.
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_lm_head(
            self,
            a: pl.Tensor[[seq_len, head_dim], pl.FP32],
            b: pl.Tensor[[head_dim, vocab_size], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, vocab_size], pl.FP32]],
        ) -> pl.Tensor[[seq_len, vocab_size], pl.FP32]:
            """[S,D] @ [D,V] → [S,V] LM head projection."""
            tile_a_l1 = pl.load(a, [0, 0], [seq_len, head_dim], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(b, [0, 0], [head_dim, vocab_size], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
            out = pl.store(tile_c_l0c, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: Q @ K^T via K-tiled matmul [S, D] @ [S, D]^T → [S, S]
        # K-dimension (head_dim) is tiled into k_tile_width=16 blocks (unrolled at
        # compile time) to satisfy the hardware src.shape == dst.shape constraint
        # for TMOV into Right memory.
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_matmul_trans_b(
            self,
            a: pl.Tensor[[seq_len, head_dim], pl.FP32],
            b: pl.Tensor[[head_dim, seq_len], pl.FP32, pl.DN],
            output: pl.Out[pl.Tensor[[seq_len, seq_len], pl.FP32]],
        ) -> pl.Tensor[[seq_len, seq_len], pl.FP32]:
            """[S,D] @ [D,S](DN) → [S,S]: Q @ K^T via K-tiled 16-wide blocks.

            Tiles K-dimension (head_dim) into 4×k_tile_width=16 blocks so each
            TMOV operates on Mat[S,16] → Right[S,16], satisfying the hardware
            src.shape == dst.shape constraint. Requires head_dim == 4*k_tile_width.
            """
            # K-tile 0: columns [0 : k_tile_width]
            a0 = pl.load(a, [0, 0], [seq_len, k_tile_width], target_memory=pl.MemorySpace.Mat)
            b0 = pl.load(b, [0, 0], [k_tile_width, seq_len], target_memory=pl.MemorySpace.Mat, transpose=True)
            a0_l = pl.move(a0, target_memory=pl.MemorySpace.Left)
            b0_r = pl.move(b0, target_memory=pl.MemorySpace.Right)
            acc: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.matmul(a0_l, b0_r)

            # K-tile 1: columns [k1 : k1+k_tile_width]
            a1 = pl.load(a, [0, k1], [seq_len, k_tile_width], target_memory=pl.MemorySpace.Mat)
            b1 = pl.load(
                b, [k1, 0], [k_tile_width, seq_len], target_memory=pl.MemorySpace.Mat, transpose=True
            )
            a1_l = pl.move(a1, target_memory=pl.MemorySpace.Left)
            b1_r = pl.move(b1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul_acc(acc, a1_l, b1_r)

            # K-tile 2: columns [k2 : k2+k_tile_width]
            a2 = pl.load(a, [0, k2], [seq_len, k_tile_width], target_memory=pl.MemorySpace.Mat)
            b2 = pl.load(
                b, [k2, 0], [k_tile_width, seq_len], target_memory=pl.MemorySpace.Mat, transpose=True
            )
            a2_l = pl.move(a2, target_memory=pl.MemorySpace.Left)
            b2_r = pl.move(b2, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul_acc(acc, a2_l, b2_r)

            # K-tile 3: columns [k3 : k3+k_tile_width]
            a3 = pl.load(a, [0, k3], [seq_len, k_tile_width], target_memory=pl.MemorySpace.Mat)
            b3 = pl.load(
                b, [k3, 0], [k_tile_width, seq_len], target_memory=pl.MemorySpace.Mat, transpose=True
            )
            a3_l = pl.move(a3, target_memory=pl.MemorySpace.Left)
            b3_r = pl.move(b3, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul_acc(acc, a3_l, b3_r)

            out = pl.store(acc, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: probs @ V → [S, D]
        # [S, S] @ [S, D] → [S, D]
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_matmul_attn(
            self,
            a: pl.Tensor[[seq_len, seq_len], pl.FP32],
            b: pl.Tensor[[seq_len, head_dim], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, head_dim], pl.FP32]],
        ) -> pl.Tensor[[seq_len, head_dim], pl.FP32]:
            """[S,S] @ [S,D] → [S,D]: probs @ V for single-head attention."""
            tile_a_l1 = pl.load(a, [0, 0], [seq_len, seq_len], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(b, [0, 0], [seq_len, head_dim], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
            out = pl.store(tile_c_l0c, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: row-wise softmax [S, S]
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_softmax(
            self,
            a: pl.Tensor[[seq_len, seq_len], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, seq_len], pl.FP32]],
        ) -> pl.Tensor[[seq_len, seq_len], pl.FP32]:
            """Row-wise numerically stable softmax."""
            tile_a: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.load(a, [0, 0], [seq_len, seq_len])

            max_tmp: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.create_tile(
                [seq_len, seq_len], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            row_max: pl.Tile[[seq_len, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)

            shifted: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.row_expand_sub(tile_a, row_max)
            exp_shifted: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.exp(shifted)

            sum_tmp: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.create_tile(
                [seq_len, seq_len], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            row_sum: pl.Tile[[seq_len, 1], pl.FP32] = pl.row_sum(exp_shifted, sum_tmp)
            result: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.row_expand_div(exp_shifted, row_sum)

            out: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.store(result, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: scale attention scores [S, S] × (1/sqrt(head_dim))
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_scale_scores(
            self,
            scores: pl.Tensor[[seq_len, seq_len], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, seq_len], pl.FP32]],
        ) -> pl.Tensor[[seq_len, seq_len], pl.FP32]:
            """Scale attention scores by 1/sqrt(head_dim)."""
            tile: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.load(scores, [0, 0], [seq_len, seq_len])
            scaled: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.mul(tile, attn_scale)
            out: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.store(scaled, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: element-wise add [S, S]
        # Used for applying causal mask to attention scores.
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_add_scores(
            self,
            a: pl.Tensor[[seq_len, seq_len], pl.FP32],
            b: pl.Tensor[[seq_len, seq_len], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, seq_len], pl.FP32]],
        ) -> pl.Tensor[[seq_len, seq_len], pl.FP32]:
            """Element-wise addition [S,S]: output = a + b."""
            tile_a: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.load(a, [0, 0], [seq_len, seq_len])
            tile_b: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.load(b, [0, 0], [seq_len, seq_len])
            result: pl.Tile[[seq_len, seq_len], pl.FP32] = pl.add(tile_a, tile_b)
            out: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.store(result, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: element-wise add [S, D]
        # Used for residual connections.
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_add(
            self,
            a: pl.Tensor[[seq_len, head_dim], pl.FP32],
            b: pl.Tensor[[seq_len, head_dim], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, head_dim], pl.FP32]],
        ) -> pl.Tensor[[seq_len, head_dim], pl.FP32]:
            """Element-wise addition [S,D]: output = a + b."""
            tile_a: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.load(
                a, [0, 0], [seq_len, head_dim], target_memory=pl.MemorySpace.Vec
            )
            tile_b: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.load(
                b, [0, 0], [seq_len, head_dim], target_memory=pl.MemorySpace.Vec
            )
            result: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.add(tile_a, tile_b)
            out: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.store(result, [0, 0], output)
            return out

        # =========================================================================
        # InCore kernel: RoPE [S, D] using [S, D/2] cos/sin tables
        # Rotate-half pattern: x_left = x[:, :D/2], x_right = x[:, D/2:]
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_rope(
            self,
            x: pl.Tensor[[seq_len, head_dim], pl.FP32],
            cos_emb: pl.Tensor[[seq_len, half_dim], pl.FP32],
            sin_emb: pl.Tensor[[seq_len, half_dim], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, head_dim], pl.FP32]],
        ) -> pl.Tensor[[seq_len, head_dim], pl.FP32]:
            """Apply Rotary Position Embedding to a [S, head_dim] tensor.

            Rotate-half RoPE:
              x_left  = x[:, :half_dim]   rotated_left  = x_left * cos - x_right * sin
              x_right = x[:, half_dim:]   rotated_right = x_right * cos + x_left * sin
            """
            x_left: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.load(x, [0, 0], [seq_len, half_dim])
            x_right: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.load(x, [0, half_dim], [seq_len, half_dim])
            cos_tile: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.load(cos_emb, [0, 0], [seq_len, half_dim])
            sin_tile: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.load(sin_emb, [0, 0], [seq_len, half_dim])

            left_cos: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.mul(x_left, cos_tile)
            right_sin: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.mul(x_right, sin_tile)
            rotated_left: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.sub(left_cos, right_sin)

            right_cos: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.mul(x_right, cos_tile)
            left_sin: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.mul(x_left, sin_tile)
            rotated_right: pl.Tile[[seq_len, half_dim], pl.FP32] = pl.add(right_cos, left_sin)

            out_left: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.store(rotated_left, [0, 0], output)
            out: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.store(rotated_right, [0, half_dim], out_left)
            return out

        # =========================================================================
        # InCore kernel: SwiGLU [S, D]
        # Formula: SiLU(gate) * up = gate * sigmoid(gate) * up
        # =========================================================================

        @pl.function(type=pl.FunctionType.InCore)
        def kernel_swiglu(
            self,
            gate: pl.Tensor[[seq_len, head_dim], pl.FP32],
            up: pl.Tensor[[seq_len, head_dim], pl.FP32],
            output: pl.Out[pl.Tensor[[seq_len, head_dim], pl.FP32]],
        ) -> pl.Tensor[[seq_len, head_dim], pl.FP32]:
            """SwiGLU: SiLU(gate) * up = gate * sigmoid(gate) * up."""
            tile_gate: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.load(
                gate, [0, 0], [seq_len, head_dim], target_memory=pl.MemorySpace.Vec
            )
            tile_up: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.load(
                up, [0, 0], [seq_len, head_dim], target_memory=pl.MemorySpace.Vec
            )

            gate_neg: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.mul(tile_gate, -1.0)
            exp_neg: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.exp(gate_neg)
            denom: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.add(exp_neg, 1.0)
            sigmoid: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.recip(denom)
            swish: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.mul(tile_gate, sigmoid)
            result: pl.Tile[[seq_len, head_dim], pl.FP32] = pl.mul(swish, tile_up)

            out: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.store(result, [0, 0], output)
            return out

        # =========================================================================
        # Top-level orchestration: minimal 1-layer, 1-head LLaMA 7B model
        # =========================================================================

        @pl.function(type=pl.FunctionType.Orchestration)
        def llama_mini_orch(  # noqa: PLR0913
            self,
            # Input hidden states
            hidden: pl.Tensor[[seq_len, head_dim], pl.FP32],
            # Causal attention mask
            causal_mask: pl.Tensor[[seq_len, seq_len], pl.FP32],
            # RoPE embeddings: positions × head_dim/2
            cos_emb: pl.Tensor[[seq_len, half_dim], pl.FP32],
            sin_emb: pl.Tensor[[seq_len, half_dim], pl.FP32],
            # QKV and projection weights [D, D]
            wq: pl.Tensor[[head_dim, head_dim], pl.FP32],
            wk: pl.Tensor[[head_dim, head_dim], pl.FP32],
            wv: pl.Tensor[[head_dim, head_dim], pl.FP32],
            w_dense: pl.Tensor[[head_dim, head_dim], pl.FP32],
            # MLP weights [D, D]
            w_gate: pl.Tensor[[head_dim, head_dim], pl.FP32],
            w_up: pl.Tensor[[head_dim, head_dim], pl.FP32],
            w_down: pl.Tensor[[head_dim, head_dim], pl.FP32],
            # LM head weight [D, V]
            w_lm: pl.Tensor[[head_dim, vocab_size], pl.FP32],
            # Output
            logits: pl.Out[pl.Tensor[[seq_len, vocab_size], pl.FP32]],
        ) -> pl.Tensor[[seq_len, vocab_size], pl.FP32]:
            """Minimal LLaMA 7B-style model forward pass (1 layer, 1 head).

            Pipeline:
              hidden [S,D]
                → Decoder Layer:
                    RMSNorm → QKV[S,D] → RoPE →
                    scaled Q@K^T[S,S] → +mask → softmax → @V[S,D] →
                    dense → residual →
                    RMSNorm → SwiGLU MLP → residual
                → Final RMSNorm
                → LM Head: [S,D] @ [D,V] → logits [S,V]
            """
            # ===== Decoder Layer =====

            # Pre-attention RMSNorm
            normed: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            normed = self.kernel_rms_norm(hidden, normed)

            # QKV projections: [S,D] @ [D,D] → [S,D] each
            q: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor([seq_len, head_dim], dtype=pl.FP32)
            q = self.kernel_matmul(normed, wq, q)
            k: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor([seq_len, head_dim], dtype=pl.FP32)
            k = self.kernel_matmul(normed, wk, k)
            v: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor([seq_len, head_dim], dtype=pl.FP32)
            v = self.kernel_matmul(normed, wv, v)

            # Apply RoPE to Q and K
            q_rot: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            q_rot = self.kernel_rope(q, cos_emb, sin_emb, q_rot)
            k_rot_buf: pl.Tensor[[head_dim, seq_len], pl.FP32, pl.DN] = pl.create_tensor(
                [head_dim, seq_len], dtype=pl.FP32, layout=pl.DN
            )
            k_rot = self.kernel_rope(k, cos_emb, sin_emb, k_rot_buf)

            # Scaled causal dot-product attention
            scores: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.create_tensor(
                [seq_len, seq_len], dtype=pl.FP32
            )
            scores = self.kernel_matmul_trans_b(q_rot, k_rot, scores)
            scaled: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.create_tensor(
                [seq_len, seq_len], dtype=pl.FP32
            )
            scaled = self.kernel_scale_scores(scores, scaled)
            masked: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.create_tensor(
                [seq_len, seq_len], dtype=pl.FP32
            )
            masked = self.kernel_add_scores(scaled, causal_mask, masked)
            probs: pl.Tensor[[seq_len, seq_len], pl.FP32] = pl.create_tensor(
                [seq_len, seq_len], dtype=pl.FP32
            )
            probs = self.kernel_softmax(masked, probs)
            attn_out: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            attn_out = self.kernel_matmul_attn(probs, v, attn_out)

            # Dense projection + first residual
            dense_out: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            dense_out = self.kernel_matmul(attn_out, w_dense, dense_out)
            attn_res: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            attn_res = self.kernel_add(hidden, dense_out, attn_res)

            # Pre-MLP RMSNorm
            normed2: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            normed2 = self.kernel_rms_norm(attn_res, normed2)

            # SwiGLU MLP
            gate: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            gate = self.kernel_matmul(normed2, w_gate, gate)
            up: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor([seq_len, head_dim], dtype=pl.FP32)
            up = self.kernel_matmul(normed2, w_up, up)
            swish_up: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            swish_up = self.kernel_swiglu(gate, up, swish_up)
            mlp_out: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            mlp_out = self.kernel_matmul(swish_up, w_down, mlp_out)

            # Second residual → h1
            h1: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor([seq_len, head_dim], dtype=pl.FP32)
            h1 = self.kernel_add(attn_res, mlp_out, h1)

            # ===== Final RMSNorm =====
            h_normed: pl.Tensor[[seq_len, head_dim], pl.FP32] = pl.create_tensor(
                [seq_len, head_dim], dtype=pl.FP32
            )
            h_normed = self.kernel_rms_norm(h1, h_normed)

            # ===== LM Head: [S,D] @ [D,V] → logits [S,V] =====
            logits = self.kernel_lm_head(h_normed, w_lm, logits)

            return logits

    return LlamaMiniProgram
