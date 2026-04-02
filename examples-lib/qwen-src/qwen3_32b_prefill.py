# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Qwen3 single-layer prefill forward (batch=16, max_seq=4096).

Each session in the batch can have a different input sequence length (up to
MAX_SEQ).  The ``seq_lens`` input tensor (shape [BATCH], INT32) carries the
per-session token count.  Tensors are padded to MAX_SEQ on the sequence axis;
the program only processes valid tokens per session.

Design goals:
- keep a decode-like structure and reuse the same primitive ops
- fuse work in three large auto_incore scopes per token-tile
- all pl.slice / pl.slice of GM tensors use 512-B-aligned shapes
  (full TOK_TILE rows even on the tail tile; padding rows are harmless)
- scope 2 (attention + KV cache write) iterates only over valid tokens
  to avoid writing garbage into the KV cache
"""

import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096
HIDDEN = 5120
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 25600
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS

EPS = 1e-6
ATTN_SCALE = 0.08838834764831845
HIDDEN_INV = 1.0 / HIDDEN

# Prefill tuning knobs (start from the decode-tuned baseline).
K_CHUNK = 256
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 32
SEQ_TILE = 120
MLP_OUT_CHUNK = 64
TOK_TILE = 4


def build_qwen3_single_layer_prefill_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    NUM_HEADS_CFG = num_heads
    NUM_KV_HEADS_CFG = num_kv_heads
    HEAD_DIM_CFG = head_dim
    KV_HIDDEN_CFG = num_kv_heads * head_dim
    INTER_CFG = intermediate_size
    Q_PER_KV_CFG = num_heads // num_kv_heads

    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    KV_OUT_BLOCKS = (KV_HIDDEN_CFG + KV_OUT_CHUNK - 1) // KV_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_CFG + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK
    CACHE_ROWS = BATCH_CFG * NUM_KV_HEADS_CFG * MAX_SEQ_CFG

    @pl.program
    class Qwen3SingleLayerPrefill:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_prefill_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ_CFG, HEAD_DIM_CFG], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ_CFG, HEAD_DIM_CFG], pl.FP32],
            k_cache: pl.Tensor[[CACHE_ROWS, HEAD_DIM_CFG], pl.BF16],
            v_cache: pl.Tensor[[CACHE_ROWS, HEAD_DIM_CFG], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            wq: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            wk: pl.Tensor[[HIDDEN_CFG, KV_HIDDEN_CFG], pl.BF16],
            wv: pl.Tensor[[HIDDEN_CFG, KV_HIDDEN_CFG], pl.BF16],
            wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16]:
            for b in pl.parallel(0, BATCH_CFG, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)
                    # Scope 1: RMSNorm + Q/K/V projections for a token tile.
                    # Uses full [TOK_TILE, ...] views from hidden_states even on the
                    # tail tile — padding rows map to allocated-but-unused MAX_SEQ
                    # slots, keeping every GM view >= 512 B aligned.
                    with pl.auto_incore():
                        sq_sum = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)
                        sq_sum = pl.mul(sq_sum, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                            valid_shape=[1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK]
                            )
                            sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))

                        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))
                        q_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.BF16)
                        k_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN_CFG], dtype=pl.BF16)
                        v_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN_CFG], dtype=pl.BF16)

                        for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                            q0 = ob * Q_OUT_CHUNK
                            q_acc = pl.create_tensor([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                            q_acc = pl.mul(q_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                            valid_shape=[1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK]
                            )
                                gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                                normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                                wq_chunk = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                                q_acc = pl.add(q_acc, pl.matmul(pl.cast(normed, target_type=pl.BF16), wq_chunk))
                            q_proj_tile = pl.assemble(q_proj_tile, pl.cast(q_acc, target_type=pl.BF16), [0, q0])

                        for ob in pl.parallel(0, KV_OUT_BLOCKS, 1, chunk=8):
                            kv0 = ob * KV_OUT_CHUNK
                            k_acc = pl.create_tensor([TOK_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                            v_acc = pl.create_tensor([TOK_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                            k_acc = pl.mul(k_acc, 0.0)
                            v_acc = pl.mul(v_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                            valid_shape=[1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK]
                            )
                                gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                                normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                                normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                                wk_chunk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                wv_chunk = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                k_acc = pl.add(k_acc, pl.matmul(normed_bf16, wk_chunk))
                                v_acc = pl.add(v_acc, pl.matmul(normed_bf16, wv_chunk))
                            k_proj_tile = pl.assemble(k_proj_tile, pl.cast(k_acc, target_type=pl.BF16), [0, kv0])
                            v_proj_tile = pl.assemble(v_proj_tile, pl.cast(v_acc, target_type=pl.BF16), [0, kv0])

                    # Scope 2: RoPE + KV cache update + causal attention.
                    # Only valid tokens are processed (for ti in range(valid_tok))
                    # to avoid writing garbage into the KV cache.  Padding rows in
                    # attn_tile stay zero; scope 3 writes them to the padding area
                    # of `out` which the caller ignores.
                    attn_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    attn_tile = pl.mul(attn_tile, 0.0)
                    for ti in pl.range(valid_tok):
                        pos = p0 + ti
                        ctx_len = pos + 1
                        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                        cos_row = pl.slice(rope_cos, [1, HEAD_DIM_CFG], [pos, 0])
                        sin_row = pl.slice(rope_sin, [1, HEAD_DIM_CFG], [pos, 0])
                        cos_lo = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                        cos_hi = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                        sin_lo = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                        sin_hi = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])

                        with pl.auto_incore():
                            attn_row = pl.create_tensor([1, HIDDEN_CFG], dtype=pl.FP32)
                            attn_row = pl.mul(attn_row, 0.0)
                            # First loop: update KV cache (separate from attention to avoid tensor view issues)
                            for kvh in pl.parallel(0, NUM_KV_HEADS_CFG, 1, chunk=4):
                                kv_col = kvh * HEAD_DIM_CFG
                                k_row = pl.cast(
                                    pl.slice(k_proj_tile, [1, HEAD_DIM_CFG], [ti, kv_col]),
                                    target_type=pl.FP32,
                                )
                                k_lo = pl.slice(k_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                                k_hi = pl.slice(k_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                                k_rot = pl.create_tensor([1, HEAD_DIM_CFG], dtype=pl.FP32)
                                k_rot = pl.assemble(
                                    k_rot,
                                    pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo)),
                                    [0, 0],
                                )
                                k_rot = pl.assemble(
                                    k_rot,
                                    pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi)),
                                    [0, HEAD_DIM_CFG // 2],
                                )
                                cache_row = b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + kvh * MAX_SEQ_CFG + pos
                                k_cache = pl.assemble(
                                    k_cache,
                                    pl.cast(k_rot, target_type=pl.BF16),
                                    [cache_row, 0],
                                )
                                v_cache = pl.assemble(
                                    v_cache,
                                    pl.slice(v_proj_tile, [1, HEAD_DIM_CFG], [ti, kv_col]),
                                    [cache_row, 0],
                                )

                            # Second loop: compute attention
                            for h in pl.parallel(0, NUM_HEADS_CFG, 1, chunk=8):
                                kvh = h // Q_PER_KV_CFG
                                q_col = h * HEAD_DIM_CFG
                                q_row = pl.cast(
                                    pl.slice(q_proj_tile, [1, HEAD_DIM_CFG], [ti, q_col]),
                                    target_type=pl.FP32,
                                )
                                q_lo = pl.slice(q_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                                q_hi = pl.slice(q_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                                q_rot = pl.create_tensor([1, HEAD_DIM_CFG], dtype=pl.FP32)
                                q_rot = pl.assemble(
                                    q_rot,
                                    pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)),
                                    [0, 0],
                                )
                                q_rot = pl.assemble(
                                    q_rot,
                                    pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)),
                                    [0, HEAD_DIM_CFG // 2],
                                )
                                q_rot_bf16 = pl.cast(q_rot, target_type=pl.BF16)

                                oi = pl.create_tensor([1, HEAD_DIM_CFG], dtype=pl.FP32)
                                li = pl.create_tensor([1, 1], dtype=pl.FP32)
                                mi = pl.create_tensor([1, 1], dtype=pl.FP32)
                                oi = pl.mul(oi, 0.0)
                                li = pl.mul(li, 0.0)
                                mi = pl.mul(mi, 0.0)

                                for sb in pl.range(ctx_blocks):
                                    s0 = sb * SEQ_TILE
                                    valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                    cache_row0 = b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + kvh * MAX_SEQ_CFG + s0
                                    k_tile = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM_CFG], [cache_row0, 0],
                                                     valid_shape=[valid_len, HEAD_DIM_CFG])
                                    v_tile = pl.slice(v_cache, [SEQ_TILE, HEAD_DIM_CFG], [cache_row0, 0],
                                                     valid_shape=[valid_len, HEAD_DIM_CFG])
                                    scores = pl.mul(pl.matmul(q_rot_bf16, k_tile, b_trans=True), ATTN_SCALE)
                                    # TODO(valid_shape): once the compiler propagates valid_shape
                                    # from k_tile, scores will auto-get vs=[1, valid_len] and the
                                    # manual scores_valid view + exp_pad can be removed.
                                    scores_valid = pl.slice(
                                        scores,
                                        [1, SEQ_TILE],
                                        [0, 0],
                                        valid_shape=[1, valid_len],
                                    )
                                    cur_mi = pl.cast(pl.row_max(scores_valid), target_type=pl.FP32)
                                    exp_scores = pl.exp(pl.row_expand_sub(scores_valid, cur_mi))
                                    cur_li = pl.cast(pl.row_sum(exp_scores), target_type=pl.FP32)
                                    exp_pad = pl.create_tensor([1, SEQ_TILE], dtype=pl.FP32)
                                    exp_pad = pl.mul(exp_pad, 0.0)
                                    exp_pad = pl.assemble(exp_pad, exp_scores, [0, 0])
                                    oi_tmp = pl.matmul(
                                        pl.cast(exp_pad, target_type=pl.BF16),
                                        v_tile,
                                        out_dtype=pl.FP32,
                                    )
                                    if sb == 0:
                                        oi = oi_tmp
                                        li = cur_li
                                        mi = cur_mi
                                    else:
                                        mi_new = pl.maximum(mi, cur_mi)
                                        alpha = pl.exp(pl.sub(mi, mi_new))
                                        beta = pl.exp(pl.sub(cur_mi, mi_new))
                                        li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                                        oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp, beta))
                                        mi = mi_new
                                ctx = pl.row_expand_div(oi, li)
                                attn_row = pl.assemble(attn_row, ctx, [0, q_col])
                            attn_tile = pl.assemble(attn_tile, attn_row, [ti, 0])

                    # Scope 3: output projection + residual + post-rms + MLP + residual.
                    with pl.auto_incore():
                        resid1_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.FP32)
                        for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                            o0 = ob * Q_OUT_CHUNK
                            o_acc = pl.create_tensor([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                            o_acc = pl.mul(o_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                a_chunk = pl.cast(
                                    pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0]),
                                    target_type=pl.BF16,
                                )
                                w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                                o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                            resid = pl.reshape(pl.cast(
                                pl.slice(hidden_states, [1, TOK_TILE, Q_OUT_CHUNK], [b, p0, o0],
                                        valid_shape=[1, valid_tok, Q_OUT_CHUNK]),
                                target_type=pl.FP32,
                            ), [TOK_TILE, Q_OUT_CHUNK])
                            resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                        sq_sum = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)
                        sq_sum = pl.mul(sq_sum, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                        post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.BF16)
                        down_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.FP32)
                        down_proj_tile = pl.mul(down_proj_tile, 0.0)

                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                            post_norm_tile = pl.assemble(
                                post_norm_tile,
                                pl.cast(normed, target_type=pl.BF16),
                                [0, k0],
                            )

                        for ob in pl.range(MLP_OUT_BLOCKS):
                            o0 = ob * MLP_OUT_CHUNK
                            gate_acc = pl.create_tensor([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                            up_acc = pl.create_tensor([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                            gate_acc = pl.mul(gate_acc, 0.0)
                            up_acc = pl.mul(up_acc, 0.0)

                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                post_chunk = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                                up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                            for dob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                                d0 = dob * Q_OUT_CHUNK
                                down_prev = pl.slice(down_proj_tile, [TOK_TILE, Q_OUT_CHUNK], [0, d0])
                                w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, Q_OUT_CHUNK], [o0, d0])
                                down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                                down_proj_tile = pl.assemble(down_proj_tile, down_next, [0, d0])
                                if ob == MLP_OUT_BLOCKS - 1:
                                    down_acc = pl.add(
                                        pl.slice(down_proj_tile, [TOK_TILE, Q_OUT_CHUNK], [0, d0]),
                                        pl.slice(resid1_tile, [TOK_TILE, Q_OUT_CHUNK], [0, d0]),
                                    )
                                    out = pl.assemble(
                                        out,
                                        pl.cast(down_acc, target_type=pl.BF16),
                                        [b, p0, d0],
                                    )

            return out

    return Qwen3SingleLayerPrefill


# ---------------------------------------------------------------------------
# Build / run helpers
# ---------------------------------------------------------------------------


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
):
    import torch  # type: ignore[import]
    from pypto.runtime import TensorSpec

    kv_hidden = num_kv_heads * head_dim
    cache_rows = batch * num_kv_heads * max_seq_len

    seq_lens_data = torch.randint(1, max_seq_len + 1, (batch,), dtype=torch.int32)

    return [
        TensorSpec("hidden_states", [batch, max_seq_len, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens_data),
        TensorSpec("rope_cos", [max_seq_len, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("rope_sin", [max_seq_len, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("out", [batch, max_seq_len, hidden_size], torch.bfloat16, is_output=True),
    ]


def compile_and_run(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
    platform: str = "a2a3",
    device_id: int = 11,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_qwen3_single_layer_prefill_program(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )

    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=None,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.Ascend950,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
        return result
    if not result.passed and result.error:
        print(f"Result: {result.error}")
    return result


if __name__ == "__main__":
    compile_and_run()
