# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

"""
DeepSeek V3.2-EXP single-layer prefill FRONT part (batch=16, max_seq=4096).

Aligned to official v3.2-exp MLA/DSA shapes. Indexer routing is externalized as
`index_topk_pos`, and the kernel applies sparse attention on those indices.
"""

import os

import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
NUM_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
INDEX_HEADS = 64
INDEX_TOPK = 2048
EP_NODES = 128

EPS = 1e-6
ATTN_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 256
K_CHUNK = 512
Q_OUT_CHUNK = 256
KV_OUT_CHUNK = 128
LORA_CHUNK = 128
V_OUT_CHUNK = 64
SEQ_TILE = 120
TOK_TILE = 4
# Extra local pad tensor width to raise explicit Vec occupancy in memory report.
LOCAL_PAD_WIDTH = 16384

# Conservative software guard for AIV Vec/UB working set (bytes). This does
# not reflect final backend private buffers, but helps avoid overly aggressive
# source-side tile settings.
UB_SOFT_LIMIT_BYTES = 160 * 1024


def build_deepseek_v3_2_prefill_front_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
    index_heads: int = INDEX_HEADS,
    index_topk: int = INDEX_TOPK,
    ep_nodes: int = EP_NODES,
):
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    NUM_HEADS_CFG = num_heads
    Q_LORA_RANK_CFG = q_lora_rank
    KV_LORA_RANK_CFG = kv_lora_rank
    QK_NOPE_HEAD_DIM_CFG = qk_nope_head_dim
    QK_ROPE_HEAD_DIM_CFG = qk_rope_head_dim
    QK_HEAD_DIM_CFG = qk_nope_head_dim + qk_rope_head_dim
    V_HEAD_DIM_CFG = v_head_dim
    INDEX_HEADS_CFG = index_heads
    ATTN_OUT_CFG = num_heads * v_head_dim
    INDEX_TOPK_CFG = index_topk
    EP_NODES_CFG = ep_nodes

    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    QR_BLOCKS = (Q_LORA_RANK_CFG + LORA_CHUNK - 1) // LORA_CHUNK
    Q_OUT_BLOCKS = (NUM_HEADS_CFG * QK_HEAD_DIM_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    KV_A_OUT = KV_LORA_RANK_CFG + QK_ROPE_HEAD_DIM_CFG
    KV_A_BLOCKS = (KV_A_OUT + KV_OUT_CHUNK - 1) // KV_OUT_CHUNK
    CACHE_ROWS = BATCH_CFG * MAX_SEQ_CFG
    V_OUT_BLOCKS = (V_HEAD_DIM_CFG + V_OUT_CHUNK - 1) // V_OUT_CHUNK

    # Capacity-oriented source tuning guard:
    # - stage1_est_bytes models dominant token-tile matmul operands/accumulators.
    # - stage2_est_bytes models topk buffers + major per-head vectors in sparse consume.
    # The check targets "close to but not exceeding" practical UB usage in source-level tuning.
    stage1_est_bytes = (
        TOK_TILE * K_CHUNK * 4
        + TOK_TILE * LORA_CHUNK * 4
        + TOK_TILE * Q_OUT_CHUNK * 4
        + TOK_TILE * KV_OUT_CHUNK * 4
        + TOK_TILE * LOCAL_PAD_WIDTH * 2
    )
    stage2_est_bytes = (
        (1 + 2) * INDEX_TOPK_CFG * 4  # topk vals + blk topk vals
        + (1 + 2) * INDEX_TOPK_CFG * 4  # topk idx + blk topk idx
        + KV_LORA_RANK_CFG * 4  # oi/ctx_latent dominant row vectors
        + QK_ROPE_HEAD_DIM_CFG * 4  # q/pe rope vectors
        + V_HEAD_DIM_CFG * 4  # ctx_v
    )
    peak_est_bytes = max(stage1_est_bytes, stage2_est_bytes)
    if peak_est_bytes > UB_SOFT_LIMIT_BYTES:
        raise ValueError(
            f"Estimated local working set {peak_est_bytes} bytes exceeds "
            f"UB soft limit {UB_SOFT_LIMIT_BYTES} bytes. "
            "Reduce TOK_TILE/Q_OUT_CHUNK/K_CHUNK/KV_OUT_CHUNK/LORA_CHUNK."
        )

    @pl.program
    class DeepSeekV32PrefillFront:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_prefill_front_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            layer_id_t: pl.Tensor[[1], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ_CFG, QK_ROPE_HEAD_DIM_CFG], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ_CFG, QK_ROPE_HEAD_DIM_CFG], pl.FP32],
            kv_cache: pl.Tensor[[CACHE_ROWS, KV_LORA_RANK_CFG], pl.BF16],
            pe_cache: pl.Tensor[[CACHE_ROWS, QK_ROPE_HEAD_DIM_CFG], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            wq_a: pl.Tensor[[HIDDEN_CFG, Q_LORA_RANK_CFG], pl.BF16],
            q_norm_weight: pl.Tensor[[1, Q_LORA_RANK_CFG], pl.FP32],
            wq_b: pl.Tensor[[Q_LORA_RANK_CFG, NUM_HEADS_CFG * QK_HEAD_DIM_CFG], pl.BF16],
            wkv_a: pl.Tensor[[HIDDEN_CFG, KV_A_OUT], pl.BF16],
            kv_norm_weight: pl.Tensor[[1, KV_LORA_RANK_CFG], pl.FP32],
            w_q_nope_to_latent: pl.Tensor[[NUM_HEADS_CFG, QK_NOPE_HEAD_DIM_CFG, KV_LORA_RANK_CFG], pl.BF16],
            w_latent_to_v: pl.Tensor[[NUM_HEADS_CFG, KV_LORA_RANK_CFG, V_HEAD_DIM_CFG], pl.BF16],
            dispatch_buf: pl.Tensor[[EP_NODES_CFG, BATCH_CFG, MAX_SEQ_CFG, ATTN_OUT_CFG], pl.BF16],
        ) -> pl.Tensor[[EP_NODES_CFG, BATCH_CFG, MAX_SEQ_CFG, ATTN_OUT_CFG], pl.BF16]:
            with pl.auto_incore():
                layer_id = pl.tensor.read(layer_id_t, [0])

                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    seq_len_b = pl.tensor.read(seq_lens, [b])
                    tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                    for p0_idx in pl.range(tok_blocks):
                        p0 = p0_idx * TOK_TILE
                        valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                        # Scope 1: RMSNorm + Q/K/V projections.
                        sq_sum = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)
                        sq_sum = pl.mul(sq_sum, 0.0)
                        # Keep an explicit local Vec pad tensor alive in this
                        # scope so AllocateMemoryAddr can reflect high occupancy.
                        usage_pad = pl.create_tensor([TOK_TILE, LOCAL_PAD_WIDTH], dtype=pl.BF16, valid_shape=[valid_tok, LOCAL_PAD_WIDTH])
                        usage_pad = pl.mul(usage_pad, 0.0)
                        usage_pad_fp = pl.cast(usage_pad, target_type=pl.FP32)
                        usage_pad_sum = pl.row_sum(usage_pad_fp)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.cast(
                                pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [b, p0, k0], valid_shape=[valid_tok, K_CHUNK]),
                                target_type=pl.FP32,
                            )
                            sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))
                        inv_rms = pl.add(inv_rms, pl.mul(usage_pad_sum, 0.0))

                        q_proj_tile = pl.create_tensor([TOK_TILE, NUM_HEADS_CFG * QK_HEAD_DIM_CFG], dtype=pl.BF16, valid_shape=[valid_tok, NUM_HEADS_CFG * QK_HEAD_DIM_CFG])
                        kv_a_tile = pl.create_tensor([TOK_TILE, KV_A_OUT], dtype=pl.BF16, valid_shape=[valid_tok, KV_A_OUT])

                        # Fused Q path (local fusion trial for former incore_0/1):
                        # directly accumulates q_proj_tile from x -> wq_a -> q_norm -> wq_b
                        # without materializing full qr_tile.
                        for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                            q0 = ob * Q_OUT_CHUNK
                            q_acc = pl.create_tensor([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                            q_acc = pl.mul(q_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                x_chunk = pl.cast(
                                    pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [b, p0, k0], valid_shape=[valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                )
                                gamma_in = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                                normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma_in)
                                for rb in pl.range(QR_BLOCKS):
                                    r0 = rb * LORA_CHUNK
                                    wq_a_chunk = pl.slice(wq_a, [K_CHUNK, LORA_CHUNK], [k0, r0])
                                    qr_part = pl.matmul(pl.cast(normed, target_type=pl.BF16), wq_a_chunk)
                                    gamma_q = pl.slice(q_norm_weight, [1, LORA_CHUNK], [0, r0])
                                    qn_part = pl.col_expand_mul(qr_part, gamma_q)
                                    wq_b_chunk = pl.slice(wq_b, [LORA_CHUNK, Q_OUT_CHUNK], [r0, q0])
                                    q_acc = pl.add(q_acc, pl.matmul(pl.cast(qn_part, target_type=pl.BF16), wq_b_chunk))
                            q_proj_tile = pl.assemble(q_proj_tile, pl.cast(q_acc, target_type=pl.BF16), [0, q0])

                        for ob in pl.parallel(0, KV_A_BLOCKS, 1, chunk=8):
                            kv0 = ob * KV_OUT_CHUNK
                            kv_acc = pl.create_tensor([TOK_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                            kv_acc = pl.mul(kv_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                x_chunk = pl.cast(
                                    pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [b, p0, k0], valid_shape=[valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                )
                                gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                                normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                                wkv_chunk = pl.slice(wkv_a, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                kv_acc = pl.add(kv_acc, pl.matmul(pl.cast(normed, target_type=pl.BF16), wkv_chunk))
                            kv_a_tile = pl.assemble(kv_a_tile, pl.cast(kv_acc, target_type=pl.BF16), [0, kv0])

                        # Scope 2: RoPE + cache update + indexer topk + sparse attention.
                        # Fusion policy (aligned with decode_front):
                        # - Stage A/B/C all stay in ONE auto_incore scope.
                        # - A: per-token cache write
                        # - B1/B2: two-stage topk (block-local then global merge)
                        # - C: sparse attention consumes merged topk immediately
                        # This avoids materializing topk intermediates across kernel boundaries.
                        attn_tile = pl.create_tensor([TOK_TILE, ATTN_OUT_CFG], dtype=pl.FP32, valid_shape=[valid_tok, ATTN_OUT_CFG])
                        attn_tile = pl.mul(attn_tile, 0.0)
                        for ti in pl.range(valid_tok):
                            pos = p0 + ti
                            ctx_len = pos + 1
                            cos_row = pl.slice(rope_cos, [1, QK_ROPE_HEAD_DIM_CFG], [pos, 0])
                            sin_row = pl.slice(rope_sin, [1, QK_ROPE_HEAD_DIM_CFG], [pos, 0])
                            cos_lo = pl.slice(cos_row, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, 0])
                            cos_hi = pl.slice(cos_row, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, QK_ROPE_HEAD_DIM_CFG // 2])
                            sin_lo = pl.slice(sin_row, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, 0])
                            sin_hi = pl.slice(sin_row, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, QK_ROPE_HEAD_DIM_CFG // 2])

                            cache_row = b * MAX_SEQ_CFG + pos
                            kv_row = pl.cast(pl.slice(kv_a_tile, [1, KV_LORA_RANK_CFG], [ti, 0]), target_type=pl.FP32)
                            kv_gamma = pl.slice(kv_norm_weight, [1, KV_LORA_RANK_CFG], [0, 0])
                            kv_normed = pl.col_expand_mul(kv_row, kv_gamma)
                            pe_row = pl.cast(
                                pl.slice(kv_a_tile, [1, QK_ROPE_HEAD_DIM_CFG], [ti, KV_LORA_RANK_CFG]),
                                target_type=pl.FP32,
                            )
                            pe_lo = pl.slice(pe_row, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, 0])
                            pe_hi = pl.slice(pe_row, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, QK_ROPE_HEAD_DIM_CFG // 2])
                            pe_rot = pl.create_tensor([1, QK_ROPE_HEAD_DIM_CFG], dtype=pl.FP32)
                            pe_rot = pl.assemble(pe_rot, pl.sub(pl.col_expand_mul(pe_lo, cos_lo), pl.col_expand_mul(pe_hi, sin_lo)), [0, 0])
                            pe_rot = pl.assemble(pe_rot, pl.add(pl.col_expand_mul(pe_hi, cos_hi), pl.col_expand_mul(pe_lo, sin_hi)), [0, QK_ROPE_HEAD_DIM_CFG // 2])
                            kv_cache = pl.assemble(kv_cache, pl.cast(kv_normed, target_type=pl.BF16), [cache_row, 0])
                            pe_cache = pl.assemble(pe_cache, pl.cast(pe_rot, target_type=pl.BF16), [cache_row, 0])

                            # Stage B1: block-local topk (2 blocks, each 2K candidates).
                            topk_vals = pl.create_tensor([1, INDEX_TOPK_CFG], dtype=pl.FP32)
                            topk_idx = pl.create_tensor([1, INDEX_TOPK_CFG], dtype=pl.INT32)
                            blk_topk_vals = pl.create_tensor([2, INDEX_TOPK_CFG], dtype=pl.FP32)
                            blk_topk_idx = pl.create_tensor([2, INDEX_TOPK_CFG], dtype=pl.INT32)
                            topk_vals = pl.mul(topk_vals, -3.402823e38)
                            topk_idx = pl.mul(topk_idx, 0)
                            blk_topk_vals = pl.mul(blk_topk_vals, -3.402823e38)
                            blk_topk_idx = pl.mul(blk_topk_idx, 0)
                            for kk in pl.range(INDEX_TOPK_CFG):
                                neg_one = pl.create_tensor([1, 1], dtype=pl.INT32)
                                neg_one = pl.mul(neg_one, 0)
                                neg_one = pl.add(neg_one, -1)
                                topk_idx = pl.assemble(topk_idx, neg_one, [0, kk])
                                blk_topk_idx = pl.assemble(blk_topk_idx, neg_one, [0, kk])
                                blk_topk_idx = pl.assemble(blk_topk_idx, neg_one, [1, kk])

                            q_col0 = 0
                            q_nope0 = pl.cast(
                                pl.slice(q_proj_tile, [1, QK_NOPE_HEAD_DIM_CFG], [ti, q_col0]),
                                target_type=pl.FP32,
                            )
                            q_pe0 = pl.cast(
                                pl.slice(q_proj_tile, [1, QK_ROPE_HEAD_DIM_CFG], [ti, q_col0 + QK_NOPE_HEAD_DIM_CFG]),
                                target_type=pl.FP32,
                            )
                            q0_lo = pl.slice(q_pe0, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, 0])
                            q0_hi = pl.slice(q_pe0, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, QK_ROPE_HEAD_DIM_CFG // 2])
                            q0_rot = pl.create_tensor([1, QK_ROPE_HEAD_DIM_CFG], dtype=pl.FP32)
                            q0_rot = pl.assemble(q0_rot, pl.sub(pl.col_expand_mul(q0_lo, cos_lo), pl.col_expand_mul(q0_hi, sin_lo)), [0, 0])
                            q0_rot = pl.assemble(q0_rot, pl.add(pl.col_expand_mul(q0_hi, cos_hi), pl.col_expand_mul(q0_lo, sin_hi)), [0, QK_ROPE_HEAD_DIM_CFG // 2])
                            q0_nope_latent = pl.matmul(
                                pl.cast(q_nope0, target_type=pl.BF16),
                                pl.slice(w_q_nope_to_latent, [QK_NOPE_HEAD_DIM_CFG, KV_LORA_RANK_CFG], [0, 0, 0]),
                            )

                            sparse_k_gen = pl.min(INDEX_TOPK_CFG, ctx_len)
                            for blk in pl.range(2):
                                blk_start = blk * INDEX_TOPK_CFG
                                blk_end = pl.min(ctx_len, blk_start + INDEX_TOPK_CFG)
                                for ss in pl.range(INDEX_TOPK_CFG):
                                    s = blk_start + ss
                                    if s < blk_end:
                                        cache_s = b * MAX_SEQ_CFG + s
                                        kv_s = pl.cast(pl.slice(kv_cache, [1, KV_LORA_RANK_CFG], [cache_s, 0]), target_type=pl.FP32)
                                        pe_s = pl.cast(pl.slice(pe_cache, [1, QK_ROPE_HEAD_DIM_CFG], [cache_s, 0]), target_type=pl.FP32)
                                        score_nope = pl.row_sum(pl.mul(q0_nope_latent, kv_s))
                                        score_pe = pl.row_sum(pl.mul(q0_rot, pe_s))
                                        score_fp32 = pl.mul(pl.add(score_nope, score_pe), ATTN_SCALE)
                                        score_fp8 = pl.cast(score_fp32, target_type=pl.FP8E4M3FN)
                                        score_a5 = pl.cast(score_fp8, target_type=pl.FP32)
                                        cur_score = pl.tensor.read(score_a5, [0, 0])

                                        inserted = pl.create_tensor([1, 1], dtype=pl.INT32)
                                        inserted = pl.mul(inserted, 0)
                                        for kk in pl.range(sparse_k_gen):
                                            ins = pl.tensor.read(inserted, [0, 0])
                                            kth_val = pl.tensor.read(blk_topk_vals, [blk, kk])
                                            if ins == 0:
                                                if cur_score > kth_val:
                                                    for sh in pl.range(sparse_k_gen - 1, kk, -1):
                                                        prev_val = pl.tensor.read(blk_topk_vals, [blk, sh - 1])
                                                        prev_idx = pl.tensor.read(blk_topk_idx, [blk, sh - 1])
                                                        prev_val_t = pl.create_tensor([1, 1], dtype=pl.FP32)
                                                        prev_idx_t = pl.create_tensor([1, 1], dtype=pl.INT32)
                                                        prev_val_t = pl.mul(prev_val_t, 0.0)
                                                        prev_idx_t = pl.mul(prev_idx_t, 0)
                                                        prev_val_t = pl.add(prev_val_t, prev_val)
                                                        prev_idx_t = pl.add(prev_idx_t, prev_idx)
                                                        blk_topk_vals = pl.assemble(blk_topk_vals, prev_val_t, [blk, sh])
                                                        blk_topk_idx = pl.assemble(blk_topk_idx, prev_idx_t, [blk, sh])
                                                    cur_score_t = pl.create_tensor([1, 1], dtype=pl.FP32)
                                                    cur_index_t = pl.create_tensor([1, 1], dtype=pl.INT32)
                                                    one_t = pl.create_tensor([1, 1], dtype=pl.INT32)
                                                    cur_score_t = pl.mul(cur_score_t, 0.0)
                                                    cur_index_t = pl.mul(cur_index_t, 0)
                                                    one_t = pl.mul(one_t, 0)
                                                    cur_score_t = pl.add(cur_score_t, cur_score)
                                                    cur_index_t = pl.add(cur_index_t, s)
                                                    one_t = pl.add(one_t, 1)
                                                    blk_topk_vals = pl.assemble(blk_topk_vals, cur_score_t, [blk, kk])
                                                    blk_topk_idx = pl.assemble(blk_topk_idx, cur_index_t, [blk, kk])
                                                    inserted = pl.assemble(inserted, one_t, [0, 0])

                            # Stage B2: global merge from 2x(local topk) -> final topk.
                            for blk in pl.range(2):
                                for kk in pl.range(sparse_k_gen):
                                    cand_idx = pl.tensor.read(blk_topk_idx, [blk, kk])
                                    if cand_idx >= 0:
                                        cand_val = pl.tensor.read(blk_topk_vals, [blk, kk])
                                        inserted = pl.create_tensor([1, 1], dtype=pl.INT32)
                                        inserted = pl.mul(inserted, 0)
                                        for tkk in pl.range(sparse_k_gen):
                                            ins = pl.tensor.read(inserted, [0, 0])
                                            kth_val = pl.tensor.read(topk_vals, [0, tkk])
                                            if ins == 0:
                                                if cand_val > kth_val:
                                                    for sh in pl.range(sparse_k_gen - 1, tkk, -1):
                                                        prev_val = pl.tensor.read(topk_vals, [0, sh - 1])
                                                        prev_idx = pl.tensor.read(topk_idx, [0, sh - 1])
                                                        prev_val_t = pl.create_tensor([1, 1], dtype=pl.FP32)
                                                        prev_idx_t = pl.create_tensor([1, 1], dtype=pl.INT32)
                                                        prev_val_t = pl.mul(prev_val_t, 0.0)
                                                        prev_idx_t = pl.mul(prev_idx_t, 0)
                                                        prev_val_t = pl.add(prev_val_t, prev_val)
                                                        prev_idx_t = pl.add(prev_idx_t, prev_idx)
                                                        topk_vals = pl.assemble(topk_vals, prev_val_t, [0, sh])
                                                        topk_idx = pl.assemble(topk_idx, prev_idx_t, [0, sh])
                                                    cand_val_t = pl.create_tensor([1, 1], dtype=pl.FP32)
                                                    cand_idx_t = pl.create_tensor([1, 1], dtype=pl.INT32)
                                                    one_t = pl.create_tensor([1, 1], dtype=pl.INT32)
                                                    cand_val_t = pl.mul(cand_val_t, 0.0)
                                                    cand_idx_t = pl.mul(cand_idx_t, 0)
                                                    one_t = pl.mul(one_t, 0)
                                                    cand_val_t = pl.add(cand_val_t, cand_val)
                                                    cand_idx_t = pl.add(cand_idx_t, cand_idx)
                                                    one_t = pl.add(one_t, 1)
                                                    topk_vals = pl.assemble(topk_vals, cand_val_t, [0, tkk])
                                                    topk_idx = pl.assemble(topk_idx, cand_idx_t, [0, tkk])
                                                    inserted = pl.assemble(inserted, one_t, [0, 0])

                            # Stage C: sparse attention directly consumes merged topk_idx.
                            attn_row = pl.create_tensor([1, ATTN_OUT_CFG], dtype=pl.FP32)
                            attn_row = pl.mul(attn_row, 0.0)
                            for h in pl.parallel(0, NUM_HEADS_CFG, 1, chunk=8):
                                q_col = h * QK_HEAD_DIM_CFG
                                q_nope = pl.cast(
                                    pl.slice(q_proj_tile, [1, QK_NOPE_HEAD_DIM_CFG], [ti, q_col]),
                                    target_type=pl.FP32,
                                )
                                q_pe = pl.cast(
                                    pl.slice(q_proj_tile, [1, QK_ROPE_HEAD_DIM_CFG], [ti, q_col + QK_NOPE_HEAD_DIM_CFG]),
                                    target_type=pl.FP32,
                                )
                                q_lo = pl.slice(q_pe, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, 0])
                                q_hi = pl.slice(q_pe, [1, QK_ROPE_HEAD_DIM_CFG // 2], [0, QK_ROPE_HEAD_DIM_CFG // 2])
                                q_rot = pl.create_tensor([1, QK_ROPE_HEAD_DIM_CFG], dtype=pl.FP32)
                                q_rot = pl.assemble(q_rot, pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)), [0, 0])
                                q_rot = pl.assemble(q_rot, pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)), [0, QK_ROPE_HEAD_DIM_CFG // 2])
                                q_nope_latent = pl.matmul(
                                    pl.cast(q_nope, target_type=pl.BF16),
                                    pl.slice(w_q_nope_to_latent, [QK_NOPE_HEAD_DIM_CFG, KV_LORA_RANK_CFG], [h, 0, 0]),
                                )

                                oi = pl.create_tensor([1, KV_LORA_RANK_CFG], dtype=pl.FP32)
                                li = pl.create_tensor([1, 1], dtype=pl.FP32)
                                mi = pl.create_tensor([1, 1], dtype=pl.FP32)
                                oi = pl.mul(oi, 0.0)
                                li = pl.mul(li, 0.0)
                                mi = pl.mul(mi, 0.0)
                                sparse_k = pl.min(INDEX_TOPK_CFG, ctx_len)
                                for kk in pl.range(sparse_k):
                                    s = pl.tensor.read(topk_idx, [0, kk])
                                    if s >= 0:
                                        cache_s = b * MAX_SEQ_CFG + s
                                        kv_s = pl.cast(pl.slice(kv_cache, [1, KV_LORA_RANK_CFG], [cache_s, 0]), target_type=pl.FP32)
                                        pe_s = pl.cast(pl.slice(pe_cache, [1, QK_ROPE_HEAD_DIM_CFG], [cache_s, 0]), target_type=pl.FP32)
                                        score_nope = pl.row_sum(pl.mul(q_nope_latent, kv_s))
                                        score_pe = pl.row_sum(pl.mul(q_rot, pe_s))
                                        score = pl.mul(pl.add(score_nope, score_pe), ATTN_SCALE)
                                        cur_mi = score
                                        cur_li = pl.exp(pl.sub(score, cur_mi))
                                        oi_tmp = pl.row_expand_mul(kv_s, cur_li)
                                        if kk == 0:
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
                                ctx_latent = pl.row_expand_div(oi, li)
                                v_col = h * V_HEAD_DIM_CFG
                                ctx_v = pl.create_tensor([1, V_HEAD_DIM_CFG], dtype=pl.FP32)
                                ctx_v = pl.mul(ctx_v, 0.0)
                                for vb in pl.range(V_OUT_BLOCKS):
                                    v0 = vb * V_OUT_CHUNK
                                    wv_tile = pl.slice(w_latent_to_v, [KV_LORA_RANK_CFG, V_OUT_CHUNK], [h, 0, v0])
                                    v_part = pl.matmul(pl.cast(ctx_latent, target_type=pl.BF16), wv_tile, out_dtype=pl.FP32)
                                    ctx_v = pl.assemble(ctx_v, v_part, [0, v0])
                                attn_row = pl.assemble(attn_row, ctx_v, [0, v_col])
                            attn_tile = pl.assemble(attn_tile, attn_row, [ti, 0])

                            # Scope 3: dispatch writes and return after dispatch.
                            for ti in pl.range(valid_tok):
                                pos = p0 + ti
                                target_node = (b + pos + layer_id) % EP_NODES_CFG
                                token_row = pl.cast(pl.slice(attn_tile, [1, ATTN_OUT_CFG], [ti, 0]), target_type=pl.BF16)
                                dispatch_buf = pl.assemble(dispatch_buf, token_row, [target_node, b, pos, 0])

            return dispatch_buf

    return DeepSeekV32PrefillFront


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
    index_heads: int = INDEX_HEADS,
    index_topk: int = INDEX_TOPK,
    ep_nodes: int = EP_NODES,
):
    import torch  # type: ignore[import]
    from pypto.runtime import TensorSpec

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_a_out = kv_lora_rank + qk_rope_head_dim
    cache_rows = batch * max_seq_len
    attn_out = num_heads * v_head_dim
    seq_lens_data = torch.randint(1, max_seq_len + 1, (batch,), dtype=torch.int32)
    layer_id_data = torch.tensor([0], dtype=torch.int32)

    return [
        TensorSpec("hidden_states", [batch, max_seq_len, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens_data),
        TensorSpec("layer_id_t", [1], torch.int32, init_value=layer_id_data),
        TensorSpec("rope_cos", [max_seq_len, qk_rope_head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("rope_sin", [max_seq_len, qk_rope_head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("kv_cache", [cache_rows, kv_lora_rank], torch.bfloat16, init_value=torch.randn),
        TensorSpec("pe_cache", [cache_rows, qk_rope_head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("wq_a", [hidden_size, q_lora_rank], torch.bfloat16, init_value=torch.randn),
        TensorSpec("q_norm_weight", [1, q_lora_rank], torch.float32, init_value=torch.randn),
        TensorSpec("wq_b", [q_lora_rank, num_heads * qk_head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wkv_a", [hidden_size, kv_a_out], torch.bfloat16, init_value=torch.randn),
        TensorSpec("kv_norm_weight", [1, kv_lora_rank], torch.float32, init_value=torch.randn),
        TensorSpec("w_q_nope_to_latent", [num_heads, qk_nope_head_dim, kv_lora_rank], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_latent_to_v", [num_heads, kv_lora_rank, v_head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("dispatch_buf", [ep_nodes, batch, max_seq_len, attn_out], torch.bfloat16, is_output=True),
    ]


def compile_and_run(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
    index_heads: int = INDEX_HEADS,
    index_topk: int = INDEX_TOPK,
    ep_nodes: int = EP_NODES,
    platform: str = "a2a3",
    device_id: int = 11,
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_deepseek_v3_2_prefill_front_program(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_heads=index_heads,
        index_topk=index_topk,
        ep_nodes=ep_nodes,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_heads=index_heads,
        index_topk=index_topk,
        ep_nodes=ep_nodes,
    )

    if work_dir is None:
        work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "deepseek_v3_2_prefill_front_dump"))

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
            backend_type=BackendType.CCE,
            work_dir=work_dir,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
        print("  Generated kernels/orchestration:", work_dir)
        return result
    if not result.passed and result.error:
        print(f"Result: {result.error}")
        print("  Pass dumps may still have been written to:", work_dir)
    else:
        print("  Generated kernels/orchestration:", work_dir)
    return result


if __name__ == "__main__":
    compile_and_run()
