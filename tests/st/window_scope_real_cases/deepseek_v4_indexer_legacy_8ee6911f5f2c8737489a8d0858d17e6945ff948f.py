# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa
"""DeepSeek-V4 Indexer (decode). Mirrors model.py Indexer (line 380-433);
golden is a port of forward's decode branch (prefill `start_pos == 0` path is omitted).
The inner Compressor is invoked via golden_compressor (placeholder)."""

import pypto.language as pl
from config import DECODE_BATCH, DECODE_SEQ, FP32_NEG_INF, INT8_AMAX_EPS, INT8_SCALE_MAX
from config import FLASH as M
from indexer_compressor import indexer_compressor

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale  # softmax_scale folded with n_heads**-0.5 (model.py Indexer:418)
MAX_SEQ_LEN = M.max_position_embeddings
OFFSET = M.sliding_window  # ScalarSpec default; = win in attention orch; added to topk_idxs (model.py:432)

# kernel-local
COMPRESS_RATIO = 4  # the indexer only runs on ratio-4 layers
IDX_TOPK = M.index_topk


INNER_ROTATE = True
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_LEN = INNER_COFF * COMPRESS_RATIO

IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
SCORE_LEN = IDX_KV_LEN
START_POS = 254  # ScalarSpec default; >0 (decode) and (START_POS+S)%COMPRESS_RATIO==0

# tiling
CACHE_TILE = 32
MAX_CACHE_BLOCKS = SCORE_LEN // CACHE_TILE
Q_CHUCK = 128
Q_OUT_CHUCK = 128
ROPE_CHUCK = 16
QH_GROUP = 2 * IDX_N_HEADS
HEAD_DIM_CHUCK = 32
D_CHUCK = 32
HEAD_CHUCK = 16
QUANT_CHUNK = 128 if T >= 64 else 256


@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],  # caller passes freqs_cis[start_pos]
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16],
    score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32],
    topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32],
    start_pos: pl.Scalar[pl.INT32],  # decode step; varies per call
    offset: pl.Scalar[pl.INT32],  # added to topk_idxs (= win from attention orch)
    inner_rotate: pl.Scalar[pl.BOOL],
):
    cache_len = (start_pos + S) // COMPRESS_RATIO
    cache_blocks = (cache_len + CACHE_TILE - 1) // CACHE_TILE

    qr_i8 = qr
    qr_scale_dq = qr_scale

    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.BF16)
    for o0 in pl.parallel(0, IDX_N_HEADS * IDX_HEAD_DIM, Q_OUT_CHUCK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj"):
            qr_tile = qr_i8[:, 0:Q_CHUCK]
            wq_tile = wq_b[0:Q_CHUCK, o0 : o0 + Q_OUT_CHUCK]
            qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)

            for q0 in pl.range(Q_CHUCK, Q_LORA, Q_CHUCK):
                qr_tile = qr_i8[:, q0 : q0 + Q_CHUCK]
                wq_tile = wq_b[q0 : q0 + Q_CHUCK, o0 : o0 + Q_OUT_CHUCK]
                qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_write"):
            qr_acc_fp32 = pl.cast(qr_acc, target_type=pl.FP32, mode="none")
            wq_scale = pl.reshape(wq_b_scale[o0 : o0 + Q_OUT_CHUCK], [1, Q_OUT_CHUCK])
            qr_dequant = pl.col_expand_mul(pl.row_expand_mul(qr_acc_fp32, qr_scale_dq), wq_scale)
            qr_proj = pl.assemble(qr_proj, pl.cast(qr_dequant, target_type=pl.BF16, mode="rint"), [0, o0])

    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)

    for o0 in pl.parallel(0, T * IDX_N_HEADS, QH_GROUP):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_slice"):
            qr_proj_rope_full = qr_proj_flat[o0 : o0 + QH_GROUP, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
            qr_proj_rope_fp32 = pl.cast(qr_proj_rope_full, target_type=pl.FP32)
            even_tile = pl.tensor.gather(qr_proj_rope_fp32, mask_pattern=pl.tile.MaskPattern.P0101)
            odd_tile = pl.tensor.gather(qr_proj_rope_fp32, mask_pattern=pl.tile.MaskPattern.P1010)
            rope_even_acc = pl.cast(
                pl.sub(pl.col_expand_mul(even_tile, cos), pl.col_expand_mul(odd_tile, sin)),
                target_type=pl.BF16,
                mode="rint",
            )
            rope_odd_acc = pl.cast(
                pl.add(pl.col_expand_mul(even_tile, sin), pl.col_expand_mul(odd_tile, cos)),
                target_type=pl.BF16,
                mode="rint",
            )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_assemble"):
            rope_even_tile = rope_even_acc[:, 0:ROPE_CHUCK]
            rope_odd_tile = rope_odd_acc[:, 0:ROPE_CHUCK]
            even_select_tile_t = even_select[:, 0:ROPE_CHUCK]
            odd_select_tile_t = odd_select[:, 0:ROPE_CHUCK]
            rope_acc = pl.matmul(rope_even_tile, even_select_tile_t, out_dtype=pl.FP32, b_trans=True)
            rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

            for r0 in pl.range(ROPE_CHUCK, ROPE_HEAD_DIM // 2, ROPE_CHUCK):
                rope_even_tile = rope_even_acc[:, r0 : r0 + ROPE_CHUCK]
                rope_odd_tile = rope_odd_acc[:, r0 : r0 + ROPE_CHUCK]
                even_select_tile_t = even_select[:, r0 : r0 + ROPE_CHUCK]
                odd_select_tile_t = odd_select[:, r0 : r0 + ROPE_CHUCK]
                rope_acc = pl.matmul_acc(rope_acc, rope_even_tile, even_select_tile_t, b_trans=True)
                rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_hadamard"):
            qr_proj_tile = qr_proj_flat[o0 : o0 + QH_GROUP, 0:HEAD_DIM_CHUCK]
            hadamard_tile = hadamard[0:HEAD_DIM_CHUCK, :]
            qr_hadamard_acc = pl.matmul(qr_proj_tile, hadamard_tile, out_dtype=pl.FP32)

            for h0 in pl.range(HEAD_DIM_CHUCK, IDX_NOPE_HEAD_DIM, HEAD_DIM_CHUCK):
                qr_proj_tile = qr_proj_flat[o0 : o0 + QH_GROUP, h0 : h0 + HEAD_DIM_CHUCK]
                hadamard_tile = hadamard[h0 : h0 + HEAD_DIM_CHUCK, :]
                qr_hadamard_acc = pl.matmul_acc(qr_hadamard_acc, qr_proj_tile, hadamard_tile)

            for h0 in pl.range(IDX_NOPE_HEAD_DIM, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                rope_h0 = h0 - IDX_NOPE_HEAD_DIM
                qr_proj_rope_chunk = pl.cast(
                    rope_acc[:, rope_h0 : rope_h0 + HEAD_DIM_CHUCK], target_type=pl.BF16, mode="rint"
                )
                hadamard_tile = hadamard[h0 : h0 + HEAD_DIM_CHUCK, :]
                qr_hadamard_acc = pl.matmul_acc(qr_hadamard_acc, qr_proj_rope_chunk, hadamard_tile)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_hadamard_quant"):
            qh_amax = pl.full([1, QH_GROUP], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                qh_a_f32 = pl.cast(
                    pl.cast(qr_hadamard_acc[:, h0 : h0 + HEAD_DIM_CHUCK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
                qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_GROUP])
                qh_amax = pl.maximum(qh_amax, qh_a_max)
            qh_scale_quant_row = pl.div(pl.full([1, QH_GROUP], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
            qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_GROUP, 1])
            qr_hadamard_scale_dq = pl.assemble(qr_hadamard_scale_dq, qh_scale_dq, [o0, 0])
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_GROUP, 1])
            for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                qh_q_f32 = pl.cast(
                    pl.cast(qr_hadamard_acc[:, h1 : h1 + HEAD_DIM_CHUCK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
                qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
                qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
                qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
                qr_hadamard_i8 = pl.assemble(qr_hadamard_i8, qh_i8, [o0, h1])

    weights = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    x_flat = pl.reshape(x, [T, D])
    for h0 in pl.parallel(0, IDX_N_HEADS, HEAD_CHUCK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="weights_proj"):
            x_tile = x_flat[:, 0:D_CHUCK]
            weights_proj_tile = weights_proj[0:D_CHUCK, h0 : h0 + HEAD_CHUCK]
            weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)

            for d0 in pl.range(D_CHUCK, D, D_CHUCK):
                x_tile = x_flat[:, d0 : d0 + D_CHUCK]
                weights_proj_tile = weights_proj[d0 : d0 + D_CHUCK, h0 : h0 + HEAD_CHUCK]
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="weights_write"):
            weights_scale = pl.mul(weights_acc, WEIGHTS_SCALE)
            weights = pl.assemble(weights, weights_scale, [0, h0])

    inner_kv, inner_kv_state, inner_score_state, idx_kv_cache = indexer_compressor(
        x,
        inner_kv,
        inner_kv_state,
        inner_score_state,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        cos,
        sin,
        even_select,
        odd_select,
        hadamard,
        idx_kv_cache,
        start_pos,
        inner_rotate,
    )

    kv_cache_flat = pl.reshape(idx_kv_cache, [B * IDX_KV_LEN, IDX_HEAD_DIM])
    score_flat = pl.reshape(score, [T, SCORE_LEN])
    topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])

    for b in pl.parallel(B):
        t0 = b * S
        q0 = b * S * IDX_N_HEADS
        kv0 = b * IDX_KV_LEN

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_init"):
            neg_inf_score = pl.full([S, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
            score_flat = pl.assemble(score_flat, neg_inf_score, [t0, 0])

        for cb in pl.parallel(cache_blocks):
            cache0 = cb * CACHE_TILE
            valid_len = pl.min(CACHE_TILE, cache_len - cache0)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_quant"):
                kv_cache_tile_i8 = pl.create_tensor([CACHE_TILE, IDX_HEAD_DIM], dtype=pl.INT8)
                kv_amax = pl.full([1, CACHE_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                    kv_a_tile = kv_cache_flat[
                        kv0 + cache0 : kv0 + cache0 + CACHE_TILE, h0 : h0 + HEAD_DIM_CHUCK
                    ]
                    kv_a_f32 = pl.cast(kv_a_tile, target_type=pl.FP32)
                    kv_a_abs = pl.maximum(kv_a_f32, pl.neg(kv_a_f32))
                    kv_a_max = pl.reshape(pl.row_max(kv_a_abs), [1, CACHE_TILE])
                    kv_amax = pl.maximum(kv_amax, kv_a_max)
                kv_scale_quant_row = pl.div(
                    pl.full([1, CACHE_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax
                )
                kv_cache_scale_dq = pl.reshape(pl.recip(kv_scale_quant_row), [CACHE_TILE, 1])
                kv_scale_quant = pl.reshape(kv_scale_quant_row, [CACHE_TILE, 1])
                for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                    kv_q_tile = kv_cache_flat[
                        kv0 + cache0 : kv0 + cache0 + CACHE_TILE, h1 : h1 + HEAD_DIM_CHUCK
                    ]
                    kv_q_f32 = pl.cast(kv_q_tile, target_type=pl.FP32)
                    kv_q_scaled = pl.row_expand_mul(kv_q_f32, kv_scale_quant)
                    kv_q_i32 = pl.cast(kv_q_scaled, target_type=pl.INT32, mode="rint")
                    kv_q_half = pl.cast(kv_q_i32, target_type=pl.FP16, mode="round")
                    kv_q_i8 = pl.cast(kv_q_half, target_type=pl.INT8, mode="trunc")
                    kv_cache_tile_i8 = pl.assemble(kv_cache_tile_i8, kv_q_i8, [0, h1])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_accum"):
                qr_hadamard_tile = qr_hadamard_i8[q0 : q0 + S * IDX_N_HEADS, :]
                score_acc = pl.matmul(kv_cache_tile_i8, qr_hadamard_tile, out_dtype=pl.INT32, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_weighted_reduce"):
                qh_scale = pl.reshape(
                    qr_hadamard_scale_dq[q0 : q0 + S * IDX_N_HEADS, :], [1, S * IDX_N_HEADS]
                )
                score_tile = pl.cast(score_acc, target_type=pl.FP32, mode="none")
                score_tile = pl.col_expand_mul(pl.row_expand_mul(score_tile, kv_cache_scale_dq), qh_scale)
                for s in pl.range(S):
                    score_tile_s = score_tile[:, s * IDX_N_HEADS : (s + 1) * IDX_N_HEADS]
                    relu_score_s = pl.maximum(score_tile_s, pl.mul(score_tile_s, 0.0))
                    weights_row_s = pl.reshape(weights[t0 + s : t0 + s + 1, :], [1, IDX_N_HEADS])
                    weighted_score_s_t = pl.col_expand_mul(relu_score_s, weights_row_s)
                    weighted_score_s = pl.reshape(pl.row_sum(weighted_score_s_t), [1, CACHE_TILE])
                    weighted_score_valid_s = pl.slice(
                        weighted_score_s,
                        [1, CACHE_TILE],
                        [0, 0],
                        valid_shape=[1, valid_len],
                    )
                    score_flat = pl.assemble(score_flat, weighted_score_valid_s, [t0 + s, cache0])

        if cache_len > 0:
            for s in pl.parallel(S):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="topk"):
                    invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
                    topk_idxs_flat = pl.assemble(topk_idxs_flat, invalid_idxs, [t0 + s, 0])
                    offset_i32 = pl.cast(offset, target_type=pl.INT32)
                    score_row = score_flat[t0 + s : t0 + s + 1, :]
                    idx_init = pl.tensor.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
                    sorted_score_tile = pl.tensor.sort32(score_row, idx_init)
                    sorted_score_tile = pl.tensor.mrgsort(sorted_score_tile, block_len=64)
                    sorted_score_tile = pl.tensor.mrgsort(sorted_score_tile, block_len=256)
                    sorted_score_tile = pl.tensor.mrgsort(sorted_score_tile, block_len=1024)
                    topk_pairs = sorted_score_tile[:, 0 : 2 * IDX_TOPK]
                    topk_idxs_tile = pl.tensor.gather(
                        topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    raw_topk_idxs = pl.create_tensor([1, IDX_TOPK], dtype=pl.INT32)
                    raw_topk_idxs = pl.assemble(raw_topk_idxs, topk_idxs_tile, [0, 0])
                    valid_topk = pl.min(IDX_TOPK, cache_len)
                    topk_idxs_valid = pl.slice(
                        raw_topk_idxs,
                        [1, IDX_TOPK],
                        [0, 0],
                        valid_shape=[1, valid_topk],
                    )
                    topk_idxs_flat = pl.assemble(
                        topk_idxs_flat, pl.add(topk_idxs_valid, offset_i32), [t0 + s, 0]
                    )
        else:
            for s in pl.parallel(S):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="topk_init"):
                    invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
                    topk_idxs_flat = pl.assemble(topk_idxs_flat, invalid_idxs, [t0 + s, 0])

    return score, idx_kv_cache, topk_idxs


@pl.jit
def indexer_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16]],
    score: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.INT32]],
    start_pos: pl.Scalar[pl.INT32],
    offset: pl.Scalar[pl.INT32],
    inner_rotate: pl.Scalar[pl.BOOL],
):
    score, idx_kv_cache, topk_idxs = indexer(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        weights_proj,
        cos,
        sin,
        even_select,
        odd_select,
        hadamard,
        inner_kv,
        inner_kv_state,
        inner_score_state,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        score,
        topk_idxs,
        start_pos,
        offset,
        inner_rotate,
    )
    return score, idx_kv_cache, topk_idxs


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _quant_w_per_output_channel(w):
    """Per-output-channel INT8 quant for [in_features, out_features] weights."""
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_indexer(tensors):
    """Torch reference for Indexer.forward decode branch; prefill `start_pos == 0` path is omitted."""
    import torch
    from indexer_compressor import golden_compressor

    x = tensors["x"].float()
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    weights_proj = tensors["weights_proj"].float()
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()

    start_pos = int(tensors["start_pos"])
    offset = int(tensors["offset"])

    bsz, seqlen, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM
    end_pos = start_pos + seqlen

    if start_pos == 0:
        return

    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)

    x_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
    y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)

    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    q = q @ hadamard
    # W8A8C16: q and Indexer Cache are quantized per row to INT8 for score matmul,
    # then dequantized with q_scale * kv_scale.
    # flash: fp4_act_quant on q (FP4 simulation).

    inner_tensors = {
        "x": tensors["x"],
        "kv": tensors["inner_kv"],
        "kv_state": tensors["inner_kv_state"],
        "score_state": tensors["inner_score_state"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cos": tensors["cos"],
        "sin": tensors["sin"],
        "even_select": tensors["even_select"],
        "odd_select": tensors["odd_select"],
        "hadamard": tensors["hadamard"],
        "kv_cache": tensors["idx_kv_cache"],
        "start_pos": tensors["start_pos"],
        "rotate": tensors["inner_rotate"],
    }
    golden_compressor(inner_tensors)

    weights = (x @ weights_proj) * WEIGHTS_SCALE

    cache_len = end_pos // ratio

    idx_kv_cache = tensors["idx_kv_cache"].float()
    kv_view = idx_kv_cache[:bsz, :cache_len]

    q_i8, q_scale = _int8_quant_per_row(q.reshape(B * S * IDX_N_HEADS, IDX_HEAD_DIM))
    kv_i8, kv_scale = _int8_quant_per_row(kv_view.reshape(B * cache_len, IDX_HEAD_DIM))
    q_i8 = q_i8.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(B, S, IDX_N_HEADS, 1)
    kv_i8 = kv_i8.view(B, cache_len, IDX_HEAD_DIM)
    kv_scale = kv_scale.view(B, cache_len, 1)
    score_i32 = torch.einsum("bshd,btd->bsht", q_i8.to(torch.int32), kv_i8.to(torch.int32))
    score = score_i32.float() * q_scale * kv_scale.view(B, 1, 1, cache_len)
    score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=2)
    score_full = torch.full((bsz, seqlen, SCORE_LEN), FP32_NEG_INF, dtype=torch.float32)
    score_full[..., :cache_len] = score.to(torch.float32)
    tensors["score"][:] = score_full

    k = min(IDX_TOPK, cache_len)
    _, idx = score.topk(k, dim=-1)
    topk_idxs = torch.full((bsz, seqlen, SCORE_LEN), -1, dtype=torch.int32)
    topk_idxs[..., :k] = idx.to(torch.int32)
    topk_idxs[..., :k] += offset

    tensors["topk_idxs"][:] = topk_idxs.view(B, S, SCORE_LEN)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.rand(B, S, D)

    def init_qr():
        return torch.rand(T, Q_LORA)

    def init_wq_b():
        return torch.rand(Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM)

    def init_weights_proj():
        return torch.rand(D, IDX_N_HEADS)

    def init_cos():
        return torch.rand(1, ROPE_HEAD_DIM // 2)

    def init_sin():
        return torch.rand(1, ROPE_HEAD_DIM // 2)

    def init_odd_select():
        M = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            M[2 * i + 1, i] = 1
        return M

    def init_even_select():
        M = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            M[2 * i, i] = 1
        return M

    def init_hadamard():
        return torch.rand(IDX_HEAD_DIM, IDX_HEAD_DIM) * (IDX_HEAD_DIM**-0.5)

    def init_inner_kv_state():
        return torch.zeros(B, INNER_STATE_LEN, INNER_OUT_DIM)

    def init_inner_score_state():
        return torch.zeros(B, INNER_STATE_LEN, INNER_OUT_DIM)

    def init_inner_wkv():
        return torch.rand(D, INNER_OUT_DIM)

    def init_inner_wgate():
        return torch.rand(D, INNER_OUT_DIM)

    def init_inner_ape():
        return torch.rand(COMPRESS_RATIO, INNER_OUT_DIM)

    def init_inner_norm_w():
        return torch.ones(INNER_HEAD_DIM)

    def init_idx_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, IDX_HEAD_DIM)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    qr_i8, qr_scale = _int8_quant_per_row(init_qr())
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [1, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [1, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec(
            "even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select
        ),
        TensorSpec(
            "odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select
        ),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_kv", [B, S, INNER_HEAD_DIM], torch.float32),
        TensorSpec(
            "inner_kv_state",
            [B, INNER_STATE_LEN, INNER_OUT_DIM],
            torch.float32,
            init_value=init_inner_kv_state,
        ),
        TensorSpec(
            "inner_score_state",
            [B, INNER_STATE_LEN, INNER_OUT_DIM],
            torch.float32,
            init_value=init_inner_score_state,
        ),
        TensorSpec("inner_wkv", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.float32, init_value=init_inner_norm_w),
        TensorSpec(
            "idx_kv_cache",
            [B, IDX_KV_LEN, IDX_HEAD_DIM],
            torch.bfloat16,
            init_value=init_idx_kv_cache,
            is_output=True,
        ),
        # Outputs are fixed to SCORE_LEN; positions past cache_len are -inf for score and -1 for topk_idxs.
        TensorSpec("score", [B, S, SCORE_LEN], torch.float32, is_output=True),
        TensorSpec("topk_idxs", [B, S, SCORE_LEN], torch.int32, is_output=True),
        ScalarSpec("start_pos", torch.int32, START_POS),
        ScalarSpec("offset", torch.int32, OFFSET),
        ScalarSpec("inner_rotate", torch.bool, INNER_ROTATE),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=indexer_test,
        specs=build_tensor_specs(),
        golden_fn=golden_indexer,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "score": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "topk_idxs": topk_pair_compare("score"),
            "idx_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / IDX_KV_LEN),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
