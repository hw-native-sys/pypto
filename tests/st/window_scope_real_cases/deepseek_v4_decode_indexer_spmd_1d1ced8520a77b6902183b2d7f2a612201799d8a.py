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
from config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FP32_NEG_INF,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from config import FLASH as M
from decode_indexer_compressor import indexer_compressor

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
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
OFFSET = M.sliding_window

# kernel-local
COMPRESS_RATIO = 4  # the indexer only runs on ratio-4 layers
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = 64
INNER_STATE_BLOCK_NUM = B * INNER_STATE_MAX_BLOCKS
INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_MAX_BLOCKS = 64
IDX_CACHE_BLOCK_NUM = B * IDX_CACHE_MAX_BLOCKS
SCORE_LEN = IDX_KV_LEN
START_POS = 254

# tiling
CACHE_TILE = 32
assert BLOCK_SIZE % CACHE_TILE == 0, "CACHE_TILE must not cross a paged idx_kv_cache block"
Q_TILE = 128
Q_OUT_TILE = 256
QR_PROJ_ROW_TILE = 16
HEAD_DIM_TILE = 32
D_TILE = 32
WEIGHTS_ROW_TILE = 32
B_TILE = 4
TOPK_TILE = 16


@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32],
    topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
):
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(
        IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE,
        name_hint="qr_proj",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        o0 = idx * Q_OUT_TILE
        qr_acc = pl.create_tensor([T, Q_OUT_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
            q0 = kb * Q_TILE
            qr_tile = qr[:, q0 : q0 + Q_TILE]
            wq_tile = wq_b[q0 : q0 + Q_TILE, o0 : o0 + Q_OUT_TILE]
            if q0 == 0:
                qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
            else:
                qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
        wq_scale = pl.reshape(wq_b_scale[o0 : o0 + Q_OUT_TILE], [1, Q_OUT_TILE])
        for r0 in pl.range(0, T, QR_PROJ_ROW_TILE):
            acc_fp32 = pl.cast(qr_acc[r0 : r0 + QR_PROJ_ROW_TILE, :], target_type=pl.FP32, mode="none")
            scale_dq = qr_scale[r0 : r0 + QR_PROJ_ROW_TILE, :]
            qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, scale_dq), wq_scale)
            qr_proj[r0 : r0 + QR_PROJ_ROW_TILE, o0 : o0 + Q_OUT_TILE] = qr_dequant

    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_rope_out = pl.create_tensor([T * IDX_N_HEADS, ROPE_HEAD_DIM], dtype=pl.BF16)
    for idx in pl.spmd(T * IDX_N_HEADS // 32, name_hint="qr_rope"):
        o0 = idx * 32
        token_idx = o0 // IDX_N_HEADS
        batch_idx = token_idx // S
        cos_b = cos[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        qr_rope_slice = qr_proj_flat[o0 : o0 + 32, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
        even_tile = pl.gather(qr_rope_slice, mask_pattern=pl.tile.MaskPattern.P0101)
        odd_tile = pl.gather(qr_rope_slice, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_even = pl.sub(pl.col_expand_mul(even_tile, cos_b), pl.col_expand_mul(odd_tile, sin_b))
        rope_odd = pl.add(pl.col_expand_mul(even_tile, sin_b), pl.col_expand_mul(odd_tile, cos_b))
        rope_buf = pl.full([32, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_buf = pl.tensor.scatter(rope_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_buf)
        rope_buf = pl.tensor.scatter(rope_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_buf)
        qr_rope_out[o0 : o0 + 32, :] = pl.cast(rope_buf, target_type=pl.BF16, mode="rint")

    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for idx in pl.spmd(
        T * IDX_N_HEADS // 512, name_hint="qr_hadamard_quant", optimizations=[pl.split(pl.SplitMode.UP_DOWN)]
    ):
        o0 = idx * 512
        for ro in pl.range(0, 512, 128):
            qh_nope = pl.cast(
                qr_proj_flat[o0 + ro : o0 + ro + 128, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint"
            )
            qh_rope = qr_rope_out[o0 + ro : o0 + ro + 128, :]
            qh_acc = pl.matmul(qh_nope, hadamard[0:IDX_NOPE_HEAD_DIM, :], out_dtype=pl.FP32)
            qr_hadamard_acc = pl.matmul_acc(qh_acc, qh_rope, hadamard[IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM, :])
            qh_amax = pl.full([1, 128], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
                qh_a_f32 = qr_hadamard_acc[0:128, h0 : h0 + HEAD_DIM_TILE]
                qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
                qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, 128])
                qh_amax = pl.maximum(qh_amax, qh_a_max)
            qh_scale_quant_row = pl.div(pl.full([1, 128], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
            qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [128, 1])
            qr_hadamard_scale_dq[o0 + ro : o0 + ro + 128, :] = qh_scale_dq
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [128, 1])
            for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
                qh_q_f32 = qr_hadamard_acc[0:128, h1 : h1 + HEAD_DIM_TILE]
                qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
                qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
                qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
                qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
                qr_hadamard_i8[o0 + ro : o0 + ro + 128, h1 : h1 + HEAD_DIM_TILE] = qh_i8

    x_flat = pl.reshape(x, [T, D])
    weights = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    for idx in pl.spmd(T // WEIGHTS_ROW_TILE, name_hint="weights_proj"):
        wrow0 = idx * WEIGHTS_ROW_TILE
        weights_acc = pl.create_tensor([WEIGHTS_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.pipeline(0, D // D_TILE, stage=2):
            d0 = db * D_TILE
            x_tile = x_flat[wrow0 : wrow0 + WEIGHTS_ROW_TILE, d0 : d0 + D_TILE]
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            if d0 == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
        weights[wrow0 : wrow0 + WEIGHTS_ROW_TILE, :] = pl.mul(weights_acc, WEIGHTS_SCALE)

    inner_kv, inner_compress_state, idx_kv_cache = indexer_compressor(
        x,
        inner_kv,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        cos,
        sin,
        hadamard,
        idx_kv_cache,
        idx_block_table,
        start_pos,
    )

    kv_cache_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM])
    idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])
    score_flat = pl.reshape(score, [T, SCORE_LEN])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_init"):
        for si0 in pl.range(0, T, S):
            score_flat[si0 : si0 + S, :] = pl.full([S, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF)

    for bg in pl.spmd(B // B_TILE, name_hint="score"):
        for bi in pl.range(B_TILE):
            b = bg * B_TILE + bi
            sp_b = pl.read(start_pos, [b])
            clen_b = (sp_b + S) // COMPRESS_RATIO
            cblk_b = (clen_b + CACHE_TILE - 1) // CACHE_TILE
            tb = b * S
            qb = b * S * IDX_N_HEADS
            for cb in pl.range(cblk_b):
                cache0 = cb * CACHE_TILE
                valid_len = pl.min(CACHE_TILE, clen_b - cache0)
                idx_blk_off = cache0 // BLOCK_SIZE
                idx_intra = cache0 % BLOCK_SIZE
                idx_blk_id = pl.cast(
                    pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + idx_blk_off]),
                    pl.INDEX,
                )
                kv0 = idx_blk_id * BLOCK_SIZE + idx_intra
                # --- KV INT8 quant scale (full-128-dim amax), kept in UB ---
                kv_amax = pl.full([1, CACHE_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
                    kv_a_tile = kv_cache_flat[kv0 : kv0 + CACHE_TILE, h0 : h0 + HEAD_DIM_TILE]
                    kv_a_f32 = pl.cast(kv_a_tile, target_type=pl.FP32)
                    kv_a_abs = pl.maximum(kv_a_f32, pl.neg(kv_a_f32))
                    kv_a_max = pl.reshape(pl.row_max(kv_a_abs), [1, CACHE_TILE])
                    kv_amax = pl.maximum(kv_amax, kv_a_max)
                kv_scale_quant_row = pl.div(
                    pl.full([1, CACHE_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax
                )
                kv_cache_scale_dq = pl.reshape(pl.recip(kv_scale_quant_row), [CACHE_TILE, 1])
                kv_scale_quant = pl.reshape(kv_scale_quant_row, [CACHE_TILE, 1])
                kv_q_full_f32 = pl.cast(
                    kv_cache_flat[kv0 : kv0 + CACHE_TILE, 0:IDX_HEAD_DIM], target_type=pl.FP32
                )
                kv_q_full_scaled = pl.row_expand_mul(kv_q_full_f32, kv_scale_quant)
                kv_q_full_i32 = pl.cast(kv_q_full_scaled, target_type=pl.INT32, mode="rint")
                kv_q_full_half = pl.cast(kv_q_full_i32, target_type=pl.FP16, mode="round")
                kv_q_i8_full = pl.cast(kv_q_full_half, target_type=pl.INT8, mode="trunc")
                for s in pl.range(S):
                    qr_full = qr_hadamard_i8[
                        qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS, 0:IDX_HEAD_DIM
                    ]
                    score_acc_s = pl.matmul(kv_q_i8_full, qr_full, out_dtype=pl.INT32, b_trans=True)
                    qh_scale_s = pl.reshape(
                        qr_hadamard_scale_dq[qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS, :],
                        [1, IDX_N_HEADS],
                    )
                    score_tile_s = pl.cast(score_acc_s, target_type=pl.FP32, mode="none")
                    score_tile_s = pl.col_expand_mul(
                        pl.row_expand_mul(score_tile_s, kv_cache_scale_dq), qh_scale_s
                    )
                    relu_score_s = pl.maximum(score_tile_s, pl.mul(score_tile_s, 0.0))
                    weights_row_s = pl.reshape(weights[tb + s : tb + s + 1, :], [1, IDX_N_HEADS])
                    weighted_score_s_t = pl.col_expand_mul(relu_score_s, weights_row_s)
                    weighted_score_s = pl.reshape(pl.row_sum(weighted_score_s_t), [1, CACHE_TILE])
                    weighted_score_valid_s = pl.fillpad(
                        pl.set_validshape(weighted_score_s, 1, valid_len), pad_value=pl.PadValue.min
                    )
                    weighted_score_valid_s = pl.maximum(
                        weighted_score_valid_s,
                        pl.full([1, CACHE_TILE], dtype=pl.FP32, value=FP32_NEG_INF),
                    )
                    score_flat[tb + s : tb + s + 1, cache0 : cache0 + CACHE_TILE] = weighted_score_valid_s

    topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])
    for idx in pl.spmd(T // TOPK_TILE, name_hint="topk"):
        t0 = idx * TOPK_TILE
        for ti in pl.range(0, TOPK_TILE):
            t = t0 + ti
            invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
            topk_idxs_flat[t : t + 1, :] = invalid_idxs
            batch_idx = t // S
            start_pos_b = pl.read(start_pos, [batch_idx])
            cache_len_b = (start_pos_b + S) // COMPRESS_RATIO
            if cache_len_b > 0:
                offset_i32 = pl.cast(offset, target_type=pl.INT32)
                score_row = score_flat[t : t + 1, :]
                idx_init = pl.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
                sorted_score_tile = pl.sort32(score_row, idx_init)
                sorted_score_tile = pl.mrgsort(sorted_score_tile, block_len=64)
                sorted_score_tile = pl.mrgsort(sorted_score_tile, block_len=256)
                sorted_score_tile = pl.mrgsort(sorted_score_tile, block_len=1024)
                topk_pairs = sorted_score_tile[:, 0 : 2 * IDX_TOPK]
                topk_idxs_tile = pl.gather(
                    topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                )
                valid_topk = pl.min(IDX_TOPK, cache_len_b)
                topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
                topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)

    return score, idx_kv_cache, topk_idxs


@pl.jit
def indexer_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.INT32]],
    start_pos: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
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
        hadamard,
        inner_kv,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_block_table,
        score,
        topk_idxs,
        start_pos,
        offset,
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
    from decode_indexer_compressor import golden_compressor

    x = tensors["x"].float()
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    weights_proj = tensors["weights_proj"].float()
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()

    start_pos_t = tensors["start_pos"]
    offset = int(tensors["offset"])

    bsz, seqlen, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)

    x_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v = cos.view(B, 1, 1, -1)
    sin_v = sin.view(B, 1, 1, -1)
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
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cos": tensors["cos"],
        "sin": tensors["sin"],
        "hadamard": tensors["hadamard"],
        "compress_state": tensors["inner_compress_state"],
        "compress_state_block_table": tensors["inner_compress_state_block_table"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_block_table": tensors["idx_block_table"],
        "start_pos": tensors["start_pos"],
    }
    golden_compressor(inner_tensors)

    weights = (x @ weights_proj) * WEIGHTS_SCALE

    idx_kv_cache = tensors["idx_kv_cache"].float()
    idx_block_table = tensors["idx_block_table"]
    score_full = torch.full((bsz, seqlen, SCORE_LEN), FP32_NEG_INF, dtype=torch.float32)
    topk_idxs = torch.full((bsz, seqlen, SCORE_LEN), -1, dtype=torch.int32)
    q_i8, q_scale = _int8_quant_per_row(q.reshape(B * S * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(B, S, IDX_N_HEADS, 1)

    for b in range(bsz):
        start_pos = int(start_pos_t[b].item())
        cache_len = (start_pos + seqlen) // ratio
        if cache_len <= 0:
            continue

        kv_rows = []
        for slot in range(cache_len):
            blk_id = int(idx_block_table[b, slot // BLOCK_SIZE].item())
            kv_rows.append(idx_kv_cache[blk_id, slot % BLOCK_SIZE, 0])
        kv_view = torch.stack(kv_rows, dim=0).unsqueeze(0)
        kv_i8, kv_scale = _int8_quant_per_row(kv_view.reshape(cache_len, IDX_HEAD_DIM))
        kv_i8 = kv_i8.view(cache_len, IDX_HEAD_DIM)
        kv_scale = kv_scale.view(cache_len, 1)
        score_i32 = torch.einsum("shd,td->sht", q_i8[b].to(torch.int32), kv_i8.to(torch.int32))
        score = score_i32.float() * q_scale[b] * kv_scale.view(1, 1, cache_len)
        score = (torch.relu(score) * weights[b].unsqueeze(-1)).sum(dim=1)
        score_full[b, :, :cache_len] = score.to(torch.float32)

        k = min(IDX_TOPK, cache_len)
        _, idx = score.topk(k, dim=-1)
        topk_idxs[b, :, :k] = idx.to(torch.int32)
        topk_idxs[b, :, :k] += offset

    tensors["score"][:] = score_full

    tensors["topk_idxs"][:] = topk_idxs.view(B, S, SCORE_LEN)


def build_tensor_specs(start_pos: int = START_POS, hetero_start_pos: bool = False):
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
        return torch.rand(B, ROPE_HEAD_DIM // 2)

    def init_sin():
        return torch.rand(B, ROPE_HEAD_DIM // 2)

    def init_hadamard():
        return torch.rand(IDX_HEAD_DIM, IDX_HEAD_DIM) * (IDX_HEAD_DIM**-0.5)

    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM)
        state[:, :, INNER_OUT_DIM:] = FP32_NEG_INF
        return state

    def init_inner_compress_state_block_table():
        tbl = torch.full((B, INNER_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(INNER_STATE_MAX_BLOCKS):
                tbl[b, j] = b * INNER_STATE_MAX_BLOCKS + j
        return tbl

    def init_inner_wkv():
        return torch.rand(D, INNER_OUT_DIM)

    def init_inner_wgate():
        return torch.rand(D, INNER_OUT_DIM)

    def init_inner_ape():
        return torch.rand(COMPRESS_RATIO, INNER_OUT_DIM)

    def init_inner_norm_w():
        return torch.ones(INNER_HEAD_DIM)

    def init_idx_kv_cache():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM)

    def init_idx_block_table():
        tbl = torch.full((B, IDX_CACHE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(IDX_CACHE_MAX_BLOCKS):
                tbl[b, j] = b * IDX_CACHE_MAX_BLOCKS + j
        return tbl

    def init_start_pos():
        vals = torch.full((B,), start_pos, dtype=torch.int32)
        if hetero_start_pos:
            pattern = torch.tensor(
                [COMPRESS_RATIO * 2 - 1, COMPRESS_RATIO - 1, COMPRESS_RATIO - S, 0], dtype=torch.int32
            )
            for b in range(B):
                vals[b] = pattern[b % int(pattern.numel())]
        return vals

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
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_kv", [B, S, INNER_HEAD_DIM], torch.float32),
        TensorSpec(
            "inner_compress_state",
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            torch.float32,
            init_value=init_inner_compress_state,
        ),
        TensorSpec(
            "inner_compress_state_block_table",
            [B, INNER_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_inner_compress_state_block_table,
        ),
        TensorSpec("inner_wkv", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.float32, init_value=init_inner_norm_w),
        TensorSpec(
            "idx_kv_cache",
            [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM],
            torch.bfloat16,
            init_value=init_idx_kv_cache,
            is_output=True,
        ),
        TensorSpec(
            "idx_block_table", [B, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table
        ),
        # Outputs are fixed to SCORE_LEN; positions past cache_len are -inf for score and -1 for topk_idxs.
        TensorSpec("score", [B, S, SCORE_LEN], torch.float32, is_output=True),
        TensorSpec("topk_idxs", [B, S, SCORE_LEN], torch.int32, is_output=True),
        TensorSpec("start_pos", [B], torch.int32, init_value=init_start_pos),
        ScalarSpec("offset", torch.int32, OFFSET),
    ]


if __name__ == "__main__":
    import argparse

    import torch
    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--start-pos", type=int, default=START_POS)
    parser.add_argument("--hetero-start-pos", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # topk_pair_compare expects a tensor whose [..., i] entry is the score paired
    # with idx[..., i] (sorted along the top-k axis). Here `score` is per-key
    # (input-space) so it isn't pre-sorted; recover the paired scores on the fly
    # by gathering `score[topk_idxs - OFFSET]` over the valid first IDX_TOPK
    # slots, then delegate.
    def topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        score = actual_outputs["score"]
        a_top = actual[..., :IDX_TOPK]
        e_top = expected[..., :IDX_TOPK]
        a_orig = (a_top.long() - OFFSET).clamp(min=0, max=score.shape[-1] - 1)
        paired = torch.gather(score, dim=-1, index=a_orig)
        synth_actual = {**actual_outputs, "_topk_paired_scores": paired}
        return topk_pair_compare("_topk_paired_scores")(
            a_top,
            e_top,
            actual_outputs=synth_actual,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )

    topk_idxs_compare.__name__ = "topk_pair_compare"

    result = run_jit(
        fn=indexer_test,
        specs=build_tensor_specs(args.start_pos, args.hetero_start_pos),
        golden_fn=golden_indexer,
        runtime_dir=args.runtime_dir,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "score": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "topk_idxs": topk_idxs_compare,
            "idx_kv_cache": ratio_allclose(
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / (IDX_CACHE_BLOCK_NUM * BLOCK_SIZE)
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
