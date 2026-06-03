# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=4 overlap).

Uses overlapping state layout with 8 slots.
Front slots 0-3 at columns [0:HEAD_DIM], back slots 4-7 at columns [HEAD_DIM:OUT_DIM].
Tree reduction for softmax+pool. State shift after compression."""

import pypto.language as pl
from config import DECODE_BATCH, DECODE_SEQ
from config import FLASH as M

# model config
B = DECODE_BATCH
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.index_head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.index_nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-4 overlapping compressor)
COMPRESS_RATIO = 4
ROTATE = True
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
START_POS = COMPRESS_RATIO - 1  # ScalarSpec default exercises S-token window-boundary scatter

# tiling
ROPE_CHUCK = 32
K_CHUNK = 128
OUT_CHUNK = 64
NEXT_OUT_CHUNK = 32
HEAD_CHUNK = 32
HEAD_DIM_CHUCK = 128
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    ape_row = pl.cast(start_pos % COMPRESS_RATIO, target_type=pl.INDEX)
    pre_tokens = COMPRESS_RATIO - (start_pos % COMPRESS_RATIO)
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])

    scatter_stop = S
    if pre_tokens < S:
        scatter_stop = pre_tokens
    for o0 in pl.parallel(0, OUT_DIM, OUT_CHUNK):
        for s in pl.parallel(scatter_stop):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_score_proj"):
                x_tile = pl.reshape(x[:, s : s + 1, 0:K_CHUNK], [B, K_CHUNK])
                wkv_tile = wkv[0:K_CHUNK, o0 : o0 + OUT_CHUNK]
                wgate_tile = wgate[0:K_CHUNK, o0 : o0 + OUT_CHUNK]
                kv_tile = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_tile = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)

                for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                    x_tile = pl.reshape(x[:, s : s + 1, k0 : k0 + K_CHUNK], [B, K_CHUNK])
                    wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                    wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                    kv_tile = pl.matmul_acc(kv_tile, x_tile, wkv_tile)
                    score_tile = pl.matmul_acc(score_tile, x_tile, wgate_tile)

                token_ape_row = (ape_row + s) % COMPRESS_RATIO
                ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                ape_base = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                slot_col0_s = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, slot_col0_s + o0])
                score_state_flat = pl.assemble(score_state_flat, score_tile, [0, slot_col0_s + o0])

    pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.BF16)
    kv_final = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])

    if (start_pos % COMPRESS_RATIO) + S >= COMPRESS_RATIO:
        for hb in pl.parallel(0, HEAD_BLOCKS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool"):
                h0 = hb * HEAD_CHUNK
                last_col0 = (STATE_LEN - 1) * OUT_DIM + HEAD_DIM + h0
                mi = score_state_flat[:, last_col0 : last_col0 + HEAD_CHUNK]
                li = pl.exp(pl.sub(mi, mi))
                oi = kv_state_flat[:, last_col0 : last_col0 + HEAD_CHUNK]

                for s in pl.range(0, COMPRESS_RATIO):
                    front_col0 = s * OUT_DIM + h0
                    front_score = score_state_flat[:, front_col0 : front_col0 + HEAD_CHUNK]
                    front_kv = kv_state_flat[:, front_col0 : front_col0 + HEAD_CHUNK]
                    mi_next_front = pl.maximum(mi, front_score)
                    alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                    beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                    li = pl.add(pl.mul(alpha_front, li), beta_front)
                    oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                    mi = mi_next_front

                for s in pl.range(COMPRESS_RATIO, STATE_LEN - 1):
                    back_col0 = s * OUT_DIM + HEAD_DIM + h0
                    back_score = score_state_flat[:, back_col0 : back_col0 + HEAD_CHUNK]
                    back_kv = kv_state_flat[:, back_col0 : back_col0 + HEAD_CHUNK]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_kv = pl.assemble(pooled_kv, pooled_chunk, [0, h0])

        for s in pl.range(0, COMPRESS_RATIO, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_shift"):
                src_col0 = (COMPRESS_RATIO + s) * OUT_DIM
                dst_col0 = s * OUT_DIM
                for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                    dep_col0 = o0 % HEAD_DIM
                    dep_tile = pooled_kv[:, dep_col0 : dep_col0 + OUT_CHUNK]
                    dep_zero = pl.sub(dep_tile, dep_tile)
                    kv_tile = kv_state_flat[:, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                    score_tile = score_state_flat[:, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                    kv_tile = pl.add(kv_tile, dep_zero)
                    score_tile = pl.add(score_tile, dep_zero)
                    kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, dst_col0 + o0])
                    score_state_flat = pl.assemble(score_state_flat, score_tile, [0, dst_col0 + o0])

        norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
            partial_sq = pl.full([1, B], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
                # Golden applies rmsnorm to kv.to(torch.bfloat16), then casts to FP32 inside rmsnorm.
                kv_rms_chunk = pl.cast(
                    pl.cast(pooled_kv[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(kv_rms_chunk, kv_rms_chunk)), [1, B]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [B, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
                kv_norm_chunk = pl.cast(
                    pl.cast(pooled_kv[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                gamma = norm_w_2d[:, k0 : k0 + HEAD_CHUNK]
                normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                normed_kv = pl.assemble(
                    normed_kv, pl.cast(normed_chunk, target_type=pl.BF16, mode="rint"), [0, k0]
                )

        kv_rope = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.BF16)
        kv_proj_even = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        kv_proj_odd = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        rope_even = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.BF16)
        rope_odd = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_slice"):
            kv_rope = pl.assemble(kv_rope, normed_kv[:, NOPE_HEAD_DIM:HEAD_DIM], [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_slice"):
            kv_rope_tile = kv_rope[:, 0:ROPE_CHUCK]
            even_select_tile = even_select[0:ROPE_CHUCK, :]
            odd_select_tile = odd_select[0:ROPE_CHUCK, :]
            even_acc = pl.matmul(kv_rope_tile, even_select_tile, out_dtype=pl.FP32)
            odd_acc = pl.matmul(kv_rope_tile, odd_select_tile, out_dtype=pl.FP32)

            for r0 in pl.range(ROPE_CHUCK, ROPE_HEAD_DIM, ROPE_CHUCK):
                kv_rope_tile = kv_rope[:, r0 : r0 + ROPE_CHUCK]
                even_select_tile = even_select[r0 : r0 + ROPE_CHUCK, :]
                odd_select_tile = odd_select[r0 : r0 + ROPE_CHUCK, :]
                even_acc = pl.matmul_acc(even_acc, kv_rope_tile, even_select_tile)
                odd_acc = pl.matmul_acc(odd_acc, kv_rope_tile, odd_select_tile)
            kv_proj_even = pl.assemble(kv_proj_even, even_acc, [0, 0])
            kv_proj_odd = pl.assemble(kv_proj_odd, odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_apply"):
            even_tile = kv_proj_even[:, :]
            odd_tile = kv_proj_odd[:, :]
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
            rope_even = pl.assemble(rope_even, rope_even_acc, [0, 0])
            rope_odd = pl.assemble(rope_odd, rope_odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_assemble"):
            rope_even_tile = rope_even[:, 0:ROPE_CHUCK]
            rope_odd_tile = rope_odd[:, 0:ROPE_CHUCK]
            even_select_tile_t = even_select[:, 0:ROPE_CHUCK]
            odd_select_tile_t = odd_select[:, 0:ROPE_CHUCK]
            rope_acc = pl.matmul(rope_even_tile, even_select_tile_t, out_dtype=pl.FP32, b_trans=True)
            rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

            for r0 in pl.range(ROPE_CHUCK, ROPE_HEAD_DIM // 2, ROPE_CHUCK):
                rope_even_tile = rope_even[:, r0 : r0 + ROPE_CHUCK]
                rope_odd_tile = rope_odd[:, r0 : r0 + ROPE_CHUCK]
                even_select_tile_t = even_select[:, r0 : r0 + ROPE_CHUCK]
                odd_select_tile_t = odd_select[:, r0 : r0 + ROPE_CHUCK]
                rope_acc = pl.matmul_acc(rope_acc, rope_even_tile, even_select_tile_t, b_trans=True)
                rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_write"):
            normed_kv = pl.assemble(
                normed_kv, pl.cast(rope_acc, target_type=pl.BF16, mode="rint"), [0, NOPE_HEAD_DIM]
            )

        if rotate:
            for o0 in pl.range(0, HEAD_DIM, OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
                    kv_proj_tile = normed_kv[:, 0:HEAD_DIM]
                    hadamard_tile = hadamard[0:HEAD_DIM, o0 : o0 + OUT_CHUNK]
                    kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
                    kv_final = pl.assemble(kv_final, kv_hadamard_acc, [0, o0])
        else:
            for o0 in pl.parallel(0, HEAD_DIM, OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_write"):
                    kv_out_tile = normed_kv[:, o0 : o0 + OUT_CHUNK]
                    kv_final = pl.assemble(kv_final, pl.cast(kv_out_tile, target_type=pl.FP32), [0, o0])

        kv_cache_flat = pl.reshape(kv_cache, [B * IDX_KV_LEN, HEAD_DIM])
        cache_col = start_pos // COMPRESS_RATIO
        for b_idx in pl.parallel(B):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write"):
                kv_row_fp32 = kv_final[b_idx : b_idx + 1, 0:HEAD_DIM]
                kv_flat = pl.assemble(kv_flat, kv_row_fp32, [b_idx * S, 0])
                cache_row = b_idx * IDX_KV_LEN + cache_col
                kv_cache_flat = pl.assemble(
                    kv_cache_flat,
                    pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint"),
                    [cache_row, 0],
                )
        if pre_tokens < S:
            for o0 in pl.parallel(0, OUT_DIM, NEXT_OUT_CHUNK):
                for s in pl.range(pre_tokens, S):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_score_proj_next"):
                        x_tile_next = pl.reshape(x[:, s : s + 1, 0:K_CHUNK], [B, K_CHUNK])
                        wkv_tile_next = wkv[0:K_CHUNK, o0 : o0 + NEXT_OUT_CHUNK]
                        wgate_tile_next = wgate[0:K_CHUNK, o0 : o0 + NEXT_OUT_CHUNK]
                        kv_tile_next = pl.matmul(x_tile_next, wkv_tile_next, out_dtype=pl.FP32)
                        score_tile_next = pl.matmul(x_tile_next, wgate_tile_next, out_dtype=pl.FP32)

                        for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                            x_tile_next = pl.reshape(x[:, s : s + 1, k0 : k0 + K_CHUNK], [B, K_CHUNK])
                            wkv_tile_next = wkv[k0 : k0 + K_CHUNK, o0 : o0 + NEXT_OUT_CHUNK]
                            wgate_tile_next = wgate[k0 : k0 + K_CHUNK, o0 : o0 + NEXT_OUT_CHUNK]
                            kv_tile_next = pl.matmul_acc(kv_tile_next, x_tile_next, wkv_tile_next)
                            score_tile_next = pl.matmul_acc(score_tile_next, x_tile_next, wgate_tile_next)

                        shift_dep_col0 = (COMPRESS_RATIO - 1) * OUT_DIM + o0
                        dep_zero_next = pl.sub(
                            kv_state_flat[:, shift_dep_col0 : shift_dep_col0 + NEXT_OUT_CHUNK],
                            kv_state_flat[:, shift_dep_col0 : shift_dep_col0 + NEXT_OUT_CHUNK],
                        )
                        kv_tile_next = pl.add(kv_tile_next, dep_zero_next)
                        score_tile_next = pl.add(score_tile_next, dep_zero_next)
                        token_ape_row_next = (ape_row + s) % COMPRESS_RATIO
                        ape_tile_next = ape[
                            token_ape_row_next : token_ape_row_next + 1, o0 : o0 + NEXT_OUT_CHUNK
                        ]
                        ape_base_next = pl.full([B, NEXT_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        score_tile_next = pl.add(score_tile_next, pl.col_expand(ape_base_next, ape_tile_next))
                        slot_col0_s = (COMPRESS_RATIO + token_ape_row_next) * OUT_DIM
                        kv_state_flat = pl.assemble(kv_state_flat, kv_tile_next, [0, slot_col0_s + o0])
                        score_state_flat = pl.assemble(
                            score_state_flat, score_tile_next, [0, slot_col0_s + o0]
                        )

        if pre_tokens < S:
            for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                for s in pl.range(pre_tokens):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_shift_boundary"):
                        token_ape_row = (ape_row + s) % COMPRESS_RATIO
                        src_col0 = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                        dst_col0 = token_ape_row * OUT_DIM
                        kv_tile = kv_state_flat[:, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                        score_tile = score_state_flat[:, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                        kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, dst_col0 + o0])
                        score_state_flat = pl.assemble(score_state_flat, score_tile, [0, dst_col0 + o0])

        kv_cache = pl.reshape(kv_cache_flat, [B, IDX_KV_LEN, HEAD_DIM])

    kv_state = pl.reshape(kv_state_flat, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [B, STATE_LEN, OUT_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv, kv_state, score_state, kv_cache


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    kv_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    score_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    kv, kv_state, score_state, kv_cache = indexer_compressor(
        x,
        kv,
        kv_state,
        score_state,
        wkv,
        wgate,
        ape,
        norm_w,
        cos,
        sin,
        even_select,
        odd_select,
        hadamard,
        kv_cache,
        start_pos,
        rotate,
    )
    return kv, kv_state, score_state, kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

    x = tensors["x"].float()
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()
    kv_cache = tensors["kv_cache"]
    start_pos = int(tensors["start_pos"])
    rotate = bool(tensors["rotate"])
    bsz, _, _ = x.shape
    ratio, d, rd = COMPRESS_RATIO, hadamard.shape[0], tensors["even_select"].shape[0]

    kv = x @ wkv  # [B, S, OUT_DIM]
    score = x @ wgate  # [B, S, OUT_DIM]
    kv_proj = kv

    pre_tokens = min(S, ratio - (start_pos % ratio))
    should_compress = pre_tokens < S or (start_pos + S) % ratio == 0

    # Per-token ape add + scatter to wrapped overlap slots.
    ape_row_g = start_pos % ratio
    for s in range(pre_tokens):
        token_ape_row = (ape_row_g + s) % ratio
        score[:, s, :] = score[:, s, :] + ape[token_ape_row]
        kv_state[:bsz, ratio + token_ape_row] = kv[:, s, :]
        score_state[:bsz, ratio + token_ape_row] = score[:, s, :]

    if should_compress:
        kvs = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)
        scs = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1)
        kv = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)
        kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
        score_state[:bsz, :ratio] = score_state[:bsz, ratio:]

    if pre_tokens < S:
        for s in range(pre_tokens, S):
            token_ape_row = (ape_row_g + s) % ratio
            score[:, s, :] = score[:, s, :] + ape[token_ape_row]
            kv_state[:bsz, ratio + token_ape_row] = kv_proj[:, s, :]
            score_state[:bsz, ratio + token_ape_row] = score[:, s, :]

    tensors["kv_state"][:] = kv_state
    tensors["score_state"][:] = score_state

    if not should_compress:
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return (w * x).to(torch.bfloat16)

    kv = rmsnorm(kv.to(torch.bfloat16), norm_w)

    x_pair = kv[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
    y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)

    kv = torch.cat([kv[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1).float()

    if rotate:
        kv = kv @ hadamard
    # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0.
    tensors["kv"][:, 0:1, :] = kv

    kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)

    tensors["kv_cache"][:] = kv_cache


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.rand(B, S, D)

    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)

    def init_score_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)

    def init_wkv():
        return torch.rand(D, OUT_DIM)

    def init_wgate():
        return torch.rand(D, OUT_DIM)

    def init_ape():
        return torch.rand(COMPRESS_RATIO, OUT_DIM)

    def init_norm_w():
        return torch.ones(HEAD_DIM)

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
        return torch.rand(HEAD_DIM, HEAD_DIM) * (HEAD_DIM**-0.5)

    def init_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, HEAD_DIM)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec(
            "kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True
        ),
        TensorSpec(
            "score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_score_state, is_output=True
        ),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [1, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [1, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec(
            "even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select
        ),
        TensorSpec(
            "odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select
        ),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec(
            "kv_cache", [B, IDX_KV_LEN, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True
        ),
        ScalarSpec("start_pos", torch.int32, START_POS),
        ScalarSpec("rotate", torch.bool, ROTATE),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(),
        golden_fn=golden_compressor,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "kv_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "score_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / IDX_KV_LEN),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
