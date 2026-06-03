# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa
"""DeepSeek-V4 Indexer KV Compressor (decode incremental, ratio=4 overlap)."""

import pypto.language as pl
from config import BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, DECODE_BATCH, DECODE_SEQ, FP32_NEG_INF
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
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_MAX_BLOCKS = 64
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
IDX_CACHE_MAX_BLOCKS = 64
IDX_CACHE_BLOCK_NUM = B * IDX_CACHE_MAX_BLOCKS
START_POS = COMPRESS_RATIO - 1  # default start_pos exercises S-token window-boundary scatter

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
B_TILE = 64
HEAD_TILE = 64
HEAD_DIM_TILE = 128
RMS_TILE = 16


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[
        [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(
        compress_state, [COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM]
    )
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    for idx in pl.spmd(B * S * OUT_DIM // (B_TILE * OUT_TILE), name_hint="kv_score_proj"):
        global_row0 = (idx // (OUT_DIM // OUT_TILE)) * B_TILE
        o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_tile = x_flat[global_row0 : global_row0 + B_TILE, k0 : k0 + K_TILE]
            wkv_tile = wkv[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            wgate_tile = wgate[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

        kv_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = score_acc

    kv_proj_by_batch = pl.reshape(kv_proj_scratch, [B, S * OUT_DIM])
    score_proj_by_batch = pl.reshape(score_proj_scratch, [B, S * OUT_DIM])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_paged"):
        for c_idx in pl.range(B):
            start_pos_b = pl.read(start_pos, [c_idx])
            ape_row_b = pl.cast(start_pos_b % COMPRESS_RATIO, target_type=pl.INDEX)

            for s in pl.pipeline(S, stage=2):
                abs_pos_s = start_pos_b + s
                state_blk_off = abs_pos_s // COMPRESS_STATE_BLOCK_SIZE
                state_intra = abs_pos_s % COMPRESS_STATE_BLOCK_SIZE
                state_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, state_blk_off]), pl.INDEX)
                state_row = state_blk_id * COMPRESS_STATE_BLOCK_SIZE + state_intra
                proj_col0 = s * OUT_DIM
                token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                kv_tile = kv_proj_by_batch[c_idx : c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                score_tile = score_proj_by_batch[c_idx : c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                ape_tile = ape[token_ape_row : token_ape_row + 1, 0:OUT_DIM]
                score_tile = pl.add(score_tile, ape_tile)
                compress_state_flat[state_row : state_row + 1, 0:OUT_DIM] = kv_tile
                compress_state_flat[state_row : state_row + 1, OUT_DIM : 2 * OUT_DIM] = score_tile

    pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    for c_idx in pl.spmd(B, name_hint="softmax_pool"):
        start_pos_b = pl.read(start_pos, [c_idx])
        pos_b = start_pos_b % COMPRESS_RATIO
        pre_tokens_b = COMPRESS_RATIO - pos_b
        boundary_end_b = start_pos_b + pre_tokens_b - 1
        cur_window_start_b = boundary_end_b - COMPRESS_RATIO + 1
        prev_window_start_b = cur_window_start_b - COMPRESS_RATIO

        if pos_b + S >= COMPRESS_RATIO:
            for hb in pl.range(HEAD_DIM // HEAD_TILE):
                h0 = hb * HEAD_TILE
                last_abs = cur_window_start_b + COMPRESS_RATIO - 1
                last_blk_off = last_abs // COMPRESS_STATE_BLOCK_SIZE
                last_intra = last_abs % COMPRESS_STATE_BLOCK_SIZE
                last_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, last_blk_off]), pl.INDEX)
                last_row = last_blk_id * COMPRESS_STATE_BLOCK_SIZE + last_intra
                last_col0 = OUT_DIM + HEAD_DIM + h0
                mi = compress_state_flat[last_row : last_row + 1, last_col0 : last_col0 + HEAD_TILE]
                li = pl.exp(pl.sub(mi, mi))
                oi = compress_state_flat[last_row : last_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]

                for s in pl.range(0, COMPRESS_RATIO):
                    prev_abs = prev_window_start_b + s
                    if prev_abs >= 0:
                        prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                        prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                        prev_blk_id = pl.cast(
                            pl.read(compress_state_block_table, [c_idx, prev_blk_off]), pl.INDEX
                        )
                        prev_row = prev_blk_id * COMPRESS_STATE_BLOCK_SIZE + prev_intra
                        front_score = compress_state_flat[
                            prev_row : prev_row + 1, OUT_DIM + h0 : OUT_DIM + h0 + HEAD_TILE
                        ]
                        front_kv = compress_state_flat[prev_row : prev_row + 1, h0 : h0 + HEAD_TILE]
                        mi_next_front = pl.maximum(mi, front_score)
                        alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                        beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                        li = pl.add(pl.mul(alpha_front, li), beta_front)
                        oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                        mi = mi_next_front

                for s in pl.range(0, COMPRESS_RATIO - 1):
                    cur_abs = cur_window_start_b + s
                    cur_blk_off = cur_abs // COMPRESS_STATE_BLOCK_SIZE
                    cur_intra = cur_abs % COMPRESS_STATE_BLOCK_SIZE
                    cur_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, cur_blk_off]), pl.INDEX)
                    cur_row = cur_blk_id * COMPRESS_STATE_BLOCK_SIZE + cur_intra
                    back_col0 = OUT_DIM + HEAD_DIM + h0
                    back_score = compress_state_flat[cur_row : cur_row + 1, back_col0 : back_col0 + HEAD_TILE]
                    back_kv = compress_state_flat[
                        cur_row : cur_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE
                    ]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_kv[c_idx : c_idx + 1, h0 : h0 + HEAD_TILE] = pooled_chunk

    normed_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="rmsnorm_rope"):
        batch_base = batch_base_idx * RMS_TILE
        cos_b = cos[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        partial_sq = pl.full([1, RMS_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE]
            gamma = norm_w_2d[:, k0 : k0 + HEAD_TILE]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE] = pl.cast(
                normed_chunk, target_type=pl.BF16, mode="rint"
            )

        kv_rope_norm = pooled_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        even_tile = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P0101)
        odd_tile = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_even = pl.sub(pl.mul(even_tile, cos_b), pl.mul(odd_tile, sin_b))
        rope_odd = pl.add(pl.mul(even_tile, sin_b), pl.mul(odd_tile, cos_b))
        rope_buf = pl.full([RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_buf = pl.tensor.scatter(rope_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_buf)
        rope_buf = pl.tensor.scatter(rope_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_buf)
        normed_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM] = pl.cast(
            rope_buf, target_type=pl.BF16, mode="rint"
        )

    kv_final = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="kv_hadamard"):
        batch_base = batch_base_idx * RMS_TILE
        kv_proj_tile = normed_kv[batch_base : batch_base + RMS_TILE, 0:HEAD_DIM]
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0:HEAD_DIM, o0 : o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[batch_base : batch_base + RMS_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc

    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="kv_and_cache_write"):
        batch_base = batch_base_idx * RMS_TILE
        for inner in pl.range(RMS_TILE):
            c_idx = batch_base + inner
            start_pos_b = pl.read(start_pos, [c_idx])
            pos_b = start_pos_b % COMPRESS_RATIO
            cache_col = start_pos_b // COMPRESS_RATIO
            if pos_b + S >= COMPRESS_RATIO:
                kv_row_fp32 = kv_final[c_idx : c_idx + 1, 0:HEAD_DIM]
                kv_flat[c_idx * S : c_idx * S + 1, :] = kv_row_fp32
                idx_blk_off = cache_col // BLOCK_SIZE
                idx_intra = cache_col % BLOCK_SIZE
                idx_blk_id = pl.cast(pl.read(idx_block_table, [c_idx, idx_blk_off]), pl.INDEX)
                cache_row = idx_blk_id * BLOCK_SIZE + idx_intra
                idx_kv_cache_flat[cache_row : cache_row + 1, :] = pl.cast(
                    kv_row_fp32, target_type=pl.BF16, mode="rint"
                )

    compress_state = pl.reshape(
        compress_state_flat, [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM]
    )
    idx_kv_cache = pl.reshape(idx_kv_cache_flat, [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv, compress_state, idx_kv_cache


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    compress_state: pl.Out[
        pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    kv, compress_state, idx_kv_cache = indexer_compressor(
        x,
        kv,
        compress_state,
        compress_state_block_table,
        wkv,
        wgate,
        ape,
        norm_w,
        cos,
        sin,
        hadamard,
        idx_kv_cache,
        idx_block_table,
        start_pos,
    )
    return kv, compress_state, idx_kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

    x = tensors["x"].float()
    compress_state = tensors["compress_state"]
    compress_state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()
    idx_kv_cache = tensors["idx_kv_cache"]
    idx_block_table = tensors["idx_block_table"]
    start_pos_t = tensors["start_pos"]
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv  # [B, S, OUT_DIM]
    score = x @ wgate  # [B, S, OUT_DIM]

    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        start_pos = int(start_pos_t[b].item())
        pre_tokens = min(S, ratio - (start_pos % ratio))
        should_compress = pre_tokens < S or (start_pos + S) % ratio == 0
        boundary_end = start_pos + pre_tokens - 1
        cur_window_start = boundary_end - ratio + 1
        prev_window_start = cur_window_start - ratio

        # Per-token ape add + state scatter through the compressor-state block table.
        ape_row_g = start_pos % ratio
        for s in range(S):
            token_ape_row = (ape_row_g + s) % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            abs_pos = start_pos + s
            blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
            intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
            compress_state[blk_id, intra, :OUT_DIM] = kv[b, s, :]
            compress_state[blk_id, intra, OUT_DIM:] = score[b, s, :]

        if should_compress:
            should_compress_rows[b] = True
            kv_rows = []
            score_rows = []
            for s in range(ratio):
                abs_pos = prev_window_start + s
                if abs_pos < 0:
                    kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device))
                    score_rows.append(
                        torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device)
                    )
                    continue
                blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
                intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
                kv_rows.append(compress_state[blk_id, intra, :HEAD_DIM])
                score_rows.append(compress_state[blk_id, intra, OUT_DIM : OUT_DIM + HEAD_DIM])
            for s in range(ratio):
                abs_pos = cur_window_start + s
                blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
                intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
                kv_rows.append(compress_state[blk_id, intra, HEAD_DIM:OUT_DIM])
                score_rows.append(compress_state[blk_id, intra, OUT_DIM + HEAD_DIM :])
            kvs = torch.stack(kv_rows, dim=0).unsqueeze(0)
            scs = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)

    tensors["compress_state"][:] = compress_state

    if not bool(should_compress_rows.any()):
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x

    for b in range(bsz):
        if not bool(should_compress_rows[b]):
            continue
        start_pos = int(start_pos_t[b].item())
        kv_b = rmsnorm(pooled[b : b + 1], norm_w)

        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_v, sin_v = cos[b].view(-1), sin[b].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        kv_b = kv_b.to(torch.bfloat16).float() @ hadamard
        # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0.
        tensors["kv"][b : b + 1, 0:1, :] = kv_b

        cache_col = start_pos // ratio
        blk_id = int(idx_block_table[b, cache_col // BLOCK_SIZE].item())
        idx_kv_cache[blk_id, cache_col % BLOCK_SIZE, 0] = kv_b[0, 0]

    tensors["idx_kv_cache"][:] = idx_kv_cache


def build_tensor_specs(start_pos: int = START_POS, hetero_start_pos: bool = False):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_x():
        return torch.rand(B, S, D)

    def init_compress_state():
        state = torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        state[:, :, OUT_DIM:] = FP32_NEG_INF
        return state

    def init_compress_state_block_table():
        tbl = torch.full((B, COMPRESS_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(COMPRESS_STATE_MAX_BLOCKS):
                tbl[b, j] = b * COMPRESS_STATE_MAX_BLOCKS + j
        return tbl

    def init_wkv():
        return torch.rand(D, OUT_DIM)

    def init_wgate():
        return torch.rand(D, OUT_DIM)

    def init_ape():
        return torch.rand(COMPRESS_RATIO, OUT_DIM)

    def init_norm_w():
        return torch.ones(HEAD_DIM)

    def init_cos():
        return torch.rand(B, ROPE_HEAD_DIM // 2)

    def init_sin():
        return torch.rand(B, ROPE_HEAD_DIM // 2)

    def init_hadamard():
        return torch.rand(HEAD_DIM, HEAD_DIM) * (HEAD_DIM**-0.5)

    def init_idx_kv_cache():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)

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
                [0, COMPRESS_RATIO - S, COMPRESS_RATIO - 1, COMPRESS_RATIO * 2 - 1], dtype=torch.int32
            )
            for b in range(B):
                vals[b] = pattern[b % int(pattern.numel())]
        return vals

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec(
            "compress_state",
            [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
            is_output=True,
        ),
        TensorSpec(
            "compress_state_block_table",
            [B, COMPRESS_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec(
            "idx_kv_cache",
            [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_idx_kv_cache,
            is_output=True,
        ),
        TensorSpec(
            "idx_block_table", [B, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table
        ),
        TensorSpec("start_pos", [B], torch.int32, init_value=init_start_pos),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help="Decode start position for no-compression/aligned/crossing coverage.",
    )
    parser.add_argument(
        "--hetero-start-pos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a grouped per-batch start_pos pattern.",
    )
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos, args.hetero_start_pos),
        golden_fn=golden_compressor,
        runtime_dir=args.runtime_dir,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "idx_kv_cache": ratio_allclose(
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / (IDX_CACHE_BLOCK_NUM * BLOCK_SIZE)
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
