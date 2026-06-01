# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Faithful reproduction of issue #1457 problem 2: DeepSeek-V4 sparse_attn runtime failure.

This is the real ``models/deepseek/v4/sparse_attn.py`` kernel from pypto-lib commit
``c225e5a3`` (the commit named in issue #1457), ported verbatim into the ST harness:
the kernel body, the torch golden, and the tensor specs are copied unchanged; only the
``config`` constants are inlined (FLASH preset) and the ``@pl.jit`` entry is replaced by
a ``@pl.program`` wrapper + ``PTOTestCase``.

Per the issue, compile and golden pass but the runtime orchestrator deadlocks
(``orch_error_code=2`` / AICPU ``507018``) and never produces ``attn_out``.  Running this
ST on a2a3 hardware should reproduce that failure (the on-board CI job will go red);
``--codegen-only`` should still pass (the failure is at runtime, not compile).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

# --- inlined from pypto-lib config.py FLASH preset @ c225e5a3 ---
DECODE_BATCH = 64  # B: tokens per decode step
DECODE_SEQ = 2  # S: 2 tokens per step (MTP)
BLOCK_SIZE = 128  # paged-KV page size / weight-quant block size
INT8_SCALE_MAX = 127.0  # per-row INT8 quant: clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4  # amax floor: avoids 127/0 on all-zero rows


class _FlashConfig:
    """Subset of the FLASH DeepSeekV4Config preset referenced by sparse_attn."""

    hidden_size = 4096
    num_attention_heads = 64
    head_dim = 512
    qk_rope_head_dim = 64
    o_lora_rank = 1024
    o_groups = 8
    sliding_window = 128
    max_position_embeddings = 8192
    index_topk = 512

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5


M = _FlashConfig()


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
IDX_TOPK = M.index_topk
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# kernel-local
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 128
ORI_MAX_BLOCKS = 1  # paged-KV pool: ori (sliding-window) blocks per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 64  # paged-KV pool: compressed blocks per batch
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# tiling
GATHER_TOKEN_TILE = 4
ATTN_TOKEN_TILE = 8
ROPE_TOKEN_TILE = 4
ROPE_PACK_TOKEN_TILE = 32
ROPE_PACK_GROUP_TILE = 1
ROPE_PACK_SPMD_BLOCKS = (T // ROPE_PACK_TOKEN_TILE) * O_GROUPS
MATMUL_ROW_PAD = 16
SPARSE_ATTN_TILE = 64
SPARSE_ATTN_BLOCKS = (TOPK + SPARSE_ATTN_TILE - 1) // SPARSE_ATTN_TILE
ROPE_CHUNK = 16
ROPE_INTERLEAVE_CHUNK = 2 * ROPE_CHUNK
A_K_CHUNK = 128
A_N_CHUNK = 128
B_K_CHUNK = 128
B_N_CHUNK = 128 if T >= 128 else 256
QUANT_CHUNK = 32 if T >= 128 else (128 if T >= 64 else 256)
QUANT_TOKEN_TILE = 8


def get_standalone_cmp_valid(compress_ratio: int) -> int:
    """Map demo compress-ratio modes to the valid compressed-cache tail length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return IDX_TOPK
    if compress_ratio == 128:
        return MAX_SEQ_LEN // compress_ratio
    raise ValueError(
        f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}"
    )


@pl.inline
def sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    A_K_BLOCKS = O_GROUP_IN // A_K_CHUNK
    A_N_BLOCKS = O_LORA // A_N_CHUNK
    A_AMAX_BLOCKS = O_GROUPS * A_N_BLOCKS
    B_K_BLOCKS = (O_GROUPS * O_LORA) // B_K_CHUNK
    B_N_BLOCKS = D // B_N_CHUNK

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [B * CMP_MAX_BLOCKS])
    cmp_sparse_indices_flat = pl.reshape(cmp_sparse_indices, [T * TOPK])
    sparse_kv = pl.create_tensor([T * TOPK, HEAD_DIM], dtype=pl.BF16)
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.BF16)
    sparse_exp = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, SPARSE_ATTN_TILE], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, HEAD_DIM], dtype=pl.FP32)
    sparse_mi = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    sparse_li = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    sparse_oi = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.FP32)
    o_proj_even = pl.create_tensor([T * H, HALF_ROPE], dtype=pl.FP32)
    o_proj_odd = pl.create_tensor([T * H, HALF_ROPE], dtype=pl.FP32)
    rope_even_interleave_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    rope_odd_interleave_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)

    # Stage 1: gather the sparse KV rows selected by the sliding-window path and
    # the compressed-cache path into one per-token packed KV list.
    for gather_t0 in pl.parallel(0, T, GATHER_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_gather_kv_topk_tile"):
            for gather_dt in pl.range(GATHER_TOKEN_TILE):
                gather_t = gather_t0 + gather_dt
                gather_b = gather_t // S
                gather_s = gather_t - gather_b * S
                gather_seq_final = pl.read(seqused_kv, [gather_b])
                gather_seq_used = gather_seq_final - S + 1 + gather_s
                gather_window_valid = pl.min(WIN, gather_seq_used)
                gather_cmp_valid = gather_seq_used - gather_window_valid
                gather_cmp_topk_valid = pl.min(IDX_TOPK, gather_cmp_valid)
                gather_sparse_k = gather_window_valid + gather_cmp_topk_valid
                gather_ori_block_base = gather_b * ORI_MAX_BLOCKS
                gather_cmp_block_base = gather_b * CMP_MAX_BLOCKS
                gather_sparse_idx_base = gather_t * TOPK
                gather_sparse_kv_base = gather_t * TOPK

                # The standalone decode contract uses a contiguous full-window
                # prefix, so copy that prefix as one row block and leave the
                # truly sparse compressed tail on the dynamic row-gather path.
                gather_ori_blk = pl.cast(pl.read(ori_block_table_flat, [gather_ori_block_base]), pl.INDEX)
                gather_ori_row = gather_ori_blk * BLOCK_SIZE
                window_rows = pl.slice(
                    ori_kv_flat,
                    [WIN, HEAD_DIM],
                    [gather_ori_row, 0],
                    valid_shape=[gather_window_valid, HEAD_DIM],
                )
                sparse_kv = pl.assemble(sparse_kv, window_rows, [gather_sparse_kv_base, 0])

                # Append compressed-cache hits after the window prefix.
                for gather_cmp_kk in pl.range(gather_cmp_topk_valid):
                    gather_cmp_idx_pos = gather_sparse_idx_base + gather_window_valid + gather_cmp_kk
                    gather_cmp_raw_idx = pl.read(cmp_sparse_indices_flat, [gather_cmp_idx_pos])
                    gather_cmp_slot = gather_cmp_raw_idx - WIN
                    gather_cmp_block_slot = gather_cmp_slot // BLOCK_SIZE
                    gather_cmp_block_pos = gather_cmp_block_base + gather_cmp_block_slot
                    gather_cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [gather_cmp_block_pos]), pl.INDEX)
                    gather_cmp_intra = gather_cmp_slot % BLOCK_SIZE
                    gather_cmp_row = gather_cmp_blk * BLOCK_SIZE + gather_cmp_intra
                    sparse_kv = pl.assemble(
                        sparse_kv,
                        cmp_kv_flat[gather_cmp_row : gather_cmp_row + 1, 0:HEAD_DIM],
                        [gather_sparse_kv_base + gather_window_valid + gather_cmp_kk, 0],
                    )

                # Keep padded rows deterministic for ratio-0/128 sanity modes.
                zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for gather_pad_kk in pl.range(gather_sparse_k, TOPK):
                    sparse_kv = pl.assemble(
                        sparse_kv, zero_kv_row, [gather_sparse_kv_base + gather_pad_kk, 0]
                    )

    for attn_t0 in pl.parallel(0, T, ATTN_TOKEN_TILE):
        for h0 in pl.parallel(0, H, MATMUL_ROW_PAD):
            # Stage 2a: QK + tile-local softmax for every sparse-K tile.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_qk_softmax_tile"):
                for qk_dt in pl.range(ATTN_TOKEN_TILE):
                    qk_t = attn_t0 + qk_dt
                    qk_b = qk_t // S
                    qk_s = qk_t - qk_b * S
                    qk_seq_final = pl.read(seqused_kv, [qk_b])
                    qk_seq_used = qk_seq_final - S + 1 + qk_s
                    qk_window_valid = pl.min(WIN, qk_seq_used)
                    qk_cmp_valid = qk_seq_used - qk_window_valid
                    qk_cmp_topk_valid = pl.min(IDX_TOPK, qk_cmp_valid)
                    qk_sparse_k = qk_window_valid + qk_cmp_topk_valid
                    qk_sparse_kv_base = qk_t * TOPK
                    qk_head_row = qk_t * H + h0
                    qk_q_batch = q_flat[qk_head_row : qk_head_row + MATMUL_ROW_PAD, 0:HEAD_DIM]

                    for qk_sb in pl.range(SPARSE_ATTN_BLOCKS):
                        qk_tile_start = qk_sb * SPARSE_ATTN_TILE
                        if qk_tile_start < qk_sparse_k:
                            qk_tile_valid = pl.min(SPARSE_ATTN_TILE, qk_sparse_k - qk_tile_start)
                            qk_kv_tile = sparse_kv[
                                qk_sparse_kv_base + qk_tile_start : qk_sparse_kv_base
                                + qk_tile_start
                                + SPARSE_ATTN_TILE,
                                0:HEAD_DIM,
                            ]
                            qk_raw_scores = pl.matmul(qk_q_batch, qk_kv_tile, b_trans=True, out_dtype=pl.FP32)
                            qk_scores_valid = pl.slice(
                                pl.mul(qk_raw_scores, SOFTMAX_SCALE),
                                [MATMUL_ROW_PAD, SPARSE_ATTN_TILE],
                                [0, 0],
                                valid_shape=[MATMUL_ROW_PAD, qk_tile_valid],
                            )
                            qk_scores = pl.fillpad(qk_scores_valid, pad_value=pl.PadValue.min)
                            qk_mi = pl.row_max(qk_scores)
                            qk_exp_scores = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                            qk_exp_scores_bf16 = pl.cast(qk_exp_scores, target_type=pl.BF16)
                            qk_li = pl.row_sum(pl.cast(qk_exp_scores_bf16, target_type=pl.FP32))
                            qk_block_row = qk_t * H * SPARSE_ATTN_BLOCKS + qk_sb * H + h0
                            sparse_exp = pl.assemble(sparse_exp, qk_exp_scores_bf16, [qk_block_row, 0])
                            sparse_blk_mi = pl.assemble(sparse_blk_mi, qk_mi, [qk_block_row, 0])
                            sparse_blk_li = pl.assemble(sparse_blk_li, qk_li, [qk_block_row, 0])

            # Stage 2b: PV for each sparse-K tile. Keep the online merge in a
            # separate scope under FLASH so the AIV live set stays bounded.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_pv_tile"):
                for pv_dt in pl.range(ATTN_TOKEN_TILE):
                    pv_t = attn_t0 + pv_dt
                    pv_b = pv_t // S
                    pv_s = pv_t - pv_b * S
                    pv_seq_final = pl.read(seqused_kv, [pv_b])
                    pv_seq_used = pv_seq_final - S + 1 + pv_s
                    pv_window_valid = pl.min(WIN, pv_seq_used)
                    pv_cmp_valid = pv_seq_used - pv_window_valid
                    pv_cmp_topk_valid = pl.min(IDX_TOPK, pv_cmp_valid)
                    pv_sparse_k = pv_window_valid + pv_cmp_topk_valid
                    pv_sparse_kv_base = pv_t * TOPK
                    for pv_sb in pl.range(SPARSE_ATTN_BLOCKS):
                        pv_tile_start = pv_sb * SPARSE_ATTN_TILE
                        if pv_tile_start < pv_sparse_k:
                            pv_block_row = pv_t * H * SPARSE_ATTN_BLOCKS + pv_sb * H + h0
                            pv_exp = sparse_exp[
                                pv_block_row : pv_block_row + MATMUL_ROW_PAD, 0:SPARSE_ATTN_TILE
                            ]
                            pv_kv_tile = sparse_kv[
                                pv_sparse_kv_base + pv_tile_start : pv_sparse_kv_base
                                + pv_tile_start
                                + SPARSE_ATTN_TILE,
                                0:HEAD_DIM,
                            ]
                            pv_oi_tmp = pl.matmul(pv_exp, pv_kv_tile, out_dtype=pl.FP32)
                            sparse_blk_oi = pl.assemble(sparse_blk_oi, pv_oi_tmp, [pv_block_row, 0])

            # Stage 2c: online-softmax merge across sparse-K tiles.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_merge_tile"):
                for merge_dt in pl.range(ATTN_TOKEN_TILE):
                    merge_t = attn_t0 + merge_dt
                    merge_b = merge_t // S
                    merge_s = merge_t - merge_b * S
                    merge_seq_final = pl.read(seqused_kv, [merge_b])
                    merge_seq_used = merge_seq_final - S + 1 + merge_s
                    merge_window_valid = pl.min(WIN, merge_seq_used)
                    merge_cmp_valid = merge_seq_used - merge_window_valid
                    merge_cmp_topk_valid = pl.min(IDX_TOPK, merge_cmp_valid)
                    merge_sparse_k = merge_window_valid + merge_cmp_topk_valid
                    merge_head_row = merge_t * H + h0
                    merge_block_row0 = merge_t * H * SPARSE_ATTN_BLOCKS + h0
                    merge_mi = sparse_blk_mi[merge_block_row0 : merge_block_row0 + MATMUL_ROW_PAD, 0:1]
                    merge_li = sparse_blk_li[merge_block_row0 : merge_block_row0 + MATMUL_ROW_PAD, 0:1]
                    merge_oi = sparse_blk_oi[merge_block_row0 : merge_block_row0 + MATMUL_ROW_PAD, 0:HEAD_DIM]

                    for merge_sb in pl.range(1, SPARSE_ATTN_BLOCKS):
                        merge_tile_start = merge_sb * SPARSE_ATTN_TILE
                        if merge_tile_start < merge_sparse_k:
                            merge_block_row = merge_t * H * SPARSE_ATTN_BLOCKS + merge_sb * H + h0
                            merge_cur_mi = sparse_blk_mi[
                                merge_block_row : merge_block_row + MATMUL_ROW_PAD, 0:1
                            ]
                            merge_cur_li = sparse_blk_li[
                                merge_block_row : merge_block_row + MATMUL_ROW_PAD, 0:1
                            ]
                            merge_cur_oi = sparse_blk_oi[
                                merge_block_row : merge_block_row + MATMUL_ROW_PAD, 0:HEAD_DIM
                            ]
                            merge_mi_new = pl.maximum(merge_mi, merge_cur_mi)
                            merge_alpha = pl.exp(pl.sub(merge_mi, merge_mi_new))
                            merge_beta = pl.exp(pl.sub(merge_cur_mi, merge_mi_new))
                            merge_li = pl.add(pl.mul(merge_alpha, merge_li), pl.mul(merge_beta, merge_cur_li))
                            merge_oi = pl.add(
                                pl.row_expand_mul(merge_oi, merge_alpha),
                                pl.row_expand_mul(merge_cur_oi, merge_beta),
                            )
                            merge_mi = merge_mi_new

                    sparse_mi = pl.assemble(sparse_mi, merge_mi, [merge_head_row, 0])
                    sparse_li = pl.assemble(sparse_li, merge_li, [merge_head_row, 0])
                    sparse_oi = pl.assemble(sparse_oi, merge_oi, [merge_head_row, 0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_norm_tile"):
                for norm_dt in pl.range(ATTN_TOKEN_TILE):
                    norm_t = attn_t0 + norm_dt
                    norm_attn_head_row = norm_t * H + h0
                    norm_oi = sparse_oi[norm_attn_head_row : norm_attn_head_row + MATMUL_ROW_PAD, 0:HEAD_DIM]
                    norm_mi = sparse_mi[norm_attn_head_row : norm_attn_head_row + MATMUL_ROW_PAD, 0:1]
                    norm_li = sparse_li[norm_attn_head_row : norm_attn_head_row + MATMUL_ROW_PAD, 0:1]
                    norm_sink_bias = pl.reshape(attn_sink[h0 : h0 + MATMUL_ROW_PAD], [MATMUL_ROW_PAD, 1])
                    norm_sink_tile = pl.add(pl.sub(norm_mi, norm_mi), norm_sink_bias)
                    norm_denom = pl.add(norm_li, pl.exp(pl.sub(norm_sink_tile, norm_mi)))
                    oi_out = pl.row_expand_div(norm_oi, norm_denom)
                    attn_stage_row = pl.cast(
                        oi_out[0:MATMUL_ROW_PAD, 0:HEAD_DIM],
                        target_type=pl.BF16,
                    )
                    attn_rope_stage = pl.assemble(
                        attn_rope_stage,
                        attn_stage_row[0:MATMUL_ROW_PAD, NOPE_DIM:HEAD_DIM],
                        [norm_attn_head_row, 0],
                    )

                    for norm_head_i in pl.range(MATMUL_ROW_PAD):
                        norm_global_head = h0 + norm_head_i
                        norm_g = norm_global_head // HEADS_PER_GROUP
                        norm_hh = norm_global_head - norm_g * HEADS_PER_GROUP
                        norm_pack_row = norm_g * T + norm_t
                        norm_head_col = norm_hh * HEAD_DIM
                        o_packed = pl.assemble(
                            o_packed,
                            attn_stage_row[norm_head_i : norm_head_i + 1, 0:NOPE_DIM],
                            [norm_pack_row, norm_head_col],
                        )

    # Stage 3: inverse RoPE on the rope slice of the attention output by
    # deinterleaving even/odd lanes, rotating them, then reinterleaving.
    for rope_t0 in pl.parallel(0, T, ROPE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_rope_slice_tile"):
            for rope_dt in pl.range(ROPE_TOKEN_TILE):
                rope_slice_t = rope_t0 + rope_dt
                rope_slice_head_row = rope_slice_t * H

                # Split interleaved rope lanes into even and odd halves.
                for rope_slice_r0 in pl.range(0, HALF_ROPE, ROPE_CHUNK):
                    rope_tile = attn_rope_stage[
                        rope_slice_head_row : rope_slice_head_row + H,
                        2 * rope_slice_r0 : 2 * rope_slice_r0 + ROPE_INTERLEAVE_CHUNK,
                    ]
                    rope_slice_even_chunk = pl.matmul(rope_tile, even_select_local, out_dtype=pl.FP32)
                    rope_slice_odd_chunk = pl.matmul(rope_tile, odd_select_local, out_dtype=pl.FP32)
                    o_proj_even = pl.assemble(
                        o_proj_even, rope_slice_even_chunk, [rope_slice_head_row, rope_slice_r0]
                    )
                    o_proj_odd = pl.assemble(
                        o_proj_odd, rope_slice_odd_chunk, [rope_slice_head_row, rope_slice_r0]
                    )

    # Stage 3: rotate even/odd halves and immediately reinterleave them. Keeping
    # the BF16 rint cast local avoids the new orchestration SSA collision on
    # written-then-read RoPE GM scratch tensors.
    for rope_apply_t0 in pl.parallel(0, T, ROPE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_rope_apply_assemble_tile"):
            for rope_apply_dt in pl.range(ROPE_TOKEN_TILE):
                rope_apply_t = rope_apply_t0 + rope_apply_dt
                rope_apply_head_row = rope_apply_t * H

                # Reinterleave the rotated even/odd halves back to rope lane order.
                for rope_asm_r0 in pl.range(0, HALF_ROPE, ROPE_CHUNK):
                    cos_chunk = pl.cast(
                        freqs_cos[rope_apply_t : rope_apply_t + 1, rope_asm_r0 : rope_asm_r0 + ROPE_CHUNK],
                        target_type=pl.FP32,
                    )
                    sin_chunk = pl.cast(
                        freqs_sin[rope_apply_t : rope_apply_t + 1, rope_asm_r0 : rope_asm_r0 + ROPE_CHUNK],
                        target_type=pl.FP32,
                    )
                    rope_apply_even_chunk = o_proj_even[
                        rope_apply_head_row : rope_apply_head_row + H,
                        rope_asm_r0 : rope_asm_r0 + ROPE_CHUNK,
                    ]
                    rope_apply_odd_chunk = o_proj_odd[
                        rope_apply_head_row : rope_apply_head_row + H,
                        rope_asm_r0 : rope_asm_r0 + ROPE_CHUNK,
                    ]
                    rope_even_acc = pl.add(
                        pl.col_expand_mul(rope_apply_even_chunk, cos_chunk),
                        pl.col_expand_mul(rope_apply_odd_chunk, sin_chunk),
                    )
                    rope_odd_acc = pl.sub(
                        pl.col_expand_mul(rope_apply_odd_chunk, cos_chunk),
                        pl.col_expand_mul(rope_apply_even_chunk, sin_chunk),
                    )
                    rope_rot_even_chunk = pl.cast(rope_even_acc, target_type=pl.BF16, mode="rint")
                    rope_rot_odd_chunk = pl.cast(rope_odd_acc, target_type=pl.BF16, mode="rint")
                    rope_even_interleave = pl.matmul(
                        rope_rot_even_chunk,
                        even_select_local,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    rope_odd_interleave = pl.matmul(
                        rope_rot_odd_chunk,
                        odd_select_local,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    rope_even_interleave_buf = pl.assemble(
                        rope_even_interleave_buf,
                        rope_even_interleave,
                        [rope_apply_head_row, 2 * rope_asm_r0],
                    )
                    rope_odd_interleave_buf = pl.assemble(
                        rope_odd_interleave_buf,
                        rope_odd_interleave,
                        [rope_apply_head_row, 2 * rope_asm_r0],
                    )

    for rope_pack_block in pl.spmd(ROPE_PACK_SPMD_BLOCKS, name_hint="cfa_proj_rope_pack_group_spmd"):
        rope_pack_token_block = rope_pack_block // O_GROUPS
        rope_pack_g = rope_pack_block - rope_pack_token_block * O_GROUPS
        rope_combine_t0 = rope_pack_token_block * ROPE_PACK_TOKEN_TILE

        for rope_combine_dt in pl.range(ROPE_PACK_TOKEN_TILE):
            rope_combine_t = rope_combine_t0 + rope_combine_dt
            rope_pack_head_row = rope_combine_t * H + rope_pack_g * HEADS_PER_GROUP

            # Merge and write only this group's inverse-RoPE tail.
            rope_even_tile = rope_even_interleave_buf[
                rope_pack_head_row : rope_pack_head_row + HEADS_PER_GROUP,
                0:ROPE_DIM,
            ]
            rope_odd_tile = rope_odd_interleave_buf[
                rope_pack_head_row : rope_pack_head_row + HEADS_PER_GROUP,
                0:ROPE_DIM,
            ]
            rope_full = pl.cast(
                pl.add(rope_even_tile, rope_odd_tile),
                target_type=pl.BF16,
            )
            rope_pack_row = rope_pack_g * T + rope_combine_t
            for rope_pack_hh in pl.range(HEADS_PER_GROUP):
                rope_pack_head_col = rope_pack_hh * HEAD_DIM + NOPE_DIM
                o_packed = pl.assemble(
                    o_packed,
                    rope_full[rope_pack_hh : rope_pack_hh + 1, 0:ROPE_DIM],
                    [rope_pack_row, rope_pack_head_col],
                )

    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    o_r_amax_parts = pl.create_tensor([A_AMAX_BLOCKS, T], dtype=pl.FP32)
    o_r_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)

    # Stage 5: grouped BF16 projection `o_packed @ wo_a^T`, producing the
    # low-rank intermediate activation `o_r`.
    for g in pl.parallel(0, O_GROUPS, 1):
        row_base_o = g * T
        out_col_g = g * O_LORA

        for nb in pl.parallel(0, A_N_BLOCKS, 1):
            n0 = nb * A_N_CHUNK

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_a_accum"):
                # K-split BF16 matmul for one wo_a output tile.
                xa0_chunk = o_packed[row_base_o : row_base_o + T, 0:A_K_CHUNK]
                wa0_chunk = wo_a[g : g + 1, n0 : n0 + A_N_CHUNK, 0:A_K_CHUNK]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, A_K_BLOCKS, stage=2):
                    k0 = kb * A_K_CHUNK
                    xa_k_chunk = o_packed[row_base_o : row_base_o + T, k0 : k0 + A_K_CHUNK]
                    wa_k_chunk = wo_a[g : g + 1, n0 : n0 + A_N_CHUNK, k0 : k0 + A_K_CHUNK]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_a_store_amax"):
                # Store BF16 activations and expose the tile's row-wise partial amax for quant.
                acc_a_2d = pl.reshape(acc_a, [T, A_N_CHUNK])
                acc_a_bf16 = pl.cast(
                    acc_a_2d,
                    target_type=pl.BF16,
                )
                o_r[:, out_col_g + n0 : out_col_g + n0 + A_N_CHUNK] = acc_a_bf16
                acc_a_f32 = pl.cast(acc_a_bf16, target_type=pl.FP32)
                acc_a_abs = pl.maximum(acc_a_f32, pl.neg(acc_a_f32))
                acc_a_amax = pl.reshape(pl.row_max(acc_a_abs), [1, T])
                amax_part_row = g * A_N_BLOCKS + nb
                o_r_amax_parts[amax_part_row : amax_part_row + 1, 0:T] = acc_a_amax

    # Stage 6: per-row symmetric INT8 quantization of `o_r` for the W8A8C16
    # second projection stage.
    for quant_t0 in pl.parallel(0, T, QUANT_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_quant_tile"):
            or_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for ab in pl.range(0, A_AMAX_BLOCKS, 1):
                or_a_part = o_r_amax_parts[ab : ab + 1, quant_t0 : quant_t0 + QUANT_TOKEN_TILE]
                or_amax = pl.maximum(or_amax, or_a_part)
            or_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
            or_scale_dq = pl.reshape(pl.recip(or_sq_row), [QUANT_TOKEN_TILE, 1])
            o_r_scale_dq[quant_t0 : quant_t0 + QUANT_TOKEN_TILE, 0:1] = or_scale_dq
            or_sq_col = pl.reshape(or_sq_row, [QUANT_TOKEN_TILE, 1])
            for k1 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
                or_q_f32 = pl.cast(
                    o_r[quant_t0 : quant_t0 + QUANT_TOKEN_TILE, k1 : k1 + QUANT_CHUNK], target_type=pl.FP32
                )
                or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
                or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
                or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
                o_r_i8[quant_t0 : quant_t0 + QUANT_TOKEN_TILE, k1 : k1 + QUANT_CHUNK] = pl.cast(
                    or_q_half,
                    target_type=pl.INT8,
                    mode="trunc",
                )

    # Stage 7: INT8 projection `o_r_i8 @ wo_b^T`, then dequantize with the
    # activation and weight scales into the final BF16 output.
    for nb in pl.parallel(0, B_N_BLOCKS, 1):
        n0 = nb * B_N_CHUNK

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_accum"):
            # K-split INT8 GEMM for one output-channel tile.
            xb0_chunk = o_r_i8[:, 0:B_K_CHUNK]
            wb0_chunk = wo_b[n0 : n0 + B_N_CHUNK, 0:B_K_CHUNK]
            acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
            for kb in pl.pipeline(1, B_K_BLOCKS, stage=2):
                k0 = kb * B_K_CHUNK
                xb_k_chunk = o_r_i8[:, k0 : k0 + B_K_CHUNK]
                wb_k_chunk = wo_b[n0 : n0 + B_N_CHUNK, k0 : k0 + B_K_CHUNK]
                acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_store"):
            # Apply the per-row and per-channel dequant scales before casting to BF16.
            wb_scale_chunk = pl.reshape(wo_b_scale[n0 : n0 + B_N_CHUNK], [1, B_N_CHUNK])
            attn_chunk = pl.cast(acc_b, target_type=pl.FP32, mode="none")
            attn_chunk = pl.col_expand_mul(pl.row_expand_mul(attn_chunk, o_r_scale_dq), wb_scale_chunk)
            attn_out[:, n0 : n0 + B_N_CHUNK] = pl.cast(attn_chunk, target_type=pl.BF16, mode="rint")

    return attn_out


@pl.program
class SparseAttnProgram:
    """``@pl.program`` wrapper around the verbatim sparse_attn kernel (ST entry)."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
        ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
        ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
        cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
        cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
        cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
        attn_sink: pl.Tensor[[H], pl.FP32],
        seqused_kv: pl.Tensor[[B], pl.INT32],
        freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
        freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
        even_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], pl.BF16],
        odd_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], pl.BF16],
        wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
        wo_b_scale: pl.Tensor[[D], pl.FP32],
        attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    ) -> pl.Tensor[[T, D], pl.BF16]:
        return sparse_attn(
            q,
            ori_kv,
            ori_block_table,
            cmp_kv,
            cmp_block_table,
            cmp_sparse_indices,
            attn_sink,
            seqused_kv,
            freqs_cos,
            freqs_sin,
            even_select_local,
            odd_select_local,
            wo_a,
            wo_b,
            wo_b_scale,
            attn_out,
        )


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path."""

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis."""

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_sparse_attn(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    attn_sink = tensors["attn_sink"].float()
    seqused_kv = tensors["seqused_kv"]
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)

    # Per-query-token attention. seqused_kv stores each batch's final sparse
    # length for this decode chunk; token t derives its causal length from s.
    for t in range(T):
        b = t // S
        s = t - b * S
        seq_used = int(seqused_kv[b].item()) - S + 1 + s
        window_valid = min(WIN, seq_used)
        cmp_valid = max(seq_used - window_valid, 0)
        gathered = []

        for raw in cmp_sparse_indices[t].tolist():
            if raw < 0:
                continue
            if raw < WIN:
                if raw >= window_valid:
                    continue
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[blk_id, intra, 0])
            else:
                cmp_slot = raw - WIN
                if cmp_slot >= cmp_valid:
                    continue
                blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[blk_id, intra, 0])

        if not gathered:
            continue

        kv_b = torch.stack(gathered, dim=0)
        q_t = q[t]
        scores = (q_t @ kv_b.T) * SOFTMAX_SCALE
        score_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - score_max)
        oi_num = exp_scores @ kv_b
        li = exp_scores.sum(dim=-1, keepdim=True)
        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[t] = oi_num / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    seq_per_batch = T // B
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)

    tensors["attn_out"][:] = out.to(torch.bfloat16)


def build_tensor_specs(
    compress_ratio: int = DEFAULT_COMPRESS_RATIO,
    causal_regression_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""

    cmp_valid = get_standalone_cmp_valid(compress_ratio)
    sparse_k = WIN + cmp_valid

    def seeded_uniform(shape, seed):
        """Create a deterministic centered uniform tensor for repeatable tests."""
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.rand(*shape, generator=generator) - 0.5

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = seeded_uniform((T, H, HEAD_DIM), 1)
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = seeded_uniform((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 2)
        if causal_regression_fixture:
            kv[0, WIN - 1, 0].fill_(8.0)
        return kv

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return seeded_uniform((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 3)

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_ori_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_cmp_sparse_indices():
        """Build the sparse index list with a full window prefix and padded compressed tail."""
        win_part = torch.arange(WIN, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        cmp_part = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
        cmp_part[:, :cmp_valid] = (
            (torch.arange(cmp_valid, dtype=torch.int32) + WIN).unsqueeze(0).expand(T, -1)
        )
        indices = torch.cat([win_part, cmp_part], dim=-1).contiguous()
        if causal_regression_fixture:
            indices[0, WIN - 1] = WIN - 1
        return indices

    def init_seqused_kv():
        """Expose the demo sequence-used length that matches the chosen ratio mode."""
        return torch.full((B,), sparse_k, dtype=torch.int32)

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    def init_odd_select_local():
        """Build the chunk-local selector that extracts odd rope lanes from interleaved inputs."""
        matrix = torch.zeros((ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK))
        for i in range(ROPE_CHUNK):
            matrix[2 * i + 1, i] = 1
        return matrix

    def init_even_select_local():
        """Build the chunk-local selector that extracts even rope lanes from interleaved inputs."""
        matrix = torch.zeros((ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK))
        for i in range(ROPE_CHUNK):
            matrix[2 * i, i] = 1
        return matrix

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights."""
        return seeded_uniform((O_GROUPS, O_LORA, O_GROUP_IN), 4) / (O_GROUP_IN**0.5)

    wo_b_bf16 = (seeded_uniform((D, O_GROUPS * O_LORA), 5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        """Initialize the second-stage output-projection weights in per-channel INT8 form."""
        return wo_b_i8

    def init_wo_b_scale():
        """Initialize the dequant scales paired with the INT8 second-stage weights."""
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], DataType.BF16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], DataType.BF16, init_value=init_ori_kv),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], DataType.INT32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], DataType.BF16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], DataType.INT32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK], DataType.INT32, init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink", [H], DataType.FP32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], DataType.INT32, init_value=init_seqused_kv),
        TensorSpec("freqs_cos", [T, ROPE_DIM], DataType.BF16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], DataType.BF16, init_value=init_sin),
        TensorSpec(
            "even_select_local",
            [ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],
            DataType.BF16,
            init_value=init_even_select_local,
        ),
        TensorSpec(
            "odd_select_local",
            [ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],
            DataType.BF16,
            init_value=init_odd_select_local,
        ),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], DataType.BF16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], DataType.INT8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], DataType.FP32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], DataType.BF16, is_output=True),
    ]


class SparseAttnReproCase(PTOTestCase):
    """#1457 problem 2: real DeepSeek-V4 sparse_attn — runtime deadlock repro."""

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_name(self) -> str:
        return "sparse_attn_repro_1457"

    def define_tensors(self) -> list[TensorSpec]:
        return build_tensor_specs(DEFAULT_COMPRESS_RATIO, causal_regression_fixture=False)

    def get_program(self) -> Any:
        return SparseAttnProgram

    def compute_expected(self, tensors, params=None):
        golden_sparse_attn(tensors)


class TestSparseAttnRepro1457:
    """Issue #1457 problem 2: real sparse_attn kernel reproduction."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_sparse_attn(self, test_runner, platform):
        """Compile + golden pass; on a2a3 hardware the runtime deadlocks (orch_error_code=2)."""
        result = test_runner.run(SparseAttnReproCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
