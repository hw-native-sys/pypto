# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa
"""Qwen3-14B single-layer decode forward.

Scope 1:
  1. RMSNorm of input hidden states
  2. Q/K/V projection via matmul

Per-head q_norm / k_norm

Scope 2 (fused flash-attention, explicit pl.spmd, bidirectional cube<->vec):
  1. K RoPE + paged cache write, V paged cache write, Q RoPE + pad.
  2. fa_fused (ONE mixed root, dispatched via pl.spmd): QK matmul (cube) +
     tail-masked softmax (vec) + SV matmul (cube), per sequence block.
     Both cross-lane handoffs are CV BOUNDARY MOVES (cube QK -> vec
     softmax = C2V; vec exp -> cube SV = V2C), not GM round-trips.
     Dispatch: pl.spmd(TOTAL_Q_GROUPS // 2) with the UP_DOWN split applied
     to the outlined InCore scope; each spmd block owns a Q-group PAIR and
     pipelines the two groups with pl.pipeline(stage=2) to give the
     bidirectional pipe its ping-pong buffering. NO chunk / auto_chunk
     (deprecated). Per-block oi_tmp / mi / li are stashed to GM.
  3. online_softmax (separate vec-only pl.spmd region): online recurrence
     over blocks (reading the per-block GM tiles), then trim
     Q_HEAD_PAD->Q_HEAD_BATCH, cast to BF16, assemble into attn_row. Kept
     separate because the recurrence is sequential while fa_fused is
     pipelined, and the flatten reshape conflicts with the UP_DOWN split.

  NOTE: BLOCK_SIZE=128 (SEQ_TILE=128 in config.py) keeps each K/V tile at
  32 KB so the cube L0B holds two double-buffered tiles within the 64 KB
  platform limit. The Q_HEAD_BATCH trim runs in the non-UP_DOWN
  online_softmax region, so the official Qwen3-14B Q_HEAD_BATCH=5 (odd) is
  fine; fa_fused itself splits on Q_HEAD_PAD=16 (always even).

Scope 3:
  1. Output projection: attn_out × wo
  2. Residual addition with hidden_states
  3. Post-attention RMSNorm
  4. MLP: gate/up projections, SiLU activation, down projection
  5. Final residual addition

The final RMSNorm and LM head projection live in rms_lm_head.py.
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl
from config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    DOWN_MLP_CHUNK,
    DOWN_OUT_CHUNK,
    EPS,
    FINAL_RMS_K_CHUNK,
    HALF_DIM,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN,
    KV_OUT_CHUNK,
    KV_PROJ_K_CHUNK,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LM_HEAD_K_CHUNK,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ,
    MLP_OUT_CHUNK,
    MLP_SPMD_INNER,
    NUM_HEADS,
    NUM_KV_HEADS,
    OUT_PROJ_K_CHUNK,
    OUT_PROJ_N_CHUNK,
    Q_GROUPS,
    Q_HEAD_BATCH,
    Q_HEAD_PAD,
    Q_OUT_CHUNK,
    Q_PER_KV,
    ROPE_SEQ_DYN,
    TOTAL_Q_GROUPS,
    USER_BATCH_DYN,
    VOCAB,
    VOCAB_CHUNK,
)
from rms_lm_head import rms_lm_head, rms_lm_head_single_chunk, rms_only

# The MLP SPMD grouping bundles MLP_SPMD_INNER output blocks per dispatch,
# so the output-block count must be an exact multiple.
assert (INTERMEDIATE // MLP_OUT_CHUNK) % MLP_SPMD_INNER == 0


@pl.jit.inline
def decode_layer(
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
    hidden_blocks = HIDDEN // K_CHUNK
    # NOTE: out_proj / gate_up_silu / down_proj use the block-count
    # constants HIDDEN // Q_OUT_CHUNK, INTERMEDIATE // MLP_OUT_CHUNK,
    # and HIDDEN // K_CHUNK inlined directly on pl.spmd(...) — pl.spmd
    # outlines its body to a top-level function and SSA-verifies the
    # block count outside the JIT-inlined scope, so a local alias
    # defined here would trip "used outside its defining scope".
    head_dim_inv = HEAD_DIM_INV
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    bt_stride = pl.tensor.dim(block_table, 0) // user_batch
    batch_padded = BATCH
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cache_base = layer_idx * decode_layer_cache_rows

    # Intermediate FP32 tensors between scope 1 and scope 2.
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    # Scope 1: input RMSNorm + Q/K/V projection.
    # The JIT inline path follows the fixed-BATCH single-layer kernel
    # contract, so every matmul tile has a static M dim of BATCH_TILE.
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        normed_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
            partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(decode_scope1_hidden_blocks):
                sq_k0 = kb * INPUT_PROJ_K_CHUNK
                sq_chunk = pl.cast(
                    pl.slice(
                        current_hidden,
                        [BATCH_TILE, INPUT_PROJ_K_CHUNK],
                        [b0, sq_k0],
                    ),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(sq_chunk, sq_chunk)), [1, BATCH_TILE]),
                )
            variance = pl.reshape(
                pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                [BATCH_TILE, 1],
            )
            inv_rms = pl.recip(pl.sqrt(variance))

            for kb in pl.range(decode_scope1_hidden_blocks):
                norm_k0 = kb * INPUT_PROJ_K_CHUNK
                norm_chunk = pl.cast(
                    pl.slice(
                        current_hidden,
                        [BATCH_TILE, INPUT_PROJ_K_CHUNK],
                        [b0, norm_k0],
                    ),
                    target_type=pl.FP32,
                )
                gamma = pl.slice(input_rms_weight, [1, INPUT_PROJ_K_CHUNK], [layer_idx, norm_k0])
                normed = pl.col_expand_mul(pl.row_expand_mul(norm_chunk, inv_rms), gamma)
                normed_tile = pl.assemble(
                    normed_tile,
                    pl.cast(normed, target_type=pl.BF16),
                    [0, norm_k0],
                )

        for q0 in pl.parallel(0, HIDDEN, Q_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(decode_scope1_hidden_blocks):
                    q_k0 = kb * INPUT_PROJ_K_CHUNK
                    q_tile_a = pl.slice(normed_tile, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [0, q_k0])
                    q_tile_b = pl.slice(wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + q_k0, q0])
                    if q_k0 == 0:
                        q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                    else:
                        q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
                q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

        for kv0 in pl.parallel(0, KV_HIDDEN, KV_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj"):
                k_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(decode_scope1_hidden_blocks):
                    k_k0 = kb * INPUT_PROJ_K_CHUNK
                    k_tile_a = pl.slice(normed_tile, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [0, k_k0])
                    k_tile_b = pl.slice(
                        wk, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k_k0, kv0]
                    )
                    if k_k0 == 0:
                        k_acc = pl.matmul(k_tile_a, k_tile_b, out_dtype=pl.FP32)
                    else:
                        k_acc = pl.matmul_acc(k_acc, k_tile_a, k_tile_b)
                k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj"):
                v_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(decode_scope1_hidden_blocks):
                    v_k0 = kb * INPUT_PROJ_K_CHUNK
                    v_tile_a = pl.slice(normed_tile, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [0, v_k0])
                    v_tile_b = pl.slice(
                        wv, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + v_k0, kv0]
                    )
                    if v_k0 == 0:
                        v_acc = pl.matmul(v_tile_a, v_tile_b, out_dtype=pl.FP32)
                    else:
                        v_acc = pl.matmul_acc(v_acc, v_tile_a, v_tile_b)
                v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

    # HF-style per-head q_norm / k_norm before RoPE, matching the original
    # single-layer qwen3_decode grouping by KV head.
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm"):
            for h in pl.range(NUM_KV_HEADS):
                q0 = h * Q_PER_KV * HEAD_DIM
                q_chunk = pl.reshape(
                    pl.slice(q_proj, [BATCH_TILE, Q_HEAD_BATCH * HEAD_DIM], [b0, q0]),
                    [BATCH_TILE * Q_HEAD_BATCH, HEAD_DIM],
                )
                q_sq_sum = pl.row_sum(pl.mul(q_chunk, q_chunk))
                q_inv_rms = pl.rsqrt(pl.add(pl.mul(q_sq_sum, head_dim_inv), EPS))
                q_chunk_norm = pl.col_expand_mul(
                    pl.row_expand_mul(q_chunk, q_inv_rms),
                    pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                )
                q_chunk_norm_flat = pl.reshape(q_chunk_norm, [BATCH_TILE, Q_HEAD_BATCH * HEAD_DIM])
                q_proj_norm = pl.assemble(q_proj_norm, q_chunk_norm_flat, [b0, q0])

                k0 = h * HEAD_DIM
                k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [b0, k0])
                k_sq_sum = pl.row_sum(pl.mul(k_chunk, k_chunk))
                k_inv_rms = pl.rsqrt(pl.add(pl.mul(k_sq_sum, head_dim_inv), EPS))
                k_chunk_norm = pl.col_expand_mul(
                    pl.row_expand_mul(k_chunk, k_inv_rms),
                    pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                )
                k_proj_norm = pl.assemble(k_proj_norm, k_chunk_norm, [b0, k0])

    # Scope 2: RoPE + KV cache update + grouped decode attention.
    # NOTE: rope_kv_cache could not be merged into fa_fused — see the
    # comment block on the rope_kv_cache pl.at below for the codegen
    # constraints that block it.
    #
    # fa_fused and online_softmax used to live inside `for b in
    # pl.parallel(user_batch)` and dispatch (TOTAL_Q_GROUPS // 2) and
    # TOTAL_Q_GROUPS spmd lanes per batch respectively (16 separate root
    # launches each). They are now promoted to TOP-LEVEL flat pl.spmd
    # dispatches of BATCH * (TOTAL_Q_GROUPS // 2) = 64 and
    # BATCH * TOTAL_Q_GROUPS = 128 lanes — one dispatch per kernel call
    # instead of 16. Same total task count, but no per-batch launch and
    # barrier overhead, and the runtime sees one big spmd pool to load-
    # balance across cube cores instead of 16 small ones. Each spmd
    # block decodes spmd_idx -> (b, g2 or gi) and re-reads its per-b
    # ctx_len / block_table_base.
    #
    # Consequence: all_oi_tmp / all_cur_mi / all_cur_li are promoted to
    # GLOBAL tensors covering all BATCH batches (was per-b creation).
    # attn_out is now written directly by online_softmax (no per-b
    # attn_row intermediate). rope_kv_cache stays as `for b in
    # pl.parallel(user_batch)` because its cos/sin slot writes still
    # need per-batch dynamic state.
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    all_q_padded = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM],
        dtype=pl.BF16,
    )
    all_oi_tmp = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, HEAD_DIM],
        dtype=pl.FP32,
    )
    all_cur_mi = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, 1],
        dtype=pl.FP32,
    )
    all_cur_li = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, 1],
        dtype=pl.FP32,
    )

    # Per-batch rope_kv_cache. Scope 2 only touches runtime-visible rows;
    # padded rows stay zero.
    for b in pl.parallel(user_batch):
        # ctx_blocks / block_table_base no longer needed here — fa_fused
        # and online_softmax re-derive their own per-block versions after
        # being promoted to top-level pl.spmd.
        ctx_len = pl.tensor.read(seq_lens, [b])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE
        cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, HALF_DIM], [0, 0])
        cos_hi = pl.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
        sin_lo = pl.slice(sin_row, [1, HALF_DIM], [0, 0])
        sin_hi = pl.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])

        # rope_kv_cache (vec-only). Stays as its own pl.at for two
        # reasons that make a merge into fa_fused infeasible:
        #   1. K/V slot writes go to k_cache / v_cache (GM) and the
        #      fa_fused QK/SV matmuls then slice the same GM tensor.
        #      Writing AND slicing the same tensor inside one InCore
        #      region fails codegen with "Tensor view not found for
        #      parameter: k_cache__tile". Cross-region barriers from
        #      pl.at boundaries are needed for the slot write to be
        #      visible to the matmul reads.
        #   2. Q RoPE writes a Q_HEAD_BATCH=5-row data slab + a
        #      Q_HEAD_PAD-Q_HEAD_BATCH=11-row zero pad into the
        #      all_q_padded bridge. Inside the fa_fused mixed root,
        #      ExpandMixedKernel applies an UP_DOWN row split, and
        #      ptoas rejects the resulting 5/11-row subviews with
        #      "'pto.subview' op valid_row must be positive when
        #      constant". Pre-padding to Q_HEAD_PAD rows before
        #      rotation doesn't help because the 5-row assemble at
        #      offset 0 still produces a 5-row subview after split.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
            for ki in pl.range(NUM_KV_HEADS):
                kv_col = ki * HEAD_DIM
                cache_row = layer_cache_base + (slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + slot_offset
                k_lo = pl.slice(k_proj_norm, [1, HALF_DIM], [b, kv_col])
                k_hi = pl.slice(k_proj_norm, [1, HALF_DIM], [b, kv_col + HALF_DIM])
                rot_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                rot_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                k_cache = pl.assemble(
                    k_cache,
                    pl.cast(rot_lo, target_type=pl.BF16),
                    [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache,
                    pl.cast(rot_hi, target_type=pl.BF16),
                    [cache_row, HALF_DIM],
                )
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(
                        pl.slice(v_proj, [1, HEAD_DIM], [b, kv_col]),
                        target_type=pl.BF16,
                    ),
                    [cache_row, 0],
                )
                q_base = ki * Q_PER_KV
                q_block = pl.reshape(
                    pl.slice(q_proj_norm, [1, Q_HEAD_BATCH * HEAD_DIM], [b, q_base * HEAD_DIM]),
                    [Q_HEAD_BATCH, HEAD_DIM],
                )
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH, HALF_DIM], [0, 0])
                q_hi = pl.slice(q_block, [Q_HEAD_BATCH, HALF_DIM], [0, HALF_DIM])
                rot_lo_bf16 = pl.cast(
                    pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)),
                    target_type=pl.BF16,
                )
                rot_hi_bf16 = pl.cast(
                    pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)),
                    target_type=pl.BF16,
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    rot_lo_bf16,
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD, 0],
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    rot_hi_bf16,
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD, HALF_DIM],
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(
                        pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),
                        target_type=pl.BF16,
                    ),
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                )

    # fa_fused promoted to TOP-LEVEL flat spmd: BATCH * (TOTAL_Q_GROUPS // 2)
    # lanes total dispatched in one shot (replaces 16 per-batch dispatches
    # of 4 lanes each). Each spmd block decodes spmd_idx -> (b, g2) and
    # re-reads its per-b ctx_len / block_table_base. Different blocks
    # have unique (b, gi), so each q_padded slice is unique — unlike the
    # previously-attempted FA_SB_CHUNKS path where two blocks shared the
    # same (b, gi).
    #
    # Scope-2 attention: QK matmul + tail-masked softmax + SV matmul fused
    # into ONE mixed root (fa_fused) using CV boundary moves for both
    # cross-lane directions (cube QK -> vec softmax = C2V; vec exp -> cube
    # SV = V2C). Each spmd block owns a Q-group PAIR and pipelines the two
    # groups with pl.pipeline(2, stage=2) — that supplies the bidirectional
    # pipe's ping-pong buffering (without it the bidirectional pipe
    # deadlocks; with one Q group per block it silently writes only half
    # the attention output).
    for fa_spmd_idx in pl.spmd(
        BATCH * (TOTAL_Q_GROUPS // 2),
        name_hint="fa_fused",
    ):
        fa_b = fa_spmd_idx // (TOTAL_Q_GROUPS // 2)
        fa_g2 = fa_spmd_idx % (TOTAL_Q_GROUPS // 2)
        # Clamp the per-b reads of USER_BATCH_DYN-backed tensors so padded
        # spmd lanes (fa_b >= user_batch when runtime_user_batch < BATCH)
        # never index past the dynamic dim. Padded lanes re-process batch
        # (user_batch - 1)'s ctx_len / block_table entries; their writes
        # to all_oi_tmp / all_cur_mi / all_cur_li still land in their own
        # (padded) (b, gi) slot of the BATCH-sized global scratch and are
        # never read back by online_softmax for valid os_b.
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_ctx_blocks = (fa_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        fa_block_table_base = fa_b_safe * bt_stride

        for gp in pl.pipeline(2, stage=2):
            gi = fa_g2 * 2 + gp
            kvh = gi // Q_GROUPS
            q_padded_row = fa_b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
            q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, HEAD_DIM], [q_padded_row, 0])
            # g_base indexes per (b, gi) into the global per-call scratch.
            g_base = (fa_b * TOTAL_Q_GROUPS + gi) * MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD

            for sb in pl.range(fa_ctx_blocks):
                s0 = sb * BLOCK_SIZE
                valid_len = pl.min(BLOCK_SIZE, fa_ctx_len - s0)
                fa_block_table_idx = fa_block_table_base + sb
                fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_block_table_idx]), pl.INDEX)
                fa_cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE

                # QK matmul (cube). First vec consumer (pl.mul) triggers the
                # C2V boundary move; set_validshape runs on the vec tile.
                k_tile = k_cache[fa_cache_row : fa_cache_row + BLOCK_SIZE, :]
                raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                scores_scaled = pl.mul(raw_scores, decode_attn_scale)
                # set_validshape's row operand is the POST-split tile
                # height: the surrounding spmd body is a cube+vec mixed
                # root and ExpandMixedKernel applies an UP_DOWN row split,
                # so the vec lane sees Q_HEAD_PAD // 2 rows per chunk.
                scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_PAD // 2, valid_len)
                scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                cur_mi = pl.row_max(scores)
                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                cur_li = pl.row_sum(exp_scores_fp32)

                # SV matmul (cube) reads exp directly -> V2C boundary move.
                v_tile = v_cache[fa_cache_row : fa_cache_row + BLOCK_SIZE, :]
                oi_tmp = pl.matmul(exp_scores_bf16, v_tile, out_dtype=pl.FP32)

                all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [g_base + sb * Q_HEAD_PAD, 0])
                all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + sb * Q_HEAD_PAD, 0])
                all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + sb * Q_HEAD_PAD, 0])

    # online_softmax promoted to TOP-LEVEL flat spmd: BATCH * TOTAL_Q_GROUPS
    # = 128 lanes in one dispatch (was 16 per-batch dispatches of 8).
    # Each block writes its (b, gi) tile of attn_out directly — no
    # per-batch attn_row intermediate.
    for os_spmd_idx in pl.spmd(BATCH * TOTAL_Q_GROUPS, name_hint="online_softmax"):
        os_b = os_spmd_idx // TOTAL_Q_GROUPS
        os_gi = os_spmd_idx % TOTAL_Q_GROUPS
        # See the matching clamp on fa_b above: padded spmd lanes
        # (os_b >= user_batch) must not index past seq_lens. Their
        # attn_out write lands in a padded row that the host trims away.
        os_b_safe = pl.min(os_b, user_batch - 1)
        os_ctx_len = pl.tensor.read(seq_lens, [os_b_safe])
        os_ctx_blocks = (os_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        kvh = os_gi // Q_GROUPS
        qg = os_gi - kvh * Q_GROUPS
        q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH
        g_base = (os_b * TOTAL_Q_GROUPS + os_gi) * MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD
        oi = all_oi_tmp[g_base : g_base + Q_HEAD_PAD, :]
        mi = all_cur_mi[g_base : g_base + Q_HEAD_PAD, :]
        li = all_cur_li[g_base : g_base + Q_HEAD_PAD, :]
        for sb in pl.range(1, os_ctx_blocks):
            rec = g_base + sb * Q_HEAD_PAD
            oi_tmp_valid = all_oi_tmp[rec : rec + Q_HEAD_PAD, :]
            online_cur_mi = all_cur_mi[rec : rec + Q_HEAD_PAD, :]
            online_cur_li = all_cur_li[rec : rec + Q_HEAD_PAD, :]
            mi_new = pl.maximum(mi, online_cur_mi)
            alpha = pl.exp(pl.sub(mi, mi_new))
            beta = pl.exp(pl.sub(online_cur_mi, mi_new))
            li = pl.add(pl.mul(alpha, li), pl.mul(beta, online_cur_li))
            oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp_valid, beta))
            mi = mi_new

        ctx = pl.row_expand_div(oi, li)
        ctx_valid = ctx[0:Q_HEAD_BATCH, :]
        ctx_flat_bf16 = pl.cast(pl.reshape(ctx_valid, [1, Q_HEAD_BATCH * HEAD_DIM]), target_type=pl.BF16)
        attn_out = pl.assemble(attn_out, ctx_flat_bf16, [os_b, q_base * HEAD_DIM])

    # Scope 3: output projection + residual + post RMSNorm + MLP + residual.
    # Loops over batch_padded so every iteration processes a full
    # [BATCH_TILE, *] tile (a2a3 matmul M-tile constraint).
    # Both out_proj and down_proj fuse their cube matmul reduction with
    # the vec residual epilogue inside a single mixed cube+vec pl.at
    # (UP_DOWN row-split) — accumulator stays on-chip and the merged
    # region pairs cube/vec ping-pong on half-rows.
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.FP32)

        # ob iterations are independent (each writes a disjoint o0-column
        # slice of resid1_tile), so dispatch them with pl.spmd — both the
        # cube matmul and the vec residual epilogue spread across cores.
        # The spmd body is implicitly InCore (no surrounding pl.at), and
        # cube K-loop + vec cast/add/assemble share that single mixed
        # cube+vec region so o_acc bypasses the GM round-trip and the
        # compiler can interleave cube and vec ops in the same region.
        # optimizations=[pl.split(UP_DOWN)] is required (see gate_up_silu
        # below): without it the merged region holds o_acc on L0C across
        # the entire K-loop AND keeps the full BATCH_TILE-row vec UB
        # workspace alive, exceeding the per-core on-chip budget under
        # --max-seq with decode_q_out_blocks lanes in parallel.
        for ob in pl.spmd(
            HIDDEN // Q_OUT_CHUNK,
            name_hint="out_proj",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            o0 = ob * Q_OUT_CHUNK

            a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
            w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, o0])
            o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
            for kb in pl.range(1, hidden_blocks):
                k0 = kb * K_CHUNK
                a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + k0, o0])
                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

            # Vec residual epilogue inside the same spmd InCore region —
            # no GM round-trip for o_acc; UP_DOWN split lets cube + vec
            # share the per-core UB budget.
            resid = pl.cast(
                pl.slice(current_hidden, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]),
                target_type=pl.FP32,
            )
            resid_sum = pl.add(o_acc, resid)
            resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

        post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden_blocks):
                post_sq_k0 = kb * K_CHUNK
                post_sq_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_sq_k0])
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(post_sq_chunk, post_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

            for kb in pl.range(hidden_blocks):
                post_norm_k0 = kb * K_CHUNK
                post_norm_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_norm_k0])
                post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, post_norm_k0])
                post_normed = pl.col_expand_mul(
                    pl.row_expand_mul(post_norm_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1])),
                    post_gamma,
                )
                normed_bf16 = pl.cast(post_normed, target_type=pl.BF16)
                post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, post_norm_k0])

        mlp_tile = pl.create_tensor([BATCH_TILE, INTERMEDIATE], dtype=pl.BF16)
        # Fused gate_proj + up_proj + SiLU as ONE mixed cube+vec spmd
        # body (same mix-mode used by fa_fused / out_proj / down_proj):
        #   - gate and up matmuls share a SINGLE K-loop that loads
        #     post_chunk once per K-tile and feeds both wg and wu matmuls,
        #     halving post_norm_tile traffic from L1.
        #   - gate_acc / up_acc stay on-chip across the K reduction; the
        #     former gate_group / up_group FP32 GM scratch tiles
        #     ([BATCH_TILE, MLP_GROUP_CHUNK] each, 32 KiB per ob_base) are
        #     gone.
        #   - SiLU (neg/exp/recip/mul/mul/cast) runs as the vec epilogue
        #     in the same spmd body, then assembles BF16 into mlp_tile.
        # optimizations=[pl.split(UP_DOWN)] is required (UB exceeds the
        # 192 KiB per-core limit otherwise — the [BATCH_TILE=16,
        # MLP_OUT_CHUNK=256] FP32 vec chain piles up). UP_DOWN splits
        # BATCH_TILE rows 8/8 so cube + vec ping-pong on half-rows.
        # One pl.spmd dispatches every ob lane directly (no outer
        # pl.parallel chunking) — MLP_SPMD_INNER is no longer needed
        # since lanes are fully independent after the gate_group /
        # up_group GM bridge was eliminated.
        for ob in pl.spmd(
            INTERMEDIATE // MLP_OUT_CHUNK,
            name_hint="gate_up_silu",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            mlp_o0 = ob * MLP_OUT_CHUNK

            # Combined K-loop: one post_chunk load per kb feeds both
            # gate and up matmuls.
            post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
            wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0])
            wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0])
            gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
            up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
            for kb in pl.range(1, hidden_blocks):
                k0 = kb * K_CHUNK
                post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + k0, mlp_o0])
                wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + k0, mlp_o0])
                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

            # Vec SiLU epilogue: sigmoid(gate) * gate * up -> BF16
            # -> mlp_tile slot at mlp_o0. No GM round-trip; cube
            # accumulators feed vec directly via C2V boundary move.
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
            mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, mlp_o0])

        # dob iterations are independent (each writes a disjoint d0-column
        # slice of next_hidden), so dispatch them with pl.spmd — both
        # the cube matmul and the vec residual epilogue spread across
        # cores. The spmd body is implicitly InCore (no pl.at), and the
        # cube K-loop reduction + vec add/cast/assemble share that one
        # mixed cube+vec region so down_acc bypasses the fp32_chunk_gm
        # GM scratch the split form used. optimizations=[UP_DOWN] is
        # required (same constraint as out_proj / gate_up_silu — without
        # it the per-core UB budget is exceeded under --max-seq with all
        # hidden_blocks lanes dispatched in parallel).
        for dob in pl.spmd(
            HIDDEN // K_CHUNK,
            name_hint="down_proj",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            d0 = dob * K_CHUNK

            mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
            w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base, d0])
            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
            for ob in pl.range(1, INTERMEDIATE // MLP_OUT_CHUNK):
                down_o0 = ob * MLP_OUT_CHUNK
                down_mlp_chunk_bf16 = pl.slice(
                    mlp_tile,
                    [BATCH_TILE, MLP_OUT_CHUNK],
                    [0, down_o0],
                )
                w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base + down_o0, d0])
                down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)

            # Vec residual epilogue inside the same spmd InCore region —
            # no GM round-trip for down_acc. cast(BF16) here was once
            # split out to "preserve ND layout"; the UP_DOWN-merged
            # mixed root produces the same BF16 ND output via the C2V
            # boundary move (matches fa_fused / out_proj).
            resid_chunk_fp32 = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])
            out_chunk = pl.add(down_acc, resid_chunk_fp32)
            out_chunk_cast = pl.cast(out_chunk, target_type=pl.BF16)
            next_hidden = pl.assemble(next_hidden, out_chunk_cast, [b0, d0])

    return next_hidden


@pl.jit
def test_decode_layer(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    user_batch = pl.tensor.dim(hidden_states, 0)
    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for kb in pl.range(HIDDEN // K_CHUNK):
                copy_k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, copy_k0])

    next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    current_hidden = decode_layer(
        current_hidden,
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        post_rms_weight,
        w_gate,
        w_up,
        w_down,
        next_hidden,
        0,
    )
    out = rms_lm_head(current_hidden, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


@pl.jit
def test_decode_layer_no_lm_head(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    """Same as ``test_decode_layer`` but swaps the LM head for an RMSNorm-only
    tail (the LM-head matmul over the full vocabulary is skipped). ``out``
    stays at its zero init."""
    user_batch = pl.tensor.dim(hidden_states, 0)
    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for kb in pl.range(HIDDEN // K_CHUNK):
                copy_k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, copy_k0])

    next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    current_hidden = decode_layer(
        current_hidden,
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        post_rms_weight,
        w_gate,
        w_up,
        w_down,
        next_hidden,
        0,
    )
    out = rms_only(current_hidden, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


@pl.jit
def test_decode_layer_single_lm_head(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    """Same as ``test_decode_layer`` but the LM head runs only its first
    ``VOCAB_CHUNK`` output tile (one ``ob`` iteration). Columns past
    ``VOCAB_CHUNK`` stay at the zero init."""
    user_batch = pl.tensor.dim(hidden_states, 0)
    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for kb in pl.range(HIDDEN // K_CHUNK):
                copy_k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, copy_k0])

    next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    current_hidden = decode_layer(
        current_hidden,
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        post_rms_weight,
        w_gate,
        w_up,
        w_down,
        next_hidden,
        0,
    )
    out = rms_lm_head_single_chunk(current_hidden, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    vocab_size: int = VOCAB,
    use_max_seq: bool = False,
):
    import torch
    from golden import TensorSpec

    # Host allocates every batch-dependent tensor at the user-visible
    # batch (no host pad / no host trim). The kernel internally rounds
    # up to BATCH_TILE, zero-pads via valid_shape on input loads, and
    # trims via vec-to-vec textract on the BF16 output. A single
    # compiled program serves any batch <= host capacity (USER_BATCH_DYN
    # / KV_CACHE_ROWS_DYN / BLOCK_TABLE_FLAT_DYN are pl.dynamic dims).
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    vocab = vocab_size
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    cache_rows = num_blocks * num_kv_heads * BLOCK_SIZE
    synthetic_proj_scale = 0.5

    if use_max_seq:
        seq_lens_seed = torch.full((batch,), max_seq, dtype=torch.int32)
    else:
        seq_lens_seed = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq():
        return torch.rand(hidden_size, hidden_size) / hidden_size**0.5

    def init_wk():
        return torch.rand(hidden_size, kv_hidden) / hidden_size**0.5

    def init_wv():
        return synthetic_proj_scale * torch.rand(hidden_size, kv_hidden) / hidden_size**0.5

    def init_q_norm_weight():
        return torch.ones(1, head_dim)

    def init_k_norm_weight():
        return torch.ones(1, head_dim)

    def init_seq_lens():
        return seq_lens_seed.clone()

    def init_block_table():
        return torch.arange(num_blocks, dtype=torch.int32)

    def init_slot_mapping():
        slots = torch.empty(batch, dtype=torch.int32)
        for b in range(batch):
            pos = int(seq_lens_seed[b].item()) - 1
            logical_block = pos // BLOCK_SIZE
            page_offset = pos % BLOCK_SIZE
            phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
            slots[b] = phys_block * BLOCK_SIZE + page_offset
        return slots

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return synthetic_proj_scale * (torch.rand(cache_rows, head_dim) - 0.5)

    def init_wo():
        return synthetic_proj_scale * (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size**0.5

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return synthetic_proj_scale * (torch.rand(hidden_size, inter) - 0.5) / hidden_size**0.5

    def init_w_up():
        return synthetic_proj_scale * (torch.rand(hidden_size, inter) - 0.5) / hidden_size**0.5

    def init_w_down():
        return synthetic_proj_scale * (torch.rand(inter, hidden_size) - 0.5) / inter**0.5

    def init_final_norm_weight():
        return torch.ones(1, hidden_size)

    def init_lm_head_weight():
        return synthetic_proj_scale * (torch.rand(vocab, hidden_size) - 0.5) / hidden_size**0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=init_rms_weight),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16, init_value=init_wv),
        TensorSpec("q_norm_weight", [1, head_dim], torch.float32, init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [1, head_dim], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("block_table", [batch * MAX_BLOCKS_PER_SEQ], torch.int32, init_value=init_block_table),
        TensorSpec("slot_mapping", [batch], torch.int32, init_value=init_slot_mapping),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32, init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32, init_value=init_rope_sin),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16, init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16, init_value=init_v_cache),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, inter], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, inter], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [inter, hidden_size], torch.bfloat16, init_value=init_w_down),
        TensorSpec("final_norm_weight", [1, hidden_size], torch.float32, init_value=init_final_norm_weight),
        TensorSpec("lm_head_weight", [vocab, hidden_size], torch.bfloat16, init_value=init_lm_head_weight),
        TensorSpec("out", [batch, vocab], torch.float32, is_output=True),
    ]


def golden_decode_layer(tensors):
    """PyTorch reference: scope1 (RMSNorm + projection), scope2 (attention), scope3 (output + MLP)."""
    import math

    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    final_norm_weight = tensors["final_norm_weight"]
    lm_head_weight = tensors["lm_head_weight"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-6
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ

    def tiled_matmul(lhs, rhs, k_chunk, n_chunk):
        out = torch.zeros(lhs.shape[0], rhs.shape[1], dtype=torch.float32)
        for n0 in range(0, rhs.shape[1], n_chunk):
            acc = torch.zeros(lhs.shape[0], n_chunk, dtype=torch.float32)
            for k0 in range(0, lhs.shape[1], k_chunk):
                acc = (
                    acc
                    + lhs[:, k0 : k0 + k_chunk].float()
                    @ rhs[
                        k0 : k0 + k_chunk,
                        n0 : n0 + n_chunk,
                    ].float()
                )
            out[:, n0 : n0 + n_chunk] = acc
        return out

    def chunked_row_sq_sum(x, k_chunk):
        acc = torch.zeros(x.shape[0], 1, dtype=torch.float32)
        for k0 in range(0, x.shape[1], k_chunk):
            x_chunk = x[:, k0 : k0 + k_chunk]
            acc = acc + (x_chunk * x_chunk).sum(dim=-1, keepdim=True)
        return acc

    def tiled_lm_head(lhs, rhs_t, k_chunk, vocab_chunk):
        out = torch.zeros(lhs.shape[0], rhs_t.shape[0], dtype=torch.float32)
        for k0 in range(0, lhs.shape[1], k_chunk):
            out = out + lhs[:, k0 : k0 + k_chunk].float() @ rhs_t[:, k0 : k0 + k_chunk].float().T
        return out

    q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, INPUT_PROJ_K_CHUNK):
            x_chunk = x_tile[:, k0 : k0 + INPUT_PROJ_K_CHUNK]
            sq_sum = sq_sum + (x_chunk**2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        rms = torch.sqrt(variance)
        normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

        q_proj[b0:b_end, :] = tiled_matmul(normed, wq, INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK)
        k_proj[b0:b_end, :] = tiled_matmul(normed, wk, KV_PROJ_K_CHUNK, KV_OUT_CHUNK)
        v_proj[b0:b_end, :] = tiled_matmul(normed, wv, KV_PROJ_K_CHUNK, KV_OUT_CHUNK)

    attn_out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_variance = k_heads.pow(2).mean(dim=-1, keepdim=True)
        k_heads = k_heads * torch.rsqrt(k_variance + eps) * k_norm_weight.float()
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
            dim=-1,
        )
        slot = int(slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE

        for ki in range(num_kv_heads):
            cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
            k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_row, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        q_heads = q_proj[b].view(num_heads, head_dim)
        q_variance = q_heads.pow(2).mean(dim=-1, keepdim=True)
        q_heads = q_heads * torch.rsqrt(q_variance + eps) * q_norm_weight.float()
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
            dim=-1,
        )

        attn_row = torch.zeros(1, hidden_size, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                gi = kvh * q_groups + qg
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = min(BLOCK_SIZE, ctx_len - s0)
                    pbid = int(block_table[b * max_ctx_blocks + sb].item())
                    cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                    k_tile = k_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]
                    v_tile = v_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < BLOCK_SIZE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                ctx_flat_bf16 = ctx.reshape(1, -1).to(torch.bfloat16)
                attn_row[
                    :,
                    q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim,
                ] = ctx_flat_bf16

        attn_out[b : b + 1, :] = attn_row

    o_proj = tiled_matmul(attn_out, wo, OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK)
    resid1 = o_proj + hidden_states.float()

    variance = chunked_row_sq_sum(resid1, K_CHUNK) / hidden_size
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = tiled_matmul(normed_bf16, w_gate, K_CHUNK, MLP_OUT_CHUNK)
    up = tiled_matmul(normed_bf16, w_up, K_CHUNK, MLP_OUT_CHUNK)
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = tiled_matmul(mlp_bf16, w_down, DOWN_MLP_CHUNK, DOWN_OUT_CHUNK)

    final_hidden = (down + resid1).bfloat16()

    variance = chunked_row_sq_sum(final_hidden.float(), FINAL_RMS_K_CHUNK) / hidden_size
    inv_rms = torch.rsqrt(variance + eps)
    final_normed = (final_hidden.float() * inv_rms * final_norm_weight.float()).bfloat16()

    tensors["out"][:] = tiled_lm_head(
        final_normed,
        lm_head_weight,
        LM_HEAD_K_CHUNK,
        VOCAB_CHUNK,
    )


def golden_decode_layer_no_lm_head(tensors):
    """Golden for ``test_decode_layer_no_lm_head``: runs the full reference
    body (so cached weights / KV cache stay identical) and then zeroes
    ``out`` to match the kernel, which leaves ``out`` at its zero init."""
    golden_decode_layer(tensors)
    tensors["out"].zero_()


def golden_decode_layer_single_lm_head(tensors):
    """Golden for ``test_decode_layer_single_lm_head``: runs the full
    reference body and zeroes every column past ``VOCAB_CHUNK`` to match
    the kernel, which only writes the first ``VOCAB_CHUNK`` outputs."""
    golden_decode_layer(tensors)
    tensors["out"][:, VOCAB_CHUNK:] = 0


if __name__ == "__main__":
    import argparse
    import sys

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=BATCH,
        help=(
            "User-visible batch size. Host allocates every "
            "batch-dependent tensor at exactly this size; "
            "the kernel internally rounds up to BATCH_TILE "
            "(%d), zero-pads input loads via valid_shape, "
            "and trims the BF16 output via vec-to-vec "
            "textract. A single compiled program serves "
            "any batch <= host KV-cache capacity. Default: "
            "%%(default)s" % BATCH_TILE
        ),
    )
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument(
        "--export-kernel-insight",
        action="store_true",
        default=False,
        help=(
            "After a successful run, export msprof op-simulator Insight traces "
            "for all generated InCore kernels under the same build_output dir."
        ),
    )
    parser.add_argument(
        "--kernel-insight-func",
        action="append",
        default=[],
        help="Only export this generated kernel function; can be repeated.",
    )
    parser.add_argument(
        "--lm-head",
        choices=["full", "skip", "single"],
        default="full",
        help=(
            "Control the LM-head tail. 'full' (default) runs every VOCAB_CHUNK "
            "iteration; 'skip' compiles a variant that only runs the final "
            "RMSNorm and never touches the LM-head matmul; 'single' runs just "
            "one VOCAB_CHUNK iteration. Useful for shrinking compile/run cost "
            "when profiling decode_layer."
        ),
    )
    args = parser.parse_args()

    fn, golden_fn = {
        "full": (test_decode_layer, golden_decode_layer),
        "skip": (test_decode_layer_no_lm_head, golden_decode_layer_no_lm_head),
        "single": (test_decode_layer_single_lm_head, golden_decode_layer_single_lm_head),
    }[args.lm_head]

    result = run_jit(
        fn=fn,
        specs=build_tensor_specs(batch=args.batch, use_max_seq=args.max_seq),
        golden_fn=golden_fn,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=3e-3,
        atol=3e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

    if args.export_kernel_insight:
        if result.work_dir is None:
            print("kernel insight export failed: run result has no build_output directory", file=sys.stderr)
            raise SystemExit(1)
        from tools.export_all_kernel_insight import StepError
        from tools.export_all_kernel_insight import main as export_kernel_insight

        export_args = ["--build-dir", str(result.work_dir)]
        for func in args.kernel_insight_func:
            export_args.extend(["--func", func])
        try:
            export_rc = export_kernel_insight(export_args)
        except StepError as exc:
            print(f"kernel insight export failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if export_rc != 0:
            raise SystemExit(export_rc)
