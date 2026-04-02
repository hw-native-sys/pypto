# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 single-layer decode forward, pa4-style — TILELET-aware version.

Each session in the batch can have a different context length (up to MAX_SEQ).
The ``seq_lens`` input tensor (shape [BATCH], INT32) carries the per-session
context length; the decode position is derived as ``pos = seq_lens[b] - 1``.

Hardware TILELET / TILE constraints
------------------------------------
The target processor requires all computation operands to reside in fixed-size
on-chip storage:

  * **Vector operations** (adds, mul, exp, rsqrt, …): each operand and result
    must be a TILELET of at most **2 KB** (2048 bytes).
  * **CUBE operations** (matmul): each operand must be a TILE of at most
    **16 KB** (16384 bytes).

All chunk-size constants below are chosen to **simultaneously maximise BOTH
vector TILELET and cube TILE utilisation**.  With ``BATCH_TILE=4`` and
``K_CHUNK=128`` the key reduction-direction vector tiles are
``[4, 128] FP32 = 2 KB = TILELET MAX``, and the matmul weight tiles are
``[128, 64] BF16 = 16 KB = TILE MAX``.  Output-direction accumulators are
``[4, 64] FP32 = 1 KB`` (50%) — the maximum possible without exceeding the
16 KB cube limit on the weight tile ``[K_CHUNK × OUT_CHUNK]``.
Where an intermediate buffer is larger (e.g. ``[BATCH_TILE, HIDDEN]``), it is
used only via ``pl.slice`` / ``pl.assemble`` and never passed directly to a
vector or cube instruction; zero-initialisation of such buffers is performed
tile-by-tile.

Design goals:
- decode only (one new token per batch item)
- single Transformer layer
- batch = 16 by default
- per-session KV cache depth up to 4096
- fewer, larger auto_incore scopes
- fused outer loops where practical
- all pl.slice of GM tensors are >= 512 B (alignment rule);
  small decode-only per-head vectors [1, 128] BF16 (256 B) are known exceptions
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

# Vector TILELET budget (2 KB = 2048 B, FP32 = 4 B/elem):
#   [BATCH_TILE, K_CHUNK]       FP32 = [4,128] × 4 = 2048 B = 2 KB  ✓ MAX
#   [BATCH_TILE, Q_OUT_CHUNK]   FP32 = [4, 64] × 4 = 1024 B = 1 KB  ✓
#   [BATCH_TILE, KV_OUT_CHUNK]  FP32 = [4, 64] × 4 = 1024 B = 1 KB  ✓
#   [BATCH_TILE, MLP_OUT_CHUNK] FP32 = [4, 64] × 4 = 1024 B = 1 KB  ✓
#   [BATCH_TILE, K_CHUNK]       FP32 = [4,128] × 4 = 2048 B = 2 KB  ✓ MAX (down proj add)
#   [Q_HEAD_BATCH, HEAD_DIM]    FP32 = [8,128] × 4 = 4096 B = 4 KB  (attn, 2×TILELET)
#   [Q_HEAD_BATCH, SEQ_TILE]   FP32 = [8, 64] × 4 = 2048 B = 2 KB  ✓ MAX (attn scores)
#   [NUM_KV_HEADS, HEAD_DIM//2] FP32 = [8, 64] × 4 = 2048 B = 2 KB  ✓ MAX (K RoPE)
#
# Cube TILE budget (16 KB = 16384 B, BF16 = 2 B/elem):
#   [K_CHUNK, Q_OUT_CHUNK]      BF16 = [128, 64] × 2 = 16384 B = 16 KB ✓ MAX
#   [SEQ_TILE, HEAD_DIM]        BF16 = [ 64,128] × 2 = 16384 B = 16 KB ✓ MAX (attn)
#   [K_CHUNK, KV_OUT_CHUNK]     BF16 = [128, 64] × 2 = 16384 B = 16 KB ✓ MAX
#   [K_CHUNK, MLP_OUT_CHUNK]    BF16 = [128, 64] × 2 = 16384 B = 16 KB ✓ MAX
#   [MLP_OUT_CHUNK, K_CHUNK]    BF16 = [ 64,128] × 2 = 16384 B = 16 KB ✓ MAX (down proj)
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
SEQ_TILE = 64
MLP_OUT_CHUNK = 64
BATCH_TILE = 4
# Q_HEAD_BATCH=8 so that li/mi can be shaped [Q_HEAD_BATCH, 1] via
# pl.full([1, Q_HEAD_BATCH]) + pl.reshape; Q_HEAD_PAD=16 pads the matmul
# M-dimension to a cube fractal-friendly multiple
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16


def build_qwen3_single_layer_decode_program(
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

    HIDDEN_BLOCKS = HIDDEN_CFG // K_CHUNK
    Q_OUT_BLOCKS = HIDDEN_CFG // Q_OUT_CHUNK
    KV_OUT_BLOCKS = KV_HIDDEN_CFG // KV_OUT_CHUNK
    MLP_OUT_BLOCKS = INTER_CFG // MLP_OUT_CHUNK
    PROJ_OUTER_BLOCKS = max(Q_OUT_BLOCKS, KV_OUT_BLOCKS)
    CACHE_ROWS = BATCH_CFG * NUM_KV_HEADS_CFG * MAX_SEQ_CFG
    Q_GROUPS = Q_PER_KV_CFG // Q_HEAD_BATCH
    TOTAL_Q_GROUPS = NUM_KV_HEADS_CFG * Q_GROUPS

    @pl.program
    class Qwen3SingleLayerDecode:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_decode_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
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
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            q_proj = pl.create_tensor([BATCH_CFG, HIDDEN_CFG], dtype=pl.BF16)
            k_proj = pl.create_tensor([BATCH_CFG, KV_HIDDEN_CFG], dtype=pl.BF16)
            v_proj = pl.create_tensor([BATCH_CFG, KV_HIDDEN_CFG], dtype=pl.BF16)
            attn_out = pl.create_tensor([BATCH_CFG, HIDDEN_CFG], dtype=pl.FP32)

            # Scope 1: input RMSNorm + Q/K/V projection.
            with pl.auto_incore():
                # Compute per-row squared sum in BATCH_TILE chunks so that
                # each vector tile is [BATCH_TILE, K_CHUNK] FP32 = 2 KB.
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    partial_sq = pl.create_tensor([BATCH_TILE, 1], dtype=pl.FP32)
                    partial_sq = pl.mul(partial_sq, 0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(partial_sq, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                    inv_rms_tile = pl.rsqrt(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS))

                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=4):
                        q0 = ob * Q_OUT_CHUNK
                        q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        q_acc = pl.mul(q_acc, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.cast(
                                pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                                target_type=pl.FP32,
                            )
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_tile), gamma)
                            wq_chunk = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_acc = pl.add(q_acc, pl.matmul(pl.cast(normed, target_type=pl.BF16), wq_chunk))
                        q_proj = pl.assemble(q_proj, pl.cast(q_acc, target_type=pl.BF16), [b0, q0])

                    for ob in pl.parallel(0, KV_OUT_BLOCKS, 1, chunk=8):
                        kv0 = ob * KV_OUT_CHUNK
                        k_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                        v_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                        k_acc = pl.mul(k_acc, 0.0)
                        v_acc = pl.mul(v_acc, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.cast(
                                pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                                target_type=pl.FP32,
                            )
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_tile), gamma)
                            normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                            wk_chunk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            wv_chunk = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            k_acc = pl.add(k_acc, pl.matmul(normed_bf16, wk_chunk))
                            v_acc = pl.add(v_acc, pl.matmul(normed_bf16, wv_chunk))
                        k_proj = pl.assemble(k_proj, pl.cast(k_acc, target_type=pl.BF16), [b0, kv0])
                        v_proj = pl.assemble(v_proj, pl.cast(v_acc, target_type=pl.BF16), [b0, kv0])

            # Scope 2: RoPE + cache update + decode attention.
            # K RoPE batches all NUM_KV_HEADS=8 heads together so that
            # RoPE half-vectors are [8, 64] FP32 = 2 KB = TILELET MAX.
            # Q attention batches Q_HEAD_BATCH=8 Q heads per group;
            # matmul M-dim is padded to Q_HEAD_PAD=16 for cube fractal alignment.
            # Attention cube tiles [64,128] BF16 = 16 KB remain at MAX.
            for b in pl.parallel(BATCH_CFG):
                ctx_len = pl.tensor.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                cos_row = pl.slice(rope_cos, [1, HEAD_DIM_CFG], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, HEAD_DIM_CFG], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                cos_hi = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                sin_lo = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                sin_hi = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])

                k_group = pl.create_tensor([NUM_KV_HEADS_CFG, HEAD_DIM_CFG], dtype=pl.FP32)
                with pl.incore():
                    # Stage 1a: gather all KV heads into a shared tensor buffer.
                    for ki in pl.range(NUM_KV_HEADS_CFG):
                        kv_col = ki * HEAD_DIM_CFG
                        k_group = pl.assemble(
                            k_group,
                            pl.cast(pl.slice(k_proj, [1, HEAD_DIM_CFG], [b, kv_col]),
                                    target_type=pl.FP32),
                            [ki, 0],
                        )

                k_rot_tensor = pl.create_tensor([NUM_KV_HEADS_CFG, HEAD_DIM_CFG], dtype=pl.FP32)
                with pl.incore():
                    # Stage 1b: rotate K halves and assemble to GM (no concat).
                    k_lo = pl.slice(k_group, [NUM_KV_HEADS_CFG, HEAD_DIM_CFG // 2], [0, 0])
                    k_hi = pl.slice(k_group, [NUM_KV_HEADS_CFG, HEAD_DIM_CFG // 2],
                                    [0, HEAD_DIM_CFG // 2])
                    rot_lo = pl.sub(
                        pl.col_expand_mul(k_lo, cos_lo),
                        pl.col_expand_mul(k_hi, sin_lo),
                    )
                    rot_hi = pl.add(
                        pl.col_expand_mul(k_hi, cos_hi),
                        pl.col_expand_mul(k_lo, sin_hi),
                    )
                    k_rot_tensor = pl.assemble(k_rot_tensor, rot_lo, [0, 0])
                    k_rot_tensor = pl.assemble(k_rot_tensor, rot_hi, [0, HEAD_DIM_CFG // 2])

                with pl.incore():
                    # Stage 1c: update the caches from rotated K tensor.
                    for ki in pl.range(NUM_KV_HEADS_CFG):
                        cache_row = b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + ki * MAX_SEQ_CFG + pos
                        k_cache = pl.assemble(
                            k_cache,
                            pl.cast(pl.slice(k_rot_tensor, [1, HEAD_DIM_CFG], [ki, 0]),
                                    target_type=pl.BF16),
                            [cache_row, 0],
                        )
                        v_cache = pl.assemble(
                            v_cache,
                            pl.slice(v_proj, [1, HEAD_DIM_CFG], [b, ki * HEAD_DIM_CFG]),
                            [cache_row, 0],
                        )

                attn_row = pl.create_tensor([1, HIDDEN_CFG], dtype=pl.FP32)

                # Manually split the decode attention into smaller incore stages so
                # each outlined kernel has a single cross-core payload size.
                for gi in pl.parallel(0, TOTAL_Q_GROUPS, 1):
                    kvh = gi // Q_GROUPS
                    qg = gi - kvh * Q_GROUPS
                    q_base = kvh * Q_PER_KV_CFG + qg * Q_HEAD_BATCH

                    q_group = pl.create_tensor([Q_HEAD_BATCH, HEAD_DIM_CFG], dtype=pl.FP32)
                    with pl.incore():
                        # Stage 2a: gather the Q-head group into a tensor buffer.
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * HEAD_DIM_CFG
                            q_group = pl.assemble(
                                q_group,
                                pl.cast(pl.slice(q_proj, [1, HEAD_DIM_CFG], [b, q_col]),
                                        target_type=pl.FP32),
                                [qi, 0],
                            )

                    q_rot_bf16 = pl.create_tensor([Q_HEAD_BATCH, HEAD_DIM_CFG], dtype=pl.BF16)
                    with pl.incore():
                        # Stage 2b: apply RoPE halves, cast and assemble to GM (no concat).
                        q_lo = pl.slice(q_group, [Q_HEAD_BATCH, HEAD_DIM_CFG // 2], [0, 0])
                        q_hi = pl.slice(q_group, [Q_HEAD_BATCH, HEAD_DIM_CFG // 2],
                                        [0, HEAD_DIM_CFG // 2])
                        q_rot_lo = pl.sub(
                            pl.col_expand_mul(q_lo, cos_lo),
                            pl.col_expand_mul(q_hi, sin_lo),
                        )
                        q_rot_hi = pl.add(
                            pl.col_expand_mul(q_hi, cos_hi),
                            pl.col_expand_mul(q_lo, sin_hi),
                        )
                        q_rot_lo_bf16 = pl.cast(q_rot_lo, target_type=pl.BF16)
                        q_rot_hi_bf16 = pl.cast(q_rot_hi, target_type=pl.BF16)
                        q_rot_bf16 = pl.assemble(q_rot_bf16, q_rot_lo_bf16, [0, 0])
                        q_rot_bf16 = pl.assemble(q_rot_bf16, q_rot_hi_bf16, [0, HEAD_DIM_CFG // 2])

                    with pl.incore():
                        oi = pl.full([Q_HEAD_BATCH, HEAD_DIM_CFG], dtype=pl.FP32, value=0.0)
                        li_flat = pl.full([1, Q_HEAD_BATCH], dtype=pl.FP32, value=0.0)
                        li = pl.reshape(li_flat, [Q_HEAD_BATCH, 1])
                        mi_flat = pl.full([1, Q_HEAD_BATCH], dtype=pl.FP32, value=0.0)
                        mi = pl.reshape(mi_flat, [Q_HEAD_BATCH, 1])

                    # Pad Q for cube fractal alignment.
                    q_padded = pl.create_tensor([Q_HEAD_PAD, HEAD_DIM_CFG], dtype=pl.BF16)
                    with pl.incore():
                        q_bf16_tile = pl.slice(q_rot_bf16, [Q_HEAD_BATCH, HEAD_DIM_CFG], [0, 0])
                        q_padded = pl.assemble(q_padded, q_bf16_tile, [0, 0])

                    for sb in pl.range(ctx_blocks):
                        s0 = sb * SEQ_TILE
                        valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                        cache_row0 = b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + kvh * MAX_SEQ_CFG + s0

                        raw_scores_pad = pl.create_tensor([Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                        with pl.incore():
                            # QK matmul: padded Q × K^T.
                            k_tile = pl.slice(
                                k_cache,
                                [SEQ_TILE, HEAD_DIM_CFG],
                                [cache_row0, 0],
                            )
                            raw_scores_pad = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)

                        with pl.incore():
                            # Softmax: slice valid rows from padded scores.
                            scores_valid = pl.slice(
                                raw_scores_pad,
                                [Q_HEAD_BATCH, SEQ_TILE],
                                [0, 0],
                                valid_shape=[Q_HEAD_BATCH, valid_len],
                            )
                            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                            # 2. scale after fillpad
                            scores = pl.mul(scores_padded, ATTN_SCALE)
                            # 3. row_max, exp
                            cur_mi = pl.row_max(scores)
                            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                            # 4. BF16 round-trip before row_sum (li matches SV matmul weights)
                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                            exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                            cur_li = pl.row_sum(exp_scores_fp32)

                        # Pad exp_scores for SV matmul.
                        exp_padded = pl.create_tensor([Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                        with pl.incore():
                            exp_tile = pl.slice(exp_scores_bf16, [Q_HEAD_BATCH, SEQ_TILE], [0, 0])
                            exp_padded = pl.assemble(exp_padded, exp_tile, [0, 0])

                        oi_tmp_pad = pl.create_tensor([Q_HEAD_PAD, HEAD_DIM_CFG], dtype=pl.FP32)
                        with pl.incore():
                            # SV matmul: padded exp_scores × V.
                            v_tile = pl.slice(
                                v_cache,
                                [SEQ_TILE, HEAD_DIM_CFG],
                                [cache_row0, 0],
                            )
                            oi_tmp_pad = pl.matmul(exp_padded, v_tile, out_dtype=pl.FP32)

                        with pl.incore():
                            # Slice valid rows from padded SV result.
                            oi_tmp = pl.slice(oi_tmp_pad, [Q_HEAD_BATCH, HEAD_DIM_CFG], [0, 0])
                            if sb == 0:
                                oi = oi_tmp
                                li = cur_li
                                mi = cur_mi
                            else:
                                mi_new = pl.maximum(mi, cur_mi)
                                alpha = pl.exp(pl.sub(mi, mi_new))
                                beta = pl.exp(pl.sub(cur_mi, mi_new))
                                li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                                oi = pl.add(pl.row_expand_mul(oi, alpha),
                                            pl.row_expand_mul(oi_tmp, beta))
                                mi = mi_new

                    with pl.incore():
                        ctx = pl.row_expand_div(oi, li)
                        ctx_flat = pl.reshape(ctx, [1, Q_HEAD_BATCH * HEAD_DIM_CFG])
                        attn_row = pl.assemble(
                            attn_row, ctx_flat, [0, q_base * HEAD_DIM_CFG],
                        )

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            # Scope 3: output projection + residual + post RMSNorm + MLP + residual.
            with pl.auto_incore():
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)

                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                        o0 = ob * Q_OUT_CHUNK
                        o_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        o_acc = pl.mul(o_acc, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.cast(
                                pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0]),
                                target_type=pl.BF16,
                            )
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]),
                            target_type=pl.FP32,
                        )
                        resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                    sq_sum = pl.create_tensor([BATCH_TILE, 1], dtype=pl.FP32)
                    sq_sum = pl.mul(sq_sum, 0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                    inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                    post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                    down_proj_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    for zi in pl.range(HIDDEN_BLOCKS):
                        z0 = zi * K_CHUNK
                        down_zero_chunk = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)
                        down_zero_chunk = pl.mul(down_zero_chunk, 0.0)
                        down_proj_tile = pl.assemble(down_proj_tile, down_zero_chunk, [0, z0])

                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                        post_norm_tile = pl.assemble(post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    for ob in pl.range(MLP_OUT_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK
                        gate_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        up_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        gate_acc = pl.mul(gate_acc, 0.0)
                        up_acc = pl.mul(up_acc, 0.0)

                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                            up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                        for dob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                            d0 = dob * K_CHUNK
                            down_prev = pl.slice(down_proj_tile, [BATCH_TILE, K_CHUNK], [0, d0])
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                            down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                            down_proj_tile = pl.assemble(down_proj_tile, down_next, [0, d0])

                    for ob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                        o0 = ob * K_CHUNK
                        down_acc = pl.add(
                            pl.slice(down_proj_tile, [BATCH_TILE, K_CHUNK], [0, o0]),
                            pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, o0]),
                        )
                        out = pl.assemble(out, pl.cast(down_acc, target_type=pl.BF16), [b0, o0])

            return out

    return Qwen3SingleLayerDecode


# ---------------------------------------------------------------------------
# Build / run helpers
# ---------------------------------------------------------------------------


def golden_qwen3_decode(tensors, params):
    import torch

    batch = BATCH
    max_seq_len = MAX_SEQ
    hidden_size = HIDDEN
    num_heads = NUM_HEADS
    num_kv_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    intermediate_size = INTERMEDIATE
    kv_hidden = num_kv_heads * head_dim
    q_per_kv = num_heads // num_kv_heads
    eps = EPS
    attn_scale = ATTN_SCALE

    hidden_states = tensors["hidden_states"]
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"]
    v_cache = tensors["v_cache"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    out = tensors["out"]

    q_proj = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.bfloat16)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.bfloat16)

    for b0 in range(0, batch, BATCH_TILE):
        valid_batch = min(BATCH_TILE, batch - b0)
        b_slice = b0 + valid_batch

        x_tile = hidden_states[b0:b_slice, :].float()

        sq_sum = torch.zeros(valid_batch, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, K_CHUNK):
            k_chunk_size = min(K_CHUNK, hidden_size - k0)
            x_chunk = x_tile[:, k0:k0+k_chunk_size]
            sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)

        inv_rms = torch.rsqrt(sq_sum / hidden_size + eps)
        normed = x_tile * inv_rms * input_rms_weight.float()

        for q0 in range(0, hidden_size, Q_OUT_CHUNK):
            q_chunk_size = min(Q_OUT_CHUNK, hidden_size - q0)
            q_acc = torch.zeros(valid_batch, q_chunk_size, dtype=torch.float32)
            for k0 in range(0, hidden_size, K_CHUNK):
                k_chunk_size = min(K_CHUNK, hidden_size - k0)
                normed_chunk = normed[:, k0:k0+k_chunk_size].bfloat16()
                wq_chunk = wq[k0:k0+k_chunk_size, q0:q0+q_chunk_size]
                q_acc = q_acc + torch.matmul(normed_chunk, wq_chunk).float()
            q_proj[b0:b_slice, q0:q0+q_chunk_size] = q_acc.bfloat16()

        for kv0 in range(0, kv_hidden, KV_OUT_CHUNK):
            kv_chunk_size = min(KV_OUT_CHUNK, kv_hidden - kv0)
            k_acc = torch.zeros(valid_batch, kv_chunk_size, dtype=torch.float32)
            v_acc = torch.zeros(valid_batch, kv_chunk_size, dtype=torch.float32)
            for k0 in range(0, hidden_size, K_CHUNK):
                k_chunk_size = min(K_CHUNK, hidden_size - k0)
                normed_chunk = normed[:, k0:k0+k_chunk_size].bfloat16()
                wk_chunk = wk[k0:k0+k_chunk_size, kv0:kv0+kv_chunk_size]
                wv_chunk = wv[k0:k0+k_chunk_size, kv0:kv0+kv_chunk_size]
                k_acc = k_acc + torch.matmul(normed_chunk, wk_chunk).float()
                v_acc = v_acc + torch.matmul(normed_chunk, wv_chunk).float()
            k_proj[b0:b_slice, kv0:kv0+kv_chunk_size] = k_acc.bfloat16()
            v_proj[b0:b_slice, kv0:kv0+kv_chunk_size] = v_acc.bfloat16()

    attn_out = torch.zeros(batch, hidden_size, dtype=torch.float32)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1

        cos_row = rope_cos[pos, :].float()
        sin_row = rope_sin[pos, :].float()
        cos_lo = cos_row[:head_dim // 2]
        cos_hi = cos_row[head_dim // 2:]
        sin_lo = sin_row[:head_dim // 2]
        sin_hi = sin_row[head_dim // 2:]

        k_group = torch.zeros(num_kv_heads, head_dim, dtype=torch.float32)
        for ki in range(num_kv_heads):
            kv_col = ki * head_dim
            k_group[ki, :] = k_proj[b, kv_col:kv_col+head_dim].float()

        k_lo = k_group[:, :head_dim // 2]
        k_hi = k_group[:, head_dim // 2:]
        k_rot_lo = k_lo * cos_lo.unsqueeze(0) - k_hi * sin_lo.unsqueeze(0)
        k_rot_hi = k_hi * cos_hi.unsqueeze(0) + k_lo * sin_hi.unsqueeze(0)
        k_rot = torch.cat([k_rot_lo, k_rot_hi], dim=-1)

        for ki in range(num_kv_heads):
            cache_row = b * num_kv_heads * max_seq_len + ki * max_seq_len + pos
            k_cache[cache_row, :] = k_rot[ki, :].bfloat16()
            v_cache[cache_row, :] = v_proj[b, ki * head_dim:(ki+1) * head_dim]

        attn_row = torch.zeros(1, hidden_size, dtype=torch.float32)

        q_groups = q_per_kv // Q_HEAD_BATCH
        total_q_groups = num_kv_heads * q_groups

        for gi in range(total_q_groups):
            kvh = gi // q_groups
            qg = gi - kvh * q_groups
            q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH

            q_group = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
            for qi in range(Q_HEAD_BATCH):
                q_col = (q_base + qi) * head_dim
                q_group[qi, :] = q_proj[b, q_col:q_col+head_dim].float()

            q_lo = q_group[:, :head_dim // 2]
            q_hi = q_group[:, head_dim // 2:]
            q_rot_lo = q_lo * cos_lo.unsqueeze(0) - q_hi * sin_lo.unsqueeze(0)
            q_rot_hi = q_hi * cos_hi.unsqueeze(0) + q_lo * sin_hi.unsqueeze(0)
            q_rot = torch.cat([q_rot_lo, q_rot_hi], dim=-1)
            q_rot_bf16 = q_rot.bfloat16().float()

            oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
            li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
            mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

            ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

            for sb in range(ctx_blocks):
                s0 = sb * SEQ_TILE
                valid_len = min(SEQ_TILE, ctx_len - s0)
                cache_row0 = b * num_kv_heads * max_seq_len + kvh * max_seq_len + s0

                k_tile = k_cache[cache_row0:cache_row0+valid_len, :].float()
                v_tile = v_cache[cache_row0:cache_row0+valid_len, :].float()

                q_padded = torch.zeros(Q_HEAD_PAD, head_dim, dtype=torch.bfloat16)
                q_padded[:Q_HEAD_BATCH, :] = q_rot.bfloat16()

                raw_scores_pad = torch.matmul(q_padded.float(), k_tile.transpose(0, 1))

                scores_valid = raw_scores_pad[:Q_HEAD_BATCH, :valid_len]

                scores_padded = torch.full((Q_HEAD_BATCH, SEQ_TILE), float('-inf'), dtype=torch.float32)
                scores_padded[:, :valid_len] = scores_valid

                scores = scores_padded * attn_scale

                cur_mi = scores.max(dim=-1, keepdim=True)[0]
                exp_scores = torch.exp(scores - cur_mi)

                exp_scores_bf16 = exp_scores.bfloat16()
                exp_scores_fp32 = exp_scores_bf16.float()
                cur_li = exp_scores_fp32.sum(dim=-1, keepdim=True)

                exp_padded = torch.zeros(Q_HEAD_PAD, SEQ_TILE, dtype=torch.bfloat16)
                exp_padded[:Q_HEAD_BATCH, :] = exp_scores_bf16

                oi_tmp_pad = torch.matmul(exp_padded.float(), v_tile)
                oi_tmp = oi_tmp_pad[:Q_HEAD_BATCH, :]

                if sb == 0:
                    oi = oi_tmp
                    li = cur_li
                    mi = cur_mi
                else:
                    mi_new = torch.maximum(mi, cur_mi)
                    alpha = torch.exp(mi - mi_new)
                    beta = torch.exp(cur_mi - mi_new)
                    li = alpha * li + beta * cur_li
                    oi = alpha * oi + beta * oi_tmp
                    mi = mi_new

            ctx = oi / li
            ctx_flat = ctx.reshape(1, Q_HEAD_BATCH * head_dim)
            attn_row[0, q_base * head_dim:(q_base + Q_HEAD_BATCH) * head_dim] = ctx_flat

        attn_out[b, :] = attn_row[0, :]

    for b0 in range(0, batch, BATCH_TILE):
        valid_batch = min(BATCH_TILE, batch - b0)
        b_slice = b0 + valid_batch

        resid1_tile = torch.zeros(valid_batch, hidden_size, dtype=torch.float32)

        for o0 in range(0, hidden_size, Q_OUT_CHUNK):
            o_chunk_size = min(Q_OUT_CHUNK, hidden_size - o0)
            o_acc = torch.zeros(valid_batch, o_chunk_size, dtype=torch.float32)
            for k0 in range(0, hidden_size, K_CHUNK):
                k_chunk_size = min(K_CHUNK, hidden_size - k0)
                a_chunk = attn_out[b0:b_slice, k0:k0+k_chunk_size].bfloat16()
                w_chunk = wo[k0:k0+k_chunk_size, o0:o0+o_chunk_size]
                o_acc = o_acc + torch.matmul(a_chunk, w_chunk).float()
            resid = hidden_states[b0:b_slice, o0:o0+o_chunk_size].float()
            resid1_tile[:, o0:o0+o_chunk_size] = o_acc + resid

        sq_sum_post = torch.zeros(valid_batch, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, K_CHUNK):
            k_chunk_size = min(K_CHUNK, hidden_size - k0)
            x_chunk = resid1_tile[:, k0:k0+k_chunk_size]
            sq_sum_post = sq_sum_post + (x_chunk ** 2).sum(dim=-1, keepdim=True)

        inv_rms_post = torch.rsqrt(sq_sum_post / hidden_size + eps)
        post_norm_tile = resid1_tile * inv_rms_post * post_rms_weight.float()
        post_norm_bf16 = post_norm_tile.bfloat16()

        down_proj_tile = torch.zeros(valid_batch, hidden_size, dtype=torch.float32)

        for o0 in range(0, intermediate_size, MLP_OUT_CHUNK):
            o_chunk_size = min(MLP_OUT_CHUNK, intermediate_size - o0)
            gate_acc = torch.zeros(valid_batch, o_chunk_size, dtype=torch.float32)
            up_acc = torch.zeros(valid_batch, o_chunk_size, dtype=torch.float32)

            for k0 in range(0, hidden_size, K_CHUNK):
                k_chunk_size = min(K_CHUNK, hidden_size - k0)
                post_chunk = post_norm_bf16[:, k0:k0+k_chunk_size]
                wg = w_gate[k0:k0+k_chunk_size, o0:o0+o_chunk_size]
                wu = w_up[k0:k0+k_chunk_size, o0:o0+o_chunk_size]
                gate_acc = gate_acc + torch.matmul(post_chunk, wg).float()
                up_acc = up_acc + torch.matmul(post_chunk, wu).float()

            sigmoid = torch.sigmoid(gate_acc)
            mlp_chunk = gate_acc * sigmoid * up_acc
            mlp_chunk_bf16 = mlp_chunk.bfloat16().float()

            for d0 in range(0, hidden_size, K_CHUNK):
                d_chunk_size = min(K_CHUNK, hidden_size - d0)
                down_prev = down_proj_tile[:, d0:d0+d_chunk_size]
                w_down_chunk = w_down[o0:o0+o_chunk_size, d0:d0+d_chunk_size]
                down_proj_tile[:, d0:d0+d_chunk_size] = down_prev + torch.matmul(mlp_chunk_bf16, w_down_chunk).float()

        final_out = down_proj_tile + resid1_tile
        out[b0:b_slice, :] = final_out.bfloat16()


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
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=torch.randn),
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
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
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
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_qwen3_single_layer_decode_program(
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
        golden=golden_qwen3_decode,
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
    if not result.passed and result.error:
        print(f"Result: {result.error}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
    )
    if not result.passed:
        raise SystemExit(1)
