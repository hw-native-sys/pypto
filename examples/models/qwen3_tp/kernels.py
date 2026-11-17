# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# The original portions below are licensed under Apache-2.0.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from nalinaly/vllm-ascend e7cb166290dfcbf2f997aa67f01b323be643fe0e.
# Modified for standalone PyPTO: separate chip entries and host-coordinated SHMEM.
# See LICENSE in this directory for the original license.

"""Qwen3-14B TP2/TP4/TP8 compute kernels; no vLLM runtime dependency."""

import os

import pypto.language as pl
import pypto.language.distributed as pld

TP_WORLD = int(os.environ.get("WORLD_SIZE", "2"))
if TP_WORLD not in (2, 4, 8):
    raise ValueError(f"Qwen3 requires WORLD_SIZE 2/4/8, got {TP_WORLD}")

HIDDEN = 5120


HIDDEN_TP = HIDDEN // TP_WORLD


KV_HIDDEN_TP = 1024 // TP_WORLD


INTER_TP = 17408 // TP_WORLD


K_CHUNK = 128 if TP_WORLD == 8 else 256


N_CHUNK = 128


TOK = 32


HIDDEN_K_BLOCKS = HIDDEN // K_CHUNK


HIDDEN_N_BLOCKS = HIDDEN // N_CHUNK


HIDDEN_TP_K_BLOCKS = HIDDEN_TP // K_CHUNK


HIDDEN_TP_N_BLOCKS = HIDDEN_TP // N_CHUNK


KV_N_BLOCKS = KV_HIDDEN_TP // N_CHUNK


INTER_K_BLOCKS = INTER_TP // K_CHUNK


INTER_N_BLOCKS = INTER_TP // N_CHUNK


@pl.jit.inline
def gemm_h_htp_step(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    for nb in pl.range(HIDDEN_TP_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HIDDEN_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_h_htp_incore(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    return gemm_h_htp_step(left, weight, out)


@pl.jit.inline
def gemm_h_kv_step(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    for nb in pl.range(KV_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HIDDEN_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_h_kv_incore(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    return gemm_h_kv_step(left, weight, out)


@pl.jit.inline
def gemm_htp_h_step(
    left: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    weight: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    for nb in pl.range(HIDDEN_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HIDDEN_TP_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_htp_h_incore(
    left: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    weight: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return gemm_htp_h_step(left, weight, out)


@pl.jit.inline
def gemm_h_inter_step(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    for nb in pl.range(INTER_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HIDDEN_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_h_inter_incore(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    return gemm_h_inter_step(left, weight, out)


@pl.jit.inline
def gemm_inter_h_step(
    left: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    weight: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    for nb in pl.range(HIDDEN_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, INTER_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_inter_h_incore(
    left: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    weight: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return gemm_inter_h_step(left, weight, out)


HEAD_DIM = 128


HALF_DIM = 64


HEADS_TP = 40 // TP_WORLD


KV_HEADS_TP = 8 // TP_WORLD


Q_PER_KV = 5


NUM_LAYERS = 40


MAX_CACHE = 128


CACHE_ROWS = NUM_LAYERS * MAX_CACHE


MAX_SEQ = 4096


VOCAB = 152064


EPS = 1e-6


HIDDEN_INV = 1.0 / float(HIDDEN)


HEAD_DIM_INV = 1.0 / float(HEAD_DIM)


ATTN_SCALE = HEAD_DIM**-0.5


RMS_CHUNK = 128


SILU_CHUNK = 128


NEG_INF = -1.0e9


@pl.jit.inline
def rms_hidden(
    x: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    # The default A2/A3 vrsqrt estimate causes measurable layer-wise drift.
    sq_sum = pl.full([1, TOK], dtype=pl.FP32, value=0.0)
    for hb in pl.range(HIDDEN // RMS_CHUNK):
        h0 = hb * RMS_CHUNK
        tile = pl.cast(pl.slice(x, [TOK, RMS_CHUNK], [0, h0]), target_type=pl.FP32)
        sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(tile, tile)), [1, TOK]))
    inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS), high_precision=True), [TOK, 1])
    for hb in pl.range(HIDDEN // RMS_CHUNK):
        h0 = hb * RMS_CHUNK
        tile = pl.cast(pl.slice(x, [TOK, RMS_CHUNK], [0, h0]), target_type=pl.FP32)
        weight = pl.slice(gamma, [1, RMS_CHUNK], [0, h0])
        normed = pl.col_expand_mul(pl.row_expand_mul(tile, inv), weight)
        y = pl.assemble(y, pl.cast(normed, target_type=pl.BF16, mode="rint"), [0, h0])
    return y


@pl.jit.inline
def q_norm_rope(
    heads: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    gamma: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[TOK, HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[TOK, HEAD_DIM], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
):
    for h in pl.range(HEADS_TP):
        h0 = h * HEAD_DIM
        qh = pl.slice(heads, [TOK, HEAD_DIM], [0, h0])
        sq = pl.reshape(pl.row_sum(pl.mul(qh, qh)), [1, TOK])
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq, HEAD_DIM_INV), EPS), high_precision=True), [TOK, 1])
        normed = pl.col_expand_mul(pl.row_expand_mul(qh, inv), gamma)
        lo = pl.slice(normed, [TOK, HALF_DIM], [0, 0])
        hi = pl.slice(normed, [TOK, HALF_DIM], [0, HALF_DIM])
        cos_lo = pl.slice(cos, [TOK, HALF_DIM], [0, 0])
        cos_hi = pl.slice(cos, [TOK, HALF_DIM], [0, HALF_DIM])
        sin_lo = pl.slice(sin, [TOK, HALF_DIM], [0, 0])
        sin_hi = pl.slice(sin, [TOK, HALF_DIM], [0, HALF_DIM])
        rot_lo = pl.sub(pl.mul(lo, cos_lo), pl.mul(hi, sin_lo))
        rot_hi = pl.add(pl.mul(hi, cos_hi), pl.mul(lo, sin_hi))
        out = pl.assemble(out, pl.cast(rot_lo, target_type=pl.BF16, mode="rint"), [0, h0])
        out = pl.assemble(out, pl.cast(rot_hi, target_type=pl.BF16, mode="rint"), [0, h0 + HALF_DIM])
    return out


@pl.jit.inline
def k_norm_rope(
    heads: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    gamma: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[TOK, HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[TOK, HEAD_DIM], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
):
    for h in pl.range(KV_HEADS_TP):
        h0 = h * HEAD_DIM
        kh = pl.slice(heads, [TOK, HEAD_DIM], [0, h0])
        sq = pl.reshape(pl.row_sum(pl.mul(kh, kh)), [1, TOK])
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq, HEAD_DIM_INV), EPS), high_precision=True), [TOK, 1])
        normed = pl.col_expand_mul(pl.row_expand_mul(kh, inv), gamma)
        lo = pl.slice(normed, [TOK, HALF_DIM], [0, 0])
        hi = pl.slice(normed, [TOK, HALF_DIM], [0, HALF_DIM])
        cos_lo = pl.slice(cos, [TOK, HALF_DIM], [0, 0])
        cos_hi = pl.slice(cos, [TOK, HALF_DIM], [0, HALF_DIM])
        sin_lo = pl.slice(sin, [TOK, HALF_DIM], [0, 0])
        sin_hi = pl.slice(sin, [TOK, HALF_DIM], [0, HALF_DIM])
        rot_lo = pl.sub(pl.mul(lo, cos_lo), pl.mul(hi, sin_lo))
        rot_hi = pl.add(pl.mul(hi, cos_hi), pl.mul(lo, sin_hi))
        out = pl.assemble(out, pl.cast(rot_lo, target_type=pl.BF16, mode="rint"), [0, h0])
        out = pl.assemble(out, pl.cast(rot_hi, target_type=pl.BF16, mode="rint"), [0, h0 + HALF_DIM])
    return out


@pl.jit.inline
def swiglu(
    gate: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    up: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.BF16]],
):
    for nb in pl.range(INTER_TP // SILU_CHUNK):
        n0 = nb * SILU_CHUNK
        gate_t = pl.slice(gate, [TOK, SILU_CHUNK], [0, n0])
        up_t = pl.slice(up, [TOK, SILU_CHUNK], [0, n0])
        sigmoid = pl.recip(pl.add(pl.exp(pl.mul(gate_t, -1.0)), 1.0))
        act = pl.mul(pl.mul(gate_t, sigmoid), up_t)
        out = pl.assemble(out, pl.cast(act, target_type=pl.BF16, mode="rint"), [0, n0])
    return out


@pl.jit.inline
def residual_add(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    delta: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    for hb in pl.range(HIDDEN // RMS_CHUNK):
        h0 = hb * RMS_CHUNK
        hid = pl.cast(pl.slice(hidden, [TOK, RMS_CHUNK], [0, h0]), target_type=pl.FP32)
        add = pl.add(hid, pl.slice(delta, [TOK, RMS_CHUNK], [0, h0]))
        out = pl.assemble(out, pl.cast(add, target_type=pl.BF16, mode="rint"), [0, h0])
    return out


@pl.jit.inline
def write_kv(
    key: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    value: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    row0: pl.Scalar[pl.INDEX],
    n_tok: pl.Scalar[pl.INDEX],
):
    for t in pl.range(TOK):
        if t < n_tok:
            k_cache = pl.assemble(k_cache, pl.slice(key, [1, KV_HIDDEN_TP], [t, 0]), [row0 + t, 0])
            v_cache = pl.assemble(v_cache, pl.slice(value, [1, KV_HIDDEN_TP], [t, 0]), [row0 + t, 0])
    return k_cache, v_cache


@pl.jit.inline
def tp_embed(
    input_ids: pl.Tensor[[TOK], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    hidden: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    n_tok: pl.Scalar[pl.INDEX],
):
    for t in pl.range(TOK):
        if t < n_tok:
            token_row = pl.cast(pl.tensor.read(input_ids, [t]), pl.INDEX)
            for k0 in pl.range(0, HIDDEN, K_CHUNK):
                hidden = pl.assemble(
                    hidden,
                    pl.slice(embed_weight, [1, K_CHUNK], [token_row, k0]),
                    [t, k0],
                )
    return hidden


@pl.jit.inline
def tp_lm_head(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    logits: pl.InOut[pl.Tensor[[TOK, VOCAB], pl.FP32]],
):
    for nb in pl.range(VOCAB // N_CHUNK):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(hidden, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [N_CHUNK, K_CHUNK], [n0, 0]),
            out_dtype=pl.FP32,
            b_trans=True,
        )
        for kb in pl.range(1, HIDDEN // K_CHUNK):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(hidden, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [N_CHUNK, K_CHUNK], [n0, k0]),
                b_trans=True,
            )
        logits = pl.assemble(logits, acc, [0, n0])
    return logits


@pl.jit.incore
def tp_embed_step(
    input_ids: pl.Tensor[[TOK], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    hidden: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    n_tok: pl.Scalar[pl.INDEX],
):
    return tp_embed(input_ids, embed_weight, hidden, n_tok)


@pl.jit.incore
def rms_hidden_step(
    x: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    return rms_hidden(x, gamma, y)


@pl.jit.incore
def tp_lm_head_step(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    logits: pl.InOut[pl.Tensor[[TOK, VOCAB], pl.FP32]],
):
    return tp_lm_head(hidden, weight, logits)


ATTN_ROWS = HEADS_TP * TOK


CACHE_TILES = MAX_CACHE // TOK


@pl.jit.inline
def _cast_htp_to_bf16(
    src: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    dst: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
):
    for h in pl.range(HEADS_TP):
        h0 = h * HEAD_DIM
        dst = pl.assemble(
            dst,
            pl.cast(pl.slice(src, [TOK, HEAD_DIM], [0, h0]), target_type=pl.BF16, mode="rint"),
            [0, h0],
        )
    return dst


@pl.jit.inline
def _cast_htp_to_fp32(
    src: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    dst: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    for h in pl.range(HEADS_TP):
        h0 = h * HEAD_DIM
        dst = pl.assemble(
            dst,
            pl.cast(pl.slice(src, [TOK, HEAD_DIM], [0, h0]), target_type=pl.FP32),
            [0, h0],
        )
    return dst


@pl.jit.inline
def _cast_kv_to_bf16(
    src: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    dst: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
):
    for h in pl.range(KV_HEADS_TP):
        h0 = h * HEAD_DIM
        dst = pl.assemble(
            dst,
            pl.cast(pl.slice(src, [TOK, HEAD_DIM], [0, h0]), target_type=pl.BF16, mode="rint"),
            [0, h0],
        )
    return dst


@pl.jit.inline
def _cast_kv_to_fp32(
    src: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    dst: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    for h in pl.range(KV_HEADS_TP):
        h0 = h * HEAD_DIM
        dst = pl.assemble(
            dst,
            pl.cast(pl.slice(src, [TOK, HEAD_DIM], [0, h0]), target_type=pl.FP32),
            [0, h0],
        )
    return dst


@pl.jit.incore
def tp_hidden_rms_step(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    xn: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    """Pure AIV: normalize hidden before the QKV cube task."""
    return rms_hidden(hidden, weight, xn)


@pl.jit.incore
def tp_qkv_gemm_step(
    xn: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    wq: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
    v_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    """Pure AIC: compute all three projections after RMS has completed."""
    q_fp = gemm_h_htp_step(xn, wq, q_fp)
    k_fp = gemm_h_kv_step(xn, wk, k_fp)
    v_fp = gemm_h_kv_step(xn, wv, v_fp)
    return q_fp, k_fp, v_fp


@pl.jit.incore
def tp_qkv_post_step(  # noqa: PLR0913 - flat device ABI
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    v_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    q_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    q_bf: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
    k_bf: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
    v_bf: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
    layer_idx: pl.Scalar[pl.INDEX],
    pos0: pl.Scalar[pl.INDEX],
    n_tok: pl.Scalar[pl.INDEX],
):
    """Pure AIV: QK norm/RoPE, V cast, and cache writeback."""
    cos = pl.slice(rope_cos, [TOK, HEAD_DIM], [pos0, 0])
    sin = pl.slice(rope_sin, [TOK, HEAD_DIM], [pos0, 0])
    q_bf = q_norm_rope(q_fp, q_norm, cos, sin, q_bf)
    k_bf = k_norm_rope(k_fp, k_norm, cos, sin, k_bf)
    v_bf = _cast_kv_to_bf16(v_fp, v_bf)
    row0 = layer_idx * MAX_CACHE + pos0
    k_cache, v_cache = write_kv(k_bf, v_bf, k_cache, v_cache, row0, n_tok)
    return k_cache, v_cache, q_bf, k_bf, v_bf


@pl.jit.incore
def tp_prefill_attention_prepare_step(
    q_bf: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    k_bf: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    v_bf: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
    v_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    """Pure AIV: materialize BF16 QKV as FP32 GM inputs for attention."""
    q_fp = _cast_htp_to_fp32(q_bf, q_fp)
    k_fp = _cast_kv_to_fp32(k_bf, k_fp)
    v_fp = _cast_kv_to_fp32(v_bf, v_fp)
    return q_fp, k_fp, v_fp


@pl.jit.incore
def tp_prefill_attention_scores_step(
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    scores: pl.InOut[pl.Tensor[[ATTN_ROWS, TOK], pl.FP32]],
):
    """Pure AIC: one raw QK^T score matrix per local query head."""
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        raw = pl.matmul(
            pl.slice(q_fp, [TOK, HEAD_DIM], [0, h * HEAD_DIM]),
            pl.slice(k_fp, [TOK, HEAD_DIM], [0, kv_h * HEAD_DIM]),
            out_dtype=pl.FP32,
            b_trans=True,
        )
        scores = pl.assemble(scores, raw, [h * TOK, 0])
    return scores


@pl.jit.incore
def tp_prefill_attention_softmax_step(
    scores: pl.Tensor[[ATTN_ROWS, TOK], pl.FP32],
    mask: pl.Tensor[[TOK, TOK], pl.FP32],
    probs: pl.InOut[pl.Tensor[[ATTN_ROWS, TOK], pl.FP32]],
):
    """Pure AIV: scale, mask, and softmax each head's scores."""
    for h in pl.range(HEADS_TP):
        score_h = pl.mul(pl.slice(scores, [TOK, TOK], [h * TOK, 0]), ATTN_SCALE)
        masked = pl.add(score_h, mask)
        shifted = pl.exp(pl.row_expand_sub(masked, pl.row_max(masked)))
        prob_h = pl.row_expand_div(shifted, pl.row_sum(shifted))
        probs = pl.assemble(probs, prob_h, [h * TOK, 0])
    return probs


@pl.jit.incore
def tp_prefill_attention_context_step(
    probs: pl.Tensor[[ATTN_ROWS, TOK], pl.FP32],
    v_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    context: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    """Pure AIC: multiply softmax probabilities by V."""
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        ctx = pl.matmul(
            pl.slice(probs, [TOK, TOK], [h * TOK, 0]),
            pl.slice(v_fp, [TOK, HEAD_DIM], [0, kv_h * HEAD_DIM]),
            out_dtype=pl.FP32,
        )
        context = pl.assemble(context, ctx, [0, h * HEAD_DIM])
    return context


@pl.jit.incore
def tp_decode_attention_prepare_step(
    q_bf: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    k_cache: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.BF16],
    v_cache: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_ctx_fp: pl.InOut[pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32]],
    v_ctx_fp: pl.InOut[pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32]],
):
    """Pure AIV: cast decode Q and the selected layer cache to FP32."""
    q_fp = _cast_htp_to_fp32(q_bf, q_fp)
    for sb in pl.range(CACHE_TILES):
        s0 = sb * TOK
        for h in pl.range(KV_HEADS_TP):
            h0 = h * HEAD_DIM
            k_ctx_fp = pl.assemble(
                k_ctx_fp,
                pl.cast(
                    pl.slice(k_cache, [TOK, HEAD_DIM], [s0, h0]),
                    target_type=pl.FP32,
                ),
                [s0, h0],
            )
            v_ctx_fp = pl.assemble(
                v_ctx_fp,
                pl.cast(
                    pl.slice(v_cache, [TOK, HEAD_DIM], [s0, h0]),
                    target_type=pl.FP32,
                ),
                [s0, h0],
            )
    return q_fp, k_ctx_fp, v_ctx_fp


@pl.jit.incore
def tp_decode_attention_scores_step(
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_ctx_fp: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32],
    scores: pl.InOut[pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32]],
):
    """Pure AIC: build decode QK^T scores in TOK-wide cache tiles."""
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        for sb in pl.range(CACHE_TILES):
            s0 = sb * TOK
            raw = pl.matmul(
                pl.slice(q_fp, [TOK, HEAD_DIM], [0, h * HEAD_DIM]),
                pl.slice(k_ctx_fp, [TOK, HEAD_DIM], [s0, kv_h * HEAD_DIM]),
                out_dtype=pl.FP32,
                b_trans=True,
            )
            scores = pl.assemble(scores, raw, [h * TOK, s0])
    return scores


@pl.jit.incore
def tp_decode_attention_softmax_step(
    scores: pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32],
    mask: pl.Tensor[[TOK, MAX_CACHE], pl.FP32],
    probs: pl.InOut[pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32]],
):
    """Pure AIV: full-cache masked softmax, matching the eager reference."""
    for h in pl.range(HEADS_TP):
        score_h = pl.mul(
            pl.slice(scores, [TOK, MAX_CACHE], [h * TOK, 0]),
            ATTN_SCALE,
        )
        masked = pl.add(score_h, mask)
        shifted = pl.exp(pl.row_expand_sub(masked, pl.row_max(masked)))
        prob_h = pl.row_expand_div(shifted, pl.row_sum(shifted))
        probs = pl.assemble(probs, prob_h, [h * TOK, 0])
    return probs


@pl.jit.incore
def tp_decode_attention_context_step(
    probs: pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32],
    v_ctx_fp: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32],
    context: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    """Pure AIC: accumulate the four cache tiles into each head context."""
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        ctx = pl.matmul(
            pl.slice(probs, [TOK, TOK], [h * TOK, 0]),
            pl.slice(v_ctx_fp, [TOK, HEAD_DIM], [0, kv_h * HEAD_DIM]),
            out_dtype=pl.FP32,
        )
        for sb in pl.range(1, CACHE_TILES):
            s0 = sb * TOK
            ctx = pl.matmul_acc(
                ctx,
                pl.slice(probs, [TOK, TOK], [h * TOK, s0]),
                pl.slice(v_ctx_fp, [TOK, HEAD_DIM], [s0, kv_h * HEAD_DIM]),
            )
        context = pl.assemble(context, ctx, [0, h * HEAD_DIM])
    return context


@pl.jit.incore
def tp_attention_cast_step(
    context: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    q_bf: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
):
    """Pure AIV: quantize attention context before the row-parallel O GEMM."""
    return _cast_htp_to_bf16(context, q_bf)


@pl.jit.incore
def tp_o_gemm_step(
    context: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    wo: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    partial: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    """Pure AIC: row-parallel O projection."""
    return gemm_htp_h_step(context, wo, partial)


@pl.jit.incore
def tp_ffn_residual_rms_step(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    post_rms: pl.Tensor[[1, HIDDEN], pl.FP32],
    hidden_out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    xn: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    """Pure AIV: attention residual followed by post-attention RMSNorm."""
    hidden_out = residual_add(hidden, reduced, hidden_out)
    xn = rms_hidden(hidden_out, post_rms, xn)
    return hidden_out, xn


@pl.jit.incore
def tp_ffn_gate_up_step(
    xn: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    gate: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
    up: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    """Pure AIC: gate and up projections."""
    gate = gemm_h_inter_step(xn, w_gate, gate)
    up = gemm_h_inter_step(xn, w_up, up)
    return gate, up


@pl.jit.incore
def tp_ffn_swiglu_step(
    gate: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    up: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    act: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.BF16]],
):
    """Pure AIV: SiLU(gate) * up and BF16 conversion."""
    return swiglu(gate, up, act)


@pl.jit.incore
def tp_ffn_down_step(
    act: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    w_down: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    partial: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    """Pure AIC: row-parallel down projection."""
    return gemm_inter_h_step(act, w_down, partial)


@pl.jit.incore
def tp_residual_tail(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    hidden_out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    """Pure AIV: finish the FFN residual."""
    return residual_add(hidden, reduced, hidden_out)


@pl.jit
def gemm_h_htp_incore_chip(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    return gemm_h_htp_incore(left, weight, out)


@pl.jit
def gemm_h_kv_incore_chip(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    return gemm_h_kv_incore(left, weight, out)


@pl.jit
def gemm_htp_h_incore_chip(
    left: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    weight: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return gemm_htp_h_incore(left, weight, out)


@pl.jit
def gemm_h_inter_incore_chip(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    return gemm_h_inter_incore(left, weight, out)


@pl.jit
def gemm_inter_h_incore_chip(
    left: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    weight: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return gemm_inter_h_incore(left, weight, out)


@pl.jit
def tp_embed_step_chip(
    input_ids: pl.Tensor[[TOK], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    hidden: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    n_tok: pl.Scalar[pl.INDEX],
):
    return tp_embed_step(input_ids, embed_weight, hidden, n_tok)


@pl.jit
def rms_hidden_step_chip(
    x: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    return rms_hidden_step(x, gamma, y)


@pl.jit
def tp_lm_head_step_chip(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    logits: pl.InOut[pl.Tensor[[TOK, VOCAB], pl.FP32]],
):
    return tp_lm_head_step(hidden, weight, logits)


@pl.jit
def tp_hidden_rms_step_chip(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    xn: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    return tp_hidden_rms_step(hidden, weight, xn)


@pl.jit
def tp_qkv_gemm_step_chip(
    xn: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    wq: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
    v_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    return tp_qkv_gemm_step(xn, wq, wk, wv, q_fp, k_fp, v_fp)


@pl.jit
def tp_qkv_post_step_chip(  # noqa: PLR0913 - flat device ABI
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    v_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    q_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    q_bf: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
    k_bf: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
    v_bf: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
    layer_idx: pl.Scalar[pl.INDEX],
    pos0: pl.Scalar[pl.INDEX],
    n_tok: pl.Scalar[pl.INDEX],
):
    return tp_qkv_post_step(
        q_fp,
        k_fp,
        v_fp,
        q_norm,
        k_norm,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        q_bf,
        k_bf,
        v_bf,
        layer_idx,
        pos0,
        n_tok,
    )


@pl.jit
def tp_prefill_attention_prepare_step_chip(
    q_bf: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    k_bf: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    v_bf: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
    v_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    return tp_prefill_attention_prepare_step(q_bf, k_bf, v_bf, q_fp, k_fp, v_fp)


@pl.jit
def tp_prefill_attention_scores_step_chip(
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    scores: pl.InOut[pl.Tensor[[ATTN_ROWS, TOK], pl.FP32]],
):
    return tp_prefill_attention_scores_step(q_fp, k_fp, scores)


@pl.jit
def tp_prefill_attention_softmax_step_chip(
    scores: pl.Tensor[[ATTN_ROWS, TOK], pl.FP32],
    mask: pl.Tensor[[TOK, TOK], pl.FP32],
    probs: pl.InOut[pl.Tensor[[ATTN_ROWS, TOK], pl.FP32]],
):
    return tp_prefill_attention_softmax_step(scores, mask, probs)


@pl.jit
def tp_prefill_attention_context_step_chip(
    probs: pl.Tensor[[ATTN_ROWS, TOK], pl.FP32],
    v_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    context: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    return tp_prefill_attention_context_step(probs, v_fp, context)


@pl.jit
def tp_decode_attention_prepare_step_chip(
    q_bf: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    k_cache: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.BF16],
    v_cache: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_ctx_fp: pl.InOut[pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32]],
    v_ctx_fp: pl.InOut[pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32]],
):
    return tp_decode_attention_prepare_step(q_bf, k_cache, v_cache, q_fp, k_ctx_fp, v_ctx_fp)


@pl.jit
def tp_decode_attention_scores_step_chip(
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_ctx_fp: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32],
    scores: pl.InOut[pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32]],
):
    return tp_decode_attention_scores_step(q_fp, k_ctx_fp, scores)


@pl.jit
def tp_decode_attention_softmax_step_chip(
    scores: pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32],
    mask: pl.Tensor[[TOK, MAX_CACHE], pl.FP32],
    probs: pl.InOut[pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32]],
):
    return tp_decode_attention_softmax_step(scores, mask, probs)


@pl.jit
def tp_decode_attention_context_step_chip(
    probs: pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32],
    v_ctx_fp: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32],
    context: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    return tp_decode_attention_context_step(probs, v_ctx_fp, context)


@pl.jit
def tp_attention_cast_step_chip(
    context: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    q_bf: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
):
    return tp_attention_cast_step(context, q_bf)


@pl.jit
def tp_o_gemm_step_chip(
    context: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    wo: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    partial: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return tp_o_gemm_step(context, wo, partial)


@pl.jit
def tp_ffn_residual_rms_step_chip(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    post_rms: pl.Tensor[[1, HIDDEN], pl.FP32],
    hidden_out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    xn: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    return tp_ffn_residual_rms_step(hidden, reduced, post_rms, hidden_out, xn)


@pl.jit
def tp_ffn_gate_up_step_chip(
    xn: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    gate: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
    up: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    return tp_ffn_gate_up_step(xn, w_gate, w_up, gate, up)


@pl.jit
def tp_ffn_swiglu_step_chip(
    gate: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    up: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    act: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.BF16]],
):
    return tp_ffn_swiglu_step(gate, up, act)


@pl.jit
def tp_ffn_down_step_chip(
    act: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    w_down: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    partial: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return tp_ffn_down_step(act, w_down, partial)


@pl.jit
def tp_ffn_swiglu_down_step_chip(
    gate: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    up: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    act: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.BF16]],
    w_down: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    partial: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    """Schedule SwiGLU and Down together, preserving the BF16 intermediate."""
    act = tp_ffn_swiglu_step(gate, up, act)
    partial = tp_ffn_down_step(act, w_down, partial)
    return act, partial


@pl.jit
def tp_ffn_step_chip(  # noqa: PLR0913 - explicit caller-owned intermediates preserve traceability
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    post_rms: pl.Tensor[[1, HIDDEN], pl.FP32],
    residual: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    xn: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    w_gate: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    gate: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
    up: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
    act: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.BF16]],
    w_down: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    partial: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    """Schedule all four local FFN tasks, preserving intermediate storage."""
    residual, xn = tp_ffn_residual_rms_step(hidden, reduced, post_rms, residual, xn)
    gate, up = tp_ffn_gate_up_step(xn, w_gate, w_up, gate, up)
    act = tp_ffn_swiglu_step(gate, up, act)
    partial = tp_ffn_down_step(act, w_down, partial)
    return residual, xn, gate, up, act, partial


@pl.jit
def tp_residual_tail_chip(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    hidden_out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    return tp_residual_tail(hidden, reduced, hidden_out)


@pl.jit.incore
def publish_step(
    inp: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    data: pl.InOut[pld.DistributedTensor[[TOK, HIDDEN], pl.FP32]],
):
    for row in pl.range(TOK):
        data = pl.store(pl.load(inp, [row, 0], [1, HIDDEN]), [row, 0], data)
    return data


@pl.jit
def publish_chip(
    inp: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    data: pl.InOut[pld.DistributedTensor[[TOK, HIDDEN], pl.FP32]],
):
    return publish_step(inp, data)


@pl.jit.incore
def allreduce_step(
    data: pld.DistributedTensor[[TOK, HIDDEN], pl.FP32],
    out: pl.Out[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    ctx = pld.get_comm_ctx(data)
    nranks = pld.nranks(ctx)
    first_peer = pld.rank(ctx) * 0
    for row in pl.range(TOK):
        # Use identical rank order everywhere to avoid rank-dependent FP32 sums.
        summed = pld.tile.remote_load(data, peer=first_peer, offsets=[row, 0], shape=[1, HIDDEN])
        for peer in pl.range(1, nranks):
            remote = pld.tile.remote_load(data, peer=peer, offsets=[row, 0], shape=[1, HIDDEN])
            summed = pl.add(summed, remote)
        out = pl.store(summed, [row, 0], out)
    return out


@pl.jit
def allreduce_chip(
    data: pld.DistributedTensor[[TOK, HIDDEN], pl.FP32],
    out: pl.Out[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return allreduce_step(data, out)
