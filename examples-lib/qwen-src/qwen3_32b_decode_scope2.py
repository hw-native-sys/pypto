# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 2 — RoPE + KV cache update + grouped-query attention.

Standalone test for the attention scope of the Qwen3-32B decode layer,
with parameters aligned to qwen3_32b_decode_tilelet.py.

Valid_shape handling aligned to pypto's kernel_softmax_prepare_unaligned approach:
  - K/V tiles are loaded as full SEQ_TILE blocks without valid_shape.
  - valid_shape + fillpad is applied only on the QK scores before softmax.
  - row_sum uses BF16 round-trip precision (matching SV matmul weights).

For each batch element:
  1. Apply RoPE to K projections (all KV heads at once) and store to cache.
  2. Copy V projections directly to cache.
  3. For each Q-head group:
     a. Gather Q heads and apply RoPE.
     b. Online flash-attention over the KV cache (up to ctx_len tokens).
     c. Write normalised attention output.

Hardware TILELET / TILE sizing (at default HEAD_DIM=128):
  * K RoPE half-vectors [NUM_KV_HEADS, HEAD_DIM//2] FP32 = [8,64]*4 = 2 KB = MAX
  * Q/attention group   [Q_HEAD_BATCH, HEAD_DIM]     FP32 = [4,128]*4 = 2 KB = MAX
  * Attention K tile    [SEQ_TILE, HEAD_DIM]          BF16 = [64,128]*2 = 16 KB = MAX

Input projections are BF16; cos/sin tables are FP32; KV caches are BF16.
Output attention is FP32.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128

# Tiling constants (aligned to qwen3_32b_decode_tilelet).
Q_HEAD_BATCH = 8        # Q heads batched per attention group
Q_HEAD_PAD = 16         # padded Q rows for cube fractal alignment
SEQ_TILE = 64           # sequence tile for attention loop


def build_decode_attention_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    q_per_kv = num_heads // num_kv_heads
    cache_rows = batch * num_kv_heads * max_seq
    half_dim = head_dim // 2
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)

    @pl.program
    class DecodeAttentionProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def decode_attention(
            self,
            q_proj: pl.Tensor[[batch, hidden], pl.BF16],
            k_proj: pl.Tensor[[batch, kv_hidden], pl.BF16],
            v_proj: pl.Tensor[[batch, kv_hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            attn_out: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, hidden], pl.FP32]:
            for b in pl.parallel(batch):
                ctx_len = pl.tensor.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                k_group = pl.create_tensor([num_kv_heads, head_dim], dtype=pl.FP32)
                with pl.incore():
                    # Stage 1a: gather all KV heads into a shared tensor buffer.
                    for ki in pl.range(num_kv_heads):
                        kv_col = ki * head_dim
                        k_group = pl.assemble(
                            k_group,
                            pl.cast(pl.slice(k_proj, [1, head_dim], [b, kv_col]),
                                    target_type=pl.FP32),
                            [ki, 0],
                        )

                k_rot_tensor = pl.create_tensor([num_kv_heads, head_dim], dtype=pl.FP32)
                with pl.incore():
                    # Stage 1b: rotate K halves and assemble to GM (no concat).
                    k_lo = pl.slice(k_group, [num_kv_heads, half_dim], [0, 0])
                    k_hi = pl.slice(k_group, [num_kv_heads, half_dim],
                                    [0, half_dim])
                    rot_lo = pl.sub(
                        pl.col_expand_mul(k_lo, cos_lo),
                        pl.col_expand_mul(k_hi, sin_lo),
                    )
                    rot_hi = pl.add(
                        pl.col_expand_mul(k_hi, cos_hi),
                        pl.col_expand_mul(k_lo, sin_hi),
                    )
                    k_rot_tensor = pl.assemble(k_rot_tensor, rot_lo, [0, 0])
                    k_rot_tensor = pl.assemble(k_rot_tensor, rot_hi, [0, half_dim])

                with pl.incore():
                    # Stage 1c: update the caches from rotated K tensor.
                    for ki in pl.range(num_kv_heads):
                        cache_row = b * num_kv_heads * max_seq + ki * max_seq + pos
                        k_cache = pl.assemble(
                            k_cache,
                            pl.cast(pl.slice(k_rot_tensor, [1, head_dim], [ki, 0]),
                                    target_type=pl.BF16),
                            [cache_row, 0],
                        )
                        v_cache = pl.assemble(
                            v_cache,
                            pl.slice(v_proj, [1, head_dim], [b, ki * head_dim]),
                            [cache_row, 0],
                        )

                attn_row = pl.create_tensor([1, hidden], dtype=pl.FP32)

                # Manually split decode attention into smaller incore stages so
                # each outlined kernel has a single cross-core payload size.
                for gi in pl.parallel(0, total_q_groups, 1):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH

                    q_group = pl.create_tensor([Q_HEAD_BATCH, head_dim], dtype=pl.FP32)
                    with pl.incore():
                        # Stage 2a: gather the Q-head group into a tensor buffer.
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * head_dim
                            q_group = pl.assemble(
                                q_group,
                                pl.cast(pl.slice(q_proj, [1, head_dim], [b, q_col]),
                                        target_type=pl.FP32),
                                [qi, 0],
                            )

                    q_rot_bf16 = pl.create_tensor([Q_HEAD_BATCH, head_dim], dtype=pl.BF16)
                    with pl.incore():
                        # Stage 2b: apply RoPE halves, cast and assemble to GM (no concat).
                        q_lo = pl.slice(q_group, [Q_HEAD_BATCH, half_dim], [0, 0])
                        q_hi = pl.slice(q_group, [Q_HEAD_BATCH, half_dim],
                                        [0, half_dim])
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
                        q_rot_bf16 = pl.assemble(q_rot_bf16, q_rot_hi_bf16, [0, half_dim])

                    with pl.incore():
                        oi = pl.full([Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0)
                        li_flat = pl.full([1, Q_HEAD_BATCH], dtype=pl.FP32, value=0.0)
                        li = pl.reshape(li_flat, [Q_HEAD_BATCH, 1])
                        mi_flat = pl.full([1, Q_HEAD_BATCH], dtype=pl.FP32, value=0.0)
                        mi = pl.reshape(mi_flat, [Q_HEAD_BATCH, 1])

                    # Pad Q for cube fractal alignment.
                    q_padded = pl.create_tensor([Q_HEAD_PAD, head_dim], dtype=pl.BF16)
                    with pl.incore():
                        q_bf16_tile = pl.slice(q_rot_bf16, [Q_HEAD_BATCH, head_dim], [0, 0])
                        q_padded = pl.assemble(q_padded, q_bf16_tile, [0, 0])

                    for sb in pl.range(ctx_blocks):
                        s0 = sb * SEQ_TILE
                        valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                        cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0

                        raw_scores_pad = pl.create_tensor([Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                        with pl.incore():
                            # QK matmul: padded Q × K^T.
                            k_tile = pl.slice(
                                k_cache,
                                [SEQ_TILE, head_dim],
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
                            scores = pl.mul(scores_padded, attn_scale)
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

                        oi_tmp_pad = pl.create_tensor([Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                        with pl.incore():
                            # SV matmul: padded exp_scores × V.
                            v_tile = pl.slice(
                                v_cache,
                                [SEQ_TILE, head_dim],
                                [cache_row0, 0],
                            )
                            oi_tmp_pad = pl.matmul(exp_padded, v_tile, out_dtype=pl.FP32)

                        with pl.incore():
                            # Slice valid rows from padded SV result.
                            oi_tmp = pl.slice(oi_tmp_pad, [Q_HEAD_BATCH, head_dim], [0, 0])
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
                        ctx_flat = pl.reshape(ctx, [1, Q_HEAD_BATCH * head_dim])
                        attn_row = pl.assemble(
                            attn_row, ctx_flat, [0, q_base * head_dim],
                        )

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            return attn_out

    return DecodeAttentionProgram


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from pypto.runtime import TensorSpec

    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    cache_rows = batch * num_kv_heads * max_seq

    seq_lens_data = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    return [
        TensorSpec("q_proj", [batch, hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("k_proj", [batch, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("v_proj", [batch, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens_data),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("attn_out", [batch, hidden], torch.float32, is_output=True),
    ]


def golden_decode_attention(tensors, params):
    """PyTorch reference matching kernel BF16 precision path.

    Simulates the kernel's tiled online-softmax with BF16 matmuls:
      - Q cast to BF16 after RoPE (matching kernel QK matmul input).
      - QK/SV matmuls use BF16 inputs with FP32 accumulation.
      - BF16 round-trip on exp_scores before row_sum (matching kernel).
      - Full SEQ_TILE K/V loads with fillpad masking on scores.
    """
    import math

    import torch

    q_proj = tensors["q_proj"].float()
    k_proj = tensors["k_proj"].float()
    v_proj = tensors["v_proj"]            # keep BF16 for cache writes
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()  # BF16, cloned to avoid side effects
    v_cache = tensors["v_cache"].clone()

    batch = q_proj.shape[0]
    hidden = q_proj.shape[1]
    kv_hidden = k_proj.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    attn_out = torch.zeros(batch, hidden, dtype=torch.float32)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        # K RoPE: all KV heads together.
        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_lo, k_hi = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [
                k_lo * cos_lo - k_hi * sin_lo,
                k_hi * cos_hi + k_lo * sin_hi,
            ],
            dim=-1,
        )

        # Update caches.
        for ki in range(num_kv_heads):
            cr = b * num_kv_heads * max_seq + ki * max_seq + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim]

        # Q RoPE: all Q heads together.
        q_heads = q_proj[b].view(num_heads, head_dim)
        q_lo, q_hi = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [
                q_lo * cos_lo - q_hi * sin_lo,
                q_hi * cos_hi + q_lo * sin_hi,
            ],
            dim=-1,
        )

        # Grouped-query attention (tiled online softmax, matching kernel BF16 path).
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp = q_rot[q_base : q_base + Q_HEAD_BATCH, :]
                # Match kernel: Q cast to BF16 for QK matmul.
                q_grp_bf16 = q_grp.to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cb = b * num_kv_heads * max_seq + kvh * max_seq + s0

                    # Load full SEQ_TILE K/V tiles as BF16 (matching kernel).
                    k_tile = k_cache[cb : cb + SEQ_TILE, :]
                    v_tile = v_cache[cb : cb + SEQ_TILE, :]

                    # QK matmul: BF16 * BF16 → FP32.
                    raw_scores = q_grp_bf16.float() @ k_tile.float().T

                    # Fillpad invalid positions before scale.
                    if valid_len < SEQ_TILE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    # Online softmax: row_max → exp → BF16 round-trip → row_sum.
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                    # SV matmul: BF16 * BF16 → FP32.
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
                for qi in range(Q_HEAD_BATCH):
                    qh = q_base + qi
                    attn_out[b, qh * head_dim : (qh + 1) * head_dim] = ctx[qi]

    tensors["attn_out"][:] = attn_out


def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    platform: str = "a5",
    device_id: int = 0,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_decode_attention_program(
        batch=batch,
        max_seq=max_seq,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq=max_seq,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_decode_attention,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).\n")
        print(result.error)
    elif not result.passed and result.error:
        print(f"Result: {result.error}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
    )
    if not result.passed:
        raise SystemExit(1)
