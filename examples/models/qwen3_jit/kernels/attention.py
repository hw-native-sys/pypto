# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scope 2 attention kernels.

This file demonstrates the cross-file utility pattern for the attention
section of Qwen3-32B decode. Currently implements one representative
sub-stage — RoPE rotation + K/V cache writes + Q pad — as a single
``@pl.jit.inline`` kernel.

The full grouped-query attention (QK matmul, softmax, SV matmul, online
softmax accumulation) follows the same composition pattern; each stage
becomes another ``@pl.jit.inline`` (or ``@pl.jit.incore``) kernel called
from the entry's per-batch / per-group loops. See ``qwen3_decode.py`` and
the upstream monolithic reference for the full structure.
"""

import pypto.language as pl

from ..config import (
    BATCH,
    HALF_DIM,
    HEAD_DIM,
    MAX_SEQ,
    NUM_KV_HEADS,
    Q_HEAD_BATCH,
    Q_HEAD_PAD,
    Q_PER_KV,
    TOTAL_Q_GROUPS,
)


@pl.jit.inline
def rope_kv_cache_update(
    q_proj: pl.Tensor,
    k_proj: pl.Tensor,
    v_proj: pl.Tensor,
    seq_lens: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Out[pl.Tensor],
    v_cache: pl.Out[pl.Tensor],
    all_q_padded: pl.Out[pl.Tensor],
):
    """Per-batch: rotate K/V via RoPE, write to KV cache, rotate-and-pad Q."""
    for b in pl.parallel(BATCH):
        ctx_len = pl.read(seq_lens, [b])
        pos = ctx_len - 1

        # Keep the row dim explicit: `rope_cos[pos, ...]` now rank-reduces to 1D
        # under numpy-style indexing, but col_expand_mul needs a 2D [1, N] operand.
        cos_lo = rope_cos[pos : pos + 1, 0:HALF_DIM]
        cos_hi = rope_cos[pos : pos + 1, HALF_DIM:HEAD_DIM]
        sin_lo = rope_sin[pos : pos + 1, 0:HALF_DIM]
        sin_hi = rope_sin[pos : pos + 1, HALF_DIM:HEAD_DIM]

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
            for ki in pl.range(NUM_KV_HEADS):
                kv_col = ki * HEAD_DIM
                k_lo = k_proj[b : b + 1, kv_col : kv_col + HALF_DIM]
                k_hi = k_proj[b : b + 1, kv_col + HALF_DIM : kv_col + HEAD_DIM]

                rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))

                cache_row = b * NUM_KV_HEADS * MAX_SEQ + ki * MAX_SEQ + pos
                k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_row, HALF_DIM])

                v_row_bf16 = pl.cast(
                    v_proj[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM], target_type=pl.BF16
                )
                v_cache = pl.assemble(v_cache, v_row_bf16, [cache_row, 0])

                # Q rotate + pad
                q_base = ki * Q_PER_KV
                q_block = pl.reshape(
                    q_proj[b : b + 1, q_base * HEAD_DIM : (q_base + Q_HEAD_BATCH) * HEAD_DIM],
                    [Q_HEAD_BATCH, HEAD_DIM],
                )

                q_lo = q_block[:, 0:HALF_DIM]
                q_hi = q_block[:, HALF_DIM:HEAD_DIM]

                q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))

                q_pad_row0 = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD
                all_q_padded = pl.assemble(
                    all_q_padded, pl.cast(q_rot_lo, target_type=pl.BF16), [q_pad_row0, 0]
                )
                all_q_padded = pl.assemble(
                    all_q_padded, pl.cast(q_rot_hi, target_type=pl.BF16), [q_pad_row0, HALF_DIM]
                )

                q_pad_zero = pl.cast(
                    pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),
                    target_type=pl.BF16,
                )
                all_q_padded = pl.assemble(all_q_padded, q_pad_zero, [q_pad_row0 + Q_HEAD_BATCH, 0])
    # All three Out params are rebound via pl.assemble. Returning only k_cache
    # is intentional: pl.Out parameters share memory with the caller, so the
    # in-place writes to v_cache and all_q_padded persist across the inline
    # splice — only one of the three is returned as the SSA-name handle.
    return k_cache
