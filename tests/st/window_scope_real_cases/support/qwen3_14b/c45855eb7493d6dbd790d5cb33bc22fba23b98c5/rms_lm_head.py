# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa
"""Qwen3-14B final RMSNorm and LM head projection."""

# pyright: reportUndefinedVariable=false

import pypto.language as pl
from config import (
    BATCH,
    BATCH_TILE,
    EPS,
    FINAL_RMS_K_CHUNK,
    HIDDEN,
    HIDDEN_INV,
    LM_HEAD_K_CHUNK,
    USER_BATCH_DYN,
    VOCAB,
    VOCAB_CHUNK,
)


@pl.jit.inline
def rms_lm_head(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    user_batch = pl.tensor.dim(seq_lens, 0)

    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_sq_k0 = kb * FINAL_RMS_K_CHUNK
                final_sq_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_sq_k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(final_sq_chunk, final_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_final = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )

            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_norm_k0 = kb * FINAL_RMS_K_CHUNK
                final_hidden_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_norm_k0]),
                    target_type=pl.FP32,
                )
                final_gamma = pl.slice(final_norm_weight, [1, FINAL_RMS_K_CHUNK], [0, final_norm_k0])
                final_normed_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(final_hidden_chunk, inv_rms_final),
                    final_gamma,
                )
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(final_normed_chunk, target_type=pl.BF16),
                    [b0, final_norm_k0],
                )

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        lm_valid_rows = pl.min(BATCH_TILE, user_batch - b0)
        for ob in pl.parallel(VOCAB // VOCAB_CHUNK):
            lm_o0 = ob * VOCAB_CHUNK
            lm_acc_gm = pl.create_tensor([BATCH_TILE, VOCAB_CHUNK], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head"):
                lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, 0])
                lm_weight_chunk = pl.slice(lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [lm_o0, 0])
                lm_acc = pl.matmul(lm_hidden_chunk, lm_weight_chunk, out_dtype=pl.FP32, b_trans=True)
                for kb in pl.range(1, HIDDEN // LM_HEAD_K_CHUNK):
                    lm_k0 = kb * LM_HEAD_K_CHUNK
                    lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, lm_k0])
                    lm_weight_chunk = pl.slice(
                        lm_head_weight,
                        [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                        [lm_o0, lm_k0],
                    )
                    lm_acc = pl.matmul_acc(lm_acc, lm_hidden_chunk, lm_weight_chunk, b_trans=True)
                lm_acc_gm = pl.assemble(lm_acc_gm, lm_acc, [0, 0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_store"):
                lm_acc_chunk = pl.slice(lm_acc_gm, [BATCH_TILE, VOCAB_CHUNK], [0, 0])
                lm_acc_trimmed = pl.slice(
                    lm_acc_chunk,
                    [BATCH_TILE, VOCAB_CHUNK],
                    [0, 0],
                    valid_shape=[lm_valid_rows, VOCAB_CHUNK],
                )
                out = pl.assemble(out, lm_acc_trimmed, [b0, lm_o0])

    return out


@pl.jit.inline
def rms_only(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    """Variant of rms_lm_head that performs the final RMSNorm but skips the
    LM-head matmul entirely. ``out`` is returned untouched (the harness
    zero-inits it), and ``lm_head_weight`` is accepted but unused so the
    function signature stays interchangeable with ``rms_lm_head``.

    TODO: ``final_normed`` is written but never read or returned (the LM-head
    matmul that consumes it is gone in this skip variant). Each pl.assemble
    is a GM store with side effects, so the pypto JIT should preserve the
    RMSNorm loop, but the value is dead-data from the IR's perspective and
    a future DCE pass could elide it (Gemini review on PR #387). If the
    measured RMSNorm cost collapses to ~0 in perf traces, force ``final_normed``
    live by writing a small slice (e.g. ``out[:, :HIDDEN]`` cast to FP32) and
    mirroring the same dummy slice in ``golden_decode_layer_no_lm_head``."""
    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_sq_k0 = kb * FINAL_RMS_K_CHUNK
                final_sq_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_sq_k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(final_sq_chunk, final_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_final = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )

            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_norm_k0 = kb * FINAL_RMS_K_CHUNK
                final_hidden_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_norm_k0]),
                    target_type=pl.FP32,
                )
                final_gamma = pl.slice(final_norm_weight, [1, FINAL_RMS_K_CHUNK], [0, final_norm_k0])
                final_normed_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(final_hidden_chunk, inv_rms_final),
                    final_gamma,
                )
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(final_normed_chunk, target_type=pl.BF16),
                    [b0, final_norm_k0],
                )

    return out


@pl.jit.inline
def rms_lm_head_single_chunk(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    """Variant of rms_lm_head that runs only the first ``VOCAB_CHUNK`` of the
    LM-head matmul (one outer ``ob`` iteration). Rows past ``VOCAB_CHUNK``
    stay zero-initialised by the harness."""
    user_batch = pl.tensor.dim(seq_lens, 0)

    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_sq_k0 = kb * FINAL_RMS_K_CHUNK
                final_sq_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_sq_k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(final_sq_chunk, final_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_final = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )

            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_norm_k0 = kb * FINAL_RMS_K_CHUNK
                final_hidden_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_norm_k0]),
                    target_type=pl.FP32,
                )
                final_gamma = pl.slice(final_norm_weight, [1, FINAL_RMS_K_CHUNK], [0, final_norm_k0])
                final_normed_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(final_hidden_chunk, inv_rms_final),
                    final_gamma,
                )
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(final_normed_chunk, target_type=pl.BF16),
                    [b0, final_norm_k0],
                )

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        lm_valid_rows = pl.min(BATCH_TILE, user_batch - b0)
        lm_o0 = 0
        lm_acc_gm = pl.create_tensor([BATCH_TILE, VOCAB_CHUNK], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head"):
            lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, 0])
            lm_weight_chunk = pl.slice(lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [lm_o0, 0])
            lm_acc = pl.matmul(lm_hidden_chunk, lm_weight_chunk, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.range(1, HIDDEN // LM_HEAD_K_CHUNK):
                lm_k0 = kb * LM_HEAD_K_CHUNK
                lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, lm_k0])
                lm_weight_chunk = pl.slice(
                    lm_head_weight,
                    [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                    [lm_o0, lm_k0],
                )
                lm_acc = pl.matmul_acc(lm_acc, lm_hidden_chunk, lm_weight_chunk, b_trans=True)
            lm_acc_gm = pl.assemble(lm_acc_gm, lm_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_store"):
            lm_acc_chunk = pl.slice(lm_acc_gm, [BATCH_TILE, VOCAB_CHUNK], [0, 0])
            lm_acc_trimmed = pl.slice(
                lm_acc_chunk,
                [BATCH_TILE, VOCAB_CHUNK],
                [0, 0],
                valid_shape=[lm_valid_rows, VOCAB_CHUNK],
            )
            out = pl.assemble(out, lm_acc_trimmed, [b0, lm_o0])

    return out
