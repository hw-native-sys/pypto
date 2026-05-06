# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""RMSNorm kernels.

Two flavours, one per call site in the model:
  - ``input_rmsnorm`` — pre-attention RMSNorm of hidden_states (chunk=512, stage=4)
  - ``post_rmsnorm`` — post-attention RMSNorm of the residual (chunk=128, stage=2)

Both are ``@pl.jit.inline``: each is called exactly once, so splicing avoids
the need for a separate Function in the IR while keeping the kernel source
reusable across other models that need the same shape.
"""

import pypto.language as pl

from ..config import BATCH, EPS, HIDDEN, HIDDEN_INV, K_CHUNK, RMSNORM_K_CHUNK


@pl.jit.inline
def input_rmsnorm(
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    normed_states: pl.Out[pl.Tensor],
):
    """Two-pass RMSNorm: variance reduction, then normalisation × gamma.

    NOTE on naming: locals in the two ``pl.pipeline`` loops are given
    distinct names (``_a``/``_b``) to work around a JIT specializer
    limitation — its alpha-renamer treats reuse of the same local in two
    sibling loops as a carry-over rebind and emits an out-of-scope read.
    The hand-written ``@pl.program`` reference can reuse names because the
    parser handles loop scoping directly.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
        partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)

        for kb_a in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):
            k0_a = kb_a * RMSNORM_K_CHUNK
            x_chunk_a = pl.cast(hidden_states[:, k0_a : k0_a + RMSNORM_K_CHUNK], target_type=pl.FP32)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(x_chunk_a, x_chunk_a)), [1, BATCH]),
            )

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
        inv_rms = pl.recip(pl.sqrt(variance))

        for kb_b in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):
            k0_b = kb_b * RMSNORM_K_CHUNK
            x_chunk_b = pl.cast(hidden_states[:, k0_b : k0_b + RMSNORM_K_CHUNK], target_type=pl.FP32)
            gamma = input_rms_weight[:, k0_b : k0_b + RMSNORM_K_CHUNK]
            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk_b, inv_rms), gamma)
            normed_states = pl.assemble(normed_states, pl.cast(normed, target_type=pl.BF16), [0, k0_b])
    return normed_states


@pl.jit.inline
def post_rmsnorm(
    resid: pl.Tensor,
    post_rms_weight: pl.Tensor,
    post_norm_tile: pl.Out[pl.Tensor],
):
    """Post-attention RMSNorm. Different chunk size from input_rmsnorm.

    See ``input_rmsnorm`` for the per-loop-naming workaround rationale.
    """
    # ``resid`` is the BF16 residual stream — promote each chunk to FP32 before
    # the squared-sum + normalize passes (mirrors ``input_rmsnorm``) so the
    # accumulation doesn't lose precision or overflow.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
        sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)

        for kb_a in pl.pipeline(HIDDEN // K_CHUNK, stage=2):
            k0_a = kb_a * K_CHUNK
            resid_chunk_a = pl.cast(resid[:, k0_a : k0_a + K_CHUNK], target_type=pl.FP32)
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(pl.row_sum(pl.mul(resid_chunk_a, resid_chunk_a)), [1, BATCH]),
            )

        inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
        inv_rms_col = pl.reshape(inv_rms, [BATCH, 1])

        for kb_b in pl.pipeline(HIDDEN // K_CHUNK, stage=2):
            k0_b = kb_b * K_CHUNK
            resid_chunk_b = pl.cast(resid[:, k0_b : k0_b + K_CHUNK], target_type=pl.FP32)
            gamma = post_rms_weight[:, k0_b : k0_b + K_CHUNK]
            normed = pl.col_expand_mul(pl.row_expand_mul(resid_chunk_b, inv_rms_col), gamma)
            post_norm_tile = pl.assemble(post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0_b])
    return post_norm_tile
