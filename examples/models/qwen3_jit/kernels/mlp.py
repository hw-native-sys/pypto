# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MLP block kernels for Scope 3 stages 4-6: gate proj + up proj + SiLU.

Decorated as ``@pl.jit.inline``: the body is spliced into the entry,
which lets ``OutlineIncoreScopes`` extract each ``pl.at`` block as a
separate InCore function — same shape as the monolithic reference.
"""

import pypto.language as pl

from ..config import BATCH, HIDDEN, INTERMEDIATE, K_CHUNK, MLP_OUT_CHUNK


@pl.jit.inline
def mlp_block(
    post_norm_tile: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    mlp_tile: pl.Out[pl.Tensor],
):
    """Three ``pl.at`` scopes per output chunk: gate proj, up proj, SiLU+mul."""
    for o0 in pl.parallel(0, INTERMEDIATE, MLP_OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
            gate_acc = pl.create_tensor([BATCH, MLP_OUT_CHUNK], dtype=pl.FP32)

            for kb in pl.pipeline(0, HIDDEN // K_CHUNK, stage=2):
                k0 = kb * K_CHUNK
                post_chunk = post_norm_tile[:, k0 : k0 + K_CHUNK]
                wg = w_gate[k0 : k0 + K_CHUNK, o0 : o0 + MLP_OUT_CHUNK]

                if k0 == 0:
                    gate_acc = pl.matmul(post_chunk, wg, out_dtype=pl.FP32)
                else:
                    gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
            up_acc = pl.create_tensor([BATCH, MLP_OUT_CHUNK], dtype=pl.FP32)

            for kb_u in pl.pipeline(0, HIDDEN // K_CHUNK, stage=2):
                k0_u = kb_u * K_CHUNK
                post_chunk_u = post_norm_tile[:, k0_u : k0_u + K_CHUNK]
                wu = w_up[k0_u : k0_u + K_CHUNK, o0 : o0 + MLP_OUT_CHUNK]

                if k0_u == 0:
                    up_acc = pl.matmul(post_chunk_u, wu, out_dtype=pl.FP32)
                else:
                    up_acc = pl.matmul_acc(up_acc, post_chunk_u, wu)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
            mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, o0])
    return mlp_tile
