# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Q/K/V/Output/MLP projection kernels.

Each utility wraps both the orchestration ``pl.parallel`` loop and the
inner ``pl.at`` compute scope. Decorated as ``@pl.jit.inline`` because the
splice keeps the IR structurally close to the monolithic reference while
enabling cross-file reuse.
"""

import pypto.language as pl

from ..config import (
    BATCH,
    DOWN_K_CHUNK,
    DOWN_N_CHUNK,
    HIDDEN,
    INTERMEDIATE,
    KV_HIDDEN,
    KV_OUT_CHUNK,
    KV_PROJ_K_CHUNK,
    OUT_PROJ_K_CHUNK,
    Q_OUT_CHUNK,
    Q_PROJ_K_CHUNK,
)


@pl.jit.inline
def q_projection(
    normed_states: pl.Tensor,
    wq: pl.Tensor,
    q_proj: pl.Out[pl.Tensor],
):
    """Q projection: tiled matmul over the K dimension, parallel over N."""
    for q0 in pl.parallel(0, HIDDEN, Q_OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
            q_acc = pl.create_tensor([BATCH, Q_OUT_CHUNK], dtype=pl.FP32)

            for kb in pl.pipeline(0, HIDDEN // Q_PROJ_K_CHUNK, stage=2):
                k0 = kb * Q_PROJ_K_CHUNK
                tile_a = normed_states[:, k0 : k0 + Q_PROJ_K_CHUNK]
                tile_b = wq[k0 : k0 + Q_PROJ_K_CHUNK, q0 : q0 + Q_OUT_CHUNK]

                if k0 == 0:
                    q_acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, tile_a, tile_b)

            q_proj = pl.assemble(q_proj, q_acc, [0, q0])
    return q_proj


@pl.jit.inline
def k_projection(
    normed_states: pl.Tensor,
    wk: pl.Tensor,
    k_proj: pl.Out[pl.Tensor],
):
    """K projection.

    Note: in the upstream monolithic Qwen3 reference K and V projections share
    one ``pl.at`` scope to amortise ``normed_states`` loads. Split here because
    the JIT specializer currently emits a single-tensor return-type annotation
    even when the body returns a tuple — flagged as a follow-up.
    """
    for kv0 in pl.parallel(0, KV_HIDDEN, KV_OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj"):
            k_acc = pl.create_tensor([BATCH, KV_OUT_CHUNK], dtype=pl.FP32)
            for kb in pl.pipeline(0, HIDDEN // KV_PROJ_K_CHUNK, stage=2):
                k0 = kb * KV_PROJ_K_CHUNK
                tile_a = normed_states[:, k0 : k0 + KV_PROJ_K_CHUNK]
                tile_wk = wk[k0 : k0 + KV_PROJ_K_CHUNK, kv0 : kv0 + KV_OUT_CHUNK]
                if k0 == 0:
                    k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                else:
                    k_acc = pl.matmul_acc(k_acc, tile_a, tile_wk)
            k_proj = pl.assemble(k_proj, k_acc, [0, kv0])
    return k_proj


@pl.jit.inline
def v_projection(
    normed_states: pl.Tensor,
    wv: pl.Tensor,
    v_proj: pl.Out[pl.Tensor],
):
    """V projection — symmetric to K."""
    for kv0 in pl.parallel(0, KV_HIDDEN, KV_OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj"):
            v_acc = pl.create_tensor([BATCH, KV_OUT_CHUNK], dtype=pl.FP32)
            for kb in pl.pipeline(0, HIDDEN // KV_PROJ_K_CHUNK, stage=2):
                k0 = kb * KV_PROJ_K_CHUNK
                tile_a = normed_states[:, k0 : k0 + KV_PROJ_K_CHUNK]
                tile_wv = wv[k0 : k0 + KV_PROJ_K_CHUNK, kv0 : kv0 + KV_OUT_CHUNK]
                if k0 == 0:
                    v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                else:
                    v_acc = pl.matmul_acc(v_acc, tile_a, tile_wv)
            v_proj = pl.assemble(v_proj, v_acc, [0, kv0])
    return v_proj


@pl.jit.inline
def out_projection_residual(
    attn_out: pl.Tensor,
    hidden_states: pl.Tensor,
    wo: pl.Tensor,
    resid1_tile: pl.Out[pl.Tensor],
):
    """attn_out @ wo + hidden_states  (Scope 3 stages 1+2)."""
    for ob in pl.parallel(0, HIDDEN // Q_OUT_CHUNK, 2):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)],
            name_hint="out_proj_residual",
        ):
            for oi in pl.range(ob, ob + 2):
                o0 = oi * Q_OUT_CHUNK
                hidden_chunk = hidden_states[:, o0 : o0 + Q_OUT_CHUNK]

                o_acc = pl.create_tensor([BATCH, Q_OUT_CHUNK], dtype=pl.FP32)

                for kb in pl.pipeline(0, HIDDEN // OUT_PROJ_K_CHUNK, stage=2):
                    k0 = kb * OUT_PROJ_K_CHUNK
                    a_chunk = attn_out[:, k0 : k0 + OUT_PROJ_K_CHUNK]
                    w_chunk = wo[k0 : k0 + OUT_PROJ_K_CHUNK, o0 : o0 + Q_OUT_CHUNK]

                    if k0 == 0:
                        o_acc = pl.matmul(a_chunk, w_chunk, out_dtype=pl.FP32)
                    else:
                        o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                resid = pl.cast(hidden_chunk, target_type=pl.FP32)
                resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])
    return resid1_tile


@pl.jit.inline
def down_projection_residual(
    mlp_tile: pl.Tensor,
    resid1_tile: pl.Tensor,
    w_down: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Down projection of MLP output + final residual writeback (Scope 3 stages 7+8)."""
    for db in pl.parallel(0, HIDDEN // DOWN_N_CHUNK, 2):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)],
            name_hint="down_proj_residual",
        ):
            for di in pl.range(db, db + 2):
                d0 = di * DOWN_N_CHUNK
                resid_chunk = resid1_tile[:, d0 : d0 + DOWN_N_CHUNK]

                down_acc = pl.create_tensor([BATCH, DOWN_N_CHUNK], dtype=pl.FP32)

                for ob in pl.pipeline(0, INTERMEDIATE // DOWN_K_CHUNK, stage=2):
                    o0 = ob * DOWN_K_CHUNK
                    mlp_chunk = mlp_tile[:, o0 : o0 + DOWN_K_CHUNK]
                    w_chunk = w_down[o0 : o0 + DOWN_K_CHUNK, d0 : d0 + DOWN_N_CHUNK]

                    if o0 == 0:
                        down_acc = pl.matmul(mlp_chunk, w_chunk, out_dtype=pl.FP32)
                    else:
                        down_acc = pl.matmul_acc(down_acc, mlp_chunk, w_chunk)

                out_chunk = pl.add(down_acc, resid_chunk)
                out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [0, d0])
    return out
