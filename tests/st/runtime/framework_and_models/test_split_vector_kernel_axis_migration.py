# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SplitVectorKernel split-axis migration regression test (issue #1790).

A UP_DOWN-split mixed cube+vec scope that transposes the token (split) axis:
the RMS denominator is kept as a row vector ``[1, T_TILE]`` (token on the
column axis) via ``reshape(row_sum(...), [1, T_TILE])``, and the per-head pre
scales are extracted with ``tile.transpose`` + ``reshape(..., [T_TILE, 1])``.

Before the split-axis-migration fix, SplitVectorKernel only halved the global
row axis (dim0). For these transpose/reshape chains the token axis migrates to
the column axis, so the pass emitted numel-inconsistent / full-width tiles
(e.g. ``reshape([8, 1] -> [1, 16])``) that crashed ``memory_reuse`` with
"sharing-group member ... falls outside reuse target". The fix resolves each
tile's split axis (consumer-driven) so every tile halves on the migrated axis.

This is a host-pass (compile-time) crash, so the test runs compile-only and
needs no device.
"""

import pypto.language as pl
import pytest
import torch
from pypto.runtime.runner import RunConfig

T_DYN = pl.dynamic("T_DYN")

T_TILE = 16  # row (token) tile -- UP_DOWN halves this to 8 per AIV lane
D = 256  # hidden width
K_TILE = 128  # matmul reduction tile
MIX_PAD = 32  # linear projection width (vector padded)
HC_PAD = 8  # padded per-head width
HC_MULT = 4  # number of heads unrolled below
D_TILE = 128  # mix_x output tile
T_MAX = 128  # static upper bound for the GM scratch row
EPS = 1e-6
D_INV = 1.0 / D


@pl.jit
def hc_pre_like(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w: pl.Tensor[[MIX_PAD, D], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)

    t_dim = pl.tensor.dim(x, 0)
    mixes_gm = pl.create_tensor([T_MAX, MIX_PAD], dtype=pl.FP32)

    for ob in pl.spmd(
        t_dim // T_TILE, name_hint="hc_pre_like", optimizations=[pl.split(pl.SplitMode.UP_DOWN)]
    ):
        t0 = ob * T_TILE

        # --- RMS + linear: row-vector sq_sum accumulator (token on columns) ---
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        mix_acc = pl.create_tensor([T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            kl0 = kb * K_TILE
            x_lin = pl.cast(x[t0 : t0 + T_TILE, kl0 : kl0 + K_TILE], target_type=pl.FP32)
            x_sq = pl.mul(x_lin, x_lin)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(x_sq), [1, T_TILE]))
            w_lin = pl.slice(w, [MIX_PAD, K_TILE], [0, kl0])
            if kb == 0:
                mix_acc = pl.matmul(x_lin, w_lin, b_trans=True, out_dtype=pl.FP32)
            else:
                mix_acc = pl.matmul_acc(mix_acc, x_lin, w_lin, b_trans=True)
        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, D_INV), EPS))  # [1, T_TILE]
        inv_rms_col = pl.reshape(inv_rms, [T_TILE, 1])  # token migrates back to rows
        mixes_gm[t0 : t0 + T_TILE, 0:MIX_PAD] = pl.row_expand_mul(mix_acc, inv_rms_col)

        # --- pre: transpose token axis to a row, extract per-head columns ---
        pre_in = pl.load(mixes_gm, [t0, 0], [T_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec)
        pre_eps = pl.recip(pl.add(pl.exp(pl.neg(pre_in)), 1.0))  # [T_TILE, HC_PAD]
        pre_eps_t = pl.transpose(pre_eps, axis1=0, axis2=1)  # [HC_PAD, T_TILE]
        pre0 = pl.mul(pl.reshape(pre_eps_t[0:1, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre1 = pl.mul(pl.reshape(pre_eps_t[1:2, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre2 = pl.mul(pl.reshape(pre_eps_t[2:3, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre3 = pl.mul(pl.reshape(pre_eps_t[3:4, 0:T_TILE], [T_TILE, 1]), 1.0)

        for db in pl.range(D // D_TILE):
            d0 = db * D_TILE
            y = pl.row_expand_mul(
                pl.cast(
                    pl.load(x, [t0, d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                ),
                pre0,
            )
            y = pl.add(
                y,
                pl.row_expand_mul(
                    pl.cast(
                        pl.load(x, [t0, d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec),
                        target_type=pl.FP32,
                    ),
                    pre1,
                ),
            )
            y = pl.add(
                y,
                pl.row_expand_mul(
                    pl.cast(
                        pl.load(x, [t0, d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec),
                        target_type=pl.FP32,
                    ),
                    pre2,
                ),
            )
            y = pl.add(
                y,
                pl.row_expand_mul(
                    pl.cast(
                        pl.load(x, [t0, d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec),
                        target_type=pl.FP32,
                    ),
                    pre3,
                ),
            )
            pl.store(pl.cast(y, target_type=pl.BF16, mode="rint"), [t0, d0], out)
    return out


def _compile_only(platform: str = "a2a3sim"):
    """Compile the kernel through the full host pass pipeline (no device)."""
    T = 32  # two T_TILE rows -> exercises the split
    x = torch.empty(T, D, dtype=torch.bfloat16)
    w = torch.empty(MIX_PAD, D, dtype=torch.float32)
    out = torch.empty(T, D, dtype=torch.bfloat16)
    return hc_pre_like.compile(x, w, out, config=RunConfig(platform=platform))


def test_split_vector_kernel_axis_migration_compiles():
    """Regression for #1790: the transpose/reshape-of-split-axis scope must
    compile (pre-fix this aborted in memory_reuse with a sharing-group OOB)."""
    compiled = _compile_only()
    assert compiled is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
