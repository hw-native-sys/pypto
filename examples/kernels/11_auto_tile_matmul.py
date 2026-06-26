# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Compiler-driven L0 matmul tiling (AutoTileMatmulL0) -- the placement x K-strategy matrix.

When a matmul's ``[M, N]`` output exceeds the cube accumulator L0c (128 KB on Ascend910B),
AutoTileMatmulL0 tiles the output into ``[m, n]`` sub-tiles and **places** each one by where
the result is consumed:

  - **DDR (direct-store)** -- the result is stored to a DDR tensor; each sub-tile is stored
    straight to ``out[mi:, ni:]``.
  - **Mat/L1 (Mat-scratch)** -- the result is consumed on-chip by *another matmul*; each
    sub-tile is assembled (Acc->Mat ``tile.assemble``) into an L1/Mat scratch kept on-chip,
    so the chain never spills the intermediate to DDR.

Orthogonally, the K (reduction) dimension picks the **K-strategy**:

  - **full-K** -- the whole K fits L0a/L0b at once (``k == K``); the M/N grid is a pipelined
    nest with loop-variable offsets (BuildFullKPipelined).
  - **split-K** -- K spans >= 2 L0 blocks; each ``[m, n]`` sub-tile is its own pipelined
    K-loop (BuildSplitKGrid).

The four kernels below are the 2x2 matrix. The K-strategy is **shape-driven**: for an
``[M, N] = [256, 256]`` FP32 output the L0 chooser caps ``k = 32``, so ``K = 32`` fits L0 in
one pass (full-K) while ``K = 128`` splits into 4 K-blocks (split-K). No manual tiling --
the compiler picks the path from the shapes and the consumer.

Run:  python examples/kernels/11_auto_tile_matmul.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def ddr_split_k(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``a @ b`` -> **DDR**, **split-K**. ``[256,128] @ [128,256] = [256,256]`` (> L0c) is
    consumed by the output store, so each ``[m, n]`` sub-tile direct-stores to DDR; ``K=128``
    splits into K-blocks."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ddr_split_k"):
        c = pl.matmul(a, b, out_dtype=pl.FP32)  # [256, 256] -> M/N-tiled, stored to DDR
        out = pl.assemble(out, c, [0, 0])
    return out


@pl.jit
def ddr_full_k(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``a @ b`` -> **DDR**, **full-K**. Same as above but ``[256,32] @ [32,256]``; ``K=32``
    fits L0 in one pass (``k == K``), so the grid is the pipelined full-K nest."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ddr_full_k"):
        c = pl.matmul(a, b, out_dtype=pl.FP32)
        out = pl.assemble(out, c, [0, 0])
    return out


@pl.jit
def mat_split_k(a: pl.Tensor, b: pl.Tensor, e: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``(a @ b) @ e`` -> **Mat/L1 scratch**, **split-K**. The ``[256,256]`` intermediate
    (> L0c) is consumed on-chip by the second matmul, so it is M/N-tiled into an L1/Mat
    scratch (Acc->Mat assemble) instead of spilling to DDR; ``K=128`` splits."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mat_split_k"):
        c = pl.matmul(a, b, out_dtype=pl.FP32)  # [256, 256] -> Mat scratch (compiler-tiled)
        d = pl.matmul(c, e, out_dtype=pl.FP32)  # consumes c on-chip
        out = pl.assemble(out, d, [0, 0])
    return out


@pl.jit
def mat_full_k(a: pl.Tensor, b: pl.Tensor, e: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``(a @ b) @ e`` -> **Mat/L1 scratch**, **full-K**. Same chain but ``[256,32] @
    [32,256]``; ``K=32`` fits L0 (``k == K``), so the Mat-scratch assembles land inside the
    pipelined full-K nest with loop-variable offsets."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mat_full_k"):
        c = pl.matmul(a, b, out_dtype=pl.FP32)
        d = pl.matmul(c, e, out_dtype=pl.FP32)
        out = pl.assemble(out, d, [0, 0])
    return out


if __name__ == "__main__":
    cfg = RunConfig()
    torch.manual_seed(0)

    # DDR (direct-store): a @ b -> [256, 256]; K picks split-K (128) vs full-K (32).
    for fn, K in ((ddr_split_k, 128), (ddr_full_k, 32)):
        a = torch.randn(256, K, dtype=torch.float32)
        b = torch.randn(K, 256, dtype=torch.float32)
        out = torch.zeros((256, 256), dtype=torch.float32)
        fn(a, b, out, config=cfg)
        assert torch.allclose(out, a @ b, rtol=1e-3, atol=1e-3), f"{fn.__name__} mismatch"

    # Mat/L1 scratch: (a @ b) @ e -> [256, 64]; K picks split-K (128) vs full-K (32).
    for fn, K in ((mat_split_k, 128), (mat_full_k, 32)):
        a = torch.randn(256, K, dtype=torch.float32)
        b = torch.randn(K, 256, dtype=torch.float32)
        e = torch.randn(256, 64, dtype=torch.float32)
        out = torch.zeros((256, 64), dtype=torch.float32)
        fn(a, b, e, out, config=cfg)
        assert torch.allclose(out, (a @ b) @ e, rtol=1e-3, atol=1e-3), f"{fn.__name__} mismatch"

    print("OK")
