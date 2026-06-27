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
    scratch instead of spilling to DDR; ``K=128`` splits.

    The intermediate is **bf16** — the cube accumulates in f32 (L0C) and the FIXPIPE
    writeback to L1 downcasts to bf16/f16 (the only offset Acc->Mat path on A2/A3), which
    is also the cube's native matmul-operand precision. The explicit ``pl.cast`` is the
    standard idiom (cf. deepseek); the autotiler fuses it into the per-sub-tile Acc->Mat
    ``pto.tinsert`` that fills the scratch."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mat_split_k"):
        c = pl.matmul(a, b, out_dtype=pl.FP32)  # bf16 @ bf16 -> f32 [256, 256] (> L0c)
        cb = pl.cast(c, pl.BF16)                # FIXPIPE downcast -> bf16 Mat scratch
        d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumes the scratch on-chip
        out = pl.assemble(out, d, [0, 0])
    return out


@pl.jit
def mat_full_k(a: pl.Tensor, b: pl.Tensor, e: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``(a @ b) @ e`` -> **Mat/L1 scratch**, **full-K**. Same bf16 chain but ``[256,32] @
    [32,256]``; ``K=32`` fits L0 (``k == K``), so the Mat-scratch ``tinsert``s land inside
    the pipelined full-K nest with loop-variable offsets."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mat_full_k"):
        c = pl.matmul(a, b, out_dtype=pl.FP32)  # bf16 @ bf16 -> f32 [256, 256] (> L0c)
        cb = pl.cast(c, pl.BF16)                # FIXPIPE downcast -> bf16 Mat scratch
        d = pl.matmul(cb, e, out_dtype=pl.FP32)
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

    # Mat/L1 scratch: (a @ b) @ e -> [256, 64]; bf16 operands, bf16 on-chip intermediate.
    # K picks split-K (128) vs full-K (32). Golden models the FIXPIPE bf16 downcast of the
    # intermediate (f32 accumulate -> bf16 L1 -> f32 accumulate), so the tolerance is bf16.
    for fn, K in ((mat_split_k, 128), (mat_full_k, 32)):
        a = torch.randn(256, K, dtype=torch.bfloat16)
        b = torch.randn(K, 256, dtype=torch.bfloat16)
        e = torch.randn(256, 64, dtype=torch.bfloat16)
        out = torch.zeros((256, 64), dtype=torch.float32)
        fn(a, b, e, out, config=cfg)
        c_bf16 = (a.float() @ b.float()).to(torch.bfloat16).float()  # FIXPIPE downcast
        golden = c_bf16 @ e.float()
        assert torch.allclose(out, golden, rtol=2e-2, atol=2e-2), f"{fn.__name__} mismatch"

    print("OK")
