# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Matrix multiplication on the cube unit (64x64).

Kernels:
  matmul_64     — full 64x64 matmul in one shot
  matmul_acc_64 — K=64 split into two K=32 chunks with matmul + matmul_acc

Concepts introduced:
  - Memory hierarchy: GM -> Mat (L1) -> Left/Right (L0A/L0B) -> matmul -> Acc (L0C)
  - pl.matmul for cube unit multiplication
  - pl.matmul_acc for accumulating partial products
  - K-dimension tiling for large reductions

Run:  python examples/kernels/03_matmul.py
Next: examples/kernels/04_concat.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def matmul_64(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.incore():
        tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        pl.store(tile_c_l0c, [0, 0], c)
    return c


@pl.jit
def matmul_acc_64(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Matrix multiply with accumulation -- K=64 split into two K=32 chunks.

    First chunk initialises L0C via ``matmul``; second chunk accumulates via
    ``matmul_acc``.  The final result equals the full 64x64 matrix product.
    """
    with pl.incore():
        # First K-chunk: A[:,0:32] @ B[0:32,:] -- initialises L0C via matmul
        tile_a0_l1 = pl.load(a, [0, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
        tile_b0_l1 = pl.load(b, [0, 0], [32, 64], target_memory=pl.MemorySpace.Mat)
        tile_a0_l0a = pl.move(tile_a0_l1, target_memory=pl.MemorySpace.Left)
        tile_b0_l0b = pl.move(tile_b0_l1, target_memory=pl.MemorySpace.Right)
        acc: pl.Tile[[64, 64], pl.FP32] = pl.matmul(tile_a0_l0a, tile_b0_l0b)

        # Second K-chunk: A[:,32:64] @ B[32:64,:] -- accumulates into existing L0C
        tile_a1_l1 = pl.load(a, [0, 32], [64, 32], target_memory=pl.MemorySpace.Mat)
        tile_b1_l1 = pl.load(b, [32, 0], [32, 64], target_memory=pl.MemorySpace.Mat)
        tile_a1_l0a = pl.move(tile_a1_l1, target_memory=pl.MemorySpace.Left)
        tile_b1_l0b = pl.move(tile_b1_l1, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul_acc(acc, tile_a1_l0a, tile_b1_l0b)

        pl.store(acc, [0, 0], c)
    return c


if __name__ == "__main__":
    cfg = RunConfig()
    torch.manual_seed(0)

    a = torch.randn(64, 64, dtype=torch.float32)
    b = torch.randn(64, 64, dtype=torch.float32)

    c = torch.zeros((64, 64), dtype=torch.float32)
    matmul_64(a, b, c, config=cfg)
    assert torch.allclose(c, torch.matmul(a, b), rtol=1e-3, atol=1e-3)

    c = torch.zeros((64, 64), dtype=torch.float32)
    matmul_acc_64(a, b, c, config=cfg)
    assert torch.allclose(c, torch.matmul(a, b), rtol=1e-3, atol=1e-3)

    print("OK")
