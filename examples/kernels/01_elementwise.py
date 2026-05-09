# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tile element-wise operations: add and multiply.

Kernels:
  tile_add_128 — c = a + b  (128x128)
  tile_mul_128 — c = a * b  (128x128)
  tile_add_64  — c = a + b  (64x64)
  tile_mul_64  — c = a * b  (64x64)

Concepts introduced:
  - pl.mul for element-wise multiplication
  - Multiple @pl.jit kernels in one file

Run:  python examples/kernels/01_elementwise.py
Next: examples/kernels/02_fused_ops.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def tile_add_128(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def tile_mul_128(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.mul(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def tile_add_64(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Element-wise addition on 64x64 tiles."""
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def tile_mul_64(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Element-wise multiplication on 64x64 tiles."""
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_c = pl.mul(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c


if __name__ == "__main__":
    cfg = RunConfig()

    a128 = torch.full((128, 128), 2.0, dtype=torch.float32)
    b128 = torch.full((128, 128), 3.0, dtype=torch.float32)
    c128 = torch.zeros((128, 128), dtype=torch.float32)
    tile_add_128(a128, b128, c128, config=cfg)
    assert torch.allclose(c128, a128 + b128, rtol=1e-5, atol=1e-5)

    c128 = torch.zeros((128, 128), dtype=torch.float32)
    tile_mul_128(a128, b128, c128, config=cfg)
    assert torch.allclose(c128, a128 * b128, rtol=1e-5, atol=1e-5)

    a64 = torch.full((64, 64), 2.0, dtype=torch.float32)
    b64 = torch.full((64, 64), 3.0, dtype=torch.float32)
    c64 = torch.zeros((64, 64), dtype=torch.float32)
    tile_add_64(a64, b64, c64, config=cfg)
    assert torch.allclose(c64, a64 + b64, rtol=1e-5, atol=1e-5)

    c64 = torch.zeros((64, 64), dtype=torch.float32)
    tile_mul_64(a64, b64, c64, config=cfg)
    assert torch.allclose(c64, a64 * b64, rtol=1e-5, atol=1e-5)

    print("OK")
