# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
The simplest PyPTO program: element-wise tensor addition.

Concepts introduced:
  - @pl.jit decorator: function specializes on torch tensor shape/dtype, compiles, caches
  - pl.incore() context: a single on-chip compute scope (load tiles, compute, store back)
  - pl.Out[] marks output tensor parameters (in-place mutation)
  - Tensor (global memory) vs Tile (on-chip register) types

Run:  python examples/hello_world.py
Next: examples/kernels/01_elementwise.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c


if __name__ == "__main__":
    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    c = torch.zeros((128, 128), dtype=torch.float32)
    tile_add(a, b, c, config=RunConfig())
    expected = a + b
    assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
        f"hello_world tile_add failed: max diff = {(c - expected).abs().max().item()}"
    )
    print("OK")
