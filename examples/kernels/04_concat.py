# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tile column-wise concatenation: c[:, :16] = a, c[:, 16:] = b.

Kernels:
  tile_concat_32x32 -- c[32,32] = concat(a[32,16], b[32,16])

Concepts introduced:
  - pl.concat for column-wise tile concatenation

Run:  python examples/kernels/04_concat.py
Next: examples/kernels/05_activation.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def tile_concat_32x32(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [32, 16])
        tile_b = pl.load(b, [0, 0], [32, 16])
        tile_out: pl.Tile[[32, 32], pl.FP32] = pl.concat(tile_a, tile_b)
        pl.store(tile_out, [0, 0], c)
    return c


if __name__ == "__main__":
    cfg = RunConfig()
    torch.manual_seed(0)

    a = torch.randn(32, 16, dtype=torch.float32)
    b = torch.randn(32, 16, dtype=torch.float32)
    c = torch.zeros((32, 32), dtype=torch.float32)
    tile_concat_32x32(a, b, c, config=cfg)
    expected = torch.cat([a, b], dim=1)
    assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5)

    print("OK")
