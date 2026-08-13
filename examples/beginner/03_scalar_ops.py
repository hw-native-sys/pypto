# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Scalar operations: multiplying a tile by a scalar constant.

Kernel:
  fused_add_scale — c = (a + b) * 2.0  (vector only)

Concepts introduced:
  - Scalar operations: pl.mul(tile, 2.0)

Run:  python examples/beginner/03_scalar_ops.py
Next: examples/beginner/04_activation.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def fused_add_scale(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Fused: load a, b -> add -> scale by 2.0 -> store c."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_sum = pl.add(tile_a, tile_b)
        tile_c = pl.mul(tile_sum, 2.0)
        pl.store(tile_c, [0, 0], c)
    return c


if __name__ == "__main__":
    cfg = RunConfig()

    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    c = torch.zeros((128, 128), dtype=torch.float32)
    fused_add_scale(a, b, c, config=cfg)
    assert torch.allclose(c, (a + b) * 2.0, rtol=1e-5, atol=1e-5)

    print("OK")
