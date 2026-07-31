# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Matmul with a 16-row physical tile and five valid rows."""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

VALID_M = 5
TILE_M = 16
K = 16
N = 16


@pl.jit
def matmul_valid_shape(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        a_mat: pl.Tile[
            [TILE_M, K],
            pl.FP32,
            pl.Mem.Mat,
            pl.TileView(valid_shape=[VALID_M, K]),
        ] = pl.load(
            a,
            [0, 0],
            [TILE_M, K],
            valid_shape=[VALID_M, K],
            target_memory=pl.Mem.Mat,
        )
        b_mat = pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        a_left = pl.move(a_mat, target_memory=pl.Mem.Left)
        b_right = pl.move(b_mat, target_memory=pl.Mem.Right)
        result: pl.Tile[
            [TILE_M, N],
            pl.FP32,
            pl.Mem.Acc,
            pl.TileView(valid_shape=[VALID_M, N]),
        ] = pl.matmul(a_left, b_right)
        pl.store(result, [0, 0], output)
    return output


if __name__ == "__main__":
    cfg = RunConfig()
    torch.manual_seed(0)
    a = torch.randn(VALID_M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    output = torch.zeros(TILE_M, N, dtype=torch.float32)
    matmul_valid_shape(a, b, output, config=cfg)
    expected = torch.zeros_like(output)
    expected[:VALID_M] = torch.matmul(a, b)
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    print("OK")
