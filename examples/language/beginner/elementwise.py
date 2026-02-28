# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tile element-wise operations: add and multiply, in 128x128 and 64x64 shapes.

Programs:
  TileAdd128Program  — c = a + b  (128x128)
  TileAdd64Program   — c = a + b  (64x64)
  TileMul128Program  — c = a * b  (128x128)
  TileMul64Program   — c = a * b  (64x64)
"""

import pypto.language as pl


@pl.program
class TileAdd128Program:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
        tile_c = pl.add(tile_a, tile_b)
        out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        out_c = self.tile_add(a, b, out_c)
        return out_c


@pl.program
class TileAdd64Program:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 64])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[64, 64])
        tile_c = pl.add(tile_a, tile_b)
        out_c = pl.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        out_c: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        out_c = self.tile_add(a, b, out_c)
        return out_c


@pl.program
class TileMul128Program:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_mul(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
        tile_c = pl.mul(tile_a, tile_b)
        out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        out_c = self.tile_mul(a, b, out_c)
        return out_c


@pl.program
class TileMul64Program:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_mul(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 64])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[64, 64])
        tile_c = pl.mul(tile_a, tile_b)
        out_c = pl.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        out_c: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        out_c = self.tile_mul(a, b, out_c)
        return out_c
