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

Programs:
  TileAddProgram — c = a + b  (128x128)
  TileMulProgram — c = a * b  (128x128)

Concepts introduced:
  - pl.mul for element-wise multiplication
  - Multiple programs in one file

Run:  python examples/kernels/01_elementwise.py
Next: examples/kernels/02_fused_ops.py
"""

import pypto.language as pl


@pl.program
class TileAddProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c = self.tile_add(a, b, out_c)
        return out_c


@pl.program
class TileMulProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_mul(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.mul(tile_a, tile_b)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c = self.tile_mul(a, b, out_c)
        return out_c


@pl.program
class TileAdd64Program:
    """Element-wise addition on 64x64 tiles."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_c = pl.add(tile_a, tile_b)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        out_c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        out_c = self.tile_add(a, b, out_c)
        return out_c


@pl.program
class TileMul64Program:
    """Element-wise multiplication on 64x64 tiles."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_mul(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_c = pl.mul(tile_a, tile_b)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        out_c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        out_c = self.tile_mul(a, b, out_c)
        return out_c


# Aliases for backward compatibility with tests that use size-suffixed names
TileAdd128Program = TileAddProgram
TileMul128Program = TileMulProgram


if __name__ == "__main__":
    print("=== TileAddProgram ===")
    print(TileAddProgram.as_python())
    print("\n=== TileMulProgram ===")
    print(TileMulProgram.as_python())
