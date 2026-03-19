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

Programs:
  TileConcat32x32Program — c[32,32] = concat(a[32,16], b[32,16])
"""

import pypto.language as pl


@pl.program
class TileConcat32x32Program:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_concat(
        self,
        a: pl.Tensor[[32, 16], pl.FP32],
        b: pl.Tensor[[32, 16], pl.FP32],
        c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 16])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 16])
        tile_out = pl.concat(tile_a, tile_b)
        out_c = pl.store(tile_out, offsets=[0, 0], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[32, 16], pl.FP32], b: pl.Tensor[[32, 16], pl.FP32]
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        out_c: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor([32, 32], dtype=pl.FP32)
        out_c = self.tile_concat(a, b, out_c)
        return out_c
