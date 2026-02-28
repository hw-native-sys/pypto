# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Hello World: element-wise tensor addition using PyPTO language DSL.

Program structure:
  InCore function  ``tile_add``
    - Loads tile_a and tile_b from global memory (GM) into registers (UB).
    - Computes tile_c = tile_a + tile_b using the vector unit.
    - Stores tile_c back to the output tensor in GM.

  Orchestration function  ``orchestrator``
    - Calls ``tile_add`` once to process the whole tensor in one shot.
"""

import pypto.language as pl


@pl.program
class HelloWorldProgram:
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
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        out_c = self.tile_add(a, b, out_c)
        return out_c
