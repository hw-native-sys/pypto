# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Matrix multiplication using PyPTO language DSL (cube unit, 64x64).

Program structure:
  InCore function  ``matmul``
    - Loads tiles from GM → L1 (Mat space), then to L0A/L0B.
    - Computes matrix multiplication on the cube unit.
    - Stores L0C result directly back to GM.

  Orchestration function  ``orchestrator``
    - Calls ``matmul`` once for the full 64x64 computation.
"""

import pypto.language as pl


@pl.program
class MatmulProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        out_c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        out_c = self.matmul(a, b, out_c)
        return out_c


@pl.program
class MatmulaccProgram:
    """Matrix multiply with accumulation — K=64 split into two K=32 chunks.

    First chunk initialises L0C via ``matmul``; second chunk accumulates via
    ``matmul_acc``.  The final result equals the full 64×64 matrix product.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def matmul_acc(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # First K-chunk: A[:,0:32] @ B[0:32,:] — initialises L0C via matmul
        tile_a0_l1 = pl.load(a, offsets=[0, 0], shapes=[64, 32], target_memory=pl.MemorySpace.Mat)
        tile_b0_l1 = pl.load(b, offsets=[0, 0], shapes=[32, 64], target_memory=pl.MemorySpace.Mat)
        tile_a0_l0a = pl.move(tile_a0_l1, target_memory=pl.MemorySpace.Left)
        tile_b0_l0b = pl.move(tile_b0_l1, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(tile_a0_l0a, tile_b0_l0b)

        # Second K-chunk: A[:,32:64] @ B[32:64,:] — accumulates into existing L0C
        tile_a1_l1 = pl.load(a, offsets=[0, 32], shapes=[64, 32], target_memory=pl.MemorySpace.Mat)
        tile_b1_l1 = pl.load(b, offsets=[32, 0], shapes=[32, 64], target_memory=pl.MemorySpace.Mat)
        tile_a1_l0a = pl.move(tile_a1_l1, target_memory=pl.MemorySpace.Left)
        tile_b1_l0b = pl.move(tile_b1_l1, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul_acc(acc, tile_a1_l0a, tile_b1_l0b)

        out_c = pl.store(acc, offsets=[0, 0], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        out_c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        out_c = self.matmul_acc(a, b, out_c)
        return out_c
