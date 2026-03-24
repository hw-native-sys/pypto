# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tile assemble: write a source tile into a target tile at a specified offset.

Hardware semantics (PTO backend):
  tile.assemble maps to the TINSERT instruction. The hardware mode is inferred
  automatically from the memory spaces of the operands:

  Acc→Mat (TInsertMode::NZ) — source from Acc (L0C), target in Mat (L1):
    - target tile: in Mat (L1), fractal layout
    - source tile: in Acc (L0C), fractal layout (always FP32, output of tile.matmul)
    Data flow:
      a, b (GM) → Mat → Left/Right → tile.matmul → Acc (FP32)
      x   (GM) → Mat (FP32) [target]
      TINSERT NZ: Acc → Mat [at offset]
      Mat → Vec → GM

  Vec→Vec (TInsertMode::ND_VEC) — both tiles in Vec (UB), RowMajor/ND layout:
    - target tile: in Vec (UB), ND layout
    - source tile: in Vec (UB), ND layout
    Data flow:
      x   (GM) → Vec (UB) [target]
      src (GM) → Vec (UB) [source]
      TINSERT ND_VEC: Vec → Vec [at offset]
      Vec → GM

Programs:
  TileAssembleZeroOffsetProgram         — Acc→Mat: matmul(a[32,16], b[16,16]) → x[32,32] at [0, 0]
  TileAssembleRightOffsetProgram        — Acc→Mat: matmul(a[32,16], b[16,16]) → x[32,32] at [0, 16]
  TileAssembleVecZeroOffsetProgram      — Vec→Vec: src[32,16] → x[32,32] at [0, 0]  (left half)
  TileAssembleVecRightOffsetProgram     — Vec→Vec: src[32,16] → x[32,32] at [0, 16] (right half)
  TileAddThenAssembleZeroOffsetProgram  — Vec→Vec: add(src, delta) then assemble into x[32,32] at [0, 0]
  TileAddThenAssembleRightOffsetProgram — Vec→Vec: add(src, delta) then assemble into x[32,32] at [0, 16]
"""

import pypto.language as pl


@pl.program
class TileAssembleZeroOffsetProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        a: pl.Tensor[[32, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target into Mat (L1)
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Mat)
        # Produce Acc (L0C, FP32) via matmul: GM → Mat → Left/Right → matmul
        tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[16, 16], target_memory=pl.MemorySpace.Mat)
        tile_a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_src = pl.matmul(tile_a, tile_b)  # FP32 Acc (L0C) — same dtype as tile_x
        # Assemble: insert tile_src into tile_x at offset [0, 0]; result stays in Mat (L1)
        result = pl.tile.assemble(tile_x, tile_src, [0, 0])
        # Move Mat → Vec before store
        result_vec = pl.move(result, target_memory=pl.MemorySpace.Vec)
        out_y = pl.store(result_vec, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        a: pl.Tensor[[32, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, a, b, y)
        return y


@pl.program
class TileAssembleRightOffsetProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        a: pl.Tensor[[32, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target into Mat (L1)
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Mat)
        # Produce Acc (L0C, FP32) via matmul: GM → Mat → Left/Right → matmul
        tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[16, 16], target_memory=pl.MemorySpace.Mat)
        tile_a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_src = pl.matmul(tile_a, tile_b)  # FP32 Acc (L0C) — same dtype as tile_x
        # Assemble: insert tile_src into tile_x at offset [0, 16]; result stays in Mat (L1)
        result = pl.tile.assemble(tile_x, tile_src, [0, 16])
        # Move Mat → Vec before store
        result_vec = pl.move(result, target_memory=pl.MemorySpace.Vec)
        out_y = pl.store(result_vec, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        a: pl.Tensor[[32, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, a, b, y)
        return y


@pl.program
class TileAssembleVecZeroOffsetProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target and source directly into Vec (UB) — ND/RowMajor layout
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        # Assemble: insert tile_src into tile_x at [0, 0] — both Vec → ND_VEC mode
        result = pl.tile.assemble(tile_x, tile_src, [0, 0])
        out_y = pl.store(result, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y


@pl.program
class TileAssembleVecRightOffsetProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target and source directly into Vec (UB) — ND/RowMajor layout
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        # Assemble: insert tile_src into tile_x at [0, 16] — both Vec → ND_VEC mode
        result = pl.tile.assemble(tile_x, tile_src, [0, 16])
        out_y = pl.store(result, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y


@pl.program
class TileAddThenAssembleZeroOffsetProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        delta: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target, source, and delta into Vec (UB)
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        tile_delta = pl.load(delta, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        # Add delta to src before assembling
        tile_src_added = pl.add(tile_src, tile_delta)
        # Assemble: insert tile_src_added into tile_x at [0, 0] — both Vec → ND_VEC mode
        result = pl.tile.assemble(tile_x, tile_src_added, [0, 0])
        out_y = pl.store(result, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        delta: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_add_assemble(x, src, delta, y)
        return y


@pl.program
class TileAddThenAssembleRightOffsetProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        delta: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target, source, and delta into Vec (UB)
        tile_x = pl.load(x, offsets=[0, 0], shapes=[32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        tile_delta = pl.load(delta, offsets=[0, 0], shapes=[32, 16], target_memory=pl.MemorySpace.Vec)
        # Add delta to src before assembling
        tile_src_added = pl.add(tile_src, tile_delta)
        # Assemble: insert tile_src_added into tile_x at [0, 16] — both Vec → ND_VEC mode
        result = pl.tile.assemble(tile_x, tile_src_added, [0, 16])
        out_y = pl.store(result, offsets=[0, 0], output_tensor=y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        delta: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_add_assemble(x, src, delta, y)
        return y
