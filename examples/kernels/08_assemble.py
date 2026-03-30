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

  Acc->Mat (TInsertMode::NZ) -- source from Acc (L0C), target in Mat (L1):
    - target tile: in Mat (L1), fractal layout
    - source tile: in Acc (L0C), fractal layout (always FP32, output of tile.matmul)
    Data flow:
      a, b (GM) -> Mat -> Left/Right -> tile.matmul -> Acc (FP32)
      x   (GM) -> Mat (FP32) [target]
      TINSERT NZ: Acc -> Mat [at offset]
      Mat -> Vec -> GM

  Vec->Vec (TInsertMode::ND_VEC) -- both tiles in Vec (UB), RowMajor/ND layout:
    - target tile: in Vec (UB), ND layout
    - source tile: in Vec (UB), ND layout
    Data flow:
      x   (GM) -> Vec (UB) [target]
      src (GM) -> Vec (UB) [source]
      TINSERT ND_VEC: Vec -> Vec [at offset]
      Vec -> GM

Concepts introduced:
  - pl.tile.assemble for inserting tiles at offsets
  - pl.slice for extracting sub-tiles with dynamic offsets
  - pl.range for InCore loops (DSL for-loop)
  - Nested loops with computed offsets
  - Acc->Mat vs Vec->Vec hardware modes

Programs (one representative per distinct pattern):
  TileAssembleAccMatProgram             -- Acc->Mat: matmul result -> target at offset
  TileAssembleVecProgram                -- Vec->Vec: single-shot insert
  TileAssembleRowByRowProgram           -- Vec->Vec: loop + pl.slice + assemble
  TileAssembleDoubleLoopProgram         -- Vec->Vec: nested loops + pl.slice
  TileAssembleLoopColBroadcastProgram   -- Vec->Vec: loop with column broadcast (no slice)
  TileAssembleDoubleLoopBroadcastProgram -- Vec->Vec: nested loops, quadrant broadcast

Run:  python examples/kernels/08_assemble.py
Next: examples/models/01_ffn.py
"""

import pypto.language as pl


@pl.program
class TileAssembleAccMatProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        a: pl.Tensor[[32, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target into Mat (L1)
        tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
        # Produce Acc (L0C, FP32) via matmul: GM -> Mat -> Left/Right -> matmul
        tile_a_l1 = pl.load(a, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        tile_a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_src = pl.matmul(tile_a, tile_b)
        # Assemble: insert tile_src into the right half of tile_x at offset [0, 16]
        result = pl.tile.assemble(tile_x, tile_src, [0, 16])
        # Move Mat -> Vec before store
        result_vec = pl.move(result, target_memory=pl.MemorySpace.Vec)
        out_y = pl.store(result_vec, [0, 0], y)
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
class TileAssembleVecProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        # Load target and source into Vec (UB) -- ND/RowMajor layout
        tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
        # Assemble: insert src into the left half of x at [0, 0] -- ND_VEC mode
        result = pl.tile.assemble(tile_x, tile_src, [0, 0])
        out_y = pl.store(result, [0, 0], y)
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
class TileAssembleRowByRowProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
        for i in pl.range(32):
            row = pl.slice(tile_src, [1, 16], [i, 0])
            tile_x = pl.tile.assemble(tile_x, row, [i, 0])
        out_y = pl.store(tile_x, [0, 0], y)
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
class TileAssembleDoubleLoopProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
        for b in pl.range(4):
            for i in pl.range(8):
                row = b * 8 + i
                tile_row = pl.slice(tile_src, [1, 16], [row, 0])
                tile_x = pl.tile.assemble(tile_x, tile_row, [row, 0])
        out_y = pl.store(tile_x, [0, 0], y)
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
class TileAssembleLoopColBroadcastProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 8], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, [0, 0], [32, 8], target_memory=pl.MemorySpace.Vec)
        for c in pl.range(4):
            tile_x = pl.tile.assemble(tile_x, tile_src, [0, c * 8])
        out_y = pl.store(tile_x, [0, 0], y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[32, 8], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y


@pl.program
class TileAssembleDoubleLoopBroadcastProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_assemble(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
        tile_src = pl.load(src, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
        for b in pl.range(2):
            for c in pl.range(2):
                tile_x = pl.tile.assemble(tile_x, tile_src, [b * 16, c * 16])
        out_y = pl.store(tile_x, [0, 0], y)
        return out_y

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[32, 32], pl.FP32],
        src: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        y = self.tile_assemble(x, src, y)
        return y


if __name__ == "__main__":
    for name, prog in [
        ("AccMat", TileAssembleAccMatProgram),
        ("Vec", TileAssembleVecProgram),
        ("RowByRow", TileAssembleRowByRowProgram),
        ("DoubleLoop", TileAssembleDoubleLoopProgram),
        ("LoopColBroadcast", TileAssembleLoopColBroadcastProgram),
        ("DoubleLoopBroadcast", TileAssembleDoubleLoopBroadcastProgram),
    ]:
        print(f"=== TileAssemble{name}Program ===")
        print(prog.as_python())
        print()
