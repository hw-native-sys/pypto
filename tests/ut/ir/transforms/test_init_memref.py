# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for InitMemRefPass."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir import MemorySpace
from pypto.ir.op import block

_span = ir.Span.unknown()


def test_init_memref_simple():
    """Test InitMemRefPass with a simple load-add-store sequence (FP32 64x64).

    Memory space assignment:
        params (input_a, input_b, output) -> DDR
        tile_a, tile_b (block.load)       -> Vec (default target_memory)
        tile_sum (block.add)              -> Vec (default for block ops)
        result (block.store)              -> DDR (shares memref with output param)
    """

    # --- Before IR (no MemRef, using @pl.program) ---
    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
            tile_sum: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_sum, [0, 0], [64, 64], output)
            return result

    # Run pass
    After = passes.init_mem_ref()(Before)

    # --- Expected IR (with MemRef) ---
    span = _span
    dim64 = ir.ConstInt(64, DataType.INT64, span)
    # Size = 64 * 64 * 4 (FP32) = 16384
    memref_input_a = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 16384, 0)
    memref_input_b = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 16384, 1)
    memref_output = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 16384, 2)
    memref_tile_a = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INT64, span), 16384, 3)
    memref_tile_b = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INT64, span), 16384, 4)
    memref_tile_sum = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INT64, span), 16384, 5)

    exp_input_a = ir.Var("input_a", ir.TensorType([64, 64], DataType.FP32, memref_input_a), span)
    exp_input_b = ir.Var("input_b", ir.TensorType([64, 64], DataType.FP32, memref_input_b), span)
    exp_output = ir.Var("output", ir.TensorType([64, 64], DataType.FP32, memref_output), span)

    exp_tile_a = ir.Var("tile_a", ir.TileType([dim64, dim64], DataType.FP32, memref_tile_a), span)
    exp_tile_b = ir.Var("tile_b", ir.TileType([dim64, dim64], DataType.FP32, memref_tile_b), span)
    exp_tile_sum = ir.Var("tile_sum", ir.TileType([dim64, dim64], DataType.FP32, memref_tile_sum), span)
    # store result shares memref with output param
    exp_result = ir.Var("result", ir.TensorType([64, 64], DataType.FP32, memref_output), span)

    expected_body = ir.SeqStmts(
        [
            ir.AssignStmt(exp_tile_a, block.load(exp_input_a, offsets=[0, 0], shapes=[64, 64]), span),
            ir.AssignStmt(exp_tile_b, block.load(exp_input_b, offsets=[0, 0], shapes=[64, 64]), span),
            ir.AssignStmt(exp_tile_sum, block.add(exp_tile_a, exp_tile_b), span),
            ir.AssignStmt(
                exp_result,
                block.store(exp_tile_sum, offsets=[0, 0], shapes=[64, 64], output_tensor=exp_output),
                span,
            ),
            ir.ReturnStmt([exp_result], span),
        ],
        span,
    )
    expected_func = ir.Function(
        "main",
        [exp_input_a, exp_input_b, exp_output],
        [ir.TensorType([64, 64], DataType.FP32)],
        expected_body,
        span,
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


def test_init_memref_matmul():
    """Test InitMemRefPass with load->move->matmul->store sequence (FP16 32x32).

    Memory space assignment:
        params (input_a, input_b, output) -> DDR
        tile_a_ub (block.load, target_memory=MemorySpace.Vec) -> Vec
        tile_b_l1 (block.load, target_memory=MemorySpace.Mat) -> Mat
        tile_a_l0a (block.move, target_memory=MemorySpace.Left) -> Left
        tile_b_l0b (block.move, target_memory=MemorySpace.Right) -> Right
        tile_result (block.matmul)              -> Acc (fixed)
        result (block.store)                    -> DDR (shares memref with output)
    """

    # --- Before IR (no MemRef, using @pl.program) ---
    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP16],
            input_b: pl.Tensor[[32, 32], pl.FP16],
            output: pl.Tensor[[32, 32], pl.FP16],
        ) -> pl.Tensor[[32, 32], pl.FP16]:
            tile_a_ub: pl.Tile[[32, 32], pl.FP16] = pl.load(
                input_a, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec
            )
            tile_b_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
                input_b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
            )
            tile_a_l0a: pl.Tile[[32, 32], pl.FP16] = pl.move(tile_a_ub, target_memory=pl.MemorySpace.Left)
            tile_b_l0b: pl.Tile[[32, 32], pl.FP16] = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_result: pl.Tile[[32, 32], pl.FP16] = pl.matmul(tile_a_l0a, tile_b_l0b)
            result: pl.Tensor[[32, 32], pl.FP16] = pl.store(tile_result, [0, 0], [32, 32], output)
            return result

    # Run pass
    After = passes.init_mem_ref()(Before)

    # --- Expected IR (with MemRef) ---
    span = _span
    dim32 = ir.ConstInt(32, DataType.INT64, span)
    # Size = 32 * 32 * 2 (FP16) = 2048
    memref_input_a = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 2048, 0)
    memref_input_b = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 2048, 1)
    memref_output = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 2048, 2)
    memref_ub = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INT64, span), 2048, 3)
    memref_l1 = ir.MemRef(ir.MemorySpace.Mat, ir.ConstInt(0, DataType.INT64, span), 2048, 4)
    memref_l0a = ir.MemRef(ir.MemorySpace.Left, ir.ConstInt(0, DataType.INT64, span), 2048, 5)
    memref_l0b = ir.MemRef(ir.MemorySpace.Right, ir.ConstInt(0, DataType.INT64, span), 2048, 6)
    memref_l0c = ir.MemRef(ir.MemorySpace.Acc, ir.ConstInt(0, DataType.INT64, span), 2048, 7)

    exp_input_a = ir.Var("input_a", ir.TensorType([32, 32], DataType.FP16, memref_input_a), span)
    exp_input_b = ir.Var("input_b", ir.TensorType([32, 32], DataType.FP16, memref_input_b), span)
    exp_output = ir.Var("output", ir.TensorType([32, 32], DataType.FP16, memref_output), span)

    exp_tile_a_ub = ir.Var("tile_a_ub", ir.TileType([dim32, dim32], DataType.FP16, memref_ub), span)
    exp_tile_b_l1 = ir.Var("tile_b_l1", ir.TileType([dim32, dim32], DataType.FP16, memref_l1), span)
    exp_tile_a_l0a = ir.Var("tile_a_l0a", ir.TileType([dim32, dim32], DataType.FP16, memref_l0a), span)
    exp_tile_b_l0b = ir.Var("tile_b_l0b", ir.TileType([dim32, dim32], DataType.FP16, memref_l0b), span)
    exp_tile_result = ir.Var("tile_result", ir.TileType([dim32, dim32], DataType.FP16, memref_l0c), span)
    # store result shares memref with output param
    exp_result = ir.Var("result", ir.TensorType([32, 32], DataType.FP16, memref_output), span)

    expected_body = ir.SeqStmts(
        [
            ir.AssignStmt(
                exp_tile_a_ub,
                block.load(exp_input_a, offsets=[0, 0], shapes=[32, 32], target_memory=MemorySpace.Vec),
                span,
            ),
            ir.AssignStmt(
                exp_tile_b_l1,
                block.load(exp_input_b, offsets=[0, 0], shapes=[32, 32], target_memory=MemorySpace.Mat),
                span,
            ),
            ir.AssignStmt(exp_tile_a_l0a, block.move(exp_tile_a_ub, target_memory=MemorySpace.Left), span),
            ir.AssignStmt(exp_tile_b_l0b, block.move(exp_tile_b_l1, target_memory=MemorySpace.Right), span),
            ir.AssignStmt(exp_tile_result, block.matmul(exp_tile_a_l0a, exp_tile_b_l0b), span),
            ir.AssignStmt(
                exp_result,
                block.store(exp_tile_result, offsets=[0, 0], shapes=[32, 32], output_tensor=exp_output),
                span,
            ),
            ir.ReturnStmt([exp_result], span),
        ],
        span,
    )
    expected_func = ir.Function(
        "main",
        [exp_input_a, exp_input_b, exp_output],
        [ir.TensorType([32, 32], DataType.FP16)],
        expected_body,
        span,
    )
    Expected = ir.Program([expected_func], "test_program", span)

    ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
