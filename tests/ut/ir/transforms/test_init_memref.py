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
from pypto import ir, passes
from pypto.ir import MemorySpace


def _iter_assign_stmts(func):
    """Iterate all AssignStmt in function body (handles SeqStmts/OpStmts)."""
    if not isinstance(func.body, ir.SeqStmts):
        return
    for child in func.body.stmts:
        if isinstance(child, ir.OpStmts):
            yield from (s for s in child.stmts if isinstance(s, ir.AssignStmt))
        elif isinstance(child, ir.AssignStmt):
            yield child


def _get_tile_memrefs(func):
    """Get {var_name: memref} for all TileType variables."""
    result = {}
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var.type, ir.TileType) and stmt.var.type.memref is not None:
            result[stmt.var.name] = stmt.var.type.memref
    return result


def _get_param_memrefs(func):
    """Get {param_name: memref} for all TensorType params."""
    result = {}
    for param in func.params:
        if isinstance(param.type, ir.TensorType) and param.type.memref is not None:
            result[param.name] = param.type.memref
    return result


def _get_alloc_stmts(func):
    """Get block.alloc AssignStmts from function body."""
    allocs = []
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "block.alloc":
            allocs.append(stmt)
    return allocs


def test_init_memref_simple():
    """Test InitMemRefPass with a simple load-add-store sequence (FP32 64x64).

    Memory space assignment:
        params (input_a, input_b, output) -> DDR
        tile_a, tile_b (block.load)       -> Vec (default target_memory)
        tile_sum (block.add)              -> Vec (default for block ops)
        result (block.store)              -> DDR (shares memref with output param)

    Also verifies:
        - addr=-1 (unallocated) for all MemRefs
        - block.alloc statements are created for non-DDR MemRefs
        - Body is wrapped in SeqStmts/OpStmts structure
    """

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

    After = passes.init_mem_ref()(Before)
    func = list(After.functions.values())[0]

    # Verify body is normalized (SeqStmts > OpStmts structure)
    assert isinstance(func.body, ir.SeqStmts)
    assert isinstance(func.body.stmts[0], ir.OpStmts)

    # Verify param MemRefs: all DDR, addr=-1, size=16384
    param_memrefs = _get_param_memrefs(func)
    for name in ("input_a", "input_b", "output"):
        assert name in param_memrefs, f"param {name} should have MemRef"
        mr = param_memrefs[name]
        assert mr.memory_space_ == MemorySpace.DDR
        assert mr.addr_.value == -1
        assert mr.size_ == 16384  # 64*64*4

    # Verify tile MemRefs: Vec space, addr=-1, size=16384
    tile_memrefs = _get_tile_memrefs(func)
    for name in ("tile_a", "tile_b", "tile_sum"):
        assert name in tile_memrefs, f"tile {name} should have MemRef"
        mr = tile_memrefs[name]
        assert mr.memory_space_ == MemorySpace.Vec
        assert mr.addr_.value == -1
        assert mr.size_ == 16384

    # Verify block.alloc statements exist for non-DDR MemRefs
    allocs = _get_alloc_stmts(func)
    assert len(allocs) == 3, f"Expected 3 alloc stmts (3 Vec tiles), got {len(allocs)}"
    for alloc in allocs:
        assert alloc.value.args[1].value == -1, "alloc addr should be -1"

    # Verify store result shares memref with output param
    result_memrefs = {}
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var.type, ir.TensorType) and stmt.var.type.memref is not None:
            result_memrefs[stmt.var.name] = stmt.var.type.memref
    assert "result" in result_memrefs
    assert result_memrefs["result"] is param_memrefs["output"]


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

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP16],
            input_b: pl.Tensor[[32, 32], pl.FP16],
            output: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile_a_ub: pl.Tile[[32, 32], pl.FP16] = pl.load(
                input_a, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec
            )
            tile_b_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
                input_b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
            )
            tile_a_l0a: pl.Tile[[32, 32], pl.FP16] = pl.move(tile_a_ub, target_memory=pl.MemorySpace.Left)
            tile_b_l0b: pl.Tile[[32, 32], pl.FP16] = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_result: pl.Tile[[32, 32], pl.FP32] = pl.matmul(tile_a_l0a, tile_b_l0b)
            result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_result, [0, 0], [32, 32], output)
            return result

    After = passes.init_mem_ref()(Before)
    func = list(After.functions.values())[0]

    # Verify normalized structure
    assert isinstance(func.body, ir.SeqStmts)
    assert isinstance(func.body.stmts[0], ir.OpStmts)

    # Verify param MemRefs: all DDR
    param_memrefs = _get_param_memrefs(func)
    for name in ("input_a", "input_b", "output"):
        assert param_memrefs[name].memory_space_ == MemorySpace.DDR
        assert param_memrefs[name].addr_.value == -1

    # Verify tile MemRefs: correct memory spaces
    tile_memrefs = _get_tile_memrefs(func)
    expected_spaces = {
        "tile_a_ub": MemorySpace.Vec,
        "tile_b_l1": MemorySpace.Mat,
        "tile_a_l0a": MemorySpace.Left,
        "tile_b_l0b": MemorySpace.Right,
        "tile_result": MemorySpace.Acc,
    }
    for name, expected_space in expected_spaces.items():
        assert name in tile_memrefs, f"tile {name} should have MemRef"
        mr = tile_memrefs[name]
        assert mr.memory_space_ == expected_space, (
            f"{name}: expected {expected_space}, got {mr.memory_space_}"
        )
        assert mr.addr_.value == -1
        if name == "tile_result":
            assert mr.size_ == 4096  # 32*32*4
        else:
            assert mr.size_ == 2048  # 32*32*2

    # Verify block.alloc statements: one for each non-DDR MemRef (5 total)
    allocs = _get_alloc_stmts(func)
    assert len(allocs) == 5, f"Expected 5 alloc stmts (Vec+Mat+Left+Right+Acc), got {len(allocs)}"

    # Verify store result shares memref with output param
    result_memrefs = {}
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var.type, ir.TensorType) and stmt.var.type.memref is not None:
            result_memrefs[stmt.var.name] = stmt.var.type.memref
    assert "result" in result_memrefs
    assert result_memrefs["result"] is param_memrefs["output"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
