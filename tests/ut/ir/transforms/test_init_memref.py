# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for InitMemRefPass."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir import MemorySpace


def _iter_assign_stmts(func):
    """Iterate all AssignStmt in function body."""

    def _visit(stmt):
        if isinstance(stmt, ir.AssignStmt):
            yield stmt
        elif isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                yield from _visit(child)
        elif isinstance(stmt, ir.ForStmt):
            yield from _visit(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            yield from _visit(stmt.then_body)
            if stmt.else_body is not None:
                yield from _visit(stmt.else_body)
        elif isinstance(stmt, ir.WhileStmt):
            yield from _visit(stmt.body)

    yield from _visit(func.body)


def _get_tile_types(func):
    """Get {var_name: tile_type} for all TileType variables with memrefs."""
    result = {}
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var.type, ir.TileType) and stmt.var.type.memref is not None:
            result[stmt.var.name_hint] = stmt.var.type
    return result


def _get_param_types(func):
    """Get {param_name: tensor_type} for all TensorType params with memrefs."""
    result = {}
    for param in func.params:
        if isinstance(param.type, ir.TensorType) and param.type.memref is not None:
            result[param.name_hint] = param.type
    return result


def _first_function(program):
    """Get the first function from a Program."""
    return next(iter(program.functions.values()))


def _is_tile_alloc_assign(stmt):
    """Return True if stmt is an AssignStmt wrapping a tile.alloc call."""
    return (
        isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "tile.alloc"
    )


def _assert_leading_allocs(func, count):
    """Assert that the first count statements in the body are tile.alloc assigns."""
    assert isinstance(func.body, ir.SeqStmts)
    assert len(func.body.stmts) >= count
    assert all(_is_tile_alloc_assign(stmt) for stmt in func.body.stmts[:count])


def _get_alloc_stmts(func):
    """Get tile.alloc AssignStmts from function body."""
    allocs = []
    for stmt in _iter_assign_stmts(func):
        if _is_tile_alloc_assign(stmt):
            allocs.append(stmt)
    return allocs


def test_init_memref_simple():
    """Test InitMemRefPass with a simple load-add-store sequence (FP32 64x64).

    Memory space assignment:
        params (input_a, input_b, output) -> DDR
        tile_a, tile_b (tile.load)       -> Vec (default target_memory)
        tile_sum (tile.add)              -> Vec (default for tile ops)
        result (tile.store)              -> DDR (shares memref with output param)

    Also verifies:
        - addr=-1 (unallocated) for all MemRefs
        - tile.alloc statements are created for non-DDR MemRefs
        - Alloc statements are prepended directly to the function body's SeqStmts
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
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_sum, [0, 0], output)
            return result

    After = passes.init_mem_ref()(Before)
    func = _first_function(After)

    # Verify body is normalized and allocs are prepended directly to the body
    _assert_leading_allocs(func, 3)

    # Verify param MemRefs: all DDR, addr=-1, size=16384
    param_types = _get_param_types(func)
    for name in ("input_a", "input_b", "output"):
        assert name in param_types, f"param {name} should have MemRef"
        tensor_type = param_types[name]
        assert tensor_type.memory_space == MemorySpace.DDR
        assert tensor_type.memref.addr_.value == -1
        assert tensor_type.memref.size_ == 16384  # 64*64*4

    # Verify tile MemRefs: Vec space, addr=-1, size=16384
    tile_types = _get_tile_types(func)
    for name in ("tile_a", "tile_b", "tile_sum"):
        assert name in tile_types, f"tile {name} should have MemRef"
        tile_type = tile_types[name]
        assert tile_type.memory_space == MemorySpace.Vec
        assert tile_type.memref.addr_.value == -1
        assert tile_type.memref.size_ == 16384

    # Verify tile.alloc statements exist for non-DDR MemRefs
    allocs = _get_alloc_stmts(func)
    assert len(allocs) == 3, f"Expected 3 alloc stmts (3 Vec tiles), got {len(allocs)}"
    for alloc in allocs:
        assert alloc.value.args[1].value == -1, "alloc addr should be -1"

    # Verify store result shares memref with output param
    result_memrefs = {}
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var.type, ir.TensorType) and stmt.var.type.memref is not None:
            result_memrefs[stmt.var.name_hint] = stmt.var.type.memref
    assert "result" in result_memrefs
    assert result_memrefs["result"] is param_types["output"].memref


def test_init_memref_matmul():
    """Test InitMemRefPass with load->move->matmul->store sequence (FP16 32x32).

    Memory space assignment:
        params (input_a, input_b, output) -> DDR
        tile_a_ub (tile.load, target_memory=MemorySpace.Vec) -> Vec
        tile_b_l1 (tile.load, target_memory=MemorySpace.Mat) -> Mat
        tile_a_l0a (tile.move, target_memory=MemorySpace.Left) -> Left
        tile_b_l0b (tile.move, target_memory=MemorySpace.Right) -> Right
        tile_result (tile.matmul)              -> Acc (fixed)
        result (tile.store)                    -> DDR (shares memref with output)
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
            result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_result, [0, 0], output)
            return result

    After = passes.init_mem_ref()(Before)
    func = _first_function(After)

    # Verify normalized structure
    _assert_leading_allocs(func, 5)

    # Verify param MemRefs: all DDR
    param_types = _get_param_types(func)
    for name in ("input_a", "input_b", "output"):
        assert param_types[name].memory_space == MemorySpace.DDR
        assert param_types[name].memref.addr_.value == -1

    # Verify tile MemRefs: correct memory spaces
    tile_types = _get_tile_types(func)
    expected_spaces = {
        "tile_a_ub": MemorySpace.Vec,
        "tile_b_l1": MemorySpace.Mat,
        "tile_a_l0a": MemorySpace.Left,
        "tile_b_l0b": MemorySpace.Right,
        "tile_result": MemorySpace.Acc,
    }
    for name, expected_space in expected_spaces.items():
        assert name in tile_types, f"tile {name} should have MemRef"
        tile_type = tile_types[name]
        assert tile_type.memory_space == expected_space, (
            f"{name}: expected {expected_space}, got {tile_type.memory_space}"
        )
        assert tile_type.memref.addr_.value == -1
        if name == "tile_result":
            assert tile_type.memref.size_ == 4096  # 32*32*4
        else:
            assert tile_type.memref.size_ == 2048  # 32*32*2

    # Verify tile.alloc statements: one for each non-DDR MemRef (5 total)
    allocs = _get_alloc_stmts(func)
    assert len(allocs) == 5, f"Expected 5 alloc stmts (Vec+Mat+Left+Right+Acc), got {len(allocs)}"

    # Verify store result shares memref with output param
    result_memrefs = {}
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var.type, ir.TensorType) and stmt.var.type.memref is not None:
            result_memrefs[stmt.var.name_hint] = stmt.var.type.memref
    assert "result" in result_memrefs
    assert result_memrefs["result"] is param_types["output"].memref


def _ci(value: int) -> ir.ConstInt:
    """Shorthand for creating a ConstInt with INDEX type."""
    return ir.ConstInt(value, ir.DataType.INDEX, ir.Span.unknown())


def test_init_memref_rejects_dynamic_tile_shape():
    """InitMemRef must fail fast when allocation shape is still dynamic."""
    span = ir.Span.unknown()

    dynamic_len = ir.Var("dynamic_len", ir.ScalarType(ir.DataType.INDEX), span)
    dynamic_tile_type = ir.TileType([_ci(1), dynamic_len], ir.DataType.FP32, memory_space=MemorySpace.Vec)
    dynamic_tile = ir.Var("dynamic_tile", dynamic_tile_type, span)

    tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"aiv_idx": 0}, dynamic_tile_type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(dynamic_tile, tpop_call, span), ir.ReturnStmt([dynamic_tile], span)], span
    )
    func = ir.Function("test_func", [], [dynamic_tile_type], body, span)
    program = ir.Program([func], "test_program", span)

    with pytest.raises(Exception, match="InitMemRef requires static shape"):
        passes.init_mem_ref()(program)


def test_init_memref_tile_with_preset_memory_space():
    """Regression: tile.tpop ops with preset memory_space must get matching MemRef space.

    tile.tpop_* ops use no_memory_spec(), so ResolveMemorySpace has no deduction
    function. It must fall back to the Call's return type memory_space (set by
    ExpandMixedKernel via CleanTileType). CreateMemRef also falls back to the
    TileType's own memory_space when the var is not in the visitor map.

    Without the fix, the MemRef defaults to DDR/Vec instead of the actual space.
    """
    span = ir.Span.unknown()

    # Parameter: a DDR tensor
    input_tensor = ir.Var("input_tensor", ir.TensorType([64, 64], ir.DataType.FP32), span)

    # A tile.load produces a Vec tile (tracked by MemRefUsageVisitor)
    tile_loaded = ir.Var(
        "tile_loaded", ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec), span
    )
    load_call = ir.Call(ir.Op("tile.load"), [input_tensor, _ci(0), _ci(0), _ci(64), _ci(64)], span)
    load_stmt = ir.AssignStmt(tile_loaded, load_call, span)

    # A tile.tpop produces a Vec tile (tracked by MemRefUsageVisitor as tile.* op)
    tile_from_tpop = ir.Var(
        "tile_from_tpop",
        ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec),
        span,
    )
    tpop_call = ir.Call(
        ir.Op("tile.tpop_from_aic"),
        [],
        {"aiv_idx": 0},
        ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec),
        span,
    )
    tpop_stmt = ir.AssignStmt(tile_from_tpop, tpop_call, span)

    # A Left-space tile from tile.tpop_from_aiv (no_memory_spec → uses return type)
    tile_left = ir.Var(
        "tile_left",
        ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Left),
        span,
    )
    tpop_call_left = ir.Call(
        ir.Op("tile.tpop_from_aiv"),
        [],
        {"aiv_idx": 0},
        ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Left),
        span,
    )
    tpop_left_stmt = ir.AssignStmt(tile_left, tpop_call_left, span)

    # tile.add uses tile_from_tpop (visited as arg → triggers ProcessNormalVar)
    tile_sum = ir.Var("tile_sum", ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec), span)
    add_call = ir.Call(ir.Op("tile.add"), [tile_loaded, tile_from_tpop], span)
    add_stmt = ir.AssignStmt(tile_sum, add_call, span)

    # Store to DDR
    result_var = ir.Var("result", ir.TensorType([64, 64], ir.DataType.FP32), span)
    store_call = ir.Call(ir.Op("tile.store"), [tile_sum, _ci(0), _ci(0), input_tensor], span)
    store_stmt = ir.AssignStmt(result_var, store_call, span)

    # Return
    return_stmt = ir.ReturnStmt([result_var], span)

    body = ir.SeqStmts(
        [load_stmt, tpop_stmt, tpop_left_stmt, add_stmt, store_stmt, return_stmt],
        span,
    )

    func = ir.Function(
        "test_func",
        [(input_tensor, ir.ParamDirection.In)],
        [ir.TensorType([64, 64], ir.DataType.FP32)],
        body,
        span,
    )
    program = ir.Program([func], "test_program", span)

    # Run InitMemRefPass
    after = passes.init_mem_ref()(program)
    result_func = _first_function(after)

    # Collect TileTypes from all TileType vars (TileType exposes .memory_space)
    tile_types = _get_tile_types(result_func)

    # tile_loaded: tracked by visitor → should be Vec
    assert "tile_loaded" in tile_types
    assert tile_types["tile_loaded"].memory_space == MemorySpace.Vec

    # tile_from_tpop: tracked by visitor, ResolveMemorySpace reads Call return type → Vec
    assert "tile_from_tpop" in tile_types
    assert tile_types["tile_from_tpop"].memory_space == MemorySpace.Vec, (
        f"Expected Vec for tpop tile, got {tile_types['tile_from_tpop'].memory_space}"
    )

    # tile_left: tracked by visitor, ResolveMemorySpace reads Call return type → Left
    assert "tile_left" in tile_types
    assert tile_types["tile_left"].memory_space == MemorySpace.Left, (
        f"Expected Left for tpop tile, got {tile_types['tile_left'].memory_space}"
    )

    # tile_sum: tracked by visitor → should be Vec
    assert "tile_sum" in tile_types
    assert tile_types["tile_sum"].memory_space == MemorySpace.Vec


def test_init_memref_untracked_tile_defaults_to_ddr():
    """Regression: untracked TileType vars must sync the DDR fallback onto TileType."""
    span = ir.Span.unknown()

    input_tensor = ir.Var("input_tensor", ir.TensorType([64, 64], ir.DataType.FP32), span)
    tile_loaded = ir.Var(
        "tile_loaded", ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec), span
    )
    tile_external = ir.Var("tile_external", ir.TileType([64, 64], ir.DataType.FP32), span)
    tile_sum = ir.Var("tile_sum", ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec), span)
    result_var = ir.Var("result", ir.TensorType([64, 64], ir.DataType.FP32), span)

    load_call = ir.Call(ir.Op("tile.load"), [input_tensor, _ci(0), _ci(0), _ci(64), _ci(64)], span)
    add_call = ir.Call(ir.Op("tile.add"), [tile_external, tile_loaded], span)
    store_call = ir.Call(ir.Op("tile.store"), [tile_sum, _ci(0), _ci(0), input_tensor], span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(tile_loaded, load_call, span),
            ir.AssignStmt(tile_sum, add_call, span),
            ir.AssignStmt(result_var, store_call, span),
            ir.ReturnStmt([result_var], span),
        ],
        span,
    )

    func = ir.Function(
        "test_func",
        [(input_tensor, ir.ParamDirection.In), (tile_external, ir.ParamDirection.In)],
        [ir.TensorType([64, 64], ir.DataType.FP32)],
        body,
        span,
    )
    program = ir.Program([func], "test_program", span)

    after = passes.init_mem_ref()(program)
    result_func = _first_function(after)

    add_stmt = next(
        stmt
        for stmt in _iter_assign_stmts(result_func)
        if stmt.var.name_hint == "tile_sum" and isinstance(stmt.value, ir.Call)
    )
    add_call = cast(ir.Call, add_stmt.value)
    external_tile = add_call.args[0]
    assert isinstance(external_tile, ir.Var)
    assert external_tile.name_hint == "tile_external"
    assert isinstance(external_tile.type, ir.TileType)
    external_tile_type = external_tile.type
    assert external_tile_type.memory_space == MemorySpace.DDR
    assert external_tile_type.memref is not None
    assert cast(ir.ConstInt, external_tile_type.memref.addr_).value == -1


def test_init_memref_single_stmt_body_prepends_alloc():
    """InitMemRef prepends allocs even when the function body is a single statement."""
    span = ir.Span.unknown()

    input_tensor = ir.Var("input_tensor", ir.TensorType([64, 64], ir.DataType.FP32), span)
    tile_loaded = ir.Var(
        "tile_loaded", ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec), span
    )

    load_call = ir.Call(ir.Op("tile.load"), [input_tensor, _ci(0), _ci(0), _ci(64), _ci(64)], span)
    body = ir.AssignStmt(tile_loaded, load_call, span)

    func = ir.Function("test_func", [(input_tensor, ir.ParamDirection.In)], [], body, span)
    program = ir.Program([func], "test_program", span)

    after = passes.init_mem_ref()(program)
    result_func = _first_function(after)

    assert isinstance(result_func.body, ir.SeqStmts)
    assert len(result_func.body.stmts) == 2
    assert isinstance(result_func.body.stmts[0], ir.AssignStmt)
    assert isinstance(result_func.body.stmts[0].value, ir.Call)
    assert result_func.body.stmts[0].value.op.name == "tile.alloc"
    assert isinstance(result_func.body.stmts[1], ir.AssignStmt)
    assert result_func.body.stmts[1].var.name_hint == "tile_loaded"

    allocs = _get_alloc_stmts(result_func)
    assert len(allocs) == 1
    assert allocs[0].value.op.name == "tile.alloc"
    assert allocs[0].var.name_hint.startswith("mem_vec_")


def _find_yield_stmt(stmt):
    """Recursively find YieldStmt in a statement tree."""
    if isinstance(stmt, ir.YieldStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            result = _find_yield_stmt(child)
            if result:
                return result
    return None


def test_init_memref_for_stmt_loop_carry_memref_relationships():
    """ForStmt loop-carry variables have two MemRef sharing groups after InitMemRef.

    Group A: initValue + iter_arg (iter_arg inherits from initValue)
    Group B: yield value + return_var (return_var inherits from yield value)

    Group A and B may have different MemRefs — the yield-to-iter_arg mismatch
    is resolved later by MemoryReuse (which inserts tile.move if needed).
    """
    span = ir.Span.unknown()

    input_tensor = ir.Var("input_tensor", ir.TensorType([64, 64], ir.DataType.FP32), span)
    init_tile = ir.Var(
        "init_tile", ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec), span
    )
    init_stmt = ir.AssignStmt(
        init_tile, ir.Call(ir.Op("tile.load"), [input_tensor, _ci(0), _ci(0), _ci(64), _ci(64)], span), span
    )

    other_tile = ir.Var(
        "other_tile", ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec), span
    )
    other_stmt = ir.AssignStmt(
        other_tile, ir.Call(ir.Op("tile.load"), [input_tensor, _ci(0), _ci(0), _ci(64), _ci(64)], span), span
    )

    iter_arg = ir.IterArg("acc_iter", ir.TileType([64, 64], ir.DataType.FP32), init_tile, span)
    next_tile = ir.Var(
        "acc_next", ir.TileType([64, 64], ir.DataType.FP32, memory_space=MemorySpace.Vec), span
    )
    next_stmt = ir.AssignStmt(next_tile, ir.Call(ir.Op("tile.add"), [iter_arg, other_tile], span), span)
    return_var = ir.Var("acc_out", ir.TileType([64, 64], ir.DataType.FP32), span)

    loop_body = ir.SeqStmts([next_stmt, ir.YieldStmt([next_tile], span)], span)
    loop_stmt = ir.ForStmt(
        ir.Var("i", ir.ScalarType(ir.DataType.INDEX), span),
        _ci(0),
        _ci(4),
        _ci(1),
        [iter_arg],
        loop_body,
        [return_var],
        span,
    )
    body = ir.SeqStmts([init_stmt, other_stmt, loop_stmt, ir.ReturnStmt([return_var], span)], span)
    func = ir.Function(
        "test_func",
        [(input_tensor, ir.ParamDirection.In)],
        [ir.TileType([64, 64], ir.DataType.FP32)],
        body,
        span,
    )
    program = ir.Program([func], "test_program", span)

    after = passes.init_mem_ref()(program)
    result_func = _first_function(after)

    loop_after = cast(
        ir.ForStmt,
        next(stmt for stmt in cast(ir.SeqStmts, result_func.body).stmts if isinstance(stmt, ir.ForStmt)),
    )
    loop_iter_arg = loop_after.iter_args[0]
    loop_return_var = loop_after.return_vars[0]
    yield_stmt = _find_yield_stmt(loop_after.body)
    assert yield_stmt is not None
    yield_var = yield_stmt.value[0]

    # Find initValue after transformation (iter_arg.init_value)
    init_value = loop_iter_arg.initValue
    assert isinstance(init_value, ir.Var)

    # All 4 variables must have TileType with MemRef in Vec space
    for name, var_type in [
        ("initValue", init_value.type),
        ("iter_arg", loop_iter_arg.type),
        ("yield_value", yield_var.type),
        ("return_var", loop_return_var.type),
    ]:
        assert isinstance(var_type, ir.TileType), f"{name} should be TileType"
        assert var_type.memref is not None, f"{name} should have MemRef"
        assert var_type.memory_space == MemorySpace.Vec, f"{name} should be Vec"

    # Group A: initValue and iter_arg share the same MemRef
    init_tile = cast(ir.TileType, init_value.type)
    iter_arg_tile = cast(ir.TileType, loop_iter_arg.type)
    yield_tile = cast(ir.TileType, yield_var.type)
    return_var_tile = cast(ir.TileType, loop_return_var.type)

    assert init_tile.shares_memref_with(iter_arg_tile), "initValue and iter_arg must share MemRef"

    # Group B: yield value and return_var share the same MemRef
    assert yield_tile.shares_memref_with(return_var_tile), "yield value and return_var must share MemRef"

    # Group A and B have different MemRefs (yield gets independent MemRef from tile.add)
    assert not yield_tile.shares_memref_with(iter_arg_tile), (
        "yield value should get its own MemRef from tile.add, not share with iter_arg"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
