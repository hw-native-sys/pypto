# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for BasicMemoryReusePass using @pl.program with pl.Tile type."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.pypto_core import DataType


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure backend before each test (required by dependency analyzer)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    yield
    backend.reset_for_testing()


def _iter_assign_stmts(func):
    """Iterate all AssignStmt in function body (handles SeqStmts/OpStmts)."""
    if not isinstance(func.body, ir.SeqStmts):
        return
    for child in func.body.stmts:
        if isinstance(child, ir.OpStmts):
            for stmt in child.stmts:
                if isinstance(stmt, ir.AssignStmt):
                    yield stmt
        elif isinstance(child, ir.AssignStmt):
            yield child


def _get_var_type(func, var_name):
    """Extract ShapedType for a variable by name."""
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var, ir.MemRef):
            continue
        if stmt.var.name == var_name:
            if isinstance(stmt.var.type, ir.ShapedType):
                return stmt.var.type
    return None


def _assert_shares_memref(func, var_a, var_b):
    """Assert two variables share the same MemRef object."""
    type_a = _get_var_type(func, var_a)
    type_b = _get_var_type(func, var_b)
    assert type_a is not None, f"{var_a} should have ShapedType"
    assert type_b is not None, f"{var_b} should have ShapedType"
    assert type_a.shares_memref_with(type_b), f"{var_b} should share the same MemRef with {var_a}"


def _assert_not_shares_memref(func, var_a, var_b):
    """Assert two variables do NOT share the same MemRef object."""
    type_a = _get_var_type(func, var_a)
    type_b = _get_var_type(func, var_b)
    assert type_a is not None, f"{var_a} should have ShapedType"
    assert type_b is not None, f"{var_b} should have ShapedType"
    assert not type_a.shares_memref_with(type_b), f"{var_b} should NOT share MemRef with {var_a}"


def _prepare_and_run_memory_reuse(program):
    """Prepare IR with memrefs (test setup), then run the pass under test.

    init_mem_ref() is test setup that attaches memrefs to tiles.
    basic_memory_reuse() is the pass under test.
    """
    program = passes.init_mem_ref()(program)
    program = passes.basic_memory_reuse()(program)
    return list(program.functions.values())[0]


def _assert_all_have_memrefs(func):
    """Assert all ShapedType variables have memrefs assigned."""
    assert isinstance(func.body, ir.SeqStmts)
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var, ir.MemRef):
            continue
        if isinstance(stmt.var.type, ir.ShapedType):
            assert stmt.var.type.memref is not None, f"{stmt.var.name} should have a memref"


def _count_alloc_stmts(func):
    """Count tile.alloc AssignStmt in the function body."""
    count = 0
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "tile.alloc":
            count += 1
    return count


def _get_alloc_memref_ids(func):
    """Get the set of MemRef id_ values from tile.alloc statements."""
    ids = set()
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "tile.alloc":
            memref = stmt.var
            assert isinstance(memref, ir.MemRef), "tile.alloc LHS must be MemRef"
            ids.add(memref.id_)
    return ids


def _iter_all_assign_stmts(stmt):
    """Recursively iterate all AssignStmt in a statement tree."""
    if isinstance(stmt, ir.AssignStmt):
        yield stmt
    elif isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _iter_all_assign_stmts(child)
    elif isinstance(stmt, ir.OpStmts):
        for child in stmt.stmts:
            yield from _iter_all_assign_stmts(child)
    elif isinstance(stmt, ir.ForStmt):
        yield from _iter_all_assign_stmts(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield from _iter_all_assign_stmts(stmt.then_body)
        if stmt.else_body is not None:
            yield from _iter_all_assign_stmts(stmt.else_body)
    elif isinstance(stmt, ir.WhileStmt):
        yield from _iter_all_assign_stmts(stmt.body)


def _get_var_type_recursive(func, var_name):
    """Extract ShapedType for a variable by name, searching the full statement tree."""
    for stmt in _iter_all_assign_stmts(func.body):
        if isinstance(stmt.var, ir.MemRef):
            continue
        if stmt.var.name == var_name:
            if isinstance(stmt.var.type, ir.ShapedType):
                return stmt.var.type
    return None


def _assert_not_shares_memref_recursive(func, var_a, var_b):
    """Assert two variables do NOT share MemRef, searching the full statement tree."""
    type_a = _get_var_type_recursive(func, var_a)
    type_b = _get_var_type_recursive(func, var_b)
    assert type_a is not None, f"{var_a} should have ShapedType"
    assert type_b is not None, f"{var_b} should have ShapedType"
    assert not type_a.shares_memref_with(type_b), f"{var_b} should NOT share MemRef with {var_a}"


# ---------------------------------------------------------------------------
# Basic memory reuse tests
# ---------------------------------------------------------------------------


def test_simple():
    """tile_d reuses tile_a, tile_e reuses tile_b.

    Lifetimes: tile_a[5,7], tile_b[6,7], tile_c[7,8], tile_d[8,9], tile_e[9,10]
    tile_d[8,9] can reuse tile_a[5,7] (non-overlapping).
    tile_e[9,10] cannot reuse tile_a (occupied by tile_d), reuses tile_b[6,7].
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
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_c, tile_c)
            tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_d, tile_d)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_d")
    _assert_shares_memref(func, "tile_b", "tile_e")


def test_sequential():
    """Sequential chain: all tiles reuse tile_a (producer-consumer at same statement).

    Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
    Each consumer's def equals its input's last_use, so all chain to tile_a.
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
            tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_d, tile_d)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_c")
    _assert_shares_memref(func, "tile_b", "tile_d")
    _assert_shares_memref(func, "tile_c", "tile_e")


def test_different_sizes():
    """Different-shaped tiles cannot reuse each other's buffer.

    PTO codegen binds alloc_tile type to the buffer, so shape must match
    exactly. tile_e (64x64) reuses tile_a (64x64); tile_f (32x32) reuses
    tile_b (32x32); cross-shape reuse is forbidden despite sufficient size.
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[32, 32], pl.FP32],
            output_a: pl.Tensor[[64, 64], pl.FP32],
            output_b: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            _result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output_a)
            tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(input_b, [0, 0], [32, 32])
            _result_b: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output_b)
            tile_e: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_f: pl.Tile[[32, 32], pl.FP32] = pl.load(input_b, [0, 0], [32, 32])
            _result_e: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output_a)
            result_f: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_f, [0, 0], output_b)
            return result_f

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_e")
    _assert_shares_memref(func, "tile_b", "tile_f")
    _assert_not_shares_memref(func, "tile_a", "tile_f")
    _assert_not_shares_memref(func, "tile_b", "tile_e")


def test_empty_function():
    """Empty function should not crash."""

    @pl.program
    class Before:
        @pl.function
        def main(self, output: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            return output

    After = passes.basic_memory_reuse()(Before)
    func = list(After.functions.values())[0]

    assert func is not None
    assert func.name == "main"


def test_memref_sharing():
    """Chain: all tiles reuse tile_a (producer-consumer at same statement).

    Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,3], tile_d[3,4]
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_c")
    _assert_shares_memref(func, "tile_b", "tile_d")


def test_with_dependencies():
    """tile_d reuses tile_a, tile_e reuses tile_b (same as test_simple)."""

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
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
            tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_d, tile_d)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_d")
    _assert_shares_memref(func, "tile_b", "tile_e")


def test_transitive_conflict():
    """tile_c reuses tile_a, tile_d reuses tile_b, tile_e gets fresh memory.

    Lifetimes: tile_a[5,6], tile_b[6,7], tile_c[7,9], tile_d[8,9], tile_e[9,10]
    tile_c[7,9] reuses tile_a[5,6]. tile_d[8,9] cannot reuse tile_a (occupied by
    tile_c[7,9]), reuses tile_b[6,7]. tile_e[9,10] cannot reuse either (both occupied).
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_b, tile_b)
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
            tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_d)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_c")
    _assert_shares_memref(func, "tile_b", "tile_d")
    _assert_not_shares_memref(func, "tile_c", "tile_d")


def test_multiple_memory_spaces():
    """Memory reuse happens within the same memory space (UB tiles)."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output_a: pl.Tensor[[64, 64], pl.FP32],
            output_b: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
            _result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output_a)
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
            result_b: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output_b)
            return result_b

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_d")


# ---------------------------------------------------------------------------
# Alloc cleanup tests (manual IR construction with tile.alloc stmts)
# ---------------------------------------------------------------------------


def _build_program_with_allocs(tile_specs, op_specs):
    """Build a Program with tile.alloc stmts and operation stmts from specs."""
    span = ir.Span.unknown()
    idx = DataType.INDEX
    fp32 = DataType.FP32
    shape = [ir.ConstInt(64, idx, span), ir.ConstInt(64, idx, span)]
    tile_size = 16384

    memref_in = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, idx, span), tile_size, 0)
    memref_out = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, idx, span), tile_size, 1)
    tensor_in = ir.TensorType(shape, fp32, memref_in)
    tensor_out = ir.TensorType(shape, fp32, memref_out)

    param_in = ir.Var("input_a", tensor_in, span)
    param_out = ir.Var("output", tensor_out, span)

    var_map = {"input_a": param_in, "output": param_out}
    stmts = []

    for name, mid in tile_specs:
        mr = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(-1, idx, span), tile_size, mid)
        tt = ir.TileType(shape, fp32, mr, None)
        var_map[name] = ir.Var(name, tt, span)
        alloc_call = ir.Call(
            ir.get_op("tile.alloc"),
            [
                ir.ConstInt(ir.MemorySpace.Vec.value, idx, span),
                ir.ConstInt(-1, idx, span),
                ir.ConstInt(tile_size, idx, span),
                ir.ConstInt(mid, idx, span),
            ],
            tt,
            span,
        )
        stmts.append(ir.AssignStmt(mr, alloc_call, span))

    offsets = ir.MakeTuple([ir.ConstInt(0, idx, span), ir.ConstInt(0, idx, span)], span)
    sizes = ir.MakeTuple([ir.ConstInt(64, idx, span), ir.ConstInt(64, idx, span)], span)

    for var_name, op_name, arg_names in op_specs:
        args = [var_map[a] for a in arg_names]
        if op_name == "tile.store":
            call = ir.Call(ir.get_op(op_name), [args[0], offsets, param_out], tensor_out, span)
            result_var = ir.Var(var_name, tensor_out, span)
            var_map[var_name] = result_var
        elif op_name == "tile.load":
            result_var = var_map[var_name]
            call = ir.Call(ir.get_op(op_name), [args[0], offsets, sizes], result_var.type, span)
        else:
            result_var = var_map[var_name]
            call = ir.Call(ir.get_op(op_name), args, result_var.type, span)
        stmts.append(ir.AssignStmt(result_var, call, span))

    body = ir.SeqStmts(
        [ir.OpStmts(stmts, span), ir.ReturnStmt([var_map[op_specs[-1][0]]], span)], span
    )
    func = ir.Function(
        "main",
        [(param_in, ir.ParamDirection.In), (param_out, ir.ParamDirection.Out)],
        [tensor_out],
        body,
        span,
    )
    return ir.Program([func], "TestProgram", span)


def test_unused_alloc_removed_after_reuse():
    """Alloc stmts for MemRefs replaced by reuse should be removed."""
    prog = _build_program_with_allocs(
        tile_specs=[("tile_a", 10), ("tile_b", 11), ("tile_c", 12)],
        op_specs=[
            ("tile_a", "tile.load", ["input_a"]),
            ("tile_b", "tile.add", ["tile_a", "tile_a"]),
            ("tile_c", "tile.add", ["tile_b", "tile_b"]),
            ("result", "tile.store", ["tile_c"]),
        ],
    )
    assert _count_alloc_stmts(list(prog.functions.values())[0]) == 3
    after = passes.basic_memory_reuse()(prog)
    func = list(after.functions.values())[0]
    # tile_c reuses tile_a, so tile_c's alloc is removed. tile_a and tile_b allocs remain.
    assert _count_alloc_stmts(func) == 2


def test_no_reuse_with_overlapping_lifetimes():
    """When all lifetimes overlap, no reuse happens and all allocs are preserved."""
    prog = _build_program_with_allocs(
        tile_specs=[("tile_a", 10), ("tile_b", 11), ("tile_c", 12)],
        op_specs=[
            ("tile_a", "tile.load", ["input_a"]),
            ("tile_b", "tile.load", ["input_a"]),
            ("tile_c", "tile.add", ["tile_a", "tile_b"]),
            ("result", "tile.store", ["tile_c"]),
        ],
    )
    assert _count_alloc_stmts(list(prog.functions.values())[0]) == 3
    after = passes.basic_memory_reuse()(prog)
    func = list(after.functions.values())[0]
    # All lifetimes touch at boundaries — no reuse possible
    assert _count_alloc_stmts(func) == 3


# ---------------------------------------------------------------------------
# Dtype compatibility tests
# ---------------------------------------------------------------------------


def test_cast_output_does_not_reuse():
    """Cast changes dtype — cast output gets different-sized buffer, no cross-dtype reuse."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_cast: pl.Tile[[64, 64], pl.BF16] = pl.cast(tile_b, target_type=pl.BF16)
            tile_c: pl.Tile[[64, 64], pl.BF16] = pl.add(tile_cast, tile_cast)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    # FP32 tiles (16384 bytes) and BF16 tiles (8192 bytes) have different sizes,
    # so cross-dtype reuse is forbidden by the size-matching constraint.
    _assert_not_shares_memref(func, "tile_a", "tile_cast")
    _assert_not_shares_memref(func, "tile_b", "tile_cast")


def test_cast_among_regular_ops():
    """Cross-dtype reuse forbidden by size mismatch; same-dtype tiles reuse within group."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_cast: pl.Tile[[64, 64], pl.BF16] = pl.cast(tile_b, target_type=pl.BF16)
            tile_d: pl.Tile[[64, 64], pl.BF16] = pl.add(tile_cast, tile_cast)
            tile_e: pl.Tile[[64, 64], pl.BF16] = pl.add(tile_d, tile_d)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    # FP32 (16384 bytes) vs BF16 (8192 bytes) — different sizes, no cross-dtype reuse
    _assert_not_shares_memref(func, "tile_a", "tile_cast")
    _assert_not_shares_memref(func, "tile_b", "tile_cast")


# ---------------------------------------------------------------------------
# Fillpad compatibility tests
# ---------------------------------------------------------------------------


def test_fillpad_output_incompatible_with_input():
    """fillpad changes valid_shape and pad — output cannot reuse input."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64])
            padded: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_not_shares_memref(func, "tile_a", "padded")


def test_fillpad_same_attributes_can_reuse():
    """Two fillpad outputs with identical TileView attributes CAN reuse."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output_a: pl.Tensor[[64, 64], pl.FP32],
            output_b: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64])
            padded_a: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile_a)
            _res_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_a, [0, 0], output_a)
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64])
            padded_b: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile_b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_b, [0, 0], output_b)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_b")
    _assert_shares_memref(func, "padded_a", "padded_b")


# ---------------------------------------------------------------------------
# View operations tests (reshape/view/transpose)
# ---------------------------------------------------------------------------


def test_reshape_shares_memref_with_input():
    """Single reshape operation should share MemRef with input tile."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self, input_a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[4096, 1], pl.FP32] = pl.reshape(tile_a, [4096, 1])
            tile_c: pl.Tile[[4096, 1], pl.FP32] = pl.add(tile_b, tile_b)
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.reshape(tile_c, [64, 64])
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_b")
    _assert_shares_memref(func, "tile_c", "tile_d")


def test_reshape_chain_shares_memref():
    """Chained reshapes should all share the same MemRef."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self, input_a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[4096, 1], pl.FP32] = pl.reshape(tile_a, [4096, 1])
            tile_c: pl.Tile[[1, 4096], pl.FP32] = pl.reshape(tile_b, [1, 4096])
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.reshape(tile_c, [64, 64])
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "tile_b")
    _assert_shares_memref(func, "tile_b", "tile_c")
    _assert_shares_memref(func, "tile_c", "tile_d")
    _assert_shares_memref(func, "tile_a", "tile_d")


def test_reshape_not_broken_by_memory_reuse():
    """BasicMemoryReuse should propagate reuse to ALL variables sharing MemRef."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self, input_a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_c: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            _tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            _tile_b: pl.Tile[[4096, 1], pl.FP32] = pl.reshape(tile_a, [4096, 1])
            tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "_tile_b")
    _assert_shares_memref(func, "tile_a", "tile_c")
    _assert_shares_memref(func, "_tile_b", "tile_c")


def test_reshape_shared_buffer_can_be_reused_after_all_dead():
    """After all aliases are dead, shared buffer can be reused."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self, input_a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            _tile_b: pl.Tile[[4096, 1], pl.FP32] = pl.reshape(tile_a, [4096, 1])
            _tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
            tile_d: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_d, tile_d)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_shares_memref(func, "tile_a", "_tile_b")
    _assert_shares_memref(func, "tile_d", "tile_a")


# ---------------------------------------------------------------------------
# Inplace safety check tests
# ---------------------------------------------------------------------------


def test_inplace_unsafe_op_no_producer_consumer_reuse():
    """tile.recip must NOT reuse its input's buffer (not inplace-safe)."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(input_a, [0, 0], [32, 32])
            tile_b: pl.Tile[[32, 32], pl.FP32] = pl.recip(tile_a)
            result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_not_shares_memref(func, "tile_a", "tile_b")


def test_inplace_unsafe_op_allows_non_producer_consumer_reuse():
    """tile.recip output must never share buffer with its input regardless of dead buffers."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP32],
            input_c: pl.Tensor[[32, 32], pl.FP32],
            input_x: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(input_a, [0, 0], [32, 32])
            _s1: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_a, [0, 0], output)
            tile_c: pl.Tile[[32, 32], pl.FP32] = pl.load(input_c, [0, 0], [32, 32])
            _s2: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], output)
            tile_x: pl.Tile[[32, 32], pl.FP32] = pl.load(input_x, [0, 0], [32, 32])
            tile_b: pl.Tile[[32, 32], pl.FP32] = pl.recip(tile_x)
            result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_not_shares_memref(func, "tile_x", "tile_b")


def test_inplace_safe_op_no_producer_consumer_reuse():
    """tile.add output cannot reuse its direct input (touching lifetimes overlap)."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(input_a, [0, 0], [32, 32])
            tile_b: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_a, tile_a)
            result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_not_shares_memref(func, "tile_a", "tile_b")


def test_ands_no_producer_consumer_reuse():
    """tile.ands must NOT reuse its input's buffer (not inplace-safe)."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.INT32],
            output: pl.Tensor[[32, 32], pl.INT32],
        ) -> pl.Tensor[[32, 32], pl.INT32]:
            tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(input_a, [0, 0], [32, 32])
            tile_b: pl.Tile[[32, 32], pl.INT32] = pl.ands(tile_a, 255)
            result: pl.Tensor[[32, 32], pl.INT32] = pl.store(tile_b, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_not_shares_memref(func, "tile_a", "tile_b")


def test_ors_no_producer_consumer_reuse():
    """tile.ors must NOT reuse its input's buffer (not inplace-safe)."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.INT32],
            output: pl.Tensor[[32, 32], pl.INT32],
        ) -> pl.Tensor[[32, 32], pl.INT32]:
            tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(input_a, [0, 0], [32, 32])
            tile_b: pl.Tile[[32, 32], pl.INT32] = pl.ors(tile_a, 255)
            result: pl.Tensor[[32, 32], pl.INT32] = pl.store(tile_b, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_not_shares_memref(func, "tile_a", "tile_b")


def test_xors_no_producer_consumer_reuse():
    """tile.xors must NOT reuse its input's buffer (not inplace-safe)."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.INT32],
            input_b: pl.Tensor[[32, 32], pl.INT32],
            output: pl.Tensor[[32, 32], pl.INT32],
        ) -> pl.Tensor[[32, 32], pl.INT32]:
            tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(input_a, [0, 0], [32, 32])
            tile_tmp: pl.Tile[[32, 32], pl.INT32] = pl.load(input_b, [0, 0], [32, 32])
            tile_b: pl.Tile[[32, 32], pl.INT32] = pl.xors(tile_a, 255, tile_tmp)
            result: pl.Tensor[[32, 32], pl.INT32] = pl.store(tile_b, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_not_shares_memref(func, "tile_a", "tile_b")


def test_inplace_unsafe_two_level_transitive_chain():
    """tile.recip must not reuse a buffer occupied by its input via a two-level chain."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP32],
            input_u: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(input_a, [0, 0], [32, 32])
            tile_b: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_a, tile_a)
            _s1: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
            tile_u: pl.Tile[[32, 32], pl.FP32] = pl.load(input_u, [0, 0], [32, 32])
            tile_d: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_u, tile_u)
            _s2: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_u, [0, 0], output)
            tile_c: pl.Tile[[32, 32], pl.FP32] = pl.recip(tile_d)
            result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)

    _assert_all_have_memrefs(func)
    _assert_not_shares_memref(func, "tile_d", "tile_c")


# ---------------------------------------------------------------------------
# Yield and init_value aliasing tests
# ---------------------------------------------------------------------------


@pl.program
class _YieldTestProgram:
    """Shared program with two accumulators in a for-loop with yield."""

    @pl.function
    def main(
        self,
        input_a: pl.Tensor[[64, 64], pl.FP32],
        input_b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        gate_init: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
        up_init: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
        for _i, (gate_acc, up_acc) in pl.range(4, init_values=(gate_init, up_init)):
            chunk: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            gate_new: pl.Tile[[64, 64], pl.FP32] = pl.add(gate_acc, chunk)
            up_new: pl.Tile[[64, 64], pl.FP32] = pl.add(up_acc, chunk)
            gate_out, _up_out = pl.yield_(gate_new, up_new)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(gate_out, [0, 0], output)
        return result


def test_yield_prevents_aliasing_of_simultaneously_live_tiles():
    """Two tile accumulators inside a loop, both yielded, must NOT share MemRef."""
    func = _prepare_and_run_memory_reuse(_YieldTestProgram)
    _assert_not_shares_memref_recursive(func, "gate_new", "up_new")


def test_init_values_prevent_aliasing_of_loop_inputs():
    """Two tiles used as init_values must NOT share MemRef."""
    func = _prepare_and_run_memory_reuse(_YieldTestProgram)
    _assert_not_shares_memref_recursive(func, "gate_init", "up_init")


def test_return_prevents_aliasing_of_simultaneously_live_tiles():
    """Two tiles both live at the return point must NOT share MemRef."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output_a: pl.Tensor[[64, 64], pl.FP32],
            output_b: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
            tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
            result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output_a)
            _result_b: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output_b)
            return result_a

    func = _prepare_and_run_memory_reuse(Before)
    _assert_not_shares_memref_recursive(func, "tile_a", "tile_b")


def test_while_init_values_prevent_aliasing():
    """Two tiles used as while-loop init_values must NOT share MemRef."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[4], pl.FP32],
            input_b: pl.Tensor[[4], pl.FP32],
            output: pl.Tensor[[4], pl.FP32],
        ) -> pl.Tensor[[4], pl.FP32]:
            gate_init: pl.Tile[[4], pl.FP32] = pl.load(input_a, [0], [4])
            up_init: pl.Tile[[4], pl.FP32] = pl.load(input_b, [0], [4])
            n: pl.Scalar[pl.INT64] = 0
            for gate_acc, up_acc in pl.while_(init_values=(gate_init, up_init)):
                pl.cond(n < 4)
                chunk: pl.Tile[[4], pl.FP32] = pl.load(input_a, [0], [4])
                gate_new: pl.Tile[[4], pl.FP32] = pl.add(gate_acc, chunk)
                up_new: pl.Tile[[4], pl.FP32] = pl.add(up_acc, chunk)
                _gate_out, _up_out = pl.yield_(gate_new, up_new)
            result: pl.Tensor[[4], pl.FP32] = pl.store(_gate_out, [0], output)
            return result

    func = _prepare_and_run_memory_reuse(Before)
    _assert_not_shares_memref_recursive(func, "gate_init", "up_init")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
