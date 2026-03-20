# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for MemoryReusePass with pre-attached MemRefs (no init_mem_ref dependency)."""

import math

import pytest
from pypto import ir, passes
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile
from pypto.pypto_core import DataType

_SPAN = ir.Span.unknown()


_IDX = DataType.INDEX
_FP32 = DataType.FP32
_FP16 = DataType.FP16
_BF16 = DataType.BF16
_INT32 = DataType.INT32


def _ci(val: int) -> ir.ConstInt:
    """Create ConstInt with INDEX type."""
    return ir.ConstInt(val, _IDX, _SPAN)


def _dtype_bytes(dtype: DataType) -> int:
    """Byte size per element for a given dtype."""
    if dtype in (_FP32, _INT32):
        return 4
    if dtype in (_FP16, _BF16):
        return 2
    if dtype == DataType.INT64:
        return 8
    raise ValueError(f"Unsupported dtype: {dtype}")


class _MemRefAlloc:
    """Auto-incrementing MemRef allocator for test IR construction."""

    def __init__(self, start_id: int = 0) -> None:
        self._next_id = start_id

    def vec(self, shape: list[int], dtype: DataType) -> ir.MemRef:
        """Create a Vec-space MemRef with unique ID."""
        size = math.prod(shape) * _dtype_bytes(dtype)
        mr = ir.MemRef(ir.MemorySpace.Vec, _ci(-1), size, self._next_id)
        self._next_id += 1
        return mr

    def ddr(self, shape: list[int], dtype: DataType) -> ir.MemRef:
        """Create a DDR-space MemRef with unique ID."""
        size = math.prod(shape) * _dtype_bytes(dtype)
        mr = ir.MemRef(ir.MemorySpace.DDR, _ci(-1), size, self._next_id)
        self._next_id += 1
        return mr


def _tile_t(
    shape: list[int], dtype: DataType, memref: ir.MemRef, space: ir.MemorySpace = ir.MemorySpace.Vec
) -> ir.TileType:
    """TileType with MemRef."""
    return ir.TileType(shape, dtype, memref, None, space)


def _tile_t_with_view(
    shape: list[int],
    dtype: DataType,
    memref: ir.MemRef,
    tile_view: ir.TileView,
    space: ir.MemorySpace = ir.MemorySpace.Vec,
) -> ir.TileType:
    """TileType with MemRef and TileView."""
    return ir.TileType(shape, dtype, memref, tile_view, space)


def _tensor_t(shape: list[int], dtype: DataType, memref: ir.MemRef | None = None) -> ir.TensorType:
    """TensorType with optional MemRef."""
    if memref is not None:
        return ir.TensorType(shape, dtype, memref)
    return ir.TensorType(shape, dtype)


def _build_program(build_fn):
    """Build a Program by calling build_fn(ib, f, alloc) inside a function/program context.

    Returns the constructed Program.
    """
    alloc = _MemRefAlloc()
    ib = IRBuilder()
    with ib.program("Test") as prog:
        with ib.function("main") as f:
            build_fn(ib, f, alloc)
        prog.add_function(f.get_result())
    return prog.get_result()


def _run_reuse(program: ir.Program) -> ir.Function:
    """Run memory_reuse pass and return the first function."""
    after = passes.memory_reuse()(program)
    return next(iter(after.functions.values()))


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _iter_all_assign_stmts(stmt):
    """Recursively iterate all AssignStmt in a statement tree."""
    if isinstance(stmt, ir.AssignStmt):
        yield stmt
    elif isinstance(stmt, ir.SeqStmts):
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


def _get_var_type(func, var_name):
    """Extract ShapedType for a variable by name (recursive search)."""
    for stmt in _iter_all_assign_stmts(func.body):
        if stmt.var.name_hint == var_name:
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


def _assert_all_have_memrefs(func):
    """Assert all ShapedType variables have memrefs assigned."""
    for stmt in _iter_all_assign_stmts(func.body):
        if isinstance(stmt.var.type, ir.ShapedType):
            assert stmt.var.type.memref is not None, f"{stmt.var.name_hint} should have a memref"


def _count_alloc_stmts(func):
    """Count tile.alloc AssignStmt in the function body."""
    count = 0
    for stmt in _iter_all_assign_stmts(func.body):
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "tile.alloc":
            count += 1
    return count


def _get_alloc_memref_ids(func):
    """Get the set of MemRef id_ values from tile.alloc statements."""
    ids = set()
    for stmt in _iter_all_assign_stmts(func.body):
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "tile.alloc":
            memref = stmt.var
            assert isinstance(memref, ir.MemRef), "tile.alloc LHS must be MemRef"
            ids.add(memref.id_)
    return ids


class TestBasic:
    """Core reuse logic: chain reuse, producer-consumer, size/shape, transitive conflicts."""

    def test_simple(self):
        """tile_c, tile_d, tile_e all chain-reuse tile_a; tile_b remains independent."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            input_b = f.param("input_b", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, b_mr, c_mr, d_mr, e_mr = (alloc.vec([64, 64], _FP32) for _ in range(5))
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            tile_b = ib.let(
                "tile_b", tile.load(input_b, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, b_mr)
            )
            tile_c = ib.let("tile_c", tile.add(tile_a, tile_b), type=_tile_t([64, 64], _FP32, c_mr))
            tile_d = ib.let("tile_d", tile.mul(tile_c, tile_c), type=_tile_t([64, 64], _FP32, d_mr))
            tile_e = ib.let("tile_e", tile.add(tile_d, tile_d), type=_tile_t([64, 64], _FP32, e_mr))
            result = ib.let(
                "result", tile.store(tile_e, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_e")

    def test_sequential(self):
        """Sequential chain: all tiles reuse tile_a (producer-consumer at same statement)."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, b_mr, c_mr, d_mr, e_mr = (alloc.vec([64, 64], _FP32) for _ in range(5))
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            tile_b = ib.let("tile_b", tile.add(tile_a, tile_a), type=_tile_t([64, 64], _FP32, b_mr))
            tile_c = ib.let("tile_c", tile.add(tile_b, tile_b), type=_tile_t([64, 64], _FP32, c_mr))
            tile_d = ib.let("tile_d", tile.add(tile_c, tile_c), type=_tile_t([64, 64], _FP32, d_mr))
            tile_e = ib.let("tile_e", tile.add(tile_d, tile_d), type=_tile_t([64, 64], _FP32, e_mr))
            result = ib.let(
                "result", tile.store(tile_e, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")
        _assert_shares_memref(func, "tile_c", "tile_e")

    def test_different_sizes(self):
        """Different-shaped tiles cannot reuse each other's buffer."""

        def build(ib, f, alloc):
            in_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            in_b = f.param("input_b", ir.TensorType([32, 32], _FP32))
            out_a_mr, out_b_mr = alloc.ddr([64, 64], _FP32), alloc.ddr([32, 32], _FP32)
            out_a = f.param("output_a", _tensor_t([64, 64], _FP32, out_a_mr), direction=ir.ParamDirection.Out)
            out_b = f.param("output_b", _tensor_t([32, 32], _FP32, out_b_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([32, 32], _FP32))
            a_mr, b_mr, e_mr, f_mr = (
                alloc.vec([64, 64], _FP32),
                alloc.vec([32, 32], _FP32),
                alloc.vec([64, 64], _FP32),
                alloc.vec([32, 32], _FP32),
            )
            tile_a = ib.let("tile_a", tile.load(in_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr))
            ib.let("_result_a", tile.store(tile_a, [0, 0], out_a), type=_tensor_t([64, 64], _FP32, out_a_mr))
            tile_b = ib.let("tile_b", tile.load(in_b, [0, 0], [32, 32]), type=_tile_t([32, 32], _FP32, b_mr))
            ib.let("_result_b", tile.store(tile_b, [0, 0], out_b), type=_tensor_t([32, 32], _FP32, out_b_mr))
            tile_e = ib.let("tile_e", tile.load(in_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, e_mr))
            tile_f = ib.let("tile_f", tile.load(in_b, [0, 0], [32, 32]), type=_tile_t([32, 32], _FP32, f_mr))
            ib.let("_result_e", tile.store(tile_e, [0, 0], out_a), type=_tensor_t([64, 64], _FP32, out_a_mr))
            result_f = ib.let(
                "result_f", tile.store(tile_f, [0, 0], out_b), type=_tensor_t([32, 32], _FP32, out_b_mr)
            )
            ib.return_stmt(result_f)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_e")
        _assert_shares_memref(func, "tile_b", "tile_f")
        _assert_not_shares_memref(func, "tile_a", "tile_f")
        _assert_not_shares_memref(func, "tile_b", "tile_e")

    def test_empty_function(self):
        """Empty function should not crash."""

        def build(ib, f, alloc):
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            ib.return_stmt(output)

        func = _run_reuse(_build_program(build))
        assert func is not None
        assert func.name == "main"

    def test_transitive_conflict(self):
        """Transitive conflict: tile_c and tile_d must NOT share memory."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, b_mr, c_mr, d_mr, e_mr = (alloc.vec([64, 64], _FP32) for _ in range(5))
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            tile_b = ib.let("tile_b", tile.add(tile_a, tile_a), type=_tile_t([64, 64], _FP32, b_mr))
            tile_c = ib.let("tile_c", tile.add(tile_b, tile_b), type=_tile_t([64, 64], _FP32, c_mr))
            tile_d = ib.let("tile_d", tile.add(tile_c, tile_c), type=_tile_t([64, 64], _FP32, d_mr))
            tile_e = ib.let("tile_e", tile.add(tile_c, tile_d), type=_tile_t([64, 64], _FP32, e_mr))
            result = ib.let(
                "result", tile.store(tile_e, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_not_shares_memref(func, "tile_c", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_e")


def _build_program_with_allocs(tile_specs, op_specs):
    """Build a Program with tile.alloc stmts and operation stmts from specs.

    Args:
        tile_specs: list of (name, memref_id) for Vec tiles.
        op_specs: list of (var_name, op_name, arg_names) defining operations.
            First op uses param "input_a" as arg; others reference earlier tile vars.
            Last op is always tile.store writing to param "output".
    """
    span = _SPAN
    shape = [_ci(64), _ci(64)]
    tile_size = 16384

    memref_in = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, _IDX, span), tile_size, 0)
    memref_out = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, _IDX, span), tile_size, 1)
    tensor_in = ir.TensorType(shape, _FP32, memref_in)
    tensor_out = ir.TensorType(shape, _FP32, memref_out)

    param_in = ir.Var("input_a", tensor_in, span)
    param_out = ir.Var("output", tensor_out, span)

    var_map = {"input_a": param_in, "output": param_out}
    memref_map = {}
    stmts = []

    for name, mid in tile_specs:
        mr = ir.MemRef(ir.MemorySpace.Vec, _ci(-1), tile_size, mid)
        memref_map[name] = mr
        tt = ir.TileType(shape, _FP32, mr, None, ir.MemorySpace.Vec)
        var_map[name] = ir.Var(name, tt, span)

        alloc_call = ir.Call(
            ir.get_op("tile.alloc"),
            [
                ir.ConstInt(ir.MemorySpace.Vec.value, _IDX, span),
                _ci(-1),
                ir.ConstInt(tile_size, _IDX, span),
                ir.ConstInt(mid, _IDX, span),
            ],
            span,
        )
        stmts.append(ir.AssignStmt(mr, alloc_call, span))

    offsets = ir.MakeTuple([_ci(0), _ci(0)], span)
    sizes = ir.MakeTuple([_ci(64), _ci(64)], span)

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

    body = ir.SeqStmts([*stmts, ir.ReturnStmt([var_map[op_specs[-1][0]]], span)], span)
    func = ir.Function(
        "main",
        [(param_in, ir.ParamDirection.In), (param_out, ir.ParamDirection.Out)],
        [tensor_out],
        body,
        span,
    )
    return ir.Program([func], "TestProgram", span)


class TestAllocCleanup:
    """Tests for redundant tile.alloc removal after memory reuse."""

    def test_unused_alloc_removed_after_reuse(self):
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

        assert _count_alloc_stmts(next(iter(prog.functions.values()))) == 3

        after = passes.memory_reuse()(prog)
        func = next(iter(after.functions.values()))

        assert _count_alloc_stmts(func) == 1, (
            f"Expected 1 alloc stmt after chain reuse, got {_count_alloc_stmts(func)}"
        )

        alloc_ids = _get_alloc_memref_ids(func)
        tile_a_type = _get_var_type(func, "tile_a")
        assert tile_a_type is not None and tile_a_type.memref is not None
        assert tile_a_type.memref.id_ in alloc_ids

    def test_partial_reuse_with_overlapping_lifetimes(self):
        """When some lifetimes truly overlap, partial reuse happens."""
        prog = _build_program_with_allocs(
            tile_specs=[("tile_a", 10), ("tile_b", 11), ("tile_c", 12)],
            op_specs=[
                ("tile_a", "tile.load", ["input_a"]),
                ("tile_b", "tile.load", ["input_a"]),
                ("tile_c", "tile.add", ["tile_a", "tile_b"]),
                ("result", "tile.store", ["tile_c"]),
            ],
        )

        assert _count_alloc_stmts(next(iter(prog.functions.values()))) == 3

        after = passes.memory_reuse()(prog)
        func = next(iter(after.functions.values()))

        assert _count_alloc_stmts(func) == 2, (
            f"Expected 2 alloc stmts (tile_c reuses tile_a), got {_count_alloc_stmts(func)}"
        )


class TestDtype:
    """Tests that tiles with different dtypes do NOT reuse each other's memory."""

    def test_cross_dtype_no_reuse_same_dtype_reuse(self):
        """Cross-dtype reuse forbidden; same-dtype tiles reuse within their group."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, b_mr = alloc.vec([64, 64], _FP32), alloc.vec([64, 64], _FP32)
            cast_mr, d_mr, e_mr = (alloc.vec([64, 64], _BF16) for _ in range(3))
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            tile_b = ib.let("tile_b", tile.add(tile_a, tile_a), type=_tile_t([64, 64], _FP32, b_mr))
            tile_cast = ib.let("tile_cast", tile.cast(tile_b, _BF16), type=_tile_t([64, 64], _BF16, cast_mr))
            tile_d = ib.let("tile_d", tile.add(tile_cast, tile_cast), type=_tile_t([64, 64], _BF16, d_mr))
            tile_e = ib.let("tile_e", tile.add(tile_d, tile_d), type=_tile_t([64, 64], _BF16, e_mr))
            result = ib.let(
                "result", tile.store(tile_e, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_cast")
        _assert_not_shares_memref(func, "tile_b", "tile_cast")
        _assert_not_shares_memref(func, "tile_a", "tile_d")
        _assert_not_shares_memref(func, "tile_b", "tile_e")
        _assert_shares_memref(func, "tile_cast", "tile_d")
        _assert_shares_memref(func, "tile_cast", "tile_e")


def _make_tile_view(valid_shape: list[int], pad: ir.PadValue = ir.PadValue.null) -> ir.TileView:
    """Create a TileView with given valid_shape and pad (other fields use defaults)."""
    vs = [_ci(v) for v in valid_shape]
    return ir.TileView(vs, [], _ci(0), ir.TileLayout.row_major, ir.TileLayout.none_box, 512, pad)


class TestFillpad:
    """Tests that fillpad output does NOT reuse input due to TileView differences."""

    def test_fillpad_output_incompatible_with_input(self):
        """fillpad changes valid_shape and pad: output cannot reuse input."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, p_mr = alloc.vec([64, 64], _FP32), alloc.vec([64, 64], _FP32)
            view_in = _make_tile_view([48, 64])
            view_pad = _make_tile_view([64, 64], ir.PadValue.max)
            tile_a = ib.let(
                "tile_a",
                tile.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64]),
                type=_tile_t_with_view([64, 64], _FP32, a_mr, view_in),
            )
            padded = ib.let(
                "padded",
                tile.fillpad(tile_a, pad_value=ir.PadValue.max),
                type=_tile_t_with_view([64, 64], _FP32, p_mr, view_pad),
            )
            result = ib.let(
                "result", tile.store(padded, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "padded")

    def test_fillpad_different_pad_no_reuse(self):
        """Two fillpad outputs with different pad values cannot reuse each other."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            oa_mr, ob_mr = alloc.ddr([64, 64], _FP32), alloc.ddr([64, 64], _FP32)
            out_a = f.param("output_a", _tensor_t([64, 64], _FP32, oa_mr), direction=ir.ParamDirection.Out)
            out_b = f.param("output_b", _tensor_t([64, 64], _FP32, ob_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, pmax_mr, b_mr, pmin_mr = (alloc.vec([64, 64], _FP32) for _ in range(4))
            view_in = _make_tile_view([48, 64])
            view_max = _make_tile_view([64, 64], ir.PadValue.max)
            view_min = _make_tile_view([64, 64], ir.PadValue.min)
            tile_a = ib.let(
                "tile_a",
                tile.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64]),
                type=_tile_t_with_view([64, 64], _FP32, a_mr, view_in),
            )
            padded_max = ib.let(
                "padded_max",
                tile.fillpad(tile_a, pad_value=ir.PadValue.max),
                type=_tile_t_with_view([64, 64], _FP32, pmax_mr, view_max),
            )
            ib.let("_res_a", tile.store(padded_max, [0, 0], out_a), type=_tensor_t([64, 64], _FP32, oa_mr))
            tile_b = ib.let(
                "tile_b",
                tile.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64]),
                type=_tile_t_with_view([64, 64], _FP32, b_mr, view_in),
            )
            padded_min = ib.let(
                "padded_min",
                tile.fillpad(tile_b, pad_value=ir.PadValue.min),
                type=_tile_t_with_view([64, 64], _FP32, pmin_mr, view_min),
            )
            result = ib.let(
                "result", tile.store(padded_min, [0, 0], out_b), type=_tensor_t([64, 64], _FP32, ob_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_not_shares_memref(func, "padded_max", "padded_min")

    def test_fillpad_same_pad_can_reuse(self):
        """Two fillpad outputs with identical TileView attributes CAN reuse."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            oa_mr, ob_mr = alloc.ddr([64, 64], _FP32), alloc.ddr([64, 64], _FP32)
            out_a = f.param("output_a", _tensor_t([64, 64], _FP32, oa_mr), direction=ir.ParamDirection.Out)
            out_b = f.param("output_b", _tensor_t([64, 64], _FP32, ob_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, pa_mr, b_mr, pb_mr = (alloc.vec([64, 64], _FP32) for _ in range(4))
            view_in = _make_tile_view([48, 64])
            view_max = _make_tile_view([64, 64], ir.PadValue.max)
            tile_a = ib.let(
                "tile_a",
                tile.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64]),
                type=_tile_t_with_view([64, 64], _FP32, a_mr, view_in),
            )
            padded_a = ib.let(
                "padded_a",
                tile.fillpad(tile_a, pad_value=ir.PadValue.max),
                type=_tile_t_with_view([64, 64], _FP32, pa_mr, view_max),
            )
            ib.let("_res_a", tile.store(padded_a, [0, 0], out_a), type=_tensor_t([64, 64], _FP32, oa_mr))
            tile_b = ib.let(
                "tile_b",
                tile.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64]),
                type=_tile_t_with_view([64, 64], _FP32, b_mr, view_in),
            )
            padded_b = ib.let(
                "padded_b",
                tile.fillpad(tile_b, pad_value=ir.PadValue.max),
                type=_tile_t_with_view([64, 64], _FP32, pb_mr, view_max),
            )
            result = ib.let(
                "result", tile.store(padded_b, [0, 0], out_b), type=_tensor_t([64, 64], _FP32, ob_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "padded_a", "padded_b")


class TestViewOps:
    """Tests for view operations (reshape) with memory reuse."""

    def test_reshape_chain_shares_memref(self):
        """Chained reshapes should all share the same MemRef."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr = alloc.vec([64, 64], _FP32)
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            tile_b = ib.let("tile_b", tile.reshape(tile_a, [4096, 1]), type=_tile_t([4096, 1], _FP32, a_mr))
            tile_c = ib.let("tile_c", tile.reshape(tile_b, [1, 4096]), type=_tile_t([1, 4096], _FP32, a_mr))
            tile_d = ib.let("tile_d", tile.reshape(tile_c, [64, 64]), type=_tile_t([64, 64], _FP32, a_mr))
            result = ib.let(
                "result", tile.store(tile_d, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "tile_b", "tile_c")
        _assert_shares_memref(func, "tile_c", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_d")

    def test_reshape_not_broken_by_memory_reuse(self):
        """MemoryReuse should propagate reuse to ALL variables sharing MemRef."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            c_mr, d_mr, a_mr, e_mr = (alloc.vec([64, 64], _FP32) for _ in range(4))
            # tile_c is dead before tile_a/tile_b are defined
            tile_c = ib.let(
                "tile_c", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, c_mr)
            )
            ib.let("_tile_d", tile.add(tile_c, tile_c), type=_tile_t([64, 64], _FP32, d_mr))
            # tile_a and _tile_b share MemRef (reshape = view alias)
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            ib.let("_tile_b", tile.reshape(tile_a, [4096, 1]), type=_tile_t([4096, 1], _FP32, a_mr))
            # MemoryReuse: tile_a reuses tile_c → _tile_b also gets tile_c's MemRef
            tile_e = ib.let("tile_e", tile.add(tile_a, tile_a), type=_tile_t([64, 64], _FP32, e_mr))
            result = ib.let(
                "result", tile.store(tile_e, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "_tile_b")
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "_tile_b", "tile_c")

    def test_reshape_shared_buffer_can_be_reused_after_all_dead(self):
        """After all aliases are dead, shared buffer can be reused."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, c_mr, d_mr, e_mr = (alloc.vec([64, 64], _FP32) for _ in range(4))
            # tile_a and _tile_b share MemRef
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            ib.let("_tile_b", tile.reshape(tile_a, [4096, 1]), type=_tile_t([4096, 1], _FP32, a_mr))
            ib.let("_tile_c", tile.add(tile_a, tile_a), type=_tile_t([64, 64], _FP32, c_mr))
            # Both tile_a and _tile_b are dead → tile_d can reuse the shared buffer
            tile_d = ib.let(
                "tile_d", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, d_mr)
            )
            tile_e = ib.let("tile_e", tile.add(tile_d, tile_d), type=_tile_t([64, 64], _FP32, e_mr))
            result = ib.let(
                "result", tile.store(tile_e, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "_tile_b")
        _assert_shares_memref(func, "tile_d", "tile_a")


class TestInplaceOps:
    """Tests verifying that ops marked not_inplace_safe block producer-consumer reuse."""

    def _build_simple_op_test(self, op_fn, shape, dtype):
        """Build a simple load → op → store program for inplace safety tests."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType(shape, dtype))
            out_mr = alloc.ddr(shape, dtype)
            output = f.param("output", _tensor_t(shape, dtype, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType(shape, dtype))
            a_mr, b_mr = alloc.vec(shape, dtype), alloc.vec(shape, dtype)
            tile_a = ib.let("tile_a", tile.load(input_a, [0, 0], shape), type=_tile_t(shape, dtype, a_mr))
            tile_b = ib.let("tile_b", op_fn(tile_a), type=_tile_t(shape, dtype, b_mr))
            result = ib.let(
                "result", tile.store(tile_b, [0, 0], output), type=_tensor_t(shape, dtype, out_mr)
            )
            ib.return_stmt(result)

        return _run_reuse(_build_program(build))

    def test_inplace_unsafe_op_no_producer_consumer_reuse(self):
        """tile.recip must NOT reuse its input's buffer."""
        func = self._build_simple_op_test(tile.recip, [32, 32], _FP32)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_inplace_unsafe_op_allows_non_producer_consumer_reuse(self):
        """tile.recip output must never share a buffer with its input."""

        def build(ib, f, alloc):
            in_a = f.param("input_a", ir.TensorType([32, 32], _FP32))
            in_c = f.param("input_c", ir.TensorType([32, 32], _FP32))
            in_x = f.param("input_x", ir.TensorType([32, 32], _FP32))
            out_mr = alloc.ddr([32, 32], _FP32)
            output = f.param("output", _tensor_t([32, 32], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([32, 32], _FP32))
            a_mr, c_mr, x_mr, b_mr = (alloc.vec([32, 32], _FP32) for _ in range(4))
            tile_a = ib.let("tile_a", tile.load(in_a, [0, 0], [32, 32]), type=_tile_t([32, 32], _FP32, a_mr))
            ib.let("_s1", tile.store(tile_a, [0, 0], output), type=_tensor_t([32, 32], _FP32, out_mr))
            tile_c = ib.let("tile_c", tile.load(in_c, [0, 0], [32, 32]), type=_tile_t([32, 32], _FP32, c_mr))
            ib.let("_s2", tile.store(tile_c, [0, 0], output), type=_tensor_t([32, 32], _FP32, out_mr))
            tile_x = ib.let("tile_x", tile.load(in_x, [0, 0], [32, 32]), type=_tile_t([32, 32], _FP32, x_mr))
            tile_b = ib.let("tile_b", tile.recip(tile_x), type=_tile_t([32, 32], _FP32, b_mr))
            result = ib.let(
                "result", tile.store(tile_b, [0, 0], output), type=_tensor_t([32, 32], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_x", "tile_b")

    def test_inplace_safe_op_allows_producer_consumer_reuse(self):
        """tile.add (inplace-safe) CAN reuse its input's buffer."""
        func = self._build_simple_op_test(lambda t: tile.add(t, t), [32, 32], _FP32)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")

    def test_ands_no_producer_consumer_reuse(self):
        """tile.ands must NOT reuse its input's buffer."""
        func = self._build_simple_op_test(lambda t: tile.ands(t, 255), [32, 32], _INT32)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_xors_no_producer_consumer_reuse(self):
        """tile.xors must NOT reuse its input's buffer."""

        def build(ib, f, alloc):
            in_a = f.param("input_a", ir.TensorType([32, 32], _INT32))
            in_b = f.param("input_b", ir.TensorType([32, 32], _INT32))
            out_mr = alloc.ddr([32, 32], _INT32)
            output = f.param("output", _tensor_t([32, 32], _INT32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([32, 32], _INT32))
            a_mr, tmp_mr, b_mr = (alloc.vec([32, 32], _INT32) for _ in range(3))
            tile_a = ib.let("tile_a", tile.load(in_a, [0, 0], [32, 32]), type=_tile_t([32, 32], _INT32, a_mr))
            tile_tmp = ib.let(
                "tile_tmp", tile.load(in_b, [0, 0], [32, 32]), type=_tile_t([32, 32], _INT32, tmp_mr)
            )
            tile_b = ib.let("tile_b", tile.xors(tile_a, 255, tile_tmp), type=_tile_t([32, 32], _INT32, b_mr))
            result = ib.let(
                "result", tile.store(tile_b, [0, 0], output), type=_tensor_t([32, 32], _INT32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_inplace_unsafe_two_level_transitive_chain(self):
        """tile.recip must not reuse a buffer occupied by its input via a two-level chain."""

        def build(ib, f, alloc):
            in_a = f.param("input_a", ir.TensorType([32, 32], _FP32))
            in_u = f.param("input_u", ir.TensorType([32, 32], _FP32))
            out_mr = alloc.ddr([32, 32], _FP32)
            output = f.param("output", _tensor_t([32, 32], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([32, 32], _FP32))
            a_mr, b_mr, u_mr, d_mr, c_mr = (alloc.vec([32, 32], _FP32) for _ in range(5))
            tile_a = ib.let("tile_a", tile.load(in_a, [0, 0], [32, 32]), type=_tile_t([32, 32], _FP32, a_mr))
            tile_b = ib.let("tile_b", tile.add(tile_a, tile_a), type=_tile_t([32, 32], _FP32, b_mr))
            ib.let("_s1", tile.store(tile_b, [0, 0], output), type=_tensor_t([32, 32], _FP32, out_mr))
            tile_u = ib.let("tile_u", tile.load(in_u, [0, 0], [32, 32]), type=_tile_t([32, 32], _FP32, u_mr))
            tile_d = ib.let("tile_d", tile.add(tile_u, tile_u), type=_tile_t([32, 32], _FP32, d_mr))
            ib.let("_s2", tile.store(tile_u, [0, 0], output), type=_tensor_t([32, 32], _FP32, out_mr))
            tile_c = ib.let("tile_c", tile.recip(tile_d), type=_tile_t([32, 32], _FP32, c_mr))
            result = ib.let(
                "result", tile.store(tile_c, [0, 0], output), type=_tensor_t([32, 32], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_d", "tile_c")


# ---------------------------------------------------------------------------
# ForStmt yield fixup helpers and tests
# ---------------------------------------------------------------------------


def _find_first_for_stmt(stmt):
    """Return the first ForStmt found in a statement tree."""
    if isinstance(stmt, ir.ForStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            found = _find_first_for_stmt(child)
            if found is not None:
                return found
    return None


def _has_tile_move(stmt):
    """Check if a statement tree contains a tile.move AssignStmt."""
    for s in _iter_all_assign_stmts(stmt):
        if isinstance(s.value, ir.Call) and s.value.op.name == "tile.move":
            return True
    return False


def _build_for_loop_program(init_mrs, yield_mrs, add_overlap=False, shape=None, dtype=None):
    """Build a Program with a ForStmt whose initValue/yield can have different MemRefs.

    Args:
        init_mrs: list of MemRef for each initValue/iter_arg (Group A).
        yield_mrs: list of MemRef for each yield value/return_var (Group B).
        add_overlap: if True, adds extra tile usage that prevents reuse between
            iter_arg and yield value (forces tile.move insertion).
    """
    if shape is None:
        shape = [64, 64]
    if dtype is None:
        dtype = _FP32
    n_iters = len(init_mrs)

    # Seed allocator past the max incoming MemRef ID to avoid collisions
    max_id = max(mr.id_ for mr in (*init_mrs, *yield_mrs))
    alloc = _MemRefAlloc(start_id=max_id + 1)
    input_tensor = ir.Var("input_tensor", ir.TensorType(shape, dtype), _SPAN)

    # Create init tiles (before loop)
    init_tiles = []
    init_stmts = []
    for i, mr in enumerate(init_mrs):
        init_tt = _tile_t(shape, dtype, mr)
        init_tile = ir.Var(f"init_{i}", init_tt, _SPAN)
        load_call = ir.Call(
            ir.get_op("tile.load"),
            [
                input_tensor,
                ir.MakeTuple([_ci(0)] * len(shape), _SPAN),
                ir.MakeTuple([_ci(s) for s in shape], _SPAN),
            ],
            init_tt,
            _SPAN,
        )
        init_stmts.append(ir.AssignStmt(init_tile, load_call, _SPAN))
        init_tiles.append(init_tile)

    # Create iter_args and return_vars
    iter_args = []
    return_vars = []
    for i in range(n_iters):
        ia = ir.IterArg(f"acc_{i}", _tile_t(shape, dtype, init_mrs[i]), init_tiles[i], _SPAN)
        iter_args.append(ia)
        rv = ir.Var(f"out_{i}", _tile_t(shape, dtype, yield_mrs[i]), _SPAN)
        return_vars.append(rv)

    # Build loop body
    body_stmts = []
    yield_values = []
    for i in range(n_iters):
        if add_overlap:
            # Load a temporary tile to keep iter_arg alive past next_i's def,
            # preventing reuse of iter_arg's MemRef by next_i.
            extra_mr = alloc.vec(shape, dtype)
            extra_var = ir.Var(f"extra_{i}", _tile_t(shape, dtype, extra_mr), _SPAN)
            extra_call = ir.Call(ir.get_op("tile.add"), [iter_args[i], iter_args[i]], _SPAN)
            body_stmts.append(ir.AssignStmt(extra_var, extra_call, _SPAN))
            # next_i uses extra_i (iter_arg still alive via extra_i computation)
            next_tt = _tile_t(shape, dtype, yield_mrs[i])
            next_var = ir.Var(f"next_{i}", next_tt, _SPAN)
            add_call = ir.Call(ir.get_op("tile.add"), [extra_var, iter_args[i]], next_tt, _SPAN)
            body_stmts.append(ir.AssignStmt(next_var, add_call, _SPAN))
        else:
            next_tt = _tile_t(shape, dtype, yield_mrs[i])
            next_var = ir.Var(f"next_{i}", next_tt, _SPAN)
            add_call = ir.Call(ir.get_op("tile.add"), [iter_args[i], iter_args[i]], next_tt, _SPAN)
            body_stmts.append(ir.AssignStmt(next_var, add_call, _SPAN))
        yield_values.append(next_var)

    loop_body = ir.SeqStmts([*body_stmts, ir.YieldStmt(yield_values, _SPAN)], _SPAN)

    loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), _SPAN)
    loop_stmt = ir.ForStmt(loop_var, _ci(0), _ci(4), _ci(1), iter_args, loop_body, return_vars, _SPAN)

    # Store first return_var and return
    out_mr = alloc.ddr(shape, dtype)
    out_tensor = ir.Var("output", _tensor_t(shape, dtype, out_mr), _SPAN)
    store_call = ir.Call(
        ir.get_op("tile.store"),
        [return_vars[0], ir.MakeTuple([_ci(0)] * len(shape), _SPAN), out_tensor],
        _tensor_t(shape, dtype, out_mr),
        _SPAN,
    )
    result_var = ir.Var("result", _tensor_t(shape, dtype, out_mr), _SPAN)
    store_stmt = ir.AssignStmt(result_var, store_call, _SPAN)

    body = ir.SeqStmts([*init_stmts, loop_stmt, store_stmt, ir.ReturnStmt([result_var], _SPAN)], _SPAN)
    func = ir.Function(
        "main",
        [(input_tensor, ir.ParamDirection.In), (out_tensor, ir.ParamDirection.Out)],
        [_tensor_t(shape, dtype)],
        body,
        _SPAN,
    )
    return ir.Program([func], "TestProgram", _SPAN)


class TestYieldFixup:
    """Yield fixup for ForStmt and IfStmt — ensuring loop-carry and return variables share correct MemRef."""

    def test_tile_move_inserted_when_memrefs_diverge(self):
        """When initValue and yield value start with different MemRefs,
        the pass should unify all loop-carry vars to share one MemRef."""
        alloc = _MemRefAlloc()
        init_mr = alloc.vec([64, 64], _FP32)
        yield_mr = alloc.vec([64, 64], _FP32)
        assert init_mr.id_ != yield_mr.id_, "precondition: MemRefs start different"
        # add_overlap=True adds extra usage to prevent trivial producer-consumer reuse
        prog = _build_for_loop_program([init_mr], [yield_mr], add_overlap=True)

        after = passes.memory_reuse()(prog)
        func = next(iter(after.functions.values()))

        loop = _find_first_for_stmt(func.body)
        assert loop is not None

        # After fixup: iter_arg, initValue, and return_var should all share one MemRef
        ia = loop.iter_args[0]
        assert isinstance(ia.initValue.type, ir.ShapedType)
        assert isinstance(ia.type, ir.ShapedType)
        assert ia.type.shares_memref_with(ia.initValue.type), "iter_arg should share initValue's MemRef"

        rv = loop.return_vars[0]
        assert isinstance(rv.type, ir.ShapedType)
        assert rv.type.shares_memref_with(ia.type), "return_var should share iter_arg's MemRef"

    def test_no_tile_move_when_memrefs_match(self):
        """When initValue and yield value already share MemRef, no tile.move is needed."""
        alloc = _MemRefAlloc()
        shared_mr = alloc.vec([64, 64], _FP32)
        prog = _build_for_loop_program([shared_mr], [shared_mr])

        after = passes.memory_reuse()(prog)
        func = next(iter(after.functions.values()))

        loop = _find_first_for_stmt(func.body)
        assert loop is not None
        assert not _has_tile_move(loop.body), "No tile.move needed when MemRefs already match"

        ia = loop.iter_args[0]
        assert isinstance(ia.initValue.type, ir.ShapedType)
        assert isinstance(ia.type, ir.ShapedType)
        assert ia.type.shares_memref_with(ia.initValue.type)

        rv = loop.return_vars[0]
        assert isinstance(rv.type, ir.ShapedType)
        assert rv.type.shares_memref_with(ia.type), "return_var should share iter_arg's MemRef"

    def test_multiple_iter_args_partial_mismatch(self):
        """With 2 iter_args, tile.move inserted only for the mismatched pair."""
        alloc = _MemRefAlloc()
        # First iter_arg: MemRefs match (no move needed)
        shared_mr = alloc.vec([64, 64], _FP32)
        # Second iter_arg: MemRefs differ (move needed)
        init_mr_2 = alloc.vec([64, 64], _FP32)
        yield_mr_2 = alloc.vec([64, 64], _FP32)

        prog = _build_for_loop_program([shared_mr, init_mr_2], [shared_mr, yield_mr_2], add_overlap=True)

        after = passes.memory_reuse()(prog)
        func = next(iter(after.functions.values()))

        loop = _find_first_for_stmt(func.body)
        assert loop is not None
        assert len(loop.iter_args) == 2

        # Both iter_args should share their initValue's MemRef, and return_vars should match
        for i in range(2):
            ia = loop.iter_args[i]
            assert isinstance(ia.initValue.type, ir.ShapedType)
            assert isinstance(ia.type, ir.ShapedType)
            assert ia.type.shares_memref_with(ia.initValue.type), (
                f"iter_arg[{i}] should share initValue's MemRef"
            )
            rv = loop.return_vars[i]
            assert isinstance(rv.type, ir.ShapedType)
            assert rv.type.shares_memref_with(ia.type), f"return_var[{i}] should share iter_arg's MemRef"

    def test_if_stmt_return_var_memref_patched(self):
        """After reuse changes a branch variable's MemRef, the IfStmt's
        return_var should be patched to reflect the updated MemRef."""
        span = _SPAN
        shape = [64, 64]
        alloc = _MemRefAlloc()

        input_tensor = ir.Var("input_tensor", ir.TensorType(shape, _FP32), span)
        cond_param = ir.Var("cond_param", ir.ScalarType(DataType.INDEX), span)
        out_mr = alloc.ddr(shape, _FP32)
        output = ir.Var("output", _tensor_t(shape, _FP32, out_mr), span)

        # tile_a: dead before IfStmt
        a_mr = alloc.vec(shape, _FP32)
        a_tt = _tile_t(shape, _FP32, a_mr)
        tile_a = ir.Var("tile_a", a_tt, span)
        load_a = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            a_tt,
            span,
        )
        # Consume tile_a immediately so it's dead before IfStmt
        store_a = ir.Call(
            ir.get_op("tile.store"),
            [tile_a, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        store_a_var = ir.Var("store_a", _tensor_t(shape, _FP32, out_mr), span)

        # Then branch: tile_b = load (could reuse tile_a's MemRef)
        b_mr = alloc.vec(shape, _FP32)
        b_tt = _tile_t(shape, _FP32, b_mr)
        tile_b = ir.Var("tile_b", b_tt, span)
        load_b = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            b_tt,
            span,
        )
        then_body = ir.SeqStmts([ir.AssignStmt(tile_b, load_b, span), ir.YieldStmt([tile_b], span)], span)

        # Else branch: tile_c = load (could also reuse tile_a)
        c_mr = alloc.vec(shape, _FP32)
        c_tt = _tile_t(shape, _FP32, c_mr)
        tile_c = ir.Var("tile_c", c_tt, span)
        load_c = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            c_tt,
            span,
        )
        else_body = ir.SeqStmts([ir.AssignStmt(tile_c, load_c, span), ir.YieldStmt([tile_c], span)], span)

        # IfStmt with return_var (initially pointing to a different MemRef)
        rv_mr = alloc.vec(shape, _FP32)
        rv = ir.Var("if_result", _tile_t(shape, _FP32, rv_mr), span)
        cond = ir.Lt(cond_param, _ci(2), _IDX, span)
        if_stmt = ir.IfStmt(cond, then_body, else_body, [rv], span)

        # Store if_result
        store_rv = ir.Call(
            ir.get_op("tile.store"),
            [rv, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        result_var = ir.Var("result", _tensor_t(shape, _FP32, out_mr), span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(tile_a, load_a, span),
                ir.AssignStmt(store_a_var, store_a, span),
                if_stmt,
                ir.AssignStmt(result_var, store_rv, span),
                ir.ReturnStmt([result_var], span),
            ],
            span,
        )
        func = ir.Function(
            "main",
            [
                (input_tensor, ir.ParamDirection.In),
                (cond_param, ir.ParamDirection.In),
                (output, ir.ParamDirection.Out),
            ],
            [ir.TensorType(shape, _FP32)],
            body,
            span,
        )
        prog = ir.Program([func], "Test", span)

        after = passes.memory_reuse()(prog)
        func_out = next(iter(after.functions.values()))

        # tile_b and tile_c should reuse tile_a (tile_a is dead before IfStmt)
        _assert_shares_memref(func_out, "tile_a", "tile_b")
        _assert_shares_memref(func_out, "tile_b", "tile_c")

        # After reuse, if_result's MemRef should be patched by YieldFixupMutator
        # to match the then-branch yield value's MemRef (which is now tile_a's MemRef)
        if_result_type = _get_var_type(func_out, "if_result")
        tile_b_type = _get_var_type(func_out, "tile_b")
        if if_result_type is not None and tile_b_type is not None:
            assert if_result_type.shares_memref_with(tile_b_type), (
                "if_result should share MemRef with tile_b after YieldFixupMutator patches it"
            )


class TestControlFlow:
    """Tests for correct lifetime analysis across control flow boundaries."""

    def test_var_used_in_nested_if_not_reused_in_loop(self):
        """Variable defined before loop, used inside IfStmt within loop body,
        must NOT have its MemRef reused by other loop-body variables."""
        span = _SPAN
        shape = [64, 64]
        alloc = _MemRefAlloc()

        # Params
        input_tensor = ir.Var("input_tensor", ir.TensorType(shape, _FP32), span)
        out_mr = alloc.ddr(shape, _FP32)
        output = ir.Var("output", _tensor_t(shape, _FP32, out_mr), span)

        # tile_a: defined before loop, used inside IfStmt in loop body
        a_mr = alloc.vec(shape, _FP32)
        tile_a_tt = _tile_t(shape, _FP32, a_mr)
        tile_a = ir.Var("tile_a", tile_a_tt, span)
        load_call = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            tile_a_tt,
            span,
        )
        load_stmt = ir.AssignStmt(tile_a, load_call, span)

        # Loop: for i in range(4)
        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)

        # iter_arg: acc initialized with tile_a
        acc_mr = alloc.vec(shape, _FP32)
        acc_tt = _tile_t(shape, _FP32, acc_mr)
        ia = ir.IterArg("acc", acc_tt, tile_a, span)

        # Inside loop body:
        #   if (i < 2):
        #     tile_c = tile.add(acc, tile_a)   ← tile_a used here!
        #     yield tile_c
        #   else:
        #     yield acc
        c_mr = alloc.vec(shape, _FP32)
        c_tt = _tile_t(shape, _FP32, c_mr)
        tile_c = ir.Var("tile_c", c_tt, span)
        add_call = ir.Call(ir.get_op("tile.add"), [ia, tile_a], c_tt, span)
        then_assign = ir.AssignStmt(tile_c, add_call, span)

        # Then branch: assign tile_c, yield tile_c
        then_yield = ir.YieldStmt([tile_c], span)
        then_body = ir.SeqStmts([then_assign, then_yield], span)

        # Else branch: yield acc
        else_yield = ir.YieldStmt([ia], span)
        else_body = else_yield

        # IfStmt with return_var
        if_rv_mr = alloc.vec(shape, _FP32)
        if_rv = ir.Var("if_result", _tile_t(shape, _FP32, if_rv_mr), span)
        cond = ir.Lt(loop_var, _ci(2), _IDX, span)
        if_stmt = ir.IfStmt(cond, then_body, else_body, [if_rv], span)

        # Loop body: IfStmt then yield if_rv
        loop_yield = ir.YieldStmt([if_rv], span)
        loop_body = ir.SeqStmts([if_stmt, loop_yield], span)

        # ForStmt
        rv_mr = alloc.vec(shape, _FP32)
        rv = ir.Var("loop_out", _tile_t(shape, _FP32, rv_mr), span)
        for_stmt = ir.ForStmt(loop_var, _ci(0), _ci(4), _ci(1), [ia], loop_body, [rv], span)

        # Store and return
        store_call = ir.Call(
            ir.get_op("tile.store"),
            [rv, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        result_var = ir.Var("result", _tensor_t(shape, _FP32, out_mr), span)
        store_stmt = ir.AssignStmt(result_var, store_call, span)

        body = ir.SeqStmts(
            [load_stmt, for_stmt, store_stmt, ir.ReturnStmt([result_var], span)],
            span,
        )
        func = ir.Function(
            "main",
            [(input_tensor, ir.ParamDirection.In), (output, ir.ParamDirection.Out)],
            [ir.TensorType(shape, _FP32)],
            body,
            span,
        )
        prog = ir.Program([func], "Test", span)

        after = passes.memory_reuse()(prog)
        func_out = next(iter(after.functions.values()))

        # tile_a must NOT share MemRef with tile_c — tile_a is live through the loop
        _assert_not_shares_memref(func_out, "tile_a", "tile_c")

    def test_different_if_branches_can_share(self):
        """Variables in different IfStmt branches should be able to share MemRef
        since they have non-overlapping lifetimes."""
        span = _SPAN
        shape = [64, 64]
        alloc = _MemRefAlloc()

        input_tensor = ir.Var("input_tensor", ir.TensorType(shape, _FP32), span)
        out_mr = alloc.ddr(shape, _FP32)
        output = ir.Var("output", _tensor_t(shape, _FP32, out_mr), span)

        # Create a scalar condition
        cond_param = ir.Var("cond_param", ir.ScalarType(DataType.INDEX), span)

        # Then branch: tile_b = tile.load(...)
        b_mr = alloc.vec(shape, _FP32)
        b_tt = _tile_t(shape, _FP32, b_mr)
        tile_b = ir.Var("tile_b", b_tt, span)
        load_b = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            b_tt,
            span,
        )
        then_body = ir.SeqStmts([ir.AssignStmt(tile_b, load_b, span), ir.YieldStmt([tile_b], span)], span)

        # Else branch: tile_c = tile.load(...)
        c_mr = alloc.vec(shape, _FP32)
        c_tt = _tile_t(shape, _FP32, c_mr)
        tile_c = ir.Var("tile_c", c_tt, span)
        load_c = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            c_tt,
            span,
        )
        else_body = ir.SeqStmts([ir.AssignStmt(tile_c, load_c, span), ir.YieldStmt([tile_c], span)], span)

        # IfStmt with return_var
        rv_mr = alloc.vec(shape, _FP32)
        rv = ir.Var("if_result", _tile_t(shape, _FP32, rv_mr), span)
        cond = ir.Lt(cond_param, _ci(2), _IDX, span)
        if_stmt = ir.IfStmt(cond, then_body, else_body, [rv], span)

        # Store and return
        store_call = ir.Call(
            ir.get_op("tile.store"),
            [rv, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        result_var = ir.Var("result", _tensor_t(shape, _FP32, out_mr), span)

        body = ir.SeqStmts(
            [if_stmt, ir.AssignStmt(result_var, store_call, span), ir.ReturnStmt([result_var], span)],
            span,
        )
        func = ir.Function(
            "main",
            [
                (input_tensor, ir.ParamDirection.In),
                (cond_param, ir.ParamDirection.In),
                (output, ir.ParamDirection.Out),
            ],
            [ir.TensorType(shape, _FP32)],
            body,
            span,
        )
        prog = ir.Program([func], "Test", span)

        after = passes.memory_reuse()(prog)
        func_out = next(iter(after.functions.values()))

        # tile_b and tile_c are in different branches — they CAN share MemRef
        _assert_shares_memref(func_out, "tile_b", "tile_c")

    def test_loop_local_var_can_be_reused(self):
        """Variables defined AND used entirely within a single loop iteration
        can still be reused with other loop-local variables."""
        span = _SPAN
        shape = [64, 64]
        alloc = _MemRefAlloc()

        input_tensor = ir.Var("input_tensor", ir.TensorType(shape, _FP32), span)
        out_mr = alloc.ddr(shape, _FP32)
        output = ir.Var("output", _tensor_t(shape, _FP32, out_mr), span)
        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)

        # Loop body: tile_x = load; tile_y = add(tile_x, tile_x); tile_z = add(tile_y, tile_y); yield tile_z
        x_mr, y_mr, z_mr = alloc.vec(shape, _FP32), alloc.vec(shape, _FP32), alloc.vec(shape, _FP32)
        x_tt, y_tt, z_tt = (
            _tile_t(shape, _FP32, x_mr),
            _tile_t(shape, _FP32, y_mr),
            _tile_t(shape, _FP32, z_mr),
        )
        tile_x = ir.Var("tile_x", x_tt, span)
        tile_y = ir.Var("tile_y", y_tt, span)
        tile_z = ir.Var("tile_z", z_tt, span)

        # Use a zero init for iter_arg
        init_mr = alloc.vec(shape, _FP32)
        init_tt = _tile_t(shape, _FP32, init_mr)
        init_tile = ir.Var("init_tile", init_tt, span)
        create_call = ir.Call(ir.get_op("tile.create"), [ir.ConstFloat(0.0, _FP32, span)], init_tt, span)
        init_stmt = ir.AssignStmt(init_tile, create_call, span)

        ia = ir.IterArg("acc", init_tt, init_tile, span)

        load_call = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            x_tt,
            span,
        )
        add_call_1 = ir.Call(ir.get_op("tile.add"), [tile_x, tile_x], y_tt, span)
        add_call_2 = ir.Call(ir.get_op("tile.add"), [tile_y, tile_y], z_tt, span)

        body_stmts: list[ir.AssignStmt | ir.EvalStmt] = [
            ir.AssignStmt(tile_x, load_call, span),
            ir.AssignStmt(tile_y, add_call_1, span),
            ir.AssignStmt(tile_z, add_call_2, span),
        ]
        seq_items: list[ir.Stmt] = [*body_stmts, ir.YieldStmt([tile_z], span)]
        loop_body = ir.SeqStmts(seq_items, span)

        rv_mr = alloc.vec(shape, _FP32)
        rv = ir.Var("loop_out", _tile_t(shape, _FP32, rv_mr), span)
        for_stmt = ir.ForStmt(loop_var, _ci(0), _ci(4), _ci(1), [ia], loop_body, [rv], span)

        store_call = ir.Call(
            ir.get_op("tile.store"),
            [rv, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        result_var = ir.Var("result", _tensor_t(shape, _FP32, out_mr), span)

        body = ir.SeqStmts(
            [
                init_stmt,
                for_stmt,
                ir.AssignStmt(result_var, store_call, span),
                ir.ReturnStmt([result_var], span),
            ],
            span,
        )
        func = ir.Function(
            "main",
            [(input_tensor, ir.ParamDirection.In), (output, ir.ParamDirection.Out)],
            [ir.TensorType(shape, _FP32)],
            body,
            span,
        )
        prog = ir.Program([func], "Test", span)

        after = passes.memory_reuse()(prog)
        func_out = next(iter(after.functions.values()))

        # tile_x and tile_z should share MemRef (both loop-local, non-overlapping)
        _assert_shares_memref(func_out, "tile_x", "tile_z")

    def test_nested_for_loops_outer_var_extends_to_outer_end(self):
        """Variable defined before nested loops, used in inner loop body —
        lifetime must extend to the END of the OUTER loop (not just inner)."""
        span = _SPAN
        shape = [64, 64]
        alloc = _MemRefAlloc()

        input_tensor = ir.Var("input_tensor", ir.TensorType(shape, _FP32), span)
        out_mr = alloc.ddr(shape, _FP32)
        output = ir.Var("output", _tensor_t(shape, _FP32, out_mr), span)

        # tile_a: defined before both loops
        a_mr = alloc.vec(shape, _FP32)
        a_tt = _tile_t(shape, _FP32, a_mr)
        tile_a = ir.Var("tile_a", a_tt, span)
        load_a = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            a_tt,
            span,
        )

        # Inner loop: for j in range(4): tile_b = add(acc_inner, tile_a); yield tile_b
        j_var = ir.Var("j", ir.ScalarType(DataType.INDEX), span)
        init_inner_mr = alloc.vec(shape, _FP32)
        init_inner_tt = _tile_t(shape, _FP32, init_inner_mr)
        init_inner = ir.Var("init_inner", init_inner_tt, span)
        create_inner = ir.Call(
            ir.get_op("tile.create"), [ir.ConstFloat(0.0, _FP32, span)], init_inner_tt, span
        )

        ia_inner = ir.IterArg("acc_inner", init_inner_tt, init_inner, span)
        b_mr = alloc.vec(shape, _FP32)
        b_tt = _tile_t(shape, _FP32, b_mr)
        tile_b = ir.Var("tile_b", b_tt, span)
        add_b = ir.Call(ir.get_op("tile.add"), [ia_inner, tile_a], b_tt, span)  # tile_a used in inner loop!

        inner_body = ir.SeqStmts([ir.AssignStmt(tile_b, add_b, span), ir.YieldStmt([tile_b], span)], span)
        inner_rv_mr = alloc.vec(shape, _FP32)
        inner_rv = ir.Var("inner_out", _tile_t(shape, _FP32, inner_rv_mr), span)
        inner_for = ir.ForStmt(j_var, _ci(0), _ci(4), _ci(1), [ia_inner], inner_body, [inner_rv], span)

        # Outer loop: for i in range(4): { inner_for; tile_d = add(acc_outer, inner_out); yield tile_d }
        i_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        init_outer_mr = alloc.vec(shape, _FP32)
        init_outer_tt = _tile_t(shape, _FP32, init_outer_mr)
        init_outer = ir.Var("init_outer", init_outer_tt, span)
        create_outer = ir.Call(
            ir.get_op("tile.create"), [ir.ConstFloat(0.0, _FP32, span)], init_outer_tt, span
        )

        ia_outer = ir.IterArg("acc_outer", init_outer_tt, init_outer, span)
        d_mr = alloc.vec(shape, _FP32)
        d_tt = _tile_t(shape, _FP32, d_mr)
        tile_d = ir.Var("tile_d", d_tt, span)
        add_d = ir.Call(ir.get_op("tile.add"), [ia_outer, inner_rv], d_tt, span)

        outer_body = ir.SeqStmts(
            [inner_for, ir.AssignStmt(tile_d, add_d, span), ir.YieldStmt([tile_d], span)],
            span,
        )
        outer_rv_mr = alloc.vec(shape, _FP32)
        outer_rv = ir.Var("outer_out", _tile_t(shape, _FP32, outer_rv_mr), span)
        outer_for = ir.ForStmt(i_var, _ci(0), _ci(4), _ci(1), [ia_outer], outer_body, [outer_rv], span)

        # Store and return
        store_call = ir.Call(
            ir.get_op("tile.store"),
            [outer_rv, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        result_var = ir.Var("result", _tensor_t(shape, _FP32, out_mr), span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(tile_a, load_a, span),
                ir.AssignStmt(init_inner, create_inner, span),
                ir.AssignStmt(init_outer, create_outer, span),
                outer_for,
                ir.AssignStmt(result_var, store_call, span),
                ir.ReturnStmt([result_var], span),
            ],
            span,
        )
        func = ir.Function(
            "main",
            [(input_tensor, ir.ParamDirection.In), (output, ir.ParamDirection.Out)],
            [ir.TensorType(shape, _FP32)],
            body,
            span,
        )
        prog = ir.Program([func], "Test", span)

        after = passes.memory_reuse()(prog)
        func_out = next(iter(after.functions.values()))

        # tile_a used in inner loop but defined outside outer loop → must NOT be reused
        # by tile_b (inner loop body) or tile_d (outer loop body)
        _assert_not_shares_memref(func_out, "tile_a", "tile_b")
        _assert_not_shares_memref(func_out, "tile_a", "tile_d")

    def test_if_without_else_branch(self):
        """IfStmt with only then branch (no else) should not crash and
        correctly track variable uses inside then body."""
        span = _SPAN
        shape = [64, 64]
        alloc = _MemRefAlloc()

        input_tensor = ir.Var("input_tensor", ir.TensorType(shape, _FP32), span)
        cond_param = ir.Var("cond_param", ir.ScalarType(DataType.INDEX), span)
        out_mr = alloc.ddr(shape, _FP32)
        output = ir.Var("output", _tensor_t(shape, _FP32, out_mr), span)

        # tile_a defined before if
        a_mr = alloc.vec(shape, _FP32)
        a_tt = _tile_t(shape, _FP32, a_mr)
        tile_a = ir.Var("tile_a", a_tt, span)
        load_a = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            a_tt,
            span,
        )

        # Then branch: tile_b = add(tile_a, tile_a); store(tile_b, ...)
        b_mr = alloc.vec(shape, _FP32)
        b_tt = _tile_t(shape, _FP32, b_mr)
        tile_b = ir.Var("tile_b", b_tt, span)
        add_b = ir.Call(ir.get_op("tile.add"), [tile_a, tile_a], b_tt, span)
        store_b = ir.Call(
            ir.get_op("tile.store"),
            [tile_b, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        store_b_var = ir.Var("store_b", _tensor_t(shape, _FP32, out_mr), span)
        then_body = ir.SeqStmts(
            [ir.AssignStmt(tile_b, add_b, span), ir.AssignStmt(store_b_var, store_b, span)], span
        )

        cond = ir.Lt(cond_param, _ci(2), _IDX, span)
        if_stmt = ir.IfStmt(cond, then_body, None, [], span)  # No else, no return_vars

        # tile_c defined after if — tile_a should still be alive through IfStmt
        c_mr = alloc.vec(shape, _FP32)
        c_tt = _tile_t(shape, _FP32, c_mr)
        tile_c = ir.Var("tile_c", c_tt, span)
        add_c = ir.Call(ir.get_op("tile.add"), [tile_a, tile_a], c_tt, span)

        store_c = ir.Call(
            ir.get_op("tile.store"),
            [tile_c, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        result_var = ir.Var("result", _tensor_t(shape, _FP32, out_mr), span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(tile_a, load_a, span),
                if_stmt,
                ir.AssignStmt(tile_c, add_c, span),
                ir.AssignStmt(result_var, store_c, span),
                ir.ReturnStmt([result_var], span),
            ],
            span,
        )
        func = ir.Function(
            "main",
            [
                (input_tensor, ir.ParamDirection.In),
                (cond_param, ir.ParamDirection.In),
                (output, ir.ParamDirection.Out),
            ],
            [ir.TensorType(shape, _FP32)],
            body,
            span,
        )
        prog = ir.Program([func], "Test", span)

        after = passes.memory_reuse()(prog)
        func_out = next(iter(after.functions.values()))

        # tile_a is used both inside IfStmt (then branch) and after it → still alive
        # tile_b (inside then) overlaps with tile_a → cannot reuse
        _assert_not_shares_memref(func_out, "tile_a", "tile_b")
        # tile_c is after tile_a's last use → can reuse tile_a (greedy first-fit)
        _assert_shares_memref(func_out, "tile_a", "tile_c")

    def test_for_with_if_multiple_vars_competing(self):
        """ForStmt with IfStmt inside, multiple variables from before the loop
        used inside the if — tests that ALL outer variables are correctly extended."""
        span = _SPAN
        shape = [64, 64]
        alloc = _MemRefAlloc()

        input_tensor = ir.Var("input_tensor", ir.TensorType(shape, _FP32), span)
        out_mr = alloc.ddr(shape, _FP32)
        output = ir.Var("output", _tensor_t(shape, _FP32, out_mr), span)

        # Two tiles defined before the loop
        a_mr, b_mr = alloc.vec(shape, _FP32), alloc.vec(shape, _FP32)
        a_tt, b_tt = _tile_t(shape, _FP32, a_mr), _tile_t(shape, _FP32, b_mr)
        tile_a = ir.Var("tile_a", a_tt, span)
        tile_b = ir.Var("tile_b", b_tt, span)
        load_a = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            a_tt,
            span,
        )
        load_b = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            b_tt,
            span,
        )

        # Loop: for i in range(4)
        i_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        init_mr = alloc.vec(shape, _FP32)
        init_tt = _tile_t(shape, _FP32, init_mr)
        init_tile = ir.Var("init_tile", init_tt, span)
        create_init = ir.Call(ir.get_op("tile.create"), [ir.ConstFloat(0.0, _FP32, span)], init_tt, span)
        ia = ir.IterArg("acc", init_tt, init_tile, span)

        # Inside loop: if (i < 2): tile_c = add(tile_a, tile_b) else: tile_d = add(tile_b, tile_a)
        c_mr, d_mr = alloc.vec(shape, _FP32), alloc.vec(shape, _FP32)
        c_tt, d_tt = _tile_t(shape, _FP32, c_mr), _tile_t(shape, _FP32, d_mr)
        tile_c = ir.Var("tile_c", c_tt, span)
        tile_d = ir.Var("tile_d", d_tt, span)
        add_c = ir.Call(ir.get_op("tile.add"), [tile_a, tile_b], c_tt, span)
        add_d = ir.Call(ir.get_op("tile.add"), [tile_b, tile_a], d_tt, span)

        then_body = ir.SeqStmts([ir.AssignStmt(tile_c, add_c, span), ir.YieldStmt([tile_c], span)], span)
        else_body = ir.SeqStmts([ir.AssignStmt(tile_d, add_d, span), ir.YieldStmt([tile_d], span)], span)
        if_rv_mr = alloc.vec(shape, _FP32)
        if_rv = ir.Var("if_result", _tile_t(shape, _FP32, if_rv_mr), span)
        cond = ir.Lt(i_var, _ci(2), _IDX, span)
        if_stmt = ir.IfStmt(cond, then_body, else_body, [if_rv], span)

        # Yield if_result from loop
        loop_body = ir.SeqStmts([if_stmt, ir.YieldStmt([if_rv], span)], span)
        rv_mr = alloc.vec(shape, _FP32)
        rv = ir.Var("loop_out", _tile_t(shape, _FP32, rv_mr), span)
        for_stmt = ir.ForStmt(i_var, _ci(0), _ci(4), _ci(1), [ia], loop_body, [rv], span)

        store_call = ir.Call(
            ir.get_op("tile.store"),
            [rv, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        result_var = ir.Var("result", _tensor_t(shape, _FP32, out_mr), span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(tile_a, load_a, span),
                ir.AssignStmt(tile_b, load_b, span),
                ir.AssignStmt(init_tile, create_init, span),
                for_stmt,
                ir.AssignStmt(result_var, store_call, span),
                ir.ReturnStmt([result_var], span),
            ],
            span,
        )
        func = ir.Function(
            "main",
            [(input_tensor, ir.ParamDirection.In), (output, ir.ParamDirection.Out)],
            [ir.TensorType(shape, _FP32)],
            body,
            span,
        )
        prog = ir.Program([func], "Test", span)

        after = passes.memory_reuse()(prog)
        func_out = next(iter(after.functions.values()))

        # tile_a and tile_b are both used inside the nested IfStmt in the loop —
        # their lifetimes extend to loop end, so tile_c and tile_d cannot reuse them
        _assert_not_shares_memref(func_out, "tile_a", "tile_c")
        _assert_not_shares_memref(func_out, "tile_a", "tile_d")
        _assert_not_shares_memref(func_out, "tile_b", "tile_c")
        _assert_not_shares_memref(func_out, "tile_b", "tile_d")
        # tile_c and tile_d are in different branches — they CAN share
        _assert_shares_memref(func_out, "tile_c", "tile_d")

    def test_branch_local_var_does_not_leak(self):
        """A variable defined and consumed entirely inside one IfStmt branch
        should have a short lifetime and not block reuse after the IfStmt."""
        span = _SPAN
        shape = [64, 64]
        alloc = _MemRefAlloc()

        input_tensor = ir.Var("input_tensor", ir.TensorType(shape, _FP32), span)
        cond_param = ir.Var("cond_param", ir.ScalarType(DataType.INDEX), span)
        out_mr = alloc.ddr(shape, _FP32)
        output = ir.Var("output", _tensor_t(shape, _FP32, out_mr), span)

        # tile_a: loaded before if
        a_mr = alloc.vec(shape, _FP32)
        a_tt = _tile_t(shape, _FP32, a_mr)
        tile_a = ir.Var("tile_a", a_tt, span)
        load_a = ir.Call(
            ir.get_op("tile.load"),
            [input_tensor, ir.MakeTuple([_ci(0), _ci(0)], span), ir.MakeTuple([_ci(64), _ci(64)], span)],
            a_tt,
            span,
        )

        # Then branch: tile_b = add(tile_a, tile_a); yield tile_b
        b_mr = alloc.vec(shape, _FP32)
        b_tt = _tile_t(shape, _FP32, b_mr)
        tile_b = ir.Var("tile_b", b_tt, span)
        add_b = ir.Call(ir.get_op("tile.add"), [tile_a, tile_a], b_tt, span)
        then_body = ir.SeqStmts([ir.AssignStmt(tile_b, add_b, span), ir.YieldStmt([tile_b], span)], span)

        # Else branch: yield tile_a (tile_a is the fallback)
        else_body = ir.YieldStmt([tile_a], span)

        if_rv_mr = alloc.vec(shape, _FP32)
        if_rv = ir.Var("if_result", _tile_t(shape, _FP32, if_rv_mr), span)
        cond = ir.Lt(cond_param, _ci(2), _IDX, span)
        if_stmt = ir.IfStmt(cond, then_body, else_body, [if_rv], span)

        # tile_e = add(if_result, if_result) — defined AFTER IfStmt
        e_mr = alloc.vec(shape, _FP32)
        e_tt = _tile_t(shape, _FP32, e_mr)
        tile_e = ir.Var("tile_e", e_tt, span)
        add_e = ir.Call(ir.get_op("tile.add"), [if_rv, if_rv], e_tt, span)

        store_call = ir.Call(
            ir.get_op("tile.store"),
            [tile_e, ir.MakeTuple([_ci(0), _ci(0)], span), output],
            _tensor_t(shape, _FP32, out_mr),
            span,
        )
        result_var = ir.Var("result", _tensor_t(shape, _FP32, out_mr), span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(tile_a, load_a, span),
                if_stmt,
                ir.AssignStmt(tile_e, add_e, span),
                ir.AssignStmt(result_var, store_call, span),
                ir.ReturnStmt([result_var], span),
            ],
            span,
        )
        func = ir.Function(
            "main",
            [
                (input_tensor, ir.ParamDirection.In),
                (cond_param, ir.ParamDirection.In),
                (output, ir.ParamDirection.Out),
            ],
            [ir.TensorType(shape, _FP32)],
            body,
            span,
        )
        prog = ir.Program([func], "Test", span)

        after = passes.memory_reuse()(prog)
        func_out = next(iter(after.functions.values()))

        # tile_b is local to then-branch (last use at then-yield).
        # tile_e is defined after IfStmt. tile_b's lifetime should NOT leak
        # beyond the then-branch, so tile_e CAN reuse tile_a (tile_a's last use
        # is in the else-yield which ends before tile_e's def)
        _assert_shares_memref(func_out, "tile_a", "tile_e")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
