# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for BasicMemoryReusePass with pre-attached MemRefs (no init_mem_ref dependency)."""

import math

import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile
from pypto.pypto_core import DataType

_SPAN = ir.Span.unknown()


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure backend before each test (required by DependencyAnalyzer)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)
    yield
    backend.reset_for_testing()


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
    """Run basic_memory_reuse pass and return the first function."""
    after = passes.basic_memory_reuse()(program)
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


class TestBasicMemoryReuse:
    """Tests for BasicMemoryReusePass with TileType variables."""

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

    def test_memref_sharing(self):
        """Chain: all tiles reuse tile_a (producer-consumer at same statement)."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, b_mr, c_mr, d_mr = (alloc.vec([64, 64], _FP32) for _ in range(4))
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            tile_b = ib.let("tile_b", tile.add(tile_a, tile_a), type=_tile_t([64, 64], _FP32, b_mr))
            tile_c = ib.let("tile_c", tile.add(tile_b, tile_b), type=_tile_t([64, 64], _FP32, c_mr))
            tile_d = ib.let("tile_d", tile.add(tile_c, tile_c), type=_tile_t([64, 64], _FP32, d_mr))
            result = ib.let(
                "result", tile.store(tile_d, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")

    def test_with_dependencies(self):
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
            tile_d = ib.let("tile_d", tile.add(tile_c, tile_c), type=_tile_t([64, 64], _FP32, d_mr))
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

    def test_multiple_memory_spaces(self):
        """Memory reuse happens within the same memory space (UB tiles)."""

        def build(ib, f, alloc):
            in_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            in_b = f.param("input_b", ir.TensorType([64, 64], _FP32))
            out_a_mr, out_b_mr = alloc.ddr([64, 64], _FP32), alloc.ddr([64, 64], _FP32)
            out_a = f.param("output_a", _tensor_t([64, 64], _FP32, out_a_mr), direction=ir.ParamDirection.Out)
            out_b = f.param("output_b", _tensor_t([64, 64], _FP32, out_b_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, b_mr, c_mr, d_mr = (alloc.vec([64, 64], _FP32) for _ in range(4))
            tile_a = ib.let("tile_a", tile.load(in_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr))
            tile_b = ib.let("tile_b", tile.load(in_b, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, b_mr))
            tile_c = ib.let("tile_c", tile.add(tile_a, tile_b), type=_tile_t([64, 64], _FP32, c_mr))
            ib.let("_result_a", tile.store(tile_c, [0, 0], out_a), type=_tensor_t([64, 64], _FP32, out_a_mr))
            tile_d = ib.let("tile_d", tile.add(tile_c, tile_c), type=_tile_t([64, 64], _FP32, d_mr))
            result_b = ib.let(
                "result_b", tile.store(tile_d, [0, 0], out_b), type=_tensor_t([64, 64], _FP32, out_b_mr)
            )
            ib.return_stmt(result_b)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_d")


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

    body = ir.SeqStmts([ir.OpStmts(stmts, span), ir.ReturnStmt([var_map[op_specs[-1][0]]], span)], span)
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

        after = passes.basic_memory_reuse()(prog)
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

        after = passes.basic_memory_reuse()(prog)
        func = next(iter(after.functions.values()))

        assert _count_alloc_stmts(func) == 2, (
            f"Expected 2 alloc stmts (tile_c reuses tile_a), got {_count_alloc_stmts(func)}"
        )


class TestDtypeCompatibility:
    """Tests that tiles with different dtypes do NOT reuse each other's memory."""

    def test_cast_output_does_not_reuse(self):
        """Cast changes dtype: no cross-dtype reuse; same-dtype tiles still reuse."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr, b_mr = alloc.vec([64, 64], _FP32), alloc.vec([64, 64], _FP32)
            cast_mr, c_mr = alloc.vec([64, 64], _BF16), alloc.vec([64, 64], _BF16)
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            tile_b = ib.let("tile_b", tile.add(tile_a, tile_a), type=_tile_t([64, 64], _FP32, b_mr))
            tile_cast = ib.let("tile_cast", tile.cast(tile_b, _BF16), type=_tile_t([64, 64], _BF16, cast_mr))
            tile_c = ib.let("tile_c", tile.add(tile_cast, tile_cast), type=_tile_t([64, 64], _BF16, c_mr))
            result = ib.let(
                "result", tile.store(tile_c, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_cast")
        _assert_not_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_cast", "tile_c")

    def test_cast_among_regular_ops(self):
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


class TestFillpadCompatibility:
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


class TestViewOperationsMemoryReuse:
    """Tests for view operations (reshape) with memory reuse."""

    def test_reshape_shares_memref_with_input(self):
        """Single reshape operation should share MemRef with input tile."""

        def build(ib, f, alloc):
            input_a = f.param("input_a", ir.TensorType([64, 64], _FP32))
            out_mr = alloc.ddr([64, 64], _FP32)
            output = f.param("output", _tensor_t([64, 64], _FP32, out_mr), direction=ir.ParamDirection.Out)
            f.return_type(ir.TensorType([64, 64], _FP32))
            a_mr = alloc.vec([64, 64], _FP32)
            c_mr = alloc.vec([4096, 1], _FP32)
            tile_a = ib.let(
                "tile_a", tile.load(input_a, [0, 0], [64, 64]), type=_tile_t([64, 64], _FP32, a_mr)
            )
            # reshape shares MemRef with input
            tile_b = ib.let("tile_b", tile.reshape(tile_a, [4096, 1]), type=_tile_t([4096, 1], _FP32, a_mr))
            tile_c = ib.let("tile_c", tile.add(tile_b, tile_b), type=_tile_t([4096, 1], _FP32, c_mr))
            # reshape shares MemRef with input
            tile_d = ib.let("tile_d", tile.reshape(tile_c, [64, 64]), type=_tile_t([64, 64], _FP32, c_mr))
            result = ib.let(
                "result", tile.store(tile_d, [0, 0], output), type=_tensor_t([64, 64], _FP32, out_mr)
            )
            ib.return_stmt(result)

        func = _run_reuse(_build_program(build))
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "tile_c", "tile_d")

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
        """BasicMemoryReuse should propagate reuse to ALL variables sharing MemRef."""

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
            # BasicMemoryReuse: tile_a reuses tile_c → _tile_b also gets tile_c's MemRef
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


class TestInplaceSafetyCheck:
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

    def test_ors_no_producer_consumer_reuse(self):
        """tile.ors must NOT reuse its input's buffer."""
        func = self._build_simple_op_test(lambda t: tile.ors(t, 255), [32, 32], _INT32)
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
    if isinstance(stmt, (ir.SeqStmts, ir.OpStmts)):
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

    loop_body = ir.SeqStmts([ir.OpStmts(body_stmts, _SPAN), ir.YieldStmt(yield_values, _SPAN)], _SPAN)

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

    body = ir.SeqStmts(
        [
            ir.OpStmts(init_stmts, _SPAN),
            loop_stmt,
            ir.OpStmts([store_stmt], _SPAN),
            ir.ReturnStmt([result_var], _SPAN),
        ],
        _SPAN,
    )
    func = ir.Function(
        "main",
        [(input_tensor, ir.ParamDirection.In), (out_tensor, ir.ParamDirection.Out)],
        [_tensor_t(shape, dtype)],
        body,
        _SPAN,
    )
    return ir.Program([func], "TestProgram", _SPAN)


class TestForStmtYieldFixup:
    """Tests for ForStmt yield fixup — ensuring all 4 loop-carry variables share one MemRef."""

    def test_tile_move_inserted_when_memrefs_diverge(self):
        """When initValue and yield value start with different MemRefs,
        the pass should unify all loop-carry vars to share one MemRef."""
        alloc = _MemRefAlloc()
        init_mr = alloc.vec([64, 64], _FP32)
        yield_mr = alloc.vec([64, 64], _FP32)
        assert init_mr.id_ != yield_mr.id_, "precondition: MemRefs start different"
        # add_overlap=True adds extra usage to prevent trivial producer-consumer reuse
        prog = _build_for_loop_program([init_mr], [yield_mr], add_overlap=True)

        after = passes.basic_memory_reuse()(prog)
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

        after = passes.basic_memory_reuse()(prog)
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

        after = passes.basic_memory_reuse()(prog)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
