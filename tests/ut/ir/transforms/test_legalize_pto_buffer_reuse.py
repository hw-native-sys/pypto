# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for LegalizePTOBufferReuse pass.

Verifies that the pass correctly splits MemRefs when multiple tile
variables sharing the same MemRef have incompatible root TileBufSignatures
(different shape/dtype/layout), while preserving legal sharing for
view-like operations (fillpad, reshape).
"""

import math

import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.builder import IRBuilder
from pypto.pypto_core import DataType

_SPAN = ir.Span.unknown()
_IDX = DataType.INDEX
_FP32 = DataType.FP32
_FP16 = DataType.FP16


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure backend before each test."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)
    yield
    backend.reset_for_testing()


def _ci(val: int) -> ir.ConstInt:
    return ir.ConstInt(val, _IDX, _SPAN)


def _dtype_bytes(dtype: DataType) -> int:
    if dtype in (_FP32,):
        return 4
    if dtype in (_FP16,):
        return 2
    raise ValueError(f"Unsupported dtype: {dtype}")


class _MemRefAlloc:
    def __init__(self, start_id: int = 0) -> None:
        self._next_id = start_id

    def vec(self, shape: list[int], dtype: DataType) -> ir.MemRef:
        size = math.prod(shape) * _dtype_bytes(dtype)
        mr = ir.MemRef(ir.MemorySpace.Vec, _ci(-1), size, self._next_id)
        self._next_id += 1
        return mr


def _tile_t(
    shape: list[int], dtype: DataType, memref: ir.MemRef, space: ir.MemorySpace = ir.MemorySpace.Vec
) -> ir.TileType:
    return ir.TileType(shape, dtype, memref, None, space)


def _tile_t_with_view(
    shape: list[int],
    dtype: DataType,
    memref: ir.MemRef,
    tile_view: ir.TileView,
    space: ir.MemorySpace = ir.MemorySpace.Vec,
) -> ir.TileType:
    return ir.TileType(shape, dtype, memref, tile_view, space)


def _tensor_t(shape: list[int], dtype: DataType) -> ir.TensorType:
    return ir.TensorType(shape, dtype)


def _build_program(build_fn):
    alloc = _MemRefAlloc()
    ib = IRBuilder()
    with ib.program("Test") as prog:
        with ib.function("main") as f:
            build_fn(ib, f, alloc)
        prog.add_function(f.get_result())
    return prog.get_result()


def _run_legalize(program: ir.Program) -> ir.Function:
    after = passes.legalize_pto_buffer_reuse()(program)
    return next(iter(after.functions.values()))


def _get_mlir_code(result: str | dict[str, str]) -> str:
    """Normalize generate() output to a single MLIR string."""
    return result if isinstance(result, str) else "".join(result.values())


def _get_alloc_tile_lines(mlir_code: str) -> list[str]:
    """Return normalized alloc_tile lines from generated MLIR."""
    return [line.strip() for line in mlir_code.splitlines() if "pto.alloc_tile" in line]


def _generate_legalized_mlir(program: ir.Program) -> str:
    """Run legalization then generate MLIR."""
    legalized = passes.legalize_pto_buffer_reuse()(program)
    return _get_mlir_code(codegen.PTOCodegen().generate(legalized))


def _get_alloc_addrs(alloc_lines: list[str]) -> list[str]:
    """Extract alloc_tile addr values after asserting that each line carries one."""
    for line in alloc_lines:
        assert "addr =" in line, f"Expected addr attribute in alloc_tile: {line}"
    return [line.split("addr = ")[1].split()[0] for line in alloc_lines]


def _iter_all_assign_stmts(stmt):
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


def _get_var_type(func, var_name):
    for stmt in _iter_all_assign_stmts(func.body):
        if stmt.var.name_hint == var_name:
            if isinstance(stmt.var.type, ir.ShapedType):
                return stmt.var.type
    return None


def _assert_shares_memref(func, var_a, var_b):
    ta = _get_var_type(func, var_a)
    tb = _get_var_type(func, var_b)
    assert ta is not None, f"Variable '{var_a}' not found"
    assert tb is not None, f"Variable '{var_b}' not found"
    assert ta.memref is tb.memref, f"'{var_a}' and '{var_b}' should share the same MemRef"


def _assert_different_memref(func, var_a, var_b):
    ta = _get_var_type(func, var_a)
    tb = _get_var_type(func, var_b)
    assert ta is not None, f"Variable '{var_a}' not found"
    assert tb is not None, f"Variable '{var_b}' not found"
    assert ta.memref is not tb.memref, f"'{var_a}' and '{var_b}' should have different MemRefs"


# ---------------------------------------------------------------------------
# Tests: identical signatures keep shared MemRef
# ---------------------------------------------------------------------------


class TestLegalSharingPreserved:
    """Same-shape same-dtype tiles sharing a MemRef should keep sharing."""

    def test_same_signature_keeps_shared(self):
        shared = _MemRefAlloc().vec([32, 32], _FP32)

        input_t = _tensor_t([32, 32], _FP32)
        output_t = _tensor_t([32, 32], _FP32)
        tile1_type = _tile_t([32, 32], _FP32, shared)
        tile2_type = _tile_t([32, 32], _FP32, shared)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", tile1_type, _SPAN)
        t2 = ir.Var("t2", tile2_type, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes = ir.MakeTuple([_ci(32), _ci(32)], _SPAN)

        load_call = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes], {}, tile1_type, _SPAN)
        adds_call = ir.Call(ir.Op("tile.adds"), [t1, ir.ConstFloat(1.0, _FP32, _SPAN)], {}, tile2_type, _SPAN)
        result_var = ir.Var("result", output_t, _SPAN)
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load_call, _SPAN),
                ir.AssignStmt(t2, adds_call, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [(a_var, ir.ParamDirection.In), (b_var, ir.ParamDirection.Out)],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        result_func = _run_legalize(program)
        _assert_shares_memref(result_func, "t1", "t2")

    def test_fillpad_view_keeps_shared(self):
        """fillpad changes pad but keeps same shape → legal view."""
        shared = _MemRefAlloc().vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([128, 128], _FP32)
        load_type = _tile_t([128, 128], _FP32, shared)

        padded_view = ir.TileView()
        padded_view.valid_shape = [_ci(128), _ci(128)]
        padded_view.pad = ir.PadValue.max
        padded_type = _tile_t_with_view([128, 128], _FP32, shared, padded_view)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("loaded", load_type, _SPAN)
        t2 = ir.Var("padded", padded_type, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)

        load_call = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes], {}, load_type, _SPAN)
        fillpad_call = ir.Call(
            ir.Op("tile.fillpad"),
            [t1],
            {"pad_value": ir.PadValue.max},
            padded_type,
            _SPAN,
        )
        result_var = ir.Var("result", output_t, _SPAN)
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load_call, _SPAN),
                ir.AssignStmt(t2, fillpad_call, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [(a_var, ir.ParamDirection.In), (b_var, ir.ParamDirection.Out)],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        result_func = _run_legalize(program)
        _assert_shares_memref(result_func, "loaded", "padded")


# ---------------------------------------------------------------------------
# Tests: incompatible signatures cause split
# ---------------------------------------------------------------------------


class TestIllegalSharingSplit:
    """Tiles with incompatible root signatures should be split."""

    def test_different_shape_same_memref_splits(self):
        """Two writers with different shapes → must split."""
        alloc = _MemRefAlloc()
        shared = alloc.vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([128, 128], _FP32)

        view_128 = ir.TileView()
        view_128.valid_shape = [_ci(128), _ci(128)]
        view_64 = ir.TileView()
        view_64.valid_shape = [_ci(64), _ci(64)]

        tile1_type = _tile_t_with_view([128, 128], _FP32, shared, view_128)
        tile2_type = _tile_t_with_view([64, 64], _FP32, shared, view_64)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", tile1_type, _SPAN)
        t2 = ir.Var("t2", tile2_type, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes_128 = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)
        shapes_64 = ir.MakeTuple([_ci(64), _ci(64)], _SPAN)

        load1 = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes_128], {}, tile1_type, _SPAN)
        load2 = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes_64], {}, tile2_type, _SPAN)
        result_var = ir.Var("result", output_t, _SPAN)
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load1, _SPAN),
                ir.AssignStmt(t2, load2, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [(a_var, ir.ParamDirection.In), (b_var, ir.ParamDirection.Out)],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        result_func = _run_legalize(program)
        _assert_different_memref(result_func, "t1", "t2")

    def test_split_propagates_through_view_chain(self):
        """A split writer's legal views should follow the new MemRef."""
        alloc = _MemRefAlloc()
        shared = alloc.vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([64, 64], _FP32)

        view_128 = ir.TileView()
        view_128.valid_shape = [_ci(128), _ci(128)]
        view_64 = ir.TileView()
        view_64.valid_shape = [_ci(64), _ci(64)]
        padded_view = ir.TileView()
        padded_view.valid_shape = [_ci(64), _ci(64)]
        padded_view.pad = ir.PadValue.max

        tile1_type = _tile_t_with_view([128, 128], _FP32, shared, view_128)
        tile2_type = _tile_t_with_view([64, 64], _FP32, shared, view_64)
        tile3_type = _tile_t_with_view([64, 64], _FP32, shared, padded_view)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", tile1_type, _SPAN)
        t2 = ir.Var("t2", tile2_type, _SPAN)
        t3 = ir.Var("t3", tile3_type, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes_128 = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)
        shapes_64 = ir.MakeTuple([_ci(64), _ci(64)], _SPAN)

        load1 = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes_128], {}, tile1_type, _SPAN)
        load2 = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes_64], {}, tile2_type, _SPAN)
        fillpad = ir.Call(
            ir.Op("tile.fillpad"),
            [t2],
            {"pad_value": ir.PadValue.max},
            tile3_type,
            _SPAN,
        )
        result_var = ir.Var("result", output_t, _SPAN)
        store_call = ir.Call(ir.Op("tile.store"), [t3, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load1, _SPAN),
                ir.AssignStmt(t2, load2, _SPAN),
                ir.AssignStmt(t3, fillpad, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [(a_var, ir.ParamDirection.In), (b_var, ir.ParamDirection.Out)],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        result_func = _run_legalize(program)
        _assert_different_memref(result_func, "t1", "t2")
        _assert_shares_memref(result_func, "t2", "t3")
        _assert_different_memref(result_func, "t1", "t3")


# ---------------------------------------------------------------------------
# Integration test: legalize + codegen
# ---------------------------------------------------------------------------


class TestLegalizeWithCodegen:
    """End-to-end: legalize pass + codegen produces valid MLIR."""

    def test_fillpad_shared_still_single_alloc(self):
        """After legalization, fillpad sharing still produces one alloc_tile."""
        shared = _MemRefAlloc().vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([128, 128], _FP32)
        load_type = _tile_t([128, 128], _FP32, shared)

        padded_view = ir.TileView()
        padded_view.valid_shape = [_ci(128), _ci(128)]
        padded_view.pad = ir.PadValue.max
        padded_type = _tile_t_with_view([128, 128], _FP32, shared, padded_view)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", load_type, _SPAN)
        t2 = ir.Var("t2", padded_type, _SPAN)
        result_var = ir.Var("result", output_t, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)

        load_call = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes], {}, load_type, _SPAN)
        fillpad_call = ir.Call(
            ir.Op("tile.fillpad"),
            [t1],
            {"pad_value": ir.PadValue.max},
            padded_type,
            _SPAN,
        )
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load_call, _SPAN),
                ir.AssignStmt(t2, fillpad_call, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [(a_var, ir.ParamDirection.In), (b_var, ir.ParamDirection.Out)],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        mlir_code = _generate_legalized_mlir(program)
        alloc_lines = _get_alloc_tile_lines(mlir_code)
        assert len(alloc_lines) == 2, (
            f"Expected two alloc_tiles for per-var alloc (same MemRef, same addr), got: {alloc_lines}"
        )
        assert "%c-1" not in mlir_code
        addr_values = _get_alloc_addrs(alloc_lines)
        assert addr_values[0] == addr_values[1], f"Expected same addr for shared MemRef, got: {addr_values}"

    def test_fillpad_dynamic_valid_row_keeps_shared_addr(self):
        """Dynamic valid_row and fillpad should keep one shared address after legalization."""
        shared = _MemRefAlloc().vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([128, 128], _FP32)
        valid_rows = ir.Var("m", ir.ScalarType(_IDX), _SPAN)

        load_view = ir.TileView()
        load_view.valid_shape = [valid_rows, _ci(128)]
        load_type = _tile_t_with_view([128, 128], _FP32, shared, load_view)

        padded_view = ir.TileView()
        padded_view.valid_shape = [_ci(128), _ci(128)]
        padded_view.pad = ir.PadValue.max
        padded_type = _tile_t_with_view([128, 128], _FP32, shared, padded_view)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", load_type, _SPAN)
        t2 = ir.Var("t2", padded_type, _SPAN)
        result_var = ir.Var("result", output_t, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)

        load_call = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes], {}, load_type, _SPAN)
        fillpad_call = ir.Call(
            ir.Op("tile.fillpad"),
            [t1],
            {"pad_value": ir.PadValue.max},
            padded_type,
            _SPAN,
        )
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load_call, _SPAN),
                ir.AssignStmt(t2, fillpad_call, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [
                (a_var, ir.ParamDirection.In),
                (b_var, ir.ParamDirection.Out),
                (valid_rows, ir.ParamDirection.In),
            ],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        mlir_code = _generate_legalized_mlir(program)
        alloc_lines = _get_alloc_tile_lines(mlir_code)

        assert len(alloc_lines) == 2, (
            f"Expected two alloc_tiles for per-var alloc (same MemRef, same addr), got: {alloc_lines}"
        )
        addr_values = _get_alloc_addrs(alloc_lines)
        assert addr_values[0] == addr_values[1], f"Expected same addr for shared MemRef, got: {addr_values}"

        dynamic_allocs = [line for line in alloc_lines if "valid_row = %" in line]
        assert len(dynamic_allocs) == 1, f"Expected one alloc_tile with dynamic valid_row, got: {alloc_lines}"
        assert "valid_col = %" not in dynamic_allocs[0], (
            f"Did not expect dynamic valid_col in alloc_tile: {dynamic_allocs[0]}"
        )
        assert "v_row=?" in dynamic_allocs[0], (
            f"Expected dynamic v_row in tile_buf type, got: {dynamic_allocs[0]}"
        )
        assert "v_col=128" in dynamic_allocs[0], (
            f"Expected static v_col=128 in tile_buf type, got: {dynamic_allocs[0]}"
        )

        padded_allocs = [line for line in alloc_lines if "pad=2>" in line]
        assert len(padded_allocs) == 1, f"Expected one padded alloc_tile after fillpad, got: {alloc_lines}"

    def test_incompatible_shape_split_produces_two_allocs(self):
        """After legalization, split MemRefs produce separate alloc_tiles."""
        alloc = _MemRefAlloc()
        shared = alloc.vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([128, 128], _FP32)

        view_128 = ir.TileView()
        view_128.valid_shape = [_ci(128), _ci(128)]
        view_64 = ir.TileView()
        view_64.valid_shape = [_ci(64), _ci(64)]

        tile1_type = _tile_t_with_view([128, 128], _FP32, shared, view_128)
        tile2_type = _tile_t_with_view([64, 64], _FP32, shared, view_64)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", tile1_type, _SPAN)
        t2 = ir.Var("t2", tile2_type, _SPAN)
        result_var = ir.Var("result", output_t, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes_128 = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)
        shapes_64 = ir.MakeTuple([_ci(64), _ci(64)], _SPAN)

        load1 = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes_128], {}, tile1_type, _SPAN)
        load2 = ir.Call(ir.Op("tile.load"), [a_var, offsets, shapes_64], {}, tile2_type, _SPAN)
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load1, _SPAN),
                ir.AssignStmt(t2, load2, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [(a_var, ir.ParamDirection.In), (b_var, ir.ParamDirection.Out)],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        mlir_code = _generate_legalized_mlir(program)
        alloc_lines = _get_alloc_tile_lines(mlir_code)
        assert len(alloc_lines) == 2, f"Expected two alloc_tiles after split, got: {alloc_lines}"

        _get_alloc_addrs(alloc_lines)

        sizes_128 = [line for line in alloc_lines if "rows=128" in line and "cols=128" in line]
        sizes_64 = [line for line in alloc_lines if "rows=64" in line and "cols=64" in line]
        assert len(sizes_128) == 1, f"Expected one 128x128 alloc: {alloc_lines}"
        assert len(sizes_64) == 1, f"Expected one 64x64 alloc: {alloc_lines}"
