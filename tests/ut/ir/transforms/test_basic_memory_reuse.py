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
from pypto import ir, passes
from pypto.pypto_core import DataType


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
    program = passes.init_mem_ref()(program)  # Test setup: attach memrefs
    program = passes.basic_memory_reuse()(program)  # Pass under test
    return list(program.functions.values())[0]


def _assert_all_have_memrefs(func):
    """Assert all ShapedType variables have memrefs assigned."""
    assert isinstance(func.body, ir.SeqStmts)
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var.type, ir.ShapedType):
            assert stmt.var.type.memref is not None, f"{stmt.var.name} should have a memref"


def _count_alloc_stmts(func):
    """Count block.alloc AssignStmt in the function body."""
    count = 0
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "block.alloc":
            count += 1
    return count


def _get_alloc_memref_ids(func):
    """Get the set of MemRef id_ values from block.alloc statements."""
    ids = set()
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "block.alloc":
            memref = stmt.var
            assert isinstance(memref, ir.MemRef), "block.alloc LHS must be MemRef"
            ids.add(memref.id_)
    return ids


class TestBasicMemoryReuse:
    """Tests for BasicMemoryReusePass with TileType variables."""

    def test_simple(self):
        """tile_d reuses tile_a, tile_e reuses tile_b (transitive conflict prevents both from tile_a).

        Lifetimes: tile_a[0,2], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_b", "tile_e")

    def test_sequential(self):
        """Sequential chain: tile_c reuses tile_a, tile_d reuses tile_b, tile_e reuses tile_c.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")
        _assert_shares_memref(func, "tile_c", "tile_e")

    def test_different_sizes(self):
        """Small tile (32x32) can reuse large tile (64x64) buffer, not vice versa.

        tile_d (32x32) reuses tile_a (64x64) since 64x64 >= 32x32.
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
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(input_b, [0, 0], [32, 32])
                tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
                _result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], [64, 64], output_a)
                tile_d: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_b, tile_b)
                result_b: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_d, [0, 0], [32, 32], output_b)
                return result_b

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_d")

    def test_empty_function(self):
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

    def test_memref_sharing(self):
        """Chain: tile_c reuses tile_a, tile_d reuses tile_b.

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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")

    def test_with_dependencies(self):
        """tile_d reuses tile_a, tile_e reuses tile_b (transitive conflict).

        Lifetimes: tile_a[0,2], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
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
                tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
                tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_b", "tile_e")

    def test_transitive_conflict(self):
        """Transitive conflict: tile_c and tile_d must NOT share memory.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,4], tile_d[3,4], tile_e[4,5]
        tile_c reuses tile_a, tile_d reuses tile_b (not tile_a, conflict with tile_c).
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")
        _assert_not_shares_memref(func, "tile_c", "tile_d")

    def test_multiple_memory_spaces(self):
        """Memory reuse happens within the same memory space (UB tiles).

        Verifies that variables in DDR don't reuse UB memory and vice versa.
        Parameters are in DDR, tiles are in UB.

        Lifetimes: tile_a[0,2], tile_b[1,2], tile_c[2,4], tile_d[4,5]
        tile_d should reuse tile_a's UB memory.
        """

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
                # Load creates UB tiles
                tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
                # Compute creates more UB tiles (tile_a and tile_b die here)
                tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
                # Store to first output (intermediate result)
                _result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], [64, 64], output_a)
                # More UB computation (tile_c dies here)
                tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
                # Store final result
                result_b: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], [64, 64], output_b)
                return result_b

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # tile_d should reuse UB memory from tile_a
        _assert_shares_memref(func, "tile_a", "tile_d")


def _build_program_with_allocs(tile_specs, op_specs):
    """Build a Program with block.alloc stmts and operation stmts from specs.

    Args:
        tile_specs: list of (name, memref_id) for Vec tiles.
        op_specs: list of (var_name, op_name, arg_names) defining operations.
            First op uses param "input_a" as arg; others reference earlier tile vars.
            Last op is always block.store writing to param "output".
    """
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
    memref_map = {}
    stmts = []

    for name, mid in tile_specs:
        mr = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(-1, idx, span), tile_size, mid)
        memref_map[name] = mr
        tt = ir.TileType(shape, fp32, mr)
        var_map[name] = ir.Var(name, tt, span)

        alloc_call = ir.Call(
            ir.get_op("block.alloc"),
            [
                ir.ConstInt(ir.MemorySpace.Vec.value, idx, span),
                ir.ConstInt(-1, idx, span),
                ir.ConstInt(tile_size, idx, span),
                ir.ConstInt(mid, idx, span),
            ],
            span,
        )
        stmts.append(ir.AssignStmt(mr, alloc_call, span))

    offsets = ir.MakeTuple([ir.ConstInt(0, idx, span), ir.ConstInt(0, idx, span)], span)
    sizes = ir.MakeTuple([ir.ConstInt(64, idx, span), ir.ConstInt(64, idx, span)], span)

    for var_name, op_name, arg_names in op_specs:
        args = [var_map[a] for a in arg_names]
        if op_name == "block.store":
            call = ir.Call(ir.get_op(op_name), [args[0], offsets, sizes, param_out], tensor_out, span)
            result_var = ir.Var(var_name, tensor_out, span)
            var_map[var_name] = result_var
        elif op_name == "block.load":
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
    """Tests for redundant block.alloc removal after memory reuse."""

    def test_unused_alloc_removed_after_reuse(self):
        """Alloc stmts for MemRefs replaced by reuse should be removed.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,3]
        tile_c reuses tile_a → memref_c alloc removed → 2 allocs remain.
        """
        prog = _build_program_with_allocs(
            tile_specs=[("tile_a", 10), ("tile_b", 11), ("tile_c", 12)],
            op_specs=[
                ("tile_a", "block.load", ["input_a"]),
                ("tile_b", "block.add", ["tile_a", "tile_a"]),
                ("tile_c", "block.add", ["tile_b", "tile_b"]),
                ("result", "block.store", ["tile_c"]),
            ],
        )

        assert _count_alloc_stmts(list(prog.functions.values())[0]) == 3

        after = passes.basic_memory_reuse()(prog)
        func = list(after.functions.values())[0]

        assert _count_alloc_stmts(func) == 2, (
            f"Expected 2 alloc stmts after reuse, got {_count_alloc_stmts(func)}"
        )

        alloc_ids = _get_alloc_memref_ids(func)
        tile_a_type = _get_var_type(func, "tile_a")
        tile_b_type = _get_var_type(func, "tile_b")
        assert tile_a_type is not None and tile_a_type.memref is not None
        assert tile_b_type is not None and tile_b_type.memref is not None
        assert tile_a_type.memref.id_ in alloc_ids
        assert tile_b_type.memref.id_ in alloc_ids

    def test_no_alloc_removed_when_no_reuse(self):
        """When all lifetimes overlap, no reuse happens and all alloc stmts are preserved."""
        prog = _build_program_with_allocs(
            tile_specs=[("tile_a", 10), ("tile_b", 11), ("tile_c", 12)],
            op_specs=[
                ("tile_a", "block.load", ["input_a"]),
                ("tile_b", "block.load", ["input_a"]),
                ("tile_c", "block.add", ["tile_a", "tile_b"]),
                ("result", "block.store", ["tile_c"]),
            ],
        )

        assert _count_alloc_stmts(list(prog.functions.values())[0]) == 3

        after = passes.basic_memory_reuse()(prog)
        func = list(after.functions.values())[0]

        assert _count_alloc_stmts(func) == 3, "All alloc stmts should be preserved when no reuse occurs"


class TestViewOperationsMemoryReuse:
    """Tests for view operations (reshape/view/transpose) with memory reuse."""

    def test_reshape_shares_memref_with_input(self):
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # tile_b should share MemRef with tile_a (view operation)
        _assert_shares_memref(func, "tile_a", "tile_b")
        # tile_d should share MemRef with tile_c (view operation)
        _assert_shares_memref(func, "tile_c", "tile_d")

    def test_reshape_chain_shares_memref(self):
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # All tiles in the chain should share the same MemRef
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "tile_b", "tile_c")
        _assert_shares_memref(func, "tile_c", "tile_d")
        # Transitive: tile_a and tile_d should also share
        _assert_shares_memref(func, "tile_a", "tile_d")

    def test_reshape_not_broken_by_memory_reuse(self):
        """BasicMemoryReuse should propagate reuse to ALL variables sharing MemRef."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, input_a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # tile_c is dead before tile_a/tile_b are defined
                tile_c: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
                _tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)

                # tile_a and tile_b share MemRef (from InitMemRef)
                tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
                _tile_b: pl.Tile[[4096, 1], pl.FP32] = pl.reshape(tile_a, [4096, 1])

                # BasicMemoryReuse should identify: tile_a can reuse tile_c
                # When tile_a reuses tile_c, tile_b should ALSO get tile_c's MemRef
                tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # Verify tile_a and tile_b still share MemRef (propagated reuse)
        _assert_shares_memref(func, "tile_a", "_tile_b")
        # Verify both reused tile_c's buffer
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "_tile_b", "tile_c")

    def test_reshape_shared_buffer_can_be_reused_after_all_dead(self):
        """After all aliases are dead, shared buffer can be reused."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, input_a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # tile_a and tile_b share MemRef
                tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
                _tile_b: pl.Tile[[4096, 1], pl.FP32] = pl.reshape(tile_a, [4096, 1])
                _tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_a)
                # Both tile_a and tile_b are dead after this point

                # tile_d can reuse the shared buffer (tile_a/tile_b)
                tile_d: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
                tile_e: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], [64, 64], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # tile_a and tile_b should still share MemRef
        _assert_shares_memref(func, "tile_a", "_tile_b")
        # tile_d should reuse the shared buffer (either tile_a or tile_b, they're the same)
        _assert_shares_memref(func, "tile_d", "tile_a")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
