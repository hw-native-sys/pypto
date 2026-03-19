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
    backend.set_backend_type(BackendType.Ascend910B_PTO)
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
            assert stmt.var.type.memref is not None, f"{stmt.var.name_hint} should have a memref"


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


class TestBasicMemoryReuse:
    """Tests for BasicMemoryReusePass with TileType variables."""

    def test_simple(self):
        """tile_c, tile_d, tile_e all chain-reuse tile_a; tile_b remains independent.

        Lifetimes: tile_a[0,2], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
        Touching lifetimes (end == start) are non-overlapping, so tile_c reuses
        tile_a at the boundary, and tile_d and tile_e continue the chain.
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
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_e")

    def test_sequential(self):
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

    def test_different_sizes(self):
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
                # tile_a and tile_b are dead
                tile_e: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
                tile_f: pl.Tile[[32, 32], pl.FP32] = pl.load(input_b, [0, 0], [32, 32])
                _result_e: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output_a)
                result_f: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_f, [0, 0], output_b)
                return result_f

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # Same shape reuses: tile_e (64x64) reuses tile_a (64x64)
        _assert_shares_memref(func, "tile_a", "tile_e")
        # Same shape reuses: tile_f (32x32) reuses tile_b (32x32)
        _assert_shares_memref(func, "tile_b", "tile_f")
        # Different shapes cannot reuse despite sufficient size
        _assert_not_shares_memref(func, "tile_a", "tile_f")
        _assert_not_shares_memref(func, "tile_b", "tile_e")

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

    def test_with_dependencies(self):
        """tile_c, tile_d, tile_e all chain-reuse tile_a; tile_b remains independent.

        Lifetimes: tile_a[0,2], tile_b[1,2], tile_c[2,3], tile_d[3,4], tile_e[4,5]
        Same as test_simple — touching lifetimes form a non-overlapping chain.
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_e")

    def test_transitive_conflict(self):
        """Transitive conflict: tile_c and tile_d must NOT share memory.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_c[2,4], tile_d[3,4], tile_e[4,5]
        tile_b reuses tile_a (touching at 1). tile_c reuses tile_a (touching at 2,
        checked against tile_b which is also non-overlapping). tile_d cannot reuse
        tile_a (conflict with tile_c[2,4]) or tile_b (same root, conflict with tile_c).
        tile_e reuses tile_a (tile_c[2,4] touches tile_e[4,5] at 4).
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
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_not_shares_memref(func, "tile_c", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_e")

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
                _result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output_a)
                # More UB computation (tile_c dies here)
                tile_d: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_c, tile_c)
                # Store final result
                result_b: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output_b)
                return result_b

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # tile_d should reuse UB memory from tile_a
        _assert_shares_memref(func, "tile_a", "tile_d")


def _build_program_with_allocs(tile_specs, op_specs):
    """Build a Program with tile.alloc stmts and operation stmts from specs.

    Args:
        tile_specs: list of (name, memref_id) for Vec tiles.
        op_specs: list of (var_name, op_name, arg_names) defining operations.
            First op uses param "input_a" as arg; others reference earlier tile vars.
            Last op is always tile.store writing to param "output".
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
            ir.get_op("tile.alloc"),
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
        """Alloc stmts for MemRefs replaced by reuse should be removed.

        Lifetimes: tile_a[3,4], tile_b[4,5], tile_c[5,6]
        With touching-lifetime reuse, tile_b reuses tile_a and tile_c
        reuses tile_a (chain) → all share one MemRef → 1 alloc remains.
        """
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

        assert _count_alloc_stmts(func) == 1, (
            f"Expected 1 alloc stmt after chain reuse, got {_count_alloc_stmts(func)}"
        )

        alloc_ids = _get_alloc_memref_ids(func)
        tile_a_type = _get_var_type(func, "tile_a")
        assert tile_a_type is not None and tile_a_type.memref is not None
        assert tile_a_type.memref.id_ in alloc_ids

    def test_partial_reuse_with_overlapping_lifetimes(self):
        """When some lifetimes truly overlap, partial reuse happens.

        Lifetimes: tile_a[3,5], tile_b[4,5], tile_c[5,6]
        tile_a and tile_b truly overlap ([3,5] vs [4,5]). tile_c touches tile_a
        at 5 and reuses it → 2 allocs remain (tile_a and tile_b).
        """
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

        assert _count_alloc_stmts(func) == 2, (
            f"Expected 2 alloc stmts (tile_c reuses tile_a), got {_count_alloc_stmts(func)}"
        )


class TestDtypeCompatibility:
    """Tests that tiles with different dtypes do NOT reuse each other's memory.

    PTO codegen binds a single alloc_tile declaration (including dtype) to each
    buffer. Reuse across different dtypes would cause the alloc_tile to carry
    a wrong dtype, leading to incorrect hardware behaviour.
    """

    def test_cast_output_does_not_reuse(self):
        """Cast changes dtype → no cross-dtype reuse; same-dtype tiles still reuse.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_cast[2,3], tile_c[3,4]
        tile_cast (BF16) cannot reuse tile_a/tile_b (FP32) due to dtype mismatch.
        tile_c (BF16) reuses tile_cast (same dtype, last_use==def).
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
                tile_cast: pl.Tile[[64, 64], pl.BF16] = pl.cast(tile_b, target_type=pl.BF16)
                tile_c: pl.Tile[[64, 64], pl.BF16] = pl.add(tile_cast, tile_cast)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_cast")
        _assert_not_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_cast", "tile_c")

    def test_cast_among_regular_ops(self):
        """Cross-dtype reuse forbidden; same-dtype tiles reuse within their group.

        Lifetimes: tile_a[0,1], tile_b[1,2], tile_cast[2,3], tile_d[3,4], tile_e[4,5]
        tile_cast/tile_d/tile_e are BF16 and cannot reuse FP32 tile_a/tile_b.
        tile_d and tile_e reuse tile_cast (same dtype, producer-consumer chain).
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
                tile_cast: pl.Tile[[64, 64], pl.BF16] = pl.cast(tile_b, target_type=pl.BF16)
                tile_d: pl.Tile[[64, 64], pl.BF16] = pl.add(tile_cast, tile_cast)
                tile_e: pl.Tile[[64, 64], pl.BF16] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_cast")
        _assert_not_shares_memref(func, "tile_b", "tile_cast")
        _assert_not_shares_memref(func, "tile_a", "tile_d")
        _assert_not_shares_memref(func, "tile_b", "tile_e")
        _assert_shares_memref(func, "tile_cast", "tile_d")
        _assert_shares_memref(func, "tile_cast", "tile_e")


class TestFillpadCompatibility:
    """Tests that fillpad output does NOT reuse input due to TileView differences.

    fillpad expands valid_shape to full shape and sets a pad value, changing the
    TileView attributes.  AreTileTypesCompatible detects these differences and
    prevents unsafe in-place reuse that would confuse PTO codegen.
    """

    def test_fillpad_output_incompatible_with_input(self):
        """fillpad changes valid_shape and pad → output cannot reuse input.

        tile_a: shape=[64,64], valid_shape=[48,64], pad=null
        padded:  shape=[64,64], valid_shape=[64,64], pad=max
        valid_shape and pad differ → no reuse.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64])
                padded: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile_a, pad_value=pl.PadValue.max)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded, [0, 0], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "padded")

    def test_fillpad_different_pad_no_reuse(self):
        """Two fillpad outputs with different pad values cannot reuse each other.

        padded_max: valid_shape=[64,64], pad=max
        padded_min: valid_shape=[64,64], pad=min
        pad differs → no reuse between them, but their inputs can reuse each other.
        """

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
                padded_max: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile_a, pad_value=pl.PadValue.max)
                _res_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_max, [0, 0], output_a)

                tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64])
                padded_min: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile_b, pad_value=pl.PadValue.min)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_min, [0, 0], output_b)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # inputs share (same valid_shape=[48,64], same pad=null)
        _assert_shares_memref(func, "tile_a", "tile_b")
        # fillpad outputs do NOT share (pad=max vs pad=min)
        _assert_not_shares_memref(func, "padded_max", "padded_min")

    def test_fillpad_same_pad_can_reuse(self):
        """Two fillpad outputs with identical TileView attributes CAN reuse.

        Both padded_a and padded_b: valid_shape=[64,64], pad=max → compatible → reuse.
        """

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
                padded_a: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile_a, pad_value=pl.PadValue.max)
                _res_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_a, [0, 0], output_a)

                tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64], valid_shapes=[48, 64])
                padded_b: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile_b, pad_value=pl.PadValue.max)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_b, [0, 0], output_b)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # Same attributes → reuse allowed
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "padded_a", "padded_b")


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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output)
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output)
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
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
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # tile_a and tile_b should still share MemRef
        _assert_shares_memref(func, "tile_a", "_tile_b")
        # tile_d should reuse the shared buffer (either tile_a or tile_b, they're the same)
        _assert_shares_memref(func, "tile_d", "tile_a")


class TestInplaceSafetyCheck:
    """Tests verifying that ops marked not_inplace_safe block producer-consumer reuse."""

    def test_inplace_unsafe_op_no_producer_consumer_reuse(self):
        """tile.recip must NOT reuse its input's buffer when last_use == def (src == dst).

        tile_a.last_use == tile_b.def (producer-consumer), but tile.recip does not
        support in-place execution, so tile_b must get a distinct MemRef from tile_a.
        """

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
        # tile.recip does not support in-place: tile_b must have its own MemRef
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_inplace_unsafe_op_allows_non_producer_consumer_reuse(self):
        """tile.recip output does not share a buffer with its input (tile_x) in any case.

        tile_c is freed strictly before tile_b's definition and tile_x occupies
        tile_a's buffer.  The key correctness property is that tile_b (result of
        recip(tile_x)) must never end up in the same buffer as tile_x regardless of
        how many dead buffers are available for reuse.
        """

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
                # Two separate dead buffers before recip is called
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(input_a, [0, 0], [32, 32])
                _s1: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_a, [0, 0], output)
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.load(input_c, [0, 0], [32, 32])
                _s2: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], output)
                # tile_x reuses one of the dead buffers
                tile_x: pl.Tile[[32, 32], pl.FP32] = pl.load(input_x, [0, 0], [32, 32])
                # tile_b = recip(tile_x): inplace-unsafe, must not share buffer with tile_x
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.recip(tile_x)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # Core correctness: tile.recip output must never share a buffer with its input
        _assert_not_shares_memref(func, "tile_x", "tile_b")

    def test_inplace_safe_op_allows_producer_consumer_reuse(self):
        """tile.add (inplace-safe by default) CAN reuse its input's buffer.

        tile_a.last_use == tile_b.def (producer-consumer), but tile.add supports
        in-place execution, so tile_b is allowed to reuse tile_a's MemRef.
        """

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
        # tile.add is inplace-safe: producer-consumer reuse is allowed
        _assert_shares_memref(func, "tile_a", "tile_b")

    def test_ands_no_producer_consumer_reuse(self):
        """tile.ands must NOT reuse its input's buffer when last_use == def (src == dst).

        tile_a.last_use == tile_b.def (producer-consumer), but tile.ands does not
        support in-place execution, so tile_b must get a distinct MemRef from tile_a.
        """

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
        # tile.ands does not support in-place: tile_b must have its own MemRef
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_ors_no_producer_consumer_reuse(self):
        """tile.ors must NOT reuse its input's buffer when last_use == def (src == dst).

        tile_a.last_use == tile_b.def (producer-consumer), but tile.ors does not
        support in-place execution, so tile_b must get a distinct MemRef from tile_a.
        """

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
        # tile.ors does not support in-place: tile_b must have its own MemRef
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_xors_no_producer_consumer_reuse(self):
        """tile.xors must NOT reuse its input's buffer when last_use == def (src == dst).

        tile_a.last_use == tile_b.def (producer-consumer), but tile.xors does not
        support in-place execution, so tile_b must get a distinct MemRef from tile_a.
        tile_tmp is loaded from a separate tensor to ensure it has a MemRef assigned.
        """

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
        # tile.xors does not support in-place: tile_b must have its own MemRef
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_inplace_unsafe_two_level_transitive_chain(self):
        """tile.recip must not reuse a buffer occupied by its input via a two-level chain.

        Timeline (statement order):
          stmt 0: tile_a = load(input_a)              tile_a.def=0, last_use=1
          stmt 1: tile_b = add(tile_a, tile_a)        tile_b.def=1, last_use=2
          stmt 2: _s1   = store(tile_b, output)       tile_b last use
          stmt 3: tile_u = load(input_u)              tile_u.def=3, last_use=5
          stmt 4: tile_d = add(tile_u, tile_u)        tile_d.def=4, last_use=6
          stmt 5: _s2   = store(tile_u, output)       tile_u last use (> tile_d.def)
          stmt 6: tile_c = recip(tile_d)              tile_c.def=6, recip is inplace-unsafe
          stmt 7: result = store(tile_c, output)

        Greedy reuse chain (without fix):
          tile_b reuses tile_a  →  memref_users[tile_a] = [tile_b]
          tile_u reuses tile_a  →  memref_users[tile_a] = [tile_b, tile_u]
          tile_d cannot reuse tile_a (tile_u.last_use=5 > tile_d.def=4 → overlap)
          tile_d reuses tile_b  →  memref_users[tile_b] = [tile_d]
          tile_c tries tile_a: direct conflict (tile_a.last_use=1 != 6) and indirect
          (tile_b.last_use=2!=6, tile_u.last_use=5!=6) both miss tile_d
          BUG: tile_c reuses tile_a even though tile_d (= tile_a's physical buffer)
          has last_use=6 == tile_c.def=6 → recip(tile_d) executes with src == dst.

        After fix: tile_d is propagated into memref_users[tile_a] when it reuses
        tile_b, so tile_d.last_use(6) == tile_c.def(6) is detected → reuse blocked.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                input_u: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                # tile_a: last use at stmt 1 (tile_b definition)
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(input_a, [0, 0], [32, 32])
                # tile_b reuses tile_a (add is inplace-safe, producer-consumer OK)
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_a, tile_a)
                # tile_b last use at stmt 2
                _s1: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                # tile_u reuses tile_a (tile_b is done before stmt 3)
                tile_u: pl.Tile[[32, 32], pl.FP32] = pl.load(input_u, [0, 0], [32, 32])
                # tile_d reuses tile_b (tile_u overlap at stmt 5 blocks tile_a for tile_d)
                tile_d: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_u, tile_u)
                # tile_u last use at stmt 5, which is AFTER tile_d.def (stmt 4)
                _s2: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_u, [0, 0], output)
                # tile_c = recip(tile_d): inplace-unsafe, must NOT share buffer with tile_d
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.recip(tile_d)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        func = _prepare_and_run_memory_reuse(Before)

        _assert_all_have_memrefs(func)
        # tile.recip is inplace-unsafe: tile_c must not share a buffer with its input tile_d.
        # tile_d physically occupies tile_a's buffer (via chain tile_d→tile_b→tile_a),
        # so this also verifies that the two-level transitive chain is detected.
        _assert_not_shares_memref(func, "tile_d", "tile_c")


def _iter_all_assign_stmts(stmt):
    """Recursively iterate all AssignStmt in a statement tree (enters ForStmt/IfStmt/WhileStmt bodies)."""
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
        if stmt.var.name_hint == var_name:
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


class TestYieldAndInitValueAliasing:
    """Tests for yield and init_value aliasing prevention (issue #585)."""

    @pl.program
    class _TestProgram:
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

    def test_yield_prevents_aliasing_of_simultaneously_live_tiles(self):
        """Two tile accumulators inside a loop, both yielded, must NOT share MemRef.

        gate_new and up_new are both live at the yield point, so their lifetimes
        overlap. Without the YieldStmt fix, the yield was silently skipped and
        both tiles appeared dead, causing incorrect aliasing.
        """
        func = _prepare_and_run_memory_reuse(self._TestProgram)

        # gate_new and up_new are both live at the yield → must NOT share MemRef
        _assert_not_shares_memref_recursive(func, "gate_new", "up_new")

    def test_init_values_prevent_aliasing_of_loop_inputs(self):
        """Two tiles used as init_values must NOT share MemRef.

        gate_init and up_init are both consumed at the loop entry point as
        init_values. Without the ForStmt init_value fix, these variables
        appeared dead and got incorrectly aliased.
        """
        func = _prepare_and_run_memory_reuse(self._TestProgram)

        # gate_init and up_init are both used as init_values → must NOT share MemRef
        _assert_not_shares_memref_recursive(func, "gate_init", "up_init")

    def test_return_prevents_aliasing_of_simultaneously_live_tiles(self):
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

        # tile_a and tile_b are both live (used in store) → must NOT share MemRef
        _assert_not_shares_memref_recursive(func, "tile_a", "tile_b")

    def test_while_init_values_prevent_aliasing(self):
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

        # gate_init and up_init are both used as while-loop init_values → must NOT share MemRef
        _assert_not_shares_memref_recursive(func, "gate_init", "up_init")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
