# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for MemoryReusePass using pl.function DSL style."""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_memory_reuse(program: ir.Program) -> ir.Function:
    """Run init_mem_ref + memory_reuse pipeline and return the first function."""
    p = passes.init_mem_ref()(program)
    after = passes.memory_reuse()(p)
    return next(iter(after.functions.values()))


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


class TestBasic:
    """Core reuse logic: chain reuse, producer-consumer, size/shape, transitive conflicts."""

    def test_simple(self):
        """tile_c, tile_d, tile_e all chain-reuse tile_a; tile_b remains independent."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                input_b: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [64, 64])
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.mul(tile_c, tile_c)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_a", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_e")

    def test_sequential(self):
        """Sequential chain: all tiles reuse tile_a (producer-consumer at same statement)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_b, tile_b)
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_c, tile_c)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "tile_b", "tile_d")
        _assert_shares_memref(func, "tile_c", "tile_e")

    def test_different_sizes(self):
        """Different-shaped tiles cannot reuse each other's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                input_b: pl.Tensor[[32, 32], pl.FP32],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                output_b: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                _result_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output_a)
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [32, 32])
                _result_b: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output_b)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_f: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [32, 32])
                _result_e: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output_a)
                result_f: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_f, [0, 0], output_b)
                return result_f

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_e")
        _assert_shares_memref(func, "tile_b", "tile_f")
        _assert_not_shares_memref(func, "tile_a", "tile_f")
        _assert_not_shares_memref(func, "tile_b", "tile_e")

    def test_empty_function(self):
        """Empty function should not crash."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                return output

        func = _run_memory_reuse(Before)
        assert func is not None
        assert func.name == "main"

    def test_transitive_conflict(self):
        """Transitive conflict: tile_c and tile_d must NOT share memory."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_b, tile_b)
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_c, tile_c)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_c, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_not_shares_memref(func, "tile_c", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_e")


class TestAllocCleanup:
    """Tests for redundant tile.alloc removal after memory reuse."""

    def test_unused_alloc_removed_after_reuse(self):
        """Alloc stmts for MemRefs replaced by reuse should be removed."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_b, tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        after_init = passes.init_mem_ref()(Before)
        func_before = next(iter(after_init.functions.values()))
        assert _count_alloc_stmts(func_before) == 3

        after = passes.memory_reuse()(after_init)
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

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        after_init = passes.init_mem_ref()(Before)
        func_before = next(iter(after_init.functions.values()))
        assert _count_alloc_stmts(func_before) == 3

        after = passes.memory_reuse()(after_init)
        func = next(iter(after.functions.values()))

        assert _count_alloc_stmts(func) == 2, (
            f"Expected 2 alloc stmts (tile_c reuses tile_a), got {_count_alloc_stmts(func)}"
        )


class TestDtype:
    """Tests that tiles with different dtypes do NOT reuse each other's memory."""

    def test_cross_dtype_no_reuse_same_dtype_reuse(self):
        """Cross-dtype reuse forbidden; same-dtype tiles reuse within their group."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                tile_cast: pl.Tile[[64, 64], pl.BF16, pl.MemorySpace.Vec] = pl.cast(
                    tile_b, target_type=pl.BF16
                )
                tile_d: pl.Tile[[64, 64], pl.BF16, pl.MemorySpace.Vec] = pl.add(tile_cast, tile_cast)
                tile_e: pl.Tile[[64, 64], pl.BF16, pl.MemorySpace.Vec] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_cast")
        _assert_not_shares_memref(func, "tile_b", "tile_cast")
        _assert_not_shares_memref(func, "tile_a", "tile_d")
        _assert_not_shares_memref(func, "tile_b", "tile_e")
        _assert_shares_memref(func, "tile_cast", "tile_d")
        _assert_shares_memref(func, "tile_cast", "tile_e")


class TestFillpad:
    """Tests that fillpad output does NOT reuse input due to TileView differences."""

    def test_fillpad_output_incompatible_with_input(self):
        """fillpad changes valid_shape and pad: output cannot reuse input."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_a, pad_value=pl.PadValue.max
                )
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "padded")

    def test_fillpad_different_pad_no_reuse(self):
        """Two fillpad outputs with different pad values cannot reuse each other."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                output_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded_max: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_a, pad_value=pl.PadValue.max
                )
                _res_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_max, [0, 0], output_a)
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded_min: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_b, pad_value=pl.PadValue.min
                )
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_min, [0, 0], output_b)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_not_shares_memref(func, "padded_max", "padded_min")

    def test_fillpad_same_pad_can_reuse(self):
        """Two fillpad outputs with identical TileView attributes CAN reuse."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                output_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_a, pad_value=pl.PadValue.max
                )
                _res_a: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_a, [0, 0], output_a)
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_a, [0, 0], [64, 64], valid_shapes=[48, 64]
                )
                padded_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.fillpad(
                    tile_b, pad_value=pl.PadValue.max
                )
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(padded_b, [0, 0], output_b)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "padded_a", "padded_b")


class TestViewOps:
    """Tests for view operations (reshape) with memory reuse."""

    def test_reshape_chain_shares_memref(self):
        """Chained reshapes should all share the same MemRef."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_a, [4096, 1])
                tile_c: pl.Tile[[1, 4096], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_b, [1, 4096])
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_c, [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "tile_b", "tile_c")
        _assert_shares_memref(func, "tile_c", "tile_d")
        _assert_shares_memref(func, "tile_a", "tile_d")

    def test_reshape_not_broken_by_memory_reuse(self):
        """MemoryReuse should propagate reuse to ALL variables sharing MemRef."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # tile_c is dead before tile_a/tile_b are defined
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                _tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_c, tile_c)
                # tile_a and _tile_b share MemRef (reshape = view alias)
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                _tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_a, [4096, 1])
                # MemoryReuse: tile_a reuses tile_c -> _tile_b also gets tile_c's MemRef
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "_tile_b")
        _assert_shares_memref(func, "tile_a", "tile_c")
        _assert_shares_memref(func, "_tile_b", "tile_c")

    def test_reshape_shared_buffer_can_be_reused_after_all_dead(self):
        """After all aliases are dead, shared buffer can be reused."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # tile_a and _tile_b share MemRef
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                _tile_b: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(tile_a, [4096, 1])
                _tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                # Both tile_a and _tile_b are dead -> tile_d can reuse the shared buffer
                tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_d, tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "_tile_b")
        _assert_shares_memref(func, "tile_d", "tile_a")


class TestInplaceOps:
    """Tests verifying that ops marked not_inplace_safe block producer-consumer reuse."""

    def test_inplace_unsafe_op_no_producer_consumer_reuse(self):
        """tile.recip must NOT reuse its input's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.recip(tile_a)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_inplace_unsafe_op_allows_non_producer_consumer_reuse(self):
        """tile.recip output must never share a buffer with its input."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                input_c: pl.Tensor[[32, 32], pl.FP32],
                input_x: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                _s1: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_a, [0, 0], output)
                tile_c: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_c, [0, 0], [32, 32])
                _s2: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], output)
                tile_x: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_x, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.recip(tile_x)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_x", "tile_b")

    def test_inplace_safe_op_allows_producer_consumer_reuse(self):
        """tile.add (inplace-safe) CAN reuse its input's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_shares_memref(func, "tile_a", "tile_b")

    def test_ands_no_producer_consumer_reuse(self):
        """tile.ands must NOT reuse its input's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.INT32],
                output: pl.Out[pl.Tensor[[32, 32], pl.INT32]],
            ) -> pl.Tensor[[32, 32], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.ands(tile_a, 255)
                result: pl.Tensor[[32, 32], pl.INT32] = pl.store(tile_b, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_xors_no_producer_consumer_reuse(self):
        """tile.xors must NOT reuse its input's buffer."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.INT32],
                input_b: pl.Tensor[[32, 32], pl.INT32],
                output: pl.Out[pl.Tensor[[32, 32], pl.INT32]],
            ) -> pl.Tensor[[32, 32], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_tmp: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32, pl.MemorySpace.Vec] = pl.xors(tile_a, 255, tile_tmp)
                result: pl.Tensor[[32, 32], pl.INT32] = pl.store(tile_b, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_a", "tile_b")

    def test_inplace_unsafe_two_level_transitive_chain(self):
        """tile.recip must not reuse a buffer occupied by its input via a two-level chain."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP32],
                input_u: pl.Tensor[[32, 32], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                _s1: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_b, [0, 0], output)
                tile_u: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_u, [0, 0], [32, 32])
                tile_d: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_u, tile_u)
                _s2: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_u, [0, 0], output)
                tile_c: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Vec] = pl.recip(tile_d)
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        _assert_all_have_memrefs(func)
        _assert_not_shares_memref(func, "tile_d", "tile_c")


class TestYieldFixup:
    """Yield fixup for ForStmt and IfStmt -- ensuring loop-carry and return variables share correct MemRef."""

    def test_tile_move_inserted_when_memrefs_diverge(self):
        """When initValue and yield value start with different MemRefs,
        the pass should unify all loop-carry vars to share one MemRef."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc_0,) in pl.range(0, 4, init_values=(init_0,)):
                    # extra_0 keeps acc_0 alive past next_0's definition
                    extra_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(extra_0, acc_0)
                    out_0 = pl.yield_(next_0)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(out_0, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
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

    def test_simple_loop_memrefs_unified(self):
        """Simple loop: after reuse, iter_arg/initValue/return_var share MemRef."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc_0,) in pl.range(0, 4, init_values=(init_0,)):
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                    out_0 = pl.yield_(next_0)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(out_0, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        loop = _find_first_for_stmt(func.body)
        assert loop is not None

        ia = loop.iter_args[0]
        assert isinstance(ia.initValue.type, ir.ShapedType)
        assert isinstance(ia.type, ir.ShapedType)
        assert ia.type.shares_memref_with(ia.initValue.type)

        rv = loop.return_vars[0]
        assert isinstance(rv.type, ir.ShapedType)
        assert rv.type.shares_memref_with(ia.type), "return_var should share iter_arg's MemRef"

    def test_multiple_iter_args_partial_mismatch(self):
        """With 2 iter_args, tile.move inserted only for the mismatched pair."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                init_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc_0, acc_1) in pl.range(0, 4, init_values=(init_0, init_1)):
                    # extra ops keep acc alive past next's definition (add_overlap pattern)
                    extra_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_0, acc_0)
                    next_0: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(extra_0, acc_0)
                    extra_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_1, acc_1)
                    next_1: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(extra_1, acc_1)
                    out_0, out_1 = pl.yield_(next_0, next_1)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(out_0, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
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

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # tile_a: dead before IfStmt
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                _: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output)
                # IfStmt with return vars
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_b)
                else:
                    tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_c)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(if_result, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)

        # tile_b and tile_c should reuse tile_a (tile_a is dead before IfStmt)
        _assert_shares_memref(func, "tile_a", "tile_b")
        _assert_shares_memref(func, "tile_b", "tile_c")

        # After reuse, if_result's MemRef should be patched by YieldFixupMutator
        if_result_type = _get_var_type(func, "if_result")
        tile_b_type = _get_var_type(func, "tile_b")
        if if_result_type is not None and tile_b_type is not None:
            assert if_result_type.shares_memref_with(tile_b_type), (
                "if_result should share MemRef with tile_b after YieldFixupMutator patches it"
            )


class TestControlFlow:
    """Tests for correct lifetime analysis across control flow boundaries."""

    def test_var_used_in_nested_if_not_reused_in_loop(self):
        """Variable defined before loop, used inside IfStmt within loop body,
        must NOT have its MemRef reused by other loop-body variables."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for i, (acc,) in pl.range(0, 4, init_values=(tile_a,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc, tile_a)
                        if_result = pl.yield_(tile_c)
                    else:
                        if_result = pl.yield_(acc)
                    loop_out = pl.yield_(if_result)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        # tile_a must NOT share MemRef with tile_c -- tile_a is live through the loop
        _assert_not_shares_memref(func, "tile_a", "tile_c")

    def test_different_if_branches_can_share(self):
        """Variables in different IfStmt branches should be able to share MemRef
        since they have non-overlapping lifetimes."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_b)
                else:
                    tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_c)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(if_result, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        # tile_b and tile_c are in different branches -- they CAN share MemRef
        _assert_shares_memref(func, "tile_b", "tile_c")

    def test_loop_local_var_can_be_reused(self):
        """Variables defined AND used entirely within a single loop iteration
        can still be reused with other loop-local variables."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                for _i, (acc,) in pl.range(0, 4, init_values=(init_tile,)):
                    tile_x: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    tile_y: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_x, tile_x)
                    tile_z: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_y, tile_y)
                    loop_out = pl.yield_(tile_z)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        # tile_x and tile_z should share MemRef (both loop-local, non-overlapping)
        _assert_shares_memref(func, "tile_x", "tile_z")

    def test_nested_for_loops_outer_var_extends_to_outer_end(self):
        """Variable defined before nested loops, used in inner loop body --
        lifetime must extend to the END of the OUTER loop (not just inner)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                init_outer: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                for _i, (acc_outer,) in pl.range(0, 4, init_values=(init_outer,)):
                    init_inner: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                        [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    for _j, (acc_inner,) in pl.range(0, 4, init_values=(init_inner,)):
                        # tile_a used in inner loop!
                        tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_inner, tile_a)
                        inner_out = pl.yield_(tile_b)
                    tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_outer, inner_out)
                    outer_out = pl.yield_(tile_d)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(outer_out, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        # tile_a used in inner loop but defined outside outer loop -> must NOT be reused
        _assert_not_shares_memref(func, "tile_a", "tile_b")
        _assert_not_shares_memref(func, "tile_a", "tile_d")

    def test_if_without_else_branch(self):
        """IfStmt with only then branch (no else) should not crash and
        correctly track variable uses inside then body."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                    _: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
                    pl.yield_()
                # tile_c defined after if -- tile_a should still be alive through IfStmt
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        # tile_a is used both inside IfStmt (then branch) and after it -> still alive
        # tile_b (inside then) overlaps with tile_a -> cannot reuse
        _assert_not_shares_memref(func, "tile_a", "tile_b")
        # tile_c is after tile_a's last use -> can reuse tile_a (greedy first-fit)
        _assert_shares_memref(func, "tile_a", "tile_c")

    def test_for_with_if_multiple_vars_competing(self):
        """ForStmt with IfStmt inside, multiple variables from before the loop
        used inside the if -- tests that ALL outer variables are correctly extended."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
                    [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                for i, (acc,) in pl.range(0, 4, init_values=(init_tile,)):
                    if i < 2:
                        tile_c: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                        if_result = pl.yield_(tile_c)
                    else:
                        tile_d: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_b, tile_a)
                        if_result = pl.yield_(tile_d)
                    loop_out = pl.yield_(if_result)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(loop_out, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        # tile_a and tile_b are both used inside the nested IfStmt in the loop --
        # their lifetimes extend to loop end, so tile_c and tile_d cannot reuse them
        _assert_not_shares_memref(func, "tile_a", "tile_c")
        _assert_not_shares_memref(func, "tile_a", "tile_d")
        _assert_not_shares_memref(func, "tile_b", "tile_c")
        _assert_not_shares_memref(func, "tile_b", "tile_d")
        # tile_c and tile_d are in different branches -- they CAN share
        _assert_shares_memref(func, "tile_c", "tile_d")

    def test_branch_local_var_does_not_leak(self):
        """A variable defined and consumed entirely inside one IfStmt branch
        should have a short lifetime and not block reuse after the IfStmt."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond_param: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                if cond_param < 2:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_a)
                    if_result = pl.yield_(tile_b)
                else:
                    if_result = pl.yield_(tile_a)
                tile_e: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(if_result, if_result)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_e, [0, 0], output)
                return result

        func = _run_memory_reuse(Before)
        # tile_b is local to then-branch. tile_e is defined after IfStmt.
        # tile_a's last use is in the else-yield which ends before tile_e's def
        _assert_shares_memref(func, "tile_a", "tile_e")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
