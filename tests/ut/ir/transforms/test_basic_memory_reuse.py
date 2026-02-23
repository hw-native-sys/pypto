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


def _get_var_type(func, var_name):
    """Extract ShapedType for a variable by name."""
    if not isinstance(func.body, ir.SeqStmts):
        return None
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name == var_name:
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
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.ShapedType):
            assert stmt.var.type.memref is not None, f"{stmt.var.name} should have a memref"


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
