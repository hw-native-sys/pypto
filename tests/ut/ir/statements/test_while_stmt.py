# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for WhileStmt class."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


class TestWhileStmt:
    """Test WhileStmt class."""

    def test_while_stmt_creation(self):
        """Test creating a WhileStmt instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        condition = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        while_stmt = ir.WhileStmt(condition, [], assign, [], span)

        assert while_stmt is not None
        assert while_stmt.span.filename == "test.py"
        assert isinstance(while_stmt.condition, ir.Lt)
        assert isinstance(while_stmt.body, ir.AssignStmt)

    def test_while_stmt_has_attributes(self):
        """Test that WhileStmt has condition, iter_args, body, and return_vars attributes."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        condition = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        assign1 = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        assign2 = ir.AssignStmt(x, ir.Add(x, ir.ConstInt(1, dtype, span), dtype, span), span)
        body_seq = ir.SeqStmts([assign1, assign2], span)
        while_stmt = ir.WhileStmt(condition, [], body_seq, [], span)

        assert while_stmt.condition is not None
        assert isinstance(while_stmt.condition, ir.Lt)
        assert isinstance(while_stmt.body, ir.SeqStmts)
        assert len(while_stmt.body.stmts) == 2
        assert len(while_stmt.iter_args) == 0
        assert len(while_stmt.return_vars) == 0

    def test_while_stmt_with_iter_args(self):
        """Test WhileStmt with iteration arguments (SSA form)."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64

        # Create iter_arg
        init_val = ir.ConstInt(0, dtype, span)
        x_iter = ir.IterArg("x", ir.ScalarType(dtype), init_val, span)

        # Condition uses iter_arg
        condition = ir.Lt(x_iter, ir.ConstInt(10, dtype, span), dtype, span)

        # Body updates iter_arg
        x_next = ir.Add(x_iter, ir.ConstInt(1, dtype, span), dtype, span)
        yield_stmt = ir.YieldStmt([x_next], span)

        # Return var captures final value
        x_final = ir.Var("x_final", ir.ScalarType(dtype), span)

        while_stmt = ir.WhileStmt(condition, [x_iter], yield_stmt, [x_final], span)

        assert len(while_stmt.iter_args) == 1
        assert len(while_stmt.return_vars) == 1
        assert cast(ir.IterArg, while_stmt.iter_args[0]).name == "x"
        assert cast(ir.Var, while_stmt.return_vars[0]).name == "x_final"
        assert isinstance(while_stmt.body, ir.YieldStmt)

    def test_while_stmt_is_stmt(self):
        """Test that WhileStmt is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        condition = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        while_stmt = ir.WhileStmt(condition, [], assign, [], span)

        assert isinstance(while_stmt, ir.Stmt)
        assert isinstance(while_stmt, ir.IRNode)

    def test_while_stmt_immutability(self):
        """Test that WhileStmt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        condition = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        new_condition = ir.Lt(x, ir.ConstInt(20, dtype, span), dtype, span)
        assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
        while_stmt = ir.WhileStmt(condition, [], assign, [], span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            while_stmt.condition = new_condition  # type: ignore
        with pytest.raises(AttributeError):
            while_stmt.body = ir.AssignStmt(x, ir.ConstInt(1, dtype, span), span)  # type: ignore

    def test_while_stmt_structural_equal(self):
        """Test structural equality of WhileStmt instances."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        condition = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        assign = ir.AssignStmt(x, ir.Add(x, ir.ConstInt(1, dtype, span), dtype, span), span)

        while_stmt1 = ir.WhileStmt(condition, [], assign, [], span)
        while_stmt2 = ir.WhileStmt(condition, [], assign, [], span)

        # Structural equality
        assert ir.structural_equal(while_stmt1, while_stmt2)

        # Different condition
        condition2 = ir.Lt(x, ir.ConstInt(20, dtype, span), dtype, span)
        while_stmt3 = ir.WhileStmt(condition2, [], assign, [], span)
        assert not ir.structural_equal(while_stmt1, while_stmt3)

    def test_while_stmt_structural_hash(self):
        """Test structural hashing of WhileStmt instances."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        condition = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        assign = ir.AssignStmt(x, ir.Add(x, ir.ConstInt(1, dtype, span), dtype, span), span)

        while_stmt1 = ir.WhileStmt(condition, [], assign, [], span)
        while_stmt2 = ir.WhileStmt(condition, [], assign, [], span)

        # Structurally equal nodes should have same hash
        assert ir.structural_hash(while_stmt1) == ir.structural_hash(while_stmt2)

        # Different condition should have different hash
        condition2 = ir.Lt(x, ir.ConstInt(20, dtype, span), dtype, span)
        while_stmt3 = ir.WhileStmt(condition2, [], assign, [], span)
        assert ir.structural_hash(while_stmt1) != ir.structural_hash(while_stmt3)


class TestWhileStmtIterArgMutatorRemap:
    """Regression tests for IRMutator IterArg pointer remapping on WhileStmt (issue #517)."""

    def test_structural_equal_after_pass_with_iter_args(self):
        """Test that structural equality works after an IRMutator-based pass on WhileStmt with iter_args.

        Same bug as ForStmt: when an IRMutator visits a WhileStmt, if an IterArg's
        initValue_ changes, a new IterArg object is created but the body still
        references the old pointer, breaking structural equality comparison.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def f(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                acc: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tile.create(
                    [4], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                n: pl.Scalar[pl.INT64] = 0
                for (a,) in pl.while_(init_values=(acc,)):
                    pl.cond(n < 2)
                    t: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.load(x, [0], [4])
                    s: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.add(a, t)
                    r: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.yield_(s)
                out: pl.Tensor[[4], pl.FP32] = pl.store(r, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[4], pl.FP32]:
                out: pl.Tensor[[4], pl.FP32] = pl.create_tensor([4], dtype=pl.FP32)
                r: pl.Tensor[[4], pl.FP32] = self.f(x, out)
                return r

        After = passes.infer_tile_memory_space()(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def f(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                acc: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tile.create(
                    [4], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                n: pl.Scalar[pl.INT64] = 0
                for (a,) in pl.while_(init_values=(acc,)):
                    pl.cond(n < 2)
                    t: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.load(x, [0], [4])
                    s: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.add(a, t)
                    r: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.yield_(s)
                out: pl.Tensor[[4], pl.FP32] = pl.store(r, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[4], pl.FP32]:
                out: pl.Tensor[[4], pl.FP32] = pl.create_tensor([4], dtype=pl.FP32)
                r: pl.Tensor[[4], pl.FP32] = self.f(x, out)
                return r

        ir.assert_structural_equal(After, After)
        ir.assert_structural_equal(Before, Expected)
        # This was failing before the fix (issue #517):
        ir.assert_structural_equal(After, Expected)

    def test_structural_equal_after_pass_with_condition_referencing_iter_arg(self):
        """Test WhileStmt where the condition uses a scalar IterArg alongside a tile IterArg.

        Exercises the condition visitation order: iter_args must be visited and
        remapped before the condition, since WhileStmt conditions typically
        reference IterArg variables (e.g., a loop counter).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def f(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                acc: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tile.create(
                    [4], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                cnt: pl.Scalar[pl.INT64] = 0
                for c, a in pl.while_(init_values=(cnt, acc)):
                    pl.cond(c < 2)
                    t: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.load(x, [0], [4])
                    s: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.add(a, t)
                    c2: pl.Scalar[pl.INT64] = c + 1
                    rc, ra = pl.yield_(c2, s)
                out: pl.Tensor[[4], pl.FP32] = pl.store(ra, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[4], pl.FP32]:
                out: pl.Tensor[[4], pl.FP32] = pl.create_tensor([4], dtype=pl.FP32)
                r: pl.Tensor[[4], pl.FP32] = self.f(x, out)
                return r

        After = passes.infer_tile_memory_space()(Before)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def f(
                self,
                x: pl.Tensor[[4], pl.FP32],
                out: pl.Out[pl.Tensor[[4], pl.FP32]],
            ) -> pl.Tensor[[4], pl.FP32]:
                acc: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tile.create(
                    [4], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                cnt: pl.Scalar[pl.INT64] = 0
                for c, a in pl.while_(init_values=(cnt, acc)):
                    pl.cond(c < 2)
                    t: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.load(x, [0], [4])
                    s: pl.Tile[[4], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.add(a, t)
                    c2: pl.Scalar[pl.INT64] = c + 1
                    rc, ra = pl.yield_(c2, s)
                out: pl.Tensor[[4], pl.FP32] = pl.store(ra, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[4], pl.FP32]) -> pl.Tensor[[4], pl.FP32]:
                out: pl.Tensor[[4], pl.FP32] = pl.create_tensor([4], dtype=pl.FP32)
                r: pl.Tensor[[4], pl.FP32] = self.f(x, out)
                return r

        ir.assert_structural_equal(After, After)
        ir.assert_structural_equal(Before, Expected)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
