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

import pytest
from pypto import DataType, ir


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
