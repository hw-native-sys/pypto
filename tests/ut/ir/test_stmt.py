# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Stmt base class."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestStmt:
    """Test Stmt base class."""

    def test_stmt_creation(self):
        """Test creating a Stmt instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        stmt = ir.Stmt(span)
        assert stmt is not None
        assert stmt.span.filename == "test.py"

    def test_stmt_has_span(self):
        """Test that Stmt has span attribute."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        stmt = ir.Stmt(span)
        assert stmt.span.begin_line == 10
        assert stmt.span.begin_column == 5

    def test_stmt_is_irnode(self):
        """Test that Stmt is an instance of IRNode."""
        span = ir.Span.unknown()
        stmt = ir.Stmt(span)
        assert isinstance(stmt, ir.IRNode)

    def test_stmt_immutability(self):
        """Test that Stmt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        stmt = ir.Stmt(span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            stmt.span = ir.Span("other.py", 2, 2, 2, 5)  # type: ignore

    def test_stmt_with_unknown_span(self):
        """Test creating Stmt with unknown span."""
        span = ir.Span.unknown()
        stmt = ir.Stmt(span)
        assert stmt.span.is_valid() is False


class TestAssignStmt:
    """Test AssignStmt class."""

    def test_assign_stmt_creation(self):
        """Test creating an AssignStmt instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        assert assign is not None
        assert assign.span.filename == "test.py"
        assert cast(ir.Var, assign.var).name == "x"
        assert cast(ir.Var, assign.value).name == "y"

    def test_assign_stmt_has_lhs_rhs(self):
        """Test that AssignStmt has var and value attributes."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(a, b, span)

        assert assign.var is not None
        assert assign.value is not None
        assert cast(ir.Var, assign.var).name == "a"
        assert cast(ir.Var, assign.value).name == "b"

    def test_assign_stmt_is_stmt(self):
        """Test that AssignStmt is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        assert isinstance(assign, ir.Stmt)
        assert isinstance(assign, ir.IRNode)

    def test_assign_stmt_immutability(self):
        """Test that AssignStmt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            assign.var = ir.Var("z", ir.ScalarType(dtype), span)  # type: ignore
        with pytest.raises(AttributeError):
            assign.value = ir.Var("w", ir.ScalarType(dtype), span)  # type: ignore

    def test_assign_stmt_with_different_expressions(self):
        """Test AssignStmt with different expression types."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64

        # Test with Var on value
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assert cast(ir.Var, assign1.var).name == "x"
        assert cast(ir.Var, assign1.value).name == "y"

        # Test with ConstInt on value
        c5 = ir.ConstInt(5, dtype, span)
        assign2 = ir.AssignStmt(x, c5, span)
        assert cast(ir.Var, assign2.var).name == "x"
        assert cast(ir.ConstInt, assign2.value).value == 5

        # Test with Call on value
        op = ir.Op("add")
        z = ir.Var("z", ir.ScalarType(dtype), span)
        call = ir.Call(op, [x, z], span)
        assign3 = ir.AssignStmt(y, call, span)
        assert cast(ir.Var, assign3.var).name == "y"
        assert isinstance(assign3.value, ir.Call)

        # Test with binary expression on value
        add_expr = ir.Add(x, z, dtype, span)
        assign4 = ir.AssignStmt(x, add_expr, span)
        assert cast(ir.Var, assign4.var).name == "x"
        assert isinstance(assign4.value, ir.Add)


class TestIfStmt:
    """Test IfStmt class."""

    def test_if_stmt_creation(self):
        """Test creating an IfStmt instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)
        if_stmt = ir.IfStmt(condition, [assign], [], span)

        assert if_stmt is not None
        assert if_stmt.span.filename == "test.py"
        assert isinstance(if_stmt.condition, ir.Eq)
        assert len(if_stmt.then_body) == 1
        assert len(if_stmt.else_body) == 0

    def test_if_stmt_has_attributes(self):
        """Test that IfStmt has condition, then_body, and else_body attributes."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        condition = ir.Lt(a, b, dtype, span)
        assign1 = ir.AssignStmt(a, b, span)
        assign2 = ir.AssignStmt(b, a, span)
        if_stmt = ir.IfStmt(condition, [assign1], [assign2], span)

        assert if_stmt.condition is not None
        assert len(if_stmt.then_body) == 1
        assert len(if_stmt.else_body) == 1
        assert isinstance(if_stmt.then_body[0], ir.AssignStmt)
        assert isinstance(if_stmt.else_body[0], ir.AssignStmt)

    def test_if_stmt_is_stmt(self):
        """Test that IfStmt is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)
        if_stmt = ir.IfStmt(condition, [assign], [], span)

        assert isinstance(if_stmt, ir.Stmt)
        assert isinstance(if_stmt, ir.IRNode)

    def test_if_stmt_immutability(self):
        """Test that IfStmt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)
        if_stmt = ir.IfStmt(condition, [assign], [], span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            if_stmt.condition = ir.Eq(y, x, dtype, span)  # type: ignore
        with pytest.raises(AttributeError):
            if_stmt.then_body = []  # type: ignore
        with pytest.raises(AttributeError):
            if_stmt.else_body = []  # type: ignore

    def test_if_stmt_with_empty_else_body(self):
        """Test IfStmt with empty else_body."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)
        if_stmt = ir.IfStmt(condition, [assign], [], span)

        assert len(if_stmt.else_body) == 0

    def test_if_stmt_with_different_condition_types(self):
        """Test IfStmt with different condition expression types."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        # Test with Eq condition
        condition1 = ir.Eq(x, y, dtype, span)
        if_stmt1 = ir.IfStmt(condition1, [assign], [], span)
        assert isinstance(if_stmt1.condition, ir.Eq)

        # Test with Lt condition
        condition2 = ir.Lt(x, y, dtype, span)
        if_stmt2 = ir.IfStmt(condition2, [assign], [], span)
        assert isinstance(if_stmt2.condition, ir.Lt)

        # Test with And condition
        condition3 = ir.And(x, y, dtype, span)
        if_stmt3 = ir.IfStmt(condition3, [assign], [], span)
        assert isinstance(if_stmt3.condition, ir.And)

    def test_if_stmt_with_multiple_statements(self):
        """Test IfStmt with multiple statements in then_body and else_body."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)
        assign3 = ir.AssignStmt(z, x, span)
        if_stmt = ir.IfStmt(condition, [assign1, assign2], [assign3], span)

        assert len(if_stmt.then_body) == 2
        assert len(if_stmt.else_body) == 1


class TestYieldStmt:
    """Test YieldStmt class."""

    def test_yield_stmt_creation_with_value(self):
        """Test creating a YieldStmt instance with a value."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        yield_stmt = ir.YieldStmt([x], span)

        assert yield_stmt is not None
        assert yield_stmt.span.filename == "test.py"
        assert len(yield_stmt.value) == 1
        assert yield_stmt.value[0].name == "x"

    def test_yield_stmt_creation_without_value(self):
        """Test creating a YieldStmt instance without a value."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        yield_stmt = ir.YieldStmt(span)

        assert yield_stmt is not None
        assert yield_stmt.span.filename == "test.py"
        assert len(yield_stmt.value) == 0

    def test_yield_stmt_has_value_attribute(self):
        """Test that YieldStmt has value attribute."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        yield_stmt = ir.YieldStmt([a], span)

        assert len(yield_stmt.value) == 1
        assert yield_stmt.value[0].name == "a"

    def test_yield_stmt_is_stmt(self):
        """Test that YieldStmt is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        yield_stmt = ir.YieldStmt([x], span)

        assert isinstance(yield_stmt, ir.Stmt)
        assert isinstance(yield_stmt, ir.IRNode)

    def test_yield_stmt_immutability(self):
        """Test that YieldStmt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        yield_stmt = ir.YieldStmt([x], span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            yield_stmt.value = [y]  # type: ignore

    def test_yield_stmt_with_multiple_vars(self):
        """Test YieldStmt with multiple variables."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        # Test with single Var
        yield_stmt1 = ir.YieldStmt([x], span)
        assert len(yield_stmt1.value) == 1
        assert yield_stmt1.value[0].name == "x"

        # Test with multiple Vars
        yield_stmt2 = ir.YieldStmt([x, y], span)
        assert len(yield_stmt2.value) == 2
        assert yield_stmt2.value[0].name == "x"
        assert yield_stmt2.value[1].name == "y"

        # Test with three Vars
        yield_stmt3 = ir.YieldStmt([x, y, z], span)
        assert len(yield_stmt3.value) == 3
        assert yield_stmt3.value[0].name == "x"
        assert yield_stmt3.value[1].name == "y"
        assert yield_stmt3.value[2].name == "z"


class TestForStmt:
    """Test ForStmt class."""

    def test_for_stmt_creation(self):
        """Test creating a ForStmt instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)
        for_stmt = ir.ForStmt(i, start, stop, step, [assign], span)

        assert for_stmt is not None
        assert for_stmt.span.filename == "test.py"
        assert cast(ir.Var, for_stmt.loop_var).name == "i"
        assert isinstance(for_stmt.start, ir.ConstInt)
        assert isinstance(for_stmt.stop, ir.ConstInt)
        assert isinstance(for_stmt.step, ir.ConstInt)
        assert len(for_stmt.body) == 1

    def test_for_stmt_has_attributes(self):
        """Test that ForStmt has loop_var, start, stop, step, and body attributes."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        loop_var = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(2, dtype, span)
        assign1 = ir.AssignStmt(loop_var, start, span)
        assign2 = ir.AssignStmt(loop_var, stop, span)
        for_stmt = ir.ForStmt(loop_var, start, stop, step, [assign1, assign2], span)

        assert for_stmt.loop_var is not None
        assert for_stmt.start is not None
        assert for_stmt.stop is not None
        assert for_stmt.step is not None
        assert len(for_stmt.body) == 2
        assert cast(ir.Var, for_stmt.loop_var).name == "i"
        assert cast(ir.ConstInt, for_stmt.start).value == 0
        assert cast(ir.ConstInt, for_stmt.stop).value == 10
        assert cast(ir.ConstInt, for_stmt.step).value == 2

    def test_for_stmt_is_stmt(self):
        """Test that ForStmt is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)
        for_stmt = ir.ForStmt(i, start, stop, step, [assign], span)

        assert isinstance(for_stmt, ir.Stmt)
        assert isinstance(for_stmt, ir.IRNode)

    def test_for_stmt_immutability(self):
        """Test that ForStmt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        j = ir.Var("j", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)
        for_stmt = ir.ForStmt(i, start, stop, step, [assign], span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            for_stmt.loop_var = j  # type: ignore
        with pytest.raises(AttributeError):
            for_stmt.start = ir.ConstInt(1, dtype, span)  # type: ignore
        with pytest.raises(AttributeError):
            for_stmt.stop = ir.ConstInt(20, dtype, span)  # type: ignore
        with pytest.raises(AttributeError):
            for_stmt.step = ir.ConstInt(2, dtype, span)  # type: ignore
        with pytest.raises(AttributeError):
            for_stmt.body = []  # type: ignore

    def test_for_stmt_with_empty_body(self):
        """Test ForStmt with empty body."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], span)

        assert len(for_stmt.body) == 0

    def test_for_stmt_with_different_expression_types(self):
        """Test ForStmt with different expression types for start, stop, step."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(i, x, span)

        # Test with Var expressions
        for_stmt1 = ir.ForStmt(i, x, y, z, [assign], span)
        assert isinstance(for_stmt1.start, ir.Var)
        assert isinstance(for_stmt1.stop, ir.Var)
        assert isinstance(for_stmt1.step, ir.Var)

        # Test with ConstInt expressions
        start_const = ir.ConstInt(0, dtype, span)
        stop_const = ir.ConstInt(10, dtype, span)
        step_const = ir.ConstInt(1, dtype, span)
        for_stmt2 = ir.ForStmt(i, start_const, stop_const, step_const, [assign], span)
        assert isinstance(for_stmt2.start, ir.ConstInt)
        assert isinstance(for_stmt2.stop, ir.ConstInt)
        assert isinstance(for_stmt2.step, ir.ConstInt)

        # Test with binary expression
        add_expr = ir.Add(x, y, dtype, span)
        for_stmt3 = ir.ForStmt(i, start_const, add_expr, step_const, [assign], span)
        assert isinstance(for_stmt3.stop, ir.Add)

    def test_for_stmt_with_multiple_statements(self):
        """Test ForStmt with multiple statements in body."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign1 = ir.AssignStmt(i, x, span)
        assign2 = ir.AssignStmt(x, y, span)
        assign3 = ir.AssignStmt(y, i, span)
        for_stmt = ir.ForStmt(i, start, stop, step, [assign1, assign2, assign3], span)

        assert len(for_stmt.body) == 3
        assert isinstance(for_stmt.body[0], ir.AssignStmt)
        assert isinstance(for_stmt.body[1], ir.AssignStmt)
        assert isinstance(for_stmt.body[2], ir.AssignStmt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
