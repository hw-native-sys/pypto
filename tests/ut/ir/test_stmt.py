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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
