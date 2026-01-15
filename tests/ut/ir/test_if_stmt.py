# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IfStmt class."""

import pytest
from pypto import DataType, ir


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
        if_stmt = ir.IfStmt(condition, assign, [], span)

        assert if_stmt is not None
        assert if_stmt.span.filename == "test.py"
        assert isinstance(if_stmt.condition, ir.Eq)
        assert isinstance(if_stmt.then_body, ir.AssignStmt)
        assert if_stmt.else_body is None

    def test_if_stmt_has_attributes(self):
        """Test that IfStmt has condition, then_body, and else_body attributes."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        condition = ir.Lt(a, b, dtype, span)
        assign1 = ir.AssignStmt(a, b, span)
        assign2 = ir.AssignStmt(b, a, span)
        if_stmt = ir.IfStmt(condition, assign1, assign2, [], span)

        assert if_stmt.condition is not None
        assert isinstance(if_stmt.then_body, ir.AssignStmt)
        assert if_stmt.else_body is not None
        assert isinstance(if_stmt.else_body, ir.AssignStmt)
        assert len(if_stmt.return_vars) == 0

    def test_if_stmt_is_stmt(self):
        """Test that IfStmt is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)
        if_stmt = ir.IfStmt(condition, assign, [], span)

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
        if_stmt = ir.IfStmt(condition, assign, [], span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            if_stmt.condition = ir.Eq(y, x, dtype, span)  # type: ignore
        with pytest.raises(AttributeError):
            if_stmt.then_body = []  # type: ignore
        with pytest.raises(AttributeError):
            if_stmt.else_body = []  # type: ignore
        with pytest.raises(AttributeError):
            if_stmt.return_vars = []  # type: ignore

    def test_if_stmt_with_empty_else_body(self):
        """Test IfStmt with empty else_body."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)
        if_stmt = ir.IfStmt(condition, assign, [], span)

        assert if_stmt.else_body is None

    def test_if_stmt_with_different_condition_types(self):
        """Test IfStmt with different condition expression types."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)

        # Test with Eq condition
        condition1 = ir.Eq(x, y, dtype, span)
        if_stmt1 = ir.IfStmt(condition1, assign, [], span)
        assert isinstance(if_stmt1.condition, ir.Eq)

        # Test with Lt condition
        condition2 = ir.Lt(x, y, dtype, span)
        if_stmt2 = ir.IfStmt(condition2, assign, [], span)
        assert isinstance(if_stmt2.condition, ir.Lt)

        # Test with And condition
        condition3 = ir.And(x, y, dtype, span)
        if_stmt3 = ir.IfStmt(condition3, assign, [], span)
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
        then_seq = ir.SeqStmts([assign1, assign2], span)
        if_stmt = ir.IfStmt(condition, then_seq, assign3, [], span)

        assert isinstance(if_stmt.then_body, ir.SeqStmts)
        assert len(if_stmt.then_body.stmts) == 2
        assert if_stmt.else_body is not None
        assert isinstance(if_stmt.else_body, ir.AssignStmt)

    def test_if_stmt_with_return_vars(self):
        """Test IfStmt with return_vars."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        c = ir.Var("c", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        # IfStmt with empty return_vars
        if_stmt1 = ir.IfStmt(condition, assign, [], span)
        assert len(if_stmt1.return_vars) == 0

        # IfStmt with single return variable
        if_stmt2 = ir.IfStmt(condition, assign, [a], span)
        assert len(if_stmt2.return_vars) == 1
        assert if_stmt2.return_vars[0].name == "a"

        # IfStmt with multiple return variables
        if_stmt3 = ir.IfStmt(condition, assign, [a, b, c], span)
        assert len(if_stmt3.return_vars) == 3
        assert if_stmt3.return_vars[0].name == "a"
        assert if_stmt3.return_vars[1].name == "b"
        assert if_stmt3.return_vars[2].name == "c"


class TestIfStmtHash:
    """Tests for IfStmt hash function."""

    def test_if_stmt_same_structure_hash(self):
        """Test IfStmt nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x1, y1, dtype, span)
        condition2 = ir.Eq(x2, y2, dtype, span)
        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        if_stmt1 = ir.IfStmt(condition1, assign1, [], span)
        if_stmt2 = ir.IfStmt(condition2, assign2, [], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        # Different variable pointers result in different hashes without auto_mapping
        assert hash1 != hash2

    def test_if_stmt_different_condition_hash(self):
        """Test IfStmt nodes with different condition hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x, y, dtype, span)
        condition2 = ir.Lt(x, z, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt1 = ir.IfStmt(condition1, assign, [], span)
        if_stmt2 = ir.IfStmt(condition2, assign, [], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        assert hash1 != hash2

    def test_if_stmt_different_then_body_hash(self):
        """Test IfStmt nodes with different then_body hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)

        if_stmt1 = ir.IfStmt(condition, assign1, [], span)
        if_stmt2 = ir.IfStmt(condition, assign2, [], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        assert hash1 != hash2

    def test_if_stmt_different_else_body_hash(self):
        """Test IfStmt nodes with different else_body hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)

        if_stmt1 = ir.IfStmt(condition, assign1, assign1, [], span)
        if_stmt2 = ir.IfStmt(condition, assign1, assign2, [], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        assert hash1 != hash2

    def test_if_stmt_different_return_vars_hash(self):
        """Test IfStmt nodes with different return_vars hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt1 = ir.IfStmt(condition, assign, [a], span)
        if_stmt2 = ir.IfStmt(condition, assign, [b], span)
        if_stmt3 = ir.IfStmt(condition, assign, [a, b], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        hash3 = ir.structural_hash(if_stmt3)
        assert hash1 == hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_if_stmt_empty_vs_non_empty_return_vars_hash(self):
        """Test IfStmt nodes with empty and non-empty return_vars hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        a = ir.Var("a", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt1 = ir.IfStmt(condition, assign, [], span)
        if_stmt2 = ir.IfStmt(condition, assign, [a], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        assert hash1 != hash2

    def test_if_stmt_with_nullopt_else_body_hash(self):
        """Test IfStmt nodes with nullopt else_body hash correctly."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x1, y1, dtype, span)
        condition2 = ir.Eq(x2, y2, dtype, span)
        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        # Both have nullopt else_body
        if_stmt1 = ir.IfStmt(condition1, assign1, None, [], span)
        if_stmt2 = ir.IfStmt(condition2, assign2, None, [], span)

        hash1 = ir.structural_hash(if_stmt1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(if_stmt2, enable_auto_mapping=True)
        assert hash1 == hash2

        # One with nullopt else_body, one with else_body
        assign_else = ir.AssignStmt(y1, x1, span)
        if_stmt3 = ir.IfStmt(condition1, assign1, assign_else, [], span)

        hash3 = ir.structural_hash(if_stmt3, enable_auto_mapping=True)
        assert hash1 != hash3


class TestIfStmtEquality:
    """Tests for IfStmt structural equality function."""

    def test_if_stmt_structural_equal(self):
        """Test structural equality of IfStmt nodes."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x1, y1, dtype, span)
        condition2 = ir.Eq(x2, y2, dtype, span)
        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        if_stmt1 = ir.IfStmt(condition1, assign1, [], span)
        if_stmt2 = ir.IfStmt(condition2, assign2, [], span)

        # Different variable pointers, so not equal without auto_mapping
        assert not ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=False)
        # With auto_mapping, they should be equal
        assert ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=True)

    def test_if_stmt_different_condition_not_equal(self):
        """Test IfStmt nodes with different condition are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x, y, dtype, span)
        condition2 = ir.Lt(x, z, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt1 = ir.IfStmt(condition1, assign, [], span)
        if_stmt2 = ir.IfStmt(condition2, assign, [], span)

        assert not ir.structural_equal(if_stmt1, if_stmt2)

    def test_if_stmt_different_then_body_not_equal(self):
        """Test IfStmt nodes with different then_body are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)

        if_stmt1 = ir.IfStmt(condition, assign1, [], span)
        if_stmt2 = ir.IfStmt(condition, assign2, [], span)

        assert not ir.structural_equal(if_stmt1, if_stmt2)

    def test_if_stmt_different_else_body_not_equal(self):
        """Test IfStmt nodes with different else_body are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)

        if_stmt1 = ir.IfStmt(condition, assign1, assign1, [], span)
        if_stmt2 = ir.IfStmt(condition, assign1, assign2, [], span)

        assert not ir.structural_equal(if_stmt1, if_stmt2)

    def test_if_stmt_different_from_base_stmt_not_equal(self):
        """Test IfStmt and base Stmt nodes are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt = ir.IfStmt(condition, assign, [], span)
        stmt = ir.Stmt(span)

        assert not ir.structural_equal(if_stmt, stmt)

    def test_if_stmt_different_return_vars_not_equal(self):
        """Test IfStmt nodes with different return_vars are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt1 = ir.IfStmt(condition, assign, [a], span)
        if_stmt2 = ir.IfStmt(condition, assign, [b], span)
        if_stmt3 = ir.IfStmt(condition, assign, [a, b], span)

        assert ir.structural_equal(if_stmt1, if_stmt2)
        assert not ir.structural_equal(if_stmt1, if_stmt3)
        assert not ir.structural_equal(if_stmt2, if_stmt3)

    def test_if_stmt_empty_vs_non_empty_return_vars_not_equal(self):
        """Test IfStmt nodes with empty and non-empty return_vars are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        a = ir.Var("a", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt1 = ir.IfStmt(condition, assign, [], span)
        if_stmt2 = ir.IfStmt(condition, assign, [a], span)

        assert not ir.structural_equal(if_stmt1, if_stmt2)

    def test_if_stmt_with_nullopt_else_body_equal(self):
        """Test IfStmt nodes with nullopt else_body are equal when structurally same."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x1, y1, dtype, span)
        condition2 = ir.Eq(x2, y2, dtype, span)
        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        # Both have nullopt else_body
        if_stmt1 = ir.IfStmt(condition1, assign1, None, [], span)
        if_stmt2 = ir.IfStmt(condition2, assign2, None, [], span)

        # With auto_mapping, they should be equal
        assert ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=True)
        # Without auto_mapping, they should not be equal (different variable pointers)
        assert not ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=False)


class TestIfStmtAutoMapping:
    """Tests for auto mapping feature with IfStmt."""

    def test_auto_mapping_with_if_stmt(self):
        """Test auto mapping with IfStmt."""
        # Build: if x == y then x = y else y = x
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition1 = ir.Eq(x1, y1, DataType.INT64, ir.Span.unknown())
        assign1_then = ir.AssignStmt(x1, y1, ir.Span.unknown())
        assign1_else = ir.AssignStmt(y1, x1, ir.Span.unknown())
        if_stmt1 = ir.IfStmt(condition1, assign1_then, assign1_else, [], ir.Span.unknown())

        # Build: if a == b then a = b else b = a
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition2 = ir.Eq(a, b, DataType.INT64, ir.Span.unknown())
        assign2_then = ir.AssignStmt(a, b, ir.Span.unknown())
        assign2_else = ir.AssignStmt(b, a, ir.Span.unknown())
        if_stmt2 = ir.IfStmt(condition2, assign2_then, assign2_else, [], ir.Span.unknown())

        assert ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=True)
        assert not ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=False)

        hash_with_auto1 = ir.structural_hash(if_stmt1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(if_stmt2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        hash_without_auto1 = ir.structural_hash(if_stmt1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(if_stmt2, enable_auto_mapping=False)
        assert hash_without_auto1 != hash_without_auto2

    def test_auto_mapping_if_stmt_different_structure(self):
        """Test auto mapping with IfStmt where structure differs."""
        # Build: if x == y then x = y
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition1 = ir.Eq(x1, y1, DataType.INT64, ir.Span.unknown())
        assign1 = ir.AssignStmt(x1, y1, ir.Span.unknown())
        if_stmt1 = ir.IfStmt(condition1, assign1, [], ir.Span.unknown())

        # Build: if a == b then a = b else b = a
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition2 = ir.Eq(a, b, DataType.INT64, ir.Span.unknown())
        assign2_then = ir.AssignStmt(a, b, ir.Span.unknown())
        assign2_else = ir.AssignStmt(b, a, ir.Span.unknown())
        if_stmt2 = ir.IfStmt(condition2, assign2_then, assign2_else, [], ir.Span.unknown())

        # Different structure (one has else, one doesn't)
        assert not ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=True)

    def test_auto_mapping_if_stmt_with_return_vars(self):
        """Test auto mapping with IfStmt that has return_vars."""
        # Build: if x == y then x = y return r1, r2
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        r1 = ir.Var("r1", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        r2 = ir.Var("r2", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition1 = ir.Eq(x1, y1, DataType.INT64, ir.Span.unknown())
        assign1 = ir.AssignStmt(x1, y1, ir.Span.unknown())
        if_stmt1 = ir.IfStmt(condition1, assign1, [r1, r2], ir.Span.unknown())

        # Build: if a == b then a = b return s1, s2
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        s1 = ir.Var("s1", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        s2 = ir.Var("s2", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition2 = ir.Eq(a, b, DataType.INT64, ir.Span.unknown())
        assign2 = ir.AssignStmt(a, b, ir.Span.unknown())
        if_stmt2 = ir.IfStmt(condition2, assign2, [s1, s2], ir.Span.unknown())

        # With auto_mapping, they should be equal (return_vars are DefField)
        assert ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=True)
        # Without auto_mapping, they should not be equal
        assert not ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=False)

        hash_with_auto1 = ir.structural_hash(if_stmt1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(if_stmt2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        hash_without_auto1 = ir.structural_hash(if_stmt1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(if_stmt2, enable_auto_mapping=False)
        assert hash_without_auto1 != hash_without_auto2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
