# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ForStmt class."""

from typing import cast

import pytest
from pypto import DataType, ir


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
        for_stmt = ir.ForStmt(i, start, stop, step, [], assign, [], span)

        assert for_stmt is not None
        assert for_stmt.span.filename == "test.py"
        assert cast(ir.Var, for_stmt.loop_var).name == "i"
        assert isinstance(for_stmt.start, ir.ConstInt)
        assert isinstance(for_stmt.stop, ir.ConstInt)
        assert isinstance(for_stmt.step, ir.ConstInt)
        assert isinstance(for_stmt.body, ir.AssignStmt)

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
        body_seq = ir.SeqStmts([assign1, assign2], span)
        for_stmt = ir.ForStmt(loop_var, start, stop, step, [], body_seq, [], span)

        assert for_stmt.loop_var is not None
        assert for_stmt.start is not None
        assert for_stmt.stop is not None
        assert for_stmt.step is not None
        assert isinstance(for_stmt.body, ir.SeqStmts)
        assert len(for_stmt.body.stmts) == 2
        assert cast(ir.Var, for_stmt.loop_var).name == "i"
        assert cast(ir.ConstInt, for_stmt.start).value == 0
        assert cast(ir.ConstInt, for_stmt.stop).value == 10
        assert cast(ir.ConstInt, for_stmt.step).value == 2
        assert len(for_stmt.return_vars) == 0

    def test_for_stmt_is_stmt(self):
        """Test that ForStmt is an instance of Stmt."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], assign, [], span)

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
        for_stmt = ir.ForStmt(i, start, stop, step, [], assign, [], span)

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
        with pytest.raises(AttributeError):
            for_stmt.return_vars = []  # type: ignore

    def test_for_stmt_with_empty_body(self):
        """Test ForStmt with empty body."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        # Empty body should use a SeqStmts with empty list
        empty_body = ir.SeqStmts([], span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], empty_body, [], span)

        assert isinstance(for_stmt.body, ir.SeqStmts)
        assert len(for_stmt.body.stmts) == 0

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
        for_stmt1 = ir.ForStmt(i, x, y, z, [], assign, [], span)
        assert isinstance(for_stmt1.start, ir.Var)
        assert isinstance(for_stmt1.stop, ir.Var)
        assert isinstance(for_stmt1.step, ir.Var)

        # Test with ConstInt expressions
        start_const = ir.ConstInt(0, dtype, span)
        stop_const = ir.ConstInt(10, dtype, span)
        step_const = ir.ConstInt(1, dtype, span)
        for_stmt2 = ir.ForStmt(i, start_const, stop_const, step_const, [], assign, [], span)
        assert isinstance(for_stmt2.start, ir.ConstInt)
        assert isinstance(for_stmt2.stop, ir.ConstInt)
        assert isinstance(for_stmt2.step, ir.ConstInt)

        # Test with binary expression
        add_expr = ir.Add(x, y, dtype, span)
        for_stmt3 = ir.ForStmt(i, start_const, add_expr, step_const, [], assign, [], span)
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
        body_seq = ir.SeqStmts([assign1, assign2, assign3], span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], body_seq, [], span)

        assert isinstance(for_stmt.body, ir.SeqStmts)
        assert len(for_stmt.body.stmts) == 3
        assert isinstance(for_stmt.body.stmts[0], ir.AssignStmt)
        assert isinstance(for_stmt.body.stmts[1], ir.AssignStmt)
        assert isinstance(for_stmt.body.stmts[2], ir.AssignStmt)

    def test_for_stmt_with_return_vars(self):
        """Test ForStmt with return_vars."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        # ForStmt with empty return_vars
        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign, [], span)
        assert len(for_stmt1.return_vars) == 0

        # ForStmt with single return variable
        for_stmt2 = ir.ForStmt(i, start, stop, step, [], assign, [x], span)
        assert len(for_stmt2.return_vars) == 1
        assert for_stmt2.return_vars[0].name == "x"

        # ForStmt with multiple return variables
        for_stmt3 = ir.ForStmt(i, start, stop, step, [], assign, [x, y, z], span)
        assert len(for_stmt3.return_vars) == 3
        assert for_stmt3.return_vars[0].name == "x"
        assert for_stmt3.return_vars[1].name == "y"
        assert for_stmt3.return_vars[2].name == "z"

    def test_for_stmt_with_iter_args(self):
        """Test ForStmt with iter_args."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        # Create IterArg instances
        init_value1 = ir.ConstInt(5, dtype, span)
        iter_arg1 = ir.IterArg("arg1", ir.ScalarType(dtype), init_value1, span)

        init_value2 = x
        iter_arg2 = ir.IterArg("arg2", ir.ScalarType(dtype), init_value2, span)

        # ForStmt with empty iter_args
        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign, [], span)
        assert len(for_stmt1.iter_args) == 0

        # ForStmt with single iter_arg
        for_stmt2 = ir.ForStmt(i, start, stop, step, [iter_arg1], assign, [], span)
        assert len(for_stmt2.iter_args) == 1
        assert for_stmt2.iter_args[0].name == "arg1"

        # ForStmt with multiple iter_args
        for_stmt3 = ir.ForStmt(i, start, stop, step, [iter_arg1, iter_arg2], assign, [], span)
        assert len(for_stmt3.iter_args) == 2
        assert for_stmt3.iter_args[0].name == "arg1"
        assert for_stmt3.iter_args[1].name == "arg2"


class TestForStmtHash:
    """Tests for ForStmt hash function."""

    def test_for_stmt_same_structure_hash(self):
        """Test ForStmt nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i1 = ir.Var("i", ir.ScalarType(dtype), span)
        start1 = ir.ConstInt(0, dtype, span)
        stop1 = ir.ConstInt(10, dtype, span)
        step1 = ir.ConstInt(1, dtype, span)
        assign1 = ir.AssignStmt(i1, start1, span)
        for_stmt1 = ir.ForStmt(i1, start1, stop1, step1, [], assign1, [], span)

        i2 = ir.Var("i", ir.ScalarType(dtype), span)
        start2 = ir.ConstInt(0, dtype, span)
        stop2 = ir.ConstInt(10, dtype, span)
        step2 = ir.ConstInt(1, dtype, span)
        assign2 = ir.AssignStmt(i2, start2, span)
        for_stmt2 = ir.ForStmt(i2, start2, stop2, step2, [], assign2, [], span)

        hash1 = ir.structural_hash(for_stmt1)
        hash2 = ir.structural_hash(for_stmt2)
        assert hash1 == hash2

    def test_for_stmt_different_loop_var_hash(self):
        """Test ForStmt nodes with different loop vars hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        j = ir.Var("j", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign, [], span)
        for_stmt2 = ir.ForStmt(j, start, stop, step, [], assign, [], span)

        hash1 = ir.structural_hash(for_stmt1)
        hash2 = ir.structural_hash(for_stmt2)
        assert hash1 != hash2

    def test_for_stmt_different_range_hash(self):
        """Test ForStmt nodes with different range values hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start1 = ir.ConstInt(0, dtype, span)
        stop1 = ir.ConstInt(10, dtype, span)
        step1 = ir.ConstInt(1, dtype, span)
        start2 = ir.ConstInt(0, dtype, span)
        stop2 = ir.ConstInt(20, dtype, span)
        step2 = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start1, span)

        for_stmt1 = ir.ForStmt(i, start1, stop1, step1, [], assign, [], span)
        for_stmt2 = ir.ForStmt(i, start2, stop2, step2, [], assign, [], span)

        hash1 = ir.structural_hash(for_stmt1)
        hash2 = ir.structural_hash(for_stmt2)
        assert hash1 != hash2

    def test_for_stmt_different_body_hash(self):
        """Test ForStmt nodes with different body statements hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign1 = ir.AssignStmt(i, start, span)
        assign2 = ir.AssignStmt(i, x, span)

        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign1, [], span)
        for_stmt2 = ir.ForStmt(i, start, stop, step, [], assign2, [], span)

        hash1 = ir.structural_hash(for_stmt1)
        hash2 = ir.structural_hash(for_stmt2)
        assert hash1 != hash2

    def test_for_stmt_different_return_vars_hash(self):
        """Test ForStmt nodes with different return_vars hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign, [x], span)
        for_stmt2 = ir.ForStmt(i, start, stop, step, [], assign, [y], span)
        for_stmt3 = ir.ForStmt(i, start, stop, step, [], assign, [x, y], span)

        hash1 = ir.structural_hash(for_stmt1)
        hash2 = ir.structural_hash(for_stmt2)
        hash3 = ir.structural_hash(for_stmt3)
        assert hash1 == hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_for_stmt_empty_vs_non_empty_return_vars_hash(self):
        """Test ForStmt nodes with empty and non-empty return_vars hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign, [], span)
        for_stmt2 = ir.ForStmt(i, start, stop, step, [], assign, [x], span)

        hash1 = ir.structural_hash(for_stmt1)
        hash2 = ir.structural_hash(for_stmt2)
        assert hash1 != hash2


class TestForStmtEquality:
    """Tests for ForStmt structural equality function."""

    def test_for_stmt_structural_equal(self):
        """Test structural equality of ForStmt nodes."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i1 = ir.Var("i", ir.ScalarType(dtype), span)
        start1 = ir.ConstInt(0, dtype, span)
        stop1 = ir.ConstInt(10, dtype, span)
        step1 = ir.ConstInt(1, dtype, span)
        assign1 = ir.AssignStmt(i1, start1, span)
        for_stmt1 = ir.ForStmt(i1, start1, stop1, step1, [], assign1, [], span)

        i2 = ir.Var("i", ir.ScalarType(dtype), span)
        start2 = ir.ConstInt(0, dtype, span)
        stop2 = ir.ConstInt(10, dtype, span)
        step2 = ir.ConstInt(1, dtype, span)
        assign2 = ir.AssignStmt(i2, start2, span)
        for_stmt2 = ir.ForStmt(i2, start2, stop2, step2, [], assign2, [], span)

        ir.assert_structural_equal(for_stmt1, for_stmt2)

    def test_for_stmt_different_loop_var_equal(self):
        """Test ForStmt nodes with different loop vars are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        j = ir.Var("j", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign, [], span)
        for_stmt2 = ir.ForStmt(j, start, stop, step, [], assign, [], span)

        ir.assert_structural_equal(for_stmt1, for_stmt2)

    def test_for_stmt_different_range_not_equal(self):
        """Test ForStmt nodes with different range values are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start1 = ir.ConstInt(0, dtype, span)
        stop1 = ir.ConstInt(10, dtype, span)
        step1 = ir.ConstInt(1, dtype, span)
        start2 = ir.ConstInt(0, dtype, span)
        stop2 = ir.ConstInt(20, dtype, span)
        step2 = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start1, span)

        for_stmt1 = ir.ForStmt(i, start1, stop1, step1, [], assign, [], span)
        for_stmt2 = ir.ForStmt(i, start2, stop2, step2, [], assign, [], span)

        assert not ir.structural_equal(for_stmt1, for_stmt2)

    def test_for_stmt_different_body_not_equal(self):
        """Test ForStmt nodes with different body statements are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign1 = ir.AssignStmt(i, start, span)
        assign2 = ir.AssignStmt(i, x, span)

        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign1, [], span)
        for_stmt2 = ir.ForStmt(i, start, stop, step, [], assign2, [], span)

        assert not ir.structural_equal(for_stmt1, for_stmt2)

    def test_for_stmt_empty_vs_non_empty_body_not_equal(self):
        """Test ForStmt nodes with empty and non-empty body lists are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        empty_body = ir.SeqStmts([], span)
        for_stmt1 = ir.ForStmt(i, start, stop, step, [], empty_body, [], span)
        for_stmt2 = ir.ForStmt(i, start, stop, step, [], assign, [], span)

        assert not ir.structural_equal(for_stmt1, for_stmt2)

    def test_for_stmt_different_from_base_stmt_not_equal(self):
        """Test ForStmt and different Stmt type are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], assign, [], span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        other_stmt = ir.YieldStmt([x], span)

        assert not ir.structural_equal(for_stmt, other_stmt)

    def test_for_stmt_different_return_vars_not_equal(self):
        """Test ForStmt nodes with different return_vars are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign, [x], span)
        for_stmt2 = ir.ForStmt(i, start, stop, step, [], assign, [y], span)
        for_stmt3 = ir.ForStmt(i, start, stop, step, [], assign, [x, y], span)

        ir.assert_structural_equal(for_stmt1, for_stmt2)
        assert not ir.structural_equal(for_stmt1, for_stmt3)
        assert not ir.structural_equal(for_stmt2, for_stmt3)

    def test_for_stmt_empty_vs_non_empty_return_vars_not_equal(self):
        """Test ForStmt nodes with empty and non-empty return_vars are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        assign = ir.AssignStmt(i, start, span)

        for_stmt1 = ir.ForStmt(i, start, stop, step, [], assign, [], span)
        for_stmt2 = ir.ForStmt(i, start, stop, step, [], assign, [x], span)

        assert not ir.structural_equal(for_stmt1, for_stmt2)


class TestForStmtAutoMapping:
    """Tests for auto mapping feature with ForStmt."""

    def test_auto_mapping_with_for_stmt(self):
        """Test auto mapping with ForStmt."""
        # Build: for i in range(0, 10, 1): i = 0
        i1 = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        start1 = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop1 = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        assign1 = ir.AssignStmt(i1, start1, ir.Span.unknown())
        for_stmt1 = ir.ForStmt(i1, start1, stop1, step1, [], assign1, [], ir.Span.unknown())

        # Build: for j in range(0, 10, 1): j = 0
        j = ir.Var("j", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        start2 = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop2 = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step2 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        assign2 = ir.AssignStmt(j, start2, ir.Span.unknown())
        for_stmt2 = ir.ForStmt(j, start2, stop2, step2, [], assign2, [], ir.Span.unknown())

        ir.assert_structural_equal(for_stmt1, for_stmt2, enable_auto_mapping=True)
        ir.assert_structural_equal(for_stmt1, for_stmt2, enable_auto_mapping=False)

        hash_with_auto1 = ir.structural_hash(for_stmt1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(for_stmt2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        hash_without_auto1 = ir.structural_hash(for_stmt1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(for_stmt2, enable_auto_mapping=False)
        assert hash_without_auto1 == hash_without_auto2

    def test_auto_mapping_for_stmt_different_structure(self):
        """Test auto mapping with ForStmt where structures differ."""
        # Build: for i in range(0, 10, 1): i = 0
        i1 = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        start1 = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop1 = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        assign1 = ir.AssignStmt(i1, start1, ir.Span.unknown())
        for_stmt1 = ir.ForStmt(i1, start1, stop1, step1, [], assign1, [], ir.Span.unknown())

        # Build: for j in range(0, 20, 1): j = 0
        j = ir.Var("j", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        start2 = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop2 = ir.ConstInt(20, DataType.INT64, ir.Span.unknown())
        step2 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        assign2 = ir.AssignStmt(j, start2, ir.Span.unknown())
        for_stmt2 = ir.ForStmt(j, start2, stop2, step2, [], assign2, [], ir.Span.unknown())

        # Different stop values should not be equal
        assert not ir.structural_equal(for_stmt1, for_stmt2, enable_auto_mapping=True)

    def test_auto_mapping_for_stmt_with_return_vars(self):
        """Test auto mapping with ForStmt that has return_vars."""
        # Build: for i in range(0, 10, 1): i = 0 return i, x
        i1 = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        start1 = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop1 = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        assign1 = ir.AssignStmt(i1, start1, ir.Span.unknown())
        for_stmt1 = ir.ForStmt(i1, start1, stop1, step1, [], assign1, [i1, x1], ir.Span.unknown())

        # Build: for j in range(0, 10, 1): j = 0 return j, y
        j = ir.Var("j", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        start2 = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop2 = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step2 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        assign2 = ir.AssignStmt(j, start2, ir.Span.unknown())
        for_stmt2 = ir.ForStmt(j, start2, stop2, step2, [], assign2, [j, y], ir.Span.unknown())

        ir.assert_structural_equal(for_stmt1, for_stmt2, enable_auto_mapping=True)
        ir.assert_structural_equal(for_stmt1, for_stmt2, enable_auto_mapping=False)

        hash_with_auto1 = ir.structural_hash(for_stmt1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(for_stmt2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        hash_without_auto1 = ir.structural_hash(for_stmt1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(for_stmt2, enable_auto_mapping=False)
        assert hash_without_auto1 == hash_without_auto2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
