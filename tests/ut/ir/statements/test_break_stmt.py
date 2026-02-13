# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
from pypto import DataType, ir


def test_break_stmt_creation():
    """Test creating a BreakStmt."""
    span = ir.Span("test.py", 1, 1)
    stmt = ir.BreakStmt(span)

    assert isinstance(stmt, ir.BreakStmt)
    assert isinstance(stmt, ir.Stmt)
    assert stmt.span.filename == "test.py"


def test_break_stmt_python_print():
    """Test printing a BreakStmt as Python code."""
    span = ir.Span("test.py", 1, 1)
    stmt = ir.BreakStmt(span)

    code = ir.python_print(stmt)
    assert code == "break"


def test_break_stmt_structural_hash():
    """Test that two BreakStmts produce the same hash."""
    span1 = ir.Span("a.py", 1, 1)
    span2 = ir.Span("b.py", 2, 2)
    stmt1 = ir.BreakStmt(span1)
    stmt2 = ir.BreakStmt(span2)

    assert ir.structural_hash(stmt1) == ir.structural_hash(stmt2)


def test_break_stmt_structural_equal():
    """Test structural equality of BreakStmts."""
    span1 = ir.Span("a.py", 1, 1)
    span2 = ir.Span("b.py", 2, 2)
    stmt1 = ir.BreakStmt(span1)
    stmt2 = ir.BreakStmt(span2)

    ir.assert_structural_equal(stmt1, stmt2)


def test_break_stmt_not_equal_to_continue():
    """Test that BreakStmt is not structurally equal to ContinueStmt."""
    span = ir.Span("test.py", 1, 1)
    break_stmt = ir.BreakStmt(span)
    continue_stmt = ir.ContinueStmt(span)

    assert not ir.structural_equal(break_stmt, continue_stmt)


def test_break_stmt_serialization():
    """Test serializing and deserializing a BreakStmt."""
    span = ir.Span("test.py", 1, 1)
    stmt = ir.BreakStmt(span)

    data = ir.serialize(stmt)
    restored = ir.deserialize(data)

    assert isinstance(restored, ir.BreakStmt)
    ir.assert_structural_equal(stmt, restored)


def test_break_stmt_not_equal_to_other_stmt_types():
    """Test that BreakStmt is not structurally equal to other statement types."""
    span = ir.Span("test.py", 1, 1)
    dtype = DataType.INT64
    break_stmt = ir.BreakStmt(span)
    x = ir.Var("x", ir.ScalarType(dtype), span)
    assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
    yield_stmt = ir.YieldStmt([x], span)
    return_stmt = ir.ReturnStmt([x], span)

    assert not ir.structural_equal(break_stmt, assign)
    assert not ir.structural_equal(break_stmt, yield_stmt)
    assert not ir.structural_equal(break_stmt, return_stmt)


def test_break_stmt_immutability():
    """Test that BreakStmt span attribute is immutable."""
    span = ir.Span("test.py", 1, 1)
    stmt = ir.BreakStmt(span)

    with pytest.raises(AttributeError):
        stmt.span = ir.Span("other.py", 2, 2)  # type: ignore


class TestBreakInForLoop:
    """Test BreakStmt composed inside ForStmt bodies."""

    def _make_for_with_body(self, body: ir.Stmt) -> ir.ForStmt:
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        return ir.ForStmt(i, start, stop, step, [], body, [], span)

    def test_for_loop_with_break_body(self):
        """Test ForStmt whose body is a single BreakStmt."""
        span = ir.Span("test.py", 1, 1)
        break_stmt = ir.BreakStmt(span)
        for_stmt = self._make_for_with_body(break_stmt)

        assert isinstance(for_stmt.body, ir.BreakStmt)

    def test_for_loop_with_break_in_seq(self):
        """Test ForStmt with break inside a SeqStmts body."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(42, dtype, span), span)
        break_stmt = ir.BreakStmt(span)
        seq = ir.SeqStmts([assign, break_stmt], span)
        for_stmt = self._make_for_with_body(seq)

        assert isinstance(for_stmt.body, ir.SeqStmts)
        assert len(for_stmt.body.stmts) == 2
        assert isinstance(for_stmt.body.stmts[0], ir.AssignStmt)
        assert isinstance(for_stmt.body.stmts[1], ir.BreakStmt)

    def test_for_loop_with_conditional_break(self):
        """Test ForStmt with break inside an IfStmt (if cond: break).

        Represents: for i in range(0, 10, 1): if i > 5: break
        """
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        cond = ir.Gt(i, ir.ConstInt(5, dtype, span), dtype, span)
        break_stmt = ir.BreakStmt(span)
        if_stmt = ir.IfStmt(cond, break_stmt, None, [], span)
        for_stmt = self._make_for_with_body(if_stmt)

        assert isinstance(for_stmt.body, ir.IfStmt)
        assert isinstance(for_stmt.body.then_body, ir.BreakStmt)

    def test_for_loop_with_break_python_print(self):
        """Test python_print of a ForStmt containing a conditional break."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        cond = ir.Gt(i, ir.ConstInt(5, dtype, span), dtype, span)
        break_stmt = ir.BreakStmt(span)
        if_stmt = ir.IfStmt(cond, break_stmt, None, [], span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], if_stmt, [], span)

        code = ir.python_print(for_stmt)
        assert "for" in code
        assert "break" in code
        assert "if" in code

    def test_for_loop_with_break_structural_equal(self):
        """Test structural equality of two ForStmts with identical conditional break."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64

        def make_for_break():
            i = ir.Var("i", ir.ScalarType(dtype), span)
            start = ir.ConstInt(0, dtype, span)
            stop = ir.ConstInt(10, dtype, span)
            step = ir.ConstInt(1, dtype, span)
            cond = ir.Gt(i, ir.ConstInt(5, dtype, span), dtype, span)
            brk = ir.BreakStmt(span)
            if_stmt = ir.IfStmt(cond, brk, None, [], span)
            return ir.ForStmt(i, start, stop, step, [], if_stmt, [], span)

        ir.assert_structural_equal(make_for_break(), make_for_break())

    def test_for_loop_with_break_serialization(self):
        """Test serialization roundtrip of a ForStmt containing a break."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        break_stmt = ir.BreakStmt(span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], break_stmt, [], span)

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        assert isinstance(restored.body, ir.BreakStmt)
        ir.assert_structural_equal(for_stmt, restored)

    def test_for_loop_break_vs_continue_not_equal(self):
        """Test that for-with-break is not equal to for-with-continue."""
        span = ir.Span("test.py", 1, 1)
        for_break = self._make_for_with_body(ir.BreakStmt(span))
        for_continue = self._make_for_with_body(ir.ContinueStmt(span))

        assert not ir.structural_equal(for_break, for_continue)


class TestBreakInWhileLoop:
    """Test BreakStmt composed inside WhileStmt bodies."""

    def test_while_loop_with_break_body(self):
        """Test WhileStmt whose body is a single BreakStmt."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        cond = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        break_stmt = ir.BreakStmt(span)
        while_stmt = ir.WhileStmt(cond, [], break_stmt, [], span)

        assert isinstance(while_stmt.body, ir.BreakStmt)

    def test_while_loop_with_conditional_break(self):
        """Test WhileStmt with conditional break: while cond: if x > 5: break."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        loop_cond = ir.Lt(x, ir.ConstInt(100, dtype, span), dtype, span)
        break_cond = ir.Gt(x, ir.ConstInt(5, dtype, span), dtype, span)
        break_stmt = ir.BreakStmt(span)
        if_stmt = ir.IfStmt(break_cond, break_stmt, None, [], span)
        while_stmt = ir.WhileStmt(loop_cond, [], if_stmt, [], span)

        assert isinstance(while_stmt.body, ir.IfStmt)
        assert isinstance(while_stmt.body.then_body, ir.BreakStmt)

    def test_while_loop_with_break_serialization(self):
        """Test serialization roundtrip of a WhileStmt containing a break."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        cond = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        break_stmt = ir.BreakStmt(span)
        while_stmt = ir.WhileStmt(cond, [], break_stmt, [], span)

        data = ir.serialize(while_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.WhileStmt)
        assert isinstance(restored.body, ir.BreakStmt)
        # WhileStmt condition has free vars; use auto-mapping after deserialization
        ir.assert_structural_equal(while_stmt, restored, enable_auto_mapping=True)

    def test_while_loop_with_break_structural_hash(self):
        """Test structural hash consistency for WhileStmt containing break."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        # Share Var instance so pointer-based hash is stable
        x = ir.Var("x", ir.ScalarType(dtype), span)
        cond = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        w1 = ir.WhileStmt(cond, [], ir.BreakStmt(span), [], span)
        w2 = ir.WhileStmt(cond, [], ir.BreakStmt(span), [], span)

        assert ir.structural_hash(w1) == ir.structural_hash(w2)


class TestBreakInNestedStructures:
    """Test BreakStmt in deeply nested IR structures."""

    def test_break_in_if_else(self):
        """Test break in then branch and continue in else branch of IfStmt."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        cond = ir.Gt(x, ir.ConstInt(5, dtype, span), dtype, span)
        break_stmt = ir.BreakStmt(span)
        continue_stmt = ir.ContinueStmt(span)
        if_stmt = ir.IfStmt(cond, break_stmt, continue_stmt, [], span)

        assert isinstance(if_stmt.then_body, ir.BreakStmt)
        assert isinstance(if_stmt.else_body, ir.ContinueStmt)

        code = ir.python_print(if_stmt)
        assert "break" in code
        assert "continue" in code

    def test_break_in_seq_with_multiple_stmts(self):
        """Test break preceded by assignments in a SeqStmts."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, ir.ConstInt(1, dtype, span), span)
        assign2 = ir.AssignStmt(y, ir.Add(x, ir.ConstInt(2, dtype, span), dtype, span), span)
        break_stmt = ir.BreakStmt(span)
        seq = ir.SeqStmts([assign1, assign2, break_stmt], span)

        assert len(seq.stmts) == 3
        assert isinstance(seq.stmts[2], ir.BreakStmt)

        # Serialization roundtrip
        data = ir.serialize(seq)
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.SeqStmts)
        assert isinstance(restored.stmts[2], ir.BreakStmt)
        ir.assert_structural_equal(seq, restored)

    def test_nested_for_with_break_in_inner_loop(self):
        """Test nested for loops where inner loop has a break.

        Represents:
            for i in range(0, 10, 1):
                for j in range(0, 5, 1):
                    break
        """
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        j = ir.Var("j", ir.ScalarType(dtype), span)

        break_stmt = ir.BreakStmt(span)
        inner_for = ir.ForStmt(
            j,
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(5, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            break_stmt,
            [],
            span,
        )
        outer_for = ir.ForStmt(
            i,
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(10, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            inner_for,
            [],
            span,
        )

        assert isinstance(outer_for.body, ir.ForStmt)
        assert isinstance(outer_for.body.body, ir.BreakStmt)

        # Structural equality
        inner_for2 = ir.ForStmt(
            ir.Var("j", ir.ScalarType(dtype), span),
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(5, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            ir.BreakStmt(span),
            [],
            span,
        )
        outer_for2 = ir.ForStmt(
            ir.Var("i", ir.ScalarType(dtype), span),
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(10, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            inner_for2,
            [],
            span,
        )
        ir.assert_structural_equal(outer_for, outer_for2)

    def test_nested_for_with_break_serialization(self):
        """Test serialization of nested for loop with break."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        j = ir.Var("j", ir.ScalarType(dtype), span)
        i = ir.Var("i", ir.ScalarType(dtype), span)

        inner_for = ir.ForStmt(
            j,
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(5, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            ir.BreakStmt(span),
            [],
            span,
        )
        outer_for = ir.ForStmt(
            i,
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(10, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            inner_for,
            [],
            span,
        )

        data = ir.serialize(outer_for)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        assert isinstance(restored.body, ir.ForStmt)
        assert isinstance(restored.body.body, ir.BreakStmt)
        ir.assert_structural_equal(outer_for, restored)

    def test_conditional_break_in_for_python_print(self):
        """Test python_print output for a realistic conditional-break pattern.

        Represents:
            for i in range(0, 100, 1):
                x = i + 1
                if x > 50:
                    break
        """
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)

        assign = ir.AssignStmt(x, ir.Add(i, ir.ConstInt(1, dtype, span), dtype, span), span)
        cond = ir.Gt(x, ir.ConstInt(50, dtype, span), dtype, span)
        if_break = ir.IfStmt(cond, ir.BreakStmt(span), None, [], span)
        body = ir.SeqStmts([assign, if_break], span)

        for_stmt = ir.ForStmt(
            i,
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(100, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            body,
            [],
            span,
        )

        code = ir.python_print(for_stmt)
        assert "for" in code
        assert "range" in code
        assert "break" in code
        assert "if" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
