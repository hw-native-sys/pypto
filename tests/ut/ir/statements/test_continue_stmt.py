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


def test_continue_stmt_creation():
    """Test creating a ContinueStmt."""
    span = ir.Span("test.py", 1, 1)
    stmt = ir.ContinueStmt(span)

    assert isinstance(stmt, ir.ContinueStmt)
    assert isinstance(stmt, ir.Stmt)
    assert stmt.span.filename == "test.py"


def test_continue_stmt_python_print():
    """Test printing a ContinueStmt as Python code."""
    span = ir.Span("test.py", 1, 1)
    stmt = ir.ContinueStmt(span)

    code = ir.python_print(stmt)
    assert code == "continue"


def test_continue_stmt_structural_hash():
    """Test that two ContinueStmts produce the same hash."""
    span1 = ir.Span("a.py", 1, 1)
    span2 = ir.Span("b.py", 2, 2)
    stmt1 = ir.ContinueStmt(span1)
    stmt2 = ir.ContinueStmt(span2)

    assert ir.structural_hash(stmt1) == ir.structural_hash(stmt2)


def test_continue_stmt_structural_equal():
    """Test structural equality of ContinueStmts."""
    span1 = ir.Span("a.py", 1, 1)
    span2 = ir.Span("b.py", 2, 2)
    stmt1 = ir.ContinueStmt(span1)
    stmt2 = ir.ContinueStmt(span2)

    ir.assert_structural_equal(stmt1, stmt2)


def test_continue_stmt_not_equal_to_break():
    """Test that ContinueStmt is not structurally equal to BreakStmt."""
    span = ir.Span("test.py", 1, 1)
    continue_stmt = ir.ContinueStmt(span)
    break_stmt = ir.BreakStmt(span)

    assert not ir.structural_equal(continue_stmt, break_stmt)


def test_continue_stmt_serialization():
    """Test serializing and deserializing a ContinueStmt."""
    span = ir.Span("test.py", 1, 1)
    stmt = ir.ContinueStmt(span)

    data = ir.serialize(stmt)
    restored = ir.deserialize(data)

    assert isinstance(restored, ir.ContinueStmt)
    ir.assert_structural_equal(stmt, restored)


def test_continue_stmt_not_equal_to_other_stmt_types():
    """Test that ContinueStmt is not structurally equal to other statement types."""
    span = ir.Span("test.py", 1, 1)
    dtype = DataType.INT64
    continue_stmt = ir.ContinueStmt(span)
    x = ir.Var("x", ir.ScalarType(dtype), span)
    assign = ir.AssignStmt(x, ir.ConstInt(0, dtype, span), span)
    yield_stmt = ir.YieldStmt([x], span)
    return_stmt = ir.ReturnStmt([x], span)

    assert not ir.structural_equal(continue_stmt, assign)
    assert not ir.structural_equal(continue_stmt, yield_stmt)
    assert not ir.structural_equal(continue_stmt, return_stmt)


def test_continue_stmt_immutability():
    """Test that ContinueStmt span attribute is immutable."""
    span = ir.Span("test.py", 1, 1)
    stmt = ir.ContinueStmt(span)

    with pytest.raises(AttributeError):
        stmt.span = ir.Span("other.py", 2, 2)  # type: ignore


class TestContinueInForLoop:
    """Test ContinueStmt composed inside ForStmt bodies."""

    def _make_for_with_body(self, body: ir.Stmt) -> ir.ForStmt:
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        return ir.ForStmt(i, start, stop, step, [], body, [], span)

    def test_for_loop_with_continue_body(self):
        """Test ForStmt whose body is a single ContinueStmt."""
        span = ir.Span("test.py", 1, 1)
        continue_stmt = ir.ContinueStmt(span)
        for_stmt = self._make_for_with_body(continue_stmt)

        assert isinstance(for_stmt.body, ir.ContinueStmt)

    def test_for_loop_with_continue_in_seq(self):
        """Test ForStmt with continue inside a SeqStmts body."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, ir.ConstInt(42, dtype, span), span)
        continue_stmt = ir.ContinueStmt(span)
        seq = ir.SeqStmts([assign, continue_stmt], span)
        for_stmt = self._make_for_with_body(seq)

        assert isinstance(for_stmt.body, ir.SeqStmts)
        assert len(for_stmt.body.stmts) == 2
        assert isinstance(for_stmt.body.stmts[0], ir.AssignStmt)
        assert isinstance(for_stmt.body.stmts[1], ir.ContinueStmt)

    def test_for_loop_with_conditional_continue(self):
        """Test ForStmt with continue inside an IfStmt (if cond: continue).

        Represents: for i in range(0, 10, 1): if i < 3: continue
        """
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        cond = ir.Lt(i, ir.ConstInt(3, dtype, span), dtype, span)
        continue_stmt = ir.ContinueStmt(span)
        if_stmt = ir.IfStmt(cond, continue_stmt, None, [], span)
        for_stmt = self._make_for_with_body(if_stmt)

        assert isinstance(for_stmt.body, ir.IfStmt)
        assert isinstance(for_stmt.body.then_body, ir.ContinueStmt)

    def test_for_loop_with_continue_python_print(self):
        """Test python_print of a ForStmt containing a conditional continue."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        cond = ir.Lt(i, ir.ConstInt(3, dtype, span), dtype, span)
        continue_stmt = ir.ContinueStmt(span)
        if_stmt = ir.IfStmt(cond, continue_stmt, None, [], span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], if_stmt, [], span)

        code = ir.python_print(for_stmt)
        assert "for" in code
        assert "continue" in code
        assert "if" in code

    def test_for_loop_with_continue_structural_equal(self):
        """Test structural equality of two ForStmts with identical conditional continue."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64

        def make_for_continue():
            i = ir.Var("i", ir.ScalarType(dtype), span)
            start = ir.ConstInt(0, dtype, span)
            stop = ir.ConstInt(10, dtype, span)
            step = ir.ConstInt(1, dtype, span)
            cond = ir.Lt(i, ir.ConstInt(3, dtype, span), dtype, span)
            cont = ir.ContinueStmt(span)
            if_stmt = ir.IfStmt(cond, cont, None, [], span)
            return ir.ForStmt(i, start, stop, step, [], if_stmt, [], span)

        ir.assert_structural_equal(make_for_continue(), make_for_continue())

    def test_for_loop_with_continue_serialization(self):
        """Test serialization roundtrip of a ForStmt containing a continue."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        start = ir.ConstInt(0, dtype, span)
        stop = ir.ConstInt(10, dtype, span)
        step = ir.ConstInt(1, dtype, span)
        continue_stmt = ir.ContinueStmt(span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], continue_stmt, [], span)

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.ForStmt)
        assert isinstance(restored.body, ir.ContinueStmt)
        ir.assert_structural_equal(for_stmt, restored)

    def test_for_loop_continue_vs_break_not_equal(self):
        """Test that for-with-continue is not equal to for-with-break."""
        span = ir.Span("test.py", 1, 1)
        for_continue = self._make_for_with_body(ir.ContinueStmt(span))
        for_break = self._make_for_with_body(ir.BreakStmt(span))

        assert not ir.structural_equal(for_continue, for_break)


class TestContinueInWhileLoop:
    """Test ContinueStmt composed inside WhileStmt bodies."""

    def test_while_loop_with_continue_body(self):
        """Test WhileStmt whose body is a single ContinueStmt."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        cond = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        continue_stmt = ir.ContinueStmt(span)
        while_stmt = ir.WhileStmt(cond, [], continue_stmt, [], span)

        assert isinstance(while_stmt.body, ir.ContinueStmt)

    def test_while_loop_with_conditional_continue(self):
        """Test WhileStmt with conditional continue: while cond: if x < 3: continue."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        loop_cond = ir.Lt(x, ir.ConstInt(100, dtype, span), dtype, span)
        skip_cond = ir.Lt(x, ir.ConstInt(3, dtype, span), dtype, span)
        continue_stmt = ir.ContinueStmt(span)
        if_stmt = ir.IfStmt(skip_cond, continue_stmt, None, [], span)
        while_stmt = ir.WhileStmt(loop_cond, [], if_stmt, [], span)

        assert isinstance(while_stmt.body, ir.IfStmt)
        assert isinstance(while_stmt.body.then_body, ir.ContinueStmt)

    def test_while_loop_with_continue_serialization(self):
        """Test serialization roundtrip of a WhileStmt containing a continue."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        cond = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        continue_stmt = ir.ContinueStmt(span)
        while_stmt = ir.WhileStmt(cond, [], continue_stmt, [], span)

        data = ir.serialize(while_stmt)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.WhileStmt)
        assert isinstance(restored.body, ir.ContinueStmt)
        # WhileStmt condition has free vars; use auto-mapping after deserialization
        ir.assert_structural_equal(while_stmt, restored, enable_auto_mapping=True)

    def test_while_loop_with_continue_structural_hash(self):
        """Test structural hash consistency for WhileStmt containing continue."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        # Share Var instance so pointer-based hash is stable
        x = ir.Var("x", ir.ScalarType(dtype), span)
        cond = ir.Lt(x, ir.ConstInt(10, dtype, span), dtype, span)
        w1 = ir.WhileStmt(cond, [], ir.ContinueStmt(span), [], span)
        w2 = ir.WhileStmt(cond, [], ir.ContinueStmt(span), [], span)

        assert ir.structural_hash(w1) == ir.structural_hash(w2)


class TestContinueInNestedStructures:
    """Test ContinueStmt in deeply nested IR structures."""

    def test_continue_in_if_else(self):
        """Test continue in then branch and break in else branch of IfStmt."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        cond = ir.Lt(x, ir.ConstInt(3, dtype, span), dtype, span)
        continue_stmt = ir.ContinueStmt(span)
        break_stmt = ir.BreakStmt(span)
        if_stmt = ir.IfStmt(cond, continue_stmt, break_stmt, [], span)

        assert isinstance(if_stmt.then_body, ir.ContinueStmt)
        assert isinstance(if_stmt.else_body, ir.BreakStmt)

        code = ir.python_print(if_stmt)
        assert "continue" in code
        assert "break" in code

    def test_continue_in_seq_with_multiple_stmts(self):
        """Test continue preceded by assignments in a SeqStmts."""
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, ir.ConstInt(1, dtype, span), span)
        assign2 = ir.AssignStmt(y, ir.Add(x, ir.ConstInt(2, dtype, span), dtype, span), span)
        continue_stmt = ir.ContinueStmt(span)
        seq = ir.SeqStmts([assign1, assign2, continue_stmt], span)

        assert len(seq.stmts) == 3
        assert isinstance(seq.stmts[2], ir.ContinueStmt)

        # Serialization roundtrip
        data = ir.serialize(seq)
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.SeqStmts)
        assert isinstance(restored.stmts[2], ir.ContinueStmt)
        ir.assert_structural_equal(seq, restored)

    def test_nested_for_with_continue_in_inner_loop(self):
        """Test nested for loops where inner loop has a continue.

        Represents:
            for i in range(0, 10, 1):
                for j in range(0, 5, 1):
                    continue
        """
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        j = ir.Var("j", ir.ScalarType(dtype), span)

        continue_stmt = ir.ContinueStmt(span)
        inner_for = ir.ForStmt(
            j,
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(5, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            continue_stmt,
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
        assert isinstance(outer_for.body.body, ir.ContinueStmt)

        # Structural equality
        inner_for2 = ir.ForStmt(
            ir.Var("j", ir.ScalarType(dtype), span),
            ir.ConstInt(0, dtype, span),
            ir.ConstInt(5, dtype, span),
            ir.ConstInt(1, dtype, span),
            [],
            ir.ContinueStmt(span),
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

    def test_nested_for_with_continue_serialization(self):
        """Test serialization of nested for loop with continue."""
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
            ir.ContinueStmt(span),
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
        assert isinstance(restored.body.body, ir.ContinueStmt)
        ir.assert_structural_equal(outer_for, restored)

    def test_conditional_continue_in_for_python_print(self):
        """Test python_print output for a realistic conditional-continue pattern.

        Represents:
            for i in range(0, 100, 1):
                if i < 10:
                    continue
                x = i * 2
        """
        span = ir.Span("test.py", 1, 1)
        dtype = DataType.INT64
        i = ir.Var("i", ir.ScalarType(dtype), span)
        x = ir.Var("x", ir.ScalarType(dtype), span)

        cond = ir.Lt(i, ir.ConstInt(10, dtype, span), dtype, span)
        if_continue = ir.IfStmt(cond, ir.ContinueStmt(span), None, [], span)
        assign = ir.AssignStmt(x, ir.Mul(i, ir.ConstInt(2, dtype, span), dtype, span), span)
        body = ir.SeqStmts([if_continue, assign], span)

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
        assert "continue" in code
        assert "if" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
