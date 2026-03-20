# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for transform_utils functions.

Tests: flatten_to_stmts, wrap_in_seq_stmts, collect_def_vars,
find_yield_stmt, get_last_yield_stmt, substitute_expr, substitute_stmt.
"""

import pytest
from pypto import DataType, ir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span() -> ir.Span:
    return ir.Span.unknown()


def _var(name: str, dtype: DataType = DataType.INT64) -> ir.Var:
    return ir.Var(name, ir.ScalarType(dtype), _span())


def _const(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INT64, _span())


def _assign(name: str, value: ir.Expr) -> tuple[ir.Var, ir.AssignStmt]:
    v = _var(name)
    return v, ir.AssignStmt(v, value, _span())


# ---------------------------------------------------------------------------
# TestFlattenToStmts
# ---------------------------------------------------------------------------


class TestFlattenToStmts:
    """Tests for ir.flatten_to_stmts."""

    def test_single_stmt(self):
        """A non-container stmt returns a single-element list."""
        _, stmt = _assign("x", _const(1))
        result = ir.flatten_to_stmts(stmt)
        assert len(result) == 1
        assert result[0] is stmt

    def test_seq_stmts(self):
        """SeqStmts returns its children."""
        _, s1 = _assign("a", _const(1))
        _, s2 = _assign("b", _const(2))
        seq = ir.SeqStmts([s1, s2], _span())
        result = ir.flatten_to_stmts(seq)
        assert len(result) == 2
        assert result[0] is s1
        assert result[1] is s2

    def test_empty_seq_stmts(self):
        """Empty SeqStmts returns empty list."""
        seq = ir.SeqStmts([], _span())
        result = ir.flatten_to_stmts(seq)
        assert len(result) == 0

    def test_yield_stmt(self):
        """YieldStmt is a leaf — returns single-element list."""
        ys = ir.YieldStmt(_span())
        result = ir.flatten_to_stmts(ys)
        assert len(result) == 1
        assert result[0] is ys


# ---------------------------------------------------------------------------
# TestWrapInSeqStmts
# ---------------------------------------------------------------------------


class TestWrapInSeqStmts:
    """Tests for ir.wrap_in_seq_stmts."""

    def test_wrap_multiple(self):
        """Wrapping multiple stmts produces a SeqStmts."""
        _, s1 = _assign("a", _const(1))
        _, s2 = _assign("b", _const(2))
        wrapped = ir.wrap_in_seq_stmts([s1, s2], _span())
        assert isinstance(wrapped, ir.SeqStmts)
        children = ir.flatten_to_stmts(wrapped)
        assert len(children) == 2

    def test_wrap_empty(self):
        """Wrapping empty list produces a SeqStmts with no children."""
        wrapped = ir.wrap_in_seq_stmts([], _span())
        assert isinstance(wrapped, ir.SeqStmts)
        assert len(ir.flatten_to_stmts(wrapped)) == 0

    def test_roundtrip(self):
        """flatten → wrap → flatten roundtrip preserves stmt count."""
        _, s1 = _assign("x", _const(10))
        _, s2 = _assign("y", _const(20))
        seq = ir.SeqStmts([s1, s2], _span())
        stmts = ir.flatten_to_stmts(seq)
        rewrapped = ir.wrap_in_seq_stmts(stmts, _span())
        assert len(ir.flatten_to_stmts(rewrapped)) == 2


# ---------------------------------------------------------------------------
# TestCollectDefVars
# ---------------------------------------------------------------------------


class TestCollectDefVars:
    """Tests for ir.collect_def_vars."""

    def test_single_assign(self):
        """Single AssignStmt yields one def var."""
        v, stmt = _assign("x", _const(42))
        result = ir.collect_def_vars(stmt)
        assert len(result) == 1
        assert result[0] is v

    def test_seq_assigns(self):
        """Multiple AssignStmts in a SeqStmts yields all def vars."""
        v1, s1 = _assign("a", _const(1))
        v2, s2 = _assign("b", _const(2))
        seq = ir.SeqStmts([s1, s2], _span())
        result = ir.collect_def_vars(seq)
        assert len(result) == 2
        assert result[0] is v1
        assert result[1] is v2

    def test_no_assigns(self):
        """A statement with no AssignStmts returns empty list."""
        ys = ir.YieldStmt(_span())
        result = ir.collect_def_vars(ys)
        assert len(result) == 0

    def test_nested_if(self):
        """Collects vars from both branches of an IfStmt."""
        v1, s1 = _assign("a", _const(1))
        v2, s2 = _assign("b", _const(2))
        cond = _const(1)
        if_stmt = ir.IfStmt(cond, s1, s2, [], _span())
        result = ir.collect_def_vars(if_stmt)
        assert len(result) == 2
        assert result[0] is v1
        assert result[1] is v2

    def test_nested_for(self):
        """Collects vars from ForStmt body."""
        loop_var = _var("i")
        v_body, body_assign = _assign("x", _const(0))
        body = ir.SeqStmts([body_assign, ir.YieldStmt(_span())], _span())
        for_stmt = ir.ForStmt(loop_var, _const(0), _const(10), _const(1), [], body, [], _span())
        result = ir.collect_def_vars(for_stmt)
        assert len(result) == 1
        assert result[0] is v_body


# ---------------------------------------------------------------------------
# TestFindYieldStmt
# ---------------------------------------------------------------------------


class TestFindYieldStmt:
    """Tests for ir.find_yield_stmt."""

    def test_direct_yield(self):
        """A YieldStmt is found directly."""
        ys = ir.YieldStmt(_span())
        assert ir.find_yield_stmt(ys) is ys

    def test_yield_in_seq(self):
        """Finds YieldStmt inside SeqStmts."""
        _, s1 = _assign("x", _const(1))
        ys = ir.YieldStmt(_span())
        seq = ir.SeqStmts([s1, ys], _span())
        assert ir.find_yield_stmt(seq) is ys

    def test_no_yield(self):
        """Returns None when no YieldStmt exists."""
        _, s1 = _assign("x", _const(1))
        assert ir.find_yield_stmt(s1) is None

    def test_finds_first(self):
        """Finds the first YieldStmt when multiple exist."""
        x = _var("x")
        y = _var("y")
        ys1 = ir.YieldStmt([x], _span())
        ys2 = ir.YieldStmt([y], _span())
        seq = ir.SeqStmts([ys1, ys2], _span())
        found = ir.find_yield_stmt(seq)
        assert found is ys1


# ---------------------------------------------------------------------------
# TestGetLastYieldStmt
# ---------------------------------------------------------------------------


class TestGetLastYieldStmt:
    """Tests for ir.get_last_yield_stmt."""

    def test_direct_yield(self):
        """A YieldStmt is returned directly."""
        ys = ir.YieldStmt(_span())
        assert ir.get_last_yield_stmt(ys) is ys

    def test_last_in_seq(self):
        """Finds the last element in SeqStmts."""
        _, s1 = _assign("x", _const(1))
        ys = ir.YieldStmt(_span())
        seq = ir.SeqStmts([s1, ys], _span())
        assert ir.get_last_yield_stmt(seq) is ys

    def test_non_yield_last(self):
        """Returns None when last element is not a YieldStmt."""
        ys = ir.YieldStmt(_span())
        _, s1 = _assign("x", _const(1))
        seq = ir.SeqStmts([ys, s1], _span())
        assert ir.get_last_yield_stmt(seq) is None

    def test_no_yield(self):
        """Returns None for a non-yield stmt."""
        _, s1 = _assign("x", _const(1))
        assert ir.get_last_yield_stmt(s1) is None

    def test_empty_seq(self):
        """Returns None for empty SeqStmts."""
        seq = ir.SeqStmts([], _span())
        assert ir.get_last_yield_stmt(seq) is None


# ---------------------------------------------------------------------------
# TestSubstituteExpr
# ---------------------------------------------------------------------------


class TestSubstituteExpr:
    """Tests for ir.substitute_expr."""

    def test_substitute_var(self):
        """Substituting a Var returns the replacement."""
        x = _var("x")
        y = _var("y")
        result = ir.substitute_expr(x, [(x, y)])
        assert result is y

    def test_no_match(self):
        """Non-matching Var is returned unchanged."""
        x = _var("x")
        y = _var("y")
        z = _var("z")
        result = ir.substitute_expr(x, [(y, z)])
        assert result is x

    def test_substitute_in_add(self):
        """Substitutes variables inside a BinaryExpr."""
        x = _var("x")
        y = _var("y")
        x_new = _var("x_new")
        add_expr = ir.Add(x, y, DataType.INT64, _span())
        result = ir.substitute_expr(add_expr, [(x, x_new)])
        # Result should be a new Add with x_new as left operand
        assert isinstance(result, ir.Add)
        assert result is not add_expr

    def test_const_unchanged(self):
        """Constants are returned as-is."""
        c = _const(42)
        result = ir.substitute_expr(c, [(_var("x"), _var("y"))])
        assert result is c

    def test_empty_map(self):
        """Empty substitution map returns the original expression."""
        x = _var("x")
        result = ir.substitute_expr(x, [])
        assert result is x


# ---------------------------------------------------------------------------
# TestSubstituteStmt
# ---------------------------------------------------------------------------


class TestSubstituteStmt:
    """Tests for ir.substitute_stmt."""

    def test_substitute_in_assign_value(self):
        """Substitutes a variable in the RHS of an AssignStmt."""
        x = _var("x")
        y = _var("y")
        y_new = _var("y_new")
        stmt = ir.AssignStmt(x, y, _span())
        result = ir.substitute_stmt(stmt, [(y, y_new)])
        assert isinstance(result, ir.AssignStmt)

    def test_empty_map(self):
        """Empty substitution map returns structurally equal stmt."""
        x = _var("x")
        stmt = ir.AssignStmt(x, _const(1), _span())
        result = ir.substitute_stmt(stmt, [])
        assert result is not None

    def test_substitute_in_seq(self):
        """Substitution works through SeqStmts."""
        x = _var("x")
        y = _var("y")
        x_new = _var("x_new")
        _, s1 = _assign("a", x)
        _, s2 = _assign("b", y)
        seq = ir.SeqStmts([s1, s2], _span())
        result = ir.substitute_stmt(seq, [(x, x_new)])
        assert isinstance(result, ir.SeqStmts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
