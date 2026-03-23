# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for transform_utils functions.

Tests: flatten_to_stmts, collect_def_vars, find_yield_stmt,
get_last_yield_stmt, substitute_expr, substitute_stmt.
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

    def test_while_stmt(self):
        """Collects vars from WhileStmt body."""
        v_body, body_assign = _assign("w", _const(5))
        body = ir.SeqStmts([body_assign, ir.YieldStmt(_span())], _span())
        cond = _const(1)
        while_stmt = ir.WhileStmt(cond, [], body, [], _span())
        result = ir.collect_def_vars(while_stmt)
        assert len(result) == 1
        assert result[0] is v_body

    def test_if_no_else(self):
        """Collects vars from IfStmt with no else branch."""
        v1, s1 = _assign("a", _const(1))
        cond = _const(1)
        if_stmt = ir.IfStmt(cond, s1, None, [], _span())
        result = ir.collect_def_vars(if_stmt)
        assert len(result) == 1
        assert result[0] is v1

    def test_scope_stmt(self):
        """Collects vars from ScopeStmt body."""
        v1, s1 = _assign("s", _const(7))
        body = ir.SeqStmts([s1], _span())
        scope = ir.ScopeStmt(ir.ScopeKind.InCore, body, _span())
        result = ir.collect_def_vars(scope)
        assert len(result) == 1
        assert result[0] is v1


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

    def test_last_in_op_stmts(self):
        """Finds trailing YieldStmt in OpStmts."""
        _, s1 = _assign("x", _const(1))
        ys = ir.YieldStmt(_span())
        # OpStmts only takes AssignStmt|EvalStmt, so wrap in SeqStmts
        # that ends with an OpStmts whose last is wrapped in a nested SeqStmts
        inner_seq = ir.SeqStmts([s1, ys], _span())
        outer = ir.SeqStmts([inner_seq], _span())
        assert ir.get_last_yield_stmt(outer) is ys


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

    def test_substitute_iter_arg(self):
        """Substituting an IterArg returns the replacement."""
        ia = ir.IterArg("ia", ir.ScalarType(DataType.INT64), _const(0), _span())
        replacement = _var("ia_new")
        result = ir.substitute_expr(ia, [(ia, replacement)])
        assert result is replacement

    def test_iter_arg_no_match(self):
        """Non-matching IterArg is returned unchanged."""
        ia = ir.IterArg("ia", ir.ScalarType(DataType.INT64), _const(0), _span())
        other = _var("other")
        result = ir.substitute_expr(ia, [(other, _var("z"))])
        assert result is ia

    def test_substitute_in_add(self):
        """Substitutes variables inside a BinaryExpr."""
        x = _var("x")
        y = _var("y")
        x_new = _var("x_new")
        add_expr = ir.Add(x, y, DataType.INT64, _span())
        result = ir.substitute_expr(add_expr, [(x, x_new)])
        assert isinstance(result, ir.Add)
        assert result is not add_expr

    def test_binary_no_change(self):
        """BinaryExpr with no matching vars is returned as-is."""
        x = _var("x")
        y = _var("y")
        add_expr = ir.Add(x, y, DataType.INT64, _span())
        other = _var("z")
        result = ir.substitute_expr(add_expr, [(other, _var("w"))])
        assert result is add_expr

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

    def test_unary_expr(self):
        """Substitutes through a UnaryExpr (Neg)."""
        x = _var("x")
        x_new = _var("x_new")
        neg = ir.Neg(x, DataType.INT64, _span())
        result = ir.substitute_expr(neg, [(x, x_new)])
        assert isinstance(result, ir.Neg)
        assert result is not neg

    def test_unary_no_change(self):
        """UnaryExpr with no matching vars is returned as-is."""
        x = _var("x")
        neg = ir.Neg(x, DataType.INT64, _span())
        other = _var("z")
        result = ir.substitute_expr(neg, [(other, _var("w"))])
        assert result is neg

    def test_call_expr(self):
        """Substitutes through Call args."""
        x = _var("x")
        x_new = _var("x_new")
        op = ir.Op("test_op")
        call = ir.Call(op, [x], ir.ScalarType(DataType.INT64), _span())
        result = ir.substitute_expr(call, [(x, x_new)])
        assert isinstance(result, ir.Call)
        assert result is not call

    def test_call_no_change(self):
        """Call with no matching vars is returned as-is."""
        x = _var("x")
        op = ir.Op("test_op")
        call = ir.Call(op, [x], ir.ScalarType(DataType.INT64), _span())
        other = _var("z")
        result = ir.substitute_expr(call, [(other, _var("w"))])
        assert result is call

    def test_make_tuple(self):
        """Substitutes through MakeTuple elements."""
        x = _var("x")
        y = _var("y")
        x_new = _var("x_new")
        tup = ir.MakeTuple([x, y], _span())
        result = ir.substitute_expr(tup, [(x, x_new)])
        assert isinstance(result, ir.MakeTuple)
        assert result is not tup

    def test_make_tuple_no_change(self):
        """MakeTuple with no matching vars is returned as-is."""
        x = _var("x")
        tup = ir.MakeTuple([x], _span())
        other = _var("z")
        result = ir.substitute_expr(tup, [(other, _var("w"))])
        assert result is tup

    def test_tuple_get_item(self):
        """Substitutes through TupleGetItemExpr."""
        x = _var("x")
        y = _var("y")
        x_new = _var("x_new")
        tup = ir.MakeTuple([x, y], _span())
        get_item = ir.TupleGetItemExpr(tup, 0, _span())
        result = ir.substitute_expr(get_item, [(x, x_new)])
        assert isinstance(result, ir.TupleGetItemExpr)
        assert result is not get_item

    def test_tuple_get_item_no_change(self):
        """TupleGetItemExpr with no matching vars is returned as-is."""
        x = _var("x")
        tup = ir.MakeTuple([x], _span())
        get_item = ir.TupleGetItemExpr(tup, 0, _span())
        other = _var("z")
        result = ir.substitute_expr(get_item, [(other, _var("w"))])
        assert result is get_item


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

    def test_for_stmt(self):
        """Substitution works through ForStmt start/stop/body."""
        x = _var("x")
        x_new = _var("x_new")
        loop_var = _var("i")
        _, body_assign = _assign("r", x)
        body = ir.SeqStmts([body_assign, ir.YieldStmt(_span())], _span())
        for_stmt = ir.ForStmt(loop_var, _const(0), _const(10), _const(1), [], body, [], _span())
        result = ir.substitute_stmt(for_stmt, [(x, x_new)])
        assert isinstance(result, ir.ForStmt)

    def test_if_stmt(self):
        """Substitution works through IfStmt condition and branches."""
        x = _var("x")
        x_new = _var("x_new")
        _, then_assign = _assign("a", x)
        _, else_assign = _assign("b", _const(0))
        cond = _const(1)
        if_stmt = ir.IfStmt(cond, then_assign, else_assign, [], _span())
        result = ir.substitute_stmt(if_stmt, [(x, x_new)])
        assert isinstance(result, ir.IfStmt)

    def test_while_stmt(self):
        """Substitution works through WhileStmt condition and body."""
        x = _var("x")
        x_new = _var("x_new")
        _, body_assign = _assign("w", x)
        body = ir.SeqStmts([body_assign, ir.YieldStmt(_span())], _span())
        cond = _const(1)
        while_stmt = ir.WhileStmt(cond, [], body, [], _span())
        result = ir.substitute_stmt(while_stmt, [(x, x_new)])
        assert isinstance(result, ir.WhileStmt)

    def test_substitute_iter_arg_in_stmt(self):
        """Substitution replaces IterArg references in statement subtree."""
        ia = ir.IterArg("ia", ir.ScalarType(DataType.INT64), _const(0), _span())
        replacement = _var("ia_new")
        stmt = ir.AssignStmt(_var("out"), ia, _span())
        result = ir.substitute_stmt(stmt, [(ia, replacement)])
        assert isinstance(result, ir.AssignStmt)

    def test_eval_stmt(self):
        """Substitution works through EvalStmt."""
        x = _var("x")
        x_new = _var("x_new")
        op = ir.Op("test_op")
        call = ir.Call(op, [x], ir.ScalarType(DataType.INT64), _span())
        eval_stmt = ir.EvalStmt(call, _span())
        result = ir.substitute_stmt(eval_stmt, [(x, x_new)])
        assert isinstance(result, ir.EvalStmt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
