# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Stmt.leading_comments metadata and ir.attach_leading_comments helper."""

import pytest
from pypto import DataType, ir


def _make_assign() -> ir.AssignStmt:
    span = ir.Span("test.py", 1, 1, 1, 10)
    x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
    y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
    return ir.AssignStmt(x, y, span)


class TestLeadingComments:
    def test_default_empty(self):
        stmt = _make_assign()
        assert stmt.leading_comments == []

    def test_attach_sets_field(self):
        stmt = _make_assign()
        result = ir.attach_leading_comments(stmt, ["first", "second"])
        assert stmt.leading_comments == ["first", "second"]
        assert result is stmt

    def test_attach_replaces_existing(self):
        stmt = _make_assign()
        ir.attach_leading_comments(stmt, ["one"])
        ir.attach_leading_comments(stmt, ["two", "three"])
        assert stmt.leading_comments == ["two", "three"]

    def test_attach_empty_clears(self):
        stmt = _make_assign()
        ir.attach_leading_comments(stmt, ["x"])
        ir.attach_leading_comments(stmt, [])
        assert stmt.leading_comments == []

    def test_python_field_is_read_only(self):
        stmt = _make_assign()
        with pytest.raises(AttributeError):
            stmt.leading_comments = ["nope"]  # type: ignore[misc]

    def test_structural_equal_ignores_comments(self):
        span = ir.Span("test.py", 1, 1, 1, 10)
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        lhs = ir.AssignStmt(x, y, span)
        rhs = ir.AssignStmt(x, y, span)
        ir.attach_leading_comments(rhs, ["annotation"])
        ir.assert_structural_equal(lhs, rhs)

    def test_structural_hash_ignores_comments(self):
        span = ir.Span("test.py", 1, 1, 1, 10)
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        lhs = ir.AssignStmt(x, y, span)
        rhs = ir.AssignStmt(x, y, span)
        ir.attach_leading_comments(rhs, ["annotation"])
        assert ir.structural_hash(lhs) == ir.structural_hash(rhs)

    def test_works_on_all_simple_stmt_kinds(self):
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)

        stmts: list[ir.Stmt] = [
            ir.AssignStmt(x, x, span),
            ir.ReturnStmt([x], span),
            ir.BreakStmt(span),
            ir.ContinueStmt(span),
            ir.EvalStmt(x, span),
        ]
        for stmt in stmts:
            assert stmt.leading_comments == []
            ir.attach_leading_comments(stmt, ["c"])
            assert stmt.leading_comments == ["c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
