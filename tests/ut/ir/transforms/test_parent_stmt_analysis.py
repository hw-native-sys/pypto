# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ParentStmtAnalysis utility class.

This tests the ParentStmtAnalysis class which builds and queries parent-child
relationships in statement trees.
"""

import pypto.language as pl
import pytest
from pypto import ir


def test_basic_parent_query():
    """Test simple parent-child relationship in a sequence of statements."""

    @pl.program
    class BasicParent:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            temp: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            result: pl.Tensor[[64], pl.FP32] = pl.add(temp, 1.0)
            return result

    func = BasicParent.get_function("main")
    assert func is not None

    # Create ParentStmtAnalysis and build map
    analysis = ir.ParentStmtAnalysis()
    analysis.build_map(func)

    # The body should be a SeqStmts containing multiple statements
    body = func.body
    assert body is not None

    # Check that body has no parent (it's the root)
    assert analysis.get_parent(body) is None
    assert not analysis.has_parent(body)


def test_nested_statements_in_seq():
    """Test parent relationships in SeqStmts containing multiple statements."""

    @pl.program
    class NestedSeq:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            a: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            b: pl.Tensor[[64], pl.FP32] = pl.add(a, 1.0)
            c: pl.Tensor[[64], pl.FP32] = pl.exp(b)
            return c

    func = NestedSeq.get_function("main")
    assert func is not None

    analysis = ir.ParentStmtAnalysis()
    analysis.build_map(func)

    body = func.body
    assert body is not None

    # If body is SeqStmts, check that all children have body as parent
    if isinstance(body, ir.SeqStmts):
        for stmt in body.stmts:
            parent = analysis.get_parent(stmt)
            assert parent is body
            assert analysis.has_parent(stmt)


def test_if_stmt_parent():
    """Test parent relationships in if statement bodies."""

    @pl.program
    class IfStmtParent:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = x
            if True:
                temp: pl.Tensor[[64], pl.FP32] = pl.mul(result, 2.0)
                result = temp
            else:
                temp2: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)
                result = temp2
            return result

    func = IfStmtParent.get_function("main")
    assert func is not None

    analysis = ir.ParentStmtAnalysis()
    analysis.build_map(func)

    # Find the if statement in the function body
    def find_if_stmt(stmt):
        """Recursively find IfStmt in statement tree."""
        if isinstance(stmt, ir.IfStmt):
            return stmt
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                result = find_if_stmt(s)
                if result:
                    return result
        return None

    if_stmt = find_if_stmt(func.body)
    if if_stmt:
        # Check that then_body and else_body have if_stmt as parent
        if if_stmt.then_body:
            then_parent = analysis.get_parent(if_stmt.then_body)
            assert then_parent is if_stmt

        if if_stmt.else_body:
            else_parent = analysis.get_parent(if_stmt.else_body)
            assert else_parent is if_stmt


def test_for_stmt_parent():
    """Test parent relationships in for loop bodies."""

    @pl.program
    class ForStmtParent:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = x
            for i in pl.range(5):
                temp: pl.Tensor[[64], pl.FP32] = pl.mul(result, 2.0)
                result = temp
            return result

    func = ForStmtParent.get_function("main")
    assert func is not None

    analysis = ir.ParentStmtAnalysis()
    analysis.build_map(func)

    # Find the for statement in the function body
    def find_for_stmt(stmt):
        """Recursively find ForStmt in statement tree."""
        if isinstance(stmt, ir.ForStmt):
            return stmt
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                result = find_for_stmt(s)
                if result:
                    return result
        return None

    for_stmt = find_for_stmt(func.body)
    if for_stmt:
        # Check that body has for_stmt as parent
        if for_stmt.body:
            body_parent = analysis.get_parent(for_stmt.body)
            assert body_parent is for_stmt


def test_root_has_no_parent():
    """Test that root statement (function body) has no parent."""

    @pl.program
    class RootNoParent:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

    func = RootNoParent.get_function("main")
    assert func is not None

    analysis = ir.ParentStmtAnalysis()
    analysis.build_map(func)

    # Root statement should have no parent
    assert analysis.get_parent(func.body) is None
    assert not analysis.has_parent(func.body)


def test_deeply_nested_structures():
    """Test parent relationships in deeply nested control flow."""

    @pl.program
    class DeeplyNested:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = x
            for i in pl.range(3):
                if i > 0:
                    for j in pl.range(2):
                        temp: pl.Tensor[[64], pl.FP32] = pl.mul(result, 2.0)
                        result = temp
            return result

    func = DeeplyNested.get_function("main")
    assert func is not None

    analysis = ir.ParentStmtAnalysis()
    analysis.build_map(func)

    # Additional verification: Find nested structures and verify their relationships
    # by directly accessing nodes in the known structure for more specific checks.
    assert isinstance(func.body, ir.SeqStmts), "Function body should be a SeqStmts"
    # The body is expected to be: assign, for, return. The for is at index 1.
    outer_for = func.body.stmts[1]
    assert isinstance(outer_for, ir.ForStmt), "Expected outer_for to be a ForStmt"

    # The for loop body contains just an if statement.
    inner_if = outer_for.body
    assert isinstance(inner_if, ir.IfStmt), "Expected inner_if to be an IfStmt"

    # The if's then_body contains just an inner for loop.
    inner_for = inner_if.then_body
    assert isinstance(inner_for, ir.ForStmt), "Expected inner_for to be a ForStmt"

    # Verify specific parent relationships
    assert analysis.get_parent(outer_for) is func.body, "Parent of outer_for should be the function body"
    assert analysis.get_parent(inner_if) is outer_for, "Parent of inner_if should be outer_for's body"
    assert analysis.get_parent(inner_for) is inner_if, "Parent of inner_for should be inner_if's then_body"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
