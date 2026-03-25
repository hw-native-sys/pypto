# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the SimplifyExpr pass.

This pass simplifies all scalar expressions in the IR using algebraic rewrite
rules and bound analysis. IRMutatorWithAnalyzer binds ForStmt loop variables
to their ranges, and ConstraintContext propagates if-branch conditions,
enabling range-aware simplification.

Tests use the @pl.program DSL where possible. Constant folding tests use
direct IR construction because Python eagerly evaluates constant expressions
(e.g., `3 + 4` becomes `7` before the DSL sees it).
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes

S = ir.Span.unknown()
IDX = DataType.INDEX


def ci(value: int, dtype: DataType = IDX) -> ir.ConstInt:
    return ir.ConstInt(value, dtype, S)


def make_var(name: str, dtype: DataType = IDX) -> ir.Var:
    return ir.Var(name, ir.ScalarType(dtype), S)


def wrap_stmts(stmts):
    if len(stmts) == 1:
        return stmts[0]
    return ir.SeqStmts(stmts, S)


def make_program(body_stmts):
    body = wrap_stmts(body_stmts)
    func = ir.Function("main", [], [], body, S)
    return ir.Program([func], "test", S)


# ============================================================================
# Pass metadata
# ============================================================================


class TestPassMetadata:
    def test_pass_name(self):
        p = passes.simplify_expr()
        assert p.get_name() == "SimplifyExpr"

    def test_pass_no_required_properties(self):
        p = passes.simplify_expr()
        assert p.get_required_properties().empty()

    def test_pass_no_produced_properties(self):
        p = passes.simplify_expr()
        assert p.get_produced_properties().empty()


# ============================================================================
# Identity simplifications (x + 0 -> x, x * 1 -> x)
# ============================================================================


class TestIdentitySimplification:
    def test_add_zero(self):
        """x + 0 should simplify to x."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i + 0

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_zero_add(self):
        """0 + x should simplify to x."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = 0 + i

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_mul_one(self):
        """x * 1 should simplify to x."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i * 1

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_sub_zero(self):
        """x - 0 should simplify to x."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i - 0

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# Constant folding (direct IR — DSL can't produce constant-only expressions)
# ============================================================================


class TestConstantFolding:
    def test_add_constants(self):
        """3 + 4 should fold to 7."""
        y = make_var("y")
        assign = ir.AssignStmt(y, ir.Add(ci(3), ci(4), IDX, S), S)
        before = make_program([assign])

        assign_exp = ir.AssignStmt(y, ci(7), S)
        expected = make_program([assign_exp])

        after = passes.simplify_expr()(before)
        ir.assert_structural_equal(after, expected)

    def test_mul_constants(self):
        """3 * 4 should fold to 12."""
        y = make_var("y")
        assign = ir.AssignStmt(y, ir.Mul(ci(3), ci(4), IDX, S), S)
        before = make_program([assign])

        assign_exp = ir.AssignStmt(y, ci(12), S)
        expected = make_program([assign_exp])

        after = passes.simplify_expr()(before)
        ir.assert_structural_equal(after, expected)

    def test_nested_constant_expr(self):
        """(2 + 3) * 4 should fold to 20."""
        y = make_var("y")
        inner = ir.Add(ci(2), ci(3), IDX, S)
        outer = ir.Mul(inner, ci(4), IDX, S)
        assign = ir.AssignStmt(y, outer, S)
        before = make_program([assign])

        assign_exp = ir.AssignStmt(y, ci(20), S)
        expected = make_program([assign_exp])

        after = passes.simplify_expr()(before)
        ir.assert_structural_equal(after, expected)


# ============================================================================
# Range-aware simplification (requires loop variable binding)
# ============================================================================


class TestRangeAwareSimplification:
    def test_floordiv_by_range_bound(self):
        """i // 8 should simplify to 0 when i is in [0, 8)."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i // 8

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = 0

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_floormod_by_range_bound(self):
        """i % 8 should simplify to i when i is in [0, 8)."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i % 8

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_floordiv_not_simplifiable(self):
        """i // 4 should NOT simplify when i is in [0, 8) — result is 0 or 1."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i // 4

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Before)

    def test_nested_loops(self):
        """Inner loop variable binding should work in nested loops."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    for j in pl.range(4):
                        _y: pl.Scalar[pl.INDEX] = j // 4

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    for j in pl.range(4):
                        _y: pl.Scalar[pl.INDEX] = 0

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# If-branch constraint propagation
# ============================================================================


class TestIfBranchConstraint:
    def test_then_branch_uses_condition(self):
        """In then-branch of `if i < 4`, i is in [0, 4) so i // 4 == 0."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    if i < 4:
                        _y: pl.Scalar[pl.INDEX] = i // 4

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    if i < 4:
                        _y: pl.Scalar[pl.INDEX] = 0

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_else_branch_uses_negated_condition(self):
        """In else-branch of `if i < 4`, Not(i<4) → i>=4 tightens bounds to [4, 8).
        Combined with loop [0, 8): i // 8 ∈ [0, 0] → 0."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    if i < 4:
                        _y: pl.Scalar[pl.INDEX] = i // 4
                    else:
                        _y: pl.Scalar[pl.INDEX] = i // 8

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    if i < 4:
                        _y: pl.Scalar[pl.INDEX] = 0
                    else:
                        _y: pl.Scalar[pl.INDEX] = 0

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_nested_if_in_loop(self):
        """Nested if inside for loop: both loop binding and condition constraint active."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(16):
                    if i < 8:
                        _y: pl.Scalar[pl.INDEX] = i // 8
                    else:
                        _z: pl.Scalar[pl.INDEX] = i // 16

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(16):
                    if i < 8:
                        _y: pl.Scalar[pl.INDEX] = 0
                    else:
                        _z: pl.Scalar[pl.INDEX] = 0

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# Comprehensive control flow (break, continue, scope, while, seq)
# ============================================================================


class TestControlFlow:
    def test_break_stmt_passthrough(self):
        """BreakStmt is a leaf — pass should simplify surrounding exprs without error."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i + 0
                    break

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i
                    break

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_continue_stmt_passthrough(self):
        """ContinueStmt is a leaf — pass should simplify surrounding exprs without error."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(4):
                    _y: pl.Scalar[pl.INDEX] = i * 1
                    continue

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(4):
                    _y: pl.Scalar[pl.INDEX] = i
                    continue

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_scope_stmt_traversal(self):
        """Pass should traverse into ScopeStmt bodies and simplify."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    with pl.incore():
                        _y: pl.Scalar[pl.INDEX] = i + 0

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    with pl.incore():
                        _y: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_while_condition_simplified(self):
        """WhileStmt condition expressions should be simplified."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX]):
                i: pl.Scalar[pl.INDEX] = 0
                while i < n + 0:
                    i = i + 1

        @pl.program
        class Expected:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX]):
                i: pl.Scalar[pl.INDEX] = 0
                while i < n:
                    i = i + 1

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_sequential_stmts(self):
        """Multiple statements should all be simplified."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i + 0
                    _z: pl.Scalar[pl.INDEX] = i * 1

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i
                    _z: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_if_with_break_and_continue(self):
        """If-branch with break/continue alongside simplifiable expressions."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    if i < 4:
                        _y: pl.Scalar[pl.INDEX] = i // 4
                        break
                    else:
                        _y: pl.Scalar[pl.INDEX] = i + 0
                        continue

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    if i < 4:
                        _y: pl.Scalar[pl.INDEX] = 0
                        break
                    else:
                        _y: pl.Scalar[pl.INDEX] = i
                        continue

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_for_loop_with_scope_and_if(self):
        """Complex nesting: for -> scope -> if with constraint propagation."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    with pl.incore():
                        if i < 4:
                            _y: pl.Scalar[pl.INDEX] = i // 4

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    with pl.incore():
                        if i < 4:
                            _y: pl.Scalar[pl.INDEX] = 0

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# No-op cases
# ============================================================================


class TestNoChange:
    def test_already_simplified(self):
        """An already-simple expression should not change."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for i in pl.range(8):
                    _y: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Before)

    def test_symbolic_loop_bounds(self):
        """Non-constant loop bounds: binding is skipped, identity simplification still works."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX]):
                for i in pl.range(n):
                    _y: pl.Scalar[pl.INDEX] = i + 0

        @pl.program
        class Expected:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX]):
                for i in pl.range(n):
                    _y: pl.Scalar[pl.INDEX] = i

        after = passes.simplify_expr()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_empty_function(self):
        """A function with no expressions should be unchanged."""
        body = ir.SeqStmts([], S)
        func = ir.Function("main", [], [], body, S)
        before = ir.Program([func], "test", S)

        after = passes.simplify_expr()(before)
        ir.assert_structural_equal(after, before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
