# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the Analyzer coordinator and ConstraintContext."""

import pytest
from pypto import DataType, ir
from pypto.arith import Analyzer, RewriteSimplifier

S = ir.Span.unknown()
INT = DataType.INT64
BOOL = DataType.BOOL


def make_var(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(INT), S)


def ci(value: int, dtype: DataType = INT) -> ir.ConstInt:
    return ir.ConstInt(value, dtype, S)


def assert_is_const_int(expr: ir.Expr, expected: int) -> None:
    assert isinstance(expr, ir.ConstInt), f"Expected ConstInt, got {type(expr).__name__}"
    assert expr.value == expected, f"Expected {expected}, got {expr.value}"


# ============================================================================
# Construction and basic usage
# ============================================================================


class TestConstruction:
    def test_create_analyzer(self):
        ana = Analyzer()
        assert ana is not None

    def test_sub_analyzers_accessible(self):
        ana = Analyzer()
        assert ana.const_int_bound is not None
        assert ana.modular_set is not None
        assert ana.rewrite_simplify is not None


# ============================================================================
# Bind (range overload)
# ============================================================================


class TestBindRange:
    def test_bind_range_propagates_to_const_int_bound(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 8)
        bound = ana.const_int_bound(x)
        assert bound.min_value == 0
        assert bound.max_value == 7

    def test_bind_range_single_value(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 5, 6)  # [5, 6) = exactly 5
        bound = ana.const_int_bound(x)
        assert bound.min_value == 5
        assert bound.max_value == 5
        # Should also substitute in rewrite simplifier
        result = ana.simplify(x)
        assert_is_const_int(result, 5)

    def test_bind_range_invalid_raises(self):
        ana = Analyzer()
        x = make_var("x")
        with pytest.raises(ValueError):
            ana.bind(x, 5, 5)  # empty range

    def test_bind_range_negative(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, -10, 0)
        bound = ana.const_int_bound(x)
        assert bound.min_value == -10
        assert bound.max_value == -1


# ============================================================================
# Bind (expression overload)
# ============================================================================


class TestBindExpr:
    def test_bind_expr_const(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, ci(42))
        bound = ana.const_int_bound(x)
        assert bound.min_value == 42
        assert bound.max_value == 42

    def test_bind_expr_variable(self):
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 10)
        ana.bind(y, ir.Add(x, ci(1), INT, S))
        bound = ana.const_int_bound(y)
        assert bound.min_value == 1
        assert bound.max_value == 10


# ============================================================================
# Simplify
# ============================================================================


class TestSimplify:
    def test_simplify_identity(self):
        ana = Analyzer()
        x = make_var("x")
        # x + 0 -> x
        result = ana.simplify(ir.Add(x, ci(0), INT, S))
        assert result is x

    def test_simplify_const_fold(self):
        ana = Analyzer()
        result = ana.simplify(ir.Add(ci(3), ci(4), INT, S))
        assert_is_const_int(result, 7)

    def test_simplify_x_minus_x(self):
        ana = Analyzer()
        x = make_var("x")
        result = ana.simplify(ir.Sub(x, x, INT, S))
        assert_is_const_int(result, 0)

    def test_simplify_with_bound_info(self):
        """Range-aware simplification: x // 8 -> 0 when x in [0, 8)."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 8)
        result = ana.simplify(ir.FloorDiv(x, ci(8), INT, S))
        assert_is_const_int(result, 0)

    def test_simplify_floormod_with_bound(self):
        """x % 8 -> x when x in [0, 8)."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 8)
        result = ana.simplify(ir.FloorMod(x, ci(8), INT, S))
        assert result is x

    def test_simplify_steps_parameter(self):
        ana = Analyzer()
        x = make_var("x")
        # steps=0 should return the expression unchanged
        expr = ir.Add(x, ci(0), INT, S)
        result = ana.simplify(expr, steps=0)
        assert result is expr

    def test_simplify_min_with_bound(self):
        """min(x, 10) -> x when x in [0, 8)."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 8)
        result = ana.simplify(ir.Min(x, ci(10), INT, S))
        assert result is x

    def test_simplify_max_with_bound(self):
        """max(x, -1) -> x when x in [0, 8)."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 8)
        result = ana.simplify(ir.Max(x, ci(-1), INT, S))
        assert result is x


# ============================================================================
# CanProve methods
# ============================================================================


class TestCanProve:
    def test_can_prove_greater_equal_true(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 10)
        assert ana.can_prove_greater_equal(x, 0) is True

    def test_can_prove_greater_equal_false(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 10)
        assert ana.can_prove_greater_equal(x, 1) is False

    def test_can_prove_less_true(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 10)
        assert ana.can_prove_less(x, 10) is True

    def test_can_prove_less_false(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 10)
        assert ana.can_prove_less(x, 9) is False

    def test_can_prove_equal_identity(self):
        ana = Analyzer()
        x = make_var("x")
        assert ana.can_prove_equal(x, x) is True

    def test_can_prove_equal_via_simplify(self):
        ana = Analyzer()
        x = make_var("x")
        lhs = ir.Add(x, ci(0), INT, S)
        assert ana.can_prove_equal(lhs, x) is True

    def test_can_prove_bool_const_true(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 10)
        # x < 10 is always true when x in [0, 10)
        cond = ir.Lt(x, ci(10), BOOL, S)
        assert ana.can_prove(cond) is True

    def test_can_prove_bool_const_false(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 10)
        # x >= 10 is always false when x in [0, 10)
        cond = ir.Ge(x, ci(10), BOOL, S)
        assert ana.can_prove(cond) is False

    def test_can_prove_greater_equal_with_expr(self):
        """x + 1 >= 1 when x in [0, 10)."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 10)
        expr = ir.Add(x, ci(1), INT, S)
        assert ana.can_prove_greater_equal(expr, 1) is True

    def test_can_prove_less_with_expr(self):
        """x + 1 < 11 when x in [0, 10)."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 10)
        expr = ir.Add(x, ci(1), INT, S)
        assert ana.can_prove_less(expr, 11) is True


# ============================================================================
# ConstraintContext
# ============================================================================


class TestConstraintContext:
    def test_context_manager(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, -10, 10)

        # Before constraint: cannot prove x >= 0
        assert ana.can_prove_greater_equal(x, 0) is False

        constraint = ir.Ge(x, ci(0), BOOL, S)
        with ana.constraint_context(constraint):
            # Within constraint: can prove x >= 0
            assert ana.can_prove_greater_equal(x, 0) is True

        # After constraint: back to original bounds
        assert ana.can_prove_greater_equal(x, 0) is False

    def test_constraint_tightens_bounds(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 100)

        constraint = ir.Lt(x, ci(10), BOOL, S)
        with ana.constraint_context(constraint):
            bound = ana.const_int_bound(x)
            assert bound.max_value == 9

        # Restored
        bound = ana.const_int_bound(x)
        assert bound.max_value == 99

    def test_nested_constraints(self):
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 100)

        with ana.constraint_context(ir.Ge(x, ci(10), BOOL, S)):
            assert ana.can_prove_greater_equal(x, 10) is True
            assert ana.can_prove_less(x, 20) is False

            with ana.constraint_context(ir.Lt(x, ci(20), BOOL, S)):
                assert ana.can_prove_greater_equal(x, 10) is True
                assert ana.can_prove_less(x, 20) is True

            # Inner scope exited
            assert ana.can_prove_less(x, 20) is False

        # Outer scope exited
        assert ana.can_prove_greater_equal(x, 10) is False

    def test_constraint_enables_simplification(self):
        """x // 8 -> 0 when constrained to x < 8 and x >= 0."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, -100, 100)

        # Without constraint, x // 8 is not simplified
        expr = ir.FloorDiv(x, ci(8), INT, S)
        result = ana.simplify(expr)
        assert not isinstance(result, ir.ConstInt)

        constraint = ir.And(
            ir.Ge(x, ci(0), BOOL, S),
            ir.Lt(x, ci(8), BOOL, S),
            BOOL,
            S,
        )
        with ana.constraint_context(constraint):
            result = ana.simplify(expr)
            assert_is_const_int(result, 0)

    # --- Multiple constraints on a single variable ---

    def test_successive_tightening_single_var(self):
        """Nested constraints progressively tighten a single variable's bounds."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 100)

        with ana.constraint_context(ir.Ge(x, ci(20), BOOL, S)):
            bound = ana.const_int_bound(x)
            assert bound.min_value == 20
            assert bound.max_value == 99

            with ana.constraint_context(ir.Lt(x, ci(50), BOOL, S)):
                bound = ana.const_int_bound(x)
                assert bound.min_value == 20
                assert bound.max_value == 49

                with ana.constraint_context(ir.Ge(x, ci(30), BOOL, S)):
                    bound = ana.const_int_bound(x)
                    assert bound.min_value == 30
                    assert bound.max_value == 49

                # Innermost exited: back to [20, 49]
                bound = ana.const_int_bound(x)
                assert bound.min_value == 20
                assert bound.max_value == 49

            # Middle exited: back to [20, 99]
            bound = ana.const_int_bound(x)
            assert bound.min_value == 20
            assert bound.max_value == 99

        # All exited: back to [0, 99]
        bound = ana.const_int_bound(x)
        assert bound.min_value == 0
        assert bound.max_value == 99

    def test_and_constraint_single_var(self):
        """A single And constraint tightens both ends at once."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, -100, 100)

        constraint = ir.And(
            ir.Ge(x, ci(10), BOOL, S),
            ir.Lt(x, ci(20), BOOL, S),
            BOOL,
            S,
        )
        with ana.constraint_context(constraint):
            bound = ana.const_int_bound(x)
            assert bound.min_value == 10
            assert bound.max_value == 19

        bound = ana.const_int_bound(x)
        assert bound.min_value == -100
        assert bound.max_value == 99

    def test_eq_constraint_single_var(self):
        """Eq constraint pins a variable to a single value."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 0, 100)

        with ana.constraint_context(ir.Eq(x, ci(42), BOOL, S)):
            bound = ana.const_int_bound(x)
            assert bound.min_value == 42
            assert bound.max_value == 42
            assert ana.can_prove_equal(x, ci(42)) is True

        bound = ana.const_int_bound(x)
        assert bound.min_value == 0
        assert bound.max_value == 99

    def test_redundant_constraint_single_var(self):
        """A constraint weaker than the existing bound is a no-op."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 10, 20)  # [10, 19]

        # x >= 0 is weaker than existing min=10
        with ana.constraint_context(ir.Ge(x, ci(0), BOOL, S)):
            bound = ana.const_int_bound(x)
            assert bound.min_value == 10  # Not weakened
            assert bound.max_value == 19

    def test_simplify_enabled_by_constraint_single_var(self):
        """Constraint makes floordiv(x, x) -> 1 by proving x != 0."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, -100, 100)

        # Without constraint: x could be 0, so floordiv(x, x) is not simplified
        expr = ir.FloorDiv(x, x, INT, S)
        result = ana.simplify(expr)
        assert not isinstance(result, ir.ConstInt)

        # With constraint x >= 1, floordiv(x, x) -> 1
        with ana.constraint_context(ir.Ge(x, ci(1), BOOL, S)):
            result = ana.simplify(expr)
            assert_is_const_int(result, 1)

    # --- Multiple constraints on multiple variables ---

    def test_independent_constraints_multi_var(self):
        """Constraints on different variables are independent."""
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)

        with ana.constraint_context(ir.Lt(x, ci(10), BOOL, S)):
            # x is [0, 9], y is still [0, 99]
            assert ana.can_prove_less(x, 10) is True
            assert ana.can_prove_less(y, 10) is False

            with ana.constraint_context(ir.Ge(y, ci(50), BOOL, S)):
                # x is [0, 9], y is [50, 99]
                assert ana.can_prove_less(x, 10) is True
                assert ana.can_prove_greater_equal(y, 50) is True

            # y restored
            assert ana.can_prove_greater_equal(y, 50) is False

        # x restored
        assert ana.can_prove_less(x, 10) is False

    def test_and_constraint_multi_var(self):
        """A single And constraint can tighten multiple variables."""
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)

        constraint = ir.And(
            ir.Ge(x, ci(10), BOOL, S),
            ir.Lt(y, ci(20), BOOL, S),
            BOOL,
            S,
        )
        with ana.constraint_context(constraint):
            assert ana.can_prove_greater_equal(x, 10) is True
            assert ana.can_prove_less(y, 20) is True

        assert ana.can_prove_greater_equal(x, 10) is False
        assert ana.can_prove_less(y, 20) is False

    def test_constraint_enables_min_max_multi_var(self):
        """Constraints on two variables enable min/max simplification."""
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)

        # Without constraints, min(x, y) cannot be simplified
        expr = ir.Min(x, y, INT, S)
        result = ana.simplify(expr)
        assert isinstance(result, ir.Min)

        # Constrain x < 10 and y >= 50 → min(x, y) = x
        constraint = ir.And(
            ir.Lt(x, ci(10), BOOL, S),
            ir.Ge(y, ci(50), BOOL, S),
            BOOL,
            S,
        )
        with ana.constraint_context(constraint):
            result = ana.simplify(expr)
            assert result is x

    def test_constraint_proves_cross_var_comparison(self):
        """Constraints on two variables let us prove cross-variable comparisons."""
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)

        # x + y < 20 when both are constrained to < 10
        constraint = ir.And(
            ir.Lt(x, ci(10), BOOL, S),
            ir.Lt(y, ci(10), BOOL, S),
            BOOL,
            S,
        )
        with ana.constraint_context(constraint):
            expr = ir.Add(x, y, INT, S)
            assert ana.can_prove_less(expr, 19) is True
            assert ana.can_prove_greater_equal(expr, 0) is True

    def test_triple_and_constraint(self):
        """Three-way And constraint (nested And nodes)."""
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)
        ana.bind(z, 0, 100)

        # (x >= 10 && y >= 20) && z >= 30
        constraint = ir.And(
            ir.And(
                ir.Ge(x, ci(10), BOOL, S),
                ir.Ge(y, ci(20), BOOL, S),
                BOOL,
                S,
            ),
            ir.Ge(z, ci(30), BOOL, S),
            BOOL,
            S,
        )
        with ana.constraint_context(constraint):
            assert ana.can_prove_greater_equal(x, 10) is True
            assert ana.can_prove_greater_equal(y, 20) is True
            assert ana.can_prove_greater_equal(z, 30) is True

        # All restored
        assert ana.can_prove_greater_equal(x, 10) is False
        assert ana.can_prove_greater_equal(y, 20) is False
        assert ana.can_prove_greater_equal(z, 30) is False

    def test_nested_multi_var_restore_order(self):
        """Verify that nested scopes on different variables restore correctly."""
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)

        with ana.constraint_context(ir.Ge(x, ci(50), BOOL, S)):
            with ana.constraint_context(ir.Ge(y, ci(60), BOOL, S)):
                assert ana.const_int_bound(x).min_value == 50
                assert ana.const_int_bound(y).min_value == 60

            # y restored, x still constrained
            assert ana.const_int_bound(x).min_value == 50
            assert ana.const_int_bound(y).min_value == 0

        # Both restored
        assert ana.const_int_bound(x).min_value == 0
        assert ana.const_int_bound(y).min_value == 0


# ============================================================================
# Cross-analyzer integration (TryCompare)
# ============================================================================


class TestCrossAnalyzer:
    def test_trycompare_enables_floordiv_simplification(self):
        """With parent Analyzer, rewrite rules use bound info for floordiv(x, x) -> 1."""
        ana = Analyzer()
        x = make_var("x")
        ana.bind(x, 1, 10)  # x is positive (non-zero)
        result = ana.simplify(ir.FloorDiv(x, x, INT, S))
        assert_is_const_int(result, 1)

    def test_trycompare_enables_min_simplification(self):
        """min(x, y) -> x when x <= y is provable."""
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 5)
        ana.bind(y, 10, 20)
        result = ana.simplify(ir.Min(x, y, INT, S))
        assert result is x

    def test_trycompare_enables_max_simplification(self):
        """max(x, y) -> y when x <= y is provable."""
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 5)
        ana.bind(y, 10, 20)
        result = ana.simplify(ir.Max(x, y, INT, S))
        assert result is y

    def test_standalone_simplifier_no_bound_awareness(self):
        """Standalone RewriteSimplifier cannot simplify min/max based on bounds."""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        # Without parent Analyzer, min(x, y) stays as-is
        result = s(ir.Min(x, y, INT, S))
        assert isinstance(result, ir.Min)

    def test_multi_variable_bounds(self):
        ana = Analyzer()
        x = make_var("x")
        y = make_var("y")
        ana.bind(x, 0, 8)
        ana.bind(y, 0, 8)
        # x + y is in [0, 14], so x + y >= 0
        expr = ir.Add(x, y, INT, S)
        assert ana.can_prove_greater_equal(expr, 0) is True
        assert ana.can_prove_less(expr, 15) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
