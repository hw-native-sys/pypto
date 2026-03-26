# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for the arith module.

These tests exercise cross-analyzer interactions and end-to-end scenarios
from tracking issue #668, including tiling patterns, multi-step simplification,
and the CanProve fallback chain (const_int_bound -> transitive_cmp -> int_set).
"""

import pytest
from pypto import DataType, ir
from pypto.arith import Analyzer

S = ir.Span.unknown()
INT = DataType.INT64
BOOL = DataType.BOOL

x = ir.Var("x", ir.ScalarType(INT), S)
y = ir.Var("y", ir.ScalarType(INT), S)
z = ir.Var("z", ir.ScalarType(INT), S)
n = ir.Var("n", ir.ScalarType(INT), S)


@pytest.fixture
def ana() -> Analyzer:
    """Fresh Analyzer per test to avoid state leakage."""
    return Analyzer()


def ci(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, INT, S)


def assert_is_const_int(expr: ir.Expr, expected: int) -> None:
    assert isinstance(expr, ir.ConstInt), f"Expected ConstInt, got {type(expr).__name__}"
    assert expr.value == expected, f"Expected {expected}, got {expr.value}"


def assert_same_expr(result: ir.Expr, expected: ir.Expr) -> None:
    ir.assert_structural_equal(result, expected)


# ============================================================================
# Tiling patterns (motivating examples from issue #668)
# ============================================================================


class TestTilingPatterns:
    """End-to-end tiling index simplification through the full Analyzer.

    The PEqualChecker<ExprPtr> uses value-based comparison for ConstInt nodes,
    so separate ci(8) calls are correctly matched as equal by the pattern matcher.
    """

    def test_div_mod_recombination(self, ana: Analyzer):
        """(i // 8) * 8 + i % 8 -> i with separate constant objects."""
        div_part = ir.Mul(ir.FloorDiv(x, ci(8), INT, S), ci(8), INT, S)
        mod_part = ir.FloorMod(x, ci(8), INT, S)
        result = ana.simplify(ir.Add(div_part, mod_part, INT, S))
        assert_same_expr(result, x)

    def test_div_mod_recombination_reversed(self, ana: Analyzer):
        """i % 8 + (i // 8) * 8 -> i (reversed operand order)."""
        mod_part = ir.FloorMod(x, ci(8), INT, S)
        div_part = ir.Mul(ir.FloorDiv(x, ci(8), INT, S), ci(8), INT, S)
        result = ana.simplify(ir.Add(mod_part, div_part, INT, S))
        assert_same_expr(result, x)

    def test_tiling_floordiv_zero(self, ana: Analyzer):
        """i // 8 -> 0 when i in [0, 8), via Analyzer with bound."""
        ana.bind(x, 0, 8)
        result = ana.simplify(ir.FloorDiv(x, ci(8), INT, S))
        assert_is_const_int(result, 0)

    def test_tiling_floormod_identity(self, ana: Analyzer):
        """i % 8 -> i when i in [0, 8), via Analyzer with bound."""
        ana.bind(x, 0, 8)
        result = ana.simplify(ir.FloorMod(x, ci(8), INT, S))
        assert_same_expr(result, x)

    def test_two_level_tiling(self, ana: Analyzer):
        """Two-level tiling: (i // 8) * 8 + i % 8 -> i."""
        ana.bind(x, 0, 64)
        outer = ir.FloorDiv(x, ci(8), INT, S)
        inner = ir.FloorMod(x, ci(8), INT, S)
        reconstructed = ir.Add(ir.Mul(outer, ci(8), INT, S), inner, INT, S)
        result = ana.simplify(reconstructed)
        assert_same_expr(result, x)

    def test_tiling_with_offset(self, ana: Analyzer):
        """(i // 4) * 4 + i % 4 + base -> i + base."""
        ana.bind(x, 0, 64)
        div_part = ir.Mul(ir.FloorDiv(x, ci(4), INT, S), ci(4), INT, S)
        mod_part = ir.FloorMod(x, ci(4), INT, S)
        recombined = ir.Add(div_part, mod_part, INT, S)
        with_offset = ir.Add(recombined, ci(100), INT, S)
        result = ana.simplify(with_offset)
        expected = ir.Add(x, ci(100), INT, S)
        assert_same_expr(result, expected)

    def test_different_divisor_values_no_recombination(self, ana: Analyzer):
        """(i // 8) * 4 + i % 8 should NOT recombine (divisor mismatch: 8 vs 4)."""
        div_part = ir.Mul(ir.FloorDiv(x, ci(8), INT, S), ci(4), INT, S)
        mod_part = ir.FloorMod(x, ci(8), INT, S)
        result = ana.simplify(ir.Add(div_part, mod_part, INT, S))
        # Should not simplify to x — divisor in mul (4) != divisor in floordiv (8)
        assert not isinstance(result, ir.Var)


# ============================================================================
# Multi-step canonical + rewrite via Analyzer.simplify()
# ============================================================================


class TestMultiStepSimplification:
    """Simplifications requiring both canonical and rewrite passes."""

    def test_coefficient_collection(self, ana: Analyzer):
        """x * 2 + x -> 3 * x via Analyzer.simplify()."""
        expr = ir.Add(ir.Mul(x, ci(2), INT, S), x, INT, S)
        result = ana.simplify(expr)
        expected = ir.Mul(x, ci(3), INT, S)
        assert_same_expr(result, expected)

    def test_coefficient_collection_reversed(self, ana: Analyzer):
        """x + x * 2 -> 3 * x."""
        expr = ir.Add(x, ir.Mul(x, ci(2), INT, S), INT, S)
        result = ana.simplify(expr)
        expected = ir.Mul(x, ci(3), INT, S)
        assert_same_expr(result, expected)

    def test_multi_variable_cancellation(self, ana: Analyzer):
        """x + y - x -> y via Analyzer.simplify()."""
        expr = ir.Sub(ir.Add(x, y, INT, S), x, INT, S)
        result = ana.simplify(expr)
        assert_same_expr(result, y)

    def test_nested_add_constant_collection(self, ana: Analyzer):
        """(x + 3) + 5 -> x + 8."""
        expr = ir.Add(ir.Add(x, ci(3), INT, S), ci(5), INT, S)
        result = ana.simplify(expr)
        expected = ir.Add(x, ci(8), INT, S)
        assert_same_expr(result, expected)

    def test_distribute_and_collect(self, ana: Analyzer):
        """(x + 1) * 2 -> 2*x + 2."""
        expr = ir.Mul(ir.Add(x, ci(1), INT, S), ci(2), INT, S)
        result = ana.simplify(expr)
        expected = ir.Add(ir.Mul(x, ci(2), INT, S), ci(2), INT, S)
        assert_same_expr(result, expected)

    def test_simplify_steps_1_vs_2(self, ana: Analyzer):
        """More steps can yield further simplification."""
        expr = ir.Sub(x, x, INT, S)
        result1 = ana.simplify(expr, steps=1)
        assert_is_const_int(result1, 0)
        result2 = ana.simplify(expr, steps=2)
        assert_is_const_int(result2, 0)


# ============================================================================
# CanProve fallback chain
# ============================================================================


class TestCanProveFallbackChain:
    """Verify CanProve uses the full chain: const_int_bound -> transitive_cmp -> int_set."""

    def test_can_prove_via_const_int_bound(self, ana: Analyzer):
        """Simple case: concrete bounds suffice."""
        ana.bind(x, 0, 10)
        assert ana.can_prove(ir.Lt(x, ci(10), BOOL, S))
        assert ana.can_prove(ir.Ge(x, ci(0), BOOL, S))

    def test_can_prove_via_transitive_cmp(self, ana: Analyzer):
        """Transitive chain: x < y, y < z => x < z, not provable by bounds alone."""
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)
        ana.bind(z, 0, 100)
        with ana.constraint_context(ir.Lt(x, y, BOOL, S)):
            with ana.constraint_context(ir.Lt(y, z, BOOL, S)):
                assert ana.can_prove(ir.Lt(x, z, BOOL, S))

    def test_can_prove_via_int_set_symbolic(self, ana: Analyzer):
        """Symbolic: x in [0, n) => x < n, requires int_set fallback."""
        ana.bind(n, 1, 1000)
        ana.int_set.bind(x, ci(0), n)
        assert ana.can_prove(ir.Lt(x, n, BOOL, S))

    def test_can_prove_symbolic_ge(self, ana: Analyzer):
        """x in [a, ...) => x >= a via int_set."""
        ana.bind(n, 0, 100)
        ana.int_set.bind(x, n, ci(1000))
        assert ana.can_prove(ir.Ge(x, n, BOOL, S))

    def test_can_prove_with_constraint_and_bounds(self, ana: Analyzer):
        """Combine bound + constraint: x in [0, 100), constrain x < 10, prove x < 10."""
        ana.bind(x, 0, 100)
        with ana.constraint_context(ir.Lt(x, ci(10), BOOL, S)):
            assert ana.can_prove(ir.Lt(x, ci(10), BOOL, S))
            assert ana.can_prove(ir.Lt(x, ci(11), BOOL, S))

    def test_can_prove_expression_with_bounds(self, ana: Analyzer):
        """x in [0, 8) => x + 1 < 9."""
        ana.bind(x, 0, 8)
        expr = ir.Add(x, ci(1), INT, S)
        assert ana.can_prove(ir.Lt(expr, ci(9), BOOL, S))


# ============================================================================
# Constraint-enabled simplification
# ============================================================================


class TestConstraintSimplification:
    """Constraints enabling simplifications that bounds alone cannot."""

    def test_constraint_enables_floordiv_simplification(self, ana: Analyzer):
        """Constrain x in [0, 8) within wider bound, then x // 8 -> 0."""
        ana.bind(x, -100, 100)
        expr = ir.FloorDiv(x, ci(8), INT, S)
        result = ana.simplify(expr)
        assert not isinstance(result, ir.ConstInt)

        cond = ir.And(ir.Ge(x, ci(0), BOOL, S), ir.Lt(x, ci(8), BOOL, S), BOOL, S)
        with ana.constraint_context(cond):
            result = ana.simplify(expr)
            assert_is_const_int(result, 0)

    def test_constraint_enables_min_simplification(self, ana: Analyzer):
        """Constrain x < 10 and y >= 50 => min(x, y) -> x."""
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)
        cond = ir.And(
            ir.Lt(x, ci(10), BOOL, S),
            ir.Ge(y, ci(50), BOOL, S),
            BOOL,
            S,
        )
        with ana.constraint_context(cond):
            result = ana.simplify(ir.Min(x, y, INT, S))
            assert_same_expr(result, x)

    def test_constraint_enables_max_simplification(self, ana: Analyzer):
        """Constrain x < 10 and y >= 50 => max(x, y) -> y."""
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)
        cond = ir.And(
            ir.Lt(x, ci(10), BOOL, S),
            ir.Ge(y, ci(50), BOOL, S),
            BOOL,
            S,
        )
        with ana.constraint_context(cond):
            result = ana.simplify(ir.Max(x, y, INT, S))
            assert_same_expr(result, y)

    def test_nested_constraint_progressive_tightening(self, ana: Analyzer):
        """Nested constraints progressively tighten, each enabling more simplification."""
        ana.bind(x, 0, 100)
        expr = ir.FloorDiv(x, ci(16), INT, S)
        result = ana.simplify(expr)
        assert not isinstance(result, ir.ConstInt)

        with ana.constraint_context(ir.Lt(x, ci(16), BOOL, S)):
            result = ana.simplify(expr)
            assert_is_const_int(result, 0)


# ============================================================================
# Cross-analyzer: int_set + transitive_cmp
# ============================================================================


class TestCrossAnalyzerIntSetTransitive:
    """Test int_set and transitive_cmp working together via CanProve."""

    def test_transitive_constraint_tightens_int_set(self, ana: Analyzer):
        """Constraint x < y propagated to int_set bounds."""
        ana.bind(x, 0, 100)
        ana.bind(y, 0, 100)
        with ana.constraint_context(ir.Lt(x, y, BOOL, S)):
            s = ana.int_set(x)
            assert s.max_value is not None

    def test_transitive_then_can_prove_symbolic(self, ana: Analyzer):
        """With transitive knowledge and symbolic int_set, can prove complex relations."""
        ana.bind(n, 10, 100)
        ana.bind(x, 0, 100)
        with ana.constraint_context(ir.Lt(x, n, BOOL, S)):
            assert ana.can_prove(ir.Lt(x, n, BOOL, S))


# ============================================================================
# Cross-analyzer: modular_set + bounds in simplification
# ============================================================================


class TestCrossAnalyzerModularBounds:
    """Modular properties combined with bounds for simplification."""

    def test_even_variable_floordiv(self, ana: Analyzer):
        """When x is known even and non-negative, simplification may leverage both."""
        ana.bind(x, 0, 100)
        expr = ir.FloorMod(ir.Mul(x, ci(2), INT, S), ci(2), INT, S)
        result = ana.simplify(expr)
        assert_is_const_int(result, 0)

    def test_modular_floordiv_simplification(self, ana: Analyzer):
        """(4 * x) // 4 -> x (modular + canonical)."""
        expr = ir.FloorDiv(ir.Mul(ci(4), x, INT, S), ci(4), INT, S)
        result = ana.simplify(expr)
        assert_same_expr(result, x)

    def test_modular_with_offset(self, ana: Analyzer):
        """(4 * x + 2) // 4 -> x, (4 * x + 2) % 4 -> 2."""
        base = ir.Add(ir.Mul(ci(4), x, INT, S), ci(2), INT, S)
        div_result = ana.simplify(ir.FloorDiv(base, ci(4), INT, S))
        assert_same_expr(div_result, x)
        mod_result = ana.simplify(ir.FloorMod(base, ci(4), INT, S))
        assert_is_const_int(mod_result, 2)


# ============================================================================
# Bound propagation through complex expressions
# ============================================================================


class TestBoundPropagation:
    """Bounds propagating through complex expression trees."""

    def test_add_bounds_propagation(self, ana: Analyzer):
        """x in [0, 8), y in [0, 4) => x + y in [0, 11]."""
        ana.bind(x, 0, 8)
        ana.bind(y, 0, 4)
        assert ana.can_prove_greater_equal(ir.Add(x, y, INT, S), 0)
        assert ana.can_prove_less(ir.Add(x, y, INT, S), 12)

    def test_mul_bounds_propagation(self, ana: Analyzer):
        """x in [0, 8) => x * 2 in [0, 14]."""
        ana.bind(x, 0, 8)
        expr = ir.Mul(x, ci(2), INT, S)
        assert ana.can_prove_greater_equal(expr, 0)
        assert ana.can_prove_less(expr, 15)

    def test_sub_bounds_propagation(self, ana: Analyzer):
        """x in [5, 10), y in [0, 3) => x - y in [3, 9]."""
        ana.bind(x, 5, 10)
        ana.bind(y, 0, 3)
        expr = ir.Sub(x, y, INT, S)
        assert ana.can_prove_greater_equal(expr, 3)
        assert ana.can_prove_less(expr, 10)

    def test_nested_expression_bounds(self, ana: Analyzer):
        """x in [0, 8) => (x + 1) * 2 in [2, 16]."""
        ana.bind(x, 0, 8)
        expr = ir.Mul(ir.Add(x, ci(1), INT, S), ci(2), INT, S)
        assert ana.can_prove_greater_equal(expr, 2)
        assert ana.can_prove_less(expr, 17)

    def test_floordiv_bounds(self, ana: Analyzer):
        """x in [0, 64) => x // 8 in [0, 7]."""
        ana.bind(x, 0, 64)
        expr = ir.FloorDiv(x, ci(8), INT, S)
        assert ana.can_prove_greater_equal(expr, 0)
        assert ana.can_prove_less(expr, 8)


# ============================================================================
# Realistic multi-variable scenarios
# ============================================================================


class TestRealisticScenarios:
    """Realistic patterns from loop transformations and tiling."""

    def test_split_loop_index_reconstruction(self, ana: Analyzer):
        """After splitting loop i into outer*4 + inner, reconstruct i.

        outer in [0, 4), inner in [0, 4) => outer * 4 + inner in [0, 15]
        (outer * 4 + inner) // 4 -> outer
        (outer * 4 + inner) % 4 -> inner
        """
        ana.bind(x, 0, 4)  # outer
        ana.bind(y, 0, 4)  # inner
        combined = ir.Add(ir.Mul(x, ci(4), INT, S), y, INT, S)

        div_result = ana.simplify(ir.FloorDiv(combined, ci(4), INT, S))
        assert_same_expr(div_result, x)

        mod_result = ana.simplify(ir.FloorMod(combined, ci(4), INT, S))
        assert_same_expr(mod_result, y)

    def test_constraint_in_if_branch_like_scenario(self, ana: Analyzer):
        """Simulate if-then-else: different simplifications in each branch.

        x in [0, 16): if x < 8 then x // 8 -> 0, else x // 16 -> 0.
        """
        ana.bind(x, 0, 16)

        with ana.constraint_context(ir.Lt(x, ci(8), BOOL, S)):
            result = ana.simplify(ir.FloorDiv(x, ci(8), INT, S))
            assert_is_const_int(result, 0)

        with ana.constraint_context(ir.Ge(x, ci(8), BOOL, S)):
            result = ana.simplify(ir.FloorDiv(x, ci(16), INT, S))
            assert_is_const_int(result, 0)

    def test_multi_variable_bound_proving(self, ana: Analyzer):
        """Two bounded vars: x in [0, 8), y in [0, 8) => x + y < 15."""
        ana.bind(x, 0, 8)
        ana.bind(y, 0, 8)
        assert ana.can_prove(ir.Lt(ir.Add(x, y, INT, S), ci(15), BOOL, S))
        assert not ana.can_prove(ir.Lt(ir.Add(x, y, INT, S), ci(14), BOOL, S))


# ============================================================================
# Exception safety and scope restoration
# ============================================================================


class TestScopeRestoration:
    """Verify constraint context properly restores state, even on exceptions."""

    def test_exception_in_constraint_restores(self, ana: Analyzer):
        """If an exception occurs inside a constraint context, state is restored."""
        ana.bind(x, 0, 100)
        try:
            with ana.constraint_context(ir.Lt(x, ci(10), BOOL, S)):
                assert ana.can_prove_less(x, 10)
                raise ValueError("intentional error")
        except ValueError:
            pass
        assert not ana.can_prove_less(x, 10)

    def test_deeply_nested_constraint_restore(self, ana: Analyzer):
        """4-level nesting restores correctly."""
        ana.bind(x, 0, 1000)
        with ana.constraint_context(ir.Ge(x, ci(10), BOOL, S)):
            with ana.constraint_context(ir.Lt(x, ci(100), BOOL, S)):
                with ana.constraint_context(ir.Ge(x, ci(50), BOOL, S)):
                    bound = ana.const_int_bound(x)
                    assert bound.min_value == 50
                    assert bound.max_value == 99
                bound = ana.const_int_bound(x)
                assert bound.min_value == 10
                assert bound.max_value == 99
            bound = ana.const_int_bound(x)
            assert bound.min_value == 10
            assert bound.max_value == 999
        bound = ana.const_int_bound(x)
        assert bound.min_value == 0
        assert bound.max_value == 999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
