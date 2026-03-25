# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for IntSetAnalyzer (symbolic interval bound propagation)."""

import pytest
from pypto import DataType, ir
from pypto.arith import Analyzer, IntSet, IntSetAnalyzer

S = ir.Span.unknown()
INT = DataType.INT64
BOOL = DataType.BOOL

x = ir.Var("x", ir.ScalarType(INT), S)
y = ir.Var("y", ir.ScalarType(INT), S)
n = ir.Var("n", ir.ScalarType(INT), S)
a = ir.Var("a", ir.ScalarType(INT), S)
b = ir.Var("b", ir.ScalarType(INT), S)
c = ir.Var("c", ir.ScalarType(INT), S)
d = ir.Var("d", ir.ScalarType(INT), S)


def ci(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, INT, S)


def assert_is_const_int(expr: ir.Expr | None, expected: int) -> None:
    assert expr is not None, "Expected ConstInt, got None (unbounded)"
    assert isinstance(expr, ir.ConstInt), f"Expected ConstInt, got {type(expr).__name__}"
    assert expr.value == expected, f"Expected {expected}, got {expr.value}"


# ============================================================================
# IntSet struct basics
# ============================================================================


class TestIntSetStruct:
    def test_everything(self):
        s = IntSet.everything()
        assert s.is_everything()
        assert not s.is_single_point()
        assert not s.is_nothing()
        assert s.min_value is None
        assert s.max_value is None

    def test_single_point(self):
        s = IntSet.single_point(ci(5))
        assert not s.is_everything()
        assert s.is_single_point()
        assert_is_const_int(s.min_value, 5)
        assert_is_const_int(s.max_value, 5)

    def test_interval(self):
        s = IntSet.interval(ci(1), ci(10))
        assert not s.is_everything()
        assert not s.is_single_point()
        assert_is_const_int(s.min_value, 1)
        assert_is_const_int(s.max_value, 10)

    def test_interval_with_none(self):
        s = IntSet(None, ci(10))
        assert not s.is_everything()
        assert s.min_value is None
        assert_is_const_int(s.max_value, 10)

    def test_repr_const(self):
        assert "42" in repr(IntSet.single_point(ci(42)))

    def test_repr_everything(self):
        assert "None" in repr(IntSet.everything())

    def test_repr_var(self):
        assert "x" in repr(IntSet.single_point(x))


# ============================================================================
# Standalone IntSetAnalyzer
# ============================================================================


class TestStandaloneIntSet:
    def test_const_int(self):
        analyzer = IntSetAnalyzer()
        s = analyzer(ci(7))
        assert s.is_single_point()
        assert_is_const_int(s.min_value, 7)

    def test_const_int_negative(self):
        analyzer = IntSetAnalyzer()
        s = analyzer(ci(-3))
        assert s.is_single_point()
        assert_is_const_int(s.min_value, -3)

    def test_unknown_var(self):
        analyzer = IntSetAnalyzer()
        s = analyzer(x)
        assert s.is_single_point()
        assert s.min_value is x

    def test_bound_var(self):
        analyzer = IntSetAnalyzer()
        analyzer.bind(x, ci(0), ci(8))  # [0, 8) = [0, 7]
        s = analyzer(x)
        assert_is_const_int(s.min_value, 0)
        assert_is_const_int(s.max_value, 7)

    def test_update_var(self):
        analyzer = IntSetAnalyzer()
        analyzer.update(x, IntSet.interval(ci(2), ci(5)))
        s = analyzer(x)
        assert_is_const_int(s.min_value, 2)
        assert_is_const_int(s.max_value, 5)


# ============================================================================
# Arithmetic propagation (concrete ranges)
# ============================================================================


class TestIntSetArithmetic:
    def test_add(self):
        ana = Analyzer()
        ana.bind(x, 0, 8)  # [0, 7]
        ana.bind(y, 0, 4)  # [0, 3]
        s = ana.int_set(ir.Add(x, y, INT, S))
        assert_is_const_int(s.min_value, 0)
        assert_is_const_int(s.max_value, 10)

    def test_sub(self):
        ana = Analyzer()
        ana.bind(x, 5, 11)  # [5, 10]
        ana.bind(y, 1, 4)  # [1, 3]
        s = ana.int_set(ir.Sub(x, y, INT, S))
        assert_is_const_int(s.min_value, 2)
        assert_is_const_int(s.max_value, 9)

    def test_mul_positive(self):
        ana = Analyzer()
        ana.bind(x, 2, 6)  # [2, 5]
        s = ana.int_set(ir.Mul(x, ci(3), INT, S))
        assert_is_const_int(s.min_value, 6)
        assert_is_const_int(s.max_value, 15)

    def test_mul_negative_multiplier(self):
        ana = Analyzer()
        ana.bind(x, 2, 6)  # [2, 5]
        s = ana.int_set(ir.Mul(x, ci(-2), INT, S))
        assert_is_const_int(s.min_value, -10)
        assert_is_const_int(s.max_value, -4)

    def test_mul_both_positive_intervals(self):
        ana = Analyzer()
        ana.bind(x, 2, 5)  # [2, 4]
        ana.bind(y, 3, 7)  # [3, 6]
        s = ana.int_set(ir.Mul(x, y, INT, S))
        assert_is_const_int(s.min_value, 6)
        assert_is_const_int(s.max_value, 24)

    def test_min(self):
        ana = Analyzer()
        ana.bind(x, 0, 11)  # [0, 10]
        ana.bind(y, 5, 16)  # [5, 15]
        s = ana.int_set(ir.Min(x, y, INT, S))
        assert_is_const_int(s.min_value, 0)
        assert_is_const_int(s.max_value, 10)

    def test_max(self):
        ana = Analyzer()
        ana.bind(x, 0, 11)  # [0, 10]
        ana.bind(y, 5, 16)  # [5, 15]
        s = ana.int_set(ir.Max(x, y, INT, S))
        assert_is_const_int(s.min_value, 5)
        assert_is_const_int(s.max_value, 15)

    def test_neg(self):
        ana = Analyzer()
        ana.bind(x, 3, 8)  # [3, 7]
        s = ana.int_set(ir.Neg(x, INT, S))
        assert_is_const_int(s.min_value, -7)
        assert_is_const_int(s.max_value, -3)


# ============================================================================
# FloorDiv and FloorMod
# ============================================================================


class TestIntSetDivMod:
    def test_floordiv_positive(self):
        ana = Analyzer()
        ana.bind(x, 0, 16)  # [0, 15]
        s = ana.int_set(ir.FloorDiv(x, ci(4), INT, S))
        assert_is_const_int(s.min_value, 0)
        assert_is_const_int(s.max_value, 3)

    def test_floordiv_non_multiple(self):
        ana = Analyzer()
        ana.bind(x, 0, 10)  # [0, 9]
        s = ana.int_set(ir.FloorDiv(x, ci(4), INT, S))
        assert_is_const_int(s.min_value, 0)
        assert_is_const_int(s.max_value, 2)

    def test_floormod_positive(self):
        ana = Analyzer()
        ana.bind(x, 0, 100)  # [0, 99]
        s = ana.int_set(ir.FloorMod(x, ci(8), INT, S))
        assert_is_const_int(s.min_value, 0)
        assert_is_const_int(s.max_value, 7)


# ============================================================================
# Symbolic bounds (the key feature)
# ============================================================================


class TestSymbolicBounds:
    def test_symbolic_upper_bound(self):
        """x in [0, n) => x + 1 in [1, n]."""
        ana = Analyzer()
        ana.int_set.bind(x, ci(0), n)  # x in [0, n) = [0, n-1]
        s = ana.int_set(ir.Add(x, ci(1), INT, S))
        assert_is_const_int(s.min_value, 1)
        assert s.max_value is not None

    def test_symbolic_range_add(self):
        """x in [a, b), y in [c, d) => x + y has symbolic bounds."""
        analyzer = IntSetAnalyzer()
        analyzer.bind(x, a, b)
        analyzer.bind(y, c, d)
        s = analyzer(ir.Add(x, y, INT, S))
        assert s.min_value is not None
        assert s.max_value is not None

    def test_symbolic_sub(self):
        """x in [0, n) => n - 1 - x has symbolic bounds."""
        ana = Analyzer()
        ana.bind(n, 1, 100)
        ana.int_set.bind(x, ci(0), n)
        n_minus_1 = ir.Sub(n, ci(1), INT, S)
        s = ana.int_set(ir.Sub(n_minus_1, x, INT, S))
        assert s.min_value is not None
        assert s.max_value is not None


# ============================================================================
# Analyzer integration
# ============================================================================


class TestAnalyzerIntegration:
    def test_analyzer_bind_propagates_to_int_set(self):
        ana = Analyzer()
        ana.bind(x, 0, 8)
        s = ana.int_set(x)
        assert_is_const_int(s.min_value, 0)
        assert_is_const_int(s.max_value, 7)

    def test_analyzer_bind_expr_propagates_to_int_set(self):
        ana = Analyzer()
        ana.bind(x, ci(42))
        s = ana.int_set(x)
        assert s.is_single_point()
        assert_is_const_int(s.min_value, 42)

    def test_int_set_explicit_bind(self):
        ana = Analyzer()
        ana.int_set.bind(x, ci(0), ci(8))
        s = ana.int_set(x)
        assert_is_const_int(s.min_value, 0)
        assert_is_const_int(s.max_value, 7)


# ============================================================================
# Constraint context
# ============================================================================


class TestIntSetConstraint:
    def test_constraint_ge_tightens(self):
        ana = Analyzer()
        ana.int_set.update(x, IntSet.everything())
        with ana.constraint_context(ir.Ge(x, ci(5), BOOL, S)):
            s = ana.int_set(x)
            assert_is_const_int(s.min_value, 5)

    def test_constraint_lt_tightens(self):
        ana = Analyzer()
        ana.bind(x, 0, 100)
        with ana.constraint_context(ir.Lt(x, ci(10), BOOL, S)):
            s = ana.int_set(x)
            assert_is_const_int(s.max_value, 9)

    def test_constraint_scope_restores(self):
        ana = Analyzer()
        ana.bind(x, 0, 100)
        assert_is_const_int(ana.int_set(x).max_value, 99)
        with ana.constraint_context(ir.Lt(x, ci(10), BOOL, S)):
            assert_is_const_int(ana.int_set(x).max_value, 9)
        assert_is_const_int(ana.int_set(x).max_value, 99)

    def test_constraint_eq_tightens(self):
        ana = Analyzer()
        ana.bind(x, 0, 100)
        with ana.constraint_context(ir.Eq(x, ci(42), BOOL, S)):
            s = ana.int_set(x)
            assert_is_const_int(s.min_value, 42)
            assert_is_const_int(s.max_value, 42)

    def test_constraint_and_compound(self):
        ana = Analyzer()
        ana.int_set.update(x, IntSet.everything())
        cond = ir.And(ir.Ge(x, ci(3), BOOL, S), ir.Lt(x, ci(10), BOOL, S), BOOL, S)
        with ana.constraint_context(cond):
            s = ana.int_set(x)
            assert_is_const_int(s.min_value, 3)
            assert_is_const_int(s.max_value, 9)


# ============================================================================
# CanProve symbolic fallback
# ============================================================================


class TestCanProveSymbolic:
    def test_can_prove_lt_symbolic(self):
        """x in [0, n), n > 0 => can prove x < n via symbolic bounds."""
        ana = Analyzer()
        ana.bind(n, 1, 1000)
        ana.int_set.bind(x, ci(0), n)
        assert ana.can_prove(ir.Lt(x, n, BOOL, S))

    def test_can_prove_ge_symbolic(self):
        """x in [a, ...) => x >= a."""
        ana = Analyzer()
        ana.bind(a, 0, 100)
        ana.int_set.bind(x, a, ci(1000))
        assert ana.can_prove(ir.Ge(x, a, BOOL, S))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
