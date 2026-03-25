# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the TransitiveComparisonAnalyzer."""

import pytest
from pypto import DataType, ir
from pypto.arith import Analyzer, CompareResult, TransitiveComparisonAnalyzer

S = ir.Span.unknown()
INT = DataType.INT64
BOOL = DataType.BOOL


def make_var(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(INT), S)


def ci(value: int, dtype: DataType = INT) -> ir.ConstInt:
    return ir.ConstInt(value, dtype, S)


# ============================================================================
# Shared state: single TCA with pre-bound variables for the whole file
# ============================================================================

tca = TransitiveComparisonAnalyzer()
x = make_var("x")
y = make_var("y")
z = make_var("z")
a = make_var("a")
b = make_var("b")
c = make_var("c")
d = make_var("d")
for _v in [x, y, z, a, b, c, d]:
    tca.bind(_v, -100, 100)


# ============================================================================
# Standalone basics
# ============================================================================


class TestStandalone:
    def test_create(self):
        assert tca is not None

    def test_unknown_for_unbound_vars(self):
        fresh = TransitiveComparisonAnalyzer()
        u = make_var("u")
        v = make_var("v")
        assert fresh.try_compare(u, v) == CompareResult.kUnknown

    def test_constant_comparison(self):
        assert tca.try_compare(ci(3), ci(5)) == CompareResult.kLT
        assert tca.try_compare(ci(5), ci(3)) == CompareResult.kGT
        assert tca.try_compare(ci(4), ci(4)) == CompareResult.kEQ


# ============================================================================
# Direct comparisons via Bind
# ============================================================================


class TestBind:
    def test_bind_expr_equality(self):
        t = TransitiveComparisonAnalyzer()
        p = make_var("p")
        q = make_var("q")
        t.bind(p, q)
        assert t.try_compare(p, q) == CompareResult.kEQ

    def test_bind_range_single_value(self):
        t = TransitiveComparisonAnalyzer()
        v = make_var("v")
        t.bind(v, 5, 6)  # exactly 5
        assert t.try_compare(v, ci(5)) == CompareResult.kEQ

    def test_bind_range_bounds(self):
        # x is bound to [-100, 100) — not enough to prove x >= 0
        assert tca.try_compare(x, ci(0)) == CompareResult.kUnknown


# ============================================================================
# Direct comparisons via EnterConstraint
# ============================================================================


class TestEnterConstraint:
    def test_enter_lt_constraint(self):
        recovery = tca.enter_constraint(ir.Lt(x, y, BOOL, S))
        assert tca.try_compare(x, y) == CompareResult.kLT
        recovery()
        assert tca.try_compare(x, y) == CompareResult.kUnknown

    def test_enter_ge_constraint(self):
        recovery = tca.enter_constraint(ir.Ge(x, y, BOOL, S))
        assert tca.try_compare(x, y) == CompareResult.kGE
        recovery()

    def test_enter_eq_constraint(self):
        recovery = tca.enter_constraint(ir.Eq(x, y, BOOL, S))
        assert tca.try_compare(x, y) == CompareResult.kEQ
        recovery()


# ============================================================================
# Transitive propagation
# ============================================================================


class TestTransitive:
    def test_transitive_le_chain(self):
        """x <= y and y <= z => x <= z."""
        r1 = tca.enter_constraint(ir.Le(x, y, BOOL, S))
        r2 = tca.enter_constraint(ir.Le(y, z, BOOL, S))
        assert tca.try_compare(x, z) == CompareResult.kLE
        r2()
        r1()

    def test_transitive_lt_chain(self):
        """x < y and y < z => x < z."""
        r1 = tca.enter_constraint(ir.Lt(x, y, BOOL, S))
        r2 = tca.enter_constraint(ir.Lt(y, z, BOOL, S))
        # x < y means x <= y-1, y < z means y <= z-1, so x <= z-2, meaning x < z
        assert tca.try_compare(x, z) == CompareResult.kLT
        r2()
        r1()

    def test_transitive_eq_and_le(self):
        """x == y and y <= z => x <= z."""
        r1 = tca.enter_constraint(ir.Eq(x, y, BOOL, S))
        r2 = tca.enter_constraint(ir.Le(y, z, BOOL, S))
        assert tca.try_compare(x, z) == CompareResult.kLE
        r2()
        r1()

    def test_transitive_ge_chain(self):
        """x >= y and y >= z => x >= z."""
        r1 = tca.enter_constraint(ir.Ge(x, y, BOOL, S))
        r2 = tca.enter_constraint(ir.Ge(y, z, BOOL, S))
        assert tca.try_compare(x, z) == CompareResult.kGE
        r2()
        r1()

    def test_transitive_longer_chain(self):
        """a <= b, b <= c, c <= d => a <= d."""
        r1 = tca.enter_constraint(ir.Le(a, b, BOOL, S))
        r2 = tca.enter_constraint(ir.Le(b, c, BOOL, S))
        r3 = tca.enter_constraint(ir.Le(c, d, BOOL, S))
        assert tca.try_compare(a, d) == CompareResult.kLE
        r3()
        r2()
        r1()

    def test_no_propagation_flag(self):
        """With propagate_inequalities=False, only direct comparisons are checked."""
        r1 = tca.enter_constraint(ir.Le(x, y, BOOL, S))
        r2 = tca.enter_constraint(ir.Le(y, z, BOOL, S))

        # Direct: x<=y is known directly
        assert tca.try_compare(x, y, propagate_inequalities=False) == CompareResult.kLE
        # Transitive: x<=z requires propagation
        assert tca.try_compare(x, z, propagate_inequalities=False) == CompareResult.kUnknown
        # With propagation: x<=z can be derived
        assert tca.try_compare(x, z, propagate_inequalities=True) == CompareResult.kLE

        r2()
        r1()


# ============================================================================
# Comparisons with offsets
# ============================================================================


class TestOffsets:
    def test_compare_with_offset(self):
        """x <= y + 3 implies x < y + 4."""
        y_plus_3 = ir.Add(y, ci(3), INT, S)
        recovery = tca.enter_constraint(ir.Le(x, y_plus_3, BOOL, S))

        # x compared to y+3 should give kLE
        assert tca.try_compare(x, y_plus_3) == CompareResult.kLE
        # x compared to y+4: we know x <= y+3, i.e. x < y+4
        y_plus_4 = ir.Add(y, ci(4), INT, S)
        assert tca.try_compare(x, y_plus_4) == CompareResult.kLT
        # x compared to y+2 — cannot prove
        y_plus_2 = ir.Add(y, ci(2), INT, S)
        assert tca.try_compare(x, y_plus_2) == CompareResult.kUnknown

        recovery()


# ============================================================================
# Integration with Analyzer
# ============================================================================

ana = Analyzer()
ax = make_var("ax")
ay = make_var("ay")
az = make_var("az")
ana.bind(ax, -100, 100)
ana.bind(ay, -100, 100)
ana.bind(az, -100, 100)


class TestAnalyzerIntegration:
    def test_transitive_cmp_accessible(self):
        assert ana.transitive_cmp is not None

    def test_bind_propagates(self):
        a2 = Analyzer()
        p = make_var("p")
        q = make_var("q")
        a2.bind(p, q)
        assert a2.transitive_cmp.try_compare(p, q) == CompareResult.kEQ

    def test_bind_range_propagates(self):
        # ax is bound to [-100, 100) — not enough to prove ax >= 0
        assert ana.transitive_cmp.try_compare(ax, ci(0)) == CompareResult.kUnknown

    def test_constraint_context_propagates(self):
        with ana.constraint_context(ir.Lt(ax, ay, BOOL, S)):
            assert ana.transitive_cmp.try_compare(ax, ay) == CompareResult.kLT
        # After exiting, constraint is gone.
        assert ana.transitive_cmp.try_compare(ax, ay) == CompareResult.kUnknown

    def test_can_prove_transitive(self):
        """CanProve should use transitive comparisons as fallback."""
        with ana.constraint_context(ir.Lt(ax, ay, BOOL, S)):
            with ana.constraint_context(ir.Lt(ay, az, BOOL, S)):
                assert ana.can_prove(ir.Lt(ax, az, BOOL, S))

    def test_can_prove_le_transitive(self):
        with ana.constraint_context(ir.Le(ax, ay, BOOL, S)):
            with ana.constraint_context(ir.Le(ay, az, BOOL, S)):
                assert ana.can_prove(ir.Le(ax, az, BOOL, S))

    def test_can_prove_ge_transitive(self):
        with ana.constraint_context(ir.Ge(ax, ay, BOOL, S)):
            with ana.constraint_context(ir.Ge(ay, az, BOOL, S)):
                assert ana.can_prove(ir.Ge(ax, az, BOOL, S))


# ============================================================================
# CompareResult enum
# ============================================================================


class TestCompareResult:
    def test_values_exist(self):
        assert CompareResult.kInconsistent is not None
        assert CompareResult.kEQ is not None
        assert CompareResult.kLT is not None
        assert CompareResult.kLE is not None
        assert CompareResult.kGT is not None
        assert CompareResult.kGE is not None
        assert CompareResult.kNE is not None
        assert CompareResult.kUnknown is not None


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_same_variable(self):
        """Comparing a variable to itself should give kEQ."""
        assert tca.try_compare(x, x) == CompareResult.kEQ

    def test_unknown_variables(self):
        """Variables never seen before should return kUnknown."""
        fresh = TransitiveComparisonAnalyzer()
        u = make_var("u")
        v = make_var("v")
        assert fresh.try_compare(u, v) == CompareResult.kUnknown

    def test_constraint_scope_isolation(self):
        """Constraints should not leak between scopes."""
        recovery = tca.enter_constraint(ir.Le(x, y, BOOL, S))
        assert tca.try_compare(x, y) == CompareResult.kLE
        recovery()
        assert tca.try_compare(x, y) == CompareResult.kUnknown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
