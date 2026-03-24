# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for CanonicalSimplifier (canonical sum-of-products simplification)."""

import pytest
from pypto import DataType, ir
from pypto.arith import CanonicalSimplifier

S = ir.Span.unknown()
INT = DataType.INT64
IDX = DataType.INDEX


def make_var(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(INT), S)


def ci(value: int, dtype: DataType = INT) -> ir.ConstInt:
    return ir.ConstInt(value, dtype, S)


def cb(value: bool) -> ir.ConstBool:
    return ir.ConstBool(value, S)


def assert_is_const_int(expr: ir.Expr, expected: int) -> None:
    assert isinstance(expr, ir.ConstInt), f"Expected ConstInt, got {type(expr).__name__}"
    assert expr.value == expected, f"Expected {expected}, got {expr.value}"


def assert_is_const_bool(expr: ir.Expr, expected: bool) -> None:
    assert isinstance(expr, ir.ConstBool), f"Expected ConstBool, got {type(expr).__name__}"
    assert expr.value == expected, f"Expected {expected}, got {expr.value}"


def assert_same_expr(expr: ir.Expr, expected: ir.Expr) -> None:
    """Assert that two expressions are the same object (pointer identity)."""
    assert expr is expected, (
        f"Expected same object, got different: {type(expr).__name__} vs {type(expected).__name__}"
    )


# ============================================================================
# Basic functionality
# ============================================================================


class TestBasics:
    def test_construction(self):
        s = CanonicalSimplifier()
        assert s is not None

    def test_identity_const(self):
        s = CanonicalSimplifier()
        c = ci(42)
        result = s(c)
        assert_is_const_int(result, 42)

    def test_identity_var(self):
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(x)
        assert_same_expr(result, x)

    def test_const_fold(self):
        s = CanonicalSimplifier()
        result = s(ir.Add(ci(3), ci(5), INT, S))
        assert_is_const_int(result, 8)

    def test_var_substitution(self):
        s = CanonicalSimplifier()
        x = make_var("x")
        s.update(x, ci(10))
        result = s(ir.Add(x, ci(5), INT, S))
        assert_is_const_int(result, 15)

    def test_remove_substitution(self):
        s = CanonicalSimplifier()
        x = make_var("x")
        s.update(x, ci(10))
        s.update(x, None)
        result = s(x)
        assert_same_expr(result, x)

    def test_float_unchanged(self):
        s = CanonicalSimplifier()
        x = ir.ConstFloat(3.14, DataType.FP32, S)
        result = s(x)
        assert result is x


# ============================================================================
# Coefficient collection
# ============================================================================


class TestCoefficientCollection:
    """Tests for sum-of-products coefficient merging."""

    def test_x_plus_x(self):
        """x + x => x * 2"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Add(x, x, INT, S))
        assert isinstance(result, ir.Mul), f"Expected Mul, got {type(result).__name__}"
        assert result.left is x
        assert isinstance(result.right, ir.ConstInt) and result.right.value == 2

    def test_x_times_2_plus_x(self):
        """x*2 + x => x * 3"""
        s = CanonicalSimplifier()
        x = make_var("x")
        expr = ir.Add(ir.Mul(x, ci(2), INT, S), x, INT, S)
        result = s(expr)
        assert isinstance(result, ir.Mul), f"Expected Mul, got {type(result).__name__}"
        assert result.left is x
        assert isinstance(result.right, ir.ConstInt) and result.right.value == 3

    def test_x_minus_x(self):
        """x - x => 0"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Sub(x, x, INT, S))
        assert_is_const_int(result, 0)

    def test_3x_minus_2x(self):
        """x*3 - x*2 => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        lhs = ir.Mul(x, ci(3), INT, S)
        rhs = ir.Mul(x, ci(2), INT, S)
        result = s(ir.Sub(lhs, rhs, INT, S))
        assert_same_expr(result, x)

    def test_constant_collection(self):
        """x + 3 + 5 => x + 8"""
        s = CanonicalSimplifier()
        x = make_var("x")
        inner = ir.Add(x, ci(3), INT, S)
        result = s(ir.Add(inner, ci(5), INT, S))
        assert isinstance(result, ir.Add), f"Expected Add, got {type(result).__name__}"
        assert result.left is x
        assert isinstance(result.right, ir.ConstInt) and result.right.value == 8

    def test_multi_variable(self):
        """x + y - x => y"""
        s = CanonicalSimplifier()
        x = make_var("x")
        y = make_var("y")
        inner = ir.Add(x, y, INT, S)
        result = s(ir.Sub(inner, x, INT, S))
        assert_same_expr(result, y)

    def test_x_plus_zero(self):
        """x + 0 => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Add(x, ci(0), INT, S))
        assert_same_expr(result, x)

    def test_zero_minus_x(self):
        """0 - x => -x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Sub(ci(0), x, INT, S))
        assert isinstance(result, ir.Neg), f"Expected Neg, got {type(result).__name__}"


# ============================================================================
# Multiplication distribution
# ============================================================================


class TestMulDistribution:
    """Tests for multiplication distributing over sums."""

    def test_mul_distribute(self):
        """(x + 1) * 2 => x * 2 + 2"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Mul(ir.Add(x, ci(1), INT, S), ci(2), INT, S))
        assert isinstance(result, ir.Add), f"Expected Add, got {type(result).__name__}"

    def test_mul_zero(self):
        """x * 0 => 0"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Mul(x, ci(0), INT, S))
        assert_is_const_int(result, 0)

    def test_mul_one(self):
        """x * 1 => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Mul(x, ci(1), INT, S))
        assert_same_expr(result, x)

    def test_mul_neg_one(self):
        """x * (-1) => -x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Mul(x, ci(-1), INT, S))
        assert isinstance(result, ir.Neg), f"Expected Neg, got {type(result).__name__}"

    def test_distribute_left(self):
        """2 * (x + 3) => x * 2 + 6"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Mul(ci(2), ir.Add(x, ci(3), INT, S), INT, S))
        assert isinstance(result, ir.Add), f"Expected Add, got {type(result).__name__}"


# ============================================================================
# FloorDiv simplification
# ============================================================================


class TestFloorDiv:
    """Tests for canonical floor division simplification."""

    def test_factor_div(self):
        """(x * 4) // 4 => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(ir.Mul(x, ci(4), INT, S), ci(4), INT, S))
        assert_same_expr(result, x)

    def test_partial_factor_div(self):
        """(x * 4) // 2 => x * 2"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(ir.Mul(x, ci(4), INT, S), ci(2), INT, S))
        assert isinstance(result, ir.Mul), f"Expected Mul, got {type(result).__name__}"
        assert result.left is x
        assert isinstance(result.right, ir.ConstInt) and result.right.value == 2

    def test_sum_div_all_divisible(self):
        """(4*x + 6*y + 8) // 2 => 2*x + 3*y + 4"""
        s = CanonicalSimplifier()
        x = make_var("x")
        y = make_var("y")
        inner = ir.Add(ir.Add(ir.Mul(x, ci(4), INT, S), ir.Mul(y, ci(6), INT, S), INT, S), ci(8), INT, S)
        result = s(ir.FloorDiv(inner, ci(2), INT, S))
        # Result should not contain FloorDiv
        assert not isinstance(result, ir.FloorDiv), "Expected simplified, got FloorDiv"

    def test_div_by_scale_factor(self):
        """(x * 6) // 3 => x * 2"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(ir.Mul(x, ci(6), INT, S), ci(3), INT, S))
        assert isinstance(result, ir.Mul), f"Expected Mul, got {type(result).__name__}"

    def test_nested_div(self):
        """(x * 12) // 4 // 3 => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        inner = ir.FloorDiv(ir.Mul(x, ci(12), INT, S), ci(4), INT, S)
        result = s(ir.FloorDiv(inner, ci(3), INT, S))
        assert_same_expr(result, x)

    def test_const_fold_div(self):
        """12 // 4 => 3"""
        s = CanonicalSimplifier()
        result = s(ir.FloorDiv(ci(12), ci(4), INT, S))
        assert_is_const_int(result, 3)


# ============================================================================
# FloorMod simplification
# ============================================================================


class TestFloorMod:
    """Tests for canonical floor modulo simplification."""

    def test_factor_mod(self):
        """(x * 4) % 4 => 0"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.FloorMod(ir.Mul(x, ci(4), INT, S), ci(4), INT, S))
        assert_is_const_int(result, 0)

    def test_all_scales_divisible_mod(self):
        """(4*x + 6*y + 5) % 2 => 1"""
        s = CanonicalSimplifier()
        x = make_var("x")
        y = make_var("y")
        inner = ir.Add(ir.Add(ir.Mul(x, ci(4), INT, S), ir.Mul(y, ci(6), INT, S), INT, S), ci(5), INT, S)
        result = s(ir.FloorMod(inner, ci(2), INT, S))
        assert_is_const_int(result, 1)

    def test_mul_scale_mod(self):
        """(x * 6) % 3 => 0"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.FloorMod(ir.Mul(x, ci(6), INT, S), ci(3), INT, S))
        assert_is_const_int(result, 0)

    def test_const_fold_mod(self):
        """13 % 4 => 1"""
        s = CanonicalSimplifier()
        result = s(ir.FloorMod(ci(13), ci(4), INT, S))
        assert_is_const_int(result, 1)


# ============================================================================
# Div-Mod recombination
# ============================================================================


class TestDivModRecombination:
    """Tests for div-mod identity: (x // c) * c + x % c → x."""

    def test_basic_recombination(self):
        """(x // 4) * 4 + x % 4 => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        div_part = ir.Mul(ir.FloorDiv(x, ci(4), INT, S), ci(4), INT, S)
        mod_part = ir.FloorMod(x, ci(4), INT, S)
        result = s(ir.Add(div_part, mod_part, INT, S))
        assert_same_expr(result, x)

    def test_recombination_reversed(self):
        """x % 4 + (x // 4) * 4 => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        mod_part = ir.FloorMod(x, ci(4), INT, S)
        div_part = ir.Mul(ir.FloorDiv(x, ci(4), INT, S), ci(4), INT, S)
        result = s(ir.Add(mod_part, div_part, INT, S))
        assert_same_expr(result, x)

    def test_recombination_with_8(self):
        """(x // 8) * 8 + x % 8 => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        div_part = ir.Mul(ir.FloorDiv(x, ci(8), INT, S), ci(8), INT, S)
        mod_part = ir.FloorMod(x, ci(8), INT, S)
        result = s(ir.Add(div_part, mod_part, INT, S))
        assert_same_expr(result, x)


# ============================================================================
# Negation
# ============================================================================


class TestNeg:
    """Tests for negation through canonical form."""

    def test_neg_const(self):
        s = CanonicalSimplifier()
        result = s(ir.Neg(ci(5), INT, S))
        assert_is_const_int(result, -5)

    def test_neg_var(self):
        """-(x) stays as Neg(x)"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Neg(x, INT, S))
        assert isinstance(result, ir.Neg), f"Expected Neg, got {type(result).__name__}"

    def test_neg_neg(self):
        """-(-(x)) => x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Neg(ir.Neg(x, INT, S), INT, S))
        assert_same_expr(result, x)

    def test_neg_sum(self):
        """-(x + 3) => -x - 3 (canonical: Neg(x) + (-3))"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Neg(ir.Add(x, ci(3), INT, S), INT, S))
        # Should not still be a Neg of Add
        assert not (isinstance(result, ir.Neg) and isinstance(result.operand, ir.Add))


# ============================================================================
# Passthrough operations
# ============================================================================


class TestPassthrough:
    """Verify non-arithmetic ops pass through correctly, simplifying children."""

    def test_comparison_const_fold(self):
        """1 + 2 == 3 => true"""
        s = CanonicalSimplifier()
        result = s(ir.Eq(ir.Add(ci(1), ci(2), INT, S), ci(3), INT, S))
        assert_is_const_bool(result, True)

    def test_comparison_simplifies_children(self):
        """(x + 0) == x should simplify the left side to x"""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s(ir.Eq(ir.Add(x, ci(0), INT, S), x, INT, S))
        # Both sides simplify to x (same pointer), but TryConstFoldBinary
        # only handles ConstInt/ConstFloat — non-const equality like x == x
        # is handled by the RewriteSimplifier's pattern rules, not here.
        assert isinstance(result, ir.Eq)
        assert result.left is x
        assert result.right is x

    def test_min_const_fold(self):
        """min(3, 5) => 3"""
        s = CanonicalSimplifier()
        result = s(ir.Min(ci(3), ci(5), INT, S))
        assert_is_const_int(result, 3)

    def test_max_const_fold(self):
        """max(3, 5) => 5"""
        s = CanonicalSimplifier()
        result = s(ir.Max(ci(3), ci(5), INT, S))
        assert_is_const_int(result, 5)

    def test_bool_and_fold(self):
        s = CanonicalSimplifier()
        result = s(ir.And(cb(True), cb(False), DataType.BOOL, S))
        assert_is_const_bool(result, False)

    def test_bool_or_fold(self):
        s = CanonicalSimplifier()
        result = s(ir.Or(cb(True), cb(False), DataType.BOOL, S))
        assert_is_const_bool(result, True)


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_large_coefficient(self):
        """x * 1000000 + x * 2000000 => x * 3000000"""
        s = CanonicalSimplifier()
        x = make_var("x")
        lhs = ir.Mul(x, ci(1000000), INT, S)
        rhs = ir.Mul(x, ci(2000000), INT, S)
        result = s(ir.Add(lhs, rhs, INT, S))
        assert isinstance(result, ir.Mul), f"Expected Mul, got {type(result).__name__}"

    def test_zero_const(self):
        """0 + 0 => 0"""
        s = CanonicalSimplifier()
        result = s(ir.Add(ci(0), ci(0), INT, S))
        assert_is_const_int(result, 0)

    def test_sub_constants(self):
        """10 - 3 => 7"""
        s = CanonicalSimplifier()
        result = s(ir.Sub(ci(10), ci(3), INT, S))
        assert_is_const_int(result, 7)

    def test_nested_add_constants(self):
        """(1 + 2) + (3 + 4) => 10"""
        s = CanonicalSimplifier()
        result = s(ir.Add(ir.Add(ci(1), ci(2), INT, S), ir.Add(ci(3), ci(4), INT, S), INT, S))
        assert_is_const_int(result, 10)

    def test_enter_constraint_standalone(self):
        """Standalone mode: enter_constraint returns None."""
        s = CanonicalSimplifier()
        x = make_var("x")
        result = s.enter_constraint(ir.Gt(x, ci(0), INT, S))
        assert result is None

    def test_index_dtype(self):
        """CanonicalSimplifier works with INDEX-typed expressions."""
        s = CanonicalSimplifier()
        x = ir.Var("x", ir.ScalarType(IDX), S)
        result = s(ir.Add(x, x, IDX, S))
        assert isinstance(result, ir.Mul)
        assert result.left is x
        assert isinstance(result.right, ir.ConstInt) and result.right.value == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
