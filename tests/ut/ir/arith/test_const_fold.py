# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for arith.fold_const (constant folding of BinaryExpr / UnaryExpr)."""

import pytest
from pypto import DataType, ir
from pypto.arith import fold_const

S = ir.Span.unknown()
INT = DataType.INT64
FP = DataType.FP32
BOOL = DataType.BOOL


def ci(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, INT, S)


def cf(value: float) -> ir.ConstFloat:
    return ir.ConstFloat(value, FP, S)


def cb(value: bool) -> ir.ConstBool:
    return ir.ConstBool(value, S)


# ============================================================================
# Binary arithmetic
# ============================================================================


class TestArithmeticFolding:
    def test_add_int(self):
        result = fold_const(ir.Add(ci(3), ci(5), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 8

    def test_sub_int(self):
        result = fold_const(ir.Sub(ci(10), ci(3), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 7

    def test_mul_int(self):
        result = fold_const(ir.Mul(ci(4), ci(7), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 28

    def test_add_float(self):
        result = fold_const(ir.Add(cf(1.5), cf(2.5), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(4.0)

    def test_sub_float(self):
        result = fold_const(ir.Sub(cf(5.0), cf(2.0), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(3.0)

    def test_mul_float(self):
        result = fold_const(ir.Mul(cf(3.0), cf(4.0), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(12.0)

    def test_negative_int(self):
        result = fold_const(ir.Add(ci(-3), ci(-5), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == -8


# ============================================================================
# Floor division and modulo
# ============================================================================


class TestFloorDivMod:
    def test_floordiv_positive(self):
        result = fold_const(ir.FloorDiv(ci(7), ci(3), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 2

    def test_floordiv_negative(self):
        """Floor division rounds toward negative infinity, not toward zero."""
        result = fold_const(ir.FloorDiv(ci(-7), ci(3), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == -3  # floor(-7/3) = -3, NOT -2 (truncation)

    def test_floormod_positive(self):
        result = fold_const(ir.FloorMod(ci(7), ci(3), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 1

    def test_floormod_negative(self):
        """Floor modulo result has same sign as divisor."""
        result = fold_const(ir.FloorMod(ci(-7), ci(3), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 2  # floormod(-7, 3) = 2, NOT -1 (truncation)

    def test_floordiv_by_zero_raises(self):
        with pytest.raises(Exception):
            fold_const(ir.FloorDiv(ci(5), ci(0), INT, S))

    def test_floormod_by_zero_raises(self):
        with pytest.raises(Exception):
            fold_const(ir.FloorMod(ci(5), ci(0), INT, S))


# ============================================================================
# Float division
# ============================================================================


class TestFloatDiv:
    def test_floatdiv(self):
        result = fold_const(ir.FloatDiv(cf(7.0), cf(2.0), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(3.5)

    def test_floatdiv_by_zero_returns_inf(self):
        """IEEE 754: float division by zero produces infinity."""
        result = fold_const(ir.FloatDiv(cf(5.0), cf(0.0), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == float("inf")

    def test_floatdiv_neg_by_zero_returns_neg_inf(self):
        """IEEE 754: -5.0 / 0.0 produces -infinity."""
        result = fold_const(ir.FloatDiv(cf(-5.0), cf(0.0), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == float("-inf")


# ============================================================================
# Power
# ============================================================================


class TestPow:
    def test_pow_int(self):
        result = fold_const(ir.Pow(ci(2), ci(10), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 1024

    def test_pow_zero(self):
        result = fold_const(ir.Pow(ci(5), ci(0), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 1

    def test_pow_one(self):
        result = fold_const(ir.Pow(ci(42), ci(1), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 42

    def test_pow_float(self):
        result = fold_const(ir.Pow(cf(2.0), cf(3.0), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(8.0)


# ============================================================================
# Min / Max
# ============================================================================


class TestMinMax:
    def test_min_int(self):
        result = fold_const(ir.Min(ci(3), ci(7), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 3

    def test_max_int(self):
        result = fold_const(ir.Max(ci(3), ci(7), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 7

    def test_min_float(self):
        result = fold_const(ir.Min(cf(1.5), cf(2.5), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(1.5)

    def test_max_float(self):
        result = fold_const(ir.Max(cf(1.5), cf(2.5), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(2.5)


# ============================================================================
# Comparisons
# ============================================================================


class TestComparisons:
    def test_eq_true(self):
        result = fold_const(ir.Eq(ci(5), ci(5), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_eq_false(self):
        result = fold_const(ir.Eq(ci(5), ci(3), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is False

    def test_ne(self):
        result = fold_const(ir.Ne(ci(5), ci(3), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_lt(self):
        result = fold_const(ir.Lt(ci(3), ci(5), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_le(self):
        result = fold_const(ir.Le(ci(5), ci(5), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_gt(self):
        result = fold_const(ir.Gt(ci(5), ci(3), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_ge(self):
        result = fold_const(ir.Ge(ci(3), ci(5), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is False

    def test_lt_float(self):
        result = fold_const(ir.Lt(cf(1.5), cf(2.5), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True


# ============================================================================
# Logical operations
# ============================================================================


class TestLogical:
    def test_and_true_true(self):
        result = fold_const(ir.And(cb(True), cb(True), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_and_true_false(self):
        result = fold_const(ir.And(cb(True), cb(False), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is False

    def test_or_false_true(self):
        result = fold_const(ir.Or(cb(False), cb(True), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_or_false_false(self):
        result = fold_const(ir.Or(cb(False), cb(False), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is False

    def test_xor_different(self):
        result = fold_const(ir.Xor(cb(True), cb(False), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_xor_same(self):
        result = fold_const(ir.Xor(cb(True), cb(True), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is False


# ============================================================================
# Bitwise operations
# ============================================================================


class TestBitwise:
    def test_bit_and(self):
        result = fold_const(ir.BitAnd(ci(0b1100), ci(0b1010), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 0b1000

    def test_bit_or(self):
        result = fold_const(ir.BitOr(ci(0b1100), ci(0b1010), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 0b1110

    def test_bit_xor(self):
        result = fold_const(ir.BitXor(ci(0b1100), ci(0b1010), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 0b0110

    def test_bit_shift_left(self):
        result = fold_const(ir.BitShiftLeft(ci(1), ci(4), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 16

    def test_bit_shift_right(self):
        result = fold_const(ir.BitShiftRight(ci(16), ci(2), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 4


# ============================================================================
# Identity simplifications (semi-constant: one operand constant)
# ============================================================================


class TestIdentityFolding:
    def test_add_zero_left(self):
        """0 + x -> x (returns rhs)."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.Add(ci(0), x, INT, S))
        assert result is x

    def test_add_zero_right(self):
        """x + 0 -> x (returns lhs)."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.Add(x, ci(0), INT, S))
        assert result is x

    def test_sub_zero(self):
        """x - 0 -> x."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.Sub(x, ci(0), INT, S))
        assert result is x

    def test_mul_one_left(self):
        """1 * x -> x."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.Mul(ci(1), x, INT, S))
        assert result is x

    def test_mul_one_right(self):
        """x * 1 -> x."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.Mul(x, ci(1), INT, S))
        assert result is x

    def test_mul_zero_left(self):
        """0 * x -> 0."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.Mul(ci(0), x, INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 0

    def test_mul_zero_right(self):
        """x * 0 -> 0."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.Mul(x, ci(0), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 0

    def test_floordiv_by_one(self):
        """x // 1 -> x."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.FloorDiv(x, ci(1), INT, S))
        assert result is x

    def test_floormod_by_one(self):
        """x % 1 -> 0."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.FloorMod(x, ci(1), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 0

    def test_non_foldable_returns_none(self):
        """Non-constant operands return None."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        y = ir.Var("y", ir.ScalarType(INT), S)
        result = fold_const(ir.Add(x, y, INT, S))
        assert result is None

    def test_non_binary_returns_none(self):
        """Non binary/unary expression returns None."""
        result = fold_const(ci(42))
        assert result is None


# ============================================================================
# Unary operations
# ============================================================================


class TestUnaryFolding:
    def test_neg_int(self):
        result = fold_const(ir.Neg(ci(5), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == -5

    def test_neg_float(self):
        result = fold_const(ir.Neg(cf(3.14), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(-3.14)

    def test_abs_positive(self):
        result = fold_const(ir.Abs(ci(5), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 5

    def test_abs_negative(self):
        result = fold_const(ir.Abs(ci(-5), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 5

    def test_abs_float(self):
        result = fold_const(ir.Abs(cf(-3.14), FP, S))
        assert isinstance(result, ir.ConstFloat)
        assert result.value == pytest.approx(3.14)

    def test_not_true(self):
        result = fold_const(ir.Not(cb(True), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is False

    def test_not_false(self):
        result = fold_const(ir.Not(cb(False), BOOL, S))
        assert isinstance(result, ir.ConstBool)
        assert result.value is True

    def test_bit_not(self):
        result = fold_const(ir.BitNot(ci(0), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == -1

    def test_unary_non_constant_returns_none(self):
        """Non-constant operand returns None."""
        x = ir.Var("x", ir.ScalarType(INT), S)
        result = fold_const(ir.Neg(x, INT, S))
        assert result is None


# ============================================================================
# Edge cases: overflow, INT64_MIN, shift bounds
# ============================================================================

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


class TestOverflowAndEdgeCases:
    def test_add_overflow_skips_folding(self):
        """Integer overflow in add should skip folding (return None)."""
        result = fold_const(ir.Add(ci(INT64_MAX), ci(1), INT, S))
        assert result is None

    def test_sub_overflow_skips_folding(self):
        result = fold_const(ir.Sub(ci(INT64_MIN), ci(1), INT, S))
        assert result is None

    def test_mul_overflow_skips_folding(self):
        result = fold_const(ir.Mul(ci(INT64_MAX), ci(2), INT, S))
        assert result is None

    def test_pow_overflow_skips_folding(self):
        result = fold_const(ir.Pow(ci(2), ci(63), INT, S))
        assert result is None  # 2^63 overflows int64_t

    def test_neg_int64_min_skips_folding(self):
        """Negating INT64_MIN overflows — should skip."""
        result = fold_const(ir.Neg(ci(INT64_MIN), INT, S))
        assert result is None

    def test_abs_int64_min_skips_folding(self):
        """abs(INT64_MIN) overflows — should skip."""
        result = fold_const(ir.Abs(ci(INT64_MIN), INT, S))
        assert result is None

    def test_shift_left_negative_count_skips(self):
        """Negative shift count — skip folding."""
        result = fold_const(ir.BitShiftLeft(ci(1), ci(-1), INT, S))
        assert result is None

    def test_shift_left_too_large_skips(self):
        """Shift count >= 64 — skip folding."""
        result = fold_const(ir.BitShiftLeft(ci(1), ci(64), INT, S))
        assert result is None

    def test_shift_right_negative_count_skips(self):
        result = fold_const(ir.BitShiftRight(ci(16), ci(-1), INT, S))
        assert result is None

    def test_floordiv_int64_min_neg1_raises(self):
        """INT64_MIN // -1 overflows — should raise."""
        with pytest.raises(Exception):
            fold_const(ir.FloorDiv(ci(INT64_MIN), ci(-1), INT, S))

    def test_pow_exponent_by_squaring(self):
        """Verify exponentiation by squaring gives correct result."""
        result = fold_const(ir.Pow(ci(3), ci(20), INT, S))
        assert isinstance(result, ir.ConstInt)
        assert result.value == 3**20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
