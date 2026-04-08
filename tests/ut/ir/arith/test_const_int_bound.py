# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for ConstIntBoundAnalyzer (bound propagation through expression trees)."""

import pytest
from pypto import DataType, ir
from pypto.arith import ConstIntBound, ConstIntBoundAnalyzer

S = ir.Span.unknown()
INT = DataType.INT64


def make_var(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(INT), S)


def ci(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, INT, S)


# Module-level shared instances
analyzer = ConstIntBoundAnalyzer()
x = make_var("x")
y = make_var("y")
z = make_var("z")
i = make_var("i")

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


# ============================================================================
# Helpers and basic structure
# ============================================================================


class TestConstIntBoundBasics:
    def test_const_int_exact_bound(self):
        b = analyzer(ci(42))
        assert b.min_value == 42
        assert b.max_value == 42
        assert b.is_const()

    def test_const_int_negative(self):
        b = analyzer(ci(-7))
        assert b.min_value == -7
        assert b.max_value == -7

    def test_const_int_zero(self):
        b = analyzer(ci(0))
        assert b.min_value == 0
        assert b.max_value == 0

    def test_unknown_var(self):
        b = analyzer(z)
        assert b.min_value == ConstIntBound.kNegInf
        assert b.max_value == ConstIntBound.kPosInf
        assert b.is_everything()

    def test_index_var_non_negative(self):
        """INDEX-typed vars are implicitly non-negative."""
        idx_var = ir.Var("n", ir.ScalarType(DataType.INDEX), S)
        b = analyzer(idx_var)
        assert b.min_value == 0
        assert b.max_value == ConstIntBound.kPosInf

    def test_bound_var(self):
        analyzer.bind(x, 0, 8)  # [0, 8) = [0, 7]
        b = analyzer(x)
        assert b.min_value == 0
        assert b.max_value == 7
        analyzer.unbind(x)

    def test_bound_repr(self):
        b = analyzer(ci(5))
        assert "5" in repr(b)

    def test_inf_repr(self):
        b = analyzer(z)
        assert "+inf" in repr(b)
        assert "-inf" in repr(b)


# ============================================================================
# Addition
# ============================================================================


class TestAddBound:
    def test_add_const(self):
        b = analyzer(ir.Add(ci(3), ci(5), INT, S))
        assert b.min_value == 8
        assert b.max_value == 8

    def test_add_var(self):
        analyzer.bind(x, 0, 11)  # [0, 10]
        analyzer.bind(y, 0, 11)  # [0, 10]
        b = analyzer(ir.Add(x, y, INT, S))
        assert b.min_value == 0
        assert b.max_value == 20
        analyzer.unbind(x)
        analyzer.unbind(y)

    def test_add_mixed_sign(self):
        analyzer.bind(x, -5, 6)  # [-5, 5]
        analyzer.bind(y, 1, 4)  # [1, 3]
        b = analyzer(ir.Add(x, y, INT, S))
        assert b.min_value == -4
        assert b.max_value == 8
        analyzer.unbind(x)
        analyzer.unbind(y)


# ============================================================================
# Subtraction
# ============================================================================


class TestSubBound:
    def test_sub_const(self):
        b = analyzer(ir.Sub(ci(10), ci(3), INT, S))
        assert b.min_value == 7
        assert b.max_value == 7

    def test_sub_var(self):
        analyzer.bind(x, 5, 11)  # [5, 10]
        analyzer.bind(y, 1, 4)  # [1, 3]
        b = analyzer(ir.Sub(x, y, INT, S))
        assert b.min_value == 2  # 5 - 3
        assert b.max_value == 9  # 10 - 1
        analyzer.unbind(x)
        analyzer.unbind(y)


# ============================================================================
# Multiplication
# ============================================================================


class TestMulBound:
    def test_mul_const(self):
        b = analyzer(ir.Mul(ci(3), ci(4), INT, S))
        assert b.min_value == 12
        assert b.max_value == 12

    def test_mul_positive(self):
        analyzer.bind(x, 2, 6)  # [2, 5]
        analyzer.bind(y, 3, 8)  # [3, 7]
        b = analyzer(ir.Mul(x, y, INT, S))
        assert b.min_value == 6  # 2 * 3
        assert b.max_value == 35  # 5 * 7
        analyzer.unbind(x)
        analyzer.unbind(y)

    def test_mul_mixed_sign(self):
        analyzer.bind(x, -5, 6)  # [-5, 5]
        b = analyzer(ir.Mul(x, x, INT, S))
        # Four-corner: min(25, -25, -25, 25) = -25, max = 25
        assert b.min_value == -25
        assert b.max_value == 25
        analyzer.unbind(x)

    def test_mul_by_zero(self):
        analyzer.bind(x, 0, 11)  # [0, 10]
        b = analyzer(ir.Mul(x, ci(0), INT, S))
        assert b.min_value == 0
        assert b.max_value == 0
        analyzer.unbind(x)


# ============================================================================
# Floor division
# ============================================================================


class TestFloorDivBound:
    def test_floordiv_simplifies_to_zero(self):
        """x in [0, 8) => x // 8 == 0."""
        analyzer.bind(x, 0, 8)  # [0, 7]
        b = analyzer(ir.FloorDiv(x, ci(8), INT, S))
        assert b.min_value == 0
        assert b.max_value == 0
        analyzer.unbind(x)

    def test_floordiv_positive(self):
        analyzer.bind(x, 0, 16)  # [0, 15]
        b = analyzer(ir.FloorDiv(x, ci(4), INT, S))
        assert b.min_value == 0  # 0 // 4
        assert b.max_value == 3  # 15 // 4
        analyzer.unbind(x)

    def test_floordiv_negative_dividend(self):
        analyzer.bind(x, -10, 0)  # [-10, -1]
        b = analyzer(ir.FloorDiv(x, ci(3), INT, S))
        assert b.min_value == -4  # -10 // 3 = -4
        assert b.max_value == -1  # -1 // 3 = -1
        analyzer.unbind(x)

    def test_floordiv_mixed_dividend(self):
        analyzer.bind(x, -5, 6)  # [-5, 5]
        b = analyzer(ir.FloorDiv(x, ci(3), INT, S))
        assert b.min_value == -2  # -5 // 3 = -2
        assert b.max_value == 1  # 5 // 3 = 1
        analyzer.unbind(x)

    def test_floordiv_negative_divisor(self):
        analyzer.bind(x, 1, 11)  # [1, 10]
        b = analyzer(ir.FloorDiv(x, ci(-3), INT, S))
        assert b.min_value == -4  # 10 // -3 = -4
        assert b.max_value == -1  # 1 // -3 = -1
        analyzer.unbind(x)

    def test_floordiv_by_larger_value(self):
        """When x < divisor, result is 0."""
        analyzer.bind(x, 0, 5)  # [0, 4]
        b = analyzer(ir.FloorDiv(x, ci(100), INT, S))
        assert b.min_value == 0
        assert b.max_value == 0
        analyzer.unbind(x)


# ============================================================================
# Floor mod
# ============================================================================


class TestFloorModBound:
    def test_floormod_basic(self):
        analyzer.bind(x, 0, 101)  # [0, 100]
        b = analyzer(ir.FloorMod(x, ci(8), INT, S))
        assert b.min_value == 0
        assert b.max_value == 7
        analyzer.unbind(x)

    def test_floormod_small_range(self):
        """When a_max < b, mod bound can be tighter."""
        analyzer.bind(x, 0, 4)  # [0, 3]
        b = analyzer(ir.FloorMod(x, ci(8), INT, S))
        assert b.min_value == 0
        assert b.max_value == 3  # min(3, 7) = 3
        analyzer.unbind(x)

    def test_floormod_const(self):
        b = analyzer(ir.FloorMod(ci(17), ci(5), INT, S))
        assert b.min_value == 0
        assert b.max_value == 4  # conservative for mod 5


# ============================================================================
# Min / Max
# ============================================================================


class TestMinMaxBound:
    def test_min_bound(self):
        analyzer.bind(x, 0, 11)  # [0, 10]
        analyzer.bind(y, 5, 16)  # [5, 15]
        b = analyzer(ir.Min(x, y, INT, S))
        assert b.min_value == 0  # min(0, 5)
        assert b.max_value == 10  # min(10, 15)
        analyzer.unbind(x)
        analyzer.unbind(y)

    def test_max_bound(self):
        analyzer.bind(x, 0, 11)  # [0, 10]
        analyzer.bind(y, 5, 16)  # [5, 15]
        b = analyzer(ir.Max(x, y, INT, S))
        assert b.min_value == 5  # max(0, 5)
        assert b.max_value == 15  # max(10, 15)
        analyzer.unbind(x)
        analyzer.unbind(y)


# ============================================================================
# Negation and Abs
# ============================================================================


class TestUnaryBound:
    def test_neg_bound(self):
        analyzer.bind(x, 2, 8)  # [2, 7]
        b = analyzer(ir.Neg(x, INT, S))
        assert b.min_value == -7
        assert b.max_value == -2
        analyzer.unbind(x)

    def test_abs_positive(self):
        analyzer.bind(x, 3, 11)  # [3, 10]
        b = analyzer(ir.Abs(x, INT, S))
        assert b.min_value == 3
        assert b.max_value == 10
        analyzer.unbind(x)

    def test_abs_negative(self):
        analyzer.bind(x, -10, -2)  # [-10, -3]
        b = analyzer(ir.Abs(x, INT, S))
        assert b.min_value == 3
        assert b.max_value == 10
        analyzer.unbind(x)

    def test_abs_mixed(self):
        analyzer.bind(x, -3, 8)  # [-3, 7]
        b = analyzer(ir.Abs(x, INT, S))
        assert b.min_value == 0
        assert b.max_value == 7
        analyzer.unbind(x)


# ============================================================================
# Comparisons and logical ops
# ============================================================================


class TestComparisonBound:
    def test_comparison_always_bool(self):
        for op_cls in [ir.Eq, ir.Ne, ir.Lt, ir.Le, ir.Gt, ir.Ge]:
            b = analyzer(op_cls(x, y, DataType.BOOL, S))
            assert b.min_value == 0
            assert b.max_value == 1

    def test_logical_always_bool(self):
        for op_cls in [ir.And, ir.Or]:
            b = analyzer(op_cls(x, y, DataType.BOOL, S))
            assert b.min_value == 0
            assert b.max_value == 1

    def test_not_always_bool(self):
        b = analyzer(ir.Not(x, DataType.BOOL, S))
        assert b.min_value == 0
        assert b.max_value == 1


# ============================================================================
# Composite expressions
# ============================================================================


class TestCompositeExprBound:
    def test_nested_add_mul(self):
        """(x + y) * 2 where x in [1, 5], y in [2, 3]."""
        analyzer.bind(x, 1, 6)  # [1, 5]
        analyzer.bind(y, 2, 4)  # [2, 3]
        expr = ir.Mul(ir.Add(x, y, INT, S), ci(2), INT, S)
        b = analyzer(expr)
        assert b.min_value == 6  # (1+2)*2
        assert b.max_value == 16  # (5+3)*2
        analyzer.unbind(x)
        analyzer.unbind(y)

    def test_index_expression(self):
        """Typical tiling: (i // 8) * 8 + i % 8 should have same bounds as i."""
        analyzer.bind(i, 0, 64)  # [0, 63]
        div_part = ir.Mul(ir.FloorDiv(i, ci(8), INT, S), ci(8), INT, S)
        mod_part = ir.FloorMod(i, ci(8), INT, S)
        expr = ir.Add(div_part, mod_part, INT, S)
        b = analyzer(expr)
        # div_part: [0, 7] * 8 = [0, 56]
        # mod_part: [0, 7]
        # sum: [0, 63]
        assert b.min_value == 0
        assert b.max_value == 63
        analyzer.unbind(i)


# ============================================================================
# Bitwise operations
# ============================================================================


class TestBitwiseBound:
    def test_bit_and_non_negative(self):
        analyzer.bind(x, 0, 256)  # [0, 255]
        analyzer.bind(y, 0, 16)  # [0, 15]
        b = analyzer(ir.BitAnd(x, y, INT, S))
        assert b.min_value == 0
        assert b.max_value == 15  # min(255, 15)
        analyzer.unbind(x)
        analyzer.unbind(y)

    def test_shift_right_non_negative(self):
        analyzer.bind(x, 0, 256)  # [0, 255]
        b = analyzer(ir.BitShiftRight(x, ci(4), INT, S))
        assert b.min_value == 0
        assert b.max_value == 15  # 255 >> 4
        analyzer.unbind(x)


# ============================================================================
# ConstBool and ConstFloat
# ============================================================================


class TestConstBoolFloat:
    def test_const_bool_true(self):
        b = analyzer(ir.ConstBool(True, S))
        assert b.min_value == 1
        assert b.max_value == 1

    def test_const_bool_false(self):
        b = analyzer(ir.ConstBool(False, S))
        assert b.min_value == 0
        assert b.max_value == 0

    def test_const_float(self):
        b = analyzer(ir.ConstFloat(3.7, DataType.FP32, S))
        assert b.min_value == 3  # floor(3.7)
        assert b.max_value == 4  # ceil(3.7)


# ============================================================================
# Power
# ============================================================================


class TestPowBound:
    def test_pow_zero_exponent(self):
        analyzer.bind(x, 1, 11)  # [1, 10]
        b = analyzer(ir.Pow(x, ci(0), INT, S))
        assert b.min_value == 1
        assert b.max_value == 1
        analyzer.unbind(x)

    def test_pow_one_exponent(self):
        analyzer.bind(x, 2, 6)  # [2, 5]
        b = analyzer(ir.Pow(x, ci(1), INT, S))
        assert b.min_value == 2
        assert b.max_value == 5
        analyzer.unbind(x)

    def test_pow_square_non_negative(self):
        analyzer.bind(x, 2, 5)  # [2, 4]
        b = analyzer(ir.Pow(x, ci(2), INT, S))
        assert b.min_value == 4  # 2^2
        assert b.max_value == 16  # 4^2
        analyzer.unbind(x)


# ============================================================================
# Multiple variables / var identity
# ============================================================================


class TestVarIdentity:
    def test_different_vars_independent(self):
        """Two vars with same name are different objects."""
        x1 = make_var("x")
        x2 = make_var("x")
        analyzer.bind(x1, 0, 10)
        analyzer.bind(x2, 100, 200)
        assert analyzer(x1).min_value == 0
        assert analyzer(x2).min_value == 100
        analyzer.unbind(x1)
        analyzer.unbind(x2)

    def test_unbound_var_is_everything(self):
        analyzer.bind(x, 0, 10)
        bx = analyzer(x)
        by = analyzer(y)
        assert bx.min_value == 0
        assert by.is_everything()
        analyzer.unbind(x)


# ============================================================================
# Edge cases: overflow, INT64 boundaries, unsigned cast
# ============================================================================


class TestEdgeCases:
    def test_neg_int64_min(self):
        """Negating INT64_MIN should not cause UB — saturate to kPosInf."""
        b = analyzer(ir.Neg(ci(INT64_MIN), INT, S))
        # -INT64_MIN overflows int64, so bound saturates
        assert b.min_value == ConstIntBound.kPosInf
        assert b.max_value == ConstIntBound.kPosInf

    def test_abs_int64_min(self):
        """Abs of INT64_MIN should saturate rather than UB."""
        b = analyzer(ir.Abs(ci(INT64_MIN), INT, S))
        assert b.min_value == ConstIntBound.kPosInf
        assert b.max_value == ConstIntBound.kPosInf

    def test_large_pow_exponent(self):
        """Large exponent should not cause O(e) slowdown — O(log e) expected."""
        analyzer.bind(x, 2, 4)  # [2, 3]
        b = analyzer(ir.Pow(x, ci(1000000), INT, S))
        # Result overflows to inf but should compute fast
        assert b.min_value == ConstIntBound.kPosInf or b.min_value > 0
        assert b.max_value == ConstIntBound.kPosInf
        analyzer.unbind(x)

    def test_cast_to_unsigned(self):
        """Cast to unsigned type should use [0, 2^bits-1] range."""
        analyzer.bind(x, -10, 300)  # [-10, 299]
        b = analyzer(ir.Cast(x, DataType.UINT8, S))
        # UINT8 range is [0, 255], intersected with [-10, 299]
        assert b.min_value == 0
        assert b.max_value == 255
        analyzer.unbind(x)

    def test_cast_to_signed(self):
        """Cast to signed type should use [-2^(bits-1), 2^(bits-1)-1] range."""
        analyzer.bind(x, -200, 200)  # [-200, 199]
        b = analyzer(ir.Cast(x, DataType.INT8, S))
        # INT8 range is [-128, 127]
        assert b.min_value == -128
        assert b.max_value == 127
        analyzer.unbind(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
