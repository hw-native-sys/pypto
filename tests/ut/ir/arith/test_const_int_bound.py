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
from pypto.arith import Analyzer, ConstIntBound, ConstIntBoundAnalyzer

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

    def test_index_var_is_not_implicitly_non_negative(self):
        """An unbound INDEX var is unbounded — the dtype implies no sign.

        `INDEX` is signed, so a runtime scalar of it can hold a negative value.
        Defaulting it to `[0, +inf)` folded users' `if idx >= 0` guards away as
        statically true and silently dropped the bounds check behind them
        (issue #2500).
        """
        idx_var = ir.Var("n", ir.ScalarType(DataType.INDEX), S)
        b = analyzer(idx_var)
        assert b.min_value == ConstIntBound.kNegInf
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

    def test_floordiv_unbounded_below_reaches_int64_min(self):
        """`x // 7` for an unbound `x` still reaches floordiv(INT64_MIN, 7).

        kNegInf is stored as INT64_MIN + 1, so floor-dividing the stored value
        recorded a minimum one above the reachable floordiv(INT64_MIN, 7), folding
        `x // 7 == -1317624576693539402` to False although `x == INT64_MIN` produces
        it.
        """
        analyzer.update(x, ConstIntBound(ConstIntBound.kNegInf, ConstIntBound.kPosInf))
        b = analyzer(ir.FloorDiv(x, ci(7), INT, S))
        assert b.min_value == INT64_MIN // 7
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


class TestFloorDivModNarrowBounds:
    @pytest.mark.parametrize("dtype", [DataType.INT8, DataType.INDEX])
    def test_in_range_divisor_retains_useful_modulo_bound(self, dtype):
        lhs = ir.Var("x", ir.ScalarType(dtype), S)
        expr = ir.FloorMod(lhs, ir.ConstInt(3, dtype, S), dtype, S)
        bound = ConstIntBoundAnalyzer()(expr)
        assert (bound.min_value, bound.max_value) == (0, 2)

    @pytest.mark.parametrize("op", [ir.FloorDiv, ir.FloorMod])
    @pytest.mark.parametrize("dtype,bits", [(DataType.INT8, 8), (DataType.INT16, 16), (DataType.INT32, 32)])
    def test_wrapped_divisor_cannot_prove_outer_sign(self, op, dtype, bits):
        """A raw positive literal emits as -3, so the result can be negative."""
        lhs = ir.Var("x", ir.ScalarType(dtype), S)
        divisor = ir.ConstInt((1 << bits) - 3, dtype, S)
        inner = op(lhs, divisor, dtype, S)
        bound_analyzer = ConstIntBoundAnalyzer()
        bound_analyzer.bind(lhs, 1, 3)
        assert bound_analyzer(inner).is_everything()

        simplifier = Analyzer()
        simplifier.bind(lhs, 1, 3)
        zero = ir.ConstInt(0, dtype, S)
        for expr in (
            ir.Min(inner, zero, dtype, S),
            ir.Max(inner, zero, dtype, S),
            ir.Le(zero, inner, DataType.BOOL, S),
        ):
            ir.assert_structural_equal(simplifier.simplify(expr), expr)

    @pytest.mark.parametrize("op", [ir.FloorDiv, ir.FloorMod])
    def test_divisor_range_narrowed_to_operation_width_is_unknown(self, op):
        """A wide operand's positive range becomes negative after i32 narrowing."""
        lhs = ir.Var("x", ir.ScalarType(DataType.INT32), S)
        rhs = make_var("divisor")
        expr = op(lhs, rhs, DataType.INT32, S)
        bound_analyzer = ConstIntBoundAnalyzer()
        bound_analyzer.bind(lhs, 1, 3)
        bound_analyzer.bind(rhs, 2**32 - 3, 2**32 - 2)
        assert bound_analyzer(expr).is_everything()

    @pytest.mark.parametrize("op", [ir.FloorDiv, ir.FloorMod])
    def test_wrapped_dividend_has_unknown_bound(self, op):
        """253 : i8 emits as -3; its stored sign cannot bound the quotient."""
        dtype = DataType.INT8
        expr = op(ir.ConstInt(253, dtype, S), ir.ConstInt(-2, dtype, S), dtype, S)
        bound_analyzer = ConstIntBoundAnalyzer()
        bound = bound_analyzer(expr)
        # Actual FloorDiv(-3,-2) is 1 and FloorMod(-3,-2) is -1. The raw
        # dividend must not establish an exact/sign bound for either operation.
        assert bound.is_everything()

    @pytest.mark.parametrize("op", [ir.FloorDiv, ir.FloorMod])
    @pytest.mark.parametrize("cast_kind", ["widen_wrapped_literal", "narrow_literal", "partial_range"])
    def test_cast_divisor_does_not_hide_wrapped_sign(self, op, cast_kind):
        dtype = DataType.INT32
        lhs = ir.Var("x", ir.ScalarType(dtype), S)
        divisor = make_var("divisor")
        if cast_kind == "widen_wrapped_literal":
            rhs = ir.Cast(ir.ConstInt(253, DataType.INT8, S), dtype, S)
        elif cast_kind == "narrow_literal":
            # A raw intersection would produce the empty bound [253, 127].
            rhs = ir.Cast(ci(253), DataType.INT8, S)
        else:
            # Clipping [126, 253] to [126, 127] would hide divisor 253 -> -3.
            rhs = ir.Cast(divisor, DataType.INT8, S)
        expr = op(lhs, rhs, dtype, S)
        bound_analyzer = ConstIntBoundAnalyzer()
        bound_analyzer.bind(lhs, 1, 3)
        bound_analyzer.bind(divisor, 126, 254)
        assert bound_analyzer(expr).is_everything()
        outer = ir.Min(expr, ir.ConstInt(0, dtype, S), dtype, S)
        simplifier = Analyzer()
        simplifier.bind(lhs, 1, 3)
        simplifier.bind(divisor, 126, 254)
        ir.assert_structural_equal(simplifier.simplify(outer), outer)


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
        """Negating INT64_MIN is unrepresentable, so nothing is known about the result.

        `-INT64_MIN` does not fit int64, and the hardware wraps rather than
        saturating: `arith.subi` on INT64_MIN yields INT64_MIN back, which is
        negative. Reporting [+inf, +inf] claimed the result was exactly
        INT64_MAX, an interval that excludes every value the node can hold.
        """
        b = analyzer(ir.Neg(ci(INT64_MIN), INT, S))
        assert b.is_everything()

    def test_abs_int64_min(self):
        """Abs of INT64_MIN is unrepresentable, so nothing is known about the result.

        Deliberately *not* [0, kPosInf]: two's-complement `abs(INT64_MIN)` is
        INT64_MIN itself, i.e. still negative, so any non-negativity claim here
        would be a fresh unsoundness rather than a conservative one.
        """
        b = analyzer(ir.Abs(ci(INT64_MIN), INT, S))
        assert b.is_everything()

    def test_negating_an_unbounded_below_interval_cannot_claim_a_sign(self):
        """`-x` where `x` has no lower bound is not provably non-negative.

        kNegInf is how "no lower bound" is spelled, so the interval contains
        INT64_MIN, and `arith.subi 0, x : i64` wraps INT64_MIN back onto itself
        -- still negative. Mirroring the endpoints reported [0, +inf], which
        folded `-x < 0` to False at the one input that satisfies it.
        """
        analyzer.bind(x, -INT64_MAX, 1)  # [kNegInf, 0] -- unbounded below
        b = analyzer(ir.Neg(x, INT, S))
        assert b.is_everything()
        analyzer.unbind(x)

    def test_abs_of_an_unbounded_below_interval_cannot_claim_a_sign(self):
        """Abs reaches the same wrap as Neg and must widen with it.

        For an all-negative interval Abs mirrors the endpoints exactly as Neg
        does, so an unbounded lower endpoint carries INT64_MIN through
        `abs(INT64_MIN) == INT64_MIN` and the result is not non-negative.
        """
        analyzer.bind(x, -INT64_MAX, 1)  # [kNegInf, 0] -- unbounded below
        b = analyzer(ir.Abs(x, INT, S))
        assert b.is_everything()
        analyzer.unbind(x)

    def test_unbounded_below_times_minus_one_cannot_claim_a_sign(self):
        """Multiplying by -1 is a negation, and shares its one wrapping input.

        The +-1 identities in the endpoint multiply must not shortcut past this:
        `x * -1` is emitted as a multiply, but it wraps on INT64_MIN exactly
        where `-x` does.
        """
        analyzer.bind(x, -INT64_MAX, 1)  # [kNegInf, 0] -- unbounded below
        b = analyzer(ir.Mul(x, ci(-1), INT, S))
        assert b.is_everything()
        analyzer.unbind(x)

    def test_subtracting_zero_from_an_unbounded_below_interval_cannot_claim_a_sign(self):
        """`0 - x` where `x` has no lower bound reaches INT64_MIN, so it wraps.

        `0 - x` is the negation of an unbounded-below interval: at `x == INT64_MIN`
        the emitted `arith.subi` yields INT64_MIN, still negative. Reading kNegInf
        as a value gave `0 - INT64_MIN + 1 == INT64_MAX` for the upper endpoint and
        reported `[0, +inf]`, which folded `0 - x < 0` to False at that input.
        """
        analyzer.bind(x, -INT64_MAX, 1)  # [kNegInf, 0] -- unbounded below
        b = analyzer(ir.Sub(ci(0), x, INT, S))
        assert b.is_everything()
        analyzer.unbind(x)

    def test_adding_a_positive_offset_keeps_int64_min_reachable(self):
        """`x + 2` where `x` has no lower bound still reaches INT64_MIN + 2.

        kNegInf is stored as INT64_MIN + 1, so treating it as a value gave
        [INT64_MIN + 3, 2] and folded `x + 2 == INT64_MIN + 2` to False although
        `x == INT64_MIN` produces exactly that without wrapping.
        """
        analyzer.bind(x, -INT64_MAX, 1)  # [kNegInf, 0] -- unbounded below
        b = analyzer(ir.Add(x, ci(2), INT, S))
        assert b.min_value == INT64_MIN + 2
        analyzer.unbind(x)

    def test_upper_endpoint_sentinel_is_not_demoted(self):
        """`z + 2` where `z` is pinned to kNegInf keeps INT64_MIN + 3 reachable.

        `z = x - 1` for `x == INT64_MIN + 2` gives the exact bound
        [kNegInf, kNegInf]. kNegInf means "reaches INT64_MIN" only as a *lower*
        endpoint; as an upper endpoint it is the exact stored value INT64_MIN + 1.
        Demoting both endpoints folded `z + 2 == INT64_MIN + 3` to False although
        `z == INT64_MIN + 1` makes it always true.
        """
        analyzer.bind(x, INT64_MIN + 2, INT64_MIN + 3)  # [INT64_MIN + 2, INT64_MIN + 2]
        b = analyzer(ir.Add(ir.Sub(x, ci(1), INT, S), ci(2), INT, S))
        assert b.max_value == INT64_MIN + 3
        analyzer.unbind(x)

    def test_large_pow_exponent(self):
        """Large exponent should not cause O(e) slowdown — O(log e) expected.

        The result overflows int64, where `arith.muli` wraps and can land on any
        value the dtype holds, so the bound is unknown rather than +inf. The
        exponent stays large to keep this a guard on the O(log e) squaring loop.
        """
        analyzer.bind(x, 2, 4)  # [2, 3]
        b = analyzer(ir.Pow(x, ci(1000000), INT, S))
        assert b.is_everything()
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

    def test_narrowing_cast_wraps_instead_of_clamping(self):
        """A source range that overflows the target wraps, so the result is the full target range.

        A scalar integer cast truncates: `[120, 139]` to INT8 yields 120..127 and -128..-117.
        Clamping to `[120, 127]` would let a caller prove `cast(x, INT8) >= 0`.
        """
        analyzer.bind(x, 120, 140)  # [120, 139]
        b = analyzer(ir.Cast(x, DataType.INT8, S))
        assert b.min_value == -128
        assert b.max_value == 127
        analyzer.unbind(x)

    def test_cast_to_uint64_is_unknown(self):
        """UINT64 has no int64 range to clamp to, so the source interval is not passed through.

        `[-10, -1]` describes none of the values a UINT64 holds — the wrap puts them
        near 2^64 — so reporting it would let a caller prove the result negative.
        """
        analyzer.bind(x, -10, 0)  # [-10, -1]
        b = analyzer(ir.Cast(x, DataType.UINT64, S))
        assert b.is_everything()
        analyzer.unbind(x)

    def test_constraint_cannot_pin_an_endpoint_onto_a_sentinel(self):
        """A constraint whose endpoint lands on a sentinel value is dropped, not recorded.

        `x > INT64_MIN` steps the endpoint to INT64_MIN + 1, which *is* kNegInf, so
        pairing it with `x < -9223372036854775806` pinned `x` to [kNegInf, kNegInf].
        Endpoint arithmetic reads that as unbounded while the fold reads it as a value,
        so `x * 2` was proved negative although `arith.muli : i64` wraps positive there
        ((-2**63 + 1) * 2 is 2). Dropping the constraint only widens the bound.
        """
        full = Analyzer()
        with full.constraint_context(ir.Gt(x, ci(INT64_MIN), DataType.BOOL, S)):
            with full.constraint_context(ir.Lt(x, ci(INT64_MIN + 2), DataType.BOOL, S)):
                b = full.const_int_bound(ir.Mul(x, ci(2), INT, S))
                # The only value x can take is INT64_MIN + 1, whose doubling wraps to 2.
                assert not b.max_value < 0, f"product proved negative: [{b.min_value}, {b.max_value}]"

    def test_narrowing_cast_keeps_a_range_that_fits(self):
        """A source range inside the target range is unchanged by the cast."""
        analyzer.bind(x, 3, 100)  # [3, 99]
        b = analyzer(ir.Cast(x, DataType.INT8, S))
        assert b.min_value == 3
        assert b.max_value == 99
        analyzer.unbind(x)

    def test_signed_64_bit_cast_keeps_an_int64_min_endpoint(self):
        """A cast between 64-bit signed types is the identity, so its bound is the source's.

        kNegInf is stored as INT64_MIN + 1, so a range check against the INT64 "range"
        [kNegInf, kPosInf] reads a genuine INT64_MIN endpoint as out of range and widens
        the result to unknown. That only loses precision by itself, but it hands
        `CanProveGreaterEqual` an interval it then misreads (see test_analyzer.py).
        """
        idx = ir.Var("idx", ir.ScalarType(DataType.INDEX), S)
        analyzer.bind(idx, INT64_MIN, INT64_MIN + 3)  # MIN, MIN+1, MIN+2
        b = analyzer(ir.Cast(idx, DataType.INT64, S))
        assert b.min_value == INT64_MIN
        assert b.max_value == INT64_MIN + 2
        analyzer.unbind(idx)

    def test_unknown_plus_pos_inf_sentinel_stays_unknown(self):
        """`x + INT64_MAX` for an unbound `x` is unknown, not `+inf`.

        The literal is numerically the `+inf` sentinel, so once stored in a bound
        it cannot be told apart from an unbounded operand. Resolving the lower
        endpoint to `+inf` would claim `x + INT64_MAX > 0` for every `x`.
        """
        b = analyzer(ir.Add(x, ci(INT64_MAX), INT, S))
        assert b.is_everything()

    def test_unknown_minus_neg_inf_sentinel_stays_unknown(self):
        """`x - (-INT64_MAX)` for an unbound `x` subtracts the `-inf` sentinel: still unknown."""
        b = analyzer(ir.Sub(x, ci(-INT64_MAX), INT, S))
        assert b.is_everything()

    def test_finite_range_plus_pos_inf_sentinel_is_not_decided(self):
        """A *finite* range plus a sentinel-valued literal must not collapse to [+inf, +inf].

        `x` in [-5, -1] gives `x + INT64_MAX` in [INT64_MAX-5, INT64_MAX-1]. The
        literal is numerically kPosInf, so the old code took the "operand is
        infinite" path for both endpoints and reported [+inf, +inf] -- an
        interval excluding every value the sum can take, which would decide
        `x + INT64_MAX == 0` as false for the wrong reason. The literal is
        refused instead, so the comparison is kept.
        """
        analyzer.bind(x, -5, 0)  # [-5, -1]
        b = analyzer(ir.Add(x, ci(INT64_MAX), INT, S))
        assert b.min_value <= INT64_MAX - 5
        assert b.max_value >= INT64_MAX - 1
        analyzer.unbind(x)

    def test_subtracting_int64_min_cancels_exactly(self):
        """`x - INT64_MIN` for `x == INT64_MIN` is 0, so `x == INT64_MIN` is decidable.

        INT64_MIN is deliberately not a sentinel (kNegInf is -INT64_MAX), so it
        stays an ordinary exact endpoint. Routing the subtraction through a
        negation used to destroy it -- `-INT64_MIN` is unrepresentable and
        saturated to +inf -- giving [+inf, +inf], an interval excluding 0 that
        folded the true predicate `x == INT64_MIN` to False.
        """
        analyzer.bind(x, INT64_MIN, INT64_MIN + 1)  # [INT64_MIN, INT64_MIN]
        b = analyzer(ir.Sub(x, ci(INT64_MIN), INT, S))
        assert b.min_value == 0
        assert b.max_value == 0
        analyzer.unbind(x)

    def test_multiplication_overflow_is_unknown(self):
        """A product that leaves int64 wraps, so its sign is not known.

        `x` in [2, 3] times 4611686018427387903 has an exact lower corner but an
        overflowing upper one. Reporting the hull of saturated corners gave
        [INT64_MAX-1, INT64_MAX] -- "definitely positive" -- while the emitted
        `arith.muli : i64` wraps to a negative value at `x == 3`.
        """
        analyzer.bind(x, 2, 4)  # [2, 3]
        b = analyzer(ir.Mul(x, ci(4611686018427387903), INT, S))
        assert not b.is_positive()
        analyzer.unbind(x)

    def test_unbounded_operand_can_overflow_so_the_product_is_unknown(self):
        """`x * x` for an unbounded non-negative `x` is not non-negative.

        An endpoint at a sentinel means the operand may be any int64, so the
        emitted `arith.muli : i64` can wrap: at `x == 3037000500` the product is
        negative. Treating an infinite endpoint as an exact value recorded the
        operand's own `[0, +inf]` for the product and "proved" `x * x >= 0`.
        """
        # `update` is inclusive, so this really pins the upper endpoint to the
        # sentinel; `bind` is half-open and would give a large finite bound.
        analyzer.update(x, ConstIntBound(0, ConstIntBound.kPosInf))
        b = analyzer(ir.Mul(x, x, INT, S))
        assert not b.is_non_negative()
        analyzer.unbind(x)

    def test_unbounded_times_a_constant_is_unknown(self):
        """Scaling an unbounded operand is not safe at any multiplier.

        An earlier revision exempted a single unbounded operand on the grounds
        that the product is "linear in the unbounded side". The multiplier's own
        magnitude decides, not the operator: against 4611686018427387903 the
        product already wraps at 3, and even `x * 8` wraps once `x` reaches
        2**60. Precision for a real scaled index is recovered at the source --
        `BoundFromOpSemantics` caps the hardware identity queries, so
        `get_subblock_idx() * 8` stays exact -- rather than by exempting the
        multiply.
        """
        analyzer.update(x, ConstIntBound(0, ConstIntBound.kPosInf))
        assert not analyzer(ir.Mul(x, ci(8), INT, S)).is_non_negative()
        analyzer.unbind(x)

    def test_multiplying_by_one_keeps_an_unbounded_endpoint(self):
        """`x * 1` and `x * -1` compute nothing, so they cannot wrap.

        `-kPosInf` is exactly `kNegInf`, so negation through a unit multiplier
        stays inside the domain and keeps the operand's own bound.
        """
        analyzer.update(x, ConstIntBound(0, ConstIntBound.kPosInf))
        assert analyzer(ir.Mul(x, ci(1), INT, S)).is_non_negative()
        negated = analyzer(ir.Mul(x, ci(-1), INT, S))
        assert negated.max_value == 0
        analyzer.unbind(x)

    def test_scaled_launch_index_stays_exact(self):
        """A hardware launch index is bounded, so products derived from it are exact.

        `get_subblock_idx()` is `[0, 2**31)` rather than `[0, +inf)`: a launch
        grid is not unbounded, and saying it is would make `idx * 8` wrap-prone
        and cost the ragged shard extent `min(max(rows - idx * 8, 0), 8)` its
        fold for nothing.
        """
        idx = ir.Call(ir.get_op("tile.get_subblock_idx"), [], ir.ScalarType(DataType.INDEX), S)
        b = analyzer(ir.Mul(idx, ir.ConstInt(8, DataType.INDEX, S), DataType.INDEX, S))
        assert b.is_non_negative()
        assert b.max_value < ConstIntBound.kPosInf

    def test_adding_zero_keeps_an_unbounded_endpoint(self):
        """`x + 0` and `x - 0` are the identity and cannot overflow.

        The overflow rule above must not widen them: rewritten index
        expressions produce these shapes, and `max(n, 0)` folds only while
        `n - 0` still carries `n`'s own non-negativity.
        """
        analyzer.update(x, ConstIntBound(0, ConstIntBound.kPosInf))
        assert analyzer(ir.Add(x, ci(0), INT, S)).is_non_negative()
        assert analyzer(ir.Sub(x, ci(0), INT, S)).is_non_negative()
        analyzer.unbind(x)

    def test_narrow_addition_wraps_to_the_whole_dtype_range(self):
        """INT8 addition wraps at its own width, so a result past 127 is not positive.

        For `i` in [120, 127], `cast(i, INT8) + 1` is emitted as
        `arith.addi : i8` and yields -128 at `i == 127`. The mathematical
        [121, 128] would prove the sum non-negative.
        """
        analyzer.bind(i, 120, 128)  # [120, 127]
        narrowed = ir.Cast(i, DataType.INT8, S)
        b = analyzer(ir.Add(narrowed, ir.ConstInt(1, DataType.INT8, S), DataType.INT8, S))
        assert b.min_value == -128
        assert b.max_value == 127
        analyzer.unbind(i)

    def test_index_addition_keeps_full_precision(self):
        """A 64-bit INDEX result is never narrowed: the width check is a no-op there.

        Guards the fold this analyzer change exists to protect -- a replicated
        K-loop's `ko + 128` must stay exact so that `ko + 128 == 0` still folds.
        """
        ko = ir.Var("ko", ir.ScalarType(DataType.INDEX), S)
        analyzer.bind(ko, 0, 512)  # [0, 511]
        b = analyzer(ir.Add(ko, ir.ConstInt(128, DataType.INDEX, S), DataType.INDEX, S))
        assert b.min_value == 128
        assert b.max_value == 639
        analyzer.unbind(ko)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
