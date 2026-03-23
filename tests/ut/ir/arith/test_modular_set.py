# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for ModularSetAnalyzer (modular arithmetic tracking through expression trees)."""

import pytest
from pypto import DataType, ir
from pypto.arith import ModularSet, ModularSetAnalyzer

S = ir.Span.unknown()
INT = DataType.INT64
BOOL = DataType.BOOL


def make_var(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(INT), S)


def ci(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, INT, S)


# ============================================================================
# Basic structure and leaf nodes
# ============================================================================


class TestModularSetBasics:
    def test_const_int_exact(self):
        analyzer = ModularSetAnalyzer()
        m = analyzer(ci(42))
        assert m.coeff == 0
        assert m.base == 42
        assert m.is_exact()

    def test_const_int_zero(self):
        analyzer = ModularSetAnalyzer()
        m = analyzer(ci(0))
        assert m.coeff == 0
        assert m.base == 0

    def test_const_int_negative(self):
        analyzer = ModularSetAnalyzer()
        m = analyzer(ci(-7))
        assert m.coeff == 0
        assert m.base == -7

    def test_unknown_var(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        m = analyzer(x)
        assert m.coeff == 1
        assert m.base == 0
        assert m.is_everything()

    def test_updated_var(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(4, 1))
        m = analyzer(x)
        assert m.coeff == 4
        assert m.base == 1

    def test_repr(self):
        m = ModularSet(4, 1)
        assert "coeff=4" in repr(m)
        assert "base=1" in repr(m)


# ============================================================================
# Arithmetic operations
# ============================================================================


class TestModularSetArithmetic:
    def test_add_consts(self):
        analyzer = ModularSetAnalyzer()
        m = analyzer(ir.Add(ci(3), ci(5), INT, S))
        assert m.coeff == 0
        assert m.base == 8

    def test_add_vars(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        y = make_var("y")
        analyzer.update(x, ModularSet(4, 1))  # x = 4k + 1
        analyzer.update(y, ModularSet(6, 2))  # y = 6j + 2
        m = analyzer(ir.Add(x, y, INT, S))
        # coeff = gcd(4, 6) = 2, base = (1 + 2) % 2 = 1
        assert m.coeff == 2
        assert m.base == 1

    def test_sub_consts(self):
        analyzer = ModularSetAnalyzer()
        m = analyzer(ir.Sub(ci(10), ci(3), INT, S))
        assert m.coeff == 0
        assert m.base == 7

    def test_sub_vars(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        y = make_var("y")
        analyzer.update(x, ModularSet(4, 3))  # x = 4k + 3
        analyzer.update(y, ModularSet(6, 1))  # y = 6j + 1
        m = analyzer(ir.Sub(x, y, INT, S))
        # coeff = gcd(4, 6) = 2, base = (3 - 1) % 2 = 0
        assert m.coeff == 2
        assert m.base == 0

    def test_mul_const_by_var(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        # 2 is {coeff=0, base=2}, x is {coeff=1, base=0}
        # pq = 0*1 = 0, pm = 0*0 = 0, qn = 2*1 = 2
        # coeff = gcd(0, gcd(0, 2)) = 2, base = 2*0 = 0
        m = analyzer(ir.Mul(ci(2), x, INT, S))
        assert m.coeff == 2
        assert m.base == 0

    def test_mul_vars(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        y = make_var("y")
        analyzer.update(x, ModularSet(2, 1))  # x = 2k + 1
        analyzer.update(y, ModularSet(3, 0))  # y = 3j
        # (2k+1)(3j) = 6kj + 3j: pq=6, pm=2*0=0, qn=1*3=3
        # coeff = gcd(6, gcd(0, 3)) = gcd(6, 3) = 3, base = 1*0 = 0
        m = analyzer(ir.Mul(x, y, INT, S))
        assert m.coeff == 3
        assert m.base == 0

    def test_neg(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(4, 1))  # x = 4k + 1
        m = analyzer(ir.Neg(x, INT, S))
        # neg: coeff=4, base=-1 → normalized to base=3
        assert m.coeff == 4
        assert m.base == 3


# ============================================================================
# Floor division and modulo
# ============================================================================


class TestModularSetFloorDivMod:
    def test_floordiv_exact(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(8, 0))  # x = 8k
        m = analyzer(ir.FloorDiv(x, ci(4), INT, S))
        # coeff % val == 0: 8 % 4 == 0, base == 0
        # result: coeff = |8/4| = 2, base = 0
        assert m.coeff == 2
        assert m.base == 0

    def test_floordiv_with_base(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(8, 4))  # x = 8k + 4
        m = analyzer(ir.FloorDiv(x, ci(4), INT, S))
        # coeff % val == 0: 8 % 4 == 0, base > 0, val > 0
        # result: coeff = 8/4 = 2, base = 4/4 = 1
        assert m.coeff == 2
        assert m.base == 1

    def test_floordiv_not_divisible(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(3, 1))  # x = 3k + 1
        m = analyzer(ir.FloorDiv(x, ci(2), INT, S))
        # 3 % 2 != 0, so everything
        assert m.coeff == 1
        assert m.base == 0

    def test_floormod_basic(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(4, 1))  # x = 4k + 1
        m = analyzer(ir.FloorMod(x, ci(2), INT, S))
        # coeff = gcd(4, 2) = 2, base % coeff = 1 % 2 = 1
        # a.base (1) > 0 → returns Entry(2, 1)
        assert m.coeff == 2
        assert m.base == 1

    def test_floormod_aligned(self):
        """x = 6k → x % 3 should have coeff 3, base 0."""
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(6, 0))  # x = 6k
        m = analyzer(ir.FloorMod(x, ci(3), INT, S))
        # coeff = gcd(6, 3) = 3, base = 0 % 3 = 0
        assert m.coeff == 3
        assert m.base == 0

    def test_two_x_mod_two(self):
        """(2*x) % 2 → coeff=2, meaning always 0 mod 2."""
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        expr = ir.FloorMod(ir.Mul(ci(2), x, INT, S), ci(2), INT, S)
        m = analyzer(expr)
        # 2*x has modular set {2, 0}
        # floormod by 2: gcd(2, 2) = 2, base = 0 % 2 = 0
        assert m.coeff == 2
        assert m.base == 0


# ============================================================================
# Min, Max
# ============================================================================


class TestModularSetMinMax:
    def test_min(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        y = make_var("y")
        analyzer.update(x, ModularSet(4, 1))
        analyzer.update(y, ModularSet(4, 1))
        m = analyzer(ir.Min(x, y, INT, S))
        # Union of {4, 1} and {4, 1} = {4, 1}
        assert m.coeff == 4
        assert m.base == 1

    def test_max_different(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        y = make_var("y")
        analyzer.update(x, ModularSet(4, 0))
        analyzer.update(y, ModularSet(4, 2))
        m = analyzer(ir.Max(x, y, INT, S))
        # Union of {4, 0} and {4, 2}: coeff=4, base0=0, base1=2
        # base0 != base1 → gcd(gcd(0, 2), 4) = gcd(2, 4) = 2, base=0
        assert m.coeff == 2
        assert m.base == 0


# ============================================================================
# Bitwise operations
# ============================================================================


class TestModularSetBitwise:
    def test_bitand_power_of_two_mask(self):
        """x & 7 is equivalent to x % 8."""
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(16, 3))  # x = 16k + 3
        m = analyzer(ir.BitAnd(x, ci(7), INT, S))
        # mask=7, mask+1=8 is power of 2
        # FloorModByConst(x, 8): gcd(16, 8) = 8, base = 3 % 8 = 3
        assert m.coeff == 8
        assert m.base == 3

    def test_shift_right(self):
        """x >> 2 is equivalent to x // 4."""
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(8, 0))  # x = 8k
        m = analyzer(ir.BitShiftRight(x, ci(2), INT, S))
        # FloorDivByConst(x, 4): 8 % 4 == 0, base == 0
        # result: coeff = 2, base = 0
        assert m.coeff == 2
        assert m.base == 0

    def test_shift_left(self):
        """x << 3 is equivalent to x * 8."""
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(3, 1))  # x = 3k + 1
        m = analyzer(ir.BitShiftLeft(x, ci(3), INT, S))
        # x * 8: coeff = gcd(0, 3*8) = 24, base = 1*8 = 8
        # Normalized: base = 8 % 24 = 8
        assert m.coeff == 24
        assert m.base == 8


# ============================================================================
# Cast
# ============================================================================


class TestModularSetCast:
    def test_cast_preserves_modular_info(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        analyzer.update(x, ModularSet(4, 1))
        cast_expr = ir.Cast(x, DataType.INT32, S)
        m = analyzer(cast_expr)
        assert m.coeff == 4
        assert m.base == 1


# ============================================================================
# Constraint entry
# ============================================================================


class TestModularSetConstraint:
    def test_enter_floormod_constraint(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        # Constraint: floormod(x, 4) == 1 → x = 4k + 1
        constraint = ir.Eq(ir.FloorMod(x, ci(4), INT, S), ci(1), BOOL, S)
        recover = analyzer.enter_constraint(constraint)
        m = analyzer(x)
        assert m.coeff == 4
        assert m.base == 1
        # Recover
        if recover is not None:
            recover()
        m = analyzer(x)
        assert m.is_everything()

    def test_enter_eq_constraint(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        # Constraint: x == 5 → exact value
        constraint = ir.Eq(x, ci(5), BOOL, S)
        recover = analyzer.enter_constraint(constraint)
        m = analyzer(x)
        assert m.coeff == 0
        assert m.base == 5
        if recover is not None:
            recover()

    def test_no_constraint_match(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        # x > 0 doesn't give modular info
        constraint = ir.Gt(x, ci(0), BOOL, S)
        recover = analyzer.enter_constraint(constraint)
        assert recover is None


# ============================================================================
# Input validation
# ============================================================================


class TestModularSetValidation:
    def test_negative_coeff_raises(self):
        analyzer = ModularSetAnalyzer()
        x = make_var("x")
        with pytest.raises(Exception, match="non-negative"):
            analyzer.update(x, ModularSet(-2, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
