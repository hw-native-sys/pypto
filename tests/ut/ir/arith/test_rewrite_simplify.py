# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for RewriteSimplifier (pattern-matching algebraic rewrite engine)."""

import pytest
from pypto import DataType, ir
from pypto.arith import RewriteSimplifier

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
        s = RewriteSimplifier()
        assert s is not None

    def test_identity_const(self):
        s = RewriteSimplifier()
        c = ci(42)
        result = s(c)
        assert_is_const_int(result, 42)

    def test_identity_var(self):
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(x)
        assert_same_expr(result, x)

    def test_const_fold(self):
        s = RewriteSimplifier()
        result = s(ir.Add(ci(3), ci(5), INT, S))
        assert_is_const_int(result, 8)

    def test_var_substitution(self):
        s = RewriteSimplifier()
        x = make_var("x")
        s.update(x, ci(10))
        result = s(ir.Add(x, ci(5), INT, S))
        assert_is_const_int(result, 15)


# ============================================================================
# Add rules
# ============================================================================


class TestAddRules:
    def test_add_zero_right(self):
        """x + 0 => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Add(x, ci(0), INT, S))
        assert_same_expr(result, x)

    def test_add_zero_left(self):
        """0 + x => x (via canonicalization c1 + x => x + c1, then x + 0 => x)"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Add(ci(0), x, INT, S))
        assert_same_expr(result, x)

    def test_add_const_reassociation(self):
        """(x + c1) + c2 => x + (c1 + c2)"""
        s = RewriteSimplifier()
        x = make_var("x")
        inner = ir.Add(x, ci(3), INT, S)
        result = s(ir.Add(inner, ci(5), INT, S))
        # Should be x + 8
        assert isinstance(result, ir.Add)
        assert result.left is x
        assert_is_const_int(result.right, 8)

    def test_sub_cancel_add(self):
        """(x - y) + y => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Add(ir.Sub(x, y, INT, S), y, INT, S))
        assert_same_expr(result, x)

    def test_add_reverse_cancel(self):
        """x + (y - x) => y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Add(x, ir.Sub(y, x, INT, S), INT, S))
        assert_same_expr(result, y)

    def test_add_self(self):
        """x + x => x * 2"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Add(x, x, INT, S))
        assert isinstance(result, ir.Mul)
        assert result.left is x
        assert_is_const_int(result.right, 2)

    def test_floordiv_mul_floormod(self):
        """floordiv(x, y) * y + floormod(x, y) => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        c = ci(4)
        fd = ir.FloorDiv(x, c, INT, S)
        fm = ir.FloorMod(x, c, INT, S)
        result = s(ir.Add(ir.Mul(fd, c, INT, S), fm, INT, S))
        assert_same_expr(result, x)

    def test_min_sub_add(self):
        """min(x - z, y) + z => min(x, y + z)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Add(ir.Min(ir.Sub(x, z, INT, S), y, INT, S), z, INT, S))
        assert isinstance(result, ir.Min)

    def test_max_min_sum(self):
        """max(x, y) + min(x, y) => x + y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Add(ir.Max(x, y, INT, S), ir.Min(x, y, INT, S), INT, S))
        assert isinstance(result, ir.Add)
        assert result.left is x
        assert result.right is y


# ============================================================================
# Sub rules
# ============================================================================


class TestSubRules:
    def test_sub_self(self):
        """x - x => 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Sub(x, x, INT, S))
        assert_is_const_int(result, 0)

    def test_sub_add_cancel_left(self):
        """(x + y) - y => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Sub(ir.Add(x, y, INT, S), y, INT, S))
        assert_same_expr(result, x)

    def test_sub_add_cancel_right(self):
        """(x + y) - x => y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Sub(ir.Add(x, y, INT, S), x, INT, S))
        assert_same_expr(result, y)

    def test_sub_const_reassociation(self):
        """(x + c1) - c2 => x + (c1 - c2)"""
        s = RewriteSimplifier()
        x = make_var("x")
        inner = ir.Add(x, ci(10), INT, S)
        result = s(ir.Sub(inner, ci(3), INT, S))
        assert isinstance(result, ir.Add)
        assert result.left is x
        assert_is_const_int(result.right, 7)

    def test_sub_extract_floormod(self):
        """x - floordiv(x, c1) * c1 => floormod(x, c1) when c1 > 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        c = ci(4)
        fd = ir.FloorDiv(x, c, INT, S)
        result = s(ir.Sub(x, ir.Mul(fd, c, INT, S), INT, S))
        assert isinstance(result, ir.FloorMod)
        assert result.left is x
        assert_is_const_int(result.right, 4)

    def test_sub_const_cross(self):
        """(c1 - x) - (c2 - y) => (y - x) + (c1 - c2)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Sub(ir.Sub(ci(10), x, INT, S), ir.Sub(ci(3), y, INT, S), INT, S))
        assert isinstance(result, ir.Add)

    def test_sub_min_cancel(self):
        """min(x, y) - min(y, x) => 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Sub(ir.Min(x, y, INT, S), ir.Min(y, x, INT, S), INT, S))
        assert_is_const_int(result, 0)

    def test_sub_max_cancel(self):
        """max(x, y) - max(y, x) => 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Sub(ir.Max(x, y, INT, S), ir.Max(y, x, INT, S), INT, S))
        assert_is_const_int(result, 0)

    def test_sub_min_add_extract(self):
        """min(x+y, z) - x => min(y, z-x)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Sub(ir.Min(ir.Add(x, y, INT, S), z, INT, S), x, INT, S))
        assert isinstance(result, ir.Min)

    def test_sub_floordiv_extended(self):
        """x - floordiv(x+y, c1)*c1 => floormod(x+y, c1) - y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        xy = ir.Add(x, y, INT, S)
        fd = ir.FloorDiv(xy, ci(4), INT, S)
        result = s(ir.Sub(x, ir.Mul(fd, ci(4), INT, S), INT, S))
        # Should contain floormod(x+y, 4)
        assert isinstance(result, ir.Sub)

    def test_sub_canonicalize_nested(self):
        """x - (y - z) => (x + z) - y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Sub(x, ir.Sub(y, z, INT, S), INT, S))
        assert isinstance(result, ir.Sub)


# ============================================================================
# Mul rules
# ============================================================================


class TestMulRules:
    def test_mul_zero(self):
        """x * 0 => 0 (via const fold when x is const, or mul rule)"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Mul(x, ci(0), INT, S))
        assert_is_const_int(result, 0)

    def test_mul_one(self):
        """x * 1 => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Mul(x, ci(1), INT, S))
        assert_same_expr(result, x)

    def test_mul_const_reassociation(self):
        """(x * c1) * c2 => x * (c1 * c2)"""
        s = RewriteSimplifier()
        x = make_var("x")
        inner = ir.Mul(x, ci(3), INT, S)
        result = s(ir.Mul(inner, ci(5), INT, S))
        assert isinstance(result, ir.Mul)
        assert result.left is x
        assert_is_const_int(result.right, 15)

    def test_mul_min_max(self):
        """min(x,y) * max(x,y) => x * y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Mul(ir.Min(x, y, INT, S), ir.Max(x, y, INT, S), INT, S))
        assert isinstance(result, ir.Mul)
        assert result.left is x
        assert result.right is y

    def test_mul_neg_flip(self):
        """(x - y) * (-3) => (y - x) * 3"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Mul(ir.Sub(x, y, INT, S), ci(-3), INT, S))
        assert isinstance(result, ir.Mul)
        assert isinstance(result.left, ir.Sub)
        assert result.left.left is y
        assert result.left.right is x
        assert_is_const_int(result.right, 3)


# ============================================================================
# FloorDiv rules
# ============================================================================


class TestFloorDivRules:
    def test_floordiv_by_one(self):
        """floordiv(x, 1) => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(x, ci(1), INT, S))
        assert_same_expr(result, x)

    def test_floordiv_factor(self):
        """floordiv(x * c1, c2) => x * (c1 / c2) when c1 % c2 == 0 and c2 > 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(ir.Mul(x, ci(6), INT, S), ci(3), INT, S))
        assert isinstance(result, ir.Mul)
        assert result.left is x
        assert_is_const_int(result.right, 2)

    def test_floordiv_nested(self):
        """floordiv(floordiv(x, c1), c2) => floordiv(x, c1*c2) when c1 > 0, c2 > 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        inner = ir.FloorDiv(x, ci(3), INT, S)
        result = s(ir.FloorDiv(inner, ci(4), INT, S))
        assert isinstance(result, ir.FloorDiv)
        assert result.left is x
        assert_is_const_int(result.right, 12)

    def test_floordiv_add_offset(self):
        """floordiv(x + c1, c2) => floordiv(x, c2) + c1/c2 when c1 % c2 == 0, c2 > 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(ir.Add(x, ci(6), INT, S), ci(3), INT, S))
        assert isinstance(result, ir.Add)
        assert isinstance(result.left, ir.FloorDiv)
        assert_is_const_int(result.right, 2)

    def test_floordiv_self(self):
        """floordiv(x, x) => 1 is dormant in standalone mode (requires x != 0 proof)"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(x, x, INT, S))
        # In standalone mode, TryCompare returns kUnknown so the rule doesn't fire
        assert isinstance(result, ir.FloorDiv)

    def test_floordiv_mul_self(self):
        """floordiv(x * c1, x) => c1 is dormant in standalone mode (requires x != 0 proof)"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(ir.Mul(x, ci(5), INT, S), x, INT, S))
        # In standalone mode, TryCompare returns kUnknown so the rule doesn't fire
        assert isinstance(result, ir.FloorDiv)

    def test_floordiv_nested_offset(self):
        """floordiv(floordiv(x, c1) + c2, c3) => floordiv(x + c1*c2, c1*c3)"""
        s = RewriteSimplifier()
        x = make_var("x")
        inner = ir.Add(ir.FloorDiv(x, ci(3), INT, S), ci(2), INT, S)
        result = s(ir.FloorDiv(inner, ci(4), INT, S))
        assert isinstance(result, ir.FloorDiv)
        # Should be floordiv(x + 6, 12)
        assert isinstance(result.left, ir.Add)
        assert_is_const_int(result.right, 12)

    def test_floordiv_sub_floormod(self):
        """floordiv(x - floormod(x, c1), c1) => floordiv(x, c1)"""
        s = RewriteSimplifier()
        x = make_var("x")
        fm = ir.FloorMod(x, ci(4), INT, S)
        result = s(ir.FloorDiv(ir.Sub(x, fm, INT, S), ci(4), INT, S))
        assert isinstance(result, ir.FloorDiv)
        assert result.left is x
        assert_is_const_int(result.right, 4)


# ============================================================================
# FloorMod rules
# ============================================================================


class TestFloorModRules:
    def test_floormod_multiple(self):
        """floormod(x * c1, c2) => 0 when c1 % c2 == 0 and c2 > 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.FloorMod(ir.Mul(x, ci(6), INT, S), ci(3), INT, S))
        assert_is_const_int(result, 0)

    def test_floormod_add_multiple_offset(self):
        """floormod(x + c1, c2) => floormod(x, c2) when c1 % c2 == 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.FloorMod(ir.Add(x, ci(6), INT, S), ci(3), INT, S))
        assert isinstance(result, ir.FloorMod)
        assert result.left is x
        assert_is_const_int(result.right, 3)

    def test_floormod_mul_var(self):
        """floormod(x * y, y) => 0 is dormant in standalone mode (requires y != 0 proof)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.FloorMod(ir.Mul(x, y, INT, S), y, INT, S))
        # In standalone mode, TryCompare returns kUnknown so the rule doesn't fire
        assert isinstance(result, ir.FloorMod)

    def test_floormod_coeff_reduction(self):
        """floormod(x * c1, c2) => floormod(x * floormod(c1, c2), c2)"""
        s = RewriteSimplifier()
        x = make_var("x")
        # floormod(x*7, 3) => floormod(x*1, 3) => floormod(x, 3)
        result = s(ir.FloorMod(ir.Mul(x, ci(7), INT, S), ci(3), INT, S))
        assert isinstance(result, ir.FloorMod)


# ============================================================================
# Min/Max rules
# ============================================================================


class TestMinMaxRules:
    def test_min_self(self):
        """min(x, x) => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Min(x, x, INT, S))
        assert_same_expr(result, x)

    def test_max_self(self):
        """max(x, x) => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Max(x, x, INT, S))
        assert_same_expr(result, x)

    def test_min_factor_sub(self):
        """min(y - x, z - x) => min(y, z) - x"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Min(ir.Sub(y, x, INT, S), ir.Sub(z, x, INT, S), INT, S))
        assert isinstance(result, ir.Sub)
        assert isinstance(result.left, ir.Min)
        assert result.right is x

    def test_max_factor_add(self):
        """max(x + y, x + z) => x + max(y, z)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Max(ir.Add(x, y, INT, S), ir.Add(x, z, INT, S), INT, S))
        assert isinstance(result, ir.Add)
        assert result.left is x
        assert isinstance(result.right, ir.Max)

    def test_min_absorption(self):
        """min(max(x, y), y) => y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Min(ir.Max(x, y, INT, S), y, INT, S))
        assert_same_expr(result, y)

    def test_max_absorption(self):
        """max(min(x, y), y) => y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Max(ir.Min(x, y, INT, S), y, INT, S))
        assert_same_expr(result, y)

    def test_min_const_reassociation(self):
        """min(min(x, c1), c2) => min(x, min(c1, c2))"""
        s = RewriteSimplifier()
        x = make_var("x")
        inner = ir.Min(x, ci(5), INT, S)
        result = s(ir.Min(inner, ci(3), INT, S))
        assert isinstance(result, ir.Min)
        assert result.left is x
        assert_is_const_int(result.right, 3)

    def test_min_const_offsets(self):
        """min(x + 3, x + 7) => x + 3"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Min(ir.Add(x, ci(3), INT, S), ir.Add(x, ci(7), INT, S), INT, S))
        assert isinstance(result, ir.Add)
        assert result.left is x
        assert_is_const_int(result.right, 3)

    def test_max_const_offsets(self):
        """max(x + 3, x + 7) => x + 7"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Max(ir.Add(x, ci(3), INT, S), ir.Add(x, ci(7), INT, S), INT, S))
        assert isinstance(result, ir.Add)
        assert result.left is x
        assert_is_const_int(result.right, 7)

    def test_min_nested_collapse(self):
        """min(min(x, y), x) => min(x, y)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        inner = ir.Min(x, y, INT, S)
        result = s(ir.Min(inner, x, INT, S))
        assert isinstance(result, ir.Min)

    def test_max_nested_collapse(self):
        """max(max(x, y), x) => max(x, y)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        inner = ir.Max(x, y, INT, S)
        result = s(ir.Max(inner, x, INT, S))
        assert isinstance(result, ir.Max)

    def test_min_cross_distribution(self):
        """min(max(x,y), max(x,z)) => max(min(y,z), x)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Min(ir.Max(x, y, INT, S), ir.Max(x, z, INT, S), INT, S))
        assert isinstance(result, ir.Max)
        assert isinstance(result.left, ir.Min)

    def test_max_cross_distribution(self):
        """max(min(x,y), min(x,z)) => min(max(y,z), x)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Max(ir.Min(x, y, INT, S), ir.Min(x, z, INT, S), INT, S))
        assert isinstance(result, ir.Min)
        assert isinstance(result.left, ir.Max)

    def test_min_scaling(self):
        """min(x * 3, y * 3) => min(x, y) * 3"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Min(ir.Mul(x, ci(3), INT, S), ir.Mul(y, ci(3), INT, S), INT, S))
        assert isinstance(result, ir.Mul)
        assert isinstance(result.left, ir.Min)
        assert_is_const_int(result.right, 3)

    def test_max_scaling(self):
        """max(x * 3, y * 3) => max(x, y) * 3"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Max(ir.Mul(x, ci(3), INT, S), ir.Mul(y, ci(3), INT, S), INT, S))
        assert isinstance(result, ir.Mul)
        assert isinstance(result.left, ir.Max)
        assert_is_const_int(result.right, 3)


# ============================================================================
# Comparison rules
# ============================================================================


class TestComparisonRules:
    def test_eq_self(self):
        """x == x => true"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Eq(x, x, DataType.BOOL, S))
        assert_is_const_bool(result, True)

    def test_ne_self(self):
        """x != x => false"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Ne(x, x, DataType.BOOL, S))
        assert_is_const_bool(result, False)

    def test_lt_self(self):
        """x < x => false"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Lt(x, x, DataType.BOOL, S))
        assert_is_const_bool(result, False)

    def test_le_self(self):
        """x <= x => true"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Le(x, x, DataType.BOOL, S))
        assert_is_const_bool(result, True)

    def test_gt_self(self):
        """x > x => false"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Gt(x, x, DataType.BOOL, S))
        assert_is_const_bool(result, False)

    def test_ge_self(self):
        """x >= x => true"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Ge(x, x, DataType.BOOL, S))
        assert_is_const_bool(result, True)

    def test_gt_delegation(self):
        """x > y delegates to y < x (Gt -> Lt delegation)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        # x + y > x + z should simplify to z < y (via Lt delegation)
        z = make_var("z")
        result = s(ir.Gt(ir.Add(x, y, INT, S), ir.Add(x, z, INT, S), DataType.BOOL, S))
        assert isinstance(result, ir.Lt)

    def test_ge_delegation(self):
        """x >= y delegates to y <= x (Ge -> Le delegation)"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Ge(ir.Add(x, y, INT, S), ir.Add(x, z, INT, S), DataType.BOOL, S))
        assert isinstance(result, ir.Le)

    def test_lt_self_offset(self):
        """x < x + z => 0 < z"""
        s = RewriteSimplifier()
        x = make_var("x")
        z = make_var("z")
        result = s(ir.Lt(x, ir.Add(x, z, INT, S), DataType.BOOL, S))
        assert isinstance(result, ir.Lt)
        assert_is_const_int(result.left, 0)
        assert result.right is z

    def test_lt_floordiv(self):
        """floordiv(x, c1) < c2 => x < c1*c2 when c1 > 0"""
        s = RewriteSimplifier()
        x = make_var("x")
        fd = ir.FloorDiv(x, ci(4), INT, S)
        result = s(ir.Lt(fd, ci(3), DataType.BOOL, S))
        assert isinstance(result, ir.Lt)
        assert result.left is x
        assert_is_const_int(result.right, 12)

    def test_lt_min_decompose(self):
        """min(x, y) < z => x < z || y < z"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Lt(ir.Min(x, y, INT, S), z, DataType.BOOL, S))
        assert isinstance(result, ir.Or)

    def test_lt_max_decompose(self):
        """max(x, y) < z => x < z && y < z"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Lt(ir.Max(x, y, INT, S), z, DataType.BOOL, S))
        assert isinstance(result, ir.And)

    def test_le_sub_cancel(self):
        """x - y <= x - z => z <= y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        result = s(ir.Le(ir.Sub(x, y, INT, S), ir.Sub(x, z, INT, S), DataType.BOOL, S))
        assert isinstance(result, ir.Le)
        assert result.left is z
        assert result.right is y

    def test_le_mul_positive(self):
        """x * 3 <= y * 3 => x <= y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Le(ir.Mul(x, ci(3), INT, S), ir.Mul(y, ci(3), INT, S), DataType.BOOL, S))
        assert isinstance(result, ir.Le)
        assert result.left is x
        assert result.right is y


# ============================================================================
# Boolean / Not rules
# ============================================================================


class TestBooleanRules:
    def test_double_negation(self):
        """!!x => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Not(ir.Not(x, DataType.BOOL, S), DataType.BOOL, S))
        assert_same_expr(result, x)

    def test_not_lt(self):
        """!(x < y) => y <= x"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Not(ir.Lt(x, y, DataType.BOOL, S), DataType.BOOL, S))
        assert isinstance(result, ir.Le)
        assert result.left is y
        assert result.right is x

    def test_not_le(self):
        """!(x <= y) => y < x"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Not(ir.Le(x, y, DataType.BOOL, S), DataType.BOOL, S))
        assert isinstance(result, ir.Lt)
        assert result.left is y
        assert result.right is x

    def test_not_eq(self):
        """!(x == y) => x != y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Not(ir.Eq(x, y, DataType.BOOL, S), DataType.BOOL, S))
        assert isinstance(result, ir.Ne)

    def test_and_contradiction(self):
        """x && !x => false"""
        s = RewriteSimplifier()
        x = make_var("x")
        not_x = ir.Not(x, DataType.BOOL, S)
        result = s(ir.And(x, not_x, DataType.BOOL, S))
        assert_is_const_bool(result, False)

    def test_or_tautology(self):
        """x || !x => true"""
        s = RewriteSimplifier()
        x = make_var("x")
        not_x = ir.Not(x, DataType.BOOL, S)
        result = s(ir.Or(x, not_x, DataType.BOOL, S))
        assert_is_const_bool(result, True)

    def test_and_idempotent(self):
        """x && x => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.And(x, x, DataType.BOOL, S))
        assert_same_expr(result, x)

    def test_or_idempotent(self):
        """x || x => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Or(x, x, DataType.BOOL, S))
        assert_same_expr(result, x)

    def test_and_eq_ne_contradiction(self):
        """x == y && x != y => false"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        eq = ir.Eq(x, y, DataType.BOOL, S)
        ne = ir.Ne(x, y, DataType.BOOL, S)
        result = s(ir.And(eq, ne, DataType.BOOL, S))
        assert_is_const_bool(result, False)

    def test_and_le_lt_contradiction(self):
        """x <= y && y < x => false"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.And(ir.Le(x, y, DataType.BOOL, S), ir.Lt(y, x, DataType.BOOL, S), DataType.BOOL, S))
        assert_is_const_bool(result, False)

    def test_and_range_contradiction(self):
        """x < 3 && 5 < x => false"""
        s = RewriteSimplifier()
        x = make_var("x")
        lt1 = ir.Lt(x, ci(3), DataType.BOOL, S)
        lt2 = ir.Lt(ci(5), x, DataType.BOOL, S)
        result = s(ir.And(lt1, lt2, DataType.BOOL, S))
        assert_is_const_bool(result, False)

    def test_or_eq_ne_tautology(self):
        """x == y || x != y => true"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        eq = ir.Eq(x, y, DataType.BOOL, S)
        ne = ir.Ne(x, y, DataType.BOOL, S)
        result = s(ir.Or(eq, ne, DataType.BOOL, S))
        assert_is_const_bool(result, True)

    def test_or_le_lt_tautology(self):
        """x <= y || y < x => true"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Or(ir.Le(x, y, DataType.BOOL, S), ir.Lt(y, x, DataType.BOOL, S), DataType.BOOL, S))
        assert_is_const_bool(result, True)

    def test_or_lt_eq_to_le(self):
        """x < y || x == y => x <= y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Or(ir.Lt(x, y, DataType.BOOL, S), ir.Eq(x, y, DataType.BOOL, S), DataType.BOOL, S))
        assert isinstance(result, ir.Le)
        assert result.left is x
        assert result.right is y

    def test_or_lt_exclusive(self):
        """x < y || y < x => x != y"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Or(ir.Lt(x, y, DataType.BOOL, S), ir.Lt(y, x, DataType.BOOL, S), DataType.BOOL, S))
        assert isinstance(result, ir.Ne)


# ============================================================================
# Neg rules
# ============================================================================


class TestNegRules:
    def test_neg_neg(self):
        """neg(neg(x)) => x"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.Neg(ir.Neg(x, INT, S), INT, S))
        assert_same_expr(result, x)

    def test_neg_sub(self):
        """neg(x - y) => y - x"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        result = s(ir.Neg(ir.Sub(x, y, INT, S), INT, S))
        assert isinstance(result, ir.Sub)
        assert result.left is y
        assert result.right is x


# ============================================================================
# Constraint / substitution
# ============================================================================


class TestConstraintAndSubstitution:
    def test_update_substitution(self):
        s = RewriteSimplifier()
        x = make_var("x")
        s.update(x, ci(7))
        assert_is_const_int(s(x), 7)

    def test_enter_constraint(self):
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        constraint = ir.Eq(x, y, DataType.BOOL, S)
        exit_fn = s.enter_constraint(constraint)
        assert callable(exit_fn)
        exit_fn()


# ============================================================================
# Combined / multi-step rewrites
# ============================================================================


class TestCombinedRewrites:
    def test_nested_add_sub(self):
        """(x + y) - y + z => x + z"""
        s = RewriteSimplifier()
        x = make_var("x")
        y = make_var("y")
        z = make_var("z")
        expr = ir.Add(ir.Sub(ir.Add(x, y, INT, S), y, INT, S), z, INT, S)
        result = s(expr)
        assert isinstance(result, ir.Add)
        assert result.left is x
        assert result.right is z

    def test_complex_mul_div(self):
        """floordiv(x * 12, 4) => x * 3"""
        s = RewriteSimplifier()
        x = make_var("x")
        result = s(ir.FloorDiv(ir.Mul(x, ci(12), INT, S), ci(4), INT, S))
        assert isinstance(result, ir.Mul)
        assert result.left is x
        assert_is_const_int(result.right, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
