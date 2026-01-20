# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for binary operation expressions: Add, Sub, Mul, Div, Mod, etc."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestArithmeticOps:
    """Tests for arithmetic binary operations."""

    def test_add_creation(self):
        """Test creating an Add expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        add_expr = ir.Add(x, y, dtype, span)

        assert cast(ir.Var, add_expr.left).name == "x"
        assert cast(ir.Var, add_expr.right).name == "y"

    def test_add_is_binary_expr(self):
        """Test that Add is an instance of BinaryExpr."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        add_expr = ir.Add(x, y, dtype, span)

        assert isinstance(add_expr, ir.BinaryExpr)
        assert isinstance(add_expr, ir.Expr)
        assert isinstance(add_expr, ir.IRNode)

    def test_sub_creation(self):
        """Test creating a Sub expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        sub_expr = ir.Sub(x, y, dtype, span)

        assert cast(ir.Var, sub_expr.left).name == "x"
        assert cast(ir.Var, sub_expr.right).name == "y"

    def test_mul_creation(self):
        """Test creating a Mul expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        mul_expr = ir.Mul(x, y, dtype, span)

        assert cast(ir.Var, mul_expr.left).name == "x"
        assert cast(ir.Var, mul_expr.right).name == "y"

    def test_floatdiv_creation(self):
        """Test creating a FloatDiv expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        div_expr = ir.FloatDiv(x, y, dtype, span)

        assert cast(ir.Var, div_expr.left).name == "x"
        assert cast(ir.Var, div_expr.right).name == "y"

    def test_floormod_creation(self):
        """Test creating a FloorMod expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        mod_expr = ir.FloorMod(x, y, dtype, span)

        assert cast(ir.Var, mod_expr.left).name == "x"
        assert cast(ir.Var, mod_expr.right).name == "y"

    def test_floordiv_creation(self):
        """Test creating a FloorDiv expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        floordiv_expr = ir.FloorDiv(x, y, dtype, span)

        assert cast(ir.Var, floordiv_expr.left).name == "x"
        assert cast(ir.Var, floordiv_expr.right).name == "y"

    def test_pow(self):
        """Test creating Pow expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        pow_expr = ir.Pow(x, y, dtype, span)
        assert cast(ir.Var, pow_expr.left).name == "x"
        assert cast(ir.Var, pow_expr.right).name == "y"

    def test_min_max(self):
        """Test creating Min and Max expressions."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        min_expr = ir.Min(x, y, dtype, span)
        assert cast(ir.Var, min_expr.left).name == "x"

        max_expr = ir.Max(x, y, dtype, span)
        assert cast(ir.Var, max_expr.left).name == "x"


class TestLogicalOps:
    """Tests for logical binary operations."""

    def test_logical_ops(self):
        """Test creating logical expressions."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        and_expr = ir.And(x, y, dtype, span)
        assert cast(ir.Var, and_expr.left).name == "x"

        or_expr = ir.Or(x, y, dtype, span)
        assert cast(ir.Var, or_expr.left).name == "x"

        xor_expr = ir.Xor(x, y, dtype, span)
        assert cast(ir.Var, xor_expr.left).name == "x"


class TestBitwiseOps:
    """Tests for bitwise binary operations."""

    def test_bitwise_ops(self):
        """Test creating bitwise expressions."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        bitand_expr = ir.BitAnd(x, y, dtype, span)
        assert cast(ir.Var, bitand_expr.left).name == "x"

        bitor_expr = ir.BitOr(x, y, dtype, span)
        assert cast(ir.Var, bitor_expr.left).name == "x"

        bitxor_expr = ir.BitXor(x, y, dtype, span)
        assert cast(ir.Var, bitxor_expr.left).name == "x"

        shl_expr = ir.BitShiftLeft(x, y, dtype, span)
        assert cast(ir.Var, shl_expr.left).name == "x"

        shr_expr = ir.BitShiftRight(x, y, dtype, span)
        assert cast(ir.Var, shr_expr.left).name == "x"


class TestUnaryOps:
    """Tests for unary operations."""

    def test_neg_creation(self):
        """Test creating a Neg expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        neg_expr = ir.Neg(x, dtype, span)

        assert cast(ir.Var, neg_expr.operand).name == "x"
        assert isinstance(neg_expr, ir.UnaryExpr)
        assert isinstance(neg_expr, ir.Expr)

    def test_abs_creation(self):
        """Test creating an Abs expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        abs_expr = ir.Abs(x, dtype, span)

        assert cast(ir.Var, abs_expr.operand).name == "x"

    def test_not_creation(self):
        """Test creating a Not expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        not_expr = ir.Not(x, dtype, span)

        assert cast(ir.Var, not_expr.operand).name == "x"

    def test_bitnot_creation(self):
        """Test creating a BitNot expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        bitnot_expr = ir.BitNot(x, dtype, span)

        assert cast(ir.Var, bitnot_expr.operand).name == "x"


class TestNestedExpressions:
    """Tests for nested expression trees."""

    def test_simple_nested_expression(self):
        """Test building a simple nested expression: (x + 5) * 2."""
        span = ir.Span("test.py", 1, 1, 1, 20)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        c5 = ir.ConstInt(5, dtype, span)
        c2 = ir.ConstInt(2, dtype, span)

        add_expr = ir.Add(x, c5, dtype, span)
        mul_expr = ir.Mul(add_expr, c2, dtype, span)

        # Verify structure
        assert isinstance(mul_expr.left, ir.Add)
        assert cast(ir.Var, mul_expr.left.left).name == "x"
        assert cast(ir.ConstInt, mul_expr.left.right).value == 5
        assert cast(ir.ConstInt, mul_expr.right).value == 2

    def test_complex_nested_expression(self):
        """Test building a complex expression: ((a + b) * (c - d))."""
        span = ir.Span("test.py", 1, 1, 1, 30)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        c = ir.Var("c", ir.ScalarType(dtype), span)
        d = ir.Var("d", ir.ScalarType(dtype), span)

        add_expr = ir.Add(a, b, dtype, span)
        sub_expr = ir.Sub(c, d, dtype, span)
        mul_expr = ir.Mul(add_expr, sub_expr, dtype, span)

        # Verify structure
        assert isinstance(mul_expr.left, ir.Add)
        assert isinstance(mul_expr.right, ir.Sub)
        assert cast(ir.Var, mul_expr.left.left).name == "a"
        assert cast(ir.Var, mul_expr.left.right).name == "b"
        assert cast(ir.Var, mul_expr.right.left).name == "c"
        assert cast(ir.Var, mul_expr.right.right).name == "d"

    def test_deeply_nested_expression(self):
        """Test building a deeply nested expression: (((x + 1) - 2) * 3) / 4."""
        span = ir.Span("test.py", 1, 1, 1, 40)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        c1 = ir.ConstInt(1, dtype, span)
        c2 = ir.ConstInt(2, dtype, span)
        c3 = ir.ConstInt(3, dtype, span)
        c4 = ir.ConstInt(4, dtype, span)

        add_expr = ir.Add(x, c1, dtype, span)
        sub_expr = ir.Sub(add_expr, c2, dtype, span)
        mul_expr = ir.Mul(sub_expr, c3, dtype, span)
        div_expr = ir.FloatDiv(mul_expr, c4, dtype, span)

        # Verify structure depth
        assert isinstance(div_expr.left, ir.Mul)
        assert isinstance(div_expr.left.left, ir.Sub)
        assert isinstance(div_expr.left.left.left, ir.Add)

    def test_all_operations(self):
        """Test expression using multiple operations."""
        span = ir.Span("test.py", 1, 1, 1, 50)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        c = ir.Var("c", ir.ScalarType(dtype), span)
        d = ir.Var("d", ir.ScalarType(dtype), span)
        e = ir.Var("e", ir.ScalarType(dtype), span)
        f = ir.Var("f", ir.ScalarType(dtype), span)

        add_expr = ir.Add(a, b, dtype, span)
        mul_expr = ir.Mul(c, d, dtype, span)
        mod_expr = ir.FloorMod(e, f, dtype, span)
        div_expr = ir.FloatDiv(mul_expr, mod_expr, dtype, span)
        final_expr = ir.Sub(add_expr, div_expr, dtype, span)

        # Verify structure
        assert isinstance(final_expr, ir.Sub)
        assert isinstance(final_expr.left, ir.Add)
        assert isinstance(final_expr.right, ir.FloatDiv)

    def test_unary_with_binary(self):
        """Test mixing unary and binary expressions: -(x + 5)."""
        span = ir.Span("test.py", 1, 1, 1, 20)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        c5 = ir.ConstInt(5, dtype, span)

        add_expr = ir.Add(x, c5, dtype, span)
        neg_expr = ir.Neg(add_expr, dtype, span)

        # Verify structure
        assert isinstance(neg_expr, ir.Neg)
        assert isinstance(neg_expr.operand, ir.Add)
        assert cast(ir.Var, neg_expr.operand.left).name == "x"


class TestImmutability:
    """Tests for immutability of expression nodes."""

    def test_expr_operands_immutable(self):
        """Test that binary expression operands cannot be modified."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        add_expr = ir.Add(x, y, dtype, span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            add_expr.left = ir.Var("z", ir.ScalarType(dtype), span)  # type: ignore


if __name__ == "__main__":
    pytest.main(["-v", __file__])
