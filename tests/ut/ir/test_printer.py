# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for IR expression printer with precedence handling."""

import pytest
from pypto import ir


def test_basic_atoms():
    """Test printing of atomic expressions (variables and constants)."""
    span = ir.Span.unknown()

    # Variables
    a = ir.Var("a", span)
    assert str(a) == "a"

    # Constants
    c = ir.ConstInt(42, span)
    assert str(c) == "42"

    c_neg = ir.ConstInt(-5, span)
    assert str(c_neg) == "-5"


def test_basic_arithmetic():
    """Test basic arithmetic operations without precedence issues."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.ConstInt(2, span)
    d = ir.ConstInt(3, span)

    # Simple addition
    expr = ir.Add(a, b, span)
    assert str(expr) == "a + b"

    # Simple subtraction
    expr = ir.Sub(a, b, span)
    assert str(expr) == "a - b"

    # Simple multiplication
    expr = ir.Mul(a, c, span)
    assert str(expr) == "a * 2"

    # Simple division
    expr = ir.FloatDiv(a, b, span)
    assert str(expr) == "a / b"

    # Floor division
    expr = ir.FloorDiv(a, c, span)
    assert str(expr) == "a // 2"

    # Modulo
    expr = ir.FloorMod(a, d, span)
    assert str(expr) == "a % 3"


def test_precedence_mul_add():
    """Test precedence: multiplication has higher precedence than addition."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.ConstInt(2, span)
    d = ir.ConstInt(3, span)

    # a * 2 + b * 3 should not have unnecessary parens
    mul1 = ir.Mul(a, c, span)
    mul2 = ir.Mul(b, d, span)
    expr = ir.Add(mul1, mul2, span)
    assert str(expr) == "a * 2 + b * 3"

    # (a + b) * (c + d) needs parens
    add1 = ir.Add(a, b, span)
    add2 = ir.Add(c, d, span)
    expr = ir.Mul(add1, add2, span)
    assert str(expr) == "(a + b) * (2 + 3)"


def test_associativity_subtraction():
    """Test left-associativity of subtraction."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.Var("c", span)

    # a - b - c means (a - b) - c (left-associative)
    sub1 = ir.Sub(a, b, span)
    expr = ir.Sub(sub1, c, span)
    assert str(expr) == "a - b - c"

    # a - (b - c) needs parens on right
    sub2 = ir.Sub(b, c, span)
    expr = ir.Sub(a, sub2, span)
    assert str(expr) == "a - (b - c)"

    # a - (b + c) needs parens (same precedence, different operator)
    add = ir.Add(b, c, span)
    expr = ir.Sub(a, add, span)
    assert str(expr) == "a - (b + c)"

    # a + (b - c) needs parens (same precedence on right)
    sub3 = ir.Sub(b, c, span)
    expr = ir.Add(a, sub3, span)
    assert str(expr) == "a + (b - c)"

    # a + (b + c) needs parens to show explicit right-association
    add1 = ir.Add(b, c, span)
    expr = ir.Add(a, add1, span)
    assert str(expr) == "a + (b + c)"

    # a // (b // c) needs parens (critical for non-associative ops)
    div1 = ir.FloorDiv(b, c, span)
    expr = ir.FloorDiv(a, div1, span)
    assert str(expr) == "a // (b // c)"


def test_power_right_associative():
    """Test right-associativity of power operator."""
    span = ir.Span.unknown()
    c2 = ir.ConstInt(2, span)
    c3 = ir.ConstInt(3, span)
    c4 = ir.ConstInt(4, span)

    # 2 ** 3 ** 4 means 2 ** (3 ** 4) (right-associative)
    pow1 = ir.Pow(c3, c4, span)
    expr = ir.Pow(c2, pow1, span)
    assert str(expr) == "2 ** 3 ** 4"

    # (2 ** 3) ** 4 needs parens on left
    pow2 = ir.Pow(c2, c3, span)
    expr = ir.Pow(pow2, c4, span)
    assert str(expr) == "(2 ** 3) ** 4"


def test_comparison_operators():
    """Test comparison operators."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.Var("c", span)

    # Basic comparisons
    assert str(ir.Eq(a, b, span)) == "a == b"
    assert str(ir.Ne(a, b, span)) == "a != b"
    assert str(ir.Lt(a, b, span)) == "a < b"
    assert str(ir.Le(a, b, span)) == "a <= b"
    assert str(ir.Gt(a, b, span)) == "a > b"
    assert str(ir.Ge(a, b, span)) == "a >= b"

    # Comparison has lower precedence than arithmetic
    add = ir.Add(a, b, span)
    expr = ir.Lt(add, c, span)
    assert str(expr) == "a + b < c"


def test_logical_operators():
    """Test logical operators with Python keywords."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.Var("c", span)

    # Basic logical ops
    assert str(ir.And(a, b, span)) == "a and b"
    assert str(ir.Or(a, b, span)) == "a or b"
    assert str(ir.Xor(a, b, span)) == "a xor b"

    # a and b or c - 'or' has lower precedence
    and_expr = ir.And(a, b, span)
    expr = ir.Or(and_expr, c, span)
    assert str(expr) == "a and b or c"

    # a or (b and c) - needs parens
    and_expr = ir.And(b, c, span)
    expr = ir.Or(a, and_expr, span)
    assert str(expr) == "a or b and c"  # No parens needed, 'and' binds tighter


def test_bitwise_operators():
    """Test bitwise operators."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.Var("c", span)

    # Basic bitwise ops
    assert str(ir.BitAnd(a, b, span)) == "a & b"
    assert str(ir.BitOr(a, b, span)) == "a | b"
    assert str(ir.BitXor(a, b, span)) == "a ^ b"
    assert str(ir.BitShiftLeft(a, b, span)) == "a << b"
    assert str(ir.BitShiftRight(a, b, span)) == "a >> b"

    # a & b | c - '|' has lower precedence than '&'
    and_expr = ir.BitAnd(a, b, span)
    expr = ir.BitOr(and_expr, c, span)
    assert str(expr) == "a & b | c"


def test_unary_operators():
    """Test unary operators."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.ConstInt(5, span)

    # Negation
    expr = ir.Neg(a, span)
    assert str(expr) == "-a"

    # Bitwise not
    expr = ir.BitNot(a, span)
    assert str(expr) == "~a"

    # Logical not
    expr = ir.Not(a, span)
    assert str(expr) == "not a"

    # Absolute value (function-style)
    expr = ir.Abs(c, span)
    assert str(expr) == "abs(5)"

    # Negation with addition needs parens
    add = ir.Add(a, b, span)
    expr = ir.Neg(add, span)
    assert str(expr) == "-(a + b)"

    # -a * b doesn't need parens (unary has higher precedence)
    neg = ir.Neg(a, span)
    expr = ir.Mul(neg, b, span)
    assert str(expr) == "-a * b"


def test_function_style_binary_ops():
    """Test Min/Max which use function call syntax."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.Var("c", span)

    # Min and Max
    expr = ir.Min(a, b, span)
    assert str(expr) == "min(a, b)"

    expr = ir.Max(a, b, span)
    assert str(expr) == "max(a, b)"

    # Nested: min(a, max(b, c))
    max_expr = ir.Max(b, c, span)
    expr = ir.Min(a, max_expr, span)
    assert str(expr) == "min(a, max(b, c))"


def test_call_expressions():
    """Test function call expressions."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.Var("c", span)

    # Simple call
    op = ir.Op("foo")
    expr = ir.Call(op, [a, b], span)
    assert str(expr) == "foo(a, b)"

    # Call with no args
    expr = ir.Call(op, [], span)
    assert str(expr) == "foo()"

    # Call with complex arguments
    add = ir.Add(a, b, span)
    expr = ir.Call(op, [add, c], span)
    assert str(expr) == "foo(a + b, c)"


def test_complex_nested_expressions():
    """Test complex nested expressions."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)
    c = ir.Var("c", span)
    d = ir.Var("d", span)
    c2 = ir.ConstInt(2, span)
    c3 = ir.ConstInt(3, span)

    # a * 2 + b * 3 - c
    mul1 = ir.Mul(a, c2, span)
    mul2 = ir.Mul(b, c3, span)
    add = ir.Add(mul1, mul2, span)
    expr = ir.Sub(add, c, span)
    assert str(expr) == "a * 2 + b * 3 - c"

    # (a + b) * (c - d)
    add = ir.Add(a, b, span)
    sub = ir.Sub(c, d, span)
    expr = ir.Mul(add, sub, span)
    assert str(expr) == "(a + b) * (c - d)"

    # a < b and b < c
    lt1 = ir.Lt(a, b, span)
    lt2 = ir.Lt(b, c, span)
    expr = ir.And(lt1, lt2, span)
    assert str(expr) == "a < b and b < c"


def test_all_division_types():
    """Test all division operator types."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)

    # Float division
    expr = ir.FloatDiv(a, b, span)
    assert str(expr) == "a / b"

    # Floor division
    expr = ir.FloorDiv(a, b, span)
    assert str(expr) == "a // b"

    # Modulo
    expr = ir.FloorMod(a, b, span)
    assert str(expr) == "a % b"


def test_abs_neg_interaction():
    """Test interaction between abs() and negation."""
    span = ir.Span.unknown()
    c = ir.ConstInt(5, span)
    a = ir.Var("a", span)

    # abs(-5)
    neg = ir.Neg(c, span)
    expr = ir.Abs(neg, span)
    assert str(expr) == "abs(-5)"

    # -abs(a)
    abs_expr = ir.Abs(a, span)
    expr = ir.Neg(abs_expr, span)
    assert str(expr) == "-abs(a)"


def test_repr_method():
    """Test __repr__ includes type information."""
    span = ir.Span.unknown()
    a = ir.Var("a", span)
    b = ir.Var("b", span)

    expr = ir.Add(a, b, span)
    repr_str = repr(expr)
    assert repr_str == "<ir.Add: a + b>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
