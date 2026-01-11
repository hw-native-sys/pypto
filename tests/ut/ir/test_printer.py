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
from pypto import DataType, ir


def test_basic_atoms():
    """Test printing of atomic expressions (variables and constants)."""
    span = ir.Span.unknown()

    # Variables
    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    assert str(a) == "a"

    # Constants
    c = ir.ConstInt(42, DataType.INT64, span)
    assert str(c) == "42"

    c_neg = ir.ConstInt(-5, DataType.INT64, span)
    assert str(c_neg) == "-5"


def test_basic_arithmetic():
    """Test basic arithmetic operations without precedence issues."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.ConstInt(2, dtype, span)
    d = ir.ConstInt(3, dtype, span)

    # Simple addition
    expr = ir.Add(a, b, dtype, span)
    assert str(expr) == "a + b"

    # Simple subtraction
    expr = ir.Sub(a, b, dtype, span)
    assert str(expr) == "a - b"

    # Simple multiplication
    expr = ir.Mul(a, c, dtype, span)
    assert str(expr) == "a * 2"

    # Simple division
    expr = ir.FloatDiv(a, b, dtype, span)
    assert str(expr) == "a / b"

    # Floor division
    expr = ir.FloorDiv(a, c, dtype, span)
    assert str(expr) == "a // 2"

    # Modulo
    expr = ir.FloorMod(a, d, dtype, span)
    assert str(expr) == "a % 3"


def test_precedence_mul_add():
    """Test precedence: multiplication has higher precedence than addition."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.ConstInt(2, dtype, span)
    d = ir.ConstInt(3, dtype, span)

    # a * 2 + b * 3 should not have unnecessary parens
    mul1 = ir.Mul(a, c, dtype, span)
    mul2 = ir.Mul(b, d, dtype, span)
    expr = ir.Add(mul1, mul2, dtype, span)
    assert str(expr) == "a * 2 + b * 3"

    # (a + b) * (c + d) needs parens
    add1 = ir.Add(a, b, dtype, span)
    add2 = ir.Add(c, d, dtype, span)
    expr = ir.Mul(add1, add2, dtype, span)
    assert str(expr) == "(a + b) * (2 + 3)"


def test_associativity_subtraction():
    """Test left-associativity of subtraction."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.Var("c", ir.ScalarType(dtype), span)

    # a - b - c means (a - b) - c (left-associative)
    sub1 = ir.Sub(a, b, dtype, span)
    expr = ir.Sub(sub1, c, dtype, span)
    assert str(expr) == "a - b - c"

    # a - (b - c) needs parens on right
    sub2 = ir.Sub(b, c, dtype, span)
    expr = ir.Sub(a, sub2, dtype, span)
    assert str(expr) == "a - (b - c)"

    # a - (b + c) needs parens (same precedence, different operator)
    add = ir.Add(b, c, dtype, span)
    expr = ir.Sub(a, add, dtype, span)
    assert str(expr) == "a - (b + c)"

    # a + (b - c) needs parens (same precedence on right)
    sub3 = ir.Sub(b, c, dtype, span)
    expr = ir.Add(a, sub3, dtype, span)
    assert str(expr) == "a + (b - c)"

    # a + (b + c) needs parens to show explicit right-association
    add1 = ir.Add(b, c, dtype, span)
    expr = ir.Add(a, add1, dtype, span)
    assert str(expr) == "a + (b + c)"

    # a // (b // c) needs parens (critical for non-associative ops)
    div1 = ir.FloorDiv(b, c, dtype, span)
    expr = ir.FloorDiv(a, div1, dtype, span)
    assert str(expr) == "a // (b // c)"


def test_power_right_associative():
    """Test right-associativity of power operator."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    c2 = ir.ConstInt(2, dtype, span)
    c3 = ir.ConstInt(3, dtype, span)
    c4 = ir.ConstInt(4, dtype, span)

    # 2 ** 3 ** 4 means 2 ** (3 ** 4) (right-associative)
    pow1 = ir.Pow(c3, c4, dtype, span)
    expr = ir.Pow(c2, pow1, dtype, span)
    assert str(expr) == "2 ** 3 ** 4"

    # (2 ** 3) ** 4 needs parens on left
    pow2 = ir.Pow(c2, c3, dtype, span)
    expr = ir.Pow(pow2, c4, dtype, span)
    assert str(expr) == "(2 ** 3) ** 4"


def test_comparison_operators():
    """Test comparison operators."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.Var("c", ir.ScalarType(dtype), span)

    # Basic comparisons
    assert str(ir.Eq(a, b, dtype, span)) == "a == b"
    assert str(ir.Ne(a, b, dtype, span)) == "a != b"
    assert str(ir.Lt(a, b, dtype, span)) == "a < b"
    assert str(ir.Le(a, b, dtype, span)) == "a <= b"
    assert str(ir.Gt(a, b, dtype, span)) == "a > b"
    assert str(ir.Ge(a, b, dtype, span)) == "a >= b"

    # Comparison has lower precedence than arithmetic
    add = ir.Add(a, b, dtype, span)
    expr = ir.Lt(add, c, dtype, span)
    assert str(expr) == "a + b < c"


def test_logical_operators():
    """Test logical operators with Python keywords."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.Var("c", ir.ScalarType(dtype), span)

    # Basic logical ops
    assert str(ir.And(a, b, dtype, span)) == "a and b"
    assert str(ir.Or(a, b, dtype, span)) == "a or b"
    assert str(ir.Xor(a, b, dtype, span)) == "a xor b"

    # a and b or c - 'or' has lower precedence
    and_expr = ir.And(a, b, dtype, span)
    expr = ir.Or(and_expr, c, dtype, span)
    assert str(expr) == "a and b or c"

    # a or (b and c) - needs parens
    and_expr = ir.And(b, c, dtype, span)
    expr = ir.Or(a, and_expr, dtype, span)
    assert str(expr) == "a or b and c"  # No parens needed, 'and' binds tighter


def test_bitwise_operators():
    """Test bitwise operators."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.Var("c", ir.ScalarType(dtype), span)

    # Basic bitwise ops
    assert str(ir.BitAnd(a, b, dtype, span)) == "a & b"
    assert str(ir.BitOr(a, b, dtype, span)) == "a | b"
    assert str(ir.BitXor(a, b, dtype, span)) == "a ^ b"
    assert str(ir.BitShiftLeft(a, b, dtype, span)) == "a << b"
    assert str(ir.BitShiftRight(a, b, dtype, span)) == "a >> b"

    # a & b | c - '|' has lower precedence than '&'
    and_expr = ir.BitAnd(a, b, dtype, span)
    expr = ir.BitOr(and_expr, c, dtype, span)
    assert str(expr) == "a & b | c"


def test_unary_operators():
    """Test unary operators."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.ConstInt(5, dtype, span)

    # Negation
    expr = ir.Neg(a, dtype, span)
    assert str(expr) == "-a"

    # Bitwise not
    expr = ir.BitNot(a, dtype, span)
    assert str(expr) == "~a"

    # Logical not
    expr = ir.Not(a, dtype, span)
    assert str(expr) == "not a"

    # Absolute value (function-style)
    expr = ir.Abs(c, dtype, span)
    assert str(expr) == "abs(5)"

    # Negation with addition needs parens
    add = ir.Add(a, b, dtype, span)
    expr = ir.Neg(add, dtype, span)
    assert str(expr) == "-(a + b)"

    # -a * b doesn't need parens (unary has higher precedence)
    neg = ir.Neg(a, dtype, span)
    expr = ir.Mul(neg, b, dtype, span)
    assert str(expr) == "-a * b"


def test_function_style_binary_ops():
    """Test Min/Max which use function call syntax."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.Var("c", ir.ScalarType(dtype), span)

    # Min and Max
    expr = ir.Min(a, b, dtype, span)
    assert str(expr) == "min(a, b)"

    expr = ir.Max(a, b, dtype, span)
    assert str(expr) == "max(a, b)"

    # Nested: min(a, max(b, c))
    max_expr = ir.Max(b, c, dtype, span)
    expr = ir.Min(a, max_expr, dtype, span)
    assert str(expr) == "min(a, max(b, c))"


def test_call_expressions():
    """Test function call expressions."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.Var("c", ir.ScalarType(dtype), span)

    # Simple call
    op = ir.Op("foo")
    expr = ir.Call(op, [a, b], span)
    assert str(expr) == "foo(a, b)"

    # Call with no args
    expr = ir.Call(op, [], span)
    assert str(expr) == "foo()"

    # Call with complex arguments
    add = ir.Add(a, b, dtype, span)
    expr = ir.Call(op, [add, c], span)
    assert str(expr) == "foo(a + b, c)"


def test_complex_nested_expressions():
    """Test complex nested expressions."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)
    c = ir.Var("c", ir.ScalarType(dtype), span)
    d = ir.Var("d", ir.ScalarType(dtype), span)
    c2 = ir.ConstInt(2, dtype, span)
    c3 = ir.ConstInt(3, dtype, span)

    # a * 2 + b * 3 - c
    mul1 = ir.Mul(a, c2, dtype, span)
    mul2 = ir.Mul(b, c3, dtype, span)
    add = ir.Add(mul1, mul2, dtype, span)
    expr = ir.Sub(add, c, dtype, span)
    assert str(expr) == "a * 2 + b * 3 - c"

    # (a + b) * (c - d)
    add = ir.Add(a, b, dtype, span)
    sub = ir.Sub(c, d, dtype, span)
    expr = ir.Mul(add, sub, dtype, span)
    assert str(expr) == "(a + b) * (c - d)"

    # a < b and b < c
    lt1 = ir.Lt(a, b, dtype, span)
    lt2 = ir.Lt(b, c, dtype, span)
    expr = ir.And(lt1, lt2, dtype, span)
    assert str(expr) == "a < b and b < c"


def test_all_division_types():
    """Test all division operator types."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)

    # Float division
    expr = ir.FloatDiv(a, b, dtype, span)
    assert str(expr) == "a / b"

    # Floor division
    expr = ir.FloorDiv(a, b, dtype, span)
    assert str(expr) == "a // b"

    # Modulo
    expr = ir.FloorMod(a, b, dtype, span)
    assert str(expr) == "a % b"


def test_abs_neg_interaction():
    """Test interaction between abs() and negation."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    c = ir.ConstInt(5, dtype, span)
    a = ir.Var("a", ir.ScalarType(dtype), span)

    # abs(-5)
    neg = ir.Neg(c, dtype, span)
    expr = ir.Abs(neg, dtype, span)
    assert str(expr) == "abs(-5)"

    # -abs(a)
    abs_expr = ir.Abs(a, dtype, span)
    expr = ir.Neg(abs_expr, dtype, span)
    assert str(expr) == "-abs(a)"


def test_repr_method():
    """Test __repr__ includes type information."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    a = ir.Var("a", ir.ScalarType(dtype), span)
    b = ir.Var("b", ir.ScalarType(dtype), span)

    expr = ir.Add(a, b, dtype, span)
    repr_str = repr(expr)
    assert repr_str == "<ir.Add: a + b>"


# ========== Statement Printing Tests ==========


def test_base_stmt_printing():
    """Test printing of base Stmt."""
    span = ir.Span.unknown()
    stmt = ir.Stmt(span)
    assert str(stmt) == "Stmt"


def test_assign_stmt_simple():
    """Test simple assignment statement."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)

    assign = ir.AssignStmt(x, y, span)
    assert str(assign) == "x = y"


def test_assign_stmt_with_constant():
    """Test assignment with constant value."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    c5 = ir.ConstInt(5, dtype, span)

    assign = ir.AssignStmt(x, c5, span)
    assert str(assign) == "x = 5"


def test_assign_stmt_with_arithmetic():
    """Test assignment with arithmetic expression."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    # x = y + z
    add = ir.Add(y, z, dtype, span)
    assign = ir.AssignStmt(x, add, span)
    assert str(assign) == "x = y + z"

    # x = y * z
    mul = ir.Mul(y, z, dtype, span)
    assign = ir.AssignStmt(x, mul, span)
    assert str(assign) == "x = y * z"


def test_assign_stmt_with_complex_expression():
    """Test assignment with complex nested expressions."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)
    c2 = ir.ConstInt(2, dtype, span)
    c3 = ir.ConstInt(3, dtype, span)

    # x = y * 2 + z * 3
    mul1 = ir.Mul(y, c2, dtype, span)
    mul2 = ir.Mul(z, c3, dtype, span)
    add = ir.Add(mul1, mul2, dtype, span)
    assign = ir.AssignStmt(x, add, span)
    assert str(assign) == "x = y * 2 + z * 3"

    # x = (y + z) * 2
    add = ir.Add(y, z, dtype, span)
    mul = ir.Mul(add, c2, dtype, span)
    assign = ir.AssignStmt(x, mul, span)
    assert str(assign) == "x = (y + z) * 2"


def test_assign_stmt_with_function_call():
    """Test assignment with function call."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    # x = foo(y, z)
    op = ir.Op("foo")
    call = ir.Call(op, [y, z], span)
    assign = ir.AssignStmt(x, call, span)
    assert str(assign) == "x = foo(y, z)"


def test_assign_stmt_with_unary_operators():
    """Test assignment with unary operators."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)

    # x = -y
    neg = ir.Neg(y, dtype, span)
    assign = ir.AssignStmt(x, neg, span)
    assert str(assign) == "x = -y"

    # x = abs(y)
    abs_expr = ir.Abs(y, dtype, span)
    assign = ir.AssignStmt(x, abs_expr, span)
    assert str(assign) == "x = abs(y)"

    # x = not y
    not_expr = ir.Not(y, dtype, span)
    assign = ir.AssignStmt(x, not_expr, span)
    assert str(assign) == "x = not y"


def test_assign_stmt_with_comparison():
    """Test assignment with comparison expression."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    # x = y < z
    lt = ir.Lt(y, z, dtype, span)
    assign = ir.AssignStmt(x, lt, span)
    assert str(assign) == "x = y < z"

    # x = y == z
    eq = ir.Eq(y, z, dtype, span)
    assign = ir.AssignStmt(x, eq, span)
    assert str(assign) == "x = y == z"


def test_assign_stmt_with_logical_operators():
    """Test assignment with logical operators."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    # x = y and z
    and_expr = ir.And(y, z, dtype, span)
    assign = ir.AssignStmt(x, and_expr, span)
    assert str(assign) == "x = y and z"

    # x = y or z
    or_expr = ir.Or(y, z, dtype, span)
    assign = ir.AssignStmt(x, or_expr, span)
    assert str(assign) == "x = y or z"


def test_assign_stmt_with_nested_expressions():
    """Test assignment with deeply nested expressions."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)
    w = ir.Var("w", ir.ScalarType(dtype), span)
    c2 = ir.ConstInt(2, dtype, span)

    # x = (y + z) * (w - 2)
    add = ir.Add(y, z, dtype, span)
    sub = ir.Sub(w, c2, dtype, span)
    mul = ir.Mul(add, sub, dtype, span)
    assign = ir.AssignStmt(x, mul, span)
    assert str(assign) == "x = (y + z) * (w - 2)"


def test_assign_stmt_with_power_operator():
    """Test assignment with power operator."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    c2 = ir.ConstInt(2, dtype, span)
    c3 = ir.ConstInt(3, dtype, span)

    # x = y ** 2
    pow_expr = ir.Pow(y, c2, dtype, span)
    assign = ir.AssignStmt(x, pow_expr, span)
    assert str(assign) == "x = y ** 2"

    # x = 2 ** 3 ** y (right-associative)
    pow1 = ir.Pow(c3, y, dtype, span)
    pow2 = ir.Pow(c2, pow1, dtype, span)
    assign = ir.AssignStmt(x, pow2, span)
    assert str(assign) == "x = 2 ** 3 ** y"


def test_assign_stmt_with_min_max():
    """Test assignment with min/max functions."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    # x = min(y, z)
    min_expr = ir.Min(y, z, dtype, span)
    assign = ir.AssignStmt(x, min_expr, span)
    assert str(assign) == "x = min(y, z)"

    # x = max(y, z)
    max_expr = ir.Max(y, z, dtype, span)
    assign = ir.AssignStmt(x, max_expr, span)
    assert str(assign) == "x = max(y, z)"


def test_assign_stmt_with_bitwise_operators():
    """Test assignment with bitwise operators."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    # x = y & z
    bit_and = ir.BitAnd(y, z, dtype, span)
    assign = ir.AssignStmt(x, bit_and, span)
    assert str(assign) == "x = y & z"

    # x = y | z
    bit_or = ir.BitOr(y, z, dtype, span)
    assign = ir.AssignStmt(x, bit_or, span)
    assert str(assign) == "x = y | z"

    # x = y << z
    shift_left = ir.BitShiftLeft(y, z, dtype, span)
    assign = ir.AssignStmt(x, shift_left, span)
    assert str(assign) == "x = y << z"


def test_assign_stmt_repr():
    """Test __repr__ for AssignStmt."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)

    assign = ir.AssignStmt(x, y, span)
    repr_str = repr(assign)
    assert repr_str == "<ir.AssignStmt: x = y>"


def test_multiple_assign_statements():
    """Test printing multiple assignment statements."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)
    c1 = ir.ConstInt(1, dtype, span)
    c2 = ir.ConstInt(2, dtype, span)

    # x = 1
    assign1 = ir.AssignStmt(x, c1, span)
    assert str(assign1) == "x = 1"

    # y = 2
    assign2 = ir.AssignStmt(y, c2, span)
    assert str(assign2) == "y = 2"

    # z = x + y
    add = ir.Add(x, y, dtype, span)
    assign3 = ir.AssignStmt(z, add, span)
    assert str(assign3) == "z = x + y"


def test_assign_stmt_with_division_types():
    """Test assignment with different division types."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    c2 = ir.ConstInt(2, dtype, span)

    # x = y / 2
    float_div = ir.FloatDiv(y, c2, dtype, span)
    assign = ir.AssignStmt(x, float_div, span)
    assert str(assign) == "x = y / 2"

    # x = y // 2
    floor_div = ir.FloorDiv(y, c2, dtype, span)
    assign = ir.AssignStmt(x, floor_div, span)
    assert str(assign) == "x = y // 2"

    # x = y % 2
    mod = ir.FloorMod(y, c2, dtype, span)
    assign = ir.AssignStmt(x, mod, span)
    assert str(assign) == "x = y % 2"


def test_if_stmt_printing():
    """Test printing of IfStmt statements."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    # Basic if statement without else
    condition = ir.Eq(x, y, dtype, span)
    assign = ir.AssignStmt(x, y, span)
    if_stmt = ir.IfStmt(condition, [assign], [], span)
    assert str(if_stmt) == "if x == y:\n  x = y"

    # If statement with else
    assign1 = ir.AssignStmt(x, y, span)
    assign2 = ir.AssignStmt(y, z, span)
    if_stmt2 = ir.IfStmt(condition, [assign1], [assign2], span)
    assert str(if_stmt2) == "if x == y:\n  x = y\nelse:\n  y = z"

    # If statement with multiple statements in then_body
    assign3 = ir.AssignStmt(z, x, span)
    if_stmt3 = ir.IfStmt(condition, [assign1, assign2], [], span)
    assert str(if_stmt3) == "if x == y:\n  x = y\n  y = z"

    # If statement with multiple statements in both branches
    if_stmt4 = ir.IfStmt(condition, [assign1, assign2], [assign3], span)
    assert str(if_stmt4) == "if x == y:\n  x = y\n  y = z\nelse:\n  z = x"

    # If statement with complex condition
    complex_condition = ir.And(ir.Lt(x, y, dtype, span), ir.Gt(z, x, dtype, span), dtype, span)
    if_stmt5 = ir.IfStmt(complex_condition, [assign1], [], span)
    assert str(if_stmt5) == "if x < y and z > x:\n  x = y"


def test_yield_stmt_printing():
    """Test printing of YieldStmt statements."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    z = ir.Var("z", ir.ScalarType(dtype), span)

    # Yield without value
    yield_stmt = ir.YieldStmt(span)
    assert str(yield_stmt) == "yield"

    # Yield with single variable
    yield_stmt2 = ir.YieldStmt([x], span)
    assert str(yield_stmt2) == "yield x"

    # Yield with multiple variables
    yield_stmt3 = ir.YieldStmt([x, y], span)
    assert str(yield_stmt3) == "yield x, y"

    # Yield with three variables
    yield_stmt4 = ir.YieldStmt([x, y, z], span)
    assert str(yield_stmt4) == "yield x, y, z"


def test_for_stmt_printing():
    """Test printing of ForStmt statements."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    i = ir.Var("i", ir.ScalarType(dtype), span)
    start = ir.ConstInt(0, dtype, span)
    stop = ir.ConstInt(10, dtype, span)
    step = ir.ConstInt(1, dtype, span)

    # For loop with empty body
    for_stmt = ir.ForStmt(i, start, stop, step, [], span)
    assert str(for_stmt) == "for i in range(0, 10, 1):\n"

    # For loop with single statement
    assign = ir.AssignStmt(i, start, span)
    for_stmt2 = ir.ForStmt(i, start, stop, step, [assign], span)
    assert str(for_stmt2) == "for i in range(0, 10, 1):\n  i = 0"

    # For loop with multiple statements
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    assign1 = ir.AssignStmt(i, x, span)
    assign2 = ir.AssignStmt(x, y, span)
    for_stmt3 = ir.ForStmt(i, start, stop, step, [assign1, assign2], span)
    assert str(for_stmt3) == "for i in range(0, 10, 1):\n  i = x\n  x = y"

    # For loop with variable expressions
    n = ir.Var("n", ir.ScalarType(dtype), span)
    m = ir.Var("m", ir.ScalarType(dtype), span)
    k = ir.Var("k", ir.ScalarType(dtype), span)
    for_stmt4 = ir.ForStmt(i, n, m, k, [assign], span)
    assert str(for_stmt4) == "for i in range(n, m, k):\n  i = 0"

    # For loop with arithmetic expressions
    start_expr = ir.Add(n, ir.ConstInt(1, dtype, span), dtype, span)
    stop_expr = ir.Mul(m, ir.ConstInt(2, dtype, span), dtype, span)
    step_expr = ir.Sub(k, ir.ConstInt(1, dtype, span), dtype, span)
    for_stmt5 = ir.ForStmt(i, start_expr, stop_expr, step_expr, [assign], span)
    assert str(for_stmt5) == "for i in range(n + 1, m * 2, k - 1):\n  i = 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
