# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IR hash and equality functionality."""

import pytest
from pypto import DataType, ir


class TestReferenceEquality:
    """Tests for reference equality (pointer-based)."""

    def test_different_pointers_not_equal(self):
        """Test that different pointers with same structure are not reference-equal."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        # Different objects, even with same content
        assert x1 != x2

    def test_different_content_not_equal(self):
        """Test that different variables are not equal."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        assert x != y

    def test_inequality_operator(self):
        """Test inequality operator works correctly."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        assert x1 != x2
        assert not (x1 == x2)


class TestStructuralHash:
    """Tests for structural hash function."""

    def test_same_structure_same_hash(self):
        """Test that expressions with same structure hash to same value."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        # Same variable name, different spans - should have same hash
        hash1 = ir.structural_hash(x1)
        hash2 = ir.structural_hash(x2)
        assert hash1 != hash2

    def test_different_var_names_different_hash(self):
        """Test that variables with different names hash differently."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        hash_x = ir.structural_hash(x)
        hash_y = ir.structural_hash(y)

        # Different names should (almost certainly) have different hashes
        assert hash_x != hash_y

    def test_different_const_values_different_hash(self):
        """Test that constants with different values hash differently."""
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())

        hash1 = ir.structural_hash(c1)
        hash2 = ir.structural_hash(c2)

        assert hash1 != hash2

    def test_same_const_value_same_hash(self):
        """Test that constants with same value hash to same value."""
        c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

        hash1 = ir.structural_hash(c1)
        hash2 = ir.structural_hash(c2)

        assert hash1 == hash2

    def test_different_operation_types_different_hash(self):
        """Test that different operation types hash differently."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
        sub_expr = ir.Sub(x, y, DataType.INT64, ir.Span.unknown())
        mul_expr = ir.Mul(x, y, DataType.INT64, ir.Span.unknown())

        hash_add = ir.structural_hash(add_expr)
        hash_sub = ir.structural_hash(sub_expr)
        hash_mul = ir.structural_hash(mul_expr)

        # Different operations should hash differently
        assert hash_add != hash_sub
        assert hash_add != hash_mul
        assert hash_sub != hash_mul

    def test_nested_expression_hash(self):
        """Test hashing of nested expressions."""
        # Build (x + 5) * 2 with different spans
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c5_1 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        c2_1 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        expr1 = ir.Mul(
            ir.Add(x1, c5_1, DataType.INT64, ir.Span.unknown()), c2_1, DataType.INT64, ir.Span.unknown()
        )

        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c5_2 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        c2_2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        expr2 = ir.Mul(
            ir.Add(x2, c5_2, DataType.INT64, ir.Span.unknown()), c2_2, DataType.INT64, ir.Span.unknown()
        )

        # Same structure, different spans - should hash to same value
        hash1 = ir.structural_hash(expr1)
        hash2 = ir.structural_hash(expr2)
        assert hash1 != hash2

    def test_operand_order_matters(self):
        """Test that operand order affects hash (x + y != y + x in structure)."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        add1 = ir.Add(x, y, DataType.INT64, ir.Span.unknown())  # x + y
        add2 = ir.Add(y, x, DataType.INT64, ir.Span.unknown())  # y + x

        hash1 = ir.structural_hash(add1)
        hash2 = ir.structural_hash(add2)

        # Different operand order should (almost certainly) hash differently
        assert hash1 != hash2

    def test_unary_expression_hash(self):
        """Test hashing of unary expressions."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        neg1 = ir.Neg(x1, DataType.INT64, ir.Span.unknown())

        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        neg2 = ir.Neg(x2, DataType.INT64, ir.Span.unknown())

        hash1 = ir.structural_hash(neg1)
        hash2 = ir.structural_hash(neg2)

        assert hash1 != hash2

    def test_call_expression_hash(self):
        """Test hashing of call expressions."""
        op1 = ir.Op("func")
        op2 = ir.Op("func")

        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        call1 = ir.Call(op1, [x, y], ir.Span.unknown())
        call2 = ir.Call(op2, [x, y], ir.Span.unknown())

        hash1 = ir.structural_hash(call1)
        hash2 = ir.structural_hash(call2)

        # Same op name and args - should hash to same value
        assert hash1 == hash2

    def test_different_op_names_different_hash(self):
        """Test that calls with different op names hash differently."""
        op1 = ir.Op("func1")
        op2 = ir.Op("func2")

        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        call1 = ir.Call(op1, [x], ir.Span.unknown())
        call2 = ir.Call(op2, [x], ir.Span.unknown())

        hash1 = ir.structural_hash(call1)
        hash2 = ir.structural_hash(call2)

        # Different op names should hash differently
        assert hash1 != hash2

    def test_stmt_same_structure_same_hash(self):
        """Test that Stmt nodes with same structure hash to same value."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        stmt1 = ir.Stmt(span1)
        stmt2 = ir.Stmt(span2)

        # Same structure (both are base Stmt with unknown span) - should hash to same value
        hash1 = ir.structural_hash(stmt1)
        hash2 = ir.structural_hash(stmt2)

        assert hash1 == hash2

    def test_stmt_different_spans_same_hash(self):
        """Test that Stmt nodes with different spans but same structure hash to same value."""
        span1 = ir.Span("file1.py", 1, 1, 1, 10)
        span2 = ir.Span("file2.py", 2, 2, 2, 20)

        stmt1 = ir.Stmt(span1)
        stmt2 = ir.Stmt(span2)

        # Different spans, but structural hash ignores span - should hash to same value
        hash1 = ir.structural_hash(stmt1)
        hash2 = ir.structural_hash(stmt2)

        assert hash1 == hash2

    def test_stmt_different_from_expr_hash(self):
        """Test that Stmt and Expr nodes hash differently."""
        span = ir.Span.unknown()

        stmt = ir.Stmt(span)
        expr = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        hash_stmt = ir.structural_hash(stmt)
        hash_expr = ir.structural_hash(expr)

        # Different IR node types should hash differently
        assert hash_stmt != hash_expr


class TestStructuralEquality:
    """Tests for structural equality function."""

    def test_same_var_structural_equal(self):
        """Test that variables with same name are structurally equal with auto mapping."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        # Different objects, need auto mapping to be equal
        assert ir.structural_equal(x1, x2, enable_auto_mapping=True)

    def test_different_var_not_structural_equal(self):
        """Test that variables with different names are not structurally equal."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        assert not ir.structural_equal(x, y)

    def test_same_const_structural_equal(self):
        """Test that constants with same value are structurally equal."""
        c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

        assert ir.structural_equal(c1, c2)

    def test_different_const_not_structural_equal(self):
        """Test that constants with different values are not structurally equal."""
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())

        assert not ir.structural_equal(c1, c2)

    def test_different_types_not_equal(self):
        """Test that different expression types are not structurally equal."""
        var = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        const = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        assert not ir.structural_equal(var, const)

    def test_binary_expr_structural_equal(self):
        """Test structural equality of binary expressions."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        add1 = ir.Add(x1, y1, DataType.INT64, ir.Span.unknown())

        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y2 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        add2 = ir.Add(x2, y2, DataType.INT64, ir.Span.unknown())

        # Same structure with auto mapping
        assert ir.structural_equal(add1, add2, enable_auto_mapping=True)

    def test_different_binary_ops_not_equal(self):
        """Test that different binary operations are not structurally equal."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
        sub_expr = ir.Sub(x, y, DataType.INT64, ir.Span.unknown())

        assert not ir.structural_equal(add_expr, sub_expr)

    def test_operand_order_matters_in_equality(self):
        """Test that operand order matters for structural equality."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        add1 = ir.Add(x, y, DataType.INT64, ir.Span.unknown())  # x + y
        add2 = ir.Add(y, x, DataType.INT64, ir.Span.unknown())  # y + x

        # Different operand order
        assert not ir.structural_equal(add1, add2)

    def test_nested_expressions_structural_equal(self):
        """Test structural equality of nested expressions."""
        # Build (x + 5) * 2
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c5_1 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        c2_1 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        expr1 = ir.Mul(
            ir.Add(x1, c5_1, DataType.INT64, ir.Span.unknown()), c2_1, DataType.INT64, ir.Span.unknown()
        )

        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c5_2 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        c2_2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        expr2 = ir.Mul(
            ir.Add(x2, c5_2, DataType.INT64, ir.Span.unknown()), c2_2, DataType.INT64, ir.Span.unknown()
        )

        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_different_nested_structure_not_equal(self):
        """Test that different nested structures are not equal."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c5 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())

        expr1 = ir.Mul(
            ir.Add(x, c5, DataType.INT64, ir.Span.unknown()), c2, DataType.INT64, ir.Span.unknown()
        )  # (x + 5) * 2
        expr2 = ir.Add(
            ir.Mul(x, c5, DataType.INT64, ir.Span.unknown()), c2, DataType.INT64, ir.Span.unknown()
        )  # (x * 5) + 2

        assert not ir.structural_equal(expr1, expr2)

    def test_unary_expr_structural_equal(self):
        """Test structural equality of unary expressions."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        neg1 = ir.Neg(x1, DataType.INT64, ir.Span.unknown())

        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        neg2 = ir.Neg(x2, DataType.INT64, ir.Span.unknown())

        assert ir.structural_equal(neg1, neg2, enable_auto_mapping=True)

    def test_different_unary_ops_not_equal(self):
        """Test that different unary operations are not equal."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        neg_expr = ir.Neg(x, DataType.INT64, ir.Span.unknown())
        abs_expr = ir.Abs(x, DataType.INT64, ir.Span.unknown())

        assert not ir.structural_equal(neg_expr, abs_expr)

    def test_call_expr_structural_equal(self):
        """Test structural equality of call expressions."""
        op1 = ir.Op("func")
        op2 = ir.Op("func")

        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        call1 = ir.Call(op1, [x, y], ir.Span.unknown())
        call2 = ir.Call(op2, [x, y], ir.Span.unknown())

        # Same op name and args
        assert ir.structural_equal(call1, call2)

    def test_different_op_names_not_equal(self):
        """Test that calls with different op names are not equal."""
        op1 = ir.Op("func1")
        op2 = ir.Op("func2")

        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        call1 = ir.Call(op1, [x], ir.Span.unknown())
        call2 = ir.Call(op2, [x], ir.Span.unknown())

        assert not ir.structural_equal(call1, call2)

    def test_different_arg_count_not_equal(self):
        """Test that calls with different argument counts are not equal."""
        op = ir.Op("func")

        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        call1 = ir.Call(op, [x], ir.Span.unknown())
        call2 = ir.Call(op, [x, y], ir.Span.unknown())

        assert not ir.structural_equal(call1, call2)

    def test_empty_call_args_equal(self):
        """Test that calls with empty args lists can be equal."""
        op1 = ir.Op("func")
        op2 = ir.Op("func")

        call1 = ir.Call(op1, [], ir.Span.unknown())
        call2 = ir.Call(op2, [], ir.Span.unknown())

        assert ir.structural_equal(call1, call2)

    def test_stmt_structural_equal(self):
        """Test structural equality of Stmt nodes."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        stmt1 = ir.Stmt(span1)
        stmt2 = ir.Stmt(span2)

        # Same structure - should be equal
        assert ir.structural_equal(stmt1, stmt2)

    def test_stmt_different_spans_structural_equal(self):
        """Test that Stmt nodes with different spans are structurally equal."""
        span1 = ir.Span("file1.py", 1, 1, 1, 10)
        span2 = ir.Span("file2.py", 2, 2, 2, 20)

        stmt1 = ir.Stmt(span1)
        stmt2 = ir.Stmt(span2)

        # Different spans, but structural equality ignores span - should be equal
        assert ir.structural_equal(stmt1, stmt2)

    def test_stmt_different_from_expr_not_equal(self):
        """Test that Stmt and Expr nodes are not structurally equal."""
        span = ir.Span.unknown()

        stmt = ir.Stmt(span)
        expr = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Different IR node types should not be equal
        assert not ir.structural_equal(stmt, expr)


class TestHashEqualityConsistency:
    """Test that hash and equality are consistent."""

    def test_equal_implies_same_hash(self):
        """Test that structurally equal expressions have the same hash."""
        # Create several pairs of structurally equal expressions
        test_cases = [
            (
                ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
            ),
            (
                ir.ConstInt(42, DataType.INT64, ir.Span.unknown()),
                ir.ConstInt(42, DataType.INT64, ir.Span.unknown()),
            ),
            (
                ir.Add(
                    ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    DataType.INT64,
                    ir.Span.unknown(),
                ),
                ir.Add(
                    ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    DataType.INT64,
                    ir.Span.unknown(),
                ),
            ),
            (
                ir.Neg(
                    ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    DataType.INT64,
                    ir.Span.unknown(),
                ),
                ir.Neg(
                    ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    DataType.INT64,
                    ir.Span.unknown(),
                ),
            ),
            (
                ir.Stmt(ir.Span.unknown()),
                ir.Stmt(ir.Span.unknown()),
            ),
        ]

        for expr1, expr2 in test_cases:
            if ir.structural_equal(expr1, expr2):
                assert ir.structural_hash(expr1) == ir.structural_hash(expr2), (
                    f"Equal expressions should have same hash: {expr1} vs {expr2}"
                )

    def test_deep_nested_consistency(self):
        """Test hash/equality consistency for deeply nested expressions."""

        # Build: (((x + 1) - 2) * 3) / 4
        def build_expr(dtype, sp):
            x = ir.Var("x", ir.ScalarType(dtype), sp)
            c1 = ir.ConstInt(1, dtype, sp)
            c2 = ir.ConstInt(2, dtype, sp)
            c3 = ir.ConstInt(3, dtype, sp)
            c4 = ir.ConstInt(4, dtype, sp)

            return ir.FloatDiv(
                ir.Mul(ir.Sub(ir.Add(x, c1, dtype, sp), c2, dtype, sp), c3, dtype, sp), c4, dtype, sp
            )

        expr1 = build_expr(DataType.INT64, ir.Span.unknown())
        expr2 = build_expr(DataType.INT64, ir.Span.unknown())

        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
        assert ir.structural_hash(expr1, enable_auto_mapping=True) == ir.structural_hash(
            expr2, enable_auto_mapping=True
        )


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_comparison_operations(self):
        """Test all comparison operation types."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        ops = [
            (ir.Eq, ir.Eq),
            (ir.Ne, ir.Ne),
            (ir.Lt, ir.Lt),
            (ir.Le, ir.Le),
            (ir.Gt, ir.Gt),
            (ir.Ge, ir.Ge),
        ]

        for op1, op2 in ops:
            expr1 = op1(x, y, DataType.INT64, ir.Span.unknown())
            expr2 = op2(x, y, DataType.INT64, ir.Span.unknown())
            assert ir.structural_equal(expr1, expr2)

    def test_logical_operations(self):
        """Test all logical operation types."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        ops = [(ir.And, ir.And), (ir.Or, ir.Or), (ir.Xor, ir.Xor)]

        for op1, op2 in ops:
            expr1 = op1(x, y, DataType.INT64, ir.Span.unknown())
            expr2 = op2(x, y, DataType.INT64, ir.Span.unknown())
            assert ir.structural_equal(expr1, expr2)

    def test_bitwise_operations(self):
        """Test all bitwise operation types."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        ops = [
            (ir.BitAnd, ir.BitAnd),
            (ir.BitOr, ir.BitOr),
            (ir.BitXor, ir.BitXor),
            (ir.BitShiftLeft, ir.BitShiftLeft),
            (ir.BitShiftRight, ir.BitShiftRight),
        ]

        for op1, op2 in ops:
            expr1 = op1(x, y, DataType.INT64, ir.Span.unknown())
            expr2 = op2(x, y, DataType.INT64, ir.Span.unknown())
            assert ir.structural_equal(expr1, expr2)

    def test_all_unary_operations(self):
        """Test all unary operation types."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        ops = [(ir.Abs, ir.Abs), (ir.Neg, ir.Neg), (ir.Not, ir.Not), (ir.BitNot, ir.BitNot)]

        for op1, op2 in ops:
            expr1 = op1(x, DataType.INT64, ir.Span.unknown())
            expr2 = op2(x, DataType.INT64, ir.Span.unknown())
            assert ir.structural_equal(expr1, expr2)

    def test_math_operations(self):
        """Test mathematical operation types."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        ops = [(ir.Min, ir.Min), (ir.Max, ir.Max), (ir.Pow, ir.Pow)]

        for op1, op2 in ops:
            expr1 = op1(x, y, DataType.INT64, ir.Span.unknown())
            expr2 = op2(x, y, DataType.INT64, ir.Span.unknown())
            assert ir.structural_equal(expr1, expr2)


class TestAutoMapping:
    """Tests for auto mapping feature in structural equality and hash."""

    def test_auto_mapping_simple_vars_equal(self):
        """Test that x+1 equals y+1 with auto mapping enabled."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1_1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c1_2 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        expr1 = ir.Add(x, c1_1, DataType.INT64, ir.Span.unknown())  # x + 1
        expr2 = ir.Add(y, c1_2, DataType.INT64, ir.Span.unknown())  # y + 1

        # Without auto mapping, they should NOT be equal
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=False)

        # With auto mapping, they SHOULD be equal
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_auto_mapping_simple_vars_not_equal(self):
        """Test that x+1 does not equal y+1 without auto mapping."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        expr1 = ir.Add(x, c1, DataType.INT64, ir.Span.unknown())  # x + 1
        expr2 = ir.Add(y, c1, DataType.INT64, ir.Span.unknown())  # y + 1

        # Without auto mapping (default), they should NOT be equal
        assert not ir.structural_equal(expr1, expr2)
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        expr1 = ir.Add(x, c1, DataType.INT64, ir.Span.unknown())  # x + 1
        expr2 = ir.Add(y, c1, DataType.INT64, ir.Span.unknown())  # y + 1

        # Without auto mapping (default), they should NOT be equal
        assert not ir.structural_equal(expr1, expr2)

    def test_auto_mapping_hash_consistency(self):
        """Test that x+1 and y+1 hash to same value with auto mapping."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1_1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c1_2 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        expr1 = ir.Add(x, c1_1, DataType.INT64, ir.Span.unknown())  # x + 1
        expr2 = ir.Add(y, c1_2, DataType.INT64, ir.Span.unknown())  # y + 1

        # Without auto mapping, hashes should be different
        hash1_no_auto = ir.structural_hash(expr1, enable_auto_mapping=False)
        hash2_no_auto = ir.structural_hash(expr2, enable_auto_mapping=False)
        assert hash1_no_auto != hash2_no_auto

        # With auto mapping, hashes should be the same
        hash1_auto = ir.structural_hash(expr1, enable_auto_mapping=True)
        hash2_auto = ir.structural_hash(expr2, enable_auto_mapping=True)
        assert hash1_auto == hash2_auto

    def test_auto_mapping_multiple_vars(self):
        """Test auto mapping with multiple different variables."""

        # Build: (x + y) * z
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        z = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr1 = ir.Mul(ir.Add(x, y, DataType.INT64, ir.Span.unknown()), z, DataType.INT64, ir.Span.unknown())

        # Build: (a + b) * c
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c = ir.Var("c", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr2 = ir.Mul(ir.Add(a, b, DataType.INT64, ir.Span.unknown()), c, DataType.INT64, ir.Span.unknown())

        # Without auto mapping, should not be equal
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=False)

        # With auto mapping, should be equal
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

        # Hashes should also match with auto mapping
        assert ir.structural_hash(expr1, enable_auto_mapping=True) == ir.structural_hash(
            expr2, enable_auto_mapping=True
        )

    def test_auto_mapping_consistent_mapping(self):
        """Test that auto mapping maintains consistent variable mapping."""
        # Build: x + x (same variable used twice)
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr1 = ir.Add(x, x, DataType.INT64, ir.Span.unknown())

        # Build: y + y (same variable used twice)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr2 = ir.Add(y, y, DataType.INT64, ir.Span.unknown())

        # With auto mapping, x maps to y consistently
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_auto_mapping_inconsistent_mapping_fails(self):
        """Test that inconsistent variable mapping is rejected."""
        # Build: x + x (same variable used twice)
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr1 = ir.Add(x, x, DataType.INT64, ir.Span.unknown())

        # Build: y + z (different variables)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        z = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr2 = ir.Add(y, z, DataType.INT64, ir.Span.unknown())

        # With auto mapping, this should fail because x can't map to both y and z
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_auto_mapping_complex_expression(self):
        """Test auto mapping with complex nested expressions."""

        # Build: ((x + 1) * (y - 2)) / (x + y)
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1_1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c2_1 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        expr1 = ir.FloatDiv(
            ir.Mul(
                ir.Add(x1, c1_1, DataType.INT64, ir.Span.unknown()),
                ir.Sub(y1, c2_1, DataType.INT64, ir.Span.unknown()),
                DataType.INT64,
                ir.Span.unknown(),
            ),
            ir.Add(x1, y1, DataType.INT64, ir.Span.unknown()),
            DataType.INT64,
            ir.Span.unknown(),
        )

        # Build: ((a + 1) * (b - 2)) / (a + b)
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1_2 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c2_2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        expr2 = ir.FloatDiv(
            ir.Mul(
                ir.Add(a, c1_2, DataType.INT64, ir.Span.unknown()),
                ir.Sub(b, c2_2, DataType.INT64, ir.Span.unknown()),
                DataType.INT64,
                ir.Span.unknown(),
            ),
            ir.Add(a, b, DataType.INT64, ir.Span.unknown()),
            DataType.INT64,
            ir.Span.unknown(),
        )

        # With auto mapping: x->a, y->b consistently
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
        assert ir.structural_hash(expr1, enable_auto_mapping=True) == ir.structural_hash(
            expr2, enable_auto_mapping=True
        )

    def test_auto_mapping_same_vars_still_equal(self):
        """Test that expressions with same variable names are still equal with auto mapping."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        expr1 = ir.Add(x1, c1, DataType.INT64, ir.Span.unknown())
        expr2 = ir.Add(x2, c2, DataType.INT64, ir.Span.unknown())

        # Should be equal both with and without auto mapping
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=False)
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_auto_mapping_default_false(self):
        """Test that auto mapping is disabled by default."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        expr1 = ir.Add(x, c1, DataType.INT64, ir.Span.unknown())
        expr2 = ir.Add(y, c1, DataType.INT64, ir.Span.unknown())

        # Default behavior should require exact variable name match
        assert not ir.structural_equal(expr1, expr2)

        # Hash should also include variable names by default
        assert ir.structural_hash(expr1) != ir.structural_hash(expr2)

    def test_auto_mapping_with_unary_ops(self):
        """Test auto mapping with unary operations."""

        # Build: -x
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr1 = ir.Neg(x, DataType.INT64, ir.Span.unknown())

        # Build: -y
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr2 = ir.Neg(y, DataType.INT64, ir.Span.unknown())

        # With auto mapping, should be equal
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
        assert ir.structural_hash(expr1, enable_auto_mapping=True) == ir.structural_hash(
            expr2, enable_auto_mapping=True
        )

    def test_auto_mapping_with_call_expr(self):
        """Test auto mapping with call expressions."""
        op = ir.Op("func")

        # Build: func(x, y)
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        call1 = ir.Call(op, [x, y], ir.Span.unknown())

        # Build: func(a, b)
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        call2 = ir.Call(op, [a, b], ir.Span.unknown())

        # With auto mapping, should be equal
        assert ir.structural_equal(call1, call2, enable_auto_mapping=True)
        assert ir.structural_hash(call1, enable_auto_mapping=True) == ir.structural_hash(
            call2, enable_auto_mapping=True
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
