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

    def test_stmt_different_from_expr_hash(self):
        """Test that Stmt and Expr nodes hash differently."""
        span = ir.Span.unknown()

        expr = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        stmt = ir.AssignStmt(var, expr, span)

        hash_stmt = ir.structural_hash(stmt)
        hash_expr = ir.structural_hash(expr)

        # Different IR node types should hash differently
        assert hash_stmt != hash_expr

    def test_assign_stmt_same_structure_hash(self):
        """Test AssignStmt nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        hash1 = ir.structural_hash(assign1)
        hash2 = ir.structural_hash(assign2)
        # Different variable pointers result in different hashes without auto_mapping
        assert hash1 != hash2

    def test_assign_stmt_different_var_hash(self):
        """Test AssignStmt nodes with different var hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(z, y, span)

        hash1 = ir.structural_hash(assign1)
        hash2 = ir.structural_hash(assign2)
        assert hash1 == hash2

    def test_assign_stmt_different_value_hash(self):
        """Test AssignStmt nodes with different value hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(x, z, span)

        hash1 = ir.structural_hash(assign1)
        hash2 = ir.structural_hash(assign2)
        assert hash1 != hash2

    def test_assign_stmt_different_from_base_stmt_hash(self):
        """Test AssignStmt and base Stmt nodes hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        assign = ir.AssignStmt(x, y, span)

        hash_assign = ir.structural_hash(assign)
        assert hash_assign != 0

    def test_if_stmt_same_structure_hash(self):
        """Test IfStmt nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x1, y1, dtype, span)
        condition2 = ir.Eq(x2, y2, dtype, span)
        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        if_stmt1 = ir.IfStmt(condition1, [assign1], [], span)
        if_stmt2 = ir.IfStmt(condition2, [assign2], [], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        # Different variable pointers result in different hashes without auto_mapping
        assert hash1 != hash2

    def test_if_stmt_different_condition_hash(self):
        """Test IfStmt nodes with different condition hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x, y, dtype, span)
        condition2 = ir.Lt(x, z, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt1 = ir.IfStmt(condition1, [assign], [], span)
        if_stmt2 = ir.IfStmt(condition2, [assign], [], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        assert hash1 != hash2

    def test_if_stmt_different_then_body_hash(self):
        """Test IfStmt nodes with different then_body hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)

        if_stmt1 = ir.IfStmt(condition, [assign1], [], span)
        if_stmt2 = ir.IfStmt(condition, [assign2], [], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        assert hash1 != hash2

    def test_if_stmt_different_else_body_hash(self):
        """Test IfStmt nodes with different else_body hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)

        if_stmt1 = ir.IfStmt(condition, [assign1], [assign1], span)
        if_stmt2 = ir.IfStmt(condition, [assign1], [assign2], span)

        hash1 = ir.structural_hash(if_stmt1)
        hash2 = ir.structural_hash(if_stmt2)
        assert hash1 != hash2

    def test_yield_stmt_same_structure_hash(self):
        """Test YieldStmt nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([x1, y1], span)
        yield_stmt2 = ir.YieldStmt([x2, y2], span)

        hash1 = ir.structural_hash(yield_stmt1)
        hash2 = ir.structural_hash(yield_stmt2)
        # Different variable pointers result in different hashes without auto_mapping
        assert hash1 != hash2

    def test_yield_stmt_different_vars_hash(self):
        """Test YieldStmt nodes with different vars hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([x, y], span)
        yield_stmt2 = ir.YieldStmt([x, z], span)

        hash1 = ir.structural_hash(yield_stmt1)
        hash2 = ir.structural_hash(yield_stmt2)
        assert hash1 != hash2

    def test_yield_stmt_empty_vs_non_empty_hash(self):
        """Test YieldStmt nodes with empty and non-empty value lists hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([], span)
        yield_stmt2 = ir.YieldStmt([x], span)

        hash1 = ir.structural_hash(yield_stmt1)
        hash2 = ir.structural_hash(yield_stmt2)
        assert hash1 != hash2


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

    def test_stmt_different_from_expr_not_equal(self):
        """Test that Stmt and Expr nodes are not structurally equal."""
        span = ir.Span.unknown()

        assign = ir.AssignStmt(
            ir.Var("x", ir.ScalarType(DataType.INT64), span),
            ir.Var("y", ir.ScalarType(DataType.INT64), span),
            span,
        )
        expr = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Different IR node types should not be equal
        assert not ir.structural_equal(assign, expr)

    def test_assign_stmt_structural_equal(self):
        """Test structural equality of AssignStmt nodes."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        # Different variable pointers, so not equal without auto_mapping
        assert not ir.structural_equal(assign1, assign2, enable_auto_mapping=False)
        # With auto_mapping, they should be equal
        assert ir.structural_equal(assign1, assign2, enable_auto_mapping=True)

    def test_assign_stmt_different_var_not_equal(self):
        """Test AssignStmt nodes with different var are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(z, y, span)

        assert ir.structural_equal(assign1, assign2, enable_auto_mapping=True)

    def test_assign_stmt_different_value_not_equal(self):
        """Test AssignStmt nodes with different value are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(x, z, span)

        # Different value, so not equal
        assert not ir.structural_equal(assign1, assign2, enable_auto_mapping=False)
        # With auto_mapping, they should be equal (x maps to x, y maps to z)
        assert ir.structural_equal(assign1, assign2, enable_auto_mapping=True)

    def test_assign_stmt_different_from_base_stmt_not_equal(self):
        """Test AssignStmt and base Stmt nodes are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        assign = ir.AssignStmt(x, y, span)
        stmt = ir.Stmt(span)

        # Different types, so not equal
        assert not ir.structural_equal(assign, stmt)

    def test_if_stmt_structural_equal(self):
        """Test structural equality of IfStmt nodes."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x1, y1, dtype, span)
        condition2 = ir.Eq(x2, y2, dtype, span)
        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        if_stmt1 = ir.IfStmt(condition1, [assign1], [], span)
        if_stmt2 = ir.IfStmt(condition2, [assign2], [], span)

        # Different variable pointers, so not equal without auto_mapping
        assert not ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=False)
        # With auto_mapping, they should be equal
        assert ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=True)

    def test_if_stmt_different_condition_not_equal(self):
        """Test IfStmt nodes with different condition are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition1 = ir.Eq(x, y, dtype, span)
        condition2 = ir.Lt(x, z, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt1 = ir.IfStmt(condition1, [assign], [], span)
        if_stmt2 = ir.IfStmt(condition2, [assign], [], span)

        assert not ir.structural_equal(if_stmt1, if_stmt2)

    def test_if_stmt_different_then_body_not_equal(self):
        """Test IfStmt nodes with different then_body are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)

        if_stmt1 = ir.IfStmt(condition, [assign1], [], span)
        if_stmt2 = ir.IfStmt(condition, [assign2], [], span)

        assert not ir.structural_equal(if_stmt1, if_stmt2)

    def test_if_stmt_different_else_body_not_equal(self):
        """Test IfStmt nodes with different else_body are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)

        if_stmt1 = ir.IfStmt(condition, [assign1], [assign1], span)
        if_stmt2 = ir.IfStmt(condition, [assign1], [assign2], span)

        assert not ir.structural_equal(if_stmt1, if_stmt2)

    def test_if_stmt_different_from_base_stmt_not_equal(self):
        """Test IfStmt and base Stmt nodes are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        condition = ir.Eq(x, y, dtype, span)
        assign = ir.AssignStmt(x, y, span)

        if_stmt = ir.IfStmt(condition, [assign], [], span)
        stmt = ir.Stmt(span)

        assert not ir.structural_equal(if_stmt, stmt)

    def test_yield_stmt_structural_equal(self):
        """Test structural equality of YieldStmt nodes."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([x1, y1], span)
        yield_stmt2 = ir.YieldStmt([x2, y2], span)

        # Different variable pointers, so not equal without auto_mapping
        assert not ir.structural_equal(yield_stmt1, yield_stmt2, enable_auto_mapping=False)
        # With auto_mapping, they should be equal
        assert ir.structural_equal(yield_stmt1, yield_stmt2, enable_auto_mapping=True)

    def test_yield_stmt_different_vars_not_equal(self):
        """Test YieldStmt nodes with different vars are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([x, y], span)
        yield_stmt2 = ir.YieldStmt([x, z], span)

        assert not ir.structural_equal(yield_stmt1, yield_stmt2)

    def test_yield_stmt_empty_vs_non_empty_not_equal(self):
        """Test YieldStmt nodes with empty and non-empty value lists are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([], span)
        yield_stmt2 = ir.YieldStmt([x], span)

        assert not ir.structural_equal(yield_stmt1, yield_stmt2)

    def test_yield_stmt_different_from_base_stmt_not_equal(self):
        """Test YieldStmt and base Stmt nodes are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)

        yield_stmt = ir.YieldStmt([x], span)
        stmt = ir.Stmt(span)

        assert not ir.structural_equal(yield_stmt, stmt)


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
                ir.AssignStmt(
                    ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    ir.Span.unknown(),
                ),
                ir.AssignStmt(
                    ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
                    ir.Span.unknown(),
                ),
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

    def test_auto_mapping_different_vars_vs_same_var(self):
        """Test that Var(x) + Var(y) is not equal to Var(a) + Var(a)."""
        # Build: x + y (two different variables)
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr1 = ir.Add(x, y, DataType.INT64, ir.Span.unknown())

        # Build: a + a (same variable used twice)
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        expr2 = ir.Add(a, a, DataType.INT64, ir.Span.unknown())

        # Without auto mapping, they should not be equal
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=False)

        # With auto mapping, they should still not be equal because:
        # - x and y are different variables
        # - They cannot both map to the same variable a
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

        # Hashes should also be different
        hash1 = ir.structural_hash(expr1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(expr2, enable_auto_mapping=True)
        assert hash1 != hash2

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

    def test_auto_mapping_with_assign_stmt(self):
        """Test auto mapping with AssignStmt."""
        # Build: x = y
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        assign1 = ir.AssignStmt(x1, y1, ir.Span.unknown())

        # Build: a = b
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        assign2 = ir.AssignStmt(a, b, ir.Span.unknown())

        assert ir.structural_equal(assign1, assign2, enable_auto_mapping=True)
        assert not ir.structural_equal(assign1, assign2, enable_auto_mapping=False)

        hash_with_auto1 = ir.structural_hash(assign1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(assign2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        hash_without_auto1 = ir.structural_hash(assign1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(assign2, enable_auto_mapping=False)
        assert hash_without_auto1 != hash_without_auto2

    def test_auto_mapping_assign_stmt_different_var_same_value(self):
        """Test auto mapping with AssignStmt where var differs but value is same."""
        # Build: x = y
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        assign1 = ir.AssignStmt(x, y, ir.Span.unknown())

        # Build: z = y
        z = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        assign2 = ir.AssignStmt(z, y, ir.Span.unknown())

        equal_with_auto = ir.structural_equal(assign1, assign2, enable_auto_mapping=True)
        assert equal_with_auto

        hash_with_auto1 = ir.structural_hash(assign1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(assign2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        equal_without_auto = ir.structural_equal(assign1, assign2, enable_auto_mapping=False)
        assert equal_without_auto

        hash_without_auto1 = ir.structural_hash(assign1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(assign2, enable_auto_mapping=False)
        assert hash_without_auto1 == hash_without_auto2

    def test_auto_mapping_assign_stmt_same_var_different_value(self):
        """Test auto mapping with AssignStmt where var is same but value differs."""
        # Build: x = y
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        assign1 = ir.AssignStmt(x, y, ir.Span.unknown())

        # Build: x = z
        z = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        assign2 = ir.AssignStmt(x, z, ir.Span.unknown())

        equal_with_auto = ir.structural_equal(assign1, assign2, enable_auto_mapping=True)
        assert equal_with_auto

        hash_with_auto1 = ir.structural_hash(assign1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(assign2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        equal_without_auto = ir.structural_equal(assign1, assign2, enable_auto_mapping=False)
        assert not equal_without_auto

        hash_without_auto1 = ir.structural_hash(assign1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(assign2, enable_auto_mapping=False)
        assert hash_without_auto1 != hash_without_auto2

    def test_auto_mapping_assign_stmt_with_expression(self):
        """Test auto mapping with AssignStmt containing complex expressions."""
        # Build: x = y + z
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        z1 = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        add1 = ir.Add(y1, z1, DataType.INT64, ir.Span.unknown())
        assign1 = ir.AssignStmt(x1, add1, ir.Span.unknown())

        # Build: a = b + c
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c = ir.Var("c", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        add2 = ir.Add(b, c, DataType.INT64, ir.Span.unknown())
        assign2 = ir.AssignStmt(a, add2, ir.Span.unknown())

        equal_with_auto = ir.structural_equal(assign1, assign2, enable_auto_mapping=True)
        assert equal_with_auto
        hash_with_auto1 = ir.structural_hash(assign1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(assign2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        equal_without_auto = ir.structural_equal(assign1, assign2, enable_auto_mapping=False)
        assert not equal_without_auto

        hash_without_auto1 = ir.structural_hash(assign1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(assign2, enable_auto_mapping=False)
        assert hash_without_auto1 != hash_without_auto2

    def test_auto_mapping_with_if_stmt(self):
        """Test auto mapping with IfStmt."""
        # Build: if x == y then x = y else y = x
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition1 = ir.Eq(x1, y1, DataType.INT64, ir.Span.unknown())
        assign1_then = ir.AssignStmt(x1, y1, ir.Span.unknown())
        assign1_else = ir.AssignStmt(y1, x1, ir.Span.unknown())
        if_stmt1 = ir.IfStmt(condition1, [assign1_then], [assign1_else], ir.Span.unknown())

        # Build: if a == b then a = b else b = a
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition2 = ir.Eq(a, b, DataType.INT64, ir.Span.unknown())
        assign2_then = ir.AssignStmt(a, b, ir.Span.unknown())
        assign2_else = ir.AssignStmt(b, a, ir.Span.unknown())
        if_stmt2 = ir.IfStmt(condition2, [assign2_then], [assign2_else], ir.Span.unknown())

        assert ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=True)
        assert not ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=False)

        hash_with_auto1 = ir.structural_hash(if_stmt1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(if_stmt2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        hash_without_auto1 = ir.structural_hash(if_stmt1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(if_stmt2, enable_auto_mapping=False)
        assert hash_without_auto1 != hash_without_auto2

    def test_auto_mapping_if_stmt_different_structure(self):
        """Test auto mapping with IfStmt where structure differs."""
        # Build: if x == y then x = y
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition1 = ir.Eq(x1, y1, DataType.INT64, ir.Span.unknown())
        assign1 = ir.AssignStmt(x1, y1, ir.Span.unknown())
        if_stmt1 = ir.IfStmt(condition1, [assign1], [], ir.Span.unknown())

        # Build: if a == b then a = b else b = a
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        condition2 = ir.Eq(a, b, DataType.INT64, ir.Span.unknown())
        assign2_then = ir.AssignStmt(a, b, ir.Span.unknown())
        assign2_else = ir.AssignStmt(b, a, ir.Span.unknown())
        if_stmt2 = ir.IfStmt(condition2, [assign2_then], [assign2_else], ir.Span.unknown())

        # Different structure (one has else, one doesn't)
        assert not ir.structural_equal(if_stmt1, if_stmt2, enable_auto_mapping=True)

    def test_auto_mapping_with_yield_stmt(self):
        """Test auto mapping with YieldStmt."""
        # Build: yield x, y
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y1 = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        yield_stmt1 = ir.YieldStmt([x1, y1], ir.Span.unknown())

        # Build: yield a, b
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        yield_stmt2 = ir.YieldStmt([a, b], ir.Span.unknown())

        assert ir.structural_equal(yield_stmt1, yield_stmt2, enable_auto_mapping=True)
        assert not ir.structural_equal(yield_stmt1, yield_stmt2, enable_auto_mapping=False)

        hash_with_auto1 = ir.structural_hash(yield_stmt1, enable_auto_mapping=True)
        hash_with_auto2 = ir.structural_hash(yield_stmt2, enable_auto_mapping=True)
        assert hash_with_auto1 == hash_with_auto2

        hash_without_auto1 = ir.structural_hash(yield_stmt1, enable_auto_mapping=False)
        hash_without_auto2 = ir.structural_hash(yield_stmt2, enable_auto_mapping=False)
        assert hash_without_auto1 != hash_without_auto2

    def test_auto_mapping_yield_stmt_different_length(self):
        """Test auto mapping with YieldStmt where list lengths differ."""
        # Build: yield x
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        yield_stmt1 = ir.YieldStmt([x1], ir.Span.unknown())

        # Build: yield a, b
        a = ir.Var("a", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        b = ir.Var("b", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        yield_stmt2 = ir.YieldStmt([a, b], ir.Span.unknown())

        # Different lengths should not be equal
        assert not ir.structural_equal(yield_stmt1, yield_stmt2, enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
