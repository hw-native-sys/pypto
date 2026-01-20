# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IR structural hash functionality."""

from pypto import DataType, ir


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

    def test_const_bool_hash(self):
        """Test that ConstBool with different values hash differently."""
        b_true1 = ir.ConstBool(True, ir.Span.unknown())
        b_true2 = ir.ConstBool(True, ir.Span.unknown())
        b_false = ir.ConstBool(False, ir.Span.unknown())

        hash_true1 = ir.structural_hash(b_true1)
        hash_true2 = ir.structural_hash(b_true2)
        hash_false = ir.structural_hash(b_false)

        # Same values should have same hash
        assert hash_true1 == hash_true2
        # Different values should have different hash
        assert hash_true1 != hash_false

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
