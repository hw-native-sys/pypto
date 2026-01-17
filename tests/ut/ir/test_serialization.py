# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Comprehensive tests for IR serialization and deserialization."""

import tempfile
from pathlib import Path
from typing import cast

import pytest
from pypto import DataType, ir


class TestBasicSerialization:
    """Tests for basic serialization of simple IR nodes."""

    def test_serialize_var(self):
        """Test serialization of Var node."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        data = ir.serialize(x)
        assert isinstance(data, bytes)
        assert len(data) > 0
        restored = ir.deserialize(data)
        assert ir.structural_equal(x, restored, enable_auto_mapping=True)

    def test_serialize_iter_arg(self):
        """Test serialization of IterArg node."""
        init_value = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        iter_arg = ir.IterArg("iter_arg", ir.ScalarType(DataType.INT64), init_value, ir.Span.unknown())

        data = ir.serialize(iter_arg)
        assert isinstance(data, bytes)
        assert len(data) > 0
        restored = ir.deserialize(data)
        restored_iter_arg = cast(ir.IterArg, restored)

        assert ir.structural_equal(iter_arg, restored, enable_auto_mapping=True)
        assert restored_iter_arg.name == "iter_arg"
        assert isinstance(restored_iter_arg.initValue, ir.ConstInt)
        assert cast(ir.ConstInt, restored_iter_arg.initValue).value == 5

    def test_serialize_iter_arg_with_expr_init_value(self):
        """Test serialization of IterArg with expression as initValue."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        init_value = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
        iter_arg = ir.IterArg("iter_arg", ir.ScalarType(DataType.INT64), init_value, ir.Span.unknown())

        data = ir.serialize(iter_arg)
        restored = ir.deserialize(data)
        restored_iter_arg = cast(ir.IterArg, restored)

        assert ir.structural_equal(iter_arg, restored, enable_auto_mapping=True)
        assert isinstance(restored_iter_arg.initValue, ir.Add)

    def test_serialize_const_int(self):
        """Test serialization of ConstInt node."""
        c = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

        data = ir.serialize(c)
        restored = ir.deserialize(data)

        assert ir.structural_equal(c, restored, enable_auto_mapping=True)

    def test_serialize_binary_expr(self):
        """Test serialization of binary expressions."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        add = ir.Add(x, y, DataType.INT64, ir.Span.unknown())

        data = ir.serialize(add)
        restored = ir.deserialize(data)

        assert ir.structural_equal(add, restored, enable_auto_mapping=True)

    def test_serialize_unary_expr(self):
        """Test serialization of unary expressions."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        neg = ir.Neg(x, DataType.INT64, ir.Span.unknown())

        data = ir.serialize(neg)
        restored = ir.deserialize(data)

        assert ir.structural_equal(neg, restored, enable_auto_mapping=True)

    def test_serialize_call(self):
        """Test serialization of Call expression."""
        op = ir.Op("func")
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        call = ir.Call(op, [x, y], ir.Span.unknown())

        data = ir.serialize(call)
        restored = ir.deserialize(data)

        assert ir.structural_equal(call, restored, enable_auto_mapping=True)


class TestComplexExpressions:
    """Tests for serialization of complex nested expressions."""

    def test_serialize_nested_arithmetic(self):
        """Test serialization of nested arithmetic expression: (x + 5) * 2."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c5 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())

        add = ir.Add(x, c5, DataType.INT64, ir.Span.unknown())
        mul = ir.Mul(add, c2, DataType.INT64, ir.Span.unknown())

        data = ir.serialize(mul)
        restored = ir.deserialize(data)

        assert ir.structural_equal(mul, restored, enable_auto_mapping=True)

    def test_serialize_deeply_nested(self):
        """Test serialization of deeply nested expression."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        c3 = ir.ConstInt(3, DataType.INT64, ir.Span.unknown())
        c4 = ir.ConstInt(4, DataType.INT64, ir.Span.unknown())

        # Build: (((x + 1) - 2) * 3) / 4
        expr = ir.FloatDiv(
            ir.Mul(
                ir.Sub(
                    ir.Add(x, c1, DataType.INT64, ir.Span.unknown()),
                    c2,
                    DataType.INT64,
                    ir.Span.unknown(),
                ),
                c3,
                DataType.INT64,
                ir.Span.unknown(),
            ),
            c4,
            DataType.INT64,
            ir.Span.unknown(),
        )

        data = ir.serialize(expr)
        restored = ir.deserialize(data)

        assert ir.structural_equal(expr, restored, enable_auto_mapping=True)


class TestPointerSharing:
    """Tests for preserving pointer sharing during serialization."""

    def test_shared_var_in_expression(self):
        """Test that shared variable pointer is preserved: x + x."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        add = ir.Add(x, x, DataType.INT64, ir.Span.unknown())  # Same x used twice

        data = ir.serialize(add)
        restored = ir.deserialize(data)
        restored_add = cast(ir.Add, restored)

        # Check structural equality
        assert ir.structural_equal(add, restored_add, enable_auto_mapping=True)

        # Check that the left and right operands are the same object
        # In the restored version, they should also be the same object
        assert restored_add.left is restored_add.right

    def test_shared_subexpression(self):
        """Test that shared subexpression pointer is preserved."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        # Create shared subexpression: x + 1
        sub = ir.Add(x, c1, DataType.INT64, ir.Span.unknown())

        # Use it twice: (x+1) * (x+1)
        mul = ir.Mul(sub, sub, DataType.INT64, ir.Span.unknown())

        data = ir.serialize(mul)
        restored = ir.deserialize(data)
        restored_mul = cast(ir.Mul, restored)

        # Check structural equality
        assert ir.structural_equal(mul, restored_mul, enable_auto_mapping=True)

        # Check that left and right are the same object
        assert restored_mul.left is restored_mul.right

    def test_complex_pointer_sharing(self):
        """Test complex pointer sharing with multiple shared nodes."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        # Create: (x + y) and reuse it
        add = ir.Add(x, y, DataType.INT64, ir.Span.unknown())

        # Create: (x + y) * x
        mul1 = ir.Mul(add, x, DataType.INT64, ir.Span.unknown())

        # Create: (x + y) + y
        add2 = ir.Add(add, y, DataType.INT64, ir.Span.unknown())

        # Combine: ((x + y) * x) - ((x + y) + y)
        sub = ir.Sub(mul1, add2, DataType.INT64, ir.Span.unknown())

        data = ir.serialize(sub)
        restored = ir.deserialize(data)
        restored_sub = cast(ir.Sub, restored)

        # Check structural equality
        assert ir.structural_equal(sub, restored_sub, enable_auto_mapping=True)

        # Verify pointer sharing is preserved
        # mul1.left and add2.left should be the same object (the original 'add')
        assert cast(ir.Add, restored_sub.left).left is cast(ir.Add, restored_sub.right).left


class TestStatementSerialization:
    """Tests for statement serialization."""

    def test_serialize_assign_stmt(self):
        """Test serialization of AssignStmt."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        assign = ir.AssignStmt(x, y, ir.Span.unknown())

        data = ir.serialize(assign)
        restored = ir.deserialize(data)

        assert ir.structural_equal(assign, restored, enable_auto_mapping=True)

    def test_serialize_if_stmt(self):
        """Test serialization of IfStmt."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        z = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        cond = ir.Gt(x, y, DataType.INT64, ir.Span.unknown())
        then_body = ir.AssignStmt(z, x, ir.Span.unknown())
        else_body = ir.AssignStmt(z, y, ir.Span.unknown())

        if_stmt = ir.IfStmt(cond, then_body, else_body, [], ir.Span.unknown())

        data = ir.serialize(if_stmt)
        restored = ir.deserialize(data)

        assert ir.structural_equal(if_stmt, restored, enable_auto_mapping=True)

    def test_serialize_if_stmt_with_nullopt_else_body(self):
        """Test serialization of IfStmt with nullopt else_body."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        z = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        cond = ir.Gt(x, y, DataType.INT64, ir.Span.unknown())
        then_body = ir.AssignStmt(z, x, ir.Span.unknown())

        # IfStmt with nullopt else_body (using constructor that only takes then_body)
        if_stmt = ir.IfStmt(cond, then_body, None, [], ir.Span.unknown())

        data = ir.serialize(if_stmt)
        restored = ir.deserialize(data)
        restored_if_stmt = cast(ir.IfStmt, restored)

        # Check structural equality
        assert ir.structural_equal(if_stmt, restored, enable_auto_mapping=True)

        # Verify that else_body is None in the restored version
        assert restored_if_stmt.else_body is None

    def test_serialize_for_stmt(self):
        """Test serialization of ForStmt."""
        i = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        body = ir.AssignStmt(x, ir.Add(x, i, DataType.INT64, ir.Span.unknown()), ir.Span.unknown())

        for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], ir.Span.unknown())

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)

        assert ir.structural_equal(for_stmt, restored, enable_auto_mapping=True)

    def test_serialize_for_stmt_with_iter_args(self):
        """Test serialization of ForStmt with iter_args."""
        i = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        # Create IterArg instances
        init_value1 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        iter_arg1 = ir.IterArg("arg1", ir.ScalarType(DataType.INT64), init_value1, ir.Span.unknown())

        init_value2 = x
        iter_arg2 = ir.IterArg("arg2", ir.ScalarType(DataType.INT64), init_value2, ir.Span.unknown())

        body = ir.AssignStmt(x, ir.Add(x, i, DataType.INT64, ir.Span.unknown()), ir.Span.unknown())

        for_stmt = ir.ForStmt(i, start, stop, step, [iter_arg1, iter_arg2], body, [], ir.Span.unknown())

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)
        restored_for_stmt = cast(ir.ForStmt, restored)

        assert ir.structural_equal(for_stmt, restored, enable_auto_mapping=True)
        assert len(restored_for_stmt.iter_args) == 2
        assert restored_for_stmt.iter_args[0].name == "arg1"
        assert restored_for_stmt.iter_args[1].name == "arg2"
        assert isinstance(restored_for_stmt.iter_args[0].initValue, ir.ConstInt)
        assert isinstance(restored_for_stmt.iter_args[1].initValue, ir.Var)

    def test_serialize_for_stmt_with_empty_iter_args(self):
        """Test serialization of ForStmt with empty iter_args."""
        i = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

        body = ir.AssignStmt(x, ir.Add(x, i, DataType.INT64, ir.Span.unknown()), ir.Span.unknown())

        for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], ir.Span.unknown())

        data = ir.serialize(for_stmt)
        restored = ir.deserialize(data)
        restored_for_stmt = cast(ir.ForStmt, restored)

        assert ir.structural_equal(for_stmt, restored, enable_auto_mapping=True)
        assert len(restored_for_stmt.iter_args) == 0

    def test_serialize_yield_stmt(self):
        """Test serialization of YieldStmt."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        yield_stmt = ir.YieldStmt([x, y], ir.Span.unknown())

        data = ir.serialize(yield_stmt)
        restored = ir.deserialize(data)

        assert ir.structural_equal(yield_stmt, restored, enable_auto_mapping=True)

    def test_serialize_return_stmt(self):
        """Test serialization of ReturnStmt."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        return_stmt = ir.ReturnStmt([x, y], ir.Span.unknown())

        data = ir.serialize(return_stmt)
        restored = ir.deserialize(data)

        assert ir.structural_equal(return_stmt, restored, enable_auto_mapping=True)

    def test_serialize_return_stmt_with_single_value(self):
        """Test serialization of ReturnStmt with single value."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        return_stmt = ir.ReturnStmt([x], ir.Span.unknown())

        data = ir.serialize(return_stmt)
        restored = ir.deserialize(data)
        restored_return = cast(ir.ReturnStmt, restored)

        assert ir.structural_equal(return_stmt, restored, enable_auto_mapping=True)
        assert len(restored_return.value) == 1

    def test_serialize_return_stmt_empty(self):
        """Test serialization of ReturnStmt without values."""
        return_stmt = ir.ReturnStmt([], ir.Span.unknown())

        data = ir.serialize(return_stmt)
        restored = ir.deserialize(data)
        restored_return = cast(ir.ReturnStmt, restored)

        assert ir.structural_equal(return_stmt, restored, enable_auto_mapping=True)
        assert len(restored_return.value) == 0

    def test_serialize_return_stmt_with_expressions(self):
        """Test serialization of ReturnStmt with complex expressions."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        # Return with binary expression
        add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
        return_stmt = ir.ReturnStmt([add_expr], ir.Span.unknown())

        data = ir.serialize(return_stmt)
        restored = ir.deserialize(data)
        restored_return = cast(ir.ReturnStmt, restored)

        assert ir.structural_equal(return_stmt, restored, enable_auto_mapping=True)
        assert len(restored_return.value) == 1
        assert isinstance(restored_return.value[0], ir.Add)

    def test_serialize_return_stmt_multiple_expressions(self):
        """Test serialization of ReturnStmt with multiple different expressions."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
        add_expr = ir.Add(x, c, DataType.INT64, ir.Span.unknown())

        return_stmt = ir.ReturnStmt([x, c, add_expr], ir.Span.unknown())

        data = ir.serialize(return_stmt)
        restored = ir.deserialize(data)
        restored_return = cast(ir.ReturnStmt, restored)

        assert ir.structural_equal(return_stmt, restored, enable_auto_mapping=True)
        assert len(restored_return.value) == 3
        assert isinstance(restored_return.value[0], ir.Var)
        assert isinstance(restored_return.value[1], ir.ConstInt)
        assert isinstance(restored_return.value[2], ir.Add)

    def test_serialize_seq_stmts(self):
        """Test serialization of SeqStmts."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        z = ir.Var("z", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        stmts: list[ir.Stmt] = [
            ir.AssignStmt(x, ir.ConstInt(1, DataType.INT64, ir.Span.unknown()), ir.Span.unknown()),
            ir.AssignStmt(y, ir.ConstInt(2, DataType.INT64, ir.Span.unknown()), ir.Span.unknown()),
            ir.AssignStmt(z, ir.Add(x, y, DataType.INT64, ir.Span.unknown()), ir.Span.unknown()),
        ]

        seq = ir.SeqStmts(stmts, ir.Span.unknown())

        data = ir.serialize(seq)
        restored = ir.deserialize(data)

        assert ir.structural_equal(seq, restored, enable_auto_mapping=True)


class TestFunctionSerialization:
    """Tests for Function and Program serialization."""

    def test_serialize_function(self):
        """Test serialization of Function."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        result = ir.Var("result", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        body = ir.SeqStmts(
            [
                ir.AssignStmt(result, ir.Add(x, y, DataType.INT64, ir.Span.unknown()), ir.Span.unknown()),
                ir.YieldStmt([result], ir.Span.unknown()),
            ],
            ir.Span.unknown(),
        )

        func = ir.Function("add_func", [x, y], [ir.ScalarType(DataType.INT64)], body, ir.Span.unknown())

        data = ir.serialize(func)
        restored = ir.deserialize(data)

        assert ir.structural_equal(func, restored, enable_auto_mapping=True)

    def test_serialize_function_with_return_stmt(self):
        """Test serialization of Function with ReturnStmt."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        # Function body with return statement
        body = ir.ReturnStmt([ir.Add(x, y, DataType.INT64, ir.Span.unknown())], ir.Span.unknown())

        func = ir.Function("add_return", [x, y], [ir.ScalarType(DataType.INT64)], body, ir.Span.unknown())

        data = ir.serialize(func)
        restored = ir.deserialize(data)
        restored_func = cast(ir.Function, restored)

        assert ir.structural_equal(func, restored, enable_auto_mapping=True)
        assert isinstance(restored_func.body, ir.ReturnStmt)
        assert len(cast(ir.ReturnStmt, restored_func.body).value) == 1

    def test_serialize_program(self):
        """Test serialization of Program."""
        # Create a simple function
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        body = ir.YieldStmt([x], ir.Span.unknown())

        func = ir.Function("identity", [x], [ir.ScalarType(DataType.INT64)], body, ir.Span.unknown())

        program = ir.Program([func], "test_program", ir.Span.unknown())

        data = ir.serialize(program)
        restored = ir.deserialize(data)

        assert ir.structural_equal(program, restored, enable_auto_mapping=True)


class TestSpanSerialization:
    """Tests for Span serialization."""

    def test_serialize_with_span(self):
        """Test that Span information is preserved."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        data = ir.serialize(x)
        restored = ir.deserialize(data)

        # Span fields should be preserved
        assert restored.span.filename == "test.py"
        assert restored.span.begin_line == 10
        assert restored.span.begin_column == 5
        assert restored.span.end_line == 10
        assert restored.span.end_column == 15


class TestFileSerialization:
    """Tests for file I/O serialization."""

    def test_serialize_to_file(self):
        """Test serialization to and from file."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
        add = ir.Add(x, c, DataType.INT64, ir.Span.unknown())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.msgpack"

            # Serialize to file
            ir.serialize_to_file(add, str(filepath))

            # Verify file exists
            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Deserialize from file
            restored = ir.deserialize_from_file(str(filepath))

            # Verify equality
            assert ir.structural_equal(add, restored, enable_auto_mapping=True)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_serialize_all_binary_ops(self):
        """Test serialization of all binary operation types."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        ops = [
            ir.Add,
            ir.Sub,
            ir.Mul,
            ir.FloorDiv,
            ir.FloorMod,
            ir.FloatDiv,
            ir.Min,
            ir.Max,
            ir.Pow,
            ir.Eq,
            ir.Ne,
            ir.Lt,
            ir.Le,
            ir.Gt,
            ir.Ge,
            ir.And,
            ir.Or,
            ir.Xor,
            ir.BitAnd,
            ir.BitOr,
            ir.BitXor,
            ir.BitShiftLeft,
            ir.BitShiftRight,
        ]

        for op_class in ops:
            expr = op_class(x, y, DataType.INT64, ir.Span.unknown())
            data = ir.serialize(expr)
            restored = ir.deserialize(data)
            assert ir.structural_equal(expr, restored, enable_auto_mapping=True)

    def test_serialize_all_unary_ops(self):
        """Test serialization of all unary operation types."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        ops = [ir.Abs, ir.Neg, ir.Not, ir.BitNot]

        for op_class in ops:
            expr = op_class(x, DataType.INT64, ir.Span.unknown())
            data = ir.serialize(expr)
            restored = ir.deserialize(data)
            assert ir.structural_equal(expr, restored, enable_auto_mapping=True)

    def test_serialize_empty_collections(self):
        """Test serialization with empty collections."""
        # YieldStmt with empty value list
        yield_empty = ir.YieldStmt([], ir.Span.unknown())
        data = ir.serialize(yield_empty)
        restored = ir.deserialize(data)
        assert ir.structural_equal(yield_empty, restored, enable_auto_mapping=True)

        # ReturnStmt with empty value list
        return_empty = ir.ReturnStmt([], ir.Span.unknown())
        data = ir.serialize(return_empty)
        restored = ir.deserialize(data)
        assert ir.structural_equal(return_empty, restored, enable_auto_mapping=True)

        # Call with empty args
        op = ir.Op("func")
        call_empty = ir.Call(op, [], ir.Span.unknown())
        data = ir.serialize(call_empty)
        restored = ir.deserialize(data)
        assert ir.structural_equal(call_empty, restored, enable_auto_mapping=True)

        # ForStmt with empty iter_args
        i = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
        stop = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
        step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        body = ir.AssignStmt(i, start, ir.Span.unknown())
        for_stmt_empty = ir.ForStmt(i, start, stop, step, [], body, [], ir.Span.unknown())
        data = ir.serialize(for_stmt_empty)
        restored = ir.deserialize(data)
        assert ir.structural_equal(for_stmt_empty, restored, enable_auto_mapping=True)

    def test_serialize_global_var(self):
        """Test serialization of GlobalVar in Call."""
        gvar = ir.GlobalVar("my_func")
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        call = ir.Call(gvar, [x], ir.Span.unknown())

        data = ir.serialize(call)
        restored = ir.deserialize(data)

        assert ir.structural_equal(call, restored, enable_auto_mapping=True)


class TestRobustness:
    """Tests for error handling and robustness."""

    def test_deserialize_invalid_data(self):
        """Test that deserializing invalid data raises an error."""
        invalid_data = b"invalid msgpack data"

        with pytest.raises(ValueError):  # Should raise some kind of error
            ir.deserialize(invalid_data)

    def test_deserialize_nonexistent_file(self):
        """Test that deserializing from nonexistent file raises an error."""
        with pytest.raises(ValueError):
            ir.deserialize_from_file("/nonexistent/path/file.msgpack")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
