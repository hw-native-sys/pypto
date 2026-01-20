# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IR Builder."""

import pytest
from pypto import DataType, ir
from pypto.ir import IRBuilder


class TestIRBuilderFunction:
    """Test IR Builder for function construction."""

    def test_simple_function_with_auto_span(self):
        """Test building a simple function with automatic span capture."""
        ib = IRBuilder()

        with ib.function("my_func") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            y = f.param("y", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # Build body
            result = ib.var("result", ir.ScalarType(DataType.INT64))
            add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
            ib.assign(result, add_expr)

        func = f.get_result()

        assert func is not None
        assert func.name == "my_func"
        assert len(func.params) == 2
        assert len(func.return_types) == 1
        assert func.params[0].name == "x"
        assert func.params[1].name == "y"
        assert func.body is not None

    def test_function_with_explicit_span(self):
        """Test building a function with explicit span."""
        ib = IRBuilder()
        my_span = ir.Span("test.py", 10, 1)

        with ib.function("explicit_func", span=my_span) as f:
            _x = f.param("x", ir.ScalarType(DataType.INT32), span=my_span)
            f.return_type(ir.ScalarType(DataType.INT32))

        func = f.get_result()

        assert func is not None
        assert func.name == "explicit_func"
        assert func.span.filename == "test.py"
        assert func.span.begin_line == 10

    def test_function_with_multiple_statements(self):
        """Test function with multiple statements in body."""
        ib = IRBuilder()

        with ib.function("multi_stmt") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            a = ib.var("a", ir.ScalarType(DataType.INT64))
            b = ib.var("b", ir.ScalarType(DataType.INT64))

            ib.assign(a, x)
            const_one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            add_expr = ir.Add(a, const_one, DataType.INT64, ir.Span.unknown())
            ib.assign(b, add_expr)

        func = f.get_result()

        assert func is not None
        assert isinstance(func.body, ir.SeqStmts)
        assert len(func.body.stmts) == 2

    def test_nested_function_error(self):
        """Test that nested functions raise an error."""
        ib = IRBuilder()

        with pytest.raises(RuntimeError, match="Cannot begin function"):
            with ib.function("outer") as f:
                f.return_type(ir.ScalarType(DataType.INT64))

                # Try to nest another function - should fail
                with ib.function("inner") as _f2:
                    pass


class TestIRBuilderForLoop:
    """Test IR Builder for loop construction."""

    def test_simple_for_loop(self):
        """Test building a simple for loop."""
        ib = IRBuilder()

        with ib.function("loop_func") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1):
                # Empty loop body
                pass

        func = f.get_result()

        assert func is not None
        # Function body should be a for loop
        assert isinstance(func.body, ir.ForStmt)
        assert func.body.loop_var.name == "i"

    def test_for_loop_with_iter_args(self):
        """Test for loop with iteration arguments."""
        ib = IRBuilder()

        with ib.function("sum_func") as f:
            n = f.param("n", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, n, 1) as loop:
                sum_iter = loop.iter_arg("sum", 0)
                sum_final = loop.return_var("sum_final")

                # Body: sum = sum + i
                add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
                yield_stmt = ir.YieldStmt([add_expr], ir.Span.unknown())  # type: ignore[arg-type]
                ib.emit(yield_stmt)
            ib.return_stmt(sum_final)
        func = f.get_result()

        assert func is not None

    def test_for_loop_iter_args_mismatch_error(self):
        """Test that mismatched iter_args and return_vars raises error."""
        ib = IRBuilder()

        # The error will be raised when exiting the for_loop context
        # Note: Error handling with context managers can be complex, so we just
        # check that RuntimeError is raised
        with pytest.raises(RuntimeError):
            with ib.function("mismatch_func") as f:
                f.return_type(ir.ScalarType(DataType.INT64))

                i = ib.var("i", ir.ScalarType(DataType.INT64))

                with ib.for_loop(i, 0, 10, 1) as loop:
                    # Add iter_arg but no return_var - should fail
                    loop.iter_arg("sum", 0)
                    # Missing loop.return_var() - will fail when exiting context


class TestIRBuilderIfStmt:
    """Test IR Builder for if statement construction."""

    def test_simple_if_stmt(self):
        """Test building a simple if statement."""
        ib = IRBuilder()

        with ib.function("if_func") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # if x > 0: result = x
            zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
            condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

            with ib.if_stmt(condition):
                result = ib.var("result", ir.ScalarType(DataType.INT64))
                ib.assign(result, x)

        func = f.get_result()

        assert func is not None
        assert isinstance(func.body, ir.IfStmt)
        assert func.body.condition is not None
        assert func.body.else_body is None

    def test_if_else_stmt(self):
        """Test building an if-else statement."""
        ib = IRBuilder()

        with ib.function("if_else_func") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())
            result = ib.var("result", ir.ScalarType(DataType.INT64))

            with ib.if_stmt(condition) as if_builder:
                # Then branch
                ib.assign(result, one)

                # Else branch
                if_builder.else_()
                ib.assign(result, zero)

        func = f.get_result()

        assert func is not None
        # When there's only one statement (the if), it becomes the body directly
        if isinstance(func.body, ir.IfStmt):
            if_stmt = func.body
        else:
            # If there are multiple statements, find the if
            assert isinstance(func.body, ir.SeqStmts)
            if_stmt = None
            for stmt in func.body.stmts:  # type: ignore[attr-defined]
                if isinstance(stmt, ir.IfStmt):
                    if_stmt = stmt
                    break
            assert if_stmt is not None

        assert if_stmt.else_body is not None


class TestIRBuilderReturnStmt:
    """Test IR Builder for return statement construction."""

    def test_simple_return_with_value(self):
        """Test building a return statement with a value."""
        ib = IRBuilder()

        with ib.function("return_func") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # return x
            ib.return_stmt(x)

        func = f.get_result()

        assert func is not None
        assert isinstance(func.body, ir.ReturnStmt)
        assert len(func.body.value) == 1

    def test_return_with_multiple_values(self):
        """Test return statement with multiple values."""
        ib = IRBuilder()

        with ib.function("multi_return_func") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            y = f.param("y", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # return x, y
            ib.return_stmt([x, y])

        func = f.get_result()

        assert func is not None
        assert isinstance(func.body, ir.ReturnStmt)
        assert len(func.body.value) == 2

    def test_return_without_value(self):
        """Test return statement without values."""
        ib = IRBuilder()

        with ib.function("void_return_func") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            # return
            ib.return_stmt()

        func = f.get_result()

        assert func is not None
        assert isinstance(func.body, ir.ReturnStmt)
        assert len(func.body.value) == 0

    def test_return_with_expression(self):
        """Test return statement with expression."""
        ib = IRBuilder()

        with ib.function("expr_return_func") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            y = f.param("y", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # return x + y
            add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
            ib.return_stmt(add_expr)

        func = f.get_result()

        assert func is not None
        assert isinstance(func.body, ir.ReturnStmt)
        assert len(func.body.value) == 1
        assert isinstance(func.body.value[0], ir.Add)

    def test_return_in_if_statement(self):
        """Test return statement inside if statement."""
        ib = IRBuilder()

        with ib.function("conditional_return") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

            with ib.if_stmt(condition) as if_builder:
                # Then branch: return 1
                ib.return_stmt(one)

                # Else branch: return 0
                if_builder.else_()
                ib.return_stmt(zero)

        func = f.get_result()

        assert func is not None
        assert isinstance(func.body, ir.IfStmt)

    def test_return_with_explicit_span(self):
        """Test return statement with explicit span."""
        ib = IRBuilder()
        my_span = ir.Span("test.py", 42, 1)

        with ib.function("span_return") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            ib.return_stmt(x, span=my_span)

        func = f.get_result()

        assert func is not None
        assert isinstance(func.body, ir.ReturnStmt)
        assert func.body.span.filename == "test.py"
        assert func.body.span.begin_line == 42


class TestIRBuilderContextQueries:
    """Test IR Builder context state queries."""

    def test_in_function_query(self):
        """Test InFunction query."""
        ib = IRBuilder()

        assert not ib.in_function()

        with ib.function("test") as f:
            assert ib.in_function()
            f.return_type(ir.ScalarType(DataType.INT64))

        assert not ib.in_function()

    def test_in_loop_query(self):
        """Test InLoop query."""
        ib = IRBuilder()

        with ib.function("test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            assert not ib.in_loop()

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1):
                assert ib.in_loop()

            assert not ib.in_loop()

    def test_in_if_query(self):
        """Test InIf query."""
        ib = IRBuilder()

        with ib.function("test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            assert not ib.in_if()

            with ib.if_stmt(1):
                assert ib.in_if()

            assert not ib.in_if()


class TestIRBuilderLet:
    """Test IR Builder let() method with type inference."""

    def test_let_with_inferred_type(self):
        """Test basic let() usage with type inference from expression."""
        ib = IRBuilder()

        with ib.function("let_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            # Create an expression with known type
            const = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

            # let() should infer the type from the expression
            x = ib.let("x", const)

            assert x.name == "x"
            assert isinstance(x.type, ir.ScalarType)
            assert x.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_let_with_type_validation(self):
        """Test let() with explicit type that matches inferred type."""
        ib = IRBuilder()

        with ib.function("validation_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            # Create an expression
            const = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

            # Provide matching type for validation
            explicit_type = ir.ScalarType(DataType.INT64)
            x = ib.let("x", const, type=explicit_type)

            assert x.name == "x"
            assert isinstance(x.type, ir.ScalarType)
            assert x.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_let_with_type_mismatch(self):
        """Test that let() raises error when explicit type doesn't match inferred type."""
        ib = IRBuilder()

        with pytest.raises(ValueError, match="Type mismatch"):
            with ib.function("mismatch_test") as f:
                f.return_type(ir.ScalarType(DataType.INT64))

                # Create INT64 expression but provide FP32 type
                const = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
                wrong_type = ir.ScalarType(DataType.FP32)

                # This should raise ValueError
                ib.let("x", const, type=wrong_type)

    def test_let_with_scalar_value(self):
        """Test let() with int/float values that get normalized."""
        ib = IRBuilder()

        with ib.function("scalar_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            # let() should handle int values via _normalize_expr
            x = ib.let("x", 42)

            assert x.name == "x"
            # Type should be inferred from the normalized expression
            assert isinstance(x.type, ir.ScalarType)

        func = f.get_result()
        assert func is not None

    def test_let_with_tensor_expr(self):
        """Test let() with tensor operation result."""
        ib = IRBuilder()

        with ib.function("tensor_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            # Create a tensor operation
            tensor_create = ir.op.tensor.create([4, 8], DataType.FP32)

            # let() should infer TensorType from the create operation
            t = ib.let("t", tensor_create)

            assert t.name == "t"
            assert isinstance(t.type, ir.TensorType)
            assert t.type.dtype == DataType.FP32

        func = f.get_result()
        assert func is not None

    def test_let_with_binary_expr(self):
        """Test let() with binary expression result."""
        ib = IRBuilder()

        with ib.function("binary_test") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            y = f.param("y", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            # Create binary expression
            add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())

            # let() should infer type from Add expression
            result = ib.let("result", add_expr)

            assert result.name == "result"
            assert isinstance(result.type, ir.ScalarType)
            assert result.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_let_with_explicit_span(self):
        """Test let() with explicit span parameter."""
        ib = IRBuilder()
        my_span = ir.Span("test.py", 100, 5)

        with ib.function("span_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            const = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
            x = ib.let("x", const, span=my_span)

            assert x.name == "x"
            assert x.span.filename == "test.py"
            assert x.span.begin_line == 100

        func = f.get_result()
        assert func is not None


class TestIRBuilderIterArgAndReturnVar:
    """Test iter_arg and return_var with type inference."""

    def test_iter_arg_with_inferred_type(self):
        """Test iter_arg with type inference from init_value."""
        ib = IRBuilder()

        with ib.function("iter_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1) as loop:
                # Type should be inferred from initial value
                sum_iter = loop.iter_arg("sum", 0)
                # Must have matching return_var
                _ = loop.return_var("sum_final")

                assert sum_iter.name == "sum"
                assert isinstance(sum_iter.type, ir.ScalarType)
                assert sum_iter.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_iter_arg_with_type_validation(self):
        """Test iter_arg with explicit type that matches inferred type."""
        ib = IRBuilder()

        with ib.function("iter_validation_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1) as loop:
                # Provide matching type for validation
                explicit_type = ir.ScalarType(DataType.INT64)
                sum_iter = loop.iter_arg("sum", 0, type=explicit_type)
                # Must have matching return_var
                _ = loop.return_var("sum_final")

                assert sum_iter.name == "sum"
                assert isinstance(sum_iter.type, ir.ScalarType)
                assert sum_iter.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_iter_arg_with_type_mismatch(self):
        """Test that iter_arg raises error when explicit type doesn't match inferred type."""
        ib = IRBuilder()

        with pytest.raises(ValueError, match="Type mismatch"):
            with ib.function("iter_mismatch_test") as f:
                f.return_type(ir.ScalarType(DataType.INT64))

                i = ib.var("i", ir.ScalarType(DataType.INT64))

                with ib.for_loop(i, 0, 10, 1) as loop:
                    # Wrong type - init_value is INT64 but we provide FP32
                    wrong_type = ir.ScalarType(DataType.FP32)
                    loop.iter_arg("sum", 0, type=wrong_type)

    def test_return_var_with_inferred_type(self):
        """Test return_var with type inference from corresponding iter_arg."""
        ib = IRBuilder()

        with ib.function("return_var_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1) as loop:
                _ = loop.iter_arg("sum", 0)
                # Type should be inferred from corresponding iter_arg
                sum_final = loop.return_var("sum_final")

                assert sum_final.name == "sum_final"
                assert isinstance(sum_final.type, ir.ScalarType)
                assert sum_final.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_return_var_with_multiple_iter_args(self):
        """Test return_var inference with multiple iter_args."""
        ib = IRBuilder()

        with ib.function("multi_return_var_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1) as loop:
                # Multiple iter_args with different types
                _ = loop.iter_arg("sum", 0)  # INT64
                _ = loop.iter_arg("count", 1)  # INT64

                # Return vars should match iter_args by index
                sum_final = loop.return_var("sum_final")  # Should be INT64
                count_final = loop.return_var("count_final")  # Should be INT64

                assert isinstance(sum_final.type, ir.ScalarType)
                assert isinstance(count_final.type, ir.ScalarType)
                assert sum_final.type.dtype == DataType.INT64
                assert count_final.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_return_var_explicit_type_validation(self):
        """Test return_var with explicit type that matches inferred type."""
        ib = IRBuilder()

        with ib.function("return_var_validation_test") as f:
            f.return_type(ir.ScalarType(DataType.INT64))

            i = ib.var("i", ir.ScalarType(DataType.INT64))

            with ib.for_loop(i, 0, 10, 1) as loop:
                _ = loop.iter_arg("sum", 0)
                # Provide explicit type that matches iter_arg type
                explicit_type = ir.ScalarType(DataType.INT64)
                sum_final = loop.return_var("sum_final", type=explicit_type)

                assert isinstance(sum_final.type, ir.ScalarType)
                assert sum_final.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None


class TestIRBuilderIfReturnVar:
    """Test if statement return_var - type must be provided explicitly."""

    def test_if_return_var_with_explicit_type(self):
        """Test if return_var requires explicit type."""
        ib = IRBuilder()

        with ib.function("if_return_test") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

            with ib.if_stmt(condition) as if_builder:
                # Type must be provided explicitly
                if_builder.return_var("result", ir.ScalarType(DataType.INT64))

                # Then branch: yield 1
                ib.emit(ir.YieldStmt([one], ir.Span.unknown()))

                # Else branch: yield 0
                if_builder.else_()
                ib.emit(ir.YieldStmt([zero], ir.Span.unknown()))

        func = f.get_result()
        assert func is not None
        # Verify the if statement has return_vars
        assert isinstance(func.body, ir.IfStmt)
        assert len(func.body.return_vars) == 1

    def test_if_return_var_with_multiple_returns(self):
        """Test if return_var with multiple return variables."""
        ib = IRBuilder()

        with ib.function("multi_if_return_test") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            two = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
            condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

            with ib.if_stmt(condition) as if_builder:
                # Both return vars need explicit types
                if_builder.return_var("result1", ir.ScalarType(DataType.INT64))
                if_builder.return_var("result2", ir.ScalarType(DataType.INT64))

                # Then branch: yield two values
                ib.emit(ir.YieldStmt([one, two], ir.Span.unknown()))

                # Else branch: yield two values
                if_builder.else_()
                ib.emit(ir.YieldStmt([zero, zero], ir.Span.unknown()))

        func = f.get_result()
        assert func is not None
        # Verify the if statement has 2 return_vars
        assert isinstance(func.body, ir.IfStmt)
        assert len(func.body.return_vars) == 2


class TestIRBuilderIfOutput:
    """Test if statement output() and outputs() methods."""

    def test_if_output_single_return_var(self):
        """Test output() with single return variable."""
        ib = IRBuilder()

        with ib.function("if_output_test") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

            with ib.if_stmt(condition) as if_builder:
                if_builder.return_var("result", ir.ScalarType(DataType.INT64))

                ib.emit(ir.YieldStmt([one], ir.Span.unknown()))
                if_builder.else_()
                ib.emit(ir.YieldStmt([zero], ir.Span.unknown()))

            # Get the output return variable
            result = if_builder.output()

            assert result.name == "result"
            assert isinstance(result.type, ir.ScalarType)
            assert result.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_if_output_multiple_return_vars(self):
        """Test output() with multiple return variables."""
        ib = IRBuilder()

        with ib.function("multi_output_test") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            two = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
            condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

            with ib.if_stmt(condition) as if_builder:
                if_builder.return_var("result1", ir.ScalarType(DataType.INT64))
                if_builder.return_var("result2", ir.ScalarType(DataType.INT64))

                ib.emit(ir.YieldStmt([one, two], ir.Span.unknown()))
                if_builder.else_()
                ib.emit(ir.YieldStmt([zero, zero], ir.Span.unknown()))

            # Get individual outputs
            result1 = if_builder.output(0)
            result2 = if_builder.output(1)

            assert result1.name == "result1"
            assert result2.name == "result2"
            assert isinstance(result1.type, ir.ScalarType)
            assert isinstance(result2.type, ir.ScalarType)
            assert result1.type.dtype == DataType.INT64
            assert result2.type.dtype == DataType.INT64

        func = f.get_result()
        assert func is not None

    def test_if_outputs_method(self):
        """Test outputs() method to get all return variables at once."""
        ib = IRBuilder()

        with ib.function("outputs_test") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            zero = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            two = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
            condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())

            with ib.if_stmt(condition) as if_builder:
                if_builder.return_var("result1", ir.ScalarType(DataType.INT64))
                if_builder.return_var("result2", ir.ScalarType(DataType.INT64))

                ib.emit(ir.YieldStmt([one, two], ir.Span.unknown()))
                if_builder.else_()
                ib.emit(ir.YieldStmt([zero, zero], ir.Span.unknown()))

            # Get all outputs at once
            results = if_builder.outputs()

            assert len(results) == 2
            assert results[0].name == "result1"
            assert results[1].name == "result2"

        func = f.get_result()
        assert func is not None


class TestIRBuilderSerialization:
    """Test that builder output can be serialized."""

    def test_serialize_builder_output(self):
        """Test serializing and deserializing builder output."""
        ib = IRBuilder()

        with ib.function("serialize_test") as f:
            x = f.param("x", ir.ScalarType(DataType.INT64))
            f.return_type(ir.ScalarType(DataType.INT64))

            result = ib.var("result", ir.ScalarType(DataType.INT64))
            one = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
            add_expr = ir.Add(x, one, DataType.INT64, ir.Span.unknown())
            ib.assign(result, add_expr)

        func = f.get_result()
        assert func is not None

        # Serialize
        data = ir.serialize(func)
        assert data is not None
        assert len(data) > 0

        # Deserialize
        restored = ir.deserialize(data)
        assert restored is not None
        assert isinstance(restored, ir.Function)

        # Check structure is preserved
        assert ir.structural_equal(func, restored, enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
