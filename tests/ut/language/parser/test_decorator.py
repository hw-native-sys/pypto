# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for @pl.function, @pl.inline, and @pl.program decorators."""

import linecache
import sys
import textwrap

import pypto
import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.language.parser.diagnostics.exceptions import (
    ParserSyntaxError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)


class TestFunctionDecorator:
    """Tests for @pl.function decorator."""

    def test_simple_function(self):
        """Test parsing simple function with no control flow."""

        @pl.function
        def add_tensors(
            x: pl.Tensor[[64, 128], pl.FP16],
            y: pl.Tensor[[64, 128], pl.FP16],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, y)
            return result

        assert isinstance(add_tensors, ir.Function)
        assert add_tensors.name == "add_tensors"
        assert len(add_tensors.params) == 2
        assert len(add_tensors.return_types) == 1

    def test_function_with_multiple_statements(self):
        """Test function with multiple statements."""

        @pl.function
        def multi_op(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            a: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            b: pl.Tensor[[64], pl.FP32] = pl.add(a, 1.0)
            c: pl.Tensor[[64], pl.FP32] = pl.sub(b, 0.5)
            return c

        assert isinstance(multi_op, ir.Function)
        assert multi_op.name == "multi_op"

    def test_function_with_multiple_params(self):
        """Test function with multiple parameters."""

        @pl.function
        def three_param(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
            z: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            temp: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
            result: pl.Tensor[[64], pl.FP32] = pl.add(temp, z)
            return result

        assert len(three_param.params) == 3

    def test_function_with_tensor_create(self):
        """Test function that creates tensors."""

        @pl.function
        def create_tensor(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            return result

        assert isinstance(create_tensor, ir.Function)

    def test_function_with_binary_ops(self):
        """Test function with binary operations."""

        @pl.function
        def binary_ops(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # Using operator overloading
            result: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), pl.create_tensor([64], dtype=pl.FP32))
            return result

        assert isinstance(binary_ops, ir.Function)

    def test_function_with_list_arguments(self):
        """Test function that uses list arguments."""

        @pl.function
        def with_lists(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[32, 64], pl.FP32]:
            # view takes list arguments
            result: pl.Tensor[[32, 64], pl.FP32] = pl.view(x, [32, 64], [0, 0])
            return result

        assert isinstance(with_lists, ir.Function)

    def test_function_with_eval_stmt(self):
        """Test parsing evaluation statements into EvalStmt."""

        @pl.function
        def with_eval_stmt(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # Standalone evaluation statements should become EvalStmt
            pl.create_tensor([32], dtype=pl.FP32)
            pl.create_tensor([64], dtype=pl.FP32)

            # Regular assignment
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        body = with_eval_stmt.body
        assert isinstance(body, ir.SeqStmts)
        assert len(body.stmts) == 4  # 2 EvalStmts + AssignStmt + ReturnStmt
        assert isinstance(body.stmts[0], ir.EvalStmt)
        assert isinstance(body.stmts[1], ir.EvalStmt)

    def test_function_serialization(self):
        """Test that parsed functions can be serialized."""

        @pl.function
        def simple(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        # Should be able to serialize
        data = pypto.ir.serialize(simple)
        assert len(data) > 0

        # Should be able to deserialize
        restored = pypto.ir.deserialize(data)
        assert isinstance(restored, ir.Function)
        assert restored.name == "simple"

    def test_function_with_different_dtypes(self):
        """Test function with various data types."""

        @pl.function
        def dtypes(
            fp16: pl.Tensor[[64], pl.FP16],
            fp32: pl.Tensor[[64], pl.FP32],
            int32: pl.Tensor[[64], pl.INT32],
        ) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(pl.cast(fp16, target_type=pl.FP32), fp32)
            return result

        assert len(dtypes.params) == 3

    def test_invalid_function_no_annotations(self):
        """Test that function without annotations raises error."""

        with pytest.raises(ParserTypeError, match="missing type annotation"):

            @pl.function
            def no_annotations(x):
                return x

    def test_function_preserves_name(self):
        """Test that function name is preserved."""

        @pl.function
        def my_custom_function_name(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert my_custom_function_name.name == "my_custom_function_name"

    def test_function_with_negative_numbers(self):
        """Test function with negative number literals."""

        @pl.function
        def with_negatives(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, -1.5)
            return result

        assert isinstance(with_negatives, ir.Function)


class TestScalarParameters:
    """Tests for Scalar parameter support in @pl.function."""

    def test_function_with_scalar_param(self):
        """Test function with scalar parameter - subscript notation."""

        @pl.function
        def add_scalar(
            x: pl.Tensor[[64], pl.FP32],
            scalar: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, scalar)
            return result

        assert isinstance(add_scalar, ir.Function)
        assert add_scalar.name == "add_scalar"
        assert len(add_scalar.params) == 2

        # Check that second parameter is ScalarType
        scalar_param = add_scalar.params[1]
        assert isinstance(scalar_param.type, ir.ScalarType)
        assert scalar_param.type.dtype == pl.FP32

    def test_function_with_multiple_scalar_params(self):
        """Test function with multiple scalar parameters."""

        @pl.function
        def scale_and_offset(
            x: pl.Tensor[[64], pl.FP32],
            scale: pl.Scalar[pl.FP32],
            offset: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            scaled: pl.Tensor[[64], pl.FP32] = pl.mul(x, scale)
            result: pl.Tensor[[64], pl.FP32] = pl.add(scaled, offset)
            return result

        assert len(scale_and_offset.params) == 3
        assert isinstance(scale_and_offset.params[1].type, ir.ScalarType)
        assert isinstance(scale_and_offset.params[2].type, ir.ScalarType)

    def test_function_with_different_scalar_types(self):
        """Test function with scalars of different types."""

        @pl.function
        def mixed_scalars(
            fp_scalar: pl.Scalar[pl.FP32],
            int_scalar: pl.Scalar[pl.INT32],
        ) -> pl.Scalar[pl.FP32]:
            return fp_scalar

        assert isinstance(mixed_scalars.params[0].type, ir.ScalarType)
        assert mixed_scalars.params[0].type.dtype == pl.FP32
        assert isinstance(mixed_scalars.params[1].type, ir.ScalarType)
        assert mixed_scalars.params[1].type.dtype == pl.INT32

    def test_function_returning_scalar(self):
        """Test function that returns a scalar."""

        @pl.function
        def return_scalar(x: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            return x

        assert isinstance(return_scalar, ir.Function)
        assert len(return_scalar.return_types) == 1
        assert isinstance(return_scalar.return_types[0], ir.ScalarType)

    def test_scalar_legacy_call_notation(self):
        """Test legacy pl.Scalar(dtype) notation (annotation uses Scalar[dtype])."""

        @pl.function
        def legacy_scalar(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            return x

        assert isinstance(legacy_scalar.params[0].type, ir.ScalarType)
        assert legacy_scalar.params[0].type.dtype == pl.FP32
        # Runtime: legacy pl.Scalar(dtype) still creates valid annotation-only instance
        assert pl.Scalar(pl.FP32).dtype == pl.FP32

    def test_block_ops_with_scalar(self):
        """Test block operations with scalar parameter."""

        @pl.function(type=pl.FunctionType.InCore)
        def block_add_scalar(
            input_tile: pl.Tensor[[64, 64], pl.FP32],
            scalar: pl.Scalar[pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile: pl.Tile[[64, 64], pl.FP32] = pl.load(input_tile, [0, 0], [64, 64])
            result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, scalar)
            output_new: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
            return output_new

        assert isinstance(block_add_scalar, ir.Function)
        assert block_add_scalar.func_type == pl.FunctionType.InCore
        assert isinstance(block_add_scalar.params[1].type, ir.ScalarType)


class TestTensorReadParsing:
    """Tests for tensor.read operation in the DSL."""

    def test_tensor_read_basic(self):
        """Test parsing pl.tensor.read with constant indices."""

        @pl.function
        def read_elem(t: pl.Tensor[[4, 8], pl.FP32]) -> pl.Scalar[pl.FP32]:
            val: pl.Scalar[pl.FP32] = pl.tensor.read(t, [0, 0])
            return val

        assert isinstance(read_elem, ir.Function)
        assert len(read_elem.return_types) == 1
        assert isinstance(read_elem.return_types[0], ir.ScalarType)

    def test_tensor_read_with_loop_index(self):
        """Test parsing pl.tensor.read with loop variable as index."""

        @pl.function
        def read_in_loop(t: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            for i in pl.range(64):
                _ = pl.tensor.read(t, [i])
            return out

        assert isinstance(read_in_loop, ir.Function)


class TestTupleReturnType:
    """Tests for tuple return type annotations in the DSL."""

    def test_tuple_return_two_tensors(self):
        """Test function with tuple[Tensor, Tensor] return type."""

        @pl.function
        def two_outputs(
            x: pl.Tensor[[64], pl.FP32],
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            b: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return a, b

        assert isinstance(two_outputs, ir.Function)
        assert len(two_outputs.return_types) == 2
        assert isinstance(two_outputs.return_types[0], ir.TensorType)
        assert isinstance(two_outputs.return_types[1], ir.TensorType)

    def test_tuple_return_mixed_types(self):
        """Test function with tuple[Tensor, Scalar] return type."""

        @pl.function
        def mixed_return(
            x: pl.Tensor[[64], pl.FP32],
            idx: pl.Scalar[pl.INT64],
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.INT64]]:
            a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return a, idx

        assert isinstance(mixed_return, ir.Function)
        assert len(mixed_return.return_types) == 2
        assert isinstance(mixed_return.return_types[0], ir.TensorType)
        assert isinstance(mixed_return.return_types[1], ir.ScalarType)


class TestProgramDecorator:
    """Tests for @pl.program decorator."""

    def test_single_function_program(self):
        """Test @pl.program with a single function."""

        @pl.program
        class SimpleProgram:
            @pl.function
            def add_one(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        assert isinstance(SimpleProgram, ir.Program)
        assert SimpleProgram.name == "SimpleProgram"
        assert len(SimpleProgram.functions) == 1

        # Verify the function is accessible
        add_func = SimpleProgram.get_function("add_one")
        assert add_func is not None
        assert add_func.name == "add_one"
        # self parameter should be stripped
        assert len(add_func.params) == 1
        assert add_func.params[0].name == "x"

    def test_multiple_functions_program(self):
        """Test @pl.program with multiple functions."""

        @pl.program
        class MathOps:
            @pl.function
            def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return result

            @pl.function
            def double(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                two: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)
                result: pl.Tensor[[1], pl.INT32] = pl.mul(x, two)
                return result

        assert isinstance(MathOps, ir.Program)
        assert MathOps.name == "MathOps"
        assert len(MathOps.functions) == 2

        # Verify both functions exist
        square_func = MathOps.get_function("square")
        double_func = MathOps.get_function("double")
        assert square_func is not None
        assert double_func is not None

    def test_cross_function_calls(self):
        """Test cross-function calls using self.method() syntax."""

        @pl.program
        class CallTest:
            @pl.function
            def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return result

            @pl.function
            def sum_of_squares(
                self, a: pl.Tensor[[1], pl.INT32], b: pl.Tensor[[1], pl.INT32]
            ) -> pl.Tensor[[1], pl.INT32]:
                # Call square method using self
                a_squared: pl.Tensor[[1], pl.INT32] = self.square(a)
                b_squared: pl.Tensor[[1], pl.INT32] = self.square(b)
                result: pl.Tensor[[1], pl.INT32] = pl.add(a_squared, b_squared)
                return result

        assert isinstance(CallTest, ir.Program)
        assert len(CallTest.functions) == 2

        # Verify sum_of_squares function exists and has proper parameters
        sum_func = CallTest.get_function("sum_of_squares")
        assert sum_func is not None
        # Should have 2 params (a, b) - self is stripped
        assert len(sum_func.params) == 2

    def test_forward_reference(self):
        """Test calling a function defined later in the class."""

        @pl.program
        class ForwardRef:
            @pl.function
            def caller(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                # Call helper which is defined below
                result: pl.Tensor[[1], pl.INT32] = self.helper(x)
                return result

            @pl.function
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = pl.mul(x, 2)
                return result

        assert isinstance(ForwardRef, ir.Program)
        assert len(ForwardRef.functions) == 2

    def test_recursive_call(self):
        """Test function calling itself recursively via self.method_name()."""

        @pl.program
        class RecursiveTest:
            @pl.function
            def factorial(self, n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                _zero: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)
                one: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)
                # Note: This is just for testing IR structure, not a real factorial implementation
                # In real DSL, we'd need if statements for base case
                result: pl.Tensor[[1], pl.INT32] = pl.add(n, one)
                return result

        assert isinstance(RecursiveTest, ir.Program)

    def test_transitive_calls(self):
        """Test transitive calls where A calls B calls C."""

        @pl.program
        class TransitiveCalls:
            @pl.function
            def a(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = self.b(x)
                return result

            @pl.function
            def b(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = self.c(x)
                return result

            @pl.function
            def c(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = pl.mul(x, 3)
                return result

        assert isinstance(TransitiveCalls, ir.Program)
        assert len(TransitiveCalls.functions) == 3

    def test_self_parameter_stripped(self):
        """Test that self parameter is properly stripped from IR."""

        @pl.program
        class SelfTest:
            @pl.function
            def test_func(
                self, x: pl.Tensor[[1], pl.INT32], y: pl.Tensor[[1], pl.INT32]
            ) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = pl.add(x, y)
                return result

        func = SelfTest.get_function("test_func")
        assert func is not None
        # Should only have x and y parameters (self stripped)
        assert len(func.params) == 2
        assert func.params[0].name == "x"
        assert func.params[1].name == "y"

    def test_program_name_from_class(self):
        """Test that program name is extracted from class name."""

        @pl.program
        class MyCustomProgram:
            @pl.function
            def dummy(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                return x

        assert MyCustomProgram.name == "MyCustomProgram"

    def test_empty_class_error(self):
        """Test that empty class raises error."""
        with pytest.raises(ParserSyntaxError):  # Should raise ParserSyntaxError

            @pl.program
            class EmptyProgram:
                pass

    def test_undefined_method_call_error(self):
        """Test that calling undefined method raises error."""
        with pytest.raises(UndefinedVariableError):  # Should raise UndefinedVariableError

            @pl.program
            class UndefinedCall:
                @pl.function
                def caller(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                    # Try to call a method that doesn't exist
                    result: pl.Tensor[[1], pl.INT32] = self.nonexistent(x)  # type: ignore
                    return result

    def test_tuple_unpacking_from_cross_function_call(self):
        """Test tuple unpacking from self.func() returning multiple values."""

        @pl.program
        class TupleUnpack:
            @pl.function
            def split(
                self, x: pl.Tensor[[64], pl.FP32]
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                return a, b

            @pl.function
            def caller(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a, b = self.split(x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

        assert isinstance(TupleUnpack, ir.Program)
        assert len(TupleUnpack.functions) == 2

        caller_func = TupleUnpack.get_function("caller")
        assert caller_func is not None


class TestProgramRoundTrip:
    """Test round-trip: parse -> print -> parse."""

    def test_roundtrip_simple_program(self):
        """Test that printing and re-parsing produces equivalent IR."""

        @pl.program
        class Original:
            @pl.function
            def add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        # Print to code
        code = pypto.ir.python_print(Original)

        # Verify code contains expected elements
        assert "@pl.program" in code
        assert "class Original:" in code
        assert "def add(self," in code  # Should have self parameter

        # Re-parse the code
        reparsed = pl.parse_program(code)

        # Verify structural equivalence
        assert isinstance(reparsed, ir.Program)
        assert reparsed.name == "Original"
        assert len(reparsed.functions) == 1

        # Verify function structure matches
        reparsed_func = reparsed.get_function("add")
        original_func = Original.get_function("add")
        assert reparsed_func is not None
        assert original_func is not None
        assert len(reparsed_func.params) == len(original_func.params)

        # Verify structural equivalence
        pypto.ir.assert_structural_equal(reparsed, Original)

    def test_roundtrip_with_cross_function_calls(self):
        """Test round-trip with cross-function calls."""

        @pl.program
        class WithCalls:
            @pl.function
            def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = pl.mul(x, 2)
                return result

            @pl.function
            def caller(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = self.helper(x)
                return result

        # Print to code
        code = pypto.ir.python_print(WithCalls)

        # Verify cross-function calls are printed with self
        assert "self.helper(" in code

        # Re-parse
        reparsed = pl.parse_program(code)

        assert isinstance(reparsed, ir.Program)
        assert len(reparsed.functions) == 2

        # Verify structural equivalence
        ir.assert_structural_equal(reparsed, WithCalls)


class TestFunctionDecoratorSourceUnavailable:
    """Tests for @pl.function when inspect.getsourcelines() fails."""

    def test_function_with_linecache_source(self):
        """Test that @pl.function works via linecache when inspect fails (e.g., exec)."""
        code = textwrap.dedent("""\
            import pypto.language as pl

            @pl.function
            def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result
        """)
        filename = "<test_linecache_function>"
        code_lines = code.splitlines(keepends=True)
        # Pre-populate linecache so the fallback strategy can find the source
        linecache.cache[filename] = (len(code), None, code_lines, filename)
        try:
            compiled = compile(code, filename, "exec")
            namespace: dict = {}
            exec(compiled, namespace)  # noqa: S102
            result = namespace["add_one"]
            assert isinstance(result, ir.Function)
            assert result.name == "add_one"
            assert len(result.params) == 1
        finally:
            linecache.cache.pop(filename, None)

    def test_function_with_orig_argv_source(self, monkeypatch):
        """Test that @pl.function works via sys.orig_argv for python -c scenarios."""
        code = textwrap.dedent("""\
            import pypto.language as pl

            @pl.function
            def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result
        """)
        # Simulate python -c by using <string> filename and setting sys.orig_argv
        monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-c", code])
        filename = "<string>"
        compiled = compile(code, filename, "exec")
        namespace: dict = {}
        exec(compiled, namespace)  # noqa: S102
        result = namespace["add_one"]
        assert isinstance(result, ir.Function)
        assert result.name == "add_one"
        assert len(result.params) == 1

    def test_function_without_source_gives_clear_error(self):
        """Test that @pl.function gives a clear ParserSyntaxError when no source is available."""
        code = textwrap.dedent("""\
            import pypto.language as pl
            from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError

            try:
                @pl.function
                def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                    return result
                assert False, "Should have raised ParserSyntaxError"
            except ParserSyntaxError as e:
                assert "Cannot retrieve source code" in str(e)
                assert "pl.parse()" in e.hint
        """)
        # Use a filename that won't be in linecache or on disk
        filename = "<no_source_available>"
        compiled = compile(code, filename, "exec")
        namespace: dict = {}
        exec(compiled, namespace)  # noqa: S102


class TestProgramDecoratorSourceUnavailable:
    """Tests for @pl.program when inspect.getsourcelines() fails."""

    def test_program_with_linecache_source(self):
        """Test that @pl.program works via linecache when inspect fails (e.g., exec)."""
        code = textwrap.dedent("""\
            import pypto.language as pl

            @pl.program
            class MyProgram:
                @pl.function
                def add_one(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                    return result
        """)
        filename = "<test_linecache_program>"
        code_lines = code.splitlines(keepends=True)
        # Pre-populate linecache so the fallback strategy can find the source
        linecache.cache[filename] = (len(code), None, code_lines, filename)
        try:
            compiled = compile(code, filename, "exec")
            namespace: dict = {}
            exec(compiled, namespace)  # noqa: S102
            result = namespace["MyProgram"]
            assert isinstance(result, ir.Program)
            assert result.name == "MyProgram"
            assert len(result.functions) == 1
        finally:
            linecache.cache.pop(filename, None)

    def test_program_with_orig_argv_source(self, monkeypatch):
        """Test that @pl.program works via sys.orig_argv for python -c scenarios."""
        code = textwrap.dedent("""\
            import pypto.language as pl

            @pl.program
            class MyProgram:
                @pl.function
                def add_one(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                    return result
        """)
        monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-c", code])
        filename = "<string>"
        compiled = compile(code, filename, "exec")
        namespace: dict = {}
        exec(compiled, namespace)  # noqa: S102
        result = namespace["MyProgram"]
        assert isinstance(result, ir.Program)
        assert result.name == "MyProgram"
        assert len(result.functions) == 1

    def test_program_without_source_gives_clear_error(self):
        """Test that @pl.program gives a clear ParserSyntaxError when no source is available."""
        code = textwrap.dedent("""\
            import pypto.language as pl
            from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError

            try:
                @pl.program
                class MyProgram:
                    @pl.function
                    def add_one(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                        result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                        return result
                assert False, "Should have raised ParserSyntaxError"
            except ParserSyntaxError as e:
                assert "Cannot retrieve source code" in str(e)
                assert "pl.parse()" in e.hint
        """)
        # Use a filename that won't be in linecache or on disk
        filename = "<no_source_available_program>"
        compiled = compile(code, filename, "exec")
        namespace: dict = {}
        exec(compiled, namespace)  # noqa: S102


class TestExternalFunctionCalls:
    """Tests for calling externally-defined @pl.function from within @pl.program."""

    def test_basic_external_call(self):
        """External @pl.function is callable and added to Program."""

        @pl.function
        def double(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        @pl.program
        class MyModel:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = double(x)
                return result

        @pl.program
        class Expected:
            @pl.function
            def double(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.double(x)
                return result

        ir.assert_structural_equal(MyModel, Expected)

    def test_external_return_type_propagation(self):
        """Return type from external function propagates to caller's variable."""

        @pl.function
        def ext_square(x: pl.Tensor[[32], pl.INT32]) -> pl.Tensor[[32], pl.INT32]:
            result: pl.Tensor[[32], pl.INT32] = pl.mul(x, x)
            return result

        @pl.program
        class TypeProp:
            @pl.function
            def main(self, x: pl.Tensor[[32], pl.INT32]) -> pl.Tensor[[32], pl.INT32]:
                y: pl.Tensor[[32], pl.INT32] = ext_square(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def ext_square(self, x: pl.Tensor[[32], pl.INT32]) -> pl.Tensor[[32], pl.INT32]:
                result: pl.Tensor[[32], pl.INT32] = pl.mul(x, x)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[32], pl.INT32]) -> pl.Tensor[[32], pl.INT32]:
                y: pl.Tensor[[32], pl.INT32] = self.ext_square(x)
                return y

        ir.assert_structural_equal(TypeProp, Expected)

    def test_multiple_external_functions(self):
        """Multiple external functions in one program."""

        @pl.function
        def ext_add(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.function
        def ext_mul(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        @pl.program
        class MultiExt:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = ext_add(x)
                z: pl.Tensor[[64], pl.FP32] = ext_mul(y)
                return z

        @pl.program
        class Expected:
            @pl.function
            def ext_add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

            @pl.function
            def ext_mul(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.ext_add(x)
                z: pl.Tensor[[64], pl.FP32] = self.ext_mul(y)
                return z

        ir.assert_structural_equal(MultiExt, Expected)

    def test_same_external_from_multiple_methods(self):
        """Same external called from 2 internal functions — added once to Program."""

        @pl.function
        def shared_helper(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class SharedExt:
            @pl.function
            def func_a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = shared_helper(x)
                return result

            @pl.function
            def func_b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = shared_helper(x)
                return result

        @pl.program
        class Expected:
            @pl.function
            def shared_helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

            @pl.function
            def func_a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.shared_helper(x)
                return result

            @pl.function
            def func_b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.shared_helper(x)
                return result

        ir.assert_structural_equal(SharedExt, Expected)

    def test_naming_conflict_with_internal_raises_error(self):
        """External with same name as internal @pl.function raises ParserSyntaxError."""

        @pl.function
        def conflicting(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        with pytest.raises(ParserSyntaxError, match="conflicts with program function"):

            @pl.program
            class Conflict:
                @pl.function
                def conflicting(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                    return result

                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = conflicting(x)
                    return result

    def test_two_externals_same_name_raises_error(self):
        """Two different external functions with same .name raises ParserSyntaxError."""

        @pl.function
        def helper(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        helper_v1 = helper

        @pl.function
        def helper(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:  # noqa: F811
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        helper_v2 = helper

        # Both have name "helper" but are different objects
        assert helper_v1 is not helper_v2

        with pytest.raises(ParserSyntaxError, match="Conflicting external functions"):

            @pl.program
            class ConflictExt:
                @pl.function
                def func_a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = helper_v1(x)
                    return result

                @pl.function
                def func_b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = helper_v2(x)
                    return result

    def test_external_roundtrip(self):
        """Print program with external function → parse → structural equality."""

        @pl.function
        def ext_add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class Original:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = ext_add_one(x)
                return result

        # Print and re-parse
        printed = ir.python_print(Original)
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Original, reparsed)

    def test_aliased_import_uses_original_name(self):
        """Aliased reference uses the function's original .name for the GlobalVar."""

        @pl.function
        def original_name(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        aliased = original_name  # Local alias

        @pl.program
        class AliasTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = aliased(x)
                return result

        @pl.program
        class Expected:
            @pl.function
            def original_name(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.original_name(x)
                return result

        ir.assert_structural_equal(AliasTest, Expected)

    def test_non_function_bare_call_still_errors(self):
        """Bare call to a regular Python function still raises UnsupportedFeatureError."""

        def regular_python_func(x):
            return x

        with pytest.raises(UnsupportedFeatureError, match="Unsupported function call"):

            @pl.program
            class BadCall:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = regular_python_func(x)
                    return result


class TestInlineFunctionCalls:
    """Tests for @pl.inline decorator and inline function expansion."""

    def test_basic_inline(self):
        """Inline expands statements in-place, no extra function in Program."""

        @pl.inline
        def double_it(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        @pl.program
        class InlineTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = double_it(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(InlineTest, Expected)

    def test_inline_return_value(self):
        """Inline return value used as expression in caller."""

        @pl.inline
        def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class ReturnTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = add_one(x)
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, 2.0)
                return z

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                y: pl.Tensor[[64], pl.FP32] = result
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, 2.0)
                return z

        ir.assert_structural_equal(ReturnTest, Expected)

    def test_inline_multiple_statements(self):
        """Multiple statements are all inlined into caller body."""

        @pl.inline
        def multi_step(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            b: pl.Tensor[[64], pl.FP32] = pl.mul(a, 2.0)
            return b

        @pl.program
        class MultiStmt:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = multi_step(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, 2.0)
                y: pl.Tensor[[64], pl.FP32] = b
                return y

        ir.assert_structural_equal(MultiStmt, Expected)

    def test_inline_no_extra_function_in_program(self):
        """Inline does NOT add a function to the Program — only @pl.function does."""

        @pl.inline
        def inlined_op(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class NoExtraFunc:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = inlined_op(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        # Verify no "inlined_op" function in the program
        assert len(NoExtraFunc.functions) == 1
        assert NoExtraFunc.get_function("inlined_op") is None
        ir.assert_structural_equal(NoExtraFunc, Expected)

    def test_inline_called_multiple_times(self):
        """Same inline called twice — fresh variable expansion each time."""

        @pl.inline
        def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class TwiceCalled:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = add_one(x)
                z: pl.Tensor[[64], pl.FP32] = add_one(y)
                return z

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                y: pl.Tensor[[64], pl.FP32] = result
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, 1.0)
                z: pl.Tensor[[64], pl.FP32] = result
                return z

        ir.assert_structural_equal(TwiceCalled, Expected)

    def test_inline_wrong_arg_count_raises_error(self):
        """Wrong number of arguments raises ParserTypeError."""

        @pl.inline
        def one_arg(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        with pytest.raises(ParserTypeError, match="expects 1 argument.*got 2"):

            @pl.program
            class WrongArgCount:
                @pl.function
                def main(
                    self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]
                ) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = one_arg(a, b)
                    return result

    def test_inline_with_closure_variables(self):
        """Inline function can reference closure variables from its definition site."""
        SCALE = 3.0

        @pl.inline
        def scale(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, SCALE)
            return result

        @pl.program
        class ClosureTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = scale(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 3.0)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(ClosureTest, Expected)

    def test_inline_structural_equality(self):
        """Program using inline produces same IR as manually writing the expanded code."""

        @pl.inline
        def inlined_add(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            tmp: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return tmp

        @pl.program
        class WithInline:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = inlined_add(x)
                return y

        @pl.program
        class ManualExpand:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                tmp: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                y: pl.Tensor[[64], pl.FP32] = tmp
                return y

        ir.assert_structural_equal(WithInline, ManualExpand)


class TestExternalFunctionControlFlow:
    """Tests for external @pl.function calls with control flow and SSA patterns."""

    def test_external_with_for_loop_iter_args(self):
        """External function containing a for loop with iter_args and yield."""

        @pl.function
        def accumulate(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = x
            for i, (acc,) in pl.range(5, init_values=(init,)):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                out = pl.yield_(new_acc)
            return out

        @pl.program
        class ExtLoopModel:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = accumulate(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def accumulate(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                    out = pl.yield_(new_acc)
                return out

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.accumulate(x)
                return y

        ir.assert_structural_equal(ExtLoopModel, Expected)

    def test_external_with_if_else_yield(self):
        """External function containing if/else with yield (SSA phi nodes)."""

        @pl.function
        def cond_scale(x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
            if flag == 0:
                out: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.mul(x, 2.0))
            else:
                out: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.add(x, 1.0))
            return out

        @pl.program
        class ExtIfModel:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = cond_scale(x, flag)
                return y

        @pl.program
        class Expected:
            @pl.function
            def cond_scale(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if flag == 0:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.mul(x, 2.0))
                else:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.add(x, 1.0))
                return out

            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.cond_scale(x, flag)
                return y

        ir.assert_structural_equal(ExtIfModel, Expected)

    def test_external_with_if_in_for_loop(self):
        """External function with if/else yield nested inside a for loop."""

        @pl.function
        def loop_cond(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = x
            for i, (acc,) in pl.range(5, init_values=(init,)):
                if i == 0:
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.mul(acc, 2.0))
                else:
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                out = pl.yield_(val)
            return out

        @pl.program
        class ExtNestedModel:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = loop_cond(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def loop_cond(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    if i == 0:
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.mul(acc, 2.0))
                    else:
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                    out = pl.yield_(val)
                return out

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.loop_cond(x)
                return y

        ir.assert_structural_equal(ExtNestedModel, Expected)

    def test_external_called_in_caller_for_loop(self):
        """External function called inside caller's for loop with iter_args."""

        @pl.function
        def step(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class CallerLoop:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    updated: pl.Tensor[[64], pl.FP32] = step(acc)
                    out = pl.yield_(updated)
                return out

        @pl.program
        class Expected:
            @pl.function
            def step(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    updated: pl.Tensor[[64], pl.FP32] = self.step(acc)
                    out = pl.yield_(updated)
                return out

        ir.assert_structural_equal(CallerLoop, Expected)

    def test_external_called_in_caller_if_yield(self):
        """External function called inside caller's if/else with yield."""

        @pl.function
        def double(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        @pl.program
        class CallerIf:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if flag == 0:
                    d: pl.Tensor[[64], pl.FP32] = double(x)
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(d)
                else:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(x)
                return out

        @pl.program
        class Expected:
            @pl.function
            def double(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                return result

            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if flag == 0:
                    d: pl.Tensor[[64], pl.FP32] = self.double(x)
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(d)
                else:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(x)
                return out

        ir.assert_structural_equal(CallerIf, Expected)

    def test_external_in_for_with_if_yield(self):
        """External function called inside if/else yield inside caller's for loop."""

        @pl.function
        def bump(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class ComplexCaller:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    if i == 0:
                        stepped: pl.Tensor[[64], pl.FP32] = bump(acc)
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(stepped)
                    else:
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                    out = pl.yield_(val)
                return out

        @pl.program
        class Expected:
            @pl.function
            def bump(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    if i == 0:
                        stepped: pl.Tensor[[64], pl.FP32] = self.bump(acc)
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(stepped)
                    else:
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                    out = pl.yield_(val)
                return out

        ir.assert_structural_equal(ComplexCaller, Expected)

    def test_external_with_multiple_iter_args(self):
        """External function with for loop using multiple iter_args and yield."""

        @pl.function
        def dual_accumulate(
            x: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            init_a: pl.Tensor[[64], pl.FP32] = x
            init_b: pl.Tensor[[64], pl.FP32] = x
            for i, (a, b) in pl.range(5, init_values=(init_a, init_b)):  # type: ignore
                new_a: pl.Tensor[[64], pl.FP32] = pl.add(a, 1.0)
                new_b: pl.Tensor[[64], pl.FP32] = pl.mul(b, 2.0)
                out_a, out_b = pl.yield_(new_a, new_b)
            result: pl.Tensor[[64], pl.FP32] = pl.add(out_a, out_b)
            return result

        @pl.program
        class ExtMultiIter:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = dual_accumulate(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def dual_accumulate(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_a: pl.Tensor[[64], pl.FP32] = x
                init_b: pl.Tensor[[64], pl.FP32] = x
                for i, (a, b) in pl.range(5, init_values=(init_a, init_b)):  # type: ignore
                    new_a: pl.Tensor[[64], pl.FP32] = pl.add(a, 1.0)
                    new_b: pl.Tensor[[64], pl.FP32] = pl.mul(b, 2.0)
                    out_a, out_b = pl.yield_(new_a, new_b)
                result: pl.Tensor[[64], pl.FP32] = pl.add(out_a, out_b)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.dual_accumulate(x)
                return y

        ir.assert_structural_equal(ExtMultiIter, Expected)


class TestInlineFunctionControlFlow:
    """Tests for @pl.inline with control flow and SSA patterns."""

    def test_inline_with_for_loop_iter_args(self):
        """Inline function containing a for loop with iter_args — expanded into caller."""

        @pl.inline
        def accumulate(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = x
            for i, (acc,) in pl.range(5, init_values=(init,)):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                out = pl.yield_(new_acc)
            return out

        @pl.program
        class InlineLoopModel:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = accumulate(x)
                return y

        # Inline expansion: for loop is emitted directly in caller body
        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                    out = pl.yield_(new_acc)
                y: pl.Tensor[[64], pl.FP32] = out
                return y

        assert len(InlineLoopModel.functions) == 1  # No extra function
        ir.assert_structural_equal(InlineLoopModel, Expected)

    def test_inline_with_if_else_yield(self):
        """Inline function containing if/else with yield (SSA phi nodes) — expanded into caller."""

        @pl.inline
        def cond_scale(x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
            if flag == 0:
                out: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.mul(x, 2.0))
            else:
                out: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.add(x, 1.0))
            return out

        @pl.program
        class InlineIfModel:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = cond_scale(x, flag)
                return y

        # Inline expansion: if/else with yield is emitted directly in caller body
        @pl.program
        class Expected:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if flag == 0:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.mul(x, 2.0))
                else:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.add(x, 1.0))
                y: pl.Tensor[[64], pl.FP32] = out
                return y

        assert len(InlineIfModel.functions) == 1
        ir.assert_structural_equal(InlineIfModel, Expected)

    def test_inline_with_if_in_for_loop(self):
        """Inline function with if/else yield nested inside a for loop — expanded into caller."""

        @pl.inline
        def loop_cond(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = x
            for i, (acc,) in pl.range(5, init_values=(init,)):
                if i == 0:
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.mul(acc, 2.0))
                else:
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                out = pl.yield_(val)
            return out

        @pl.program
        class InlineNestedModel:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = loop_cond(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    if i == 0:
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(pl.mul(acc, 2.0))
                    else:
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                    out = pl.yield_(val)
                y: pl.Tensor[[64], pl.FP32] = out
                return y

        assert len(InlineNestedModel.functions) == 1
        ir.assert_structural_equal(InlineNestedModel, Expected)

    def test_inline_called_in_caller_for_loop(self):
        """Inline function called inside caller's for loop with iter_args."""

        @pl.inline
        def step(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class CallerLoop:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    updated: pl.Tensor[[64], pl.FP32] = step(acc)
                    out = pl.yield_(updated)
                return out

        # Inline expansion happens inside the for loop body
        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    result: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                    updated: pl.Tensor[[64], pl.FP32] = result
                    out = pl.yield_(updated)
                return out

        assert len(CallerLoop.functions) == 1
        ir.assert_structural_equal(CallerLoop, Expected)

    def test_inline_called_in_caller_if_yield(self):
        """Inline function called inside caller's if/else with yield."""

        @pl.inline
        def double(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        @pl.program
        class CallerIf:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if flag == 0:
                    d: pl.Tensor[[64], pl.FP32] = double(x)
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(d)
                else:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(x)
                return out

        # Inline expansion happens inside the if-then branch
        @pl.program
        class Expected:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if flag == 0:
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                    d: pl.Tensor[[64], pl.FP32] = result
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(d)
                else:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(x)
                return out

        assert len(CallerIf.functions) == 1
        ir.assert_structural_equal(CallerIf, Expected)

    def test_inline_in_for_with_if_yield(self):
        """Inline called inside if/else yield inside caller's for loop."""

        @pl.inline
        def bump(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class ComplexCaller:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    if i == 0:
                        stepped: pl.Tensor[[64], pl.FP32] = bump(acc)
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(stepped)
                    else:
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                    out = pl.yield_(val)
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    if i == 0:
                        result: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                        stepped: pl.Tensor[[64], pl.FP32] = result
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(stepped)
                    else:
                        val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                    out = pl.yield_(val)
                return out

        assert len(ComplexCaller.functions) == 1
        ir.assert_structural_equal(ComplexCaller, Expected)

    def test_inline_with_multiple_iter_args(self):
        """Inline function with for loop using multiple iter_args — expanded into caller."""

        @pl.inline
        def dual_accumulate(
            x: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            init_a: pl.Tensor[[64], pl.FP32] = x
            init_b: pl.Tensor[[64], pl.FP32] = x
            for i, (a, b) in pl.range(5, init_values=(init_a, init_b)):  # type: ignore
                new_a: pl.Tensor[[64], pl.FP32] = pl.add(a, 1.0)
                new_b: pl.Tensor[[64], pl.FP32] = pl.mul(b, 2.0)
                out_a, out_b = pl.yield_(new_a, new_b)
            result: pl.Tensor[[64], pl.FP32] = pl.add(out_a, out_b)
            return result

        @pl.program
        class InlineMultiIter:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = dual_accumulate(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_a: pl.Tensor[[64], pl.FP32] = x
                init_b: pl.Tensor[[64], pl.FP32] = x
                for i, (a, b) in pl.range(5, init_values=(init_a, init_b)):  # type: ignore
                    new_a: pl.Tensor[[64], pl.FP32] = pl.add(a, 1.0)
                    new_b: pl.Tensor[[64], pl.FP32] = pl.mul(b, 2.0)
                    out_a, out_b = pl.yield_(new_a, new_b)
                result: pl.Tensor[[64], pl.FP32] = pl.add(out_a, out_b)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        assert len(InlineMultiIter.functions) == 1
        ir.assert_structural_equal(InlineMultiIter, Expected)

    def test_inline_as_yield_arg_in_if(self):
        """Inline used as argument to pl.yield_() inside if/else branches."""

        @pl.inline
        def scale(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        @pl.program
        class YieldInlineArg:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if flag == 0:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(scale(x))
                else:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(x)
                return out

        # Inline expansion as yield argument: statements emit before yield
        @pl.program
        class Expected:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if flag == 0:
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(result)
                else:
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(x)
                return out

        assert len(YieldInlineArg.functions) == 1
        ir.assert_structural_equal(YieldInlineArg, Expected)

    def test_inline_as_yield_arg_in_for_loop(self):
        """Inline used as argument to pl.yield_() inside a for loop."""

        @pl.inline
        def transform(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class YieldInlineLoop:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    out = pl.yield_(transform(acc))
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = x
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    result: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                    out = pl.yield_(result)
                return out

        assert len(YieldInlineLoop.functions) == 1
        ir.assert_structural_equal(YieldInlineLoop, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
