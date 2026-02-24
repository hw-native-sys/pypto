# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for @pl.function and @pl.program decorators."""

import linecache
import sys
import textwrap

import pypto
import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError, UndefinedVariableError


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
