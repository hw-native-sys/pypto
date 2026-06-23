# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for @pl.function, @pl.inline, and @pl.program decorators."""
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
# Rationale: tests access IR node internals (e.g. .var on Stmt) and chain
# through possibly-None DSL return values; pyright's type model cannot
# represent these runtime DSL constructs accurately.

import linecache
import sys
import textwrap

import pypto
import pypto.language as pl
import pypto.pypto_core.ir as _ir
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import InvalidOperationError, ParserTypeError
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
            # slice takes list arguments
            result: pl.Tensor[[32, 64], pl.FP32] = pl.slice(x, [32, 64], [0, 0])
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

    def test_scalar_legacy_call_rejects_duplicate_dtype(self):
        """Scalar legacy call rejects duplicate dtype values."""
        with pytest.raises(TypeError, match="multiple values for argument 'dtype'"):
            pl.Scalar(pl.FP32, dtype=pl.INT32)

    def test_scalar_legacy_call_rejects_unknown_kwarg(self):
        """Scalar legacy call rejects unknown keyword arguments."""
        with pytest.raises(TypeError, match="unexpected keyword argument 'typo'"):
            pl.Scalar(dtype=pl.FP32, typo=1)

    def test_tensor_call_rejects_unknown_kwarg(self):
        """Tensor call rejects unknown keyword arguments."""
        with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
            pl.Tensor([1], pl.FP32, foo=1)

    def test_tensor_call_rejects_duplicate_shape(self):
        """Tensor call rejects duplicate shape values."""
        with pytest.raises(TypeError, match="multiple values for argument 'shape'"):
            pl.Tensor([1], pl.FP32, shape=[2])

    def test_tensor_legacy_call_notation(self):
        """Legacy Tensor(shape, dtype) call still produces annotation-only instances."""
        tensor = pl.Tensor((64, 128), pl.FP16)
        assert tensor.dtype == pl.FP16
        assert tensor.shape == (64, 128)
        with pytest.raises(ValueError, match="annotation-only Tensor"):
            tensor.unwrap()

    def test_tile_ops_with_scalar(self):
        """Test tile operations with scalar parameter."""

        @pl.function(type=pl.FunctionType.InCore)
        def tile_add_scalar(
            input_tile: pl.Tensor[[64, 64], pl.FP32],
            scalar: pl.Scalar[pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile: pl.Tile[[64, 64], pl.FP32] = pl.load(input_tile, [0, 0], [64, 64])
            result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, scalar)
            output_new: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], output)
            return output_new

        assert isinstance(tile_add_scalar, ir.Function)
        assert tile_add_scalar.func_type == pl.FunctionType.InCore
        assert isinstance(tile_add_scalar.params[1].type, ir.ScalarType)


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
        assert add_func.params[0].name_hint == "x"

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
        assert func.params[0].name_hint == "x"
        assert func.params[1].name_hint == "y"

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
        code = Original.as_python()

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
        code = WithCalls.as_python()

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
        printed = Original.as_python()
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

        result = TwiceCalled.get_function("main")
        body = result.body
        assert isinstance(body, ir.SeqStmts)
        # 4 AssignStmts + 1 ReturnStmt = 5 total
        assert len(body.stmts) == 5, (
            f"Expected 5 statements, got {len(body.stmts)}: {TwiceCalled.as_python()}"
        )
        # Verify the two inlined result vars are distinct (hygiene)
        stmt0_var = body.stmts[0].var
        stmt2_var = body.stmts[2].var
        assert stmt0_var.name_hint == "result"
        assert stmt2_var.name_hint == "result"
        assert stmt0_var is not stmt2_var, "Two inline calls should produce distinct result Vars"

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

    def test_nested_inline_calls(self):
        """Nested @pl.inline calls: add_one(add_one(add_one(x)))."""

        @pl.inline
        def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.program
        class NestedCall:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = add_one(add_one(add_one(x)))
                return z

        result = NestedCall.get_function("main")
        assert result is not None
        body = result.body
        assert isinstance(body, ir.SeqStmts)
        # Each inline adds: result = add(...) + assignment to caller LHS
        # Three inlines → 3 AssignStmts + 1 for final z = ...
        # 3 result = add(...) + 1 z = result + 1 return z = 5 total
        assert len(body.stmts) == 5, f"Expected 5 statements, got {len(body.stmts)}: {NestedCall.as_python()}"
        # Verify the three inlined result vars are all distinct (hygiene)
        assert body.stmts[0].var is not body.stmts[1].var
        assert body.stmts[0].var is not body.stmts[2].var
        assert body.stmts[1].var is not body.stmts[2].var


class TestFunctionCallArgCountValidation:
    """Tests for argument count validation on @pl.function and self.method() calls."""

    def test_external_function_too_few_args(self):
        """External @pl.function called with too few args raises error."""

        @pl.function
        def compute(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
            return result

        with pytest.raises(ParserTypeError, match=r"expects 2 argument\(s\), got 1"):

            @pl.program
            class Bad:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = compute(x)
                    return result

    def test_external_function_correct_args(self):
        """External @pl.function called with correct args works."""

        @pl.function
        def compute(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
            return result

        @pl.program
        class Good:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = compute(x, y)
                return result

        assert len(Good.functions) == 2

    def test_cross_function_too_few_args(self):
        """self.method() called with too few args raises error."""

        with pytest.raises(ParserTypeError, match=r"expects 2 argument\(s\), got 1"):

            @pl.program
            class Bad:
                @pl.function
                def helper(
                    self,
                    x: pl.Tensor[[64], pl.FP32],
                    y: pl.Tensor[[64], pl.FP32],
                ) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                    return result

                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = self.helper(x)
                    return result

    def test_cross_function_correct_args(self):
        """self.method() called with correct args works."""

        @pl.program
        class Good:
            @pl.function
            def helper(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                return result

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.helper(x, y)
                return result

        assert len(Good.functions) == 2

    def test_external_function_too_many_args(self):
        """External @pl.function called with too many args raises error."""

        @pl.function
        def single_arg(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        with pytest.raises(ParserTypeError, match=r"expects 1 argument\(s\), got 2"):

            @pl.program
            class Bad:
                @pl.function
                def main(
                    self,
                    x: pl.Tensor[[64], pl.FP32],
                    y: pl.Tensor[[64], pl.FP32],
                ) -> pl.Tensor[[64], pl.FP32]:
                    result: pl.Tensor[[64], pl.FP32] = single_arg(x, y)
                    return result


class TestCrossFunctionDynamicShapeSubstitution:
    """Tests for dynamic shape variable substitution at cross-function call sites (issue #864)."""

    def test_cross_function_dynamic_shape_substitution(self):
        """Callee with dynamic [M, N] shapes, caller passes [128, 128] → return type is [128, 128]."""
        M = pl.dynamic("M")
        N = pl.dynamic("N")

        @pl.program
        class DynShape:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                result: pl.Tensor[[M, N], pl.FP32] = pl.add(a, b)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c: pl.Tensor[[128, 128], pl.FP32] = self.add_kernel(a, b)
                return c

        orch_func = DynShape.get_function("orchestrator")
        assert orch_func is not None
        body = orch_func.body
        assert isinstance(body, ir.SeqStmts)
        assign_stmt = body.stmts[0]
        assert isinstance(assign_stmt, ir.AssignStmt)
        call_expr = assign_stmt.value
        assert isinstance(call_expr, ir.Call)
        call_type = call_expr.type
        assert isinstance(call_type, ir.TensorType)
        # Verify shape dims are concrete ConstInt, not Var
        for dim in call_type.shape:
            assert isinstance(dim, ir.ConstInt), f"Expected ConstInt, got {type(dim).__name__}: {dim}"
            assert dim.value == 128

    def test_cross_function_dynamic_shape_partial(self):
        """Callee has [M, 64] — only M should be substituted."""
        M = pl.dynamic("M")

        @pl.program
        class PartialDyn:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, 64], pl.FP32],
            ) -> pl.Tensor[[M, 64], pl.FP32]:
                return a

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                c: pl.Tensor[[256, 64], pl.FP32] = self.kernel(a)
                return c

        orch_func = PartialDyn.get_function("orch")
        assert orch_func is not None
        assert isinstance(orch_func.body, ir.SeqStmts)
        assign_stmt = orch_func.body.stmts[0]
        assert isinstance(assign_stmt, ir.AssignStmt)
        call_type = assign_stmt.value.type
        assert isinstance(call_type, ir.TensorType)
        # First dim should be 256 (substituted), second should be 64 (unchanged)
        assert isinstance(call_type.shape[0], ir.ConstInt)
        assert call_type.shape[0].value == 256
        assert isinstance(call_type.shape[1], ir.ConstInt)
        assert call_type.shape[1].value == 64

    def test_cross_function_static_shapes_unchanged(self):
        """All-static shapes → no substitution needed, return types unchanged."""

        @pl.program
        class StaticShape:
            @pl.function
            def helper(
                self,
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function
            def caller(
                self,
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = self.helper(x)
                return c

        caller_func = StaticShape.get_function("caller")
        assert caller_func is not None
        assert isinstance(caller_func.body, ir.SeqStmts)
        assign_stmt = caller_func.body.stmts[0]
        assert isinstance(assign_stmt, ir.AssignStmt)
        call_type = assign_stmt.value.type
        assert isinstance(call_type, ir.TensorType)
        assert isinstance(call_type.shape[0], ir.ConstInt)
        assert call_type.shape[0].value == 64

    def test_cross_function_dynamic_shape_mismatch_raises(self):
        """Callee has [M, N], [M, N] but caller passes [128, 64], [127, 64] → M conflicts."""
        M = pl.dynamic("M")
        N = pl.dynamic("N")

        with pytest.raises(Exception, match="conflicting bindings"):

            @pl.program
            class ShapeMismatch:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    a: pl.Tensor[[M, N], pl.FP32],
                    b: pl.Tensor[[M, N], pl.FP32],
                ) -> pl.Tensor[[M, N], pl.FP32]:
                    result: pl.Tensor[[M, N], pl.FP32] = pl.add(a, b)
                    return result

                @pl.function(type=pl.FunctionType.Orchestration)
                def orch(
                    self,
                    a: pl.Tensor[[128, 64], pl.FP32],
                    b: pl.Tensor[[127, 64], pl.FP32],
                ) -> pl.Tensor[[128, 64], pl.FP32]:
                    c: pl.Tensor[[128, 64], pl.FP32] = self.kernel(a, b)
                    return c


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
            for i, (a, b) in pl.range(5, init_values=(init_a, init_b)):
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
                for i, (a, b) in pl.range(5, init_values=(init_a, init_b)):
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
            for i, (a, b) in pl.range(5, init_values=(init_a, init_b)):
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
                for i, (a, b) in pl.range(5, init_values=(init_a, init_b)):
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


class TestInlineCTKwargs:
    """@pl.inline keyword-only parameters resolve to compile-time constants.

    Verifies that int, float, bool, and str annotated keyword-only
    parameters are evaluated at each call site, type-checked, and
    injected into the inline body's closure. Covers defaults, call-site
    overrides, required params, definition-site default freezing,
    type validation, shadow rejection, and nested CT-kwarg chains."""

    # ------------------------------------------------------------------
    # Core acceptance: int, float, bool, str CT kwargs
    # ------------------------------------------------------------------

    def test_kwonly_int_loop_bound(self):
        """int CT kwarg resolves to ConstInt in a loop bound."""

        @pl.inline
        def loop_add(x: pl.Tensor[[128], pl.FP32], *, k_tile: int = 32):
            result: pl.Tensor[[128], pl.FP32] = x
            for k in pl.range(0, 128, k_tile):
                result = pl.add(result, result)
            return result

        @pl.program
        class CTIntTest:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = loop_add(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                result: pl.Tensor[[128], pl.FP32] = x
                for k in pl.range(0, 128, 32):
                    result = pl.add(result, result)
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(CTIntTest, Expected)

    def test_kwonly_float_arithmetic(self):
        """float CT kwarg resolves to ConstFloat in an arithmetic expression."""

        @pl.inline
        def add_eps(x: pl.Tensor[[64], pl.FP32], *, eps: float = 1e-6):
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, eps)
            return result

        @pl.program
        class CTFloatTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = add_eps(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1e-6)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(CTFloatTest, Expected)

    def test_kwonly_bool_arithmetic(self):
        """bool CT kwarg in a ternary expression raises UnsupportedFeatureError.

        Ternary expressions (``ast.IfExp``) are not yet supported in the parser.
        CT-if is tested separately in test_control_flow.py.
        """

        @pl.inline
        def scale(x: pl.Tensor[[64], pl.FP32], *, enable_scale: bool = True):
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0 if enable_scale else 1.0)
            return result

        with pytest.raises(UnsupportedFeatureError):

            @pl.program
            class CTBoolTest:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = scale(x)
                    return y

    def test_kwonly_str_expression(self):
        """str CT kwarg resolves to a Python str value in an expression."""

        @pl.inline
        def identity(x: pl.Tensor[[64], pl.FP32], *, tag: str = "default"):
            _ = tag  # reference the str to ensure it resolves
            result: pl.Tensor[[64], pl.FP32] = x
            return result

        @pl.program
        class CTStrTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = identity(x)
                return y

        assert len(CTStrTest.functions) == 1
        assert CTStrTest.get_function("identity") is None

    def test_all_four_types(self):
        """All four CT kwarg types on one inline: int, float, bool, str."""

        @pl.inline
        def fused(
            x: pl.Tensor[[128], pl.FP32],
            *,
            k_tile: int = 64,
            eps: float = 1e-6,
            use_fp32: bool = True,
            tag: str = "fused",
        ):
            _ = tag
            result: pl.Tensor[[128], pl.FP32] = pl.add(x, eps)
            for k in pl.range(0, 128, k_tile):
                result = pl.add(result, result)
            return result

        @pl.program
        class AllTypesTest:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = fused(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                _ = "fused"
                result: pl.Tensor[[128], pl.FP32] = pl.add(x, 1e-6)
                for k in pl.range(0, 128, 64):
                    result = pl.add(result, result)
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(AllTypesTest, Expected)

    def test_multiple_same_type_kwargs(self):
        """Multiple CT kwargs of the same type (two ints)."""

        @pl.inline
        def unfold(x: pl.Tensor[[128], pl.FP32], *, a: int = 1, b: int = 2):
            result: pl.Tensor[[128], pl.FP32] = pl.mul(x, a + b)
            return result

        @pl.program
        class MultiSame:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = unfold(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                result: pl.Tensor[[128], pl.FP32] = pl.mul(x, 3)
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(MultiSame, Expected)

    # ------------------------------------------------------------------
    # Defaults & overrides
    # ------------------------------------------------------------------

    def test_default_value_used(self):
        """CT kwarg with a default — call with no keyword uses the default."""

        @pl.inline
        def mul_by(x: pl.Tensor[[64], pl.FP32], *, scale: float = 3.0):
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, scale)
            return result

        @pl.program
        class DefaultTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = mul_by(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 3.0)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(DefaultTest, Expected)

    def test_override_default_at_call_site(self):
        """CT kwarg default overridden at call site with a different value."""

        @pl.inline
        def mul_by(x: pl.Tensor[[64], pl.FP32], *, scale: float = 3.0):
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, scale)
            return result

        @pl.program
        class OverrideTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = mul_by(x, scale=5.0)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 5.0)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(OverrideTest, Expected)

    def test_override_some_defaults(self):
        """Override some CT kwargs while leaving others at default."""

        @pl.inline
        def mixed(x: pl.Tensor[[128], pl.FP32], *, k_tile: int = 64, eps: float = 1e-6):
            result: pl.Tensor[[128], pl.FP32] = pl.add(x, eps)
            for k in pl.range(0, 128, k_tile):
                result = pl.add(result, result)
            return result

        @pl.program
        class MixedTest:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                # Override eps, keep k_tile at default (64)
                y: pl.Tensor[[128], pl.FP32] = mixed(x, eps=1e-7)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                result: pl.Tensor[[128], pl.FP32] = pl.add(x, 1e-7)
                for k in pl.range(0, 128, 64):
                    result = pl.add(result, result)
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(MixedTest, Expected)

    def test_required_ct_kwarg_omitted(self):
        """Required CT kwarg (no default) omitted at call site → ParserTypeError."""

        @pl.inline
        def req(x: pl.Tensor[[64], pl.FP32], *, k_tile: int):
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, k_tile)
            return result

        with pytest.raises(ParserTypeError, match="missing required"):

            @pl.program
            class MissingReq:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = req(x)
                    return y

    def test_default_frozen_from_def_site(self):
        """A name-based default is frozen at definition time, not re-resolved from caller."""
        DEF_K_TILE = 128

        @pl.inline
        def frozen(x: pl.Tensor[[128], pl.FP32], *, k_tile: int = DEF_K_TILE):
            result: pl.Tensor[[128], pl.FP32] = x
            for k in pl.range(0, 128, k_tile):
                result = pl.add(result, result)
            return result

        # The caller's DEF_K_TILE (if any) should NOT affect the callee.
        DEF_K_TILE = 999  # noqa: F811 — this must NOT be picked up

        @pl.program
        class FrozenDefaultTest:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = frozen(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                result: pl.Tensor[[128], pl.FP32] = x
                for k in pl.range(0, 128, 128):  # 128, not 999
                    result = pl.add(result, result)
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(FrozenDefaultTest, Expected)

    # ------------------------------------------------------------------
    # Type validation
    # ------------------------------------------------------------------

    def test_int_annotation_wrong_type(self):
        """int-annotated CT kwarg passed a str → ParserTypeError."""

        @pl.inline
        def f(x: pl.Tensor[[64], pl.FP32], *, k_tile: int = 32):
            result: pl.Tensor[[64], pl.FP32] = x
            for k in pl.range(0, 64, k_tile):
                result = pl.add(result, result)
            return result

        with pytest.raises(ParserTypeError, match="k_tile"):

            @pl.program
            class WrongType:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = f(x, k_tile="hello")
                    return y

    def test_float_annotation_wrong_type(self):
        """float-annotated CT kwarg passed a non-float → ParserTypeError."""

        @pl.inline
        def f(x: pl.Tensor[[64], pl.FP32], *, eps: float = 1.0):
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, eps)
            return result

        with pytest.raises(ParserTypeError, match="eps"):

            @pl.program
            class WrongType:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = f(x, eps="not-a-float")
                    return y

    def test_bool_annotation_int_rejected(self):
        """bool-annotated CT kwarg passed an int → ParserTypeError (exact type check)."""

        @pl.inline
        def f(x: pl.Tensor[[64], pl.FP32], *, flag: bool = True):
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0 if flag else 1.0)
            return result

        with pytest.raises(ParserTypeError, match="flag"):

            @pl.program
            class BoolInt:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = f(x, flag=1)  # int, not bool
                    return y

    def test_str_annotation_wrong_type(self):
        """str-annotated CT kwarg passed a non-str → ParserTypeError."""

        @pl.inline
        def f(x: pl.Tensor[[64], pl.FP32], *, name: str = "default"):
            _ = name
            return x

        with pytest.raises(ParserTypeError, match="name"):

            @pl.program
            class WrongType:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = f(x, name=42)
                    return y

    def test_unsupported_annotation_on_kwonly(self):
        """A keyword-only param with a non-{int, float, bool, str} annotation → error at definition."""

        with pytest.raises((ParserSyntaxError, TypeError), match="annotation|unsupported"):

            @pl.inline
            def bad(x: pl.Tensor[[64], pl.FP32], *, k_tile: "int | None" = 128):
                result: pl.Tensor[[64], pl.FP32] = x
                for k in pl.range(0, 64, k_tile):  # pyright: ignore[reportArgumentType]
                    # Rationale: body is never executed; the @pl.inline
                    # definition raises ParserSyntaxError on the unsupported
                    # "int | None" annotation before the body runs.
                    result = pl.add(result, result)
                return result

    def test_missing_annotation_on_kwonly(self):
        """A keyword-only param with no annotation → error at definition."""

        with pytest.raises((ParserSyntaxError, TypeError), match="annotation|type"):

            @pl.inline
            def bad(x: pl.Tensor[[64], pl.FP32], *, k_tile=128):
                result: pl.Tensor[[64], pl.FP32] = x
                for k in pl.range(0, 64, k_tile):
                    result = pl.add(result, result)
                return result

    # ------------------------------------------------------------------
    # Hygiene & constraints
    # ------------------------------------------------------------------

    def test_kwonly_name_collides_with_positional(self):
        """CT-kwarg name collides with a positional param → error at definition time.

        Python itself rejects ``def bad(x, *, x)`` (duplicate argument name),
        so the test uses exec to verify that the @pl.inline decorator, or
        Python's own parser, catches the collision.
        """

        with pytest.raises((ParserSyntaxError, TypeError, SyntaxError)):
            exec(
                textwrap.dedent("""\
                import pypto.language as pl
                @pl.inline
                def bad(x: pl.Tensor[[64], pl.FP32], *, x: int = 128):
                    return x
            """)
            )

    def test_body_var_shadows_ct_kwarg(self):
        """A body-defined DSL variable shadows the CT kwarg → error at parse time.

        The parser rejects plain assignments, annotated assignments,
        and augmented assignments whose target is a CT-kwarg name.
        """

        @pl.inline
        def shadow(x: pl.Tensor[[64], pl.FP32], *, k_tile: int = 128):
            # The body defines k_tile as a DSL variable; it shadows the CT kwarg
            k_tile: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
            result: pl.Tensor[[64], pl.FP32] = k_tile  # refers to DSL var, not CT kwarg
            return result

        with pytest.raises((ParserTypeError, ParserSyntaxError), match="shadow|assign"):

            @pl.program
            class ShadowTest:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = shadow(x)
                    return y

    def test_ct_kwarg_as_assign_lhs(self):
        """CT-kwarg name used as assignment LHS inside body → error at parse time."""

        @pl.inline
        def bad(x: pl.Tensor[[64], pl.FP32], *, k_tile: int = 128):
            k_tile = pl.mul(x, x)  # reassign — should be rejected
            return k_tile

        with pytest.raises((ParserTypeError, ParserSyntaxError), match="shadow|assign|LHS"):

            @pl.program
            class BadAssign:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = bad(x)
                    return y

    def test_augmented_assign_on_ct_kwarg(self):
        """CT-kwarg name used in augmented assignment (k_tile += 1) → error."""

        @pl.inline
        def bad(x: pl.Tensor[[64], pl.FP32], *, k_tile: int = 128):
            k_tile += 1
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, k_tile)
            return result

        with pytest.raises((ParserTypeError, ParserSyntaxError), match="shadow|assign|augmented"):

            @pl.program
            class BadAugAssign:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = bad(x)
                    return y

    def test_callers_local_does_not_affect_default(self):
        """Caller's scope variable with the same name as a CT-kwarg default is irrelevant."""

        @pl.inline
        def callee(x: pl.Tensor[[64], pl.FP32], *, k_tile: int = 64):
            result: pl.Tensor[[64], pl.FP32] = x
            for k in pl.range(0, 64, k_tile):
                result = pl.add(result, result)
            return result

        # Caller has its own k_tile — should NOT affect callee's default
        @pl.program
        class CallerScopeTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                k_tile: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)  # noqa: F841 — scope-isolation test fixture
                y: pl.Tensor[[64], pl.FP32] = callee(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                k_tile: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)  # noqa: F841 — scope-isolation test fixture
                result: pl.Tensor[[64], pl.FP32] = x
                for k in pl.range(0, 64, 64):
                    result = pl.add(result, result)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(CallerScopeTest, Expected)

    # ------------------------------------------------------------------
    # Backward compatibility: no kwonly → keyword args still rejected
    # ------------------------------------------------------------------

    def test_no_kwonly_still_rejects_keyword_args(self):
        """An @pl.inline without * still rejects keyword args (backward compat)."""

        @pl.inline
        def plain(x: pl.Tensor[[64], pl.FP32]):
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        with pytest.raises(ParserTypeError, match="does not accept keyword argument"):

            @pl.program
            class BadKeyword:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]):
                    y: pl.Tensor[[64], pl.FP32] = plain(x, something=1.0)
                    return y

    # ------------------------------------------------------------------
    # int semantics: loop bound, shape, slice, const, folding
    # ------------------------------------------------------------------

    def test_int_in_loop_bound(self):
        """int CT kwarg in pl.range loop bound."""

        @pl.inline
        def loop(x: pl.Tensor[[256], pl.FP32], *, tile: int = 64):
            result: pl.Tensor[[256], pl.FP32] = x
            for k in pl.range(0, 256, tile):
                result = pl.add(result, result)
            return result

        @pl.program
        class LoopTest:
            @pl.function
            def main(self, x: pl.Tensor[[256], pl.FP32]):
                y: pl.Tensor[[256], pl.FP32] = loop(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[256], pl.FP32]):
                result: pl.Tensor[[256], pl.FP32] = x
                for k in pl.range(0, 256, 64):
                    result = pl.add(result, result)
                y: pl.Tensor[[256], pl.FP32] = result
                return y

        ir.assert_structural_equal(LoopTest, Expected)

    def test_int_in_shape_dimension(self):
        """int CT kwarg as a shape dimension in pl.full."""

        @pl.inline
        def make_tile(*, d_tile: int = 64):
            result: pl.Tensor[[1, 64], pl.FP32] = pl.full([1, d_tile], dtype=pl.FP32, value=0.0)
            return result

        @pl.program
        class ShapeTest:
            @pl.function
            def main(self):
                y: pl.Tensor[[1, 64], pl.FP32] = make_tile()
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                result: pl.Tensor[[1, 64], pl.FP32] = pl.full([1, 64], dtype=pl.FP32, value=0.0)
                y: pl.Tensor[[1, 64], pl.FP32] = result
                return y

        ir.assert_structural_equal(ShapeTest, Expected)

    def test_int_in_slice_bound(self):
        """int CT kwarg in a tensor slice bound."""

        @pl.inline
        def slice_tile(x: pl.Tensor[[128, 128], pl.FP32], *, t_tile: int = 32):
            result: pl.Tensor[[32, 32], pl.FP32] = x[0:t_tile, 0:t_tile]
            return result

        @pl.program
        class SliceTest:
            @pl.function
            def main(self, x: pl.Tensor[[128, 128], pl.FP32]):
                y: pl.Tensor[[32, 32], pl.FP32] = slice_tile(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128, 128], pl.FP32]):
                result: pl.Tensor[[32, 32], pl.FP32] = x[0:32, 0:32]
                y: pl.Tensor[[32, 32], pl.FP32] = result
                return y

        ir.assert_structural_equal(SliceTest, Expected)

    def test_int_in_const(self):
        """int CT kwarg as a typed constant via pl.const."""

        @pl.inline
        def const_tile(*, k_tile: int = 64):
            result: pl.Scalar[pl.INT64] = pl.const(k_tile, pl.INT64)
            return result

        @pl.program
        class ConstTest:
            @pl.function
            def main(self):
                y: pl.Scalar[pl.INT64] = const_tile()
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                result: pl.Scalar[pl.INT64] = pl.const(64, pl.INT64)
                y: pl.Scalar[pl.INT64] = result
                return y

        ir.assert_structural_equal(ConstTest, Expected)

    def test_int_constant_folding(self):
        """Arithmetic on int CT kwargs constant-folds: k_tile * stage_depth → ConstInt(512)."""

        @pl.inline
        def folded(x: pl.Tensor[[1024], pl.FP32], *, k_tile: int = 128, stage_depth: int = 4):
            tile_count: pl.Scalar[pl.INT64] = pl.const(k_tile * stage_depth, pl.INT64)
            result: pl.Tensor[[1024], pl.FP32] = pl.add(x, tile_count)
            return result

        @pl.program
        class FoldTest:
            @pl.function
            def main(self, x: pl.Tensor[[1024], pl.FP32]):
                y: pl.Tensor[[1024], pl.FP32] = folded(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[1024], pl.FP32]):
                tile_count: pl.Scalar[pl.INT64] = pl.const(512, pl.INT64)
                result: pl.Tensor[[1024], pl.FP32] = pl.add(x, tile_count)
                y: pl.Tensor[[1024], pl.FP32] = result
                return y

        ir.assert_structural_equal(FoldTest, Expected)

    # ------------------------------------------------------------------
    # float semantics
    # ------------------------------------------------------------------

    def test_float_arithmetic(self):
        """float CT kwarg in arithmetic: sq_sum + eps."""

        @pl.inline
        def normalize(x: pl.Tensor[[64], pl.FP32], *, eps: float = 1e-6):
            sq: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
            result: pl.Tensor[[64], pl.FP32] = pl.add(sq, eps)
            return result

        @pl.program
        class FloatArith:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = normalize(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                sq: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(sq, 1e-6)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(FloatArith, Expected)

    def test_float_scalar_operand(self):
        """float CT kwarg as scalar operand in pl.mul."""

        @pl.inline
        def scale(x: pl.Tensor[[64], pl.FP32], *, scale_val: float = 2.5):
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, scale_val)
            return result

        @pl.program
        class ScalarOp:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = scale(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.5)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(ScalarOp, Expected)

    def test_float_constant_folding(self):
        """float CT kwarg participates in constant folding: 1.0 / hidden + eps."""

        @pl.inline
        def rms_approx(x: pl.Tensor[[64], pl.FP32], *, hidden: float = 512.0, eps: float = 1e-6):
            inv: pl.Scalar[pl.FP32] = pl.const(1.0 / hidden + eps, pl.FP32)
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, inv)
            return result

        @pl.program
        class FoldFloat:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = rms_approx(x)
                return y

        assert len(FoldFloat.functions) == 1

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_ct_kwarg_evaluation_order(self):
        """CT kwargs evaluated in signature order (not call-site keyword order).

        An expression like t_tile // 2 that references a prior CT kwarg sees
        the resolved value because evaluation follows signature order.
        """

        @pl.inline
        def ordered(x: pl.Tensor[[128], pl.FP32], *, t_tile: int = 64, half_tile: int = 32):
            # half_tile is independently declared, not derived from t_tile
            result: pl.Tensor[[128], pl.FP32] = pl.add(x, pl.const(t_tile + half_tile, pl.INT64))
            return result

        # Call in reversed keyword order — should still work
        @pl.program
        class OrderTest:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = ordered(x, half_tile=16, t_tile=128)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                result: pl.Tensor[[128], pl.FP32] = pl.add(x, pl.const(144, pl.INT64))
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(OrderTest, Expected)

    def test_large_int_value(self):
        """Large int CT kwarg values work (e.g. k_tile = 2**20)."""

        @pl.inline
        def big(x: pl.Tensor[[1048576], pl.FP32], *, tile: int = 1048576):
            result: pl.Tensor[[1048576], pl.FP32] = x
            for k in pl.range(0, 1048576, tile):
                result = pl.add(result, result)
            return result

        @pl.program
        class BigTest:
            @pl.function
            def main(self, x: pl.Tensor[[1048576], pl.FP32]):
                y: pl.Tensor[[1048576], pl.FP32] = big(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[1048576], pl.FP32]):
                result: pl.Tensor[[1048576], pl.FP32] = x
                for k in pl.range(0, 1048576, 1048576):
                    result = pl.add(result, result)
                y: pl.Tensor[[1048576], pl.FP32] = result
                return y

        ir.assert_structural_equal(BigTest, Expected)

    def test_negative_int_value(self):
        """Negative int CT kwarg values work."""

        @pl.inline
        def neg(x: pl.Tensor[[64], pl.FP32], *, offset: int = -1):
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, offset)
            return result

        @pl.program
        class NegTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = neg(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, -1)
                y: pl.Tensor[[64], pl.FP32] = result
                return y

        ir.assert_structural_equal(NegTest, Expected)

    def test_float_inf_nan(self):
        """Special float values (inf, nan) work as CT kwargs."""

        @pl.inline
        def with_inf(x: pl.Tensor[[64], pl.FP32], *, big: float = float("inf")):
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, big)
            return result

        @pl.program
        class InfTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = with_inf(x)
                return y

        assert len(InfTest.functions) == 1

    def test_empty_string(self):
        """Empty string as CT kwarg works."""

        @pl.inline
        def empty_str(x: pl.Tensor[[64], pl.FP32], *, tag: str = ""):
            _ = tag
            return x

        @pl.program
        class EmptyStrTest:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]):
                y: pl.Tensor[[64], pl.FP32] = empty_str(x)
                return y

        assert len(EmptyStrTest.functions) == 1

    def test_caller_expr_folds(self):
        """Caller-side expression for CT kwarg value is constant-folded before injection."""

        HEAD_DIM = 128

        @pl.inline
        def attn(x: pl.Tensor[[128], pl.FP32], *, d_tile: int = 64):
            result: pl.Tensor[[128], pl.FP32] = x
            for k in pl.range(0, 128, d_tile):
                result = pl.add(result, result)
            return result

        @pl.program
        class ExprFold:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = attn(x, d_tile=HEAD_DIM // 2)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                result: pl.Tensor[[128], pl.FP32] = x
                for k in pl.range(0, 128, 64):
                    result = pl.add(result, result)
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(ExprFold, Expected)

    # ------------------------------------------------------------------
    # Outer-to-inner CT kwarg transitivity (@pl.inline calling @pl.inline)
    # ------------------------------------------------------------------

    def test_outer_inner_ct_kwarg_chain(self):
        """An outer @pl.inline passes its own CT kwarg as the inner's call-site value."""

        @pl.inline
        def inner(x: pl.Tensor[[128], pl.FP32], *, k_tile: int = 64):
            result: pl.Tensor[[128], pl.FP32] = x
            for k in pl.range(0, 128, k_tile):
                result = pl.add(result, result)
            return result

        @pl.inline
        def outer(x: pl.Tensor[[128], pl.FP32], *, tile: int = 32):
            # tile is outer's CT kwarg; it feeds inner's k_tile
            y: pl.Tensor[[128], pl.FP32] = inner(x, k_tile=tile)
            return y

        @pl.program
        class ChainTest:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = outer(x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                # inner's body spliced with k_tile=32 (outer's default)
                result: pl.Tensor[[128], pl.FP32] = x
                for k in pl.range(0, 128, 32):
                    result = pl.add(result, result)
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(ChainTest, Expected)

    def test_outer_inner_override_propagates(self):
        """Overriding outer's CT kwarg propagates to inner."""

        @pl.inline
        def inner(x: pl.Tensor[[128], pl.FP32], *, k_tile: int = 64):
            result: pl.Tensor[[128], pl.FP32] = x
            for k in pl.range(0, 128, k_tile):
                result = pl.add(result, result)
            return result

        @pl.inline
        def outer(x: pl.Tensor[[128], pl.FP32], *, tile: int = 32):
            y: pl.Tensor[[128], pl.FP32] = inner(x, k_tile=tile)
            return y

        # Override outer's tile=16 → inner should see k_tile=16
        @pl.program
        class OverrideChain:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = outer(x, tile=16)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                result: pl.Tensor[[128], pl.FP32] = x
                for k in pl.range(0, 128, 16):
                    result = pl.add(result, result)
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(OverrideChain, Expected)

    # ------------------------------------------------------------------
    # End-to-end: rmsnorm-like pattern
    # ------------------------------------------------------------------

    def test_rmsnorm_pattern(self):
        """Full rmsnorm-like @pl.inline with k_tile and eps CT kwargs."""

        @pl.inline
        def rmsnorm(
            x: pl.Tensor[[128, 5120], pl.BF16],
            gamma_w: pl.Tensor[[1, 5120], pl.FP32],
            out: pl.Tensor[[128, 5120], pl.BF16],
            *,
            k_tile: int = 128,
            eps: float = 1e-6,
        ):
            for k in pl.range(0, 5120, k_tile):
                sq_sum: pl.Tensor[[128, 5120], pl.FP32] = pl.cast(pl.mul(x, x), pl.FP32)
                sq_sum = pl.add(sq_sum, eps)
            return out

        @pl.program
        class RmsnormTest:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[128, 5120], pl.BF16],
                gamma_w: pl.Tensor[[1, 5120], pl.FP32],
                out: pl.Tensor[[128, 5120], pl.BF16],
            ):
                y: pl.Tensor[[128, 5120], pl.BF16] = rmsnorm(x, gamma_w, out)
                return y

        assert len(RmsnormTest.functions) == 1

    def test_rmsnorm_override_eps(self):
        """rmsnorm with overridden eps for a different model size."""

        @pl.inline
        def rmsnorm(
            x: pl.Tensor[[128, 5120], pl.BF16],
            gamma_w: pl.Tensor[[1, 5120], pl.FP32],
            out: pl.Tensor[[128, 5120], pl.BF16],
            *,
            k_tile: int = 128,
            eps: float = 1e-6,
        ):
            for k in pl.range(0, 5120, k_tile):
                sq_sum: pl.Tensor[[128, 5120], pl.FP32] = pl.cast(pl.mul(x, x), pl.FP32)
                sq_sum = pl.add(sq_sum, eps)
            return out

        @pl.program
        class RmsnormOverride:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[128, 5120], pl.BF16],
                gamma_w: pl.Tensor[[1, 5120], pl.FP32],
                out: pl.Tensor[[128, 5120], pl.BF16],
            ):
                y: pl.Tensor[[128, 5120], pl.BF16] = rmsnorm(x, gamma_w, out, eps=1e-7, k_tile=64)
                return y

        assert len(RmsnormOverride.functions) == 1


class TestInlineCTKwargArrayCreate:
    """CT kwargs in @pl.inline used as pl.array.create size arguments.

    When a CT kwarg name is used directly (without rebinding), the parser
    resolves it from closure_vars -> ConstInt, which _ir_ops.create accepts
    as an Expr.  When a CT kwarg is rebound to a local variable
    (``SIZE = size``), the parser creates a Scalar Var and
    pl.array.create rejects it.

    AST constant folding (replacing CT kwarg names with ast.Constant before
    parsing) eliminates the need for rebindings and keeps pl.array.create
    on the happy path.
    """

    def test_direct_ct_kwarg_in_array_create(self):
        """Using a CT kwarg name directly in pl.array.create works -
        closure_vars -> ConstInt -> _ir_ops.create(Expr) path."""

        @pl.inline
        def make_tids(*, size: int = 4):
            tids = pl.array.create(size, pl.TASK_ID)
            return tids

        @pl.program
        class DirectTest:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = make_tids()  # noqa: F841 — inline call fixture
                return x

        fn = DirectTest.get_function("main")
        assert fn is not None
        # Verify the array.create op in the IR has ConstInt size.

        create_calls = []

        def _walk(stmt):
            if isinstance(stmt, _ir.SeqStmts):
                for s in stmt.stmts:
                    _walk(s)
            elif isinstance(stmt, _ir.AssignStmt) and isinstance(stmt.value, _ir.Call):
                if stmt.value.op.name == "array.create":
                    create_calls.append(stmt.value)
            body = getattr(stmt, "body", None)
            if body is not None:
                _walk(body)

        _walk(fn.body)
        assert len(create_calls) >= 1, "expected at least one array.create call"
        extent = create_calls[0].args[0]
        assert isinstance(extent, _ir.ConstInt), (
            f"expected ConstInt for array.create size, got {type(extent).__name__}"
        )
        assert extent.value == 4, f"expected size=4, got {extent.value}"

    def test_direct_ct_kwarg_override_in_array_create(self):
        """Call-site override of CT kwarg used in pl.array.create."""

        @pl.inline
        def make_tids(*, size: int = 4):
            tids = pl.array.create(size, pl.TASK_ID)
            return tids

        @pl.program
        class OverrideTest:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = make_tids(size=8)  # noqa: F841 — inline call fixture
                return x

        fn = OverrideTest.get_function("main")
        assert fn is not None

        create_calls = []

        def _walk(stmt):
            if isinstance(stmt, _ir.SeqStmts):
                for s in stmt.stmts:
                    _walk(s)
            elif isinstance(stmt, _ir.AssignStmt) and isinstance(stmt.value, _ir.Call):
                if stmt.value.op.name == "array.create":
                    create_calls.append(stmt.value)
            body = getattr(stmt, "body", None)
            if body is not None:
                _walk(body)

        _walk(fn.body)
        assert len(create_calls) >= 1, "expected at least one array.create call"
        extent = create_calls[0].args[0]
        assert isinstance(extent, _ir.ConstInt), (
            f"expected ConstInt for array.create size, got {type(extent).__name__}"
        )
        assert extent.value == 8, f"expected size=8, got {extent.value}"

    def test_rebinding_ct_kwarg_fails_in_array_create(self):
        """Rebinding a CT kwarg to a local name creates a Scalar, which
        pl.array.create rejects - this is the motivating bug."""

        with pytest.raises((TypeError, ParserTypeError, InvalidOperationError)):

            @pl.inline
            def make_tids(*, size: int = 4):
                N = size  # <- creates Scalar[INDEX] Var
                tids = pl.array.create(N, pl.TASK_ID)
                return tids

            @pl.program
            class RebindFail:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        tids = make_tids()  # noqa: F841 — inline call fixture
                    return x

    def test_direct_ct_kwarg_in_create_tensor_shape(self):
        """CT kwarg used directly in pl.create_tensor shape works."""

        @pl.inline
        def alloc(*, rows: int = 16, cols: int = 64):
            buf = pl.create_tensor([rows, cols], dtype=pl.FP32)
            return buf

        @pl.program
        class AllocTest:
            @pl.function
            def main(self, x: pl.Tensor[[16, 64], pl.FP32]) -> pl.Tensor[[16, 64], pl.FP32]:
                y: pl.Tensor[[16, 64], pl.FP32] = alloc()
                y = pl.add(y, x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[16, 64], pl.FP32]) -> pl.Tensor[[16, 64], pl.FP32]:
                buf = pl.create_tensor([16, 64], dtype=pl.FP32)
                y: pl.Tensor[[16, 64], pl.FP32] = buf
                y = pl.add(y, x)
                return y

        ir.assert_structural_equal(AllocTest, Expected)


class TestInlineCTKwargLoopBounds:
    """CT kwargs used in loop bounds and subscript offsets resolve to
    ConstInt, enabling DimensionsEqual across different call sites."""

    def test_ct_kwarg_loop_bound_and_subscript(self):
        """CT kwarg used as loop step and subscript bound."""

        @pl.inline
        def tiled_add(x: pl.Tensor[[128], pl.FP32], *, tile: int = 32):
            result: pl.Tensor[[128], pl.FP32] = x
            for k in pl.range(0, 128, tile):
                chunk = x[k : k + tile]
                result = pl.assemble(result, pl.add(chunk, chunk), [k])
            return result

        @pl.program
        class TiledTest:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                y: pl.Tensor[[128], pl.FP32] = tiled_add(x, tile=64)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[128], pl.FP32]):
                result: pl.Tensor[[128], pl.FP32] = x
                for k in pl.range(0, 128, 64):
                    chunk = x[k : k + 64]
                    result = pl.assemble(result, pl.add(chunk, chunk), [k])
                y: pl.Tensor[[128], pl.FP32] = result
                return y

        ir.assert_structural_equal(TiledTest, Expected)


class TestInlineInCoreScope:
    """@pl.inline expanded inside pl.parallel + pl.at(CORE_GROUP) contexts.

    These tests pin the Qwen3-14B RMSNorm migration pattern: an ``@pl.inline``
    body that uses ``pl.slice`` + ``pl.cast`` + reductions/broadcasts, called
    from inside a caller's ``for b0 in pl.parallel(...): with pl.at(...)`` block,
    returning the assembled output and the caller capturing the return.

    Coverage gap these close: ``test_rmsnorm_pattern`` uses full-tensor
    ``pl.cast`` in a plain ``@pl.function`` (no ``pl.parallel``/``pl.at``);
    the JIT toy tests use ``pl.load``/``pl.store``. No prior test combines
    ``@pl.inline`` + ``pl.slice``/``pl.cast`` + ``pl.parallel`` +
    ``pl.at(CORE_GROUP)`` + the return-and-capture output-escape pattern.
    """

    def test_rmsnorm_full_in_parallel_incore_scope(self):
        """Full RMSNorm @pl.inline called inside pl.parallel + pl.at(CORE_GROUP).

        The body accumulates sq-sum, computes inv_rms, and normalizes+assembles
        into the passed-in ``out`` tensor, then returns it (the body is
        scope-isolated — the return is the only way the assembled output
        escapes). The caller captures the return. Structural equality against
        a manual expand confirms the parse-time splice is identical to
        hand-written code in the scoped context.
        """

        @pl.inline
        def rmsnorm_tile(
            x: pl.Tensor[[32, 128], pl.BF16],
            gamma: pl.Tensor[[1, 128], pl.FP32],
            out: pl.Tensor[[32, 128], pl.BF16],
            b0,
            *,
            rows: int = 16,
            k_chunk: int = 64,
            eps: float = 1e-6,
            hidden: int = 128,
        ) -> pl.Tensor[[32, 128], pl.BF16]:
            sq_sum = pl.full([1, rows], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden // k_chunk):
                k0 = kb * k_chunk
                chunk = pl.cast(pl.slice(x, [rows, k_chunk], [b0, k0]), pl.FP32)
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, rows]),
                )
            inv_rms = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / hidden), eps)),
                [rows, 1],
            )
            for kb in pl.range(hidden // k_chunk):
                k0 = kb * k_chunk
                h = pl.cast(pl.slice(x, [rows, k_chunk], [b0, k0]), pl.FP32)
                g = pl.slice(gamma, [1, k_chunk], [0, k0])
                out = pl.assemble(
                    out,
                    pl.cast(
                        pl.col_expand_mul(pl.row_expand_mul(h, inv_rms), g),
                        pl.BF16,
                    ),
                    [b0, k0],
                )
            return out

        @pl.program
        class RmsnormScoped:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 128], pl.BF16],
                gamma: pl.Tensor[[1, 128], pl.FP32],
                out: pl.Tensor[[32, 128], pl.BF16],
            ) -> pl.Tensor[[32, 128], pl.BF16]:
                for b0 in pl.parallel(0, 32, 16):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                        out = rmsnorm_tile(x, gamma, out, b0)
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 128], pl.BF16],
                gamma: pl.Tensor[[1, 128], pl.FP32],
                out: pl.Tensor[[32, 128], pl.BF16],
            ) -> pl.Tensor[[32, 128], pl.BF16]:
                for b0 in pl.parallel(0, 32, 16):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                        sq_sum = pl.full([1, 16], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(2):
                            k0 = kb * 64
                            chunk = pl.cast(pl.slice(x, [16, 64], [b0, k0]), pl.FP32)
                            sq_sum = pl.add(
                                sq_sum,
                                pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, 16]),
                            )
                        inv_rms = pl.reshape(
                            pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / 128), 1e-6)),
                            [16, 1],
                        )
                        for kb in pl.range(2):
                            k0 = kb * 64
                            h = pl.cast(pl.slice(x, [16, 64], [b0, k0]), pl.FP32)
                            g = pl.slice(gamma, [1, 64], [0, k0])
                            out = pl.assemble(
                                out,
                                pl.cast(
                                    pl.col_expand_mul(pl.row_expand_mul(h, inv_rms), g),
                                    pl.BF16,
                                ),
                                [b0, k0],
                            )
                return out

        ir.assert_structural_equal(RmsnormScoped, Expected)

    def test_rmsnorm_recip_top_level_return(self):
        """rmsnorm_recip @pl.inline: reciprocal-only, no pl.parallel nesting.

        Mirrors decode_layer.py's ``rms_recip`` site (site 6): top-level in the
        body (not nested in pl.parallel), computes only the reciprocal, returns
        it. The caller uses the returned inv_rms in a pl.assemble. Confirms the
        return-escape pattern for a value-producing (not output-mutating)
        inline.
        """

        @pl.inline
        def rmsnorm_recip(
            x: pl.Tensor[[16, 128], pl.BF16],
            *,
            rows: int = 16,
            k_chunk: int = 64,
            eps: float = 1e-6,
            hidden: int = 128,
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            sq_sum = pl.full([1, rows], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden // k_chunk):
                k0 = kb * k_chunk
                chunk = pl.cast(x[:, k0 : k0 + k_chunk], pl.FP32)
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, rows]),
                )
            inv_rms = pl.recip(pl.sqrt(pl.reshape(pl.add(pl.mul(sq_sum, 1.0 / hidden), eps), [rows, 1])))
            return inv_rms

        @pl.program
        class RecipScoped:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out: pl.Tensor[[16, 1], pl.FP32],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip"):
                    inv_rms = rmsnorm_recip(x)
                    out = pl.assemble(out, inv_rms, [0, 0])
                return out

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out: pl.Tensor[[16, 1], pl.FP32],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip"):
                    sq_sum = pl.full([1, 16], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(2):
                        k0 = kb * 64
                        chunk = pl.cast(x[:, k0 : k0 + 64], pl.FP32)
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, 16]),
                        )
                    inv_rms = pl.recip(pl.sqrt(pl.reshape(pl.add(pl.mul(sq_sum, 1.0 / 128), 1e-6), [16, 1])))
                    out = pl.assemble(out, inv_rms, [0, 0])
                return out

        ir.assert_structural_equal(RecipScoped, Expected)

    def test_inline_in_incore_scope_no_return_raises(self):
        """An @pl.inline body that assembles into ``out`` but does not return
        it must be rejected — the body is scope-isolated, so the mutation
        cannot escape and the parser requires a return value.
        """

        @pl.inline
        def no_return(x, out, b0, *, rows: int = 16, k_chunk: int = 64):
            chunk = pl.cast(pl.slice(x, [rows, k_chunk], [b0, 0]), pl.FP32)
            out = pl.assemble(out, pl.cast(chunk, pl.BF16), [b0, 0])
            # no return — body is scope-isolated, the assemble is lost

        with pytest.raises(ParserTypeError, match="has no return value"):

            @pl.program
            class _Bad:
                @pl.function
                def main(
                    self, x: pl.Tensor[[32, 64], pl.BF16], out: pl.Tensor[[32, 64], pl.BF16]
                ) -> pl.Tensor[[32, 64], pl.BF16]:
                    for b0 in pl.parallel(0, 32, 16):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            no_return(x, out, b0)
                    return out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
