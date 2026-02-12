# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for @pl.program decorator."""

import pypto
import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError, UndefinedVariableError
from pypto.pypto_core import ir


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
    """Test round-trip: parse → print → parse."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
