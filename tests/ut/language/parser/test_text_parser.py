# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for pl.parse() and pl.loads() text parsing functions."""

import os
import tempfile

import pypto
import pypto.language as pl
import pytest
from pypto import ir


class TestParse:
    """Tests for pl.parse() function."""

    def test_parse_simple_function_with_import(self):
        """Test parsing simple function with import statement."""
        code = """
import pypto.language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "add_one"
        assert len(func.params) == 1
        assert len(func.return_types) == 1

    def test_parse_simple_function_without_import(self):
        """Test parsing simple function without import statement (auto-injected)."""
        code = """
@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "add_one"
        assert len(func.params) == 1
        assert len(func.return_types) == 1

    def test_parse_multiple_params(self):
        """Test parsing function with multiple parameters."""
        code = """
@pl.function
def add_three(
    x: pl.Tensor[[64], pl.FP32],
    y: pl.Tensor[[64], pl.FP32],
    z: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    temp: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
    result: pl.Tensor[[64], pl.FP32] = pl.add(temp, z)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "add_three"
        assert len(func.params) == 3

    def test_parse_with_for_loop(self):
        """Test parsing function with for loop control flow."""
        code = """
@pl.function
def sum_loop(x: pl.Tensor[[10], pl.FP32]) -> pl.Tensor[[10], pl.FP32]:
    init_sum: pl.Tensor[[10], pl.FP32] = pl.create_tensor([10], dtype=pl.FP32)
    for i, (running_sum,) in pl.range(5, init_values=(init_sum,)):
        new_sum: pl.Tensor[[10], pl.FP32] = pl.add(running_sum, x)
        result = pl.yield_(new_sum)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "sum_loop"

    def test_parse_with_multiple_statements(self):
        """Test parsing function with multiple statements."""
        code = """
@pl.function
def multi_op(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    a: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    b: pl.Tensor[[64], pl.FP32] = pl.add(a, 1.0)
    c: pl.Tensor[[64], pl.FP32] = pl.sub(b, 0.5)
    return c
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "multi_op"

    def test_parse_no_function_error(self):
        """Test that parsing code with no function raises ValueError."""
        code = """
x = 42
y = x + 1
"""
        with pytest.raises(ValueError, match="No @pl.function or @pl.program found"):
            pl.parse(code)

    def test_parse_multiple_functions_error(self):
        """Test that parsing code with multiple functions raises ValueError."""
        code = """
@pl.function
def func1(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x

@pl.function
def func2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ValueError, match="Multiple functions/programs found"):
            pl.parse(code)

    def test_parse_syntax_error(self):
        """Test that parsing code with syntax error raises SyntaxError."""
        code = """
@pl.function
def bad_syntax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x +
"""
        with pytest.raises(SyntaxError, match="Failed to compile code"):
            pl.parse(code)

    def test_parse_with_custom_filename(self):
        """Test that custom filename is used in error reporting."""
        code = """
@pl.function
def bad_func(x):
    return x
"""
        with pytest.raises(pl.parser.ParserError):
            pl.parse(code, filename="custom_file.py")

    def test_parse_from_import_variant(self):
        """Test parsing with 'from pypto import language as pl' variant."""
        code = """
from pypto import language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "add_one"

    def test_parse_with_different_dtypes(self):
        """Test parsing function with various data types."""
        code = """
@pl.function
def cast_op(
    fp16: pl.Tensor[[64], pl.FP16],
    fp32: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(
        pl.cast(fp16, target_type=pl.FP32), fp32
    )
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "cast_op"
        assert len(func.params) == 2


class TestLoad:
    """Tests for pl.loads() function."""

    def test_load_simple_function(self):
        """Test loading function from a file."""
        code = """
import pypto.language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.loads(temp_path)
            assert isinstance(func, ir.Function)
            assert func.name == "add_one"
            assert len(func.params) == 1
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_load_function_without_import(self):
        """Test loading function without import (auto-injected)."""
        code = """
@pl.function
def multiply(x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
    result: pl.Tensor[[32, 32], pl.FP32] = pl.mul(x, 2.0)
    return result
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.loads(temp_path)
            assert isinstance(func, ir.Function)
            assert func.name == "multiply"
        finally:
            os.unlink(temp_path)

    def test_load_file_not_found(self):
        """Test that loading non-existent file raises OSError."""
        with pytest.raises(OSError):
            pl.loads("/non/existent/path/file.py")

    def test_load_complex_function(self):
        """Test loading a complex function with control flow."""
        code = """
import pypto.language as pl

@pl.function
def complex_op(
    x: pl.Tensor[[64, 128], pl.FP16],
    y: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    # Multiple operations
    temp1: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, y)
    temp2: pl.Tensor[[64, 128], pl.FP16] = pl.mul(temp1, 2.0)
    result: pl.Tensor[[64, 128], pl.FP16] = pl.sub(temp2, x)
    return result
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.loads(temp_path)
            assert isinstance(func, ir.Function)
            assert func.name == "complex_op"
            assert len(func.params) == 2
        finally:
            os.unlink(temp_path)

    def test_load_preserves_filename_in_errors(self):
        """Test that errors reference the correct file path."""
        code = """
@pl.function
def bad_func(x):
    return x
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        with pytest.raises(pl.parser.ParserError):
            pl.loads(temp_path)
        os.unlink(temp_path)


class TestIntegration:
    """Integration tests for parse/load with existing decorator."""

    def test_decorator_and_parse_produce_same_result(self):
        """Test that @pl.function decorator and pl.parse produce equivalent results."""

        # Using decorator
        @pl.function
        def func_decorator(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        # Using parse
        code = """
@pl.function
def func_parse(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return result
"""
        func_parsed = pl.parse(code)

        # Both should be ir.Function objects with same structure
        assert isinstance(func_decorator, ir.Function)
        assert isinstance(func_parsed, ir.Function)
        assert len(func_decorator.params) == len(func_parsed.params)
        assert len(func_decorator.return_types) == len(func_parsed.return_types)

    def test_serialization_of_parsed_function(self):
        """Test that parsed functions can be serialized and deserialized."""
        code = """
@pl.function
def serializable(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        func = pl.parse(code)

        # Should be able to serialize
        data = pypto.ir.serialize(func)
        assert len(data) > 0

        # Should be able to deserialize
        restored = pypto.ir.deserialize(data)
        assert isinstance(restored, ir.Function)
        assert restored.name == "serializable"


class TestParseProgram:
    """Tests for pl.parse_program() function."""

    def test_parse_simple_program_with_import(self):
        """Test parsing simple program with import statement."""
        code = """
import pypto.language as pl

@pl.program
class SimpleProgram:
    @pl.function
    def add_one(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
        return result
"""
        program = pl.parse_program(code)
        assert isinstance(program, ir.Program)
        assert program.name == "SimpleProgram"
        assert len(program.functions) == 1

    def test_parse_simple_program_without_import(self):
        """Test parsing simple program without import statement (auto-injected)."""
        code = """
@pl.program
class SimpleProgram:
    @pl.function
    def add_one(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
        return result
"""
        program = pl.parse_program(code)
        assert isinstance(program, ir.Program)
        assert program.name == "SimpleProgram"
        assert len(program.functions) == 1

    def test_parse_program_with_multiple_functions(self):
        """Test parsing program with multiple functions."""
        code = """
@pl.program
class MathOps:
    @pl.function
    def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
        return result

    @pl.function
    def cube(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        x_sq: pl.Tensor[[1], pl.INT32] = self.square(x)
        result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x_sq)
        return result
"""
        program = pl.parse_program(code)
        assert isinstance(program, ir.Program)
        assert program.name == "MathOps"
        assert len(program.functions) == 2

    def test_parse_program_with_cross_function_calls(self):
        """Test parsing program with cross-function calls."""
        code = """
@pl.program
class CallTest:
    @pl.function
    def helper(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.mul(x, 2)
        return result

    @pl.function
    def caller(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = self.helper(x)
        return result
"""
        program = pl.parse_program(code)
        assert isinstance(program, ir.Program)
        assert len(program.functions) == 2

        # Verify self parameter was stripped
        caller_func = program.get_function("caller")
        assert caller_func is not None
        assert len(caller_func.params) == 1
        assert caller_func.params[0].name == "x"

    def test_parse_program_no_program_error(self):
        """Test that code without @pl.program raises error."""
        code = """
@pl.function
def standalone(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
        with pytest.raises(ValueError, match="Expected @pl.program but found @pl.function"):
            pl.parse_program(code)

    def test_parse_program_multiple_programs_error(self):
        """Test that multiple @pl.program classes raises error."""
        code = """
@pl.program
class Program1:
    @pl.function
    def func1(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        return x

@pl.program
class Program2:
    @pl.function
    def func2(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        return x
"""
        with pytest.raises(ValueError, match="Multiple functions/programs found"):
            pl.parse_program(code)

    def test_parse_program_syntax_error(self):
        """Test that syntax errors are properly reported."""
        code = """
@pl.program
class BadSyntax:
    @pl.function
    def broken(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        return x +
"""
        with pytest.raises(SyntaxError):
            pl.parse_program(code)


class TestLoadProgram:
    """Tests for pl.loads_program() function."""

    def test_load_program_from_file(self):
        """Test loading program from a file."""

        code = """
import pypto.language as pl

@pl.program
class FileProgram:
    @pl.function
    def add(self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
        return result
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            program = pl.loads_program(temp_path)
            assert isinstance(program, ir.Program)
            assert program.name == "FileProgram"
            assert len(program.functions) == 1
        finally:
            os.unlink(temp_path)

    def test_load_program_file_not_found(self):
        """Test that load_program raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            pl.loads_program("nonexistent_file.py")

    def test_parse_function_with_scalar_param(self):
        """Test parsing function with scalar parameter from text."""
        code = """
@pl.function
def add_scalar(x: pl.Tensor[[64], pl.FP32], scalar: pl.Scalar[pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, scalar)
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert len(func.params) == 2
        assert isinstance(func.params[1].type, ir.ScalarType)
        assert func.params[1].type.dtype == pl.FP32


class TestScalarRangeRoundTrip:
    """Tests for round-trip (print -> parse) of pl.range() with Scalar arguments."""

    def test_scalar_stop_roundtrip(self):
        """Test round-trip: pl.range(n) where n is Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        printed = pypto.ir.python_print(Before)
        # Verify the printed output contains the Scalar parameter in range
        assert "pl.Scalar[pl.INT64]" in printed
        assert "pl.range(" in printed
        assert "n" in printed

        # Re-parse the printed output
        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_start_stop_step_roundtrip(self):
        """Test round-trip: pl.range(0, n, s) where n, s are Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                n: pl.Scalar[pl.INT64],
                s: pl.Scalar[pl.INT64],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, n, s):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        printed = pypto.ir.python_print(Before)
        assert "pl.range(" in printed
        assert "n" in printed
        assert "s" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_expression_roundtrip(self):
        """Test round-trip: pl.range(n * 2) where n is Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n * 2):  # type: ignore[operator]
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        printed = pypto.ir.python_print(Before)
        assert "n * 2" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_range_with_iter_args_roundtrip(self):
        """Test round-trip: pl.range(n, init_values=(...)) where n is Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(n, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        printed = pypto.ir.python_print(Before)
        assert "pl.range(" in printed
        assert "init_values=" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_parallel_roundtrip(self):
        """Test round-trip: pl.parallel(n) where n is Scalar[INT64]."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.parallel(n):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        printed = pypto.ir.python_print(Before)
        assert "pl.parallel(" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_scalar_range_text_parse(self):
        """Test parsing Scalar range directly from text string."""
        code = """
@pl.function
def scalar_range_func(
    n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.range(n):
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return y
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "scalar_range_func"
        assert len(func.params) == 2
        assert isinstance(func.params[0].type, ir.ScalarType)

    def test_scalar_range_complex_expression_text_parse(self):
        """Test parsing Scalar range with complex expression from text string."""
        code = """
@pl.function
def complex_range(
    n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.range(0, n * 2 + 1, 1):
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    return y
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "complex_range"
