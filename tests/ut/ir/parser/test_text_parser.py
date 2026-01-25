# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for pl.parse() and pl.load() text parsing functions."""

import os
import tempfile

import pypto
import pypto.language as pl
import pytest
from pypto.pypto_core import ir


class TestParse:
    """Tests for pl.parse() function."""

    def test_parse_simple_function_with_import(self):
        """Test parsing simple function with import statement."""
        code = """
import pypto.language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
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
    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
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
    temp: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, y)
    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(temp, z)
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
    init_sum: pl.Tensor[[10], pl.FP32] = pl.op.tensor.create([10], dtype=pl.FP32)
    for i, (running_sum,) in pl.range(5, init_values=[init_sum]):
        new_sum: pl.Tensor[[10], pl.FP32] = pl.op.tensor.add(running_sum, x)
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
    a: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 2.0)
    b: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(a, 1.0)
    c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.sub(b, 0.5)
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
        with pytest.raises(ValueError, match="No @pl.function decorated functions found"):
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
        with pytest.raises(ValueError, match="Multiple functions found"):
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
        with pytest.raises(pypto.ir.parser.ParserError):
            pl.parse(code, filename="custom_file.py")

    def test_parse_from_import_variant(self):
        """Test parsing with 'from pypto import language as pl' variant."""
        code = """
from pypto import language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
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
    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(
        pl.op.tensor.cast(fp16, target_type=pl.FP32), fp32
    )
    return result
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        assert func.name == "cast_op"
        assert len(func.params) == 2


class TestLoad:
    """Tests for pl.load() function."""

    def test_load_simple_function(self):
        """Test loading function from a file."""
        code = """
import pypto.language as pl

@pl.function
def add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
    return result
"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.load(temp_path)
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
    result: pl.Tensor[[32, 32], pl.FP32] = pl.op.tensor.mul(x, 2.0)
    return result
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.load(temp_path)
            assert isinstance(func, ir.Function)
            assert func.name == "multiply"
        finally:
            os.unlink(temp_path)

    def test_load_file_not_found(self):
        """Test that loading non-existent file raises OSError."""
        with pytest.raises(OSError):
            pl.load("/non/existent/path/file.py")

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
    temp1: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.add(x, y)
    temp2: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.mul(temp1, 2.0)
    result: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.sub(temp2, x)
    return result
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            func = pl.load(temp_path)
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

        with pytest.raises(pypto.ir.parser.ParserError):
            pl.load(temp_path)
        os.unlink(temp_path)


class TestIntegration:
    """Integration tests for parse/load with existing decorator."""

    def test_decorator_and_parse_produce_same_result(self):
        """Test that @pl.function decorator and pl.parse produce equivalent results."""

        # Using decorator
        @pl.function
        def func_decorator(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
            return result

        # Using parse
        code = """
@pl.function
def func_parse(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
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
