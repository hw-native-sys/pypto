# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Python IR printer."""

import pypto
import pypto.language as pl
import pytest
from pypto import DataType, ir


class TestPythonPrinterProgram:
    """Tests for Python printer with Program nodes."""

    def test_print_empty_program(self):
        """Test printing an empty program."""
        span = ir.Span.unknown()
        program = ir.Program([], "EmptyProgram", span)

        code = pypto.ir.python_print(program)

        assert "@pl.program" in code
        assert "class EmptyProgram:" in code

    def test_print_program_with_single_function(self):
        """Test printing a program with a single function."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        add_expr = ir.Add(x, y, DataType.INT64, span)
        assign = ir.AssignStmt(x, add_expr, span)
        func = ir.Function("add", [x, y], [ir.ScalarType(DataType.INT64)], assign, span)
        program = ir.Program([func], "SingleFunc", span)

        code = pypto.ir.python_print(program)

        assert "@pl.program" in code
        assert "class SingleFunc:" in code
        assert "@pl.function" in code
        assert "def add(self," in code  # Should have self parameter
        assert "x: pl.INT64" in code or "x: pl.INT64" in code

    def test_print_program_with_multiple_functions(self):
        """Test printing a program with multiple functions."""
        span = ir.Span.unknown()

        # Create first function
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        body1 = ir.AssignStmt(x1, x1, span)
        func1 = ir.Function("func1", [x1], [ir.ScalarType(DataType.INT64)], body1, span)

        # Create second function
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        body2 = ir.AssignStmt(x2, x2, span)
        func2 = ir.Function("func2", [x2], [ir.ScalarType(DataType.INT64)], body2, span)

        program = ir.Program([func1, func2], "MultiFunc", span)

        code = pypto.ir.python_print(program)

        assert "@pl.program" in code
        assert "class MultiFunc:" in code
        assert code.count("@pl.function") == 2
        assert "def func1(self," in code
        assert "def func2(self," in code

    def test_print_program_methods_have_self(self):
        """Test that printed methods include self parameter."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        z = ir.Var("z", ir.ScalarType(DataType.INT32), span)
        add_expr = ir.Add(x, y, DataType.INT32, span)
        assign = ir.AssignStmt(z, add_expr, span)
        func = ir.Function("my_func", [x, y], [ir.ScalarType(DataType.INT32)], assign, span)
        program = ir.Program([func], "TestProgram", span)

        code = pypto.ir.python_print(program)

        # Verify self is the first parameter
        assert "def my_func(self, x:" in code

    def test_print_program_with_cross_function_calls(self):
        """Test that cross-function calls print as self.method_name()."""
        span = ir.Span.unknown()

        # Create helper function
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        helper_body = ir.AssignStmt(x, x, span)
        helper = ir.Function("helper", [x], [ir.ScalarType(DataType.INT64)], helper_body, span)

        # Create program to get GlobalVar
        temp_program = ir.Program([helper], "TempProgram", span)
        helper_gvar = temp_program.get_global_var("helper")
        assert helper_gvar is not None

        # Create main function that calls helper
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        call = ir.Call(helper_gvar, [y], span)
        main_body = ir.AssignStmt(y, call, span)
        main = ir.Function("main", [y], [ir.ScalarType(DataType.INT64)], main_body, span)

        # Create final program with both functions
        program = ir.Program([helper, main], "WithCalls", span)

        code = pypto.ir.python_print(program)

        # Verify cross-function call is printed with self
        assert "self.helper(" in code

    def test_standalone_function_no_self(self):
        """Test that standalone Function printing doesn't add self."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        body = ir.AssignStmt(x, x, span)
        func = ir.Function("standalone", [x], [ir.ScalarType(DataType.INT64)], body, span)

        code = pypto.ir.python_print(func)

        # Standalone functions should NOT have self
        assert "def standalone(x:" in code or "def standalone(x :" in code
        assert "def standalone(self," not in code

    def test_roundtrip_program_parse_print_parse(self):
        """Test that parse → print → parse produces equivalent IR."""

        @pl.program
        class Original:
            @pl.function
            def add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.op.add(x, 1.0)
                return result

        # Print to code
        code = pypto.ir.python_print(Original)

        # Re-parse
        reparsed = pl.parse_program(code)

        # Verify structural properties match
        assert isinstance(reparsed, ir.Program)
        assert reparsed.name == Original.name
        assert len(reparsed.functions) == len(Original.functions)

    def test_roundtrip_with_cross_function_calls(self):
        """Test round-trip preserves cross-function calls."""

        @pl.program
        class WithCalls:
            @pl.function
            def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = pl.op.mul(x, x)
                return result

            @pl.function
            def sum_of_squares(
                self, a: pl.Tensor[[1], pl.INT32], b: pl.Tensor[[1], pl.INT32]
            ) -> pl.Tensor[[1], pl.INT32]:
                a_sq: pl.Tensor[[1], pl.INT32] = self.square(a)
                b_sq: pl.Tensor[[1], pl.INT32] = self.square(b)
                result: pl.Tensor[[1], pl.INT32] = pl.op.add(a_sq, b_sq)
                return result

        # Print
        code = pypto.ir.python_print(WithCalls)

        # Verify printed code has self.square() calls
        assert "self.square(a)" in code
        assert "self.square(b)" in code

        # Re-parse
        reparsed = pl.parse_program(code)

        assert isinstance(reparsed, ir.Program)
        assert len(reparsed.functions) == 2

    def test_printed_program_is_valid_python(self):
        """Test that printed program code is syntactically valid Python."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        body = ir.AssignStmt(x, x, span)
        func = ir.Function("test", [x], [ir.ScalarType(DataType.INT64)], body, span)
        program = ir.Program([func], "ValidSyntax", span)

        code = pypto.ir.python_print(program)

        # Try to compile it as Python code (will raise SyntaxError if invalid)
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Printed code has invalid Python syntax: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
