# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Python IR printer."""

import ast

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.printer import python_print
from pypto.language.parser.expr_evaluator import ExprEvaluator
from pypto.language.parser.type_resolver import TypeResolver


class TestPythonPrinterProgram:
    """Tests for Python printer with Program nodes."""

    def test_print_empty_program(self):
        """Test printing an empty program."""
        span = ir.Span.unknown()
        program = ir.Program([], "EmptyProgram", span)

        code = program.as_python()

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

        code = program.as_python()

        assert "@pl.program" in code
        assert "class SingleFunc:" in code
        assert "@pl.function" in code
        assert "def add(self," in code  # Should have self parameter
        assert "x: pl.Scalar[pl.INT64]" in code

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

        code = program.as_python()

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

        code = program.as_python()

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

        code = program.as_python()

        # Verify cross-function call is printed with self
        assert "self.helper(" in code

    def test_standalone_function_no_self(self):
        """Test that standalone Function printing doesn't add self."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        body = ir.AssignStmt(x, x, span)
        func = ir.Function("standalone", [x], [ir.ScalarType(DataType.INT64)], body, span)

        code = func.as_python()

        # Standalone functions should NOT have self
        assert "def standalone(x:" in code or "def standalone(x :" in code
        assert "def standalone(self," not in code

    def test_roundtrip_program_parse_print_parse(self):
        """Test that parse → print → parse produces equivalent IR."""

        @pl.program
        class Original:
            @pl.function
            def add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        # Print to code
        code = Original.as_python()

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
                result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
                return result

            @pl.function
            def sum_of_squares(
                self, a: pl.Tensor[[1], pl.INT32], b: pl.Tensor[[1], pl.INT32]
            ) -> pl.Tensor[[1], pl.INT32]:
                a_sq: pl.Tensor[[1], pl.INT32] = self.square(a)
                b_sq: pl.Tensor[[1], pl.INT32] = self.square(b)
                result: pl.Tensor[[1], pl.INT32] = pl.add(a_sq, b_sq)
                return result

        # Print
        code = WithCalls.as_python()

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

        code = program.as_python()

        # Try to compile it as Python code (will raise SyntaxError if invalid)
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Printed code has invalid Python syntax: {e}")

    def test_print_alloc_memref_names_in_tile_annotations(self):
        """Use tile.alloc names in tile annotations when printing function bodies."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(x, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(tile_a, tile_a)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(tile_b, [0, 0], out)
                return result

        after = passes.allocate_memory_addr()(passes.init_mem_ref()(Before))
        code = after.as_python()

        assert "pl.MemRefType = pl.tile.alloc(" in code
        assert "pl.Tile[[64, 64], pl.FP32, mem_vec_" in code
        assert "pl.Tile[[64, 64], pl.FP32, pl.MemRef(" not in code


class TestPythonPrinterConstDtypeRoundtrip:
    """Tests for round-trip of constants with non-default dtypes."""

    def test_roundtrip_const_int_non_default_dtype(self):
        """Test round-trip: ConstInt with INT32 dtype survives print → parse."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x, pl.const(42, pl.INT32))
                return y

        code = Before.as_python()
        assert "pl.const(42, pl.INT32)" in code

        After = pl.parse_program(code)
        ir.assert_structural_equal(After, Before)

    def test_roundtrip_const_float_non_default_dtype(self):
        """Test round-trip: ConstFloat with FP16 dtype survives print → parse."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
                y: pl.Tensor[[64], pl.FP16] = pl.tensor.add(x, pl.const(1.0, pl.FP16))
                return y

        code = Before.as_python()
        assert "pl.const(1.0, pl.FP16)" in code

        After = pl.parse_program(code)
        ir.assert_structural_equal(After, Before)

    def test_roundtrip_default_dtype_constants_bare(self):
        """Test that default-typed constants print as bare values and round-trip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x, 1.0)
                return y

        code = Before.as_python()
        # Default FP32 should print as bare 1.0
        assert "pl.const(" not in code

        After = pl.parse_program(code)
        ir.assert_structural_equal(After, Before)

    def test_roundtrip_negative_typed_constant(self):
        """Test round-trip with negative typed constant."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
                y: pl.Tensor[[64], pl.FP16] = pl.tensor.add(x, pl.const(-2.5, pl.FP16))
                return y

        code = Before.as_python()
        assert "pl.const(-2.5, pl.FP16)" in code

        After = pl.parse_program(code)
        ir.assert_structural_equal(After, Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTileViewTensorViewPrinting:
    """Printer output fix for Python 3.10 keyword subscript syntax (Fix #323)."""

    def test_tiletype_with_tileview_no_keyword_subscript(self):
        span = ir.Span.unknown()
        tile_view = ir.TileView()
        tile_view.valid_shape = [ir.ConstInt(32, DataType.INT64, span)]
        memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INT64, span), 256, 0)
        tile_type = ir.TileType(
            [64], DataType.FP32, memref=memref, tile_view=tile_view, memory_space=ir.MemorySpace.Vec
        )

        printed = ir.python_print_type(tile_type)

        assert "tile_view=" not in printed  # must not use keyword subscript syntax
        assert "pl.TileView(" in printed  # must be emitted as a positional call

    def test_printed_type_is_valid_python_syntax(self):
        span = ir.Span.unknown()
        tile_view = ir.TileView()
        tile_view.valid_shape = [ir.ConstInt(32, DataType.INT64, span)]
        memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INT64, span), 256, 0)
        tile_type = ir.TileType(
            [64], DataType.FP32, memref=memref, tile_view=tile_view, memory_space=ir.MemorySpace.Vec
        )

        printed = "import pypto.language as pl\nresult = " + ir.python_print_type(tile_type)
        compile(printed, "<string>", "exec")  # must not raise SyntaxError

    def test_tensorview_always_emitted_when_present(self):
        tensor_view = ir.TensorView()  # all-default fields
        tensor_type = ir.TensorType([64], DataType.FP32, tensor_view=tensor_view)

        printed = ir.python_print_type(tensor_type)
        assert "pl.TensorView()" in printed  # all-default fields must still be emitted

    def test_tileview_tensorview_parseable_by_type_resolver(self):
        span = ir.Span.unknown()
        tile_view = ir.TileView()
        tile_view.valid_shape = [ir.ConstInt(32, DataType.INT64, span)]
        memref = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 256, 0)
        original = ir.TileType(
            [64], DataType.FP32, memref=memref, tile_view=tile_view, memory_space=ir.MemorySpace.DDR
        )

        printed = ir.python_print_type(original)
        node = ast.parse(printed, mode="eval").body
        resolver = TypeResolver(expr_evaluator=ExprEvaluator(closure_vars={}))
        reparsed = resolver.resolve_type(node)

        assert isinstance(reparsed, ir.TileType)
        assert reparsed.tile_view is not None


class TestDynVarAndSSARename:
    """dyn var collection and SSA var deduplication in printer."""

    def test_dyn_var_declared_in_header(self):
        N = pl.dynamic("N")

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[N], pl.FP32]) -> pl.Tensor[[N], pl.FP32]:
                return x

        src = Prog.as_python()
        assert 'N = pl.dynamic("N")' in src

    def test_dyn_var_same_name_different_identity_disambiguated(self):
        """Two distinct Var objects with same name_hint_ get disambiguated (issue #618)."""
        span = ir.Span.unknown()
        c64 = ir.ConstInt(64, DataType.INT64, span)
        c128 = ir.ConstInt(128, DataType.INT64, span)
        # Two distinct ir.Var objects both named "M"
        var_m1 = ir.Var("M", ir.ScalarType(DataType.INDEX), span)
        var_m2 = ir.Var("M", ir.ScalarType(DataType.INDEX), span)
        assert var_m1 is not var_m2

        x = ir.Var("x", ir.TensorType([var_m1, c64], DataType.FP32), span)
        ret1 = ir.ReturnStmt([x], span)
        func1 = ir.Function("func1", [x], [ir.TensorType([var_m1, c64], DataType.FP32)], ret1, span)

        y = ir.Var("y", ir.TensorType([var_m2, c128], DataType.FP32), span)
        ret2 = ir.ReturnStmt([y], span)
        func2 = ir.Function("func2", [y], [ir.TensorType([var_m2, c128], DataType.FP32)], ret2, span)

        prog = ir.Program([func1, func2], "TestProg", span)
        src = prog.as_python()

        # Both distinct vars should appear as separate pl.dynamic() declarations
        lines = [line for line in src.splitlines() if "pl.dynamic" in line]
        assert len(lines) == 2, f"Expected 2 pl.dynamic() declarations, got {len(lines)}: {lines}"
        # One should be M, the other M_1
        assert 'M = pl.dynamic("M")' in src
        assert 'M_1 = pl.dynamic("M_1")' in src

    def test_dyn_var_unique_names_not_disambiguated(self):
        """Distinct Var objects with different names are not affected (issue #618)."""
        span = ir.Span.unknown()
        var_m = ir.Var("M", ir.ScalarType(DataType.INDEX), span)
        var_n = ir.Var("N", ir.ScalarType(DataType.INDEX), span)

        x = ir.Var("x", ir.TensorType([var_m, var_n], DataType.FP32), span)
        ret = ir.ReturnStmt([x], span)
        func = ir.Function("main", [x], [ir.TensorType([var_m, var_n], DataType.FP32)], ret, span)
        prog = ir.Program([func], "TestProg", span)
        src = prog.as_python()

        assert 'M = pl.dynamic("M")' in src
        assert 'N = pl.dynamic("N")' in src
        # No suffixed names
        assert "M_1" not in src
        assert "N_1" not in src

    def test_ssa_shadowed_vars_get_unique_names(self):
        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(4):
                    x = pl.add(x, 1.0)
                return x

        after_ssa = passes.convert_to_ssa()(Prog)
        src = python_print(after_ssa)
        lhs_names = [
            line.split(":")[0].strip() for line in src.splitlines() if ": pl." in line and "=" in line
        ]
        assert len(lhs_names) == len(set(lhs_names))


class TestOpOutputNormalization:
    """Op-specific printer output normalization."""

    def test_tensor_add_scalar_prints_as_adds(self):
        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return pl.add(x, 1.0)

        src = python_print(Prog)
        assert "pl.tensor.adds(" in src
        assert "pl.tensor.add(" not in src

    def test_tile_full_dtype_as_keyword(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tile[[64], pl.FP32]) -> pl.Tile[[64], pl.FP32]:
                y: pl.Tile[[64], pl.FP32] = pl.tile.full([64], dtype=pl.FP32, value=0.0)
                return y

        src = python_print(Prog)
        assert "dtype=pl.FP32" in src
        assert "value=" in src
