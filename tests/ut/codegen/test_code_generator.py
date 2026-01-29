# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for CodeGenerator class."""

import pytest

from pypto import DataType, ir
from pypto.pypto_core import codegen, passes
from pypto.ir.pass_manager import PassManager
from pypto.ir.op import block
from pypto.ir.builder import IRBuilder
from pypto.pypto_core import ir as core_ir


class TestCodeGeneratorBasics:
    """Test basic CodeGenerator functionality."""

    def test_create_code_generator(self):
        """Test creating a CodeGenerator instance."""
        generator = codegen.CodeGenerator()
        assert generator is not None

    def test_tadd_example(self):
        """Test generating code for a simple tensor addition example."""
        ib = IRBuilder()

        with ib.function("test_tadd_simple") as f:
            # Define input and output parameters (Global Tensors -> DDR)
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            # Constants for tile
            tile_height = 128
            tile_width = 128

            # Load (should infer input_a/b as DDR)
            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

            # Compute (UB)
            tile_sum = ib.let("tile_sum", block.add(tile_a, tile_b))

            # Store (should infer output as DDR)
            result = ib.let("result", block.store(tile_sum, 0, 0, tile_height, tile_width, output))

            ib.return_stmt(result)

        func = f.get_result()
        program = ir.Program([func], "test_tadd_simple", ir.Span.unknown())

        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)
        optimized_func = list(optimized_program.functions.values())[0]

        generator = codegen.CodeGenerator()
        code = generator.Generate(optimized_func)
        print(code)

        # Verify GlobalTensor declarations are generated
        assert "GlobalTensor<float" in code
        assert "input_aGlobalType" in code
        assert "input_bGlobalType" in code
        assert "outputGlobalType" in code

        # Verify Tile type definitions are generated
        assert "Tile<TileType::Vec, float, 128, 128, BLayout::RowMajor, -1, -1>" in code
        assert "tile_aType tile_a(128, 128)" in code
        assert "tile_bType tile_b(128, 128)" in code
        assert "tile_sumType tile_sum(128, 128)" in code

        # Verify instructions are generated
        assert "TLOAD(tile_a, input_aGlobal)" in code
        assert "TLOAD(tile_b, input_bGlobal)" in code
        assert "TADD(tile_sum, tile_a, tile_b)" in code
        assert "TSTORE(outputGlobal, tile_sum)" in code


class TestControlFlowCodegen:
    """Test control flow statement code generation."""

    def test_simple_for_loop(self):
        """Test simple for loop without iter_args."""
        ib = IRBuilder()

        with ib.function("test_simple_for") as f:
            # Parameters
            input_tensor = f.param("input", ir.TensorType([128, 64], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 64], DataType.FP32))
            f.return_type(ir.TensorType([128, 64], DataType.FP32))

            # Loop variable
            i = ib.var("i", ir.ScalarType(DataType.INT32))

            # Simple for loop: for i in range(0, 4, 1)
            with ib.for_loop(i, 0, 4, 1):
                # Load tile inside loop
                tile_x = ib.let("tile_x", block.load(input_tensor, i, 0, 32, 64))
                # Store tile back
                result = ib.let("result", block.store(tile_x, i, 0, 32, 64, output_tensor))

            ib.return_stmt(result)

        func = f.get_result()
        generator = codegen.CodeGenerator()
        code = generator.Generate(func)
        # print(code)

        # Verify for loop structure
        assert "for (int64_t i = 0; i < 4; i += 1) {" in code
        assert "TLOAD(tile_x, inputGlobal)" in code
        assert "TSTORE(outputGlobal, tile_x)" in code

    def test_nested_for_loops(self):
        """Test nested for loops."""
        ib = IRBuilder()

        with ib.function("test_nested_for") as f:
            # Parameters
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            # Outer loop variable
            i = ib.var("i", ir.ScalarType(DataType.INT32))
            # Inner loop variable
            j = ib.var("j", ir.ScalarType(DataType.INT32))

            # Nested for loops
            with ib.for_loop(i, 0, 4, 1):
                with ib.for_loop(j, 0, 4, 1):
                    # Load tile inside inner loop
                    tile_x = ib.let("tile_x", block.load(input_tensor, i, j, 32, 32))
                    # Store tile back
                    result = ib.let("result", block.store(tile_x, i, j, 32, 32, output_tensor))

            ib.return_stmt(result)

        func = f.get_result()
        generator = codegen.CodeGenerator()
        code = generator.Generate(func)

        # Verify nested loop structure
        assert "for (int64_t i = 0; i < 4; i += 1) {" in code
        assert "for (int64_t j = 0; j < 4; j += 1) {" in code
        # Verify proper nesting (inner loop should appear after outer loop)
        assert code.index("for (int64_t i") < code.index("for (int64_t j")

    def test_if_statement_simple(self):
        """Test simple if statement code generation."""
        span = ir.Span.unknown()

        # Build if statement directly using IR nodes
        condition = ir.ConstBool(True, span)

        # Then body: just an assignment
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        then_assign = ir.AssignStmt(x, ir.ConstInt(5, DataType.INT32, span), span)

        # Create if statement without else
        if_stmt = ir.IfStmt(condition, then_assign, None, [], span)

        # Create a simple function with the if statement
        ret_stmt = ir.ReturnStmt([], span)
        seq = ir.SeqStmts([if_stmt, ret_stmt], span)

        func = ir.Function("test_if", [], [ir.TensorType([1], DataType.FP32)], seq, span)

        generator = codegen.CodeGenerator()
        code = generator.Generate(func)

        # Verify if structure
        assert "if (true) {" in code or "if (1) {" in code
        assert "auto x = 5;" in code

    def test_if_else_statement(self):
        """Test if-else statement code generation."""
        span = ir.Span.unknown()

        # Build condition
        a = ir.Var("a", ir.ScalarType(DataType.INT32), span)
        b = ir.Var("b", ir.ScalarType(DataType.INT32), span)
        condition = ir.Lt(a, b, DataType.INT32, span)

        # Then body
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        then_assign = ir.AssignStmt(x, ir.ConstInt(1, DataType.INT32, span), span)

        # Else body
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        else_assign = ir.AssignStmt(y, ir.ConstInt(2, DataType.INT32, span), span)

        # Create if-else statement
        if_stmt = ir.IfStmt(condition, then_assign, else_assign, [], span)

        # Create function
        # First assign a and b
        assign_a = ir.AssignStmt(a, ir.ConstInt(5, DataType.INT32, span), span)
        assign_b = ir.AssignStmt(b, ir.ConstInt(10, DataType.INT32, span), span)
        ret_stmt = ir.ReturnStmt([], span)
        seq = ir.SeqStmts([assign_a, assign_b, if_stmt, ret_stmt], span)

        func = ir.Function("test_if_else", [], [ir.TensorType([1], DataType.FP32)], seq, span)

        generator = codegen.CodeGenerator()
        code = generator.Generate(func)

        # Verify if-else structure
        assert "if ((a < b)) {" in code or "if (a < b) {" in code
        assert "} else {" in code
        assert "auto x = 1;" in code
        assert "auto y = 2;" in code


class TestRealExampleCodegen:
    """Test code generation for real examples."""

    @pytest.mark.skip(reason="Requires fixing tensor op registration")
    def test_flash_attention(self):
        from examples.ir_builder.flash_attention_builder import build_flash_attention
        func = build_flash_attention()
        program = ir.Program([func], "test_flash_attention", ir.Span.unknown())
        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)
        optimized_func = list(optimized_program.functions.values())[0]
        print(optimized_func)

        generator = codegen.CodeGenerator()
        code = generator.Generate(optimized_func)
        print(code)

    def test_sinh_example(self):
        from examples.ir_builder.sinh_taylor_codegen import build_sinh_ir
        program = build_sinh_ir()

        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)
        optimized_func = list(optimized_program.functions.values())[0]
        print(optimized_func)

        generator = codegen.CodeGenerator()
        code = generator.Generate(optimized_func)
        print(code)
