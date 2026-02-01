# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for CCECodegen class."""

from pypto import DataType, ir
from pypto.ir.builder import IRBuilder
from pypto.ir.op import block
from pypto.ir.pass_manager import PassManager
from pypto.pypto_core import codegen


class TestCCECodegenBasics:
    """Test basic CCECodegen functionality."""

    def test_create_cce_codegen(self):
        """Test creating a CCECodegen instance."""
        generator = codegen.CCECodegen()
        assert generator is not None

    def test_tadds_example(self):
        """Test generating code for a simple tensor addition with scalar example."""
        ib = IRBuilder()

        with ib.function("test_tadds_simple") as f:
            # Define input and output parameters (Global Tensors -> DDR)
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.ScalarType(DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            # Constants for tile
            tile_height = 128
            tile_width = 128

            # Load (should infer input_a as DDR)
            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))

            # Compute (UB)
            tile_sum = ib.let("tile_sum", block.adds(tile_a, input_b))

            # Store (should infer output as DDR)
            result = ib.let("result", block.store(tile_sum, 0, 0, tile_height, tile_width, output))

            ib.return_stmt(result)

        func = f.get_result()
        program = ir.Program([func], "test_tadd_simple", ir.Span.unknown())

        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)

        generator = codegen.CCECodegen()
        files = generator.generate(optimized_program)
        kernel_name = list(optimized_program.functions.values())[0].name
        code = files["kernels/" + kernel_name + ".cpp"]

        # Verify function parameters unpacking and declarations are generated
        assert "GlobalTensor<float" in code
        assert "__gm__ float* input_a" in code
        assert "float input_b" in code
        assert "outputGlobalType" in code

        # Verify Tile type definitions are generated
        assert "Tile<TileType::Vec, float, 128, 128, BLayout::RowMajor, -1, -1>" in code
        assert "tile_aType tile_a(128, 128)" in code
        assert "tile_sumType tile_sum(128, 128)" in code
        assert "TASSIGN(tile_sum, 0x10000)" in code

        # Verify instructions are generated
        assert "TLOAD(tile_a, input_aGlobal)" in code
        assert "TADDS(tile_sum, tile_a, input_b)" in code
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
        program = ir.Program([func], "test_simple_for", ir.Span.unknown())
        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/test_simple_for.cpp"]

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
        program = ir.Program([func], "test_nested_for", ir.Span.unknown())
        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/test_nested_for.cpp"]

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
        program = ir.Program([func], "test_if", ir.Span.unknown())

        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/test_if.cpp"]

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
        program = ir.Program([func], "test_if_else", ir.Span.unknown())

        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/test_if_else.cpp"]

        # Verify if-else structure
        assert "if ((a < b)) {" in code or "if (a < b) {" in code
        assert "} else {" in code
        assert "auto x = 1;" in code
        assert "auto y = 2;" in code
