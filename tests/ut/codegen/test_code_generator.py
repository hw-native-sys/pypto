# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for CodeGenerator class."""

from pypto import DataType, ir
from pypto.pypto_core import codegen, passes
from pypto.ir.op import block
from pypto.ir.builder import IRBuilder


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
            input_a = f.param("input_a", ir.TensorType([64, 64], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([64, 64], DataType.FP32))
            output = f.param("output", ir.TensorType([64, 64], DataType.FP32))
            f.return_type(ir.TensorType([64, 64], DataType.FP32))

            # Constants for tile
            tile_height = 64
            tile_width = 64

            # Load (should infer input_a/b as DDR)
            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, tile_height, tile_width))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, tile_height, tile_width))

            # Compute (UB)
            tile_sum = ib.let("tile_sum", block.add(tile_a, tile_b))

            # Store (should infer output as DDR)
            result = ib.let("result", block.store(tile_sum, 0, 0, tile_height, tile_width, output))

            ib.return_stmt(result)

        func = f.get_result()

        func = passes.InitMemRefPass().run(func)
        func = passes.BasicMemoryReusePass().run(func)
        func = passes.InsertSyncPass().run(func)

        generator = codegen.CodeGenerator()
        code = generator.Generate(func)
        print(code)

        # Verify GlobalTensor declarations are generated
        assert "GlobalTensor<float" in code
        assert "input_aGlobalType" in code
        assert "input_bGlobalType" in code
        assert "outputGlobalType" in code

        # Verify Tile type definitions are generated
        assert "Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1>" in code
        assert "tile_aType tile_a(64, 64)" in code
        assert "tile_bType tile_b(64, 64)" in code
        assert "tile_sumType tile_sum(64, 64)" in code

        # Verify instructions are generated
        assert "TLOAD(tile_a, input_aGlobal)" in code
        assert "TLOAD(tile_b, input_bGlobal)" in code
        assert "TADD(tile_sum, tile_a, tile_b)" in code
        assert "TSTORE(outputGlobal, tile_sum)" in code

