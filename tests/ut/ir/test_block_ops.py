# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for block operations."""

import pytest
from pypto.ir.builder import IRBuilder
from pypto.ir.op import block
from pypto.pypto_core import DataType, ir


class TestBlockMemoryOps:
    """Tests for block memory operations."""

    def test_ub_copy_in(self):
        """Test block.ub_copy_in operation."""
        ib = IRBuilder()

        with ib.function("test_ub_copy_in") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(
                ir.TileType(
                    [
                        ir.ConstInt(32, DataType.INT32, ir.Span.unknown()),
                        ir.ConstInt(32, DataType.INT32, ir.Span.unknown()),
                    ],
                    DataType.FP32,
                )
            )

            tile = ib.let("tile", block.ub_copy_in(input_tensor, 0, 0, 32, 32))
            ib.return_stmt(tile)

        func = f.get_result()
        assert func is not None
        assert "block.ub_copy_in" in str(func)

    def test_ub_copy_out(self):
        """Test block.ub_copy_out operation."""
        ib = IRBuilder()

        with ib.function("test_ub_copy_out") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile = ib.let("tile", block.ub_copy_in(input_tensor, 0, 0, 32, 32))
            result = ib.let("result", block.ub_copy_out(tile, 0, 0, 32, 32, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.ub_copy_out" in str(func)


class TestBlockElementwiseOps:
    """Tests for block element-wise operations."""

    def test_block_mul(self):
        """Test block.mul operation."""
        ib = IRBuilder()

        with ib.function("test_block_mul") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.ub_copy_in(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.mul(tile_a, tile_b))
            result = ib.let("result", block.ub_copy_out(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.mul" in str(func)

    def test_block_mul_scalar(self):
        """Test block.mul with scalar operation."""
        ib = IRBuilder()

        with ib.function("test_block_mul_scalar") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.mul(tile_a, 2.0))
            result = ib.let("result", block.ub_copy_out(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.mul" in str(func)

    def test_block_add(self):
        """Test block.add operation."""
        ib = IRBuilder()

        with ib.function("test_block_add") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.ub_copy_in(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.add(tile_a, tile_b))
            result = ib.let("result", block.ub_copy_out(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.add" in str(func)

    def test_block_div(self):
        """Test block.div operation."""
        ib = IRBuilder()

        with ib.function("test_block_div") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.ub_copy_in(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.div(tile_a, tile_b))
            result = ib.let("result", block.ub_copy_out(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.div" in str(func)


class TestBlockUnaryOps:
    """Tests for block unary operations."""

    def test_block_sqrt(self):
        """Test block.sqrt operation."""
        ib = IRBuilder()

        with ib.function("test_block_sqrt") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_in = ib.let("tile_in", block.ub_copy_in(input_tensor, 0, 0, 32, 32))
            tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_in))
            result = ib.let("result", block.ub_copy_out(tile_sqrt, 0, 0, 32, 32, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.sqrt" in str(func)


class TestBlockReductionOps:
    """Tests for block reduction operations."""

    def test_block_sum_no_keepdim(self):
        """Test block.sum operation without keepdim."""
        ib = IRBuilder()

        with ib.function("test_block_sum") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128], DataType.FP32))
            f.return_type(ir.TensorType([128], DataType.FP32))

            tile_in = ib.let("tile_in", block.ub_copy_in(input_tensor, 0, 0, 32, 128))
            # Sum along axis 1 (columns), result shape should be (32,) with keepdim=False
            tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=False))
            # Copy the reduced 1D tile back; width is set to 1 for the slice
            result = ib.let("result", block.ub_copy_out(tile_sum, 0, 0, 32, 1, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.sum" in str(func)

    def test_block_sum_keepdim(self):
        """Test block.sum operation with keepdim."""
        ib = IRBuilder()

        with ib.function("test_block_sum_keepdim") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
            f.return_type(ir.TensorType([128, 1], DataType.FP32))

            tile_in = ib.let("tile_in", block.ub_copy_in(input_tensor, 0, 0, 32, 128))
            # Sum along axis 1 (columns), result shape should be (32, 1)
            tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))
            result = ib.let("result", block.ub_copy_out(tile_sum, 0, 0, 32, 1, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.sum" in str(func)


class TestBlockOpsIntegration:
    """Integration tests for block operations."""

    def test_build_program_with_block_ops(self):
        """Test building a complete Program with block operations."""
        ib = IRBuilder()

        # Build first function: element-wise multiplication
        with ib.function("block_multiply") as f1:
            input_a = f1.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f1.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f1.param("output", ir.TensorType([128, 128], DataType.FP32))
            f1.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.ub_copy_in(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.mul(tile_a, tile_b))
            result = ib.let("result", block.ub_copy_out(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func1 = f1.get_result()

        # Build second function: reduction sum
        with ib.function("block_reduce_sum") as f2:
            input_tensor = f2.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f2.param("output", ir.TensorType([128, 1], DataType.FP32))
            f2.return_type(ir.TensorType([128, 1], DataType.FP32))

            tile_in = ib.let("tile_in", block.ub_copy_in(input_tensor, 0, 0, 32, 128))
            tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))
            result = ib.let("result", block.ub_copy_out(tile_sum, 0, 0, 32, 1, output_tensor))
            ib.return_stmt(result)

        func2 = f2.get_result()

        # Create a Program with both functions
        program = ir.Program([func1, func2], "block_ops_program", ir.Span.unknown())

        assert program is not None
        assert len(program.functions) == 2
        assert program.name == "block_ops_program"

        # Verify we can retrieve functions by name
        retrieved_func1 = program.get_function("block_multiply")
        assert retrieved_func1 is not None
        assert retrieved_func1.name == "block_multiply"

        retrieved_func2 = program.get_function("block_reduce_sum")
        assert retrieved_func2 is not None
        assert retrieved_func2.name == "block_reduce_sum"

        # Print program
        print(f"\n{program}")

    def test_complex_block_computation(self):
        """Test complex block computation combining multiple operations."""
        ib = IRBuilder()

        with ib.function("complex_block_computation") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            input_c = f.param("input_c", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
            f.return_type(ir.TensorType([128, 1], DataType.FP32))

            # Load tiles
            tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 128))
            tile_b = ib.let("tile_b", block.ub_copy_in(input_b, 0, 0, 32, 128))
            tile_c = ib.let("tile_c", block.ub_copy_in(input_c, 0, 0, 32, 128))

            # Compute: sqrt(a * b + c)
            tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
            tile_add = ib.let("tile_add", block.add(tile_mul, tile_c))
            tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_add))

            # Reduce along axis 1
            tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))

            # Store result
            result = ib.let("result", block.ub_copy_out(tile_sum, 0, 0, 32, 1, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.mul" in str(func)
        assert "block.add" in str(func)
        assert "block.sqrt" in str(func)
        assert "block.sum" in str(func)
        assert "block.ub_copy_in" in str(func)
        assert "block.ub_copy_out" in str(func)
        # Print function
        print(f"\n{func}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
