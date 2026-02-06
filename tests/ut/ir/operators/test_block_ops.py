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

    def test_load(self):
        """Test block.load operation."""
        ib = IRBuilder()

        with ib.function("test_load") as f:
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

            tile = ib.let("tile", block.load(input_tensor, 0, 0, 32, 32))
            ib.return_stmt(tile)

        func = f.get_result()
        assert func is not None
        assert "block.load" in str(func)

    def test_store(self):
        """Test block.store operation."""
        ib = IRBuilder()

        with ib.function("test_store") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile = ib.let("tile", block.load(input_tensor, 0, 0, 32, 32))
            result = ib.let("result", block.store(tile, 0, 0, 32, 32, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.store" in str(func)

    def test_move(self):
        """Test block.move operation."""
        ib = IRBuilder()

        with ib.function("test_move") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(
                ir.TileType(
                    [
                        ir.ConstInt(64, DataType.INT32, ir.Span.unknown()),
                        ir.ConstInt(32, DataType.INT32, ir.Span.unknown()),
                    ],
                    DataType.FP32,
                )
            )

            tile = ib.let("tile", block.load(input_tensor, 0, 0, 32, 64))
            # Move with transpose: shape [32, 64] -> [64, 32]
            moved_tile = ib.let("moved_tile", block.move(tile, target_memory=1, transpose=True))
            ib.return_stmt(moved_tile)

        func = f.get_result()
        assert func is not None
        assert "block.move" in str(func)


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

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.mul(tile_a, tile_b))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.mul" in str(func)

    def test_block_muls(self):
        """Test block.muls operation (tile * scalar)."""
        ib = IRBuilder()

        with ib.function("test_block_muls") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.muls(tile_a, 2.0))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.muls" in str(func)

    def test_block_add(self):
        """Test block.add operation."""
        ib = IRBuilder()

        with ib.function("test_block_add") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.add(tile_a, tile_b))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
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

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.div(tile_a, tile_b))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.div" in str(func)

    def test_block_sub(self):
        """Test block.sub operation."""
        ib = IRBuilder()

        with ib.function("test_block_sub") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.sub(tile_a, tile_b))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.sub" in str(func)

    def test_block_adds(self):
        """Test block.adds operation (tile + scalar)."""
        ib = IRBuilder()

        with ib.function("test_block_adds") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.adds(tile_a, 5.0))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.adds" in str(func)

    def test_block_divs(self):
        """Test block.divs operation (tile / scalar)."""
        ib = IRBuilder()

        with ib.function("test_block_divs") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.divs(tile_a, 3.0))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.divs" in str(func)

    def test_block_subs(self):
        """Test block.subs operation (tile - scalar)."""
        ib = IRBuilder()

        with ib.function("test_block_subs") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.subs(tile_a, 1.0))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.subs" in str(func)

    def test_block_maximum(self):
        """Test block.maximum operation."""
        ib = IRBuilder()

        with ib.function("test_block_maximum") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.maximum(tile_a, tile_b))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.maximum" in str(func)


class TestBlockBroadcastOps:
    """Tests for block row broadcast operations."""

    def test_block_row_expand_sub(self):
        """Test block.row_expand_sub operation."""
        ib = IRBuilder()

        with ib.function("test_block_row_expand_sub") as f:
            input_tile = f.param("input_tile", ir.TensorType([128, 128], DataType.FP32))
            input_row = f.param("input_row", ir.TensorType([128, 1], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile = ib.let("tile", block.load(input_tile, 0, 0, 32, 128))
            row_vec = ib.let("row_vec", block.load(input_row, 0, 0, 32, 1))
            tile_result = ib.let("tile_result", block.row_expand_sub(tile, row_vec))
            result = ib.let("result", block.store(tile_result, 0, 0, 32, 128, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.row_expand_sub" in str(func)

    def test_block_row_expand_div(self):
        """Test block.row_expand_div operation."""
        ib = IRBuilder()

        with ib.function("test_block_row_expand_div") as f:
            input_tile = f.param("input_tile", ir.TensorType([128, 128], DataType.FP32))
            input_row = f.param("input_row", ir.TensorType([128, 1], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile = ib.let("tile", block.load(input_tile, 0, 0, 32, 128))
            row_vec = ib.let("row_vec", block.load(input_row, 0, 0, 32, 1))
            tile_result = ib.let("tile_result", block.row_expand_div(tile, row_vec))
            result = ib.let("result", block.store(tile_result, 0, 0, 32, 128, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.row_expand_div" in str(func)

    def test_block_row_expand_mul(self):
        """Test block.row_expand_mul operation."""
        ib = IRBuilder()

        with ib.function("test_block_row_expand_mul") as f:
            input_tile = f.param("input_tile", ir.TensorType([128, 128], DataType.FP32))
            input_row = f.param("input_row", ir.TensorType([128, 1], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile = ib.let("tile", block.load(input_tile, 0, 0, 32, 128))
            row_vec = ib.let("row_vec", block.load(input_row, 0, 0, 32, 1))
            tile_result = ib.let("tile_result", block.row_expand_mul(tile, row_vec))
            result = ib.let("result", block.store(tile_result, 0, 0, 32, 128, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.row_expand_mul" in str(func)


class TestBlockUnaryOps:
    """Tests for block unary operations."""

    def test_block_neg(self):
        """Test block.neg operation."""
        ib = IRBuilder()

        with ib.function("test_block_neg") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))
            tile_neg = ib.let("tile_neg", block.neg(tile_in))
            result = ib.let("result", block.store(tile_neg, 0, 0, 32, 32, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.neg" in str(func)

    def test_block_exp(self):
        """Test block.exp operation."""
        ib = IRBuilder()

        with ib.function("test_block_exp") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))
            tile_exp = ib.let("tile_exp", block.exp(tile_in))
            result = ib.let("result", block.store(tile_exp, 0, 0, 32, 32, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.exp" in str(func)

    def test_block_recip(self):
        """Test block.recip operation."""
        ib = IRBuilder()

        with ib.function("test_block_recip") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))
            tile_recip = ib.let("tile_recip", block.recip(tile_in))
            result = ib.let("result", block.store(tile_recip, 0, 0, 32, 32, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.recip" in str(func)

    def test_block_sqrt(self):
        """Test block.sqrt operation."""
        ib = IRBuilder()

        with ib.function("test_block_sqrt") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))
            tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_in))
            result = ib.let("result", block.store(tile_sqrt, 0, 0, 32, 32, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.sqrt" in str(func)

    def test_block_rsqrt(self):
        """Test block.rsqrt operation."""
        ib = IRBuilder()

        with ib.function("test_block_rsqrt") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))
            tile_rsqrt = ib.let("tile_rsqrt", block.rsqrt(tile_in))
            result = ib.let("result", block.store(tile_rsqrt, 0, 0, 32, 32, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.rsqrt" in str(func)


class TestBlockMatMulOps:
    """Tests for block matrix multiplication operations."""

    def test_block_matmul(self):
        """Test block.matmul operation."""
        ib = IRBuilder()

        with ib.function("test_block_matmul") as f:
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 64))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 64, 32))
            tile_c = ib.let("tile_c", block.matmul(tile_a, tile_b))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.matmul" in str(func)

    def test_block_matmul_acc(self):
        """Test block.matmul_acc operation."""
        ib = IRBuilder()

        with ib.function("test_block_matmul_acc") as f:
            input_a = f.param("input_a", ir.TensorType([128, 256], DataType.FP16))
            input_b = f.param("input_b", ir.TensorType([256, 128], DataType.FP16))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            # Load first K slice
            tile_a0 = ib.let("tile_a0", block.load(input_a, 0, 0, 32, 64))
            tile_b0 = ib.let("tile_b0", block.load(input_b, 0, 0, 64, 32))

            # Initial matmul
            tile_c0 = ib.let("tile_c0", block.matmul(tile_a0, tile_b0))

            # Load second K slice
            tile_a1 = ib.let("tile_a1", block.load(input_a, 0, 64, 32, 64))
            tile_b1 = ib.let("tile_b1", block.load(input_b, 64, 0, 64, 32))

            # Accumulate
            tile_c1 = ib.let("tile_c1", block.matmul_acc(tile_c0, tile_a1, tile_b1))

            # Store result
            result = ib.let("result", block.store(tile_c1, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.matmul" in str(func)
        assert "block.matmul_acc" in str(func)


class TestBlockReductionOps:
    """Tests for block reduction operations."""

    def test_block_max(self):
        """Test block.max operation."""
        ib = IRBuilder()

        with ib.function("test_block_max") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
            f.return_type(ir.TensorType([128, 1], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))
            tile_max = ib.let("tile_max", block.max(tile_in, axis=1, keepdim=True))
            result = ib.let("result", block.store(tile_max, 0, 0, 32, 1, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.max" in str(func)

    def test_block_row_max(self):
        """Test block.row_max operation."""
        ib = IRBuilder()

        with ib.function("test_block_row_max") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
            f.return_type(ir.TensorType([128, 1], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))
            tile_row_max = ib.let("tile_row_max", block.row_max(tile_in))
            result = ib.let("result", block.store(tile_row_max, 0, 0, 32, 1, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.row_max" in str(func)

    def test_block_row_sum(self):
        """Test block.row_sum operation."""
        ib = IRBuilder()

        with ib.function("test_block_row_sum") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
            f.return_type(ir.TensorType([128, 1], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))
            tile_row_sum = ib.let("tile_row_sum", block.row_sum(tile_in))
            result = ib.let("result", block.store(tile_row_sum, 0, 0, 32, 1, output_tensor))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.row_sum" in str(func)

    def test_block_sum_no_keepdim(self):
        """Test block.sum operation without keepdim."""
        ib = IRBuilder()

        with ib.function("test_block_sum") as f:
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
            f.return_type(ir.TensorType([128, 1], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))
            # Sum along axis 1 (columns), result shape should be (32, 1) with keepdim=True
            tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))
            result = ib.let("result", block.store(tile_sum, 0, 0, 32, 1, output_tensor))
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

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))
            # Sum along axis 1 (columns), result shape should be (32, 1)
            tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))
            result = ib.let("result", block.store(tile_sum, 0, 0, 32, 1, output_tensor))
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

            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))
            tile_c = ib.let("tile_c", block.mul(tile_a, tile_b))
            result = ib.let("result", block.store(tile_c, 0, 0, 32, 32, output))
            ib.return_stmt(result)

        func1 = f1.get_result()

        # Build second function: reduction sum
        with ib.function("block_reduce_sum") as f2:
            input_tensor = f2.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f2.param("output", ir.TensorType([128, 1], DataType.FP32))
            f2.return_type(ir.TensorType([128, 1], DataType.FP32))

            tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))
            tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))
            result = ib.let("result", block.store(tile_sum, 0, 0, 32, 1, output_tensor))
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
            tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 128))
            tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 128))
            tile_c = ib.let("tile_c", block.load(input_c, 0, 0, 32, 128))

            # Compute: sqrt(a * b + c)
            tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
            tile_add = ib.let("tile_add", block.add(tile_mul, tile_c))
            tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_add))

            # Reduce along axis 1
            tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))

            # Store result
            result = ib.let("result", block.store(tile_sum, 0, 0, 32, 1, output))
            ib.return_stmt(result)

        func = f.get_result()
        assert func is not None
        assert "block.mul" in str(func)
        assert "block.add" in str(func)
        assert "block.sqrt" in str(func)
        assert "block.sum" in str(func)
        assert "block.load" in str(func)
        assert "block.store" in str(func)
        # Print function
        print(f"\n{func}")


class TestTileTransformOps:
    """Tests for tile transform operations."""

    def test_tile_view(self):
        """Test tile.view operation."""
        span = ir.Span.unknown()

        # Create a tile variable [16, 32]
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Create a view [8, 16] with offset [0, 0]
        call = block.view(tile_var, [8, 16], [0, 0])

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.view"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2

    def test_tile_reshape(self):
        """Test tile.reshape operation."""
        span = ir.Span.unknown()

        # Create a tile variable [4, 8]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim8], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        # Reshape to [8, 4]
        call = block.reshape(tile_var, [8, 4])

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.reshape"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

        # Reshape to [32, 1]
        call2 = block.reshape(tile_var, [32, 1])
        result_type2 = call2.type
        assert isinstance(result_type2, ir.TileType)
        assert len(result_type2.shape) == 2

    def test_tile_transpose(self):
        """Test tile.transpose operation."""
        span = ir.Span.unknown()

        # Create a tile [8, 16]
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose: [8, 16] -> [16, 8]
        call = block.transpose(tile_var, 0, 1)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2

    def test_tile_transpose_negative_axis(self):
        """Test tile.transpose with negative axis indices."""
        span = ir.Span.unknown()

        # Create a tile [8, 16]
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose using negative indices: axis1=-2 (0), axis2=-1 (1)
        # [8, 16] -> [16, 8]
        call = block.transpose(tile_var, -2, -1)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)

    def test_transform_operators_registered(self):
        """Test that transform operators are registered."""
        assert ir.is_op_registered("block.view")
        assert ir.is_op_registered("block.reshape")
        assert ir.is_op_registered("block.transpose")


class TestBlockBatchMatMulOps:
    """Tests for block batch matrix multiplication operations."""

    def test_batch_matmul_2d(self):
        """Test block.batch_matmul with 2D tiles (equivalent to regular matmul)."""
        span = ir.Span.unknown()

        # Create 2D tiles: [16, 32] @ [32, 64] -> [16, 64]
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim16, dim32], DataType.FP16)
        rhs_type = ir.TileType([dim32, dim64], DataType.FP16)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        # Create batch_matmul call
        call = ir.create_op_call("block.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 2
        assert result_type.dtype == DataType.FP16

    def test_batch_matmul_3d(self):
        """Test block.batch_matmul with 3D tiles (batch dimension)."""
        span = ir.Span.unknown()

        # Create 3D tiles: [4, 16, 32] @ [4, 32, 64] -> [4, 16, 64]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim4, dim16, dim32], DataType.FP32)
        rhs_type = ir.TileType([dim4, dim32, dim64], DataType.FP32)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        # Create batch_matmul call
        call = ir.create_op_call("block.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3
        assert result_type.dtype == DataType.FP32

    def test_batch_matmul_4d(self):
        """Test block.batch_matmul with 4D tiles (multiple batch dimensions)."""
        span = ir.Span.unknown()

        # Create 4D tiles: [2, 3, 16, 32] @ [2, 3, 32, 64] -> [2, 3, 16, 64]
        dim2 = ir.ConstInt(2, DataType.INT32, span)
        dim3 = ir.ConstInt(3, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim2, dim3, dim16, dim32], DataType.FP16)
        rhs_type = ir.TileType([dim2, dim3, dim32, dim64], DataType.FP16)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        # Create batch_matmul call
        call = ir.create_op_call("block.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 4
        assert result_type.dtype == DataType.FP16

    def test_batch_matmul_broadcast(self):
        """Test block.batch_matmul with broadcasting batch dimensions."""
        span = ir.Span.unknown()

        # Create tiles with different batch shapes: [1, 16, 32] @ [4, 32, 64] -> [4, 16, 64]
        dim1 = ir.ConstInt(1, DataType.INT32, span)
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim1, dim16, dim32], DataType.FP32)
        rhs_type = ir.TileType([dim4, dim32, dim64], DataType.FP32)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        # Create batch_matmul call
        call = ir.create_op_call("block.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3


class TestMultiDimensionalTileOps:
    """Tests for multi-dimensional TileType operations."""

    def test_transpose_3d(self):
        """Test transpose on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 8, 16]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose axes 0 and 2: [4, 8, 16] -> [16, 8, 4]
        call = block.transpose(tile_var, 0, 2)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3

    def test_row_max_3d(self):
        """Test row_max on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 16, 32]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim16, dim32], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        # row_max should reduce the last dimension: [4, 16, 32] -> [4, 16, 1]
        call = block.row_max(tile_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.row_max"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3

    def test_view_3d(self):
        """Test view operation on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 16, 32]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Create a view with different shape [2, 8, 16]
        new_shape = [2, 8, 16]
        offset = [0, 0, 0]
        call = block.view(tile_var, new_shape, offset)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.view"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
