# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for block operations."""

import pypto.language as pl
import pytest
from pypto.ir.op import block
from pypto.ir.pass_manager import PassManager
from pypto.pypto_core import DataType, ir


class TestBlockElementwiseOps:
    """Test suite for block-level element-wise operators (tile-tile and tile-scalar)."""

    def test_block_add(self):
        """Test block.add operator - element-wise addition of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.add" in ir_str

    def test_block_sub(self):
        """Test block.sub operator - element-wise subtraction of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.sub(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.sub" in ir_str

    def test_block_mul(self):
        """Test block.mul operator - element-wise multiplication of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.mul(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.mul" in ir_str

    def test_block_div(self):
        """Test block.div operator - element-wise division of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.div(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.div" in ir_str

    def test_block_muls(self):
        """Test block.muls operator - multiply all elements of a tile by scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.mul(tile_a, 2.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.muls" in ir_str

    def test_block_cmp(self):
        """Test block.cmp operator - element-wise comparison of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.cmp(tile_a, tile_b, cmp_type=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.cmp" in ir_str

    def test_block_cmps(self):
        """Test block.cmps operator - compare tile elements with scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.cmps(tile_a, 0.0, cmp_type=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.cmps" in ir_str


class TestBlockUnaryOps:
    """Test suite for block-level unary operators."""

    def test_block_log(self):
        """Test block.log operator - natural logarithm of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.log(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.log" in ir_str

    def test_block_abs(self):
        """Test block.abs operator - absolute value of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.abs(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.abs" in ir_str

    def test_block_relu(self):
        """Test block.relu operator - ReLU activation function."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.relu(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.relu" in ir_str

    def test_block_exp(self):
        """Test block.exp operator - exponential of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.exp(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.exp" in ir_str

    def test_block_sqrt(self):
        """Test block.sqrt operator - square root of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.sqrt(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.sqrt" in ir_str

    def test_block_neg(self):
        """Test block.neg operator - negate all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.neg(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.neg" in ir_str


class TestBlockReductionOps:
    """Test suite for block-level reduction operators."""

    def test_block_sum_axis0(self):
        """Test block.sum operator - sum along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.sum(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.sum" in ir_str

    def test_block_sum_axis1(self):
        """Test block.sum operator - sum along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 1], pl.FP32] = pl.sum(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 1], output)
                return result

        ir_str = str(Program)
        assert "block.sum" in ir_str

    def test_block_max_axis0(self):
        """Test block.max operator - max along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.max(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.max" in ir_str

    def test_block_max_axis1(self):
        """Test block.max operator - max along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 1], pl.FP32] = pl.max(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 1], output)
                return result

        ir_str = str(Program)
        assert "block.max" in ir_str

    def test_block_row_max(self):
        """Test block.row_max operation."""

        @pl.program
        class RowMaxKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_max_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 1], pl.FP32] = pl.block.create_tile(
                    [32, 1], dtype=pl.FP32, target_memory=1
                )
                tile_max: pl.Tile[[32, 1], pl.FP32] = pl.row_max(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_max, [0, 0], [32, 1], output)
                return result

        program = RowMaxKernel
        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)

        assert optimized_program is not None
        assert "block.row_max" in str(optimized_program)

    def test_block_row_sum(self):
        """Test block.row_sum operation."""

        @pl.program
        class RowSumKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_sum_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 1], pl.FP32] = pl.block.create_tile(
                    [32, 1], dtype=pl.FP32, target_memory=1
                )
                tile_sum: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_sum, [0, 0], [32, 1], output)
                return result

        program = RowSumKernel
        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)

        assert optimized_program is not None
        assert "block.row_sum" in str(optimized_program)

    def test_block_row_min(self):
        """Test block.row_min operation."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.block.create_tile(
                    [32, 128], dtype=pl.FP32, target_memory=1
                )
                tile_row_min: pl.Tile[[32, 1], pl.FP32] = pl.row_min(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_row_min, [0, 0], [32, 1], output)
                return result

        ir_str = str(Program)
        assert "block.row_min" in ir_str

    def test_block_min_axis0(self):
        """Test block.min operator - min along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.min(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.min" in ir_str

    def test_block_min_axis1(self):
        """Test block.min operator - min along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 1], pl.FP32] = pl.min(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 1], output)
                return result

        ir_str = str(Program)
        assert "block.min" in ir_str


class TestBlockBroadcastOps:
    """Test suite for block-level broadcast operators."""

    def test_block_col_expand(self):
        """Test block.col_expand operator - expand column vector to target shape."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                target: pl.Tensor[[128, 128], pl.FP32],
                col: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_target: pl.Tile[[32, 32], pl.FP32] = pl.load(target, [0, 0], [32, 32])
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand(tile_target, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.col_expand" in ir_str

    def test_block_col_expand_mul(self):
        """Test block.col_expand_mul operator - expand column and multiply with tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_mul(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.col_expand_mul" in ir_str

    def test_block_col_expand_div(self):
        """Test block.col_expand_div operator - expand column and divide tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_div(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.col_expand_div" in ir_str

    def test_block_col_expand_sub(self):
        """Test block.col_expand_sub operator - expand column and subtract from tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_sub(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.col_expand_sub" in ir_str

    def test_block_row_expand_add(self):
        """Test block.row_expand_add operator - expand row and add to tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_add(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.row_expand_add" in ir_str

    def test_block_expands(self):
        """Test block.expands operator - expand scalar to tile shape."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.expands(tile_a, 1.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.expands" in ir_str


class TestBlockMatMulOps:
    """Test suite for block-level matrix multiplication operators."""

    def test_block_matmul(self):
        """Test block.matmul operator - matrix multiplication."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.matmul" in ir_str


class TestBlockTransformOps:
    """Test suite for block-level transform operators."""

    def test_block_transpose(self):
        """Test block.transpose operator - transpose a tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                output: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_c: pl.Tile[[16, 32], pl.FP32] = pl.transpose(tile_a, axis1=0, axis2=1)
                result: pl.Tensor[[64, 128], pl.FP32] = pl.store(tile_c, [0, 0], [16, 32], output)
                return result

        ir_str = str(Program)
        assert "block.transpose" in ir_str


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
        tmp_tile = ir.Var("tmp_tile", tile_type, span)

        # row_max should reduce the last dimension: [4, 16, 32] -> [4, 16, 1]
        call = block.row_max(tile_var, tmp_tile)

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
