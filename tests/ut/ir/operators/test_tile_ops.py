# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile operations."""

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir
from pypto.backend import BackendType
from pypto.ir.op import tile
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


class TestTileElementwiseOps:
    """Test suite for tile-level element-wise operators (tile-tile and tile-scalar)."""

    def test_tile_add(self):
        """Test tile.add operator - element-wise addition of two tiles."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.add" in ir_str

    def test_tile_sub(self):
        """Test tile.sub operator - element-wise subtraction of two tiles."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sub" in ir_str

    def test_tile_mul(self):
        """Test tile.mul operator - element-wise multiplication of two tiles."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.mul" in ir_str

    def test_tile_div(self):
        """Test tile.div operator - element-wise division of two tiles."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.div" in ir_str

    def test_tile_muls(self):
        """Test tile.muls operator - multiply all elements of a tile by scalar."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.muls" in ir_str

    def test_tile_muls_preserves_tile_dtype(self):
        """tile.muls result must keep the tile's element dtype, not promote to the scalar's dtype.

        pto.tmuls requires src and dst to share the same element type, so multiplying a BF16
        tile by an FP32 scalar must produce a BF16 result (the scalar is narrowed at runtime).
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.BF16],
                output: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[128, 128], pl.BF16]:
                tile_a: pl.Tile[[32, 32], pl.BF16] = pl.load(a, [0, 0], [32, 32])
                # Scalar 0.0 is typed FP32 by default; result must still be BF16.
                tile_c: pl.Tile[[32, 32], pl.BF16] = pl.mul(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.BF16] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.muls" in ir_str
        # Confirm the result tile carries BF16 (pl.BF16 in the Python printer),
        # not a promoted FP32.  The hardware narrowing happens at runtime.
        assert "tile_c: pl.Tile[[32, 32], pl.BF16" in ir_str

    def test_tile_cmp(self):
        """Test tile.cmp operator - element-wise comparison of two tiles."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.cmp" in ir_str

    def test_tile_cmps(self):
        """Test tile.cmps operator - compare tile elements with scalar."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.cmps" in ir_str


class TestTileUnaryOps:
    """Test suite for tile-level unary operators."""

    def test_tile_log(self):
        """Test tile.log operator - natural logarithm of all elements."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.log" in ir_str

    def test_tile_abs(self):
        """Test tile.abs operator - absolute value of all elements."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.abs" in ir_str

    def test_tile_relu(self):
        """Test tile.relu operator - ReLU activation function."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.relu" in ir_str

    def test_tile_exp(self):
        """Test tile.exp operator - exponential of all elements."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.exp" in ir_str

    def test_tile_sqrt(self):
        """Test tile.sqrt operator - square root of all elements."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sqrt" in ir_str

    def test_tile_neg(self):
        """Test tile.neg operator - negate all elements."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.neg" in ir_str


class TestTileReductionOps:
    """Test suite for tile-level reduction operators."""

    def test_tile_sum_axis0(self):
        """Test tile.sum operator - sum along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.sum(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sum" in ir_str

    def test_tile_sum_axis1(self):
        """Test tile.sum operator - sum along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.sum(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sum" in ir_str

    def test_tile_max_axis0(self):
        """Test tile.max operator - max along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.max(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.max" in ir_str

    def test_tile_max_axis1(self):
        """Test tile.max operator - max along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.max(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.max" in ir_str

    def test_tile_row_max(self):
        """Test tile.row_max operation."""

        @pl.program
        class RowMaxKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_max_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 1], pl.FP32] = pl.tile.create(
                    [32, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_max: pl.Tile[[32, 1], pl.FP32] = pl.row_max(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_max, [0, 0], output)
                return result

        program = RowMaxKernel
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B_CCE)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized_program = pm.run_passes(program)

        assert optimized_program is not None
        assert "tile.row_max" in str(optimized_program)

    def test_tile_row_sum(self):
        """Test tile.row_sum operation."""

        @pl.program
        class RowSumKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_sum_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 1], pl.FP32] = pl.tile.create(
                    [32, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_sum: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_sum, [0, 0], output)
                return result

        program = RowSumKernel
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B_CCE)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized_program = pm.run_passes(program)

        assert optimized_program is not None
        assert "tile.row_sum" in str(optimized_program)

    def test_tile_row_min(self):
        """Test tile.row_min operation."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_row_min: pl.Tile[[32, 1], pl.FP32] = pl.row_min(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_row_min, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_min" in ir_str

    def test_tile_min_axis0(self):
        """Test tile.min operator - min along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.min(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.min" in ir_str

    def test_tile_min_axis1(self):
        """Test tile.min operator - min along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.min(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.min" in ir_str


class TestTileBroadcastOps:
    """Test suite for tile-level broadcast operators."""

    def test_tile_col_expand(self):
        """Test tile.col_expand operator - expand column vector to target shape."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand" in ir_str

    def test_tile_col_expand_mul(self):
        """Test tile.col_expand_mul operator - expand column and multiply with tile."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_mul" in ir_str

    def test_tile_col_expand_div(self):
        """Test tile.col_expand_div operator - expand column and divide tile."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_div" in ir_str

    def test_tile_col_expand_sub(self):
        """Test tile.col_expand_sub operator - expand column and subtract from tile."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_sub" in ir_str

    def test_tile_row_expand_add(self):
        """Test tile.row_expand_add operator - expand row and add to tile."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_add" in ir_str

    def test_tile_row_expand_sub(self):
        """Test tile.row_expand_sub operator - subtract row vector from each tile row."""

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
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_sub(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_sub" in ir_str

    def test_tile_row_expand_div(self):
        """Test tile.row_expand_div operator - divide each tile row by row vector."""

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
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_div(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_div" in ir_str

    def test_tile_row_expand_mul(self):
        """Test tile.row_expand_mul operator - multiply each tile row by row vector."""

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
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_mul(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_mul" in ir_str

    def test_tile_row_expand(self):
        """Test tile.row_expand operator - broadcast first element of each row across the row."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand" in ir_str

    def test_tile_expands(self):
        """Test tile.expands operator - expand scalar to tile shape."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.expands" in ir_str


class TestTileMatMulOps:
    """Test suite for tile-level matrix multiplication operators."""

    def test_tile_matmul(self):
        """Test tile.matmul operator - matrix multiplication."""

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
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.matmul" in ir_str

    def test_tile_matmul_acc(self):
        """Test tile.matmul_acc operator - matrix multiplication with accumulation (TMATMUL_ACC).

        Computes: acc_out = acc_in + lhs @ rhs
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                acc_in: pl.Tensor[[128, 128], pl.FP32],
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_acc: pl.Tile[[32, 32], pl.FP32] = pl.load(acc_in, [0, 0], [32, 32])
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul_acc(tile_acc, tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.matmul_acc" in ir_str

    def test_tile_matmul_bias(self):
        """Test tile.matmul_bias operator - matrix multiplication with bias add."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                bias: pl.Tensor[[1, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_bias: pl.Tile[[1, 32], pl.FP32] = pl.load(bias, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul_bias(tile_a, tile_b, tile_bias)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.matmul_bias" in ir_str

    def test_tile_gemv(self):
        """Test tile.gemv operator - general matrix-vector multiplication."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv(tile_a, tile_b)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.gemv" in ir_str

    def test_tile_gemv_acc(self):
        """Test tile.gemv_acc operator - GEMV with accumulation."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                acc_in: pl.Tensor[[1, 128], pl.FP32],
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_acc: pl.Tile[[1, 32], pl.FP32] = pl.load(acc_in, [0, 0], [1, 32])
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv_acc(tile_acc, tile_a, tile_b)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.gemv_acc" in ir_str

    def test_tile_gemv_bias(self):
        """Test tile.gemv_bias operator - GEMV with bias add."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                bias: pl.Tensor[[1, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_bias: pl.Tile[[1, 32], pl.FP32] = pl.load(bias, [0, 0], [1, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv_bias(tile_a, tile_b, tile_bias)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.gemv_bias" in ir_str


class TestTileTransformOps:
    """Test suite for tile-level transform operators."""

    def test_tile_transpose(self):
        """Test tile.transpose operator - transpose a tile."""

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
                result: pl.Tensor[[64, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.transpose" in ir_str


class TestTileSliceReshapeOps:
    """Tests for tile slice and reshape operations."""

    def test_tile_slice(self):
        """Test tile.slice operation."""
        span = ir.Span.unknown()

        # Create a tile variable [16, 32]
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Create a slice [8, 16] with offset [0, 0]
        call = tile.slice(tile_var, [8, 16], [0, 0])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.slice"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2

    def test_tile_slice_with_dynamic_valid_shape(self):
        """tile.slice keeps static allocation shape and stores dynamic valid_shape in TileView."""
        span = ir.Span.unknown()

        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        valid_n = ir.Var("valid_n", ir.ScalarType(DataType.INDEX), span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, valid_n])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.slice"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert len(result_type.shape) == 2
        assert isinstance(result_type.shape[1], ir.ConstInt)
        assert result_type.tile_view.valid_shape[1] is valid_n

    def test_tile_slice_rejects_dynamic_shape(self):
        """tile.slice shape must stay static so InitMemRef can allocate memory."""
        span = ir.Span.unknown()

        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        valid_n = ir.Var("valid_n", ir.ScalarType(DataType.INDEX), span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        with pytest.raises(Exception, match="compile-time constant"):
            tile.slice(tile_var, [8, valid_n], [0, 0])

    def test_tile_reshape(self):
        """Test tile.reshape operation."""
        span = ir.Span.unknown()

        # Create a tile variable [4, 8]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim8], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        # Reshape to [8, 4]
        call = tile.reshape(tile_var, [8, 4])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.reshape"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

        # Reshape to [32, 1]
        call2 = tile.reshape(tile_var, [32, 1])
        result_type2 = call2.type
        assert isinstance(result_type2, ir.TileType)
        assert len(result_type2.shape) == 2
        assert result_type2.tile_view is not None
        assert result_type2.tile_view.blayout == ir.TileLayout.col_major

        # Layout is inferred from target shape for vector repair
        call3 = tile.reshape(tile_var, [1, 32])
        result_type3 = call3.type
        assert isinstance(result_type3, ir.TileType)
        assert result_type3.tile_view is not None
        assert result_type3.tile_view.blayout == ir.TileLayout.row_major
        assert call3.kwargs == {}

    def test_tile_transpose(self):
        """Test tile.transpose operation."""
        span = ir.Span.unknown()

        # Create a tile [8, 16]
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose: [8, 16] -> [16, 8]
        call = tile.transpose(tile_var, 0, 1)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.transpose"
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
        call = tile.transpose(tile_var, -2, -1)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)

    def test_transform_operators_registered(self):
        """Test that transform operators are registered."""
        assert ir.is_op_registered("tile.slice")
        assert ir.is_op_registered("tile.reshape")
        assert ir.is_op_registered("tile.transpose")


class TestTileBatchMatMulOps:
    """Tests for tile batch matrix multiplication operations."""

    def test_batch_matmul_2d(self):
        """Test tile.batch_matmul with 2D tiles (equivalent to regular matmul)."""
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
        call = ir.create_op_call("tile.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 2
        assert result_type.dtype == DataType.FP16

    def test_batch_matmul_3d(self):
        """Test tile.batch_matmul with 3D tiles (batch dimension)."""
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
        call = ir.create_op_call("tile.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3
        assert result_type.dtype == DataType.FP32

    def test_batch_matmul_4d(self):
        """Test tile.batch_matmul with 4D tiles (multiple batch dimensions)."""
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
        call = ir.create_op_call("tile.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 4
        assert result_type.dtype == DataType.FP16

    def test_batch_matmul_broadcast(self):
        """Test tile.batch_matmul with broadcasting batch dimensions."""
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
        call = ir.create_op_call("tile.batch_matmul", [lhs, rhs], {}, span)

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
        call = tile.transpose(tile_var, 0, 2)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.transpose"
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
        call = tile.row_max(tile_var, tmp_tile)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.row_max"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3

    def test_slice_3d(self):
        """Test slice operation on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 16, 32]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Create a slice with different shape [2, 8, 16]
        new_shape = [2, 8, 16]
        offset = [0, 0, 0]
        call = tile.slice(tile_var, new_shape, offset)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.slice"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3


class TestTileBitwiseArithmeticOps:
    """Test suite for newly added tile-level bitwise and arithmetic ops (rem, and, or, xor)."""

    def test_tile_rem(self):
        """Test tile.rem operator - element-wise remainder of two tiles."""

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
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.rem(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.rem" in ir_str

    def test_tile_rems(self):
        """Test tile.rems operator - element-wise remainder of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.rems(tile_a, 3.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.rems" in ir_str

    def test_tile_and(self):
        """Test tile.and operator - element-wise bitwise AND of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.and_(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.and" in ir_str

    def test_tile_ands(self):
        """Test tile.ands operator - element-wise bitwise AND of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.ands(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.ands" in ir_str

    def test_tile_or(self):
        """Test tile.or operator - element-wise bitwise OR of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.or_(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.or" in ir_str

    def test_tile_ors(self):
        """Test tile.ors operator - element-wise bitwise OR of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.ors(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.ors" in ir_str

    def test_tile_xor(self):
        """Test tile.xor operator - element-wise bitwise XOR of two tiles with tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.INT32] = pl.tile.create(
                    [32, 32], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.xor(tile_a, tile_b, tmp)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.xor" in ir_str

    def test_tile_xors(self):
        """Test tile.xors operator - element-wise bitwise XOR of tile and scalar with tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.INT32] = pl.tile.create(
                    [32, 32], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.xors(tile_a, scalar, tmp)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.xors" in ir_str

    def test_tile_shl(self):
        """Test tile.shl operator - element-wise bitwise left shift of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shl(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shl" in ir_str

    def test_tile_shls(self):
        """Test tile.shls operator - element-wise bitwise left shift of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shls(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shls" in ir_str

    def test_tile_maxs(self):
        """Test tile.maxs operator - element-wise maximum of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.maxs(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.maxs" in ir_str

    def test_tile_mins(self):
        """Test tile.mins operator - element-wise minimum of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.mins(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.mins" in ir_str

    def test_tile_shr(self):
        """Test tile.shr operator - element-wise bitwise right shift of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shr(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shr" in ir_str

    def test_tile_shrs(self):
        """Test tile.shrs operator - element-wise bitwise right shift of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shrs(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shrs" in ir_str

    def test_tile_shl_preserves_lhs_dtype(self):
        """Regression: tile.shl result dtype must match LHS dtype, not the promoted type.

        When lhs is UINT16 and rhs is UINT32, the result must be UINT16 (LHS dtype),
        consistent with the scalar variant tile.shls which preserves the LHS tile dtype.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT16],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT16],
            ) -> pl.Tensor[[128, 128], pl.UINT16]:
                tile_a: pl.Tile[[16, 16], pl.UINT16] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT16] = pl.shl(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT16] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shl" in ir_str

    def test_tile_shr_preserves_lhs_dtype(self):
        """Regression: tile.shr result dtype must match LHS dtype, not the promoted type.

        When lhs is UINT16 and rhs is UINT32, the result must be UINT16 (LHS dtype),
        consistent with the scalar variant tile.shrs which preserves the LHS tile dtype.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT16],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT16],
            ) -> pl.Tensor[[128, 128], pl.UINT16]:
                tile_a: pl.Tile[[16, 16], pl.UINT16] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT16] = pl.shr(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT16] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shr" in ir_str

    def test_tile_prelu(self):
        """Test tile.prelu operator - element-wise parametric ReLU with slope and tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_x: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                slope: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tmp: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.prelu(tile_x, slope, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.prelu" in ir_str

    def test_tile_not(self):
        """Test tile.not operator - element-wise bitwise NOT of a tile (int16/uint16 only)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT16],
                output: pl.Tensor[[128, 128], pl.INT16],
            ) -> pl.Tensor[[128, 128], pl.INT16]:
                tile_a: pl.Tile[[16, 16], pl.INT16] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.INT16] = pl.not_(tile_a)
                result: pl.Tensor[[128, 128], pl.INT16] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.not" in ir_str

    def test_tile_addc(self):
        """Test tile.addc operator - element-wise addition of three tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.load(c, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.addc(tile_a, tile_b, tile_c)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.addc" in ir_str

    def test_tile_subc(self):
        """Test tile.subc operator - element-wise subtraction of three tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.load(c, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.subc(tile_a, tile_b, tile_c)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.subc" in ir_str

    def test_tile_addsc(self):
        """Test tile.addsc operator - element-wise addition of tile, scalar, and tile."""

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
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.addsc(tile_a, 2.0, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.addsc" in ir_str

    def test_tile_subsc(self):
        """Test tile.subsc operator - element-wise subtraction of tile, scalar, and tile."""

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
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.subsc(tile_a, 2.0, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.subsc" in ir_str

    def test_tile_lrelu(self):
        """Test tile.lrelu operator - element-wise leaky ReLU with scalar slope."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.lrelu(tile_a, 0.1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.lrelu" in ir_str

    def test_tile_sels(self):
        """Test tile.sels operator - select between two tiles via integer scalar mode."""

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
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.sels(tile_a, tile_b, 1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sels" in ir_str

    def test_tile_sel(self):
        """Test tile.sel operator - per-element selection between two tiles via mask tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                m: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_m: pl.Tile[[32, 32], pl.FP32] = pl.load(m, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.sel(tile_m, tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sel" in ir_str


class TestTileLoadOp:
    """Tests for tile.load operation with valid_shapes and TileView."""

    def test_load_without_valid_shapes_sets_tileview_from_shapes(self):
        """When valid_shapes not provided, TileView.valid_shape equals shapes."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)

        call = tile.load(tensor, [0, 0], [64, 128])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert len(tile_type.tile_view.valid_shape) == 2

    def test_load_with_static_valid_shapes_sets_tileview(self):
        """When valid_shapes provided as static ints, TileView.valid_shape reflects it."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)

        call = tile.load(tensor, [0, 0], [128, 128], valid_shapes=[64, 128])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert len(tile_type.tile_view.valid_shape) == 2
        # tile shape should still be [128, 128]
        assert len(tile_type.shape) == 2

    def test_load_with_dynamic_valid_shapes_sets_tileview(self):
        """When valid_shapes provided as symbolic vars, TileView.valid_shape uses them."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)
        M = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        N = ir.Var("N", ir.ScalarType(DataType.INT64), span)

        call = tile.load(tensor, [0, 0], [64, 128], valid_shapes=[M, N])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert len(tile_type.tile_view.valid_shape) == 2
        # valid_shape elements should be the symbolic vars M and N
        assert tile_type.tile_view.valid_shape[0] is M
        assert tile_type.tile_view.valid_shape[1] is N

    def test_load_via_pl_load_with_valid_shapes(self):
        """pl.load with valid_shapes propagates TileView to the output tile."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                M: pl.Scalar[pl.INT64],
                N: pl.Scalar[pl.INT64],
            ) -> pl.Tile[[128, 128], pl.FP32]:
                tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128], valid_shapes=[M, N])
                return tile

        # Just verifying it builds without error
        assert Prog is not None


class TestTileCreateOp:
    """Tests for tile.create layout inference."""

    def test_create_column_vector_uses_col_major_layout(self):
        """Static `[N, 1]` Vec tiles should infer col-major block layout."""
        call = tile.create([32, 1], DataType.FP32, ir.MemorySpace.Vec)
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert tile_type.tile_view.blayout == ir.TileLayout.col_major
        assert len(tile_type.tile_view.valid_shape) == 2

    def test_create_row_vector_keeps_row_major_layout(self):
        """Non-column-vector shapes should keep the default row-major layout."""
        call = tile.create([1, 32], DataType.FP32, ir.MemorySpace.Vec)
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert tile_type.tile_view.blayout == ir.TileLayout.row_major


class TestTileScalarOps:
    """Tests for tile scalar read/write ops (tile.read / tile.write)."""

    def test_tile_write_via_pl_write(self):
        """Test tile.write: write scalar into tile via pl.write with indices."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 16], pl.FP16],
                dst: pl.Tensor[[16, 16], pl.FP16],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                t: pl.Tile[[16, 16], pl.FP16] = pl.load(src, [0, 0], [16, 16])
                val: pl.Scalar[pl.FP16] = pl.read(t, [0, 0])
                pl.write(t, [0, 1], val)
                result: pl.Tensor[[16, 16], pl.FP16] = pl.store(t, [0, 0], dst)
                return result

        ir_str = str(Program)
        assert "tile.write" in ir_str

    def test_tile_read_write_direct(self):
        """Test tile.read/write via pl.tile.read/pl.tile.write directly."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 16], pl.FP16],
                dst: pl.Tensor[[16, 16], pl.FP16],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                t: pl.Tile[[16, 16], pl.FP16] = pl.load(src, [0, 0], [16, 16])
                val: pl.Scalar[pl.FP16] = pl.tile.read(t, [0, 0])
                pl.tile.write(t, [0, 1], val)
                result: pl.Tensor[[16, 16], pl.FP16] = pl.store(t, [0, 0], dst)
                return result

        ir_str = str(Program)
        assert "tile.read" in ir_str
        assert "tile.write" in ir_str


class TestTileAssembleOp:
    """Tests for tile.assemble operator."""

    def test_tile_assemble_basic(self):
        """Test tile.assemble type deduction returns target TileType."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        target_type = ir.TileType([dim16, dim128], DataType.FP32)
        target_var = ir.Var("target", target_type, span)

        source_type = ir.TileType([dim16, dim64], DataType.FP32)
        source_var = ir.Var("source", source_type, span)

        call = tile.assemble(target_var, source_var, [0, 0])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.assemble"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

    def test_tile_assemble_dtype_mismatch(self):
        """tile.assemble requires matching dtypes for target and source."""
        span = ir.Span.unknown()
        dim16 = ir.ConstInt(16, DataType.INT32, span)

        target_type = ir.TileType([dim16, dim16], DataType.FP32)
        target_var = ir.Var("target", target_type, span)

        source_type = ir.TileType([dim16, dim16], DataType.FP16)
        source_var = ir.Var("source", source_type, span)

        with pytest.raises(ValueError, match="same dtype"):
            tile.assemble(target_var, source_var, [0, 0])


class TestTileScatterUpdateOps:
    """Test suite for tile.scatter_update operation."""

    def test_tile_scatter_update_2d(self):
        """Test tile.scatter_update with 2D input and src."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        d = ir.ConstInt(64, DataType.INT32, span)
        b = ir.ConstInt(2, DataType.INT32, span)
        s = ir.ConstInt(4, DataType.INT32, span)
        bs = ir.ConstInt(8, DataType.INT32, span)

        input_type = ir.TileType([rows, d], DataType.FP16)
        index_type = ir.TileType([b, s], DataType.INT32)
        src_type = ir.TileType([bs, d], DataType.FP16)

        input_var = ir.Var("inp", input_type, span)
        index_var = ir.Var("idx", index_type, span)
        src_var = ir.Var("src", src_type, span)

        call = tile.scatter_update(input_var, -2, index_var, src_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.scatter_update"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2

    def test_tile_scatter_update_4d(self):
        """Test tile.scatter_update with 4D input and src."""
        span = ir.Span.unknown()
        block_num = ir.ConstInt(4, DataType.INT32, span)
        block_size = ir.ConstInt(4, DataType.INT32, span)
        one = ir.ConstInt(1, DataType.INT32, span)
        d = ir.ConstInt(64, DataType.INT32, span)
        b = ir.ConstInt(2, DataType.INT32, span)
        s = ir.ConstInt(4, DataType.INT32, span)

        input_type = ir.TileType([block_num, block_size, one, d], DataType.BF16)
        index_type = ir.TileType([b, s], DataType.INT32)
        src_type = ir.TileType([b, s, one, d], DataType.BF16)

        input_var = ir.Var("kv_cache", input_type, span)
        index_var = ir.Var("block_table", index_type, span)
        src_var = ir.Var("new_kv", src_type, span)

        call = tile.scatter_update(input_var, -2, index_var, src_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.scatter_update"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.BF16
        assert len(result_type.shape) == 4

    def test_tile_scatter_update_dtype_mismatch(self):
        """Test tile.scatter_update rejects mismatched dtypes between input and src."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        d = ir.ConstInt(64, DataType.INT32, span)
        b = ir.ConstInt(2, DataType.INT32, span)
        s = ir.ConstInt(4, DataType.INT32, span)
        bs = ir.ConstInt(8, DataType.INT32, span)

        input_type = ir.TileType([rows, d], DataType.FP16)
        index_type = ir.TileType([b, s], DataType.INT32)
        src_type = ir.TileType([bs, d], DataType.FP32)  # wrong dtype

        input_var = ir.Var("inp", input_type, span)
        index_var = ir.Var("idx", index_type, span)
        src_var = ir.Var("src", src_type, span)

        with pytest.raises(ValueError, match="src dtype"):
            tile.scatter_update(input_var, -2, index_var, src_var)

    def test_tile_scatter_update_invalid_dim(self):
        """Test tile.scatter_update rejects dim values other than -2."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        d = ir.ConstInt(64, DataType.INT32, span)
        b = ir.ConstInt(2, DataType.INT32, span)
        s = ir.ConstInt(4, DataType.INT32, span)
        bs = ir.ConstInt(8, DataType.INT32, span)

        input_type = ir.TileType([rows, d], DataType.FP16)
        index_type = ir.TileType([b, s], DataType.INT32)
        src_type = ir.TileType([bs, d], DataType.FP16)

        input_var = ir.Var("inp", input_type, span)
        index_var = ir.Var("idx", index_type, span)
        src_var = ir.Var("src", src_type, span)

        with pytest.raises(ValueError, match="dim=-2"):
            tile.scatter_update(input_var, -1, index_var, src_var)


class TestTileConcatOps:
    """Test suite for tile.concat operation."""

    def test_tile_concat(self):
        """Test tile.concat operator - concatenate two tiles along columns."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[32, 16], pl.FP32] = pl.load(b, [0, 0], [32, 16])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.concat(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.concat" in ir_str

    def test_tile_concat_ir_level(self):
        """Test tile.concat at IR level with type deduction."""
        span = ir.Span.unknown()

        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        t0_type = ir.TileType([dim32, dim16], DataType.FP32)
        t1_type = ir.TileType([dim32, dim16], DataType.FP32)
        t0_var = ir.Var("src0", t0_type, span)
        t1_var = ir.Var("src1", t1_type, span)

        call = tile.concat(t0_var, t1_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.concat"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2
        # Output cols = 16 + 16 = 32
        assert isinstance(result_type.shape[1], ir.ConstInt)
        assert result_type.shape[1].value == 32

    def test_tile_concat_dtype_mismatch(self):
        """Test tile.concat rejects mismatched dtypes."""
        span = ir.Span.unknown()

        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        t0_type = ir.TileType([dim32, dim16], DataType.FP32)
        t1_type = ir.TileType([dim32, dim16], DataType.FP16)
        t0_var = ir.Var("src0", t0_type, span)
        t1_var = ir.Var("src1", t1_type, span)

        with pytest.raises(ValueError, match="same dtype"):
            tile.concat(t0_var, t1_var)

    def test_tile_concat_row_mismatch(self):
        """Test tile.concat rejects mismatched row counts."""
        span = ir.Span.unknown()

        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        t0_type = ir.TileType([dim32, dim16], DataType.FP32)
        t1_type = ir.TileType([dim8, dim16], DataType.FP32)
        t0_var = ir.Var("src0", t0_type, span)
        t1_var = ir.Var("src1", t1_type, span)

        with pytest.raises(ValueError, match="row count must match"):
            tile.concat(t0_var, t1_var)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
