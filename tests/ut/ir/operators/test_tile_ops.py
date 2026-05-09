# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile operations."""

import math

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir.op import tile


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
                tile_c: pl.Tile[[32, 32], pl.UINT8] = pl.cmp(tile_a, tile_b, cmp_type=0)
                one_tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.full([32, 32], dtype=pl.FP32, value=1.0)
                zero_tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                selected: pl.Tile[[32, 32], pl.FP32] = pl.sel(tile_c, one_tile, zero_tile, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(selected, [0, 0], output)
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
                tile_c: pl.Tile[[32, 32], pl.UINT8] = pl.cmps(tile_a, 0.0, cmp_type=0)
                one_tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.full([32, 32], dtype=pl.FP32, value=1.0)
                zero_tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                selected: pl.Tile[[32, 32], pl.FP32] = pl.sel(tile_c, one_tile, zero_tile, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(selected, [0, 0], output)
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

    def test_tile_rsqrt_rejects_tmp_shape_mismatch(self):
        """tile.rsqrt rejects a tmp tile whose per-dim shape differs from the input."""
        span = ir.Span.unknown()
        input_type = ir.TileType([16, 64], DataType.FP32)
        tmp_type = ir.TileType([32, 64], DataType.FP32)  # rank matches, dim 0 differs
        input_var = ir.Var("src", input_type, span)
        tmp_var = ir.Var("tmp", tmp_type, span)

        with pytest.raises(ValueError, match="shape mismatch"):
            tile.rsqrt(input_var, tmp_var)

    def test_tile_rsqrt_high_precision(self):
        """tile.rsqrt accepts an optional tmp tile for the high-precision path."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.FP32] = pl.tile.create(
                    [32, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.tile.rsqrt(tile_a, tmp=tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.rsqrt" in ir_str

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

    # ------------------------------------------------------------------
    # tile.sin / tile.cos (FP32-only)
    # ------------------------------------------------------------------

    def test_tile_sin_creates_call(self):
        """tile.sin on an FP32 tile produces a Call with FP32 output of the same shape."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.FP32)
        tile_var = ir.Var("x", tile_type, span)

        call = tile.sin(tile_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.sin"

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

    def test_tile_cos_creates_call(self):
        """tile.cos on an FP32 tile produces a Call with FP32 output of the same shape."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.FP32)
        tile_var = ir.Var("x", tile_type, span)

        call = tile.cos(tile_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.cos"

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

    def test_tile_sin_rejects_fp16(self):
        """tile.sin must reject FP16 input with an error mentioning the op name and FP32."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.FP16)
        tile_var = ir.Var("x", tile_type, span)

        with pytest.raises(ValueError, match=r"tile\.sin.*FP32"):
            tile.sin(tile_var)

    def test_tile_cos_rejects_bf16(self):
        """tile.cos must reject BF16 input with an error mentioning the op name and FP32."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.BF16)
        tile_var = ir.Var("x", tile_type, span)

        with pytest.raises(ValueError, match=r"tile\.cos.*FP32"):
            tile.cos(tile_var)

    def test_tile_sin_rejects_int32(self):
        """tile.sin must reject INT32 input with an FP32-mentioning error."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.INT32)
        tile_var = ir.Var("x", tile_type, span)

        with pytest.raises(ValueError, match=r"(?i)FP32"):
            tile.sin(tile_var)


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

    def test_tile_row_max(self, ascend_backend, default_pass_manager):
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

        optimized_program = default_pass_manager.run_passes(RowMaxKernel)

        assert optimized_program is not None
        assert "tile.row_max" in str(optimized_program)

    def test_tile_row_sum(self, ascend_backend, default_pass_manager):
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

        optimized_program = default_pass_manager.run_passes(RowSumKernel)

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

    def test_tile_col_sum(self):
        """Test tile.col_sum operation (2 args: tile + tmp_tile)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_col_sum: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_sum(tile_in, tmp_tile)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_col_sum, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_sum" in ir_str

    def test_tile_col_max(self):
        """Test tile.col_max operation (1 arg)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tile_col_max: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_max(tile_in)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_col_max, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_max" in ir_str

    def test_tile_col_min(self):
        """Test tile.col_min operation (1 arg)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tile_col_min: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_min(tile_in)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_col_min, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_min" in ir_str

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
        """Test tile.row_expand operator - expand row vector to target tile shape."""

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
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand(tile_a, tile_row)
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

        with pytest.raises(ValueError, match="compile-time constant"):
            tile.slice(tile_var, [8, valid_n], [0, 0])

    @staticmethod
    def _make_slice_tile_var():
        """Build a [16, 32] FP16 tile Var for slice pad_value tests."""
        span = ir.Span.unknown()
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim32], DataType.FP16)
        return ir.Var("tile", tile_type, span)

    def test_tile_slice_with_pad_value_zero(self):
        """tile.slice writes pad_value=zero to the output tile_view.pad."""
        tile_var = self._make_slice_tile_var()
        call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=ir.PadValue.zero)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.slice"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.pad == ir.PadValue.zero
        assert len(result_type.tile_view.valid_shape) == 2
        assert isinstance(result_type.tile_view.valid_shape[0], ir.ConstInt)
        assert result_type.tile_view.valid_shape[0].value == 8
        assert isinstance(result_type.tile_view.valid_shape[1], ir.ConstInt)
        assert result_type.tile_view.valid_shape[1].value == 4

    def test_tile_slice_with_pad_value_min(self):
        """tile.slice writes pad_value=min to the output tile_view.pad."""
        tile_var = self._make_slice_tile_var()
        call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=ir.PadValue.min)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.pad == ir.PadValue.min

    def test_tile_slice_with_pad_value_max(self):
        """tile.slice writes pad_value=max to the output tile_view.pad."""
        tile_var = self._make_slice_tile_var()
        call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=ir.PadValue.max)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.pad == ir.PadValue.max

    def test_tile_slice_default_pad_is_null(self):
        """tile.slice without pad_value defaults to PadValue.null (backward compat)."""
        tile_var = self._make_slice_tile_var()
        call = tile.slice(tile_var, [8, 16], [0, 0])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.get_effective_tile_view().pad == ir.PadValue.null

    def test_tile_slice_rejects_bad_pad_value(self):
        """tile.slice rejects a non-PadValue pad_value kwarg via registry validation."""
        tile_var = self._make_slice_tile_var()
        span = tile_var.span
        shape_tuple = ir.MakeTuple(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)], span
        )
        offset_tuple = ir.MakeTuple(
            [ir.ConstInt(0, DataType.INT32, span), ir.ConstInt(0, DataType.INT32, span)], span
        )
        valid_shape_tuple = ir.MakeTuple(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(4, DataType.INT32, span)], span
        )
        with pytest.raises(TypeError, match="'pad_value'.*incompatible type"):
            ir.create_op_call(
                "tile.slice",
                [tile_var, shape_tuple, offset_tuple, valid_shape_tuple],
                {"pad_value": 5},
                span,
            )

    def test_tile_slice_accepts_numeric_sugar_pad_value(self):
        """tile.slice maps 0 / math.inf / -math.inf onto PadValue zero/max/min."""
        tile_var = self._make_slice_tile_var()
        for literal, expected_pad in [
            (0, ir.PadValue.zero),
            (math.inf, ir.PadValue.max),
            (-math.inf, ir.PadValue.min),
        ]:
            call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=literal)
            result_type = call.type
            assert isinstance(result_type, ir.TileType)
            assert result_type.tile_view is not None
            assert result_type.tile_view.pad == expected_pad

    def test_tile_slice_rejects_bad_numeric_pad_value_at_python_boundary(self):
        """Non-sugar numeric values are rejected at the Python API boundary."""
        tile_var = self._make_slice_tile_var()
        with pytest.raises(ValueError, match="fillpad pad_value"):
            tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=5)

    def test_tile_slice_pad_without_valid_shape_warns(self):
        """DSL emits a UserWarning when pad_value is set but valid_shape is None."""
        span = ir.Span.unknown()
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        tile_arg = pl.Tile(expr=tile_var)
        with pytest.warns(UserWarning, match="pad_value has no effect"):
            pl.tile.slice(tile_arg, [8, 16], [0, 0], pad_value=pl.PadValue.zero)

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

        # Reshape to [32, 1] — [N,1] infers col_major blayout.
        call2 = tile.reshape(tile_var, [32, 1])
        result_type2 = call2.type
        assert isinstance(result_type2, ir.TileType)
        assert len(result_type2.shape) == 2
        assert result_type2.get_effective_tile_view().blayout == ir.TileLayout.col_major

        # Layout is inferred from target shape for vector repair
        call3 = tile.reshape(tile_var, [1, 32])
        result_type3 = call3.type
        assert isinstance(result_type3, ir.TileType)
        assert result_type3.get_effective_tile_view().blayout == ir.TileLayout.row_major
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

    def test_tile_set_validshape(self):
        """Test tile.set_validshape with constant valid dimensions."""
        span = ir.Span.unknown()

        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim32, dim32], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        call = tile.set_validshape(tile_var, 16, 24)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.set_validshape"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2
        assert result_type.tile_view is not None
        assert len(result_type.tile_view.valid_shape) == 2

    def test_tile_set_validshape_dynamic(self):
        """Test tile.set_validshape with dynamic Scalar[INDEX] dimensions."""
        span = ir.Span.unknown()

        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim32, dim32], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)
        valid_rows = ir.Var("vr", ir.ScalarType(DataType.INDEX), span)
        valid_cols = ir.Var("vc", ir.ScalarType(DataType.INDEX), span)

        call = tile.set_validshape(tile_var, valid_rows, valid_cols)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.set_validshape"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.valid_shape[0] is valid_rows
        assert result_type.tile_view.valid_shape[1] is valid_cols

    def test_tile_set_validshape_preserves_physical_shape(self):
        """Physical shape is unchanged; only valid_shape metadata is updated."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim64], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        call = tile.set_validshape(tile_var, 8, 32)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert isinstance(result_type.shape[0], ir.ConstInt)
        assert result_type.shape[0].value == 16
        assert isinstance(result_type.shape[1], ir.ConstInt)
        assert result_type.shape[1].value == 64

    def test_tile_set_validshape_rejects_negative(self):
        """Negative constant valid dimensions are rejected."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim16], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        with pytest.raises(Exception, match="must be >= 0"):
            tile.set_validshape(tile_var, -1, 8)

    def test_tile_set_validshape_rejects_exceeding_bound(self):
        """Valid dimensions exceeding physical shape are rejected."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim16], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        with pytest.raises(Exception, match="exceeds tile bound"):
            tile.set_validshape(tile_var, 32, 8)

    def test_transform_operators_registered(self):
        """Test that transform operators are registered."""
        assert ir.is_op_registered("tile.slice")
        assert ir.is_op_registered("tile.reshape")
        assert ir.is_op_registered("tile.transpose")
        assert ir.is_op_registered("tile.set_validshape")


def _const_dims(span, *values):
    """Build a list of ConstInt dims (INT32) from Python ints."""
    return [ir.ConstInt(v, DataType.INT32, span) for v in values]


class TestTileBatchMatMulOps:
    """Tests for tile batch matrix multiplication operations."""

    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape", "input_dtype", "expected_shape"),
        [
            # 2D: [16,32] @ [32,64] -> [16,64] (regular matmul)
            ([16, 32], [32, 64], DataType.FP16, [16, 64]),
            # 3D: [4,16,32] @ [4,32,64] -> [4,16,64] (one batch dim)
            ([4, 16, 32], [4, 32, 64], DataType.FP32, [4, 16, 64]),
            # 4D: [2,3,16,32] @ [2,3,32,64] -> [2,3,16,64] (multiple batch dims, FP16 in)
            ([2, 3, 16, 32], [2, 3, 32, 64], DataType.FP16, [2, 3, 16, 64]),
            # Broadcast: [1,16,32] @ [4,32,64] -> [4,16,64]
            ([1, 16, 32], [4, 32, 64], DataType.FP32, [4, 16, 64]),
        ],
        ids=["2d", "3d", "4d", "broadcast"],
    )
    def test_batch_matmul(self, lhs_shape, rhs_shape, input_dtype, expected_shape):
        """tile.batch_matmul handles batch ranks + broadcasting; result dtype is promoted to FP32."""
        span = ir.Span.unknown()
        lhs_type = ir.TileType(_const_dims(span, *lhs_shape), input_dtype)
        rhs_type = ir.TileType(_const_dims(span, *rhs_shape), input_dtype)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        call = tile.batch_matmul(lhs, rhs, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        const_dims = [dim for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert len(const_dims) == len(result_type.shape)
        assert [dim.value for dim in const_dims] == expected_shape
        assert result_type.dtype == DataType.FP32

    def test_batch_matmul_dtype_mismatch(self):
        """Test tile.batch_matmul rejects mismatched dtypes."""
        span = ir.Span.unknown()

        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)

        lhs_type = ir.TileType([dim4, dim16, dim32], DataType.FP16)
        rhs_type = ir.TileType([dim4, dim32, dim16], DataType.FP32)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        with pytest.raises(ValueError, match="identical"):
            tile.batch_matmul(lhs, rhs, span)

    def test_batch_matmul_int_accumulation(self):
        """Test tile.batch_matmul with integer inputs produces INT32 accumulator dtype."""
        span = ir.Span.unknown()

        dim2 = ir.ConstInt(2, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)

        lhs_type = ir.TileType([dim2, dim16, dim32], DataType.INT8)
        rhs_type = ir.TileType([dim2, dim32, dim16], DataType.INT8)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        call = tile.batch_matmul(lhs, rhs, span)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.INT32

    def test_batch_matmul_output_tile_view(self):
        """Test tile.batch_matmul output has correct TileView (col_major, row_major, fractal=1024)."""
        span = ir.Span.unknown()

        dim2 = ir.ConstInt(2, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim2, dim16, dim32], DataType.FP16)
        rhs_type = ir.TileType([dim2, dim32, dim64], DataType.FP16)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        call = tile.batch_matmul(lhs, rhs, span)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        eff = result_type.get_effective_tile_view()
        assert eff.blayout == ir.TileLayout.col_major
        assert eff.slayout == ir.TileLayout.row_major
        assert eff.fractal == 1024

    @pytest.mark.parametrize(
        ("acc_shape", "lhs_shape", "rhs_shape", "input_dtype", "acc_dtype"),
        [
            # 2D: acc[16,64] += lhs[16,32] @ rhs[32,64]
            ([16, 64], [16, 32], [32, 64], DataType.FP16, DataType.FP32),
            # 3D: acc[4,16,64] += lhs[4,16,32] @ rhs[4,32,64]
            ([4, 16, 64], [4, 16, 32], [4, 32, 64], DataType.FP32, DataType.FP32),
            # 4D: multiple batch dims
            ([2, 3, 16, 64], [2, 3, 16, 32], [2, 3, 32, 64], DataType.FP16, DataType.FP32),
            # Broadcast lhs/rhs against acc batch
            ([4, 16, 64], [1, 16, 32], [4, 32, 64], DataType.FP32, DataType.FP32),
            # INT path
            ([2, 16, 64], [2, 16, 32], [2, 32, 64], DataType.INT8, DataType.INT32),
        ],
        ids=["2d", "3d", "4d", "broadcast", "int"],
    )
    def test_batch_matmul_acc(self, acc_shape, lhs_shape, rhs_shape, input_dtype, acc_dtype):
        """tile.batch_matmul_acc handles batch ranks + broadcasting; result shape == acc shape."""
        span = ir.Span.unknown()
        acc_type = ir.TileType(_const_dims(span, *acc_shape), acc_dtype)
        lhs_type = ir.TileType(_const_dims(span, *lhs_shape), input_dtype)
        rhs_type = ir.TileType(_const_dims(span, *rhs_shape), input_dtype)
        acc = ir.Var("acc", acc_type, span)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        call = tile.batch_matmul_acc(acc, lhs, rhs, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.batch_matmul_acc"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        const_dims = [dim for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert len(const_dims) == len(result_type.shape)
        assert [dim.value for dim in const_dims] == acc_shape
        assert result_type.dtype == acc_dtype

    def test_batch_matmul_acc_acc_batch_must_match_broadcast(self):
        """tile.batch_matmul_acc rejects acc batch dims that disagree with broadcast(lhs, rhs)."""
        span = ir.Span.unknown()
        acc_type = ir.TileType(_const_dims(span, 2, 16, 64), DataType.FP32)
        lhs_type = ir.TileType(_const_dims(span, 4, 16, 32), DataType.FP16)
        rhs_type = ir.TileType(_const_dims(span, 4, 32, 64), DataType.FP16)
        acc = ir.Var("acc", acc_type, span)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        with pytest.raises(ValueError, match="acc batch dim"):
            tile.batch_matmul_acc(acc, lhs, rhs, span)

    def test_batch_matmul_acc_dtype_mismatch(self):
        """tile.batch_matmul_acc rejects acc dtype that doesn't match the result dtype."""
        span = ir.Span.unknown()
        # FP inputs => FP32 acc required, but acc is FP16 here.
        acc_type = ir.TileType(_const_dims(span, 2, 16, 64), DataType.FP16)
        lhs_type = ir.TileType(_const_dims(span, 2, 16, 32), DataType.FP16)
        rhs_type = ir.TileType(_const_dims(span, 2, 32, 64), DataType.FP16)
        acc = ir.Var("acc", acc_type, span)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        with pytest.raises(ValueError, match="accumulator dtype"):
            tile.batch_matmul_acc(acc, lhs, rhs, span)

    def test_batch_matmul_acc_inner_dim_mismatch(self):
        """tile.batch_matmul_acc rejects mismatched K dims."""
        span = ir.Span.unknown()
        acc_type = ir.TileType(_const_dims(span, 2, 16, 64), DataType.FP32)
        lhs_type = ir.TileType(_const_dims(span, 2, 16, 32), DataType.FP16)
        rhs_type = ir.TileType(_const_dims(span, 2, 16, 64), DataType.FP16)  # K=16, mismatch
        acc = ir.Var("acc", acc_type, span)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        with pytest.raises(ValueError, match="inner dimensions"):
            tile.batch_matmul_acc(acc, lhs, rhs, span)

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

    def test_tile_maximums(self):
        """Test tile.maximums operator - element-wise maximum of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.maximums(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.maximums" in ir_str

    def test_tile_minimums(self):
        """Test tile.minimums operator - element-wise minimum of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.minimums(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.minimums" in ir_str

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
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.sel(tile_m, tile_a, tile_b, tmp)
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
        assert len(tile_type.get_effective_tile_view().valid_shape) == 2

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
        eff = tile_type.get_effective_tile_view()
        assert eff.blayout == ir.TileLayout.col_major
        assert len(eff.valid_shape) == 2

    def test_create_row_vector_keeps_row_major_layout(self):
        """Non-column-vector shapes should keep the default row-major layout."""
        call = tile.create([1, 32], DataType.FP32, ir.MemorySpace.Vec)
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.get_effective_tile_view().blayout == ir.TileLayout.row_major


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


class TestTileExtractOp:
    """Tests for tile.extract operator (ISA TEXTRACT Variant 1)."""

    @staticmethod
    def _make_src_var(
        rows: int = 64,
        cols: int = 256,
        dtype: DataType = DataType.FP16,
        memory_space: ir.MemorySpace | None = None,
    ) -> ir.Var:
        span = ir.Span.unknown()
        r = ir.ConstInt(rows, DataType.INT32, span)
        c = ir.ConstInt(cols, DataType.INT32, span)
        tile_type = ir.TileType([r, c], dtype, memory_space=memory_space)
        return ir.Var("src", tile_type, span)

    def test_tile_extract_basic(self):
        """tile.extract returns a TileType with the requested shape and src dtype."""
        src_var = self._make_src_var()

        call = tile.extract(src_var, 0, 0, shape=[64, 64], target_memory=ir.MemorySpace.Left)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.extract"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2
        rows, cols = result_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 64
        assert isinstance(cols, ir.ConstInt) and cols.value == 64

    def test_tile_extract_acc_to_mat(self):
        """Acc source → Mat target: src lives in Acc, dtype preserved."""
        src_var = self._make_src_var(64, 64, DataType.FP32, memory_space=ir.MemorySpace.Acc)
        src_tile_type = src_var.type
        assert isinstance(src_tile_type, ir.TileType)
        assert src_tile_type.memory_space == ir.MemorySpace.Acc

        call = tile.extract(src_var, 0, 0, shape=[32, 32], target_memory=ir.MemorySpace.Mat)

        assert call.op.name == "tile.extract"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        rows, cols = result_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 32
        assert isinstance(cols, ir.ConstInt) and cols.value == 32

    def test_tile_extract_dynamic_offset(self):
        """Runtime symbolic offsets are accepted (no compile-time bounds check fires)."""
        span = ir.Span.unknown()
        src_var = self._make_src_var()
        row = ir.Var("row", ir.ScalarType(DataType.INDEX), span)
        col = ir.Var("col", ir.ScalarType(DataType.INDEX), span)

        call = tile.extract(src_var, row, col, shape=[16, 16], target_memory=ir.MemorySpace.Left)

        assert call.op.name == "tile.extract"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        rows, cols = result_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 16
        assert isinstance(cols, ir.ConstInt) and cols.value == 16

    def test_tile_extract_shape_exceeds_src_static(self):
        """Static shape larger than src is rejected at deduction time."""
        src_var = self._make_src_var(64, 64)

        with pytest.raises(ValueError, match="exceeds src"):
            tile.extract(src_var, 0, 0, shape=[128, 128], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_offset_plus_shape_exceeds_src_static(self):
        """Constant offset + shape that walks past src is rejected at deduction."""
        src_var = self._make_src_var(64, 64)

        # offset 60 + shape 16 = 76 > 64 rows
        with pytest.raises(ValueError, match="exceeds src row"):
            tile.extract(src_var, 60, 0, shape=[16, 16], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_negative_offset_static(self):
        """Constant negative offset is rejected at deduction."""
        src_var = self._make_src_var(64, 64)

        with pytest.raises(ValueError, match="must be >= 0"):
            tile.extract(src_var, -1, 0, shape=[16, 16], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_rejects_non_index_offset(self):
        """index_row/col must be INT64/UINT64/INDEX."""
        span = ir.Span.unknown()
        src_var = self._make_src_var()
        bad = ir.Var("bad", ir.ScalarType(DataType.FP32), span)

        with pytest.raises(ValueError, match="INT64/UINT64/INDEX"):
            tile.extract(src_var, bad, 0, shape=[16, 16], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_rejects_dynamic_shape(self):
        """shape elements must be compile-time ConstInt for storage allocation."""
        span = ir.Span.unknown()
        src_var = self._make_src_var()
        dyn = ir.Var("dyn", ir.ScalarType(DataType.INDEX), span)

        with pytest.raises(ValueError, match="compile-time ConstInt"):
            tile.extract(src_var, 0, 0, shape=[dyn, 16], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_rejects_non_2d_shape(self):
        """shape must be 2D."""
        src_var = self._make_src_var()

        with pytest.raises(ValueError, match="2D"):
            tile.extract(src_var, 0, 0, shape=[16, 16, 16], target_memory=ir.MemorySpace.Left)


class TestTileScatterUpdateOps:
    """Test suite for tile.scatter_update operation."""

    @pytest.mark.parametrize(
        ("input_shape", "src_shape", "dtype"),
        [
            # 2D scatter: rows=16, src first dim = b*s = 8.
            ([16, 64], [8, 64], DataType.FP16),
            # 4D KV-cache style: [block_num, block_size, 1, d] with src [b, s, 1, d].
            ([4, 4, 1, 64], [2, 4, 1, 64], DataType.BF16),
        ],
        ids=["2d", "4d"],
    )
    def test_tile_scatter_update_valid(self, input_shape, src_shape, dtype):
        """tile.scatter_update preserves input rank/dtype across 2D and 4D inputs."""
        span = ir.Span.unknown()
        input_type = ir.TileType(_const_dims(span, *input_shape), dtype)
        index_type = ir.TileType(_const_dims(span, 2, 4), DataType.INT32)
        src_type = ir.TileType(_const_dims(span, *src_shape), dtype)
        scratch_type = ir.TileType(_const_dims(span, 1, input_shape[-1]), dtype)

        call = tile.scatter_update(
            ir.Var("inp", input_type, span),
            -2,
            ir.Var("idx", index_type, span),
            ir.Var("src", src_type, span),
            ir.Var("scratch", scratch_type, span),
        )

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.scatter_update"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == dtype
        const_dims = [dim for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert len(const_dims) == len(result_type.shape)
        assert [dim.value for dim in const_dims] == input_shape

    @pytest.mark.parametrize(
        ("src_dtype", "dim", "match"),
        [
            (DataType.FP32, -2, "src dtype"),  # input is FP16; src must match
            (DataType.FP16, -1, "dim=-2"),  # only dim=-2 is supported
        ],
        ids=["dtype_mismatch", "invalid_dim"],
    )
    def test_tile_scatter_update_rejects_invalid(self, src_dtype, dim, match):
        """tile.scatter_update validates src dtype and the dim argument."""
        span = ir.Span.unknown()
        input_type = ir.TileType(_const_dims(span, 16, 64), DataType.FP16)
        index_type = ir.TileType(_const_dims(span, 2, 4), DataType.INT32)
        src_type = ir.TileType(_const_dims(span, 8, 64), src_dtype)
        scratch_type = ir.TileType(_const_dims(span, 1, 64), DataType.FP16)

        with pytest.raises(ValueError, match=match):
            tile.scatter_update(
                ir.Var("inp", input_type, span),
                dim,
                ir.Var("idx", index_type, span),
                ir.Var("src", src_type, span),
                ir.Var("scratch", scratch_type, span),
            )


class TestTileMscatterOps:
    """Test suite for tile.mscatter operation."""

    def test_tile_mscatter_basic(self):
        """Test tile.mscatter constructs a Call returning a TensorType."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([tensor_n], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        call = tile.mscatter(src_var, idx_var, out_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.mscatter"
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert result_type.dtype == DataType.FP32

    def test_tile_mscatter_fp16(self):
        """Test tile.mscatter works with FP16 dtype."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(8, DataType.INT32, span)
        cols = ir.ConstInt(16, DataType.INT32, span)
        tensor_n = ir.ConstInt(512, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP16)
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([tensor_n], DataType.FP16)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        call = tile.mscatter(src_var, idx_var, out_var)
        assert call.op.name == "tile.mscatter"
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert result_type.dtype == DataType.FP16

    def test_tile_mscatter_src_dtype_error(self):
        """Test tile.mscatter rejects unsupported src dtype."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.UINT8)  # unsupported
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([tensor_n], DataType.UINT8)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="src dtype"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_idx_dtype_error(self):
        """Test tile.mscatter rejects non-INT32 idx dtype."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT16)  # wrong dtype
        tensor_type = ir.TensorType([tensor_n], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="idx dtype"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_rank_mismatch_error(self):
        """Test tile.mscatter rejects idx with different rank than src."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)  # 2D
        idx_type = ir.TileType([rows], DataType.INT32)  # 1D
        tensor_type = ir.TensorType([tensor_n], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="idx rank"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_dtype_mismatch_error(self):
        """Test tile.mscatter rejects output_tensor with dtype different from src."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([tensor_n], DataType.FP16)  # mismatched

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="output_tensor dtype"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_arg_count_error(self):
        """Test tile.mscatter rejects wrong number of arguments."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)

        with pytest.raises(ValueError, match="3 arguments"):
            # Missing output_tensor; call the op directly via create_op_call
            ir.create_op_call("tile.mscatter", [src_var, idx_var], {}, span)

    def test_tile_mscatter_shape_mismatch_error(self):
        """Test tile.mscatter rejects idx with different shape than src."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(16, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
            DataType.FP32,
        )
        idx_type = ir.TileType(
            [ir.ConstInt(16, DataType.INT32, span), ir.ConstInt(64, DataType.INT32, span)],
            DataType.INT32,
        )
        tensor_type = ir.TensorType([ir.ConstInt(1024, DataType.INT32, span)], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="idx shape to match src shape"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_scalar_output_error(self):
        """Test tile.mscatter rejects scalar (rank-0) output tensor."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="at least 1 dimension"):
            tile.mscatter(src_var, idx_var, out_var)


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

    @pytest.mark.parametrize(
        ("t0_shape", "t0_dtype", "t1_shape", "t1_dtype", "match"),
        [
            ([32, 16], DataType.FP32, [32, 16], DataType.FP16, "same dtype"),
            ([32, 16], DataType.FP32, [8, 16], DataType.FP32, "row count must match"),
        ],
        ids=["dtype_mismatch", "row_mismatch"],
    )
    def test_tile_concat_rejects_invalid(self, t0_shape, t0_dtype, t1_shape, t1_dtype, match):
        """tile.concat enforces matching dtype and matching row counts."""
        span = ir.Span.unknown()
        t0_type = ir.TileType(_const_dims(span, *t0_shape), t0_dtype)
        t1_type = ir.TileType(_const_dims(span, *t1_shape), t1_dtype)

        with pytest.raises(ValueError, match=match):
            tile.concat(ir.Var("src0", t0_type, span), ir.Var("src1", t1_type, span))


class TestTileFormatShapeError:
    """Regression tests for issue #824: FormatShape prints readable shapes, not pointer addresses."""

    @staticmethod
    def _make_dim(span, value):
        """Create a dim that is either a ConstInt (from ``int``) or a symbolic Var (from ``str``)."""
        if isinstance(value, str):
            return ir.Var(value, ir.ScalarType(DataType.INT32), span)
        return ir.ConstInt(value, DataType.DEFAULT_CONST_INT, span)

    @pytest.mark.parametrize(
        ("op_callable", "lhs_dims", "rhs_dims", "match"),
        [
            # Static shape mismatch surfaces the concrete dims (not pointers).
            (tile.add, [16, 16], [32, 16], r"\[16, 16\].*\[32, 16\]"),
            (tile.mul, [8, 16], [32, 16], r"\[8, 16\].*\[32, 16\]"),
            # Symbolic mismatch surfaces variable names instead of dim addresses.
            (tile.add, ["M", 16], ["N", 16], r"\[M, 16\].*\[N, 16\]"),
        ],
        ids=[
            "add_shape_mismatch_shows_readable_dims",
            "mul_shape_mismatch_shows_readable_dims",
            "add_symbolic_shape_mismatch_shows_var_names",
        ],
    )
    def test_tile_shape_mismatch_message(self, op_callable, lhs_dims, rhs_dims, match):
        """Shape-mismatch errors render dims/symbols as readable text (regression for #824)."""
        span = ir.Span.unknown()
        lhs_type = ir.TileType([self._make_dim(span, d) for d in lhs_dims], DataType.FP32)
        rhs_type = ir.TileType([self._make_dim(span, d) for d in rhs_dims], DataType.FP32)
        tile_a = ir.Var("a", lhs_type, span)
        tile_b = ir.Var("b", rhs_type, span)

        with pytest.raises(ValueError, match=match):
            op_callable(tile_a, tile_b)


class TestTileCiOp:
    """Tests for tile.ci (contiguous integer sequence generation, pto.tci)."""

    def test_tile_ci_ascending(self):
        """tile.ci returns a TileType with requested shape / dtype."""
        call = tile.ci(0, [1, 32], dtype=DataType.INT32)
        t = call.type
        assert isinstance(t, ir.TileType)
        assert t.dtype == DataType.INT32
        assert len(t.shape) == 2
        assert "tile.ci" in str(call)
        assert "descending=False" in str(call)

    def test_tile_ci_descending_kwarg_printed(self):
        """descending=True should appear in the printed IR."""
        call = tile.ci(10, [1, 16], dtype=DataType.INT32, descending=True)
        assert "descending=True" in str(call)

    def test_tile_ci_rejects_float_dtype(self):
        with pytest.raises(ValueError, match=r"INT16.*INT32.*UINT16.*UINT32"):
            tile.ci(0, [1, 32], dtype=DataType.FP32)

    def test_tile_ci_accepts_uint_dtype(self):
        call = tile.ci(0, [1, 16], dtype=DataType.UINT32)
        assert call is not None

    def test_tile_ci_rejects_cols_equal_one(self):
        with pytest.raises(ValueError, match="innermost dimension"):
            tile.ci(0, [32, 1], dtype=DataType.INT32)

    def test_tile_ci_rejects_multi_row_shape(self):
        """pto.tci only populates the first row, so leading dims must be 1."""
        with pytest.raises(ValueError, match=r"leading dimensions must be 1"):
            tile.ci(0, [4, 32], dtype=DataType.INT32)

    def test_tile_ci_rejects_start_dtype_mismatch(self):
        span = ir.Span.unknown()
        start = ir.Var("s", ir.ScalarType(DataType.INT16), span)
        with pytest.raises(ValueError, match=r"start.*dtype"):
            tile.ci(start, [1, 32], dtype=DataType.INT32)

    def test_tile_arange_alias_is_ci(self):
        assert pl.tile.arange is pl.tile.ci


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
