# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for FlattenTileNdTo2D pass."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


class TestFlattenTileNdTo2D:
    """Test FlattenTileNdTo2D pass."""

    def test_3d_tile_element_wise(self):
        """3D tile [2, 3, 4] with tile.add -> reshaped to [6, 4]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_4d_tile(self):
        """4D tile [2, 3, 4, 5] with tile.mul -> reshaped to [24, 5]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4, 5], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4, 5], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4, 5], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4, 5], pl.FP32] = pl.load(x, [0, 0, 0, 0], [2, 3, 4, 5])
                y_tile: pl.Tile[[2, 3, 4, 5], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4, 5], pl.FP32] = pl.store(y_tile, [0, 0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4, 5], pl.FP32]) -> pl.Tensor[[2, 3, 4, 5], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4, 5], pl.FP32] = pl.create_tensor([2, 3, 4, 5], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4, 5], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4, 5], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4, 5], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4, 5], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4, 5], pl.FP32] = pl.load(x, [0, 0, 0, 0], [2, 3, 4, 5])
                x_tile: pl.Tile[[24, 5], pl.FP32] = pl.tile.reshape(x_tile_nd, [24, 5])
                y_tile: pl.Tile[[24, 5], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                y_tile_nd: pl.Tile[[2, 3, 4, 5], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4, 5])
                out_0: pl.Tensor[[2, 3, 4, 5], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4, 5], pl.FP32]) -> pl.Tensor[[2, 3, 4, 5], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4, 5], pl.FP32] = pl.create_tensor([2, 3, 4, 5], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4, 5], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_2d_tile_unchanged(self):
        """2D tile [32, 64] -> no change."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                x_tile: pl.Tile[[32, 64], pl.FP32] = pl.load(x, [0, 0], [32, 64])
                y_tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[32, 64], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[32, 64], pl.FP32]:
                out_0: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
                y: pl.Tensor[[32, 64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Before)

    def test_1d_tile_unchanged(self):
        """1D tile [64] -> no change."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Before)

    def test_tile_load_store_reshape(self):
        """tile.load 3D -> reshape -> ops -> reshape -> tile.store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(y, [0, 0, 0], [2, 3, 4])
                z_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, y_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(z_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[2, 3, 4], pl.FP32],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                z: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(y, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(y_tile_nd, [6, 4])
                z_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, y_tile)
                z_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(z_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(z_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[2, 3, 4], pl.FP32],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                z: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_tile_create_shape_flattened(self):
        """tile.create([2,3,4]) -> tile.create([6,4])."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                tmp: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.create([2, 3, 4], dtype=pl.FP32)
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, tmp)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                tmp: pl.Tile[[6, 4], pl.FP32] = pl.tile.create([6, 4], dtype=pl.FP32)
                y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, tmp)
                y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_reduce_last_axis(self):
        """tile.sum on 3D tile [2, 3, 4] with axis=2 -> axis=1 after flatten."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 1], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 1], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 1], pl.FP32] = pl.tile.sum(x_tile, axis=2, keepdim=True)
                out_0: pl.Tensor[[2, 3, 1], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 1], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 1], pl.FP32] = pl.create_tensor([2, 3, 1], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 1], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 1], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 1], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                y_tile: pl.Tile[[6, 1], pl.FP32] = pl.tile.sum(x_tile, axis=1, keepdim=True)
                y_tile_nd: pl.Tile[[2, 3, 1], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 1])
                out_0: pl.Tensor[[2, 3, 1], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 1], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 1], pl.FP32] = pl.create_tensor([2, 3, 1], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 1], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_reduce_non_last_axis_error(self):
        """tile.sum with axis=0 on 3D tile -> CHECK error."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[1, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[1, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[1, 3, 4], pl.FP32] = pl.tile.sum(x_tile, axis=0, keepdim=True)
                out_0: pl.Tensor[[1, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[1, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[1, 3, 4], pl.FP32] = pl.create_tensor([1, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[1, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        with pytest.raises(Exception, match="must reduce along the last axis"):
            passes.flatten_tile_nd_to_2d()(Before)

    def test_dynamic_shape_error(self):
        """Dynamic (non-ConstInt) tile shape on >2D tile -> CHECK error."""
        span = ir.Span.unknown()

        # Create a dynamic dimension via a Var (not ConstInt)
        n_var = ir.Var("n", ir.ScalarType(DataType.INT32), span)
        dim2 = ir.ConstInt(3, DataType.INT32, span)
        dim3 = ir.ConstInt(4, DataType.INT32, span)

        # 3D tile type with one dynamic dimension
        dyn_tile_type = ir.TileType([n_var, dim2, dim3], DataType.FP32)

        # Create vars with this tile type
        x_tile = ir.Var("x_tile", dyn_tile_type, span)

        # Create tile.add call with dynamic-shaped tile
        add_op = ir.Op("tile.add")
        add_call = ir.Call(add_op, [x_tile, x_tile], dyn_tile_type, span)
        y_tile = ir.Var("y_tile", dyn_tile_type, span)
        body = ir.AssignStmt(y_tile, add_call, span)

        # Wrap in InCore function
        func = ir.Function(
            "incore_func",
            [x_tile],
            [dyn_tile_type],
            body,
            span,
            type=ir.FunctionType.InCore,
        )
        program = ir.Program([func], "test_dyn", span)

        with pytest.raises(Exception, match="must be static"):
            passes.flatten_tile_nd_to_2d()(program)

    def test_non_incore_unchanged(self):
        """Orchestration functions are not modified."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                x_tile: pl.Tile[[32, 64], pl.FP32] = pl.load(x, [0, 0], [32, 64])
                y_tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[32, 64], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[32, 64], pl.FP32]:
                out_0: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
                y: pl.Tensor[[32, 64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Before)


class TestFlattenTileNdTo2DUnaryOps:
    """Test unary tile operations on ND tiles."""

    def test_unary_exp_3d(self):
        """tile.exp on 3D tile [2, 3, 4] -> flattened to [6, 4]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.exp(x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.exp(x_tile)
                y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_unary_neg_3d(self):
        """tile.neg on 3D tile [4, 2, 8] -> flattened to [8, 8]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[4, 2, 8], pl.FP32],
                out_0: pl.Out[pl.Tensor[[4, 2, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 2, 8], pl.FP32]:
                x_tile: pl.Tile[[4, 2, 8], pl.FP32] = pl.load(x, [0, 0, 0], [4, 2, 8])
                y_tile: pl.Tile[[4, 2, 8], pl.FP32] = pl.tile.neg(x_tile)
                out_0: pl.Tensor[[4, 2, 8], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[4, 2, 8], pl.FP32]) -> pl.Tensor[[4, 2, 8], pl.FP32]:
                out_0: pl.Tensor[[4, 2, 8], pl.FP32] = pl.create_tensor([4, 2, 8], dtype=pl.FP32)
                y: pl.Tensor[[4, 2, 8], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[4, 2, 8], pl.FP32],
                out_0: pl.Out[pl.Tensor[[4, 2, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 2, 8], pl.FP32]:
                x_tile_nd: pl.Tile[[4, 2, 8], pl.FP32] = pl.load(x, [0, 0, 0], [4, 2, 8])
                x_tile: pl.Tile[[8, 8], pl.FP32] = pl.tile.reshape(x_tile_nd, [8, 8])
                y_tile: pl.Tile[[8, 8], pl.FP32] = pl.tile.neg(x_tile)
                y_tile_nd: pl.Tile[[4, 2, 8], pl.FP32] = pl.tile.reshape(y_tile, [4, 2, 8])
                out_0: pl.Tensor[[4, 2, 8], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[4, 2, 8], pl.FP32]) -> pl.Tensor[[4, 2, 8], pl.FP32]:
                out_0: pl.Tensor[[4, 2, 8], pl.FP32] = pl.create_tensor([4, 2, 8], dtype=pl.FP32)
                y: pl.Tensor[[4, 2, 8], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DScalarOps:
    """Test tile-scalar operations on ND tiles."""

    def test_muls_3d(self):
        """tile.muls on 3D tile [2, 3, 4] with scalar -> flattened to [6, 4]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.muls(x_tile, 2.0)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.muls(x_tile, 2.0)
                y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_adds_3d(self):
        """tile.adds on 3D tile [2, 4, 8] with scalar -> flattened to [8, 8]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 4, 8], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP32]:
                x_tile: pl.Tile[[2, 4, 8], pl.FP32] = pl.load(x, [0, 0, 0], [2, 4, 8])
                y_tile: pl.Tile[[2, 4, 8], pl.FP32] = pl.tile.adds(x_tile, 1.0)
                out_0: pl.Tensor[[2, 4, 8], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 4, 8], pl.FP32]) -> pl.Tensor[[2, 4, 8], pl.FP32]:
                out_0: pl.Tensor[[2, 4, 8], pl.FP32] = pl.create_tensor([2, 4, 8], dtype=pl.FP32)
                y: pl.Tensor[[2, 4, 8], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 4, 8], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 4, 8], pl.FP32] = pl.load(x, [0, 0, 0], [2, 4, 8])
                x_tile: pl.Tile[[8, 8], pl.FP32] = pl.tile.reshape(x_tile_nd, [8, 8])
                y_tile: pl.Tile[[8, 8], pl.FP32] = pl.tile.adds(x_tile, 1.0)
                y_tile_nd: pl.Tile[[2, 4, 8], pl.FP32] = pl.tile.reshape(y_tile, [2, 4, 8])
                out_0: pl.Tensor[[2, 4, 8], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 4, 8], pl.FP32]) -> pl.Tensor[[2, 4, 8], pl.FP32]:
                out_0: pl.Tensor[[2, 4, 8], pl.FP32] = pl.create_tensor([2, 4, 8], dtype=pl.FP32)
                y: pl.Tensor[[2, 4, 8], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DReduceOps:
    """Test reduction operations on ND tiles."""

    def test_tile_max_reduce_last_axis(self):
        """tile.max on 3D tile [2, 4, 8] with axis=2 -> axis=1 after flatten."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 4, 8], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 4, 1], pl.FP32]],
            ) -> pl.Tensor[[2, 4, 1], pl.FP32]:
                x_tile: pl.Tile[[2, 4, 8], pl.FP32] = pl.load(x, [0, 0, 0], [2, 4, 8])
                y_tile: pl.Tile[[2, 4, 1], pl.FP32] = pl.tile.max(x_tile, axis=2, keepdim=True)
                out_0: pl.Tensor[[2, 4, 1], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 4, 8], pl.FP32]) -> pl.Tensor[[2, 4, 1], pl.FP32]:
                out_0: pl.Tensor[[2, 4, 1], pl.FP32] = pl.create_tensor([2, 4, 1], dtype=pl.FP32)
                y: pl.Tensor[[2, 4, 1], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 4, 8], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 4, 1], pl.FP32]],
            ) -> pl.Tensor[[2, 4, 1], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 4, 8], pl.FP32] = pl.load(x, [0, 0, 0], [2, 4, 8])
                x_tile: pl.Tile[[8, 8], pl.FP32] = pl.tile.reshape(x_tile_nd, [8, 8])
                y_tile: pl.Tile[[8, 1], pl.FP32] = pl.tile.max(x_tile, axis=1, keepdim=True)
                y_tile_nd: pl.Tile[[2, 4, 1], pl.FP32] = pl.tile.reshape(y_tile, [2, 4, 1])
                out_0: pl.Tensor[[2, 4, 1], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 4, 8], pl.FP32]) -> pl.Tensor[[2, 4, 1], pl.FP32]:
                out_0: pl.Tensor[[2, 4, 1], pl.FP32] = pl.create_tensor([2, 4, 1], dtype=pl.FP32)
                y: pl.Tensor[[2, 4, 1], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_tile_min_reduce_non_last_axis_error(self):
        """tile.min with axis=1 on 3D tile -> CHECK error."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 1, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 1, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 1, 4], pl.FP32] = pl.tile.min(x_tile, axis=1, keepdim=True)
                out_0: pl.Tensor[[2, 1, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 1, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 1, 4], pl.FP32] = pl.create_tensor([2, 1, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 1, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        with pytest.raises(Exception, match="must reduce along the last axis"):
            passes.flatten_tile_nd_to_2d()(Before)


class TestFlattenTileNdTo2DChainedOps:
    """Test chained operations on ND tiles."""

    def test_chained_load_exp_add_muls_store(self):
        """Long chain: load -> exp -> add -> muls -> store on 3D tile."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                a_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.exp(x_tile)
                b_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(a_tile, x_tile)
                c_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.muls(b_tile, 0.5)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(c_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                a_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.exp(x_tile)
                b_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(a_tile, x_tile)
                c_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.muls(b_tile, 0.5)
                c_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(c_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(c_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_two_loads_sub_store(self):
        """Two 3D loads -> tile.sub -> store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[3, 4, 5], pl.FP32],
                y: pl.Tensor[[3, 4, 5], pl.FP32],
                out_0: pl.Out[pl.Tensor[[3, 4, 5], pl.FP32]],
            ) -> pl.Tensor[[3, 4, 5], pl.FP32]:
                x_tile: pl.Tile[[3, 4, 5], pl.FP32] = pl.load(x, [0, 0, 0], [3, 4, 5])
                y_tile: pl.Tile[[3, 4, 5], pl.FP32] = pl.load(y, [0, 0, 0], [3, 4, 5])
                z_tile: pl.Tile[[3, 4, 5], pl.FP32] = pl.tile.sub(x_tile, y_tile)
                out_0: pl.Tensor[[3, 4, 5], pl.FP32] = pl.store(z_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[3, 4, 5], pl.FP32],
                y: pl.Tensor[[3, 4, 5], pl.FP32],
            ) -> pl.Tensor[[3, 4, 5], pl.FP32]:
                out_0: pl.Tensor[[3, 4, 5], pl.FP32] = pl.create_tensor([3, 4, 5], dtype=pl.FP32)
                z: pl.Tensor[[3, 4, 5], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[3, 4, 5], pl.FP32],
                y: pl.Tensor[[3, 4, 5], pl.FP32],
                out_0: pl.Out[pl.Tensor[[3, 4, 5], pl.FP32]],
            ) -> pl.Tensor[[3, 4, 5], pl.FP32]:
                x_tile_nd: pl.Tile[[3, 4, 5], pl.FP32] = pl.load(x, [0, 0, 0], [3, 4, 5])
                x_tile: pl.Tile[[12, 5], pl.FP32] = pl.tile.reshape(x_tile_nd, [12, 5])
                y_tile_nd: pl.Tile[[3, 4, 5], pl.FP32] = pl.load(y, [0, 0, 0], [3, 4, 5])
                y_tile: pl.Tile[[12, 5], pl.FP32] = pl.tile.reshape(y_tile_nd, [12, 5])
                z_tile: pl.Tile[[12, 5], pl.FP32] = pl.tile.sub(x_tile, y_tile)
                z_tile_nd: pl.Tile[[3, 4, 5], pl.FP32] = pl.tile.reshape(z_tile, [3, 4, 5])
                out_0: pl.Tensor[[3, 4, 5], pl.FP32] = pl.store(z_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[3, 4, 5], pl.FP32],
                y: pl.Tensor[[3, 4, 5], pl.FP32],
            ) -> pl.Tensor[[3, 4, 5], pl.FP32]:
                out_0: pl.Tensor[[3, 4, 5], pl.FP32] = pl.create_tensor([3, 4, 5], dtype=pl.FP32)
                z: pl.Tensor[[3, 4, 5], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DHigherDims:
    """Test higher-dimensional tiles (5D+)."""

    def test_5d_tile(self):
        """5D tile [2, 2, 2, 2, 4] -> [16, 4]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 2, 2, 2, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 2, 2, 2, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 2, 2, 2, 4], pl.FP32] = pl.load(x, [0, 0, 0, 0, 0], [2, 2, 2, 2, 4])
                y_tile: pl.Tile[[2, 2, 2, 2, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32]) -> pl.Tensor[[2, 2, 2, 2, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32] = pl.create_tensor([2, 2, 2, 2, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 2, 2, 2, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 2, 2, 2, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 2, 2, 2, 4], pl.FP32] = pl.load(x, [0, 0, 0, 0, 0], [2, 2, 2, 2, 4])
                x_tile: pl.Tile[[16, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [16, 4])
                y_tile: pl.Tile[[16, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                y_tile_nd: pl.Tile[[2, 2, 2, 2, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 2, 2, 2, 4])
                out_0: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32]) -> pl.Tensor[[2, 2, 2, 2, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32] = pl.create_tensor([2, 2, 2, 2, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 2, 2, 2, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DMixedDims:
    """Test programs with mixed 2D and 3D tile operations."""

    def test_mixed_2d_and_3d_tiles(self):
        """Some tiles 2D (unchanged), some 3D (flattened) in same function."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[32, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
                out_1: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                # 3D tile -> should be flattened
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                a_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.exp(x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(a_tile, [0, 0, 0], out_0)
                # 2D tile -> should be unchanged
                y_tile: pl.Tile[[32, 64], pl.FP32] = pl.load(y, [0, 0], [32, 64])
                b_tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.add(y_tile, y_tile)
                out_1: pl.Tensor[[32, 64], pl.FP32] = pl.store(b_tile, [0, 0], out_1)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[32, 64], pl.FP32],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_1: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
                r: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, y, out_0, out_1)
                return r

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[32, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
                out_1: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                # 3D -> flattened
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                a_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.exp(x_tile)
                a_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(a_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(a_tile_nd, [0, 0, 0], out_0)
                # 2D -> unchanged
                y_tile: pl.Tile[[32, 64], pl.FP32] = pl.load(y, [0, 0], [32, 64])
                b_tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.add(y_tile, y_tile)
                out_1: pl.Tensor[[32, 64], pl.FP32] = pl.store(b_tile, [0, 0], out_1)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[32, 64], pl.FP32],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_1: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
                r: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, y, out_0, out_1)
                return r

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DMultipleStores:
    """Test multiple tile.store operations in same function."""

    def test_two_stores_same_shape(self):
        """Two separate load-compute-store chains on same 3D shape."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
                out_1: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                a_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(a_tile, [0, 0, 0], out_0)
                b_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_1: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(b_tile, [0, 0, 0], out_1)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_1: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                r: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0, out_1)
                return r

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
                out_1: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                a_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                a_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(a_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(a_tile_nd, [0, 0, 0], out_0)
                b_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                b_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(b_tile, [2, 3, 4])
                out_1: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(b_tile_nd, [0, 0, 0], out_1)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_1: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                r: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0, out_1)
                return r

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DMultipleFunctions:
    """Test programs with multiple InCore functions."""

    def test_multiple_incore_functions(self):
        """Two InCore functions with 3D tiles: both get transformed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def incore_a(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def incore_b(
                self,
                x: pl.Tensor[[3, 4, 5], pl.FP32],
                out_0: pl.Out[pl.Tensor[[3, 4, 5], pl.FP32]],
            ) -> pl.Tensor[[3, 4, 5], pl.FP32]:
                x_tile: pl.Tile[[3, 4, 5], pl.FP32] = pl.load(x, [0, 0, 0], [3, 4, 5])
                y_tile: pl.Tile[[3, 4, 5], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_0: pl.Tensor[[3, 4, 5], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[3, 4, 5], pl.FP32],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_a: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_b: pl.Tensor[[3, 4, 5], pl.FP32] = pl.create_tensor([3, 4, 5], dtype=pl.FP32)
                ra: pl.Tensor[[2, 3, 4], pl.FP32] = self.incore_a(x, out_a)
                _rb: pl.Tensor[[3, 4, 5], pl.FP32] = self.incore_b(y, out_b)
                return ra

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def incore_a(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def incore_b(
                self,
                x: pl.Tensor[[3, 4, 5], pl.FP32],
                out_0: pl.Out[pl.Tensor[[3, 4, 5], pl.FP32]],
            ) -> pl.Tensor[[3, 4, 5], pl.FP32]:
                x_tile_nd: pl.Tile[[3, 4, 5], pl.FP32] = pl.load(x, [0, 0, 0], [3, 4, 5])
                x_tile: pl.Tile[[12, 5], pl.FP32] = pl.tile.reshape(x_tile_nd, [12, 5])
                y_tile: pl.Tile[[12, 5], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                y_tile_nd: pl.Tile[[3, 4, 5], pl.FP32] = pl.tile.reshape(y_tile, [3, 4, 5])
                out_0: pl.Tensor[[3, 4, 5], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                y: pl.Tensor[[3, 4, 5], pl.FP32],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_a: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                out_b: pl.Tensor[[3, 4, 5], pl.FP32] = pl.create_tensor([3, 4, 5], dtype=pl.FP32)
                ra: pl.Tensor[[2, 3, 4], pl.FP32] = self.incore_a(x, out_a)
                _rb: pl.Tensor[[3, 4, 5], pl.FP32] = self.incore_b(y, out_b)
                return ra

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DFull:
    """Test tile.full on ND tiles."""

    def test_tile_full_3d_flattened(self):
        """tile.full([2, 3, 4], value=0.0) -> tile.full([6, 4], value=0.0)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                z_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.full([2, 3, 4], dtype=pl.FP32, value=0.0)
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, z_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                z_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.full([6, 4], dtype=pl.FP32, value=0.0)
                y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, z_tile)
                y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DFunctionTypes:
    """Test AIC/AIV function types (specialized InCore variants)."""

    def test_aic_function_transformed(self):
        """AIC function with 3D tiles is transformed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC)
            def aic_func(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.aic_func(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC)
            def aic_func(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.aic_func(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_aiv_function_transformed(self):
        """AIV function with 3D tiles is transformed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def aiv_func(
                self,
                x: pl.Tensor[[4, 2, 8], pl.FP32],
                out_0: pl.Out[pl.Tensor[[4, 2, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 2, 8], pl.FP32]:
                x_tile: pl.Tile[[4, 2, 8], pl.FP32] = pl.load(x, [0, 0, 0], [4, 2, 8])
                y_tile: pl.Tile[[4, 2, 8], pl.FP32] = pl.tile.exp(x_tile)
                out_0: pl.Tensor[[4, 2, 8], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[4, 2, 8], pl.FP32]) -> pl.Tensor[[4, 2, 8], pl.FP32]:
                out_0: pl.Tensor[[4, 2, 8], pl.FP32] = pl.create_tensor([4, 2, 8], dtype=pl.FP32)
                y: pl.Tensor[[4, 2, 8], pl.FP32] = self.aiv_func(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def aiv_func(
                self,
                x: pl.Tensor[[4, 2, 8], pl.FP32],
                out_0: pl.Out[pl.Tensor[[4, 2, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 2, 8], pl.FP32]:
                x_tile_nd: pl.Tile[[4, 2, 8], pl.FP32] = pl.load(x, [0, 0, 0], [4, 2, 8])
                x_tile: pl.Tile[[8, 8], pl.FP32] = pl.tile.reshape(x_tile_nd, [8, 8])
                y_tile: pl.Tile[[8, 8], pl.FP32] = pl.tile.exp(x_tile)
                y_tile_nd: pl.Tile[[4, 2, 8], pl.FP32] = pl.tile.reshape(y_tile, [4, 2, 8])
                out_0: pl.Tensor[[4, 2, 8], pl.FP32] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[4, 2, 8], pl.FP32]) -> pl.Tensor[[4, 2, 8], pl.FP32]:
                out_0: pl.Tensor[[4, 2, 8], pl.FP32] = pl.create_tensor([4, 2, 8], dtype=pl.FP32)
                y: pl.Tensor[[4, 2, 8], pl.FP32] = self.aiv_func(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_group_function_unchanged(self):
        """Group function is not an InCore variant -> unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Group)
            def group_func(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                return x

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.group_func(x)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Before)


class TestFlattenTileNdTo2DDataTypes:
    """Test different data types."""

    def test_fp16_3d_tile(self):
        """FP16 3D tile [2, 4, 8] -> [8, 8]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 4, 8], pl.FP16],
                out_0: pl.Out[pl.Tensor[[2, 4, 8], pl.FP16]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                x_tile: pl.Tile[[2, 4, 8], pl.FP16] = pl.load(x, [0, 0, 0], [2, 4, 8])
                y_tile: pl.Tile[[2, 4, 8], pl.FP16] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 4, 8], pl.FP16] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 4, 8], pl.FP16]) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                out_0: pl.Tensor[[2, 4, 8], pl.FP16] = pl.create_tensor([2, 4, 8], dtype=pl.FP16)
                y: pl.Tensor[[2, 4, 8], pl.FP16] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 4, 8], pl.FP16],
                out_0: pl.Out[pl.Tensor[[2, 4, 8], pl.FP16]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                x_tile_nd: pl.Tile[[2, 4, 8], pl.FP16] = pl.load(x, [0, 0, 0], [2, 4, 8])
                x_tile: pl.Tile[[8, 8], pl.FP16] = pl.tile.reshape(x_tile_nd, [8, 8])
                y_tile: pl.Tile[[8, 8], pl.FP16] = pl.tile.add(x_tile, x_tile)
                y_tile_nd: pl.Tile[[2, 4, 8], pl.FP16] = pl.tile.reshape(y_tile, [2, 4, 8])
                out_0: pl.Tensor[[2, 4, 8], pl.FP16] = pl.store(y_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 4, 8], pl.FP16]) -> pl.Tensor[[2, 4, 8], pl.FP16]:
                out_0: pl.Tensor[[2, 4, 8], pl.FP16] = pl.create_tensor([2, 4, 8], dtype=pl.FP16)
                y: pl.Tensor[[2, 4, 8], pl.FP16] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DVerifier:
    """Test TileOps2D property verifier."""

    def test_verifier_passes_after_flatten(self):
        """TileOps2D verifier passes on correctly flattened program."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)

        # Verify TileOps2D property holds
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.TileOps2D)
        passes.verify_properties(props, After, "test_verifier")

    def test_verifier_fails_on_unflatten_program(self):
        """TileOps2D verifier fails on program with >2D tile ops."""

        @pl.program
        class Unflatten:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(y_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        # Verifying TileOps2D on unflatten program should fail
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.TileOps2D)
        with pytest.raises(Exception):
            passes.verify_properties(props, Unflatten, "test_verifier_fails")


class TestFlattenTileNdTo2DPassProperties:
    """Test pass property declarations."""

    def test_pass_properties(self):
        """Verify the pass declares correct required/produced properties."""
        p = passes.flatten_tile_nd_to_2d()

        required = p.get_required_properties()
        assert required.contains(passes.IRProperty.SSAForm)
        assert required.contains(passes.IRProperty.IncoreTileOps)

        produced = p.get_produced_properties()
        assert produced.contains(passes.IRProperty.SSAForm)
        assert produced.contains(passes.IRProperty.TileOps2D)

    def test_pass_name(self):
        """Verify the pass name."""
        p = passes.flatten_tile_nd_to_2d()
        assert p.get_name() == "FlattenTileNdTo2D"


class TestFlattenTileNdTo2DReduceAndCompute:
    """Test reduce followed by further computation on 3D tiles."""

    def test_sum_then_add_3d(self):
        """Load 3D -> sum(keepdim) -> add with another tile -> store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 1], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 1], pl.FP32]:
                x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                s_tile: pl.Tile[[2, 3, 1], pl.FP32] = pl.tile.sum(x_tile, axis=2, keepdim=True)
                r_tile: pl.Tile[[2, 3, 1], pl.FP32] = pl.tile.add(s_tile, s_tile)
                out_0: pl.Tensor[[2, 3, 1], pl.FP32] = pl.store(r_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 1], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 1], pl.FP32] = pl.create_tensor([2, 3, 1], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 1], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 1], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 1], pl.FP32]:
                x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
                x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
                s_tile: pl.Tile[[6, 1], pl.FP32] = pl.tile.sum(x_tile, axis=1, keepdim=True)
                r_tile: pl.Tile[[6, 1], pl.FP32] = pl.tile.add(s_tile, s_tile)
                r_tile_nd: pl.Tile[[2, 3, 1], pl.FP32] = pl.tile.reshape(r_tile, [2, 3, 1])
                out_0: pl.Tensor[[2, 3, 1], pl.FP32] = pl.store(r_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 1], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 1], pl.FP32] = pl.create_tensor([2, 3, 1], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 1], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_create_full_add_chain(self):
        """tile.create + tile.full + tile.add chain on 3D tiles."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                a_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.create([2, 3, 4], dtype=pl.FP32)
                b_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.full([2, 3, 4], dtype=pl.FP32, value=1.0)
                c_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(a_tile, b_tile)
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(c_tile, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                a_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.create([6, 4], dtype=pl.FP32)
                b_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.full([6, 4], dtype=pl.FP32, value=1.0)
                c_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(a_tile, b_tile)
                c_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(c_tile, [2, 3, 4])
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(c_tile_nd, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
