# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for FlattenTileNdTo2D pass."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir import IRBuilder
from pypto.ir.op import tensor as tensor_ops
from pypto.ir.op import tile as tile_ops


def _load2d(
    tensor: ir.Expr,
    offsets: list,
    shapes: list,
    flat_shape: list,
    dtype: DataType,
) -> ir.Call:
    """Create tile.load Call with explicit 2D TileType (bypasses op registry type inference).

    tile.load hardware semantics: ND tensor -> 2D tile. The pass changes the result type
    directly to 2D without inserting a tile.reshape, preserving tile_view and memory_space
    from the original ND type.
    """
    nd_call = tile_ops.load(tensor, offsets, shapes, span=ir.Span.unknown())
    # Create a reference 2D tile.load to get the correct type (with proper tile_view + memory_space)
    ref_tensor = ir.Var("_ref", ir.TensorType(flat_shape, dtype), ir.Span.unknown())
    ref_offsets = [0] * len(flat_shape)
    ref_call = tile_ops.load(ref_tensor, ref_offsets, flat_shape, span=ir.Span.unknown())
    flat_type = cast(ir.TileType, ref_call.type)
    return ir.Call(nd_call.op, list(nd_call.args), nd_call.kwargs, flat_type, nd_call.span)


class TestFlattenTileNdTo2D:
    """Test FlattenTileNdTo2D pass."""

    def test_3d_tile_element_wise(self):
        """3D tile [2, 3, 4] with tile.add -> flattened to [6, 4]."""

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.add(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_4d_tile(self):
        """4D tile [2, 3, 4, 5] with tile.mul -> flattened to [24, 5]."""

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4, 5], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4, 5], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4, 5], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0, 0], [2, 3, 4, 5], [24, 5], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.mul(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0, 0], out_0, [2, 3, 4, 5]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4, 5], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4, 5], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4, 5], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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
        """Two 3D loads -> tile.add -> tile.store."""

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                y = f.param("y", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                y_tile = ib.let("y_tile", _load2d(y, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                z_tile = ib.let("z_tile", tile_ops.add(x_tile, y_tile))
                out_0_r = ib.let("out_0", tile_ops.store(z_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                y = f.param("y", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                z = ib.let("z", ir.Call(incore_gvar, [x, y, out_0], ir.Span.unknown()))
                ib.return_stmt(z)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                tmp = ib.let("tmp", tile_ops.create([6, 4], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.add(x_tile, tmp))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 1], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 1], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.sum(x_tile, axis=1, keepdim=True))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 3, 1]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 1], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 1], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.exp(x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([4, 2, 8], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([4, 2, 8], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([4, 2, 8], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [4, 2, 8], [8, 8], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.neg(x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [4, 2, 8]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([4, 2, 8], DataType.FP32))
                f.return_type(ir.TensorType([4, 2, 8], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([4, 2, 8], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.muls(x_tile, 2.0))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 4, 8], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 4, 8], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 4, 8], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 4, 8], [8, 8], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.adds(x_tile, 1.0))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 4, 8]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 4, 8], DataType.FP32))
                f.return_type(ir.TensorType([2, 4, 8], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 4, 8], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 4, 8], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 4, 1], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 4, 1], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 4, 8], [8, 8], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.max(x_tile, axis=1, keepdim=True))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 4, 1]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 4, 8], DataType.FP32))
                f.return_type(ir.TensorType([2, 4, 1], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 4, 1], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                a_tile = ib.let("a_tile", tile_ops.exp(x_tile))
                b_tile = ib.let("b_tile", tile_ops.add(a_tile, x_tile))
                c_tile = ib.let("c_tile", tile_ops.muls(b_tile, 0.5))
                out_0_r = ib.let("out_0", tile_ops.store(c_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([3, 4, 5], DataType.FP32))
                y = f.param("y", ir.TensorType([3, 4, 5], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([3, 4, 5], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([3, 4, 5], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [3, 4, 5], [12, 5], DataType.FP32))
                y_tile = ib.let("y_tile", _load2d(y, [0, 0, 0], [3, 4, 5], [12, 5], DataType.FP32))
                z_tile = ib.let("z_tile", tile_ops.sub(x_tile, y_tile))
                out_0_r = ib.let("out_0", tile_ops.store(z_tile, [0, 0, 0], out_0, [3, 4, 5]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([3, 4, 5], DataType.FP32))
                y = f.param("y", ir.TensorType([3, 4, 5], DataType.FP32))
                f.return_type(ir.TensorType([3, 4, 5], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([3, 4, 5], DataType.FP32))
                z = ib.let("z", ir.Call(incore_gvar, [x, y, out_0], ir.Span.unknown()))
                ib.return_stmt(z)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 2, 2, 2, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 2, 2, 2, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 2, 2, 2, 4], DataType.FP32))
                x_tile = ib.let(
                    "x_tile", _load2d(x, [0, 0, 0, 0, 0], [2, 2, 2, 2, 4], [16, 4], DataType.FP32)
                )
                y_tile = ib.let("y_tile", tile_ops.add(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0, 0, 0], out_0, [2, 2, 2, 2, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 2, 2, 2, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 2, 2, 2, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 2, 2, 2, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                y = f.param("y", ir.TensorType([32, 64], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                out_1 = f.param(
                    "out_1", ir.TensorType([32, 64], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                # 3D -> flattened
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                a_tile = ib.let("a_tile", tile_ops.exp(x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(a_tile, [0, 0, 0], out_0, [2, 3, 4]))
                # 2D -> unchanged
                y_tile = ib.let("y_tile", tile_ops.load(y, [0, 0], [32, 64]))
                b_tile = ib.let("b_tile", tile_ops.add(y_tile, y_tile))
                ib.let("out_1", tile_ops.store(b_tile, [0, 0], out_1))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                y = f.param("y", ir.TensorType([32, 64], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                out_1 = ib.let("out_1", tensor_ops.create([32, 64], DataType.FP32))
                r = ib.let("r", ir.Call(incore_gvar, [x, y, out_0, out_1], ir.Span.unknown()))
                ib.return_stmt(r)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                out_1 = f.param(
                    "out_1", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                a_tile = ib.let("a_tile", tile_ops.add(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(a_tile, [0, 0, 0], out_0, [2, 3, 4]))
                b_tile = ib.let("b_tile", tile_ops.mul(x_tile, x_tile))
                ib.let("out_1", tile_ops.store(b_tile, [0, 0, 0], out_1, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                out_1 = ib.let("out_1", tensor_ops.create([2, 3, 4], DataType.FP32))
                r = ib.let("r", ir.Call(incore_gvar, [x, out_0, out_1], ir.Span.unknown()))
                ib.return_stmt(r)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_a_gvar = prog.declare_function("incore_a")
            incore_b_gvar = prog.declare_function("incore_b")
            prog.declare_function("main")

            with ib.function("incore_a", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.add(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("incore_b", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([3, 4, 5], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([3, 4, 5], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([3, 4, 5], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [3, 4, 5], [12, 5], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.mul(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [3, 4, 5]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                y = f.param("y", ir.TensorType([3, 4, 5], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_a = ib.let("out_a", tensor_ops.create([2, 3, 4], DataType.FP32))
                out_b = ib.let("out_b", tensor_ops.create([3, 4, 5], DataType.FP32))
                ra = ib.let("ra", ir.Call(incore_a_gvar, [x, out_a], ir.Span.unknown()))
                _rb = ib.let("_rb", ir.Call(incore_b_gvar, [y, out_b], ir.Span.unknown()))
                ib.return_stmt(ra)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                z_tile = ib.let("z_tile", tile_ops.full([6, 4], DataType.FP32, 0.0))
                y_tile = ib.let("y_tile", tile_ops.add(x_tile, z_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            aic_gvar = prog.declare_function("aic_func")
            prog.declare_function("main")

            with ib.function("aic_func", type=ir.FunctionType.AIC) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.add(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 3, 4]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(aic_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            aiv_gvar = prog.declare_function("aiv_func")
            prog.declare_function("main")

            with ib.function("aiv_func", type=ir.FunctionType.AIV) as f:
                x = f.param("x", ir.TensorType([4, 2, 8], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([4, 2, 8], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([4, 2, 8], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [4, 2, 8], [8, 8], DataType.FP32))
                y_tile = ib.let("y_tile", tile_ops.exp(x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [4, 2, 8]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([4, 2, 8], DataType.FP32))
                f.return_type(ir.TensorType([4, 2, 8], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([4, 2, 8], DataType.FP32))
                y = ib.let("y", ir.Call(aiv_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 4, 8], DataType.FP16))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 4, 8], DataType.FP16), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 4, 8], DataType.FP16))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 4, 8], [8, 8], DataType.FP16))
                y_tile = ib.let("y_tile", tile_ops.add(x_tile, x_tile))
                out_0_r = ib.let("out_0", tile_ops.store(y_tile, [0, 0, 0], out_0, [2, 4, 8]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 4, 8], DataType.FP16))
                f.return_type(ir.TensorType([2, 4, 8], DataType.FP16))
                out_0 = ib.let("out_0", tensor_ops.create([2, 4, 8], DataType.FP16))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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

        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_0 = f.param(
                    "out_0", ir.TensorType([2, 3, 1], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 1], DataType.FP32))
                x_tile = ib.let("x_tile", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                s_tile = ib.let("s_tile", tile_ops.sum(x_tile, axis=1, keepdim=True))
                r_tile = ib.let("r_tile", tile_ops.add(s_tile, s_tile))
                out_0_r = ib.let("out_0", tile_ops.store(r_tile, [0, 0, 0], out_0, [2, 3, 1]))
                ib.return_stmt(out_0_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 1], DataType.FP32))
                out_0 = ib.let("out_0", tensor_ops.create([2, 3, 1], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out_0], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

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
                out_store: pl.Tensor[[2, 3, 4], pl.FP32] = pl.store(c_tile, [0, 0, 0], out_0, [2, 3, 4])
                return out_store

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 3, 4], pl.FP32] = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 3, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.flatten_tile_nd_to_2d()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenTileNdTo2DControlFlow:
    """Tests for FlattenTileNdTo2D with control-flow (ForStmt/IfStmt/WhileStmt).

    Regression coverage for #648: return_vars must be matched by identity, not name_hint.
    """

    def test_for_stmt_tile_iter_arg(self):
        """ForStmt with 3D tile iter_arg -> iter_arg and return_var flattened to 2D."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                t = pl.load(x, [0, 0, 0], [2, 3, 4])
                for i in pl.range(4):
                    t = pl.tile.add(t, t)
                out_0 = pl.store(t, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0 = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y = self.main_incore_0(x, out_0)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.flatten_tile_nd_to_2d()(Before)

        # Verify: all tile ops in InCore functions use ≤2D tiles
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.TileOps2D)
        passes.verify_properties(props, After, "test_for_stmt_tile_iter_arg")

    def test_for_stmt_tile_iter_arg_structural(self):
        """ForStmt with 3D tile iter_arg -> structural equality check."""

        # Before: SSA form with 3D tiles (built with IRBuilder for full control)
        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_p = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                t0 = ib.let("t", tile_ops.load(x, [0, 0, 0], [2, 3, 4]))
                i = ib.var("i", ir.ScalarType(DataType.INT64))
                with ib.for_loop(i, 0, 4, 1) as loop:
                    acc = loop.iter_arg("acc", t0)
                    loop.return_var("acc_out")
                    r = ib.let("r", tile_ops.add(acc, acc))
                    ib.emit(ir.YieldStmt([r], ir.Span.unknown()))
                acc_out = loop.output()
                out_r = ib.let("out_0", tile_ops.store(acc_out, [0, 0, 0], out_p))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Before = prog.get_result()

        After = passes.flatten_tile_nd_to_2d()(Before)

        # Expected: same structure but with 2D tiles
        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                out_p = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                t0 = ib.let("t", _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32))
                i = ib.var("i", ir.ScalarType(DataType.INT64))
                with ib.for_loop(i, 0, 4, 1) as loop:
                    acc = loop.iter_arg("acc", t0)
                    loop.return_var("acc_out")
                    r = ib.let("r", tile_ops.add(acc, acc))
                    ib.emit(ir.YieldStmt([r], ir.Span.unknown()))
                acc_out = loop.output()
                out_r = ib.let("out_0", tile_ops.store(acc_out, [0, 0, 0], out_p, [2, 3, 4]))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, out], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

        ir.assert_structural_equal(After, Expected)

    def test_if_stmt_tile_return_var(self):
        """IfStmt with 3D tile return_vars -> flattened to 2D via yield type matching."""

        # Before: SSA form with IfStmt that yields 3D tiles
        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                cond_param = f.param("cond", ir.ScalarType(DataType.BOOL))
                out_p = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))

                t0 = ib.let("t", tile_ops.load(x, [0, 0, 0], [2, 3, 4]))

                with ib.if_stmt(cond_param) as if_blk:
                    if_blk.return_var("rv", ir.TileType([2, 3, 4], DataType.FP32))
                    a = ib.let("a", tile_ops.add(t0, t0))
                    ib.emit(ir.YieldStmt([a], ir.Span.unknown()))
                    if_blk.else_()
                    b = ib.let("b", tile_ops.mul(t0, t0))
                    ib.emit(ir.YieldStmt([b], ir.Span.unknown()))
                rv = if_blk.output()

                out_r = ib.let("out_0", tile_ops.store(rv, [0, 0, 0], out_p))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                cond = f.param("cond", ir.ScalarType(DataType.BOOL))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, cond, out], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Before = prog.get_result()

        After = passes.flatten_tile_nd_to_2d()(Before)

        # Expected: same structure with 2D tiles
        ib = IRBuilder()
        with ib.program("main") as prog:
            incore_gvar = prog.declare_function("main_incore_0")
            prog.declare_function("main")

            with ib.function("main_incore_0", type=ir.FunctionType.InCore) as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                cond_param = f.param("cond", ir.ScalarType(DataType.BOOL))
                out_p = f.param(
                    "out_0", ir.TensorType([2, 3, 4], DataType.FP32), direction=ir.ParamDirection.Out
                )
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))

                load_call = _load2d(x, [0, 0, 0], [2, 3, 4], [6, 4], DataType.FP32)
                t0 = ib.let("t", load_call)

                with ib.if_stmt(cond_param) as if_blk:
                    # return_var type must match yield type (which includes tile_view from op_registry)
                    if_blk.return_var("rv", load_call.type)
                    a = ib.let("a", tile_ops.add(t0, t0))
                    ib.emit(ir.YieldStmt([a], ir.Span.unknown()))
                    if_blk.else_()
                    b = ib.let("b", tile_ops.mul(t0, t0))
                    ib.emit(ir.YieldStmt([b], ir.Span.unknown()))
                rv = if_blk.output()

                out_r = ib.let("out_0", tile_ops.store(rv, [0, 0, 0], out_p, [2, 3, 4]))
                ib.return_stmt(out_r)
            prog.add_function(f.get_result())

            with ib.function("main") as f:
                x = f.param("x", ir.TensorType([2, 3, 4], DataType.FP32))
                cond = f.param("cond", ir.ScalarType(DataType.BOOL))
                f.return_type(ir.TensorType([2, 3, 4], DataType.FP32))
                out = ib.let("out_0", tensor_ops.create([2, 3, 4], DataType.FP32))
                y = ib.let("y", ir.Call(incore_gvar, [x, cond, out], ir.Span.unknown()))
                ib.return_stmt(y)
            prog.add_function(f.get_result())
        Expected = prog.get_result()

        ir.assert_structural_equal(After, Expected)

    def test_while_stmt_tile_iter_arg(self):
        """WhileStmt with 3D tile iter_arg -> iter_arg and return_var flattened to 2D."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[2, 3, 4], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                t = pl.load(x, [0, 0, 0], [2, 3, 4])
                cond = True
                while cond:
                    t = pl.tile.add(t, t)
                    cond = False
                out_0 = pl.store(t, [0, 0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
                out_0 = pl.create_tensor([2, 3, 4], dtype=pl.FP32)
                y = self.main_incore_0(x, out_0)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.flatten_tile_nd_to_2d()(Before)

        # Verify: all tile ops in InCore functions use ≤2D tiles
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.TileOps2D)
        passes.verify_properties(props, After, "test_while_stmt_tile_iter_arg")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
