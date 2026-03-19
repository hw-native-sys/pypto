# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ConvertTensorToTileOps pass."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


class TestConvertTensorToTileOps:
    """Test ConvertTensorToTileOps pass."""

    def test_simple_elementwise_add(self):
        """tensor.add -> tile.load + tile.add + tile.store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
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

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_two_tensor_inputs(self):
        """Two tensor parameters -> two tile.load calls."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                return z

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, y)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.load(y, [0], [64])
                z_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z_tile, [0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_chained_ops(self):
        """Sequential tensor ops -> correct substitution chain."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, x_tile)
                z_tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(y_tile, y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return z

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_orchestration_unchanged(self):
        """Non-InCore functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Before)

    def test_2d_tensor(self):
        """2D tensor -> correct offsets and shapes for load/store."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[32, 64], pl.FP16]) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP16]) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP16]) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_scalar_op_conversion(self):
        """tensor.adds -> tile.adds."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.adds(x_tile, 1.0)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_exp_conversion(self):
        """tensor.exp -> tile.exp."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.exp(x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_neg_conversion(self):
        """tensor.neg -> tile.neg."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.neg(x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.neg(x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_recip_conversion(self):
        """tensor.recip -> tile.recip."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.recip(x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.recip(x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_sqrt_conversion(self):
        """tensor.sqrt -> tile.sqrt."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.sqrt(x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.sqrt(x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_row_expand_mul_conversion(self):
        """tensor.row_expand_mul -> tile.row_expand_mul."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.row_expand_mul(x, rv)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, rv)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                rv_tile: pl.Tile[[32, 1], pl.FP16] = pl.load(rv, [0, 0], [32, 1])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.row_expand_mul(x_tile, rv_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, rv, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_row_expand_div_conversion(self):
        """tensor.row_expand_div -> tile.row_expand_div."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.row_expand_div(x, rv)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, rv)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                rv_tile: pl.Tile[[32, 1], pl.FP16] = pl.load(rv, [0, 0], [32, 1])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.row_expand_div(x_tile, rv_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, rv, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_col_expand_mul_conversion(self):
        """tensor.col_expand_mul -> tile.col_expand_mul."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.col_expand_mul(x, cv)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, cv)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                cv_tile: pl.Tile[[1, 64], pl.FP16] = pl.load(cv, [0, 0], [1, 64])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.col_expand_mul(x_tile, cv_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, cv, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_conversion(self):
        """tensor.matmul -> tile.load(Mat) + tile.move(Left/Right) + tile.matmul.

        Verifies that tile.move calls do NOT contain transpose kwargs,
        and rhs tile.load carries transpose=False when b_trans is not set.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP16],
                rhs: pl.Tensor[[128, 64], pl.FP16],
            ) -> pl.Tensor[[16, 64], pl.FP16]:
                y: pl.Tensor[[16, 64], pl.FP16] = pl.matmul(lhs, rhs)
                return y

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP16],
                rhs: pl.Tensor[[128, 64], pl.FP16],
            ) -> pl.Tensor[[16, 64], pl.FP16]:
                y: pl.Tensor[[16, 64], pl.FP16] = self.main_incore_0(lhs, rhs)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP16],
                rhs: pl.Tensor[[128, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.FP16]],
            ) -> pl.Tensor[[16, 64], pl.FP16]:
                lhs_mat: pl.Tile[[16, 128], pl.FP16] = pl.load(
                    lhs, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_mat: pl.Tile[[128, 64], pl.FP16] = pl.load(
                    rhs, [0, 0], [128, 64], [128, 64], target_memory=pl.MemorySpace.Mat, transpose=False
                )
                y_tile: pl.Tile[[16, 64], pl.FP32] = pl.matmul(lhs_mat, rhs_mat)
                out_0: pl.Tensor[[16, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP16],
                rhs: pl.Tensor[[128, 64], pl.FP16],
            ) -> pl.Tensor[[16, 64], pl.FP16]:
                out_0: pl.Tensor[[16, 64], pl.FP16] = pl.create_tensor([16, 64], dtype=pl.FP16)
                y: pl.Tensor[[16, 64], pl.FP16] = self.main_incore_0(lhs, rhs, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_b_trans_conversion(self):
        """tensor.matmul(b_trans=True) -> tile.load(Mat, transpose=True) + tile.move + tile.matmul.

        Verifies that b_trans is moved from tile.move to tile.load for rhs,
        and tile.move calls have no transpose kwarg.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                y: pl.Tensor[[16, 128], pl.BF16] = pl.matmul(lhs, rhs, b_trans=True)
                return y

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(lhs, rhs)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                lhs_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    lhs, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    rhs, [0, 0], [128, 128], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                y_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(lhs_mat, rhs_mat)
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.BF16],
                rhs: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(lhs, rhs, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_assemble_tile_tile_then_cast_conversion(self):
        """tensor.create + tensor.assemble(tile,tile) + tensor.cast must not crash.

        Regression test: tensor.create → tile.create, so the subsequent
        tensor.assemble sees both args as tiles → tile.assemble (stays TileType).
        The following tensor.cast then sees a tile input and converts to tile.cast
        without error.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[1, 32], pl.FP32],
                b: pl.Tensor[[1, 32], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.BF16]:
                t: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                t = pl.assemble(t, a, [0, 0])
                t = pl.assemble(t, b, [0, 32])
                out: pl.Tensor[[1, 64], pl.BF16] = pl.cast(t, target_type=pl.BF16)
                return out

            @pl.function
            def main(
                self,
                a: pl.Tensor[[1, 32], pl.FP32],
                b: pl.Tensor[[1, 32], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.BF16]:
                out: pl.Tensor[[1, 64], pl.BF16] = self.main_incore_0(a, b)
                return out

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir_str = str(After)
        assert "tile.assemble" in ir_str
        assert "tile.cast" in ir_str

    def test_no_spurious_loads_for_explicit_tile_ops(self):
        """Regression test for #334: no redundant Vec loads when params are consumed by tile ops only.

        When an InCore function explicitly loads tensors to Mat space and uses
        tile.move/tile.matmul/tile.store (none of which are converted tensor ops),
        the pass must NOT insert extra Vec-space tile.load ops for the tensor parameters.
        The output IR must be structurally identical to the input IR.
        """

        @pl.program
        class QKMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_matmul(
                self,
                qi_0: pl.Tensor[[16, 128], pl.BF16],
                kj_t_0: pl.Tensor[[128, 128], pl.BF16],
                sij_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                qi_l1_0: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    qi_0, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                kj_l1_0: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    kj_t_0, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                qi_l0a_0: pl.Tile[[16, 128], pl.BF16] = pl.move(qi_l1_0, target_memory=pl.MemorySpace.Left)
                kj_l0b_0: pl.Tile[[128, 128], pl.BF16] = pl.move(kj_l1_0, target_memory=pl.MemorySpace.Right)
                sij_l0c_0: pl.Tile[[16, 128], pl.FP32] = pl.matmul(qi_l0a_0, kj_l0b_0)
                out_sij_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(sij_l0c_0, [0, 0], output_tensor=sij_0)
                return out_sij_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                qi_0: pl.Tensor[[16, 128], pl.BF16],
                kj_t_0: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_sij_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                out_sij_1: pl.Tensor[[16, 128], pl.FP32] = self.qk_matmul(qi_0, kj_t_0, out_sij_0)
                return out_sij_1

        After = passes.convert_tensor_to_tile_ops()(QKMatmulProgram)
        ir.assert_structural_equal(After, QKMatmulProgram)

    def test_row_expand_add_conversion(self):
        """tensor.row_expand_add -> tile.row_expand_add."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.row_expand_add(x, rv)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, rv)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                rv_tile: pl.Tile[[32, 1], pl.FP16] = pl.load(rv, [0, 0], [32, 1])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.row_expand_add(x_tile, rv_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, rv, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_row_expand_sub_conversion(self):
        """tensor.row_expand_sub -> tile.row_expand_sub."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.row_expand_sub(x, rv)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, rv)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                rv_tile: pl.Tile[[32, 1], pl.FP16] = pl.load(rv, [0, 0], [32, 1])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.row_expand_sub(x_tile, rv_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                rv: pl.Tensor[[32, 1], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, rv, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_col_expand_conversion(self):
        """tensor.col_expand -> tile.col_expand."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.col_expand(x, cv)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, cv)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                cv_tile: pl.Tile[[1, 64], pl.FP16] = pl.load(cv, [0, 0], [1, 64])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.col_expand(x_tile, cv_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, cv, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_col_expand_sub_conversion(self):
        """tensor.col_expand_sub -> tile.col_expand_sub."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.col_expand_sub(x, cv)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, cv)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                cv_tile: pl.Tile[[1, 64], pl.FP16] = pl.load(cv, [0, 0], [1, 64])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.col_expand_sub(x_tile, cv_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, cv, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_col_expand_div_conversion(self):
        """tensor.col_expand_div -> tile.col_expand_div."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.col_expand_div(x, cv)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, cv)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                cv_tile: pl.Tile[[1, 64], pl.FP16] = pl.load(cv, [0, 0], [1, 64])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.col_expand_div(x_tile, cv_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                cv: pl.Tensor[[1, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, cv, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_row_expand_conversion(self):
        """tensor.row_expand -> tile.row_expand."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = pl.row_expand(x)
                return y

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32, 64], pl.FP16]],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.tile.row_expand(x_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


class TestNestedControlFlow:
    """Test ConvertTensorToTileOps with nested control flow."""

    def test_incore_with_if_branch(self):
        """Tensor ops inside IfStmt in InCore -> tile ops in both branches."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                if n == 0:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z = pl.yield_(y)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                    z = pl.yield_(y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                z = self.main_incore_0(n, x)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                n: pl.Scalar[pl.INT64],
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                if n == 0:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, x_tile)
                    z = pl.yield_(y_tile)
                else:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                    z = pl.yield_(y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(n, x, out_0)
                return z

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_call_inside_for_loop(self):
        """Call to InCore function inside ForStmt -> tensor.create inside loop."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, acc: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(acc, acc)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc)
                    result = pl.yield_(y)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc_tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(acc_tile, acc_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, out_0)
                    result = pl.yield_(y)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_both_sides(self):
        """Both InCore (IfStmt) and orchestration (ForStmt) have nested control flow."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, acc: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                if n == 0:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(acc, acc)
                    z = pl.yield_(y)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(acc, acc)
                    z = pl.yield_(y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n)
                    result = pl.yield_(z)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc_tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                if n == 0:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(acc_tile, acc_tile)
                    z = pl.yield_(y_tile)
                else:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(acc_tile, acc_tile)
                    z = pl.yield_(y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, out_0)
                    result = pl.yield_(z)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_missing_conversion_raises_error(self):
        """TensorOp with no registered converter raises an error when encountered in InCore body."""
        span = ir.Span.unknown()
        tensor_type = ir.TensorType([4], DataType.FP32)

        x_param = ir.Var("x", tensor_type, span)
        call = ir.create_op_call("test.tensor_op_no_conv", [x_param], {}, span)
        y_var = ir.Var("y", tensor_type, span)
        body = ir.SeqStmts(
            [
                ir.AssignStmt(y_var, call, span),
                ir.ReturnStmt([y_var], span),
            ],
            span,
        )
        func = ir.Function("incore", [x_param], [tensor_type], body, span, ir.FunctionType.InCore)
        prog = ir.Program([func], "test_program", span)

        with pytest.raises(Exception, match="has no registered tile conversion"):
            passes.convert_tensor_to_tile_ops()(prog)


class TestGmLocalTensorConversion:
    """Test gm_tensor vs local_tensor differentiated conversion."""

    def test_gm_tensor_slice_to_tile_load(self):
        """gm_tensor.slice (function param) -> tile.load."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[16, 64], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                s: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.slice(x, [8, 32], [0, 0])
                y: pl.Tensor[[8, 32], pl.FP32] = pl.add(s, s)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[16, 64], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
            ) -> pl.Tensor[[8, 32], pl.FP32]:
                s_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(x, [0, 0], [8, 32])
                y_tile: pl.Tile[[8, 32], pl.FP32] = pl.tile.add(s_tile, s_tile)
                out_0: pl.Tensor[[8, 32], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 64], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                out_0: pl.Tensor[[8, 32], pl.FP32] = pl.create_tensor([8, 32], dtype=pl.FP32)
                y: pl.Tensor[[8, 32], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_local_tensor_slice_to_tile_slice(self):
        """local_tensor.slice (tensor.create result) -> tile.slice."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                t: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                s: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.slice(t, [8, 32], [0, 0])
                y: pl.Tensor[[8, 32], pl.FP32] = pl.add(s, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[8, 32], pl.FP32],
                out_0: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
            ) -> pl.Tensor[[8, 32], pl.FP32]:
                x_tile: pl.Tile[[8, 32], pl.FP32] = pl.load(x, [0, 0], [8, 32])
                t_tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.create([16, 64], dtype=pl.FP32)
                s_tile: pl.Tile[[8, 32], pl.FP32] = pl.tile.slice(t_tile, [8, 32], [0, 0])
                y_tile: pl.Tile[[8, 32], pl.FP32] = pl.tile.add(s_tile, x_tile)
                out_0: pl.Tensor[[8, 32], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                out_0: pl.Tensor[[8, 32], pl.FP32] = pl.create_tensor([8, 32], dtype=pl.FP32)
                y: pl.Tensor[[8, 32], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_consecutive_slice_converts_to_tile_slice(self):
        """Consecutive tensor.slice: first becomes tile.load, second becomes tile.slice."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                s1: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(x, [16, 32], [0, 0])
                s2: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(s1, [4, 8], [0, 0])
                return s2

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                y: pl.Tensor[[4, 8], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                s1_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(x, [0, 0], [16, 32])
                s2_tile: pl.Tile[[4, 8], pl.FP32] = pl.tile.slice(s1_tile, [4, 8], [0, 0])
                out_0: pl.Tensor[[4, 8], pl.FP32] = pl.store(s2_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                out_0: pl.Tensor[[4, 8], pl.FP32] = pl.create_tensor([4, 8], dtype=pl.FP32)
                y: pl.Tensor[[4, 8], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_triple_consecutive_slice(self):
        """Three consecutive tensor.slice: first becomes tile.load, rest become tile.slice."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[2, 4], pl.FP32]:
                s1: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(x, [32, 64], [0, 0])
                s2: pl.Tensor[[8, 16], pl.FP32] = pl.tensor.slice(s1, [8, 16], [0, 0])
                s3: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(s2, [2, 4], [0, 0])
                return s3

            @pl.function
            def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[2, 4], pl.FP32]:
                y: pl.Tensor[[2, 4], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 4], pl.FP32]:
                s1_tile: pl.Tile[[32, 64], pl.FP32] = pl.load(x, [0, 0], [32, 64])
                s2_tile: pl.Tile[[8, 16], pl.FP32] = pl.tile.slice(s1_tile, [8, 16], [0, 0])
                s3_tile: pl.Tile[[2, 4], pl.FP32] = pl.tile.slice(s2_tile, [2, 4], [0, 0])
                out_0: pl.Tensor[[2, 4], pl.FP32] = pl.store(s3_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[2, 4], pl.FP32]:
                out_0: pl.Tensor[[2, 4], pl.FP32] = pl.create_tensor([2, 4], dtype=pl.FP32)
                y: pl.Tensor[[2, 4], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_consecutive_slice_with_computation(self):
        """Consecutive slice result used in computation (tile.add)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                s1: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(x, [16, 32], [0, 0])
                s2: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(s1, [4, 8], [0, 0])
                y: pl.Tensor[[4, 8], pl.FP32] = pl.add(s2, s2)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                y: pl.Tensor[[4, 8], pl.FP32] = self.main_incore_0(x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                s1_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(x, [0, 0], [16, 32])
                s2_tile: pl.Tile[[4, 8], pl.FP32] = pl.tile.slice(s1_tile, [4, 8], [0, 0])
                y_tile: pl.Tile[[4, 8], pl.FP32] = pl.tile.add(s2_tile, s2_tile)
                out_0: pl.Tensor[[4, 8], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                out_0: pl.Tensor[[4, 8], pl.FP32] = pl.create_tensor([4, 8], dtype=pl.FP32)
                y: pl.Tensor[[4, 8], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_gm_tensor_read_stays_tensor_read(self):
        """gm_tensor.read (function param) stays as tensor.read, no Phase 1 load."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, config: pl.Tensor[[4], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                scale: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, scale)
                return y

            @pl.function
            def main(
                self, config: pl.Tensor[[4], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(config, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                config: pl.Tensor[[4], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                scale_tile: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.muls(x_tile, scale_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(
                self, config: pl.Tensor[[4], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(config, x, out_0)
                return y

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_local_tensor_read_to_tile_read(self):
        """local_tensor.read (tile from tensor.create) -> tile.read."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                t: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                v: pl.Scalar[pl.FP32] = pl.tensor.read(t, [0])
                return v

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v: pl.Scalar[pl.FP32] = self.main_incore_0(x)
                return v

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                t_tile: pl.Tile[[64], pl.FP32] = pl.tile.create([64], dtype=pl.FP32)
                v_tile: pl.Scalar[pl.FP32] = pl.tile.read(t_tile, [0])
                return v_tile

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v: pl.Scalar[pl.FP32] = self.main_incore_0(x)
                return v

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_gm_tensor_write_stays_tensor_write(self):
        """tensor.write to a gm_tensor (function parameter) stays as tensor.write."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                dst: pl.Tensor[[4], pl.FP32],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                pl.tensor.write(dst, [0], val)
                return val

            @pl.function
            def main(
                self,
                dst: pl.Tensor[[4], pl.FP32],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                result: pl.Scalar[pl.FP32] = self.main_incore_0(dst, val)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                dst: pl.Tensor[[4], pl.FP32],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                pl.tensor.write(dst, [0], val)
                return val

            @pl.function
            def main(
                self,
                dst: pl.Tensor[[4], pl.FP32],
                val: pl.Scalar[pl.FP32],
            ) -> pl.Scalar[pl.FP32]:
                result: pl.Scalar[pl.FP32] = self.main_incore_0(dst, val)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_local_tensor_write_to_tile_write(self):
        """tensor.write to a local_tensor (result of tensor.add) converts to tile.write."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]
            ) -> pl.Scalar[pl.FP32]:
                t: pl.Tensor[[4], pl.FP32] = pl.add(a, b)
                val: pl.Scalar[pl.FP32] = pl.tensor.read(a, [0])
                pl.tensor.write(t, [0], val)
                v: pl.Scalar[pl.FP32] = pl.tensor.read(t, [0])
                return v

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v: pl.Scalar[pl.FP32] = self.main_incore_0(a, b)
                return v

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]
            ) -> pl.Scalar[pl.FP32]:
                a_tile: pl.Tile[[4], pl.FP32] = pl.load(a, [0], [4])
                b_tile: pl.Tile[[4], pl.FP32] = pl.load(b, [0], [4])
                t_tile: pl.Tile[[4], pl.FP32] = pl.tile.add(a_tile, b_tile)
                val: pl.Scalar[pl.FP32] = pl.tile.read(a_tile, [0])
                pl.tile.write(t_tile, [0], val)
                v: pl.Scalar[pl.FP32] = pl.tile.read(t_tile, [0])
                return v

            @pl.function
            def main(self, a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32]) -> pl.Scalar[pl.FP32]:
                v: pl.Scalar[pl.FP32] = self.main_incore_0(a, b)
                return v

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


class TestSliceMatmulConversion:
    """Test tensor.slice + tensor.matmul conversion patterns.

    When a tensor.slice result feeds into tensor.matmul, the slice should produce
    tile.load(Mat, transpose=...) instead of tile.load(Vec), and the matmul should
    skip its own load for that operand (using the tile directly for move + matmul).
    """

    def test_slice_then_matmul_no_trans(self):
        """slice + matmul (no trans) -> tile.load(Mat, transpose=False) + move + matmul."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                b_slice: pl.Tensor[[128, 64], pl.BF16] = pl.slice(b, [128, 64], [0, 0])
                result: pl.Tensor[[16, 64], pl.BF16] = pl.matmul(a, b_slice)
                return result

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                result: pl.Tensor[[16, 64], pl.BF16] = self.main_incore_0(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                b_slice_tile: pl.Tile[[128, 64], pl.BF16] = pl.load(
                    b, [0, 0], [128, 64], [128, 64], target_memory=pl.MemorySpace.Mat, transpose=False
                )
                lhs_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    a, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Mat, transpose=False
                )
                result_tile: pl.Tile[[16, 64], pl.FP32] = pl.matmul(lhs_mat, b_slice_tile)
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.store(result_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.create_tensor([16, 64], dtype=pl.BF16)
                result: pl.Tensor[[16, 64], pl.BF16] = self.main_incore_0(a, b, out_0)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_slice_then_matmul_btrans(self):
        """slice + matmul(b_trans=True) -> tile.load(Mat, transpose=True) with swapped shapes."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                q: pl.Tensor[[1, 128], pl.BF16],
                k_cache: pl.Tensor[[120, 128], pl.BF16],
            ) -> pl.Tensor[[1, 120], pl.BF16]:
                k_slice: pl.Tensor[[120, 128], pl.BF16] = pl.slice(k_cache, [120, 128], [0, 0])
                result: pl.Tensor[[1, 120], pl.BF16] = pl.matmul(q, k_slice, b_trans=True)
                return result

            @pl.function
            def main(
                self,
                q: pl.Tensor[[1, 128], pl.BF16],
                k_cache: pl.Tensor[[120, 128], pl.BF16],
            ) -> pl.Tensor[[1, 120], pl.BF16]:
                result: pl.Tensor[[1, 120], pl.BF16] = self.main_incore_0(q, k_cache)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                q: pl.Tensor[[1, 128], pl.BF16],
                k_cache: pl.Tensor[[120, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[1, 120], pl.BF16]],
            ) -> pl.Tensor[[1, 120], pl.BF16]:
                k_slice_tile: pl.Tile[[128, 120], pl.BF16] = pl.load(
                    k_cache,
                    [0, 0],
                    [128, 120],
                    [128, 120],
                    target_memory=pl.MemorySpace.Mat,
                    transpose=True,
                )
                lhs_mat: pl.Tile[[1, 128], pl.BF16] = pl.load(
                    q,
                    [0, 0],
                    [1, 128],
                    [1, 128],
                    target_memory=pl.MemorySpace.Mat,
                    transpose=False,
                )
                result_tile: pl.Tile[[1, 120], pl.FP32] = pl.matmul(lhs_mat, k_slice_tile)
                out_0: pl.Tensor[[1, 120], pl.BF16] = pl.store(result_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                q: pl.Tensor[[1, 128], pl.BF16],
                k_cache: pl.Tensor[[120, 128], pl.BF16],
            ) -> pl.Tensor[[1, 120], pl.BF16]:
                out_0: pl.Tensor[[1, 120], pl.BF16] = pl.create_tensor([1, 120], dtype=pl.BF16)
                result: pl.Tensor[[1, 120], pl.BF16] = self.main_incore_0(q, k_cache, out_0)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_slice_then_matmul_atrans(self):
        """slice + matmul(a_trans=True) -> tile.load(Mat, transpose=True) for lhs with swapped shapes."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[128, 16], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                a_slice: pl.Tensor[[128, 16], pl.BF16] = pl.slice(a, [128, 16], [0, 0])
                result: pl.Tensor[[16, 64], pl.BF16] = pl.matmul(a_slice, b, a_trans=True)
                return result

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 16], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                result: pl.Tensor[[16, 64], pl.BF16] = self.main_incore_0(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[128, 16], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                a_slice_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    a,
                    [0, 0],
                    [16, 128],
                    [16, 128],
                    target_memory=pl.MemorySpace.Mat,
                    transpose=True,
                )
                rhs_mat: pl.Tile[[128, 64], pl.BF16] = pl.load(
                    b,
                    [0, 0],
                    [128, 64],
                    [128, 64],
                    target_memory=pl.MemorySpace.Mat,
                    transpose=False,
                )
                result_tile: pl.Tile[[16, 64], pl.FP32] = pl.matmul(a_slice_tile, rhs_mat)
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.store(result_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 16], pl.BF16],
                b: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.create_tensor([16, 64], dtype=pl.BF16)
                result: pl.Tensor[[16, 64], pl.BF16] = self.main_incore_0(a, b, out_0)
                return result

        After = passes.convert_tensor_to_tile_ops()(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
