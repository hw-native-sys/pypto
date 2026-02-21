# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ConvertTensorToBlockOps pass."""

import pypto.language as pl
from pypto import ir, passes


class TestConvertTensorToBlockOps:
    """Test ConvertTensorToBlockOps pass."""

    def test_simple_elementwise_add(self):
        """tensor.add -> block.load + block.add + block.store."""

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
                out_0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.block.add(x_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], [64], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_two_tensor_inputs(self):
        """Two tensor parameters -> two block.load calls."""

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
                out_0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.load(y, [0], [64])
                z_tile: pl.Tile[[64], pl.FP32] = pl.block.add(x_tile, y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z_tile, [0], [64], out_0)
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

        After = passes.convert_tensor_to_block_ops()(Before)
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
                out_0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.block.add(x_tile, x_tile)
                z_tile: pl.Tile[[64], pl.FP32] = pl.block.mul(y_tile, y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z_tile, [0], [64], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return z

        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_orchestration_unchanged(self):
        """Non-InCore functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = passes.convert_tensor_to_block_ops()(Before)
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
                out_0: pl.Tensor[[32, 64], pl.FP16],
            ) -> pl.Tensor[[32, 64], pl.FP16]:
                x_tile: pl.Tile[[32, 64], pl.FP16] = pl.load(x, [0, 0], [32, 64])
                y_tile: pl.Tile[[32, 64], pl.FP16] = pl.block.add(x_tile, x_tile)
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.store(y_tile, [0, 0], [32, 64], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP16]) -> pl.Tensor[[32, 64], pl.FP16]:
                out_0: pl.Tensor[[32, 64], pl.FP16] = pl.create_tensor([32, 64], dtype=pl.FP16)
                y: pl.Tensor[[32, 64], pl.FP16] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_scalar_op_conversion(self):
        """tensor.add_scalar -> block.adds."""

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
                out_0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.block.adds(x_tile, 1.0)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], [64], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_exp_conversion(self):
        """tensor.exp -> block.exp."""

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
                out_0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.block.exp(x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], [64], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)


class TestNestedControlFlow:
    """Test ConvertTensorToBlockOps with nested control flow."""

    def test_incore_with_if_branch(self):
        """Tensor ops inside IfStmt in InCore -> block ops in both branches."""

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
                out_0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                if n == 0:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.block.add(x_tile, x_tile)
                    z: pl.Tile[[64], pl.FP32] = pl.yield_(y_tile)
                else:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.block.mul(x_tile, x_tile)
                    z: pl.Tile[[64], pl.FP32] = pl.yield_(y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z, [0], [64], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(n, x, out_0)
                return z

        After = passes.convert_tensor_to_block_ops()(Before)
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
                out_0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc_tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.block.add(acc_tile, acc_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], [64], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, out_0)
                    result = pl.yield_(y)
                return result

        After = passes.convert_tensor_to_block_ops()(Before)
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
                out_0: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc_tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                if n == 0:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.block.add(acc_tile, acc_tile)
                    z: pl.Tile[[64], pl.FP32] = pl.yield_(y_tile)
                else:
                    y_tile: pl.Tile[[64], pl.FP32] = pl.block.mul(acc_tile, acc_tile)
                    z: pl.Tile[[64], pl.FP32] = pl.yield_(y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z, [0], [64], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(3, init_values=(x,)):
                    out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, out_0)
                    result = pl.yield_(z)
                return result

        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)
