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


def _prepare(program):
    """Apply prerequisite passes: ConvertToSSA -> OutlineIncoreScopes."""
    program = passes.convert_to_ssa()(program)
    program = passes.outline_incore_scopes()(program)
    return program


class TestConvertTensorToBlockOps:
    """Test ConvertTensorToBlockOps pass."""

    def test_simple_elementwise_add(self):
        """tensor.add -> block.load + block.add + block.store."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
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

        Before = _prepare(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_two_tensor_inputs(self):
        """Two tensor parameters -> two block.load calls."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    z: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
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

        Before = _prepare(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_chained_ops(self):
        """Sequential tensor ops -> correct substitution chain."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
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

        Before = _prepare(Before)
        Expected = passes.convert_to_ssa()(Expected)
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

        Before = passes.convert_to_ssa()(Before)
        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Before)

    def test_2d_tensor(self):
        """2D tensor -> correct offsets and shapes for load/store."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[32, 64], pl.FP16]) -> pl.Tensor[[32, 64], pl.FP16]:
                with pl.incore():
                    y: pl.Tensor[[32, 64], pl.FP16] = pl.add(x, x)
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

        Before = _prepare(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.convert_tensor_to_block_ops()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_end_to_end_pipeline(self):
        """Full pipeline: ConvertToSSA -> OutlineIncoreScopes -> ConvertTensorToBlockOps."""

        @pl.program
        class Input:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Apply full pipeline
        result = passes.convert_to_ssa()(Input)
        result = passes.outline_incore_scopes()(result)
        result = passes.convert_tensor_to_block_ops()(result)

        # Verify the result has the expected structure by printing
        text = ir.python_print(result)
        assert "block.load" in text
        assert "block.add" in text
        assert "block.store" in text
        assert "tensor.create" in text

    def test_scalar_op_conversion(self):
        """tensor.add_scalar -> block.adds."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return y

        Before = _prepare(Before)
        After = passes.convert_tensor_to_block_ops()(Before)
        text = ir.python_print(After)
        assert "block.load" in text
        assert "block.adds" in text
        assert "block.store" in text

    def test_exp_conversion(self):
        """tensor.exp -> block.exp."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                return y

        Before = _prepare(Before)
        After = passes.convert_tensor_to_block_ops()(Before)
        text = ir.python_print(After)
        assert "block.load" in text
        assert "block.exp" in text
        assert "block.store" in text
