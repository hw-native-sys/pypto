# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for InferTileMemorySpace pass.

Note on test strategy:
  InferTileMemorySpace sets memory_space on TileType variables. We verify by
  printing the transformed program and checking that TileType annotations contain
  the expected pl.MemorySpace.<space> positional argument.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _assert_var_memory_space(printed: str, var_name: str, memory_space: str) -> None:
    """Assert a TileType variable has the expected memory_space in printed output.

    Searches for a line containing `var_name:` with a `pl.Tile[` annotation
    and checks that it includes `pl.MemorySpace.<memory_space>`.
    """
    for line in printed.split("\n"):
        if f"{var_name}:" in line and "pl.Tile[" in line:
            assert f", pl.MemorySpace.{memory_space}" in line, (
                f"Expected pl.MemorySpace.{memory_space} for '{var_name}', but line was: {line.strip()}"
            )
            return
    raise AssertionError(f"Variable '{var_name}' with pl.Tile type not found in printed output")


class TestInferTileMemorySpaceKwargOps:
    """Test memory_space inference for ops that read from target_memory kwarg."""

    def test_load_default_vec(self):
        """tile.load without target_memory kwarg defaults to Vec."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Vec")

    def test_load_with_mat_kwarg(self):
        """tile.load(target_memory=Mat) -> Mat."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(x_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Mat")

    def test_move_with_left_kwarg(self):
        """tile.move(target_memory=Left) -> Left."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(x_left, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Mat")
        _assert_var_memory_space(printed, "x_left", "Left")

    def test_create_default_vec(self):
        """tile.create without target_memory kwarg defaults to Vec."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t_tile: pl.Tile[[64], pl.FP32] = pl.tile.create([64], dtype=pl.FP32)
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(t_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "t_tile", "Vec")


class TestInferTileMemorySpaceCubeOps:
    """Test memory_space inference for cube ops (matmul, gemv, etc.)."""

    def test_matmul_gets_acc(self):
        """tile.matmul output -> Acc."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "z_tile", "Acc")
        # Inputs loaded to Vec by default
        _assert_var_memory_space(printed, "x_tile", "Vec")
        _assert_var_memory_space(printed, "y_tile", "Vec")

    def test_matmul_full_pipeline(self):
        """Full matmul pipeline: load->Mat, move->Left/Right, matmul->Acc."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_matmul(
                self,
                qi: pl.Tensor[[16, 128], pl.BF16],
                kj_t: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                qi_l1: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    qi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                kj_l1: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    kj_t, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                qi_l0a: pl.Tile[[16, 128], pl.BF16] = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
                kj_l0b: pl.Tile[[128, 128], pl.BF16] = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)
                sij: pl.Tile[[16, 128], pl.FP32] = pl.matmul(qi_l0a, kj_l0b)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(sij, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                qi: pl.Tensor[[16, 128], pl.BF16],
                kj_t: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                sij: pl.Tensor[[16, 128], pl.FP32] = self.qk_matmul(qi, kj_t, out_0)
                return sij

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "qi_l1", "Mat")
        _assert_var_memory_space(printed, "kj_l1", "Mat")
        _assert_var_memory_space(printed, "qi_l0a", "Left")
        _assert_var_memory_space(printed, "kj_l0b", "Right")
        _assert_var_memory_space(printed, "sij", "Acc")


class TestInferTileMemorySpaceOtherOps:
    """Test memory_space inference for other tile ops (default to Vec)."""

    def test_elementwise_inherits_vec(self):
        """tile.add(vec_tile, vec_tile) inherits Vec from inputs."""

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

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Vec")
        _assert_var_memory_space(printed, "y_tile", "Vec")

    def test_elementwise_after_matmul_gets_vec(self):
        """tile.add after matmul defaults to Vec (not inherited from matmul Acc)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                w_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.add(z_tile, z_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(w_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "z_tile", "Acc")
        _assert_var_memory_space(printed, "w_tile", "Vec")

    def test_chained_elementwise_inherits(self):
        """Chained elementwise ops: add then mul both inherit Vec."""

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
                z_tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(y_tile, y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Vec")
        _assert_var_memory_space(printed, "y_tile", "Vec")
        _assert_var_memory_space(printed, "z_tile", "Vec")


class TestInferTileMemorySpaceEdgeCases:
    """Test edge cases and pass-through behavior."""

    def test_orchestration_unchanged(self):
        """Non-InCore (Orchestration) functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Before)

    def test_multiple_incore_functions(self):
        """Multiple InCore functions are all processed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def incore_a(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def incore_b(
                self,
                y: pl.Tensor[[32], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32], pl.FP16]],
            ) -> pl.Tensor[[32], pl.FP16]:
                y_tile: pl.Tile[[32], pl.FP16] = pl.load(y, [0], [32])
                out_0: pl.Tensor[[32], pl.FP16] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[32], pl.FP16],
            ) -> pl.Tensor[[64], pl.FP32]:
                out_a: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                a: pl.Tensor[[64], pl.FP32] = self.incore_a(x, out_a)
                out_b: pl.Tensor[[32], pl.FP16] = pl.create_tensor([32], dtype=pl.FP16)
                _b: pl.Tensor[[32], pl.FP16] = self.incore_b(y, out_b)
                return a

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Vec")
        _assert_var_memory_space(printed, "y_tile", "Vec")

    def test_pass_is_idempotent(self):
        """Running the pass twice produces the same result."""

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

        first_pass = passes.infer_tile_memory_space()(Before)
        second_pass = passes.infer_tile_memory_space()(first_pass)
        ir.assert_structural_equal(first_pass, second_pass)


class TestTileTargetMemoryParsing:
    """Test that target_memory in type annotations is parsed correctly."""

    def test_parse_tile_with_target_memory_3arg(self):
        """pl.Tile[[shape], dtype, pl.MemorySpace.Vec] parses target_memory."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0], [64])
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        printed = ir.python_print(Program)
        _assert_var_memory_space(printed, "x_tile", "Vec")

    def test_parse_tile_with_target_memory_mat(self):
        """pl.Tile[[shape], dtype, pl.MemorySpace.Mat] parses target_memory=Mat."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(x_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        printed = ir.python_print(Program)
        _assert_var_memory_space(printed, "x_tile", "Mat")

    def test_printed_target_memory_format(self):
        """Verify printed output includes target_memory as positional arg in TileType."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        # Verify the type annotation in printed output contains MemorySpace as positional arg
        assert "pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec]" in printed


class TestInferTileMemorySpaceInheritOps:
    """Test memory_space inference for view/transform ops that inherit from input."""

    def test_reshape_inherits_vec(self):
        """tile.reshape inherits Vec memory space from input tile."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                reshaped: pl.Tile[[8, 8], pl.FP32] = pl.tile.reshape(x_tile, [8, 8])
                flat: pl.Tile[[64], pl.FP32] = pl.tile.reshape(reshaped, [64])
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(flat, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Vec")
        _assert_var_memory_space(printed, "reshaped", "Vec")
        _assert_var_memory_space(printed, "flat", "Vec")

    def test_slice_inherits_vec(self):
        """tile.slice inherits Vec memory space from input tile."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[32], pl.FP32]],
            ) -> pl.Tensor[[32], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                sliced: pl.Tile[[32], pl.FP32] = pl.tile.slice(x_tile, [32], [0])
                out_0: pl.Tensor[[32], pl.FP32] = pl.store(sliced, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                out_0: pl.Tensor[[32], pl.FP32] = pl.create_tensor([32], dtype=pl.FP32)
                y: pl.Tensor[[32], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Vec")
        _assert_var_memory_space(printed, "sliced", "Vec")

    def test_reshape_inherits_mat(self):
        """tile.reshape inherits Mat memory space from input loaded to Mat."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                reshaped: pl.Tile[[2048], pl.BF16] = pl.tile.reshape(x_tile, [2048])
                flat: pl.Tile[[16, 128], pl.BF16] = pl.tile.reshape(reshaped, [16, 128])
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(flat, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Mat")
        _assert_var_memory_space(printed, "reshaped", "Mat")
        _assert_var_memory_space(printed, "flat", "Mat")

    def test_slice_inherits_mat(self):
        """tile.slice inherits Mat memory space from Mat input."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                sliced: pl.Tile[[16, 64], pl.BF16] = pl.tile.slice(x_tile, [16, 64], [0, 0])
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.store(sliced, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 64], pl.BF16]:
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.create_tensor([16, 64], dtype=pl.BF16)
                y: pl.Tensor[[16, 64], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Mat")
        _assert_var_memory_space(printed, "sliced", "Mat")

    def test_chained_view_ops_inherit(self):
        """reshape(slice(load(Mat))) — all inherit Mat from the load."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                sliced: pl.Tile[[16, 64], pl.BF16] = pl.tile.slice(x_tile, [16, 64], [0, 0])
                reshaped: pl.Tile[[1024], pl.BF16] = pl.tile.reshape(sliced, [1024])
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.store(reshaped, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 64], pl.BF16]:
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.create_tensor([16, 64], dtype=pl.BF16)
                y: pl.Tensor[[16, 64], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Mat")
        _assert_var_memory_space(printed, "sliced", "Mat")
        _assert_var_memory_space(printed, "reshaped", "Mat")


class TestAutoMoveInsertion:
    """Test that InferTileMemorySpace auto-inserts tile.move for input mismatches."""

    def test_matmul_auto_moves_from_vec(self):
        """tile.matmul with Vec inputs -> auto-insert moves to Left/Right."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        # Auto-inserted moves
        _assert_var_memory_space(printed, "x_tile_Left", "Left")
        _assert_var_memory_space(printed, "y_tile_Right", "Right")
        # Matmul uses moved vars
        assert "pl.tile.matmul(x_tile_Left, y_tile_Right)" in printed
        _assert_var_memory_space(printed, "z_tile", "Acc")
        # Exactly 2 auto-inserted tile.move (x_tile->Left, y_tile->Right)
        assert printed.count("pl.tile.move") == 2

    def test_matmul_auto_moves_from_mat(self):
        """tile.matmul with Mat inputs -> auto-insert moves to Left/Right."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        _assert_var_memory_space(printed, "x_tile", "Mat")
        _assert_var_memory_space(printed, "y_tile", "Mat")
        _assert_var_memory_space(printed, "x_tile_Left", "Left")
        _assert_var_memory_space(printed, "y_tile_Right", "Right")
        assert "pl.tile.matmul(x_tile_Left, y_tile_Right)" in printed

    def test_no_move_when_already_correct(self):
        """No move inserted when input already in correct space."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_tile, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        # Only 2 tile.move (the original ones), no extra auto-inserted moves
        assert printed.count("pl.tile.move") == 2
        # Matmul uses the original moved vars
        assert "pl.tile.matmul(x_left, y_right)" in printed

    def test_store_no_move_for_vec(self):
        """tile.store accepts Vec — no move needed for Vec tile."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        # No tile.move should be inserted
        assert "pl.tile.move" not in printed

    def test_store_no_move_for_acc(self):
        """tile.store accepts Acc — no move needed for matmul output."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
                y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_tile, target_memory=pl.MemorySpace.Right)
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                z: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(x, y, out_0)
                return z

        After = passes.infer_tile_memory_space()(Before)
        printed = ir.python_print(After)
        # tile.store(z_tile, ...) — z_tile is Acc, which is allowed by store
        # Only the 2 original explicit moves, no extra move for store's input
        assert printed.count("pl.tile.move") == 2
        _assert_var_memory_space(printed, "z_tile", "Acc")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
