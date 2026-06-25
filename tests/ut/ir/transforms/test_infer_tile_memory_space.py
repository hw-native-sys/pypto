# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for InferTileMemorySpace pass.

Test strategy:
  Build a `Before` program, apply the pass, and compare the result to an
  explicitly-constructed `Expected` program using `assert_structural_equal`.
  Memory-space annotations are expressed via the 3-arg `pl.Tile[...]` form.
  Auto-inserted `tile.move` ops are expressed directly in `Expected`.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes


@pytest.fixture(autouse=True)
def _reset_backend():
    """Ensure no backend is configured so TileView inference is deterministic.

    InferTileMemorySpace consults backend-specific layout specs when a backend
    is configured. Tests in other files set Ascend and may share an xdist
    worker; resetting before each test guarantees the no-backend defaults
    assumed by the Expected programs.
    """
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


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

        @pl.program
        class Expected:
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

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                x_tile_V: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.move(x_tile, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(x_tile_V, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_tile, target_memory=pl.MemorySpace.Left
                )
                x_left_V: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.move(x_left, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(x_left_V, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create([64], dtype=pl.FP32)
                x_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(t_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)


class TestInferTileMemorySpaceCubeOps:
    """Test memory_space inference for cube ops (matmul, gemv, etc.)."""

    def test_matmul_gets_acc(self):
        """tile.matmul output -> Acc; inputs auto-moved to Left/Right."""

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(x, [0, 0], [16, 128])
                y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(y, [0, 0], [128, 128])
                x_tile_L: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_tile, target_memory=pl.MemorySpace.Left
                )
                y_tile_R: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_tile, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_tile_L, y_tile_R)
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
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_matmul(
                self,
                qi: pl.Tensor[[16, 128], pl.BF16],
                kj_t: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                qi_l1: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    qi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                kj_l1: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    kj_t, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                qi_l0a: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    qi_l1, target_memory=pl.MemorySpace.Left
                )
                kj_l0b: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    kj_l1, target_memory=pl.MemorySpace.Right
                )
                sij: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(qi_l0a, kj_l0b)
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
        ir.assert_structural_equal(After, Expected)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_elementwise_after_matmul_gets_vec(self):
        """tile.add after matmul: auto-insert move Acc->Vec before add."""

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_mat: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_mat: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_mat, target_memory=pl.MemorySpace.Left
                )
                y_right: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_mat, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_left, y_right)
                # ISA constraint: Acc→Vec data path lands ND in Vec; the inserted
                # move pins blayout=row_major, slayout=none_box on its kwargs.
                z_tile_V: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.move(
                    z_tile,
                    target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                w_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(z_tile_V, z_tile_V)
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
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0], [64])
                y_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(x_tile, x_tile)
                z_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.mul(y_tile, y_tile)
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(z_tile, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def incore_a(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0], [64])
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.InCore)
            def incore_b(
                self,
                y: pl.Tensor[[32], pl.FP16],
                out_0: pl.Out[pl.Tensor[[32], pl.FP16]],
            ) -> pl.Tensor[[32], pl.FP16]:
                y_tile: pl.Tile[[32], pl.FP16, pl.MemorySpace.Vec] = pl.load(y, [0], [32])
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
        ir.assert_structural_equal(After, Expected)

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
    """Test that target_memory in type annotations is parsed correctly (parser, not pass)."""

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

        incore = Program.get_function("main_incore_0")
        assert incore is not None
        assert isinstance(incore.body, ir.SeqStmts)
        x_tile_assign = incore.body.stmts[0]
        assert isinstance(x_tile_assign, ir.AssignStmt)
        assert isinstance(x_tile_assign.var.type, ir.TileType)
        assert x_tile_assign.var.type.memory_space == ir.MemorySpace.Vec

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

        incore = Program.get_function("main_incore_0")
        assert incore is not None
        assert isinstance(incore.body, ir.SeqStmts)
        x_tile_assign = incore.body.stmts[0]
        assert isinstance(x_tile_assign, ir.AssignStmt)
        assert isinstance(x_tile_assign.var.type, ir.TileType)
        assert x_tile_assign.var.type.memory_space == ir.MemorySpace.Mat


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0], [64])
                reshaped: pl.Tile[[8, 8], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(x_tile, [8, 8])
                flat: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(reshaped, [64])
                out_0: pl.Tensor[[64], pl.FP32] = pl.store(flat, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                out_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[32], pl.FP32]],
            ) -> pl.Tensor[[32], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0], [64])
                sliced: pl.Tile[[32], pl.FP32, pl.MemorySpace.Vec] = pl.tile.slice(x_tile, [32], [0])
                out_0: pl.Tensor[[32], pl.FP32] = pl.store(sliced, [0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                out_0: pl.Tensor[[32], pl.FP32] = pl.create_tensor([32], dtype=pl.FP32)
                y: pl.Tensor[[32], pl.FP32] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
            ) -> pl.Tensor[[16, 128], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                # Mat-implicit reshape result (col_major / row_major) — printer elides.
                reshaped: pl.Tile[[2048], pl.BF16, pl.MemorySpace.Mat] = pl.tile.reshape(x_tile, [2048])
                flat: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.tile.reshape(reshaped, [16, 128])
                # Move preserves Mat layout into Vec — non-Vec-implicit, so surfaced.
                flat_V: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.move(flat, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(flat_V, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 128], pl.BF16]:
                out_0: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                y: pl.Tensor[[16, 128], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                # tile.slice now propagates the source TileView (Mat-implicit
                # col_major / row_major) into the result so pto.subview is legal,
                # which means the printer elides the implicit annotation here.
                sliced: pl.Tile[[16, 64], pl.BF16, pl.MemorySpace.Mat] = pl.tile.slice(
                    x_tile, [16, 64], [0, 0]
                )
                # The move into Vec preserves the slice's Mat-style layout
                # (col_major / row_major) on the destination buffer; the printer
                # surfaces this because it differs from the Vec-implicit defaults.
                sliced_V: pl.Tile[
                    [16, 64],
                    pl.BF16,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.move(sliced, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.store(sliced_V, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 64], pl.BF16]:
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.create_tensor([16, 64], dtype=pl.BF16)
                y: pl.Tensor[[16, 64], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            ) -> pl.Tensor[[16, 64], pl.BF16]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                # Slice and reshape stay Mat-implicit — printer elides.
                sliced: pl.Tile[[16, 64], pl.BF16, pl.MemorySpace.Mat] = pl.tile.slice(
                    x_tile, [16, 64], [0, 0]
                )
                reshaped: pl.Tile[[1024], pl.BF16, pl.MemorySpace.Mat] = pl.tile.reshape(sliced, [1024])
                # Move preserves Mat layout into Vec — non-Vec-implicit, so surfaced.
                reshaped_V: pl.Tile[
                    [1024],
                    pl.BF16,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.move(reshaped, target_memory=pl.MemorySpace.Vec)
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.store(reshaped_V, [0, 0], out_0)
                return out_0

            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.BF16]) -> pl.Tensor[[16, 64], pl.BF16]:
                out_0: pl.Tensor[[16, 64], pl.BF16] = pl.create_tensor([16, 64], dtype=pl.BF16)
                y: pl.Tensor[[16, 64], pl.BF16] = self.main_incore_0(x, out_0)
                return y

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(x, [0, 0], [16, 128])
                y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(y, [0, 0], [128, 128])
                x_tile_L: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_tile, target_memory=pl.MemorySpace.Left
                )
                y_tile_R: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_tile, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_tile_L, y_tile_R)
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
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                x_tile_L: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_tile, target_memory=pl.MemorySpace.Left
                )
                y_tile_R: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_tile, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_tile_L, y_tile_R)
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
        ir.assert_structural_equal(After, Expected)

    def test_matmul_auto_moves_acc_through_mat_to_left(self):
        """Cube output (Acc) consumed as another matmul's Left input must be
        routed Acc -> Mat -> Left.

        Hardware constraint: a2a3 mem_graph declares
            mem_graph[Acc] = {Mat, DDR}
        in `src/backend/common/soc.cpp` -- no direct edge from Acc to Left or
        Right. PTOAS's `TMovOp::verify` enforces the same set: Acc->Left is
        not in `okPair` and is rejected at codegen with
        "expects a supported tmov address-space pair for this target".

        When a producer's TileType.memory_space_ is Acc and a downstream
        op requires the value in Left, InsertMoveStmt must split the move
        into two legal hops: Acc -> Mat (in mem_graph) then Mat -> Left
        (in mem_graph). Without the intermediate hop, codegen emits a
        direct Acc->Left `tile.move` that PTOAS rejects.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 128], pl.FP32] = pl.load(
                    a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                b_mat: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                # First matmul: `ab` lives in Acc (matmul output_memory=Acc).
                ab: pl.Tile[[16, 128], pl.FP32] = pl.matmul(a_mat, b_mat)
                c_mat: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    c, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                # Second matmul consumes `ab` as Left. The pass must route
                # Acc -> Mat -> Left, not emit a direct Acc -> Left tile.move.
                abc: pl.Tile[[16, 128], pl.FP32] = pl.matmul(ab, c_mat)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(abc, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                r: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(a, b, c, out_0)
                return r

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                a_mat: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Mat] = pl.load(
                    a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                b_mat: pl.Tile[[128, 128], pl.FP32, pl.MemorySpace.Mat] = pl.load(
                    b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                # Auto-inserted moves for the first matmul (Mat -> Left/Right).
                a_L: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Left] = pl.move(
                    a_mat, target_memory=pl.MemorySpace.Left
                )
                b_R: pl.Tile[[128, 128], pl.FP32, pl.MemorySpace.Right] = pl.move(
                    b_mat, target_memory=pl.MemorySpace.Right
                )
                ab: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(a_L, b_R)
                c_mat: pl.Tile[[128, 128], pl.FP32, pl.MemorySpace.Mat] = pl.load(
                    c, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                # The fix: Acc -> Mat -> Left chain for the second matmul lhs
                # (arg 0), because direct Acc -> Left is not in `mem_graph`.
                # Pass inserts moves in arg order, so the lhs chain comes
                # before the rhs single-hop move.
                ab_M: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Mat] = pl.move(
                    ab, target_memory=pl.MemorySpace.Mat
                )
                ab_L: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Left] = pl.move(
                    ab_M, target_memory=pl.MemorySpace.Left
                )
                # Auto-inserted Mat -> Right for the second matmul rhs (arg 1).
                c_R: pl.Tile[[128, 128], pl.FP32, pl.MemorySpace.Right] = pl.move(
                    c_mat, target_memory=pl.MemorySpace.Right
                )
                abc: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(ab_L, c_R)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(abc, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                r: pl.Tensor[[16, 128], pl.FP32] = self.main_incore_0(a, b, c, out_0)
                return r

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_matmul_moves_are_inserted_at_first_consumer(self):
        """Auto-inserted moves should be materialized at first constrained use.

        For two matmuls sharing `lhs_tile`, the lhs move is inserted once before
        the first matmul, while each rhs move is inserted just before its
        respective matmul.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[4, 128], pl.BF16],
                rhs0: pl.Tensor[[128, 64], pl.BF16],
                rhs1: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                lhs_tile: pl.Tile[[4, 128], pl.BF16] = pl.load(
                    lhs, [0, 0], [4, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs0_tile: pl.Tile[[128, 64], pl.BF16] = pl.load(
                    rhs0, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat
                )
                rhs1_tile: pl.Tile[[128, 64], pl.BF16] = pl.load(
                    rhs1, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat
                )
                _acc0: pl.Tile[[4, 64], pl.FP32] = pl.matmul(lhs_tile, rhs0_tile)
                acc1: pl.Tile[[4, 64], pl.FP32] = pl.matmul(lhs_tile, rhs1_tile)
                out_0_store: pl.Tensor[[4, 64], pl.FP32] = pl.store(acc1, [0, 0], out_0)
                return out_0_store

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[4, 128], pl.BF16],
                rhs0: pl.Tensor[[128, 64], pl.BF16],
                rhs1: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                out_0: pl.Tensor[[4, 64], pl.FP32] = pl.create_tensor([4, 64], dtype=pl.FP32)
                result: pl.Tensor[[4, 64], pl.FP32] = self.main_incore_0(lhs, rhs0, rhs1, out_0)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[4, 128], pl.BF16],
                rhs0: pl.Tensor[[128, 64], pl.BF16],
                rhs1: pl.Tensor[[128, 64], pl.BF16],
                out_0: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                lhs_tile: pl.Tile[[4, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    lhs, [0, 0], [4, 128], target_memory=pl.MemorySpace.Mat
                )
                rhs0_tile: pl.Tile[[128, 64], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    rhs0, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat
                )
                rhs1_tile: pl.Tile[[128, 64], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    rhs1, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat
                )
                lhs_tile_L: pl.Tile[[4, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    lhs_tile, target_memory=pl.MemorySpace.Left
                )
                rhs0_tile_R: pl.Tile[[128, 64], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    rhs0_tile, target_memory=pl.MemorySpace.Right
                )
                _acc0: pl.Tile[[4, 64], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(lhs_tile_L, rhs0_tile_R)
                rhs1_tile_R: pl.Tile[[128, 64], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    rhs1_tile, target_memory=pl.MemorySpace.Right
                )
                acc1: pl.Tile[[4, 64], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(lhs_tile_L, rhs1_tile_R)
                out_0_store: pl.Tensor[[4, 64], pl.FP32] = pl.store(acc1, [0, 0], out_0)
                return out_0_store

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[4, 128], pl.BF16],
                rhs0: pl.Tensor[[128, 64], pl.BF16],
                rhs1: pl.Tensor[[128, 64], pl.BF16],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                out_0: pl.Tensor[[4, 64], pl.FP32] = pl.create_tensor([4, 64], dtype=pl.FP32)
                result: pl.Tensor[[4, 64], pl.FP32] = self.main_incore_0(lhs, rhs0, rhs1, out_0)
                return result

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_tile, target_memory=pl.MemorySpace.Left
                )
                y_right: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_tile, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_left, y_right)
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
        ir.assert_structural_equal(After, Expected)

    def test_eval_stmt_consumer_collects_and_inserts_move(self):
        """EvalStmt consumers should also trigger required auto-inserted moves."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                value: pl.Scalar[pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
                    x, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                pl.tile.write(x_tile, [0, 0], value)
                return out_0

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                value: pl.Scalar[pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                x_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                x_tile_V: pl.Tile[
                    [16, 16],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
                ] = pl.move(x_tile, target_memory=pl.MemorySpace.Vec)
                pl.tile.write(x_tile_V, [0, 0], value)
                return out_0

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
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

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                x_left: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_tile, target_memory=pl.MemorySpace.Left
                )
                y_right: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_tile, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_left, y_right)
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
        ir.assert_structural_equal(After, Expected)


class TestInferTileMemorySpaceSSAAlias:
    """SSA-alias propagation added by the backward-demand-inference refactor.

    `y = x` where both sides are Tile-typed must forward x's resolved memory
    space onto y. The pl.DSL parser emits these aliases when eliding no-op
    wrappers (e.g. commented-out `pl.fillpad`), and earlier pipeline stages
    also produce them. Before the refactor, aliases without an explicit
    `pl.Mem.*` annotation left y with no memory_space set and later-phase
    consumers (MoveCollector, Phase 3) diverged from x.
    """

    def test_ssa_alias_inherits_memory_space_from_source(self):
        """`y = x` inherits x's resolved memory_space. tile.store demands Vec/Acc,
        so a Mat alias requires a Mat→Vec move before the store — present in
        both Before and Expected so the pass only has to propagate the alias's
        memory_space, isolating what this test verifies."""

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
                # The alias carries no memory_space annotation — Phase 1 must
                # copy Mat over from x_tile.
                x_alias: pl.Tile[[16, 128], pl.BF16] = x_tile
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_alias, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                return self.main_incore_0(x, y, out_0)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                x_alias: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = x_tile
                x_alias_L: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_alias, target_memory=pl.MemorySpace.Left
                )
                y_tile_R: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_tile, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_alias_L, y_tile_R)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                return self.main_incore_0(x, y, out_0)

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_ssa_alias_chain_feeds_matmul(self):
        """`y = x`, `z = y`: both aliases inherit x's memory_space. Verifies
        Phase 1 handles transitive SSA-alias chains in a single forward sweep."""

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
                alias_1: pl.Tile[[16, 128], pl.BF16] = x_tile
                alias_2: pl.Tile[[16, 128], pl.BF16] = alias_1
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(alias_2, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                return self.main_incore_0(x, y, out_0)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
                )
                alias_1: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = x_tile
                alias_2: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = alias_1
                y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
                )
                alias_2_L: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    alias_2, target_memory=pl.MemorySpace.Left
                )
                y_tile_R: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_tile, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(alias_2_L, y_tile_R)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 128], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                return self.main_incore_0(x, y, out_0)

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)


class TestInferTileMemorySpaceLoopCarried:
    """ForStmt accumulator back-propagation (analyzer VisitStmt_(ForStmt)).

    When a loop body writes a non-Vec space (e.g. Acc from matmul_acc) into a
    yielded tile, the analyzer copies that space onto the matching return_var,
    the iter_arg, AND the TileType init carrier underneath the iter_arg
    (cpp lines 213-247). This is what fixes the accumulator pattern where a
    conservative `tile.create -> Vec` init would otherwise leave the final
    `tile.store` reading a Vec tile and mislead ExpandMixedKernel.
    """

    def test_forstmt_accumulator_backprops_acc_to_create_init(self):
        """`acc0 = tile.create` (Vec default) carried into a matmul_acc loop is
        back-propagated to Acc.

        Derivation (no snapshot):
        - `matmul_acc` resolves its output to Acc via `set_output_memory(Acc)`,
          so the yielded `acc_next` is Acc.
        - ForStmt back-prop sets `return_vars_[0]` (r), `iter_args_[0]` (acc),
          and the init carrier `acc0` all to Acc (cpp 230, 238, 244-246).
        - `acc0 = tile.create` is a retargetable producer, so Phase 3 rewrites
          its absent/Vec `target_memory` kwarg to Acc and refreshes its type
          (cpp 508-547, doc step 4).
        - `matmul_acc` input 0 now reads an Acc `acc`, inputs 1/2 already
          Left/Right -> no moves inserted. `tile.store` accepts {Vec, Acc}, so
          the Acc `r` needs no move.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 32], pl.BF16],
                rhs: pl.Tensor[[32, 16], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                acc0: pl.Tile[[16, 16], pl.FP32] = pl.tile.create([16, 16], dtype=pl.FP32)
                lhs_m: pl.Tile[[16, 32], pl.BF16] = pl.load(
                    lhs, [0, 0], [16, 32], target_memory=pl.MemorySpace.Mat
                )
                rhs_m: pl.Tile[[32, 16], pl.BF16] = pl.load(
                    rhs, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat
                )
                lhs_l: pl.Tile[[16, 32], pl.BF16] = pl.move(lhs_m, target_memory=pl.MemorySpace.Left)
                rhs_r: pl.Tile[[32, 16], pl.BF16] = pl.move(rhs_m, target_memory=pl.MemorySpace.Right)
                for i, (acc,) in pl.range(0, 4, 1, init_values=(acc0,)):
                    acc_next: pl.Tile[[16, 16], pl.FP32] = pl.matmul_acc(acc, lhs_l, rhs_r)
                    r = pl.yield_(acc_next)
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(r, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 32], pl.BF16],
                rhs: pl.Tensor[[32, 16], pl.BF16],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                return self.main_incore_0(lhs, rhs, out_0)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                lhs: pl.Tensor[[16, 32], pl.BF16],
                rhs: pl.Tensor[[32, 16], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                # acc0 promoted Vec -> Acc; target_memory kwarg rewritten to Acc.
                acc0: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc
                )
                lhs_m: pl.Tile[[16, 32], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    lhs, [0, 0], [16, 32], target_memory=pl.MemorySpace.Mat
                )
                rhs_m: pl.Tile[[32, 16], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    rhs, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat
                )
                lhs_l: pl.Tile[[16, 32], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    lhs_m, target_memory=pl.MemorySpace.Left
                )
                rhs_r: pl.Tile[[32, 16], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    rhs_m, target_memory=pl.MemorySpace.Right
                )
                # iter_arg acc and return_var r both back-propagated to Acc.
                for i, (acc,) in pl.range(0, 4, 1, init_values=(acc0,)):
                    acc_next: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.matmul_acc(
                        acc, lhs_l, rhs_r
                    )
                    r = pl.yield_(acc_next)
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(r, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                lhs: pl.Tensor[[16, 32], pl.BF16],
                rhs: pl.Tensor[[32, 16], pl.BF16],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                return self.main_incore_0(lhs, rhs, out_0)

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_ifstmt_return_var_init_backprops_acc(self):
        """An IfStmt return_var used as a loop init is back-propagated to Acc.

        This targets the IfStmt-return_var fallback in ForStmt analysis: the
        analyzer never visits an IfStmt return_var as an AssignStmt, so it would
        otherwise keep its annotation (Mat here). When that var is the loop init
        whose iter_arg yields Acc, cpp lines 243-246 force `var_memory_[init_var]
        = Acc`, and the fallback at cpp 222-227 reads the yielded IfStmt-result's
        TileType annotation when resolving the loop return.

        Derivation (no snapshot):
        - Both branches yield a Mat tile, so `sel` (IfStmt return_var) is Mat.
        - The loop body's `matmul_acc(acc, lhs_l, rhs_r)` resolves to Acc, so
          `acc_next` (yield) is Acc.
        - Back-prop: r -> Acc, iter_arg acc -> Acc, and `sel` (the init carrier)
          -> Acc (cpp 244-246). Phase 3 rewrites `sel`'s Var type to Acc.
        - The inner branch loads stay Mat (unchanged). The pass does not insert
          a legalization move inside the if-branches for `sel` (IfStmt yields are
          invisible to MoveCollector); this fallback only forces the annotation.
        - `tile.store` reads the Acc `r` -> no move.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                lhs: pl.Tensor[[16, 32], pl.BF16],
                rhs: pl.Tensor[[32, 16], pl.BF16],
                flag: pl.Scalar[pl.INT32],
                out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                lhs_m: pl.Tile[[16, 32], pl.BF16] = pl.load(
                    lhs, [0, 0], [16, 32], target_memory=pl.MemorySpace.Mat
                )
                rhs_m: pl.Tile[[32, 16], pl.BF16] = pl.load(
                    rhs, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat
                )
                lhs_l: pl.Tile[[16, 32], pl.BF16] = pl.move(lhs_m, target_memory=pl.MemorySpace.Left)
                rhs_r: pl.Tile[[32, 16], pl.BF16] = pl.move(rhs_m, target_memory=pl.MemorySpace.Right)
                if flag > 0:
                    a: pl.Tile[[16, 16], pl.FP32] = pl.load(
                        x, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                    )
                    sel = pl.yield_(a)
                else:
                    b: pl.Tile[[16, 16], pl.FP32] = pl.load(
                        x, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                    )
                    sel = pl.yield_(b)
                for i, (acc,) in pl.range(0, 4, 1, init_values=(sel,)):
                    acc_next: pl.Tile[[16, 16], pl.FP32] = pl.matmul_acc(acc, lhs_l, rhs_r)
                    r = pl.yield_(acc_next)
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(r, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                lhs: pl.Tensor[[16, 32], pl.BF16],
                rhs: pl.Tensor[[32, 16], pl.BF16],
                flag: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                return self.main_incore_0(x, lhs, rhs, flag, out_0)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                lhs: pl.Tensor[[16, 32], pl.BF16],
                rhs: pl.Tensor[[32, 16], pl.BF16],
                flag: pl.Scalar[pl.INT32],
                out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                lhs_m: pl.Tile[[16, 32], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    lhs, [0, 0], [16, 32], target_memory=pl.MemorySpace.Mat
                )
                rhs_m: pl.Tile[[32, 16], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                    rhs, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat
                )
                lhs_l: pl.Tile[[16, 32], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    lhs_m, target_memory=pl.MemorySpace.Left
                )
                rhs_r: pl.Tile[[32, 16], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    rhs_m, target_memory=pl.MemorySpace.Right
                )
                if flag > 0:
                    a: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat] = pl.load(
                        x, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                    )
                    # sel (IfStmt return_var) forced to Acc by ForStmt back-prop.
                    sel: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.yield_(a)
                else:
                    b: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat] = pl.load(
                        x, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                    )
                    sel: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.yield_(b)
                for i, (acc,) in pl.range(0, 4, 1, init_values=(sel,)):
                    acc_next: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.matmul_acc(
                        acc, lhs_l, rhs_r
                    )
                    r = pl.yield_(acc_next)
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(r, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                lhs: pl.Tensor[[16, 32], pl.BF16],
                rhs: pl.Tensor[[32, 16], pl.BF16],
                flag: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                return self.main_incore_0(x, lhs, rhs, flag, out_0)

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)


class TestInferTileMemorySpaceDemandBackprop:
    """Backward demand propagation through inherit-input view chains (Phase 0).

    A `tile.load` with no `target_memory` kwarg feeding a `tile.slice`
    (`OutputMemoryInheritsInput`) into a `tile.matmul` (input 0 demands Left).
    Phase 0 records the Left demand on the slice output and propagates it back
    through the slice->load inherit-input edge onto the load.
    """

    def test_load_slice_matmul_demand_clamps_to_vec_then_moves(self):
        """Left demand back-propagated to a retargetable `tile.load` is clamped
        to Vec, with the Left/Right moves inserted at the matmul.

        Derivation (no snapshot):
        - Phase 0 records matmul input-0 demand Left on `x_sl`, then propagates
          it back through the slice->load inherit-input edge onto `x_tile`
          (cpp 106-157, doc 41-50).
        - Phase 1: `x_tile = tile.load` is retargetable with demand Left. The
          clamp keeps retargetable DDR producers in {Vec, Mat} (cpp 293-303,
          doc 76-79); Left is neither, so it falls through to Vec.
        - `x_sl = tile.slice` inherits Vec from `x_tile`; `y_tile = tile.load`
          (no demand) resolves to Vec.
        - matmul demands Left/Right but the operands are Vec, so Phase 2/3
          insert `x_sl_Left` and `y_tile_Right` moves before the matmul, which
          itself resolves to Acc. `tile.store` accepts Acc -> no move.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 256], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                x_tile: pl.Tile[[16, 256], pl.BF16] = pl.load(x, [0, 0], [16, 256])
                x_sl: pl.Tile[[16, 128], pl.BF16] = pl.tile.slice(x_tile, [16, 128], [0, 0])
                y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
                z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_sl, y_tile)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 256], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                return self.main_incore_0(x, y, out_0)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[16, 256], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                # Left demand clamped to Vec on the retargetable load.
                x_tile: pl.Tile[[16, 256], pl.BF16, pl.MemorySpace.Vec] = pl.load(x, [0, 0], [16, 256])
                x_sl: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Vec] = pl.tile.slice(
                    x_tile, [16, 128], [0, 0]
                )
                y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(y, [0, 0], [128, 128])
                x_sl_L: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
                    x_sl, target_memory=pl.MemorySpace.Left
                )
                y_tile_R: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
                    y_tile, target_memory=pl.MemorySpace.Right
                )
                z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_sl_L, y_tile_R)
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
                return out_0

            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 256], pl.BF16],
                y: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_0: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                return self.main_incore_0(x, y, out_0)

        After = passes.infer_tile_memory_space()(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
