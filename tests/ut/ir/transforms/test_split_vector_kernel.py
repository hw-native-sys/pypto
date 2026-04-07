# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SplitVectorKernel pass."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend950 backend before each test and reset afterward."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def _run_split_vector_kernel(program):
    """Run convert_to_ssa then split_vector_kernel (without verification)."""
    ssa = passes.convert_to_ssa()(program)
    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.split_vector_kernel())
    ctx = passes.PassContext([], passes.VerificationLevel.NONE)
    with ctx:
        return pipeline.run(ssa)


def _assert_split_matches_expected(before_program, expected_program):
    actual = _run_split_vector_kernel(before_program)
    ir.assert_structural_equal(actual, passes.convert_to_ssa()(expected_program))


class TestSplitVectorKernelUpDown:
    """Tests for SplitMode.UP_DOWN (halve height, dim 0)."""

    def test_infers_split_mode_from_cross_core_pipe_ops(self):
        """Cross-core pipe split=1 should infer function split and trigger updown lowering."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC)
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(type=pl.FunctionType.AIV)
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=1
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INT64] = pl.tile.get_subblock_idx()
                z_vec: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0 + subblock_idx * 8, 0], out_0)
                return out_0_store

        _assert_split_matches_expected(Before, Expected)

    def test_tpop_shape_halved_and_store_offset_adjusted(self):
        """tpop result shape height halved and store offset dim0 adjusted."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INT64] = pl.tile.get_subblock_idx()
                z_vec: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0 + subblock_idx * 8, 0], out_0)
                return out_0_store

        _assert_split_matches_expected(Before, Expected)

    def test_load_shape_halved_and_offset_adjusted(self):
        """tile.load in AIV: shape halved, offset adjusted in split dim (includes add of halved tiles)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self, data: pl.Tensor[[16, 128], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                prev: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                pop_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                result: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.add(prev, pop_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self, data: pl.Tensor[[16, 128], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INT64] = pl.tile.get_subblock_idx()
                prev: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0 + subblock_idx * 8, 0], [8, 128], target_memory=pl.MemorySpace.Vec
                )
                pop_tile: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                result: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.add(prev, pop_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(
                    result, [0 + subblock_idx * 8, 0], out_0
                )
                return out_0_store

        _assert_split_matches_expected(Before, Expected)

    def test_injected_subblock_idx_avoids_name_collision(self):
        """Injected lane temp should pick a fresh name when subblock_idx already exists."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self,
                subblock_idx: pl.Tensor[[16, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                prev: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    subblock_idx, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                pop_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                result: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.add(prev, pop_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self,
                subblock_idx: pl.Tensor[[16, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx__ssa_v0: pl.Scalar[pl.INT64] = pl.tile.get_subblock_idx()
                prev: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    subblock_idx,
                    [0 + subblock_idx__ssa_v0 * 8, 0],
                    [8, 128],
                    target_memory=pl.MemorySpace.Vec,
                )
                pop_tile: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                result: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.add(prev, pop_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(
                    result, [0 + subblock_idx__ssa_v0 * 8, 0], out_0
                )
                return out_0_store

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_when_none(self):
        """Functions with no split should not be modified."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main_aiv(
                self,
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

        result = _run_split_vector_kernel(Before)
        ir.assert_structural_equal(result, passes.convert_to_ssa()(Before))

    def test_aic_tpop_from_aiv_keeps_full_tile_shape(self):
        """AIC tpop_from_aiv must not halve tile shape (cube still consumes full operand)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16]):
                a_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat, pl.TileView()] = pl.tpop_from_aiv(
                    split=0
                )
                pl.tfree_to_aiv(a_tile)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16]):
                a_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat, pl.TileView()] = pl.tpop_from_aiv(
                    split=1
                )
                pl.tfree_to_aiv(a_tile)

        _assert_split_matches_expected(Before, Expected)


class TestSplitVectorKernelLeftRight:
    """Tests for SplitMode.LEFT_RIGHT (halve width, dim 1)."""

    def test_tpop_shape_halved_and_store_offset_adjusted(self):
        """tpop result shape width halved and store offset dim1 adjusted."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=2)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INT64] = pl.tile.get_subblock_idx()
                z_vec: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=2)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(
                    z_vec, [0, 0 + subblock_idx * 64], out_0
                )
                return out_0_store

        _assert_split_matches_expected(Before, Expected)

    def test_load_shape_halved_left_right(self):
        """tile.load in AIV with LEFT_RIGHT: dim1 halved, offset dim1 adjusted."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=0)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aiv(
                self, data: pl.Tensor[[16, 128], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                prev: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                pop_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                result: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.add(prev, pop_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=2)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aiv(
                self, data: pl.Tensor[[16, 128], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INT64] = pl.tile.get_subblock_idx()
                prev: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0 + subblock_idx * 64], [16, 64], target_memory=pl.MemorySpace.Vec
                )
                pop_tile: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=2)
                result: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(prev, pop_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(
                    result, [0, 0 + subblock_idx * 64], out_0
                )
                return out_0_store

        _assert_split_matches_expected(Before, Expected)
