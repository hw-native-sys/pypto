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
from pypto.ir.printer import python_print


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

    def test_loop_iter_arg_keeps_split_tracking(self):
        """Loop iter_args seeded by halved tiles must keep split-aware store offsets."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self, data: pl.Tensor[[16, 128], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                accum: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                for i in pl.range(2):
                    out_0 = pl.store(accum, [0, 0], out_0)
                    pop_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = (
                        pl.tpop_from_aic(split=0)
                    )
                    accum = pl.add(accum, pop_tile)
                return out_0

        actual = _run_split_vector_kernel(Before)
        printed = python_print(actual)
        main_aiv = actual.get_function("main_aiv")
        assert main_aiv is not None
        loop_stmt = next(stmt for stmt in ir.flatten_to_stmts(main_aiv.body) if isinstance(stmt, ir.ForStmt))
        iter_arg_type = loop_stmt.iter_args[0].type
        assert isinstance(iter_arg_type, ir.TileType)
        assert isinstance(iter_arg_type.shape[0], ir.ConstInt)
        assert iter_arg_type.shape[0].value == 8
        assert "def main_aiv(" in printed
        assert "def main_aiv__aiv1(" not in printed
        assert "pl.tile.get_subblock_idx()" in printed
        assert "pl.tile.load(data__ssa_v0, [0 + subblock_idx * 8, 0], [8, 128], [8, 128]" in printed
        assert "pl.tile.tpop_from_aic(split=1)" in printed
        assert "pl.tile.store(accum__iter_v1, [0 + subblock_idx * 8, 0], out_0__iter_v1)" in printed

    def test_loop_return_var_keeps_split_tracking(self):
        """Loop return_vars fed by split tiles must keep split-aware store offsets after the loop."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self, data: pl.Tensor[[16, 128], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                accum: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                for i in pl.range(2):
                    pop_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = (
                        pl.tpop_from_aic(split=0)
                    )
                    accum = pl.add(accum, pop_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(accum, [0, 0], out_0)
                return out_0_store

        actual = _run_split_vector_kernel(Before)
        printed = python_print(actual)
        main_aiv = actual.get_function("main_aiv")
        assert main_aiv is not None
        loop_stmt = next(stmt for stmt in ir.flatten_to_stmts(main_aiv.body) if isinstance(stmt, ir.ForStmt))
        return_var_type = loop_stmt.return_vars[0].type
        assert isinstance(return_var_type, ir.TileType)
        assert isinstance(return_var_type.shape[0], ir.ConstInt)
        assert return_var_type.shape[0].value == 8
        assert "def main_aiv(" in printed
        assert "def main_aiv__aiv1(" not in printed
        assert "pl.tile.get_subblock_idx()" in printed
        assert "pl.tile.load(data__ssa_v0, [0 + subblock_idx * 8, 0], [8, 128], [8, 128]" in printed
        assert "pl.tile.tpop_from_aic(split=1)" in printed
        assert "pl.tile.store(accum__rv_v2, [0 + subblock_idx * 8, 0], out_0__ssa_v0)" in printed

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

    def test_for_stmt_tile_iter_arg_fp32_store_offset_adjusted(self):
        """ForStmt tile iter_arg FP32: return_var type halved and tile.store offset adjusted."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
                acc: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                    [16, 64], dtype=pl.FP32, value=0.0
                )
                for _, (acc_iter,) in pl.range(4, init_values=(acc,)):
                    pop_tile: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = (
                        pl.tpop_from_aic(split=0)
                    )
                    new_acc: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_iter, pop_tile)
                    acc_rv = pl.yield_(new_acc)
                out = pl.store(acc_rv, [0, 0], out_0)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INT64] = pl.tile.get_subblock_idx()
                acc: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                    [8, 64], dtype=pl.FP32, value=0.0
                )
                for _, (acc_iter,) in pl.range(4, init_values=(acc,)):
                    pop_tile: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                    new_acc: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_iter, pop_tile)
                    acc_rv = pl.yield_(new_acc)
                out = pl.store(acc_rv, [0 + subblock_idx * 8, 0], out_0)
                return out

        _assert_split_matches_expected(Before, Expected)

    def test_for_stmt_tile_iter_arg_store_inside_loop_offset_adjusted(self):
        """ForStmt tile iter_arg: tile.store inside loop body on iter_arg has offset adjusted."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
                acc: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                    [16, 64], dtype=pl.FP32, value=0.0
                )
                for _, (acc_iter,) in pl.range(4, init_values=(acc,)):
                    pl.store(acc_iter, [0, 0], out_0)
                    new_acc: pl.Tile[[16, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_iter, acc_iter)
                    acc_rv = pl.yield_(new_acc)
                out = pl.store(acc_rv, [0, 0], out_0)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INT64] = pl.tile.get_subblock_idx()
                acc: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                    [8, 64], dtype=pl.FP32, value=0.0
                )
                for _, (acc_iter,) in pl.range(4, init_values=(acc,)):
                    pl.store(acc_iter, [0 + subblock_idx * 8, 0], out_0)
                    new_acc: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_iter, acc_iter)
                    acc_rv = pl.yield_(new_acc)
                out = pl.store(acc_rv, [0 + subblock_idx * 8, 0], out_0)
                return out

        _assert_split_matches_expected(Before, Expected)

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

    def test_singleton_broadcast_tile_preserved(self):
        """Broadcast tile [1, 128] on split axis dim0 must stay unchanged under UP_DOWN."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 128], pl.FP32],
                gamma: pl.Tensor[[1, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                prev: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                gamma_tile: pl.Tile[[1, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    gamma, [0, 0], [1, 128], target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.col_expand_mul(prev, gamma_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0_store

        actual = _run_split_vector_kernel(Before)
        printed = python_print(actual)
        main_aiv = actual.get_function("main_aiv")
        assert main_aiv is not None
        assert "pl.tile.get_subblock_idx()" in printed
        assert "pl.tile.load(data__ssa_v0, [0 + subblock_idx * 8, 0], [8, 128], [8, 128]" in printed
        assert "pl.tile.load(gamma__ssa_v0, [0, 0], [1, 128], [1, 128]" in printed
        assert "pl.tile.col_expand_mul(" in printed
        assert "pl.tile.store(" in printed

    def test_reduce_on_split_axis_rejected(self):
        """Reduce on split axis (dim0 under UP_DOWN) must raise ValueError."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                prev: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                reduced: pl.Tile[[1, 128], pl.FP32, pl.MemorySpace.Vec] = pl.sum(prev, axis=0, keepdim=True)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(reduced, [0, 0], out_0)
                return out_0_store

        with pytest.raises(ValueError, match="reduces on the split axis"):
            _run_split_vector_kernel(Before)


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

    def test_singleton_broadcast_tile_preserved_left_right(self):
        """Broadcast tile [128, 1] on split axis dim1 must stay unchanged under LEFT_RIGHT."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 128], pl.FP32],
                gamma: pl.Tensor[[16, 1], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                prev: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                gamma_tile: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    gamma, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.row_expand_mul(prev, gamma_tile)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0_store

        actual = _run_split_vector_kernel(Before)
        printed = python_print(actual)
        main_aiv = actual.get_function("main_aiv")
        assert main_aiv is not None
        assert "pl.tile.get_subblock_idx()" in printed
        assert "pl.tile.load(data__ssa_v0, [0, 0 + subblock_idx * 64], [16, 64], [16, 64]" in printed
        assert "pl.tile.load(gamma__ssa_v0, [0, 0], [16, 1], [16, 1]" in printed
        assert "pl.tile.row_expand_mul(" in printed
        assert "pl.tile.store(" in printed
