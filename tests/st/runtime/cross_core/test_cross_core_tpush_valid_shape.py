# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime regression for preserving producer-side validShape through tpush."""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

ROWS = 16
COLS = 16
VALID_ROWS = 8
VALID_COLS = 16
SLOT_SIZE_BYTES = ROWS * COLS * 4
BUFFER_SIZE_BYTES = SLOT_SIZE_BYTES * 4


class C2VTpushValidShapeTestCase(PTOTestCase):
    """Cube sets validShape before tpush; vector pop observes the split valid region."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_c2v_tpush_valid_shape_updown_8x16"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [ROWS, COLS], DataType.BF16, init_value=1.0),
            TensorSpec("b", [ROWS, COLS], DataType.BF16, init_value=2.0),
            TensorSpec(
                "valid_shape",
                [2],
                DataType.INT64,
                init_value=torch.tensor([VALID_ROWS, VALID_COLS], dtype=torch.int64),
            ),
            TensorSpec("output", [ROWS, COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class C2VTpushValidShapeProgram:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def cube_producer(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                valid_rows: pl.Scalar[pl.INDEX],
                valid_cols: pl.Scalar[pl.INDEX],
            ):
                c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_consumer")
                pl.aic_initialize_pipe(
                    dir_mask=1,
                    slot_size=SLOT_SIZE_BYTES,
                    c2v_consumer_buf=c2v_peer,
                )

                a_mat: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Mat] = pl.load(
                    a, [0, 0], [ROWS, COLS], target_memory=pl.MemorySpace.Mat
                )
                b_mat: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Mat] = pl.load(
                    b, [0, 0], [ROWS, COLS], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Left] = pl.move(
                    a_mat, target_memory=pl.MemorySpace.Left
                )
                b_right: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Right] = pl.move(
                    b_mat, target_memory=pl.MemorySpace.Right
                )
                acc: pl.Tile[[ROWS, COLS], pl.FP32] = pl.matmul(a_left, b_right)
                narrowed: pl.Tile[
                    [ROWS, COLS],
                    pl.FP32,
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[valid_rows, valid_cols]),
                ] = pl.tile.set_validshape(acc, valid_rows, valid_cols)
                pl.tpush_to_aiv(narrowed, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def vector_consumer(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                valid_rows: pl.Scalar[pl.INDEX],
                valid_cols: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                c2v_buf = pl.reserve_buffer(
                    name="c2v_slot_buffer",
                    size=BUFFER_SIZE_BYTES,
                    base=0x2000,
                )
                pl.aiv_initialize_pipe(
                    dir_mask=1,
                    slot_size=SLOT_SIZE_BYTES,
                    c2v_consumer_buf=c2v_buf,
                )

                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                popped: pl.Tile[
                    [ROWS, COLS],
                    pl.FP32,
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[valid_rows, valid_cols]),
                ] = pl.tpop_from_aic(split=1)
                incremented: pl.Tile[[ROWS, COLS], pl.FP32] = pl.add(popped, 1.0)
                pl.tfree_to_aic(popped)
                return pl.store(incremented, [subblock_idx * VALID_ROWS, 0], output)

            @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
            def group_func(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                valid_rows: pl.Scalar[pl.INDEX],
                valid_cols: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                self.cube_producer(a, b, output, valid_rows, valid_cols)
                result = self.vector_consumer(a, b, output, valid_rows, valid_cols)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                valid_shape: pl.Tensor[[2], pl.INDEX],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                valid_rows: pl.Scalar[pl.INDEX] = pl.tensor.read(valid_shape, [0])
                valid_cols: pl.Scalar[pl.INDEX] = pl.tensor.read(valid_shape, [1])
                result = self.group_func(a, b, output, valid_rows, valid_cols)
                return result

        return C2VTpushValidShapeProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows = int(tensors["valid_shape"][0])
        valid_cols = int(tensors["valid_shape"][1])
        matmul = torch.matmul(tensors["a"].float(), tensors["b"].float())
        # Only output[:valid_rows, :valid_cols] is written; rows >= valid_rows
        # are never stored (localized store) -> mark NaN so validate_golden skips.
        tensors["output"][:] = float("nan")
        tensors["output"][:valid_rows, :valid_cols] = matmul[:valid_rows, :valid_cols] + 1.0


SV_VALID_COLS = 8  # vector-side set_validshape narrows columns; rows stay full (= ROWS)


class V2SetValidShapeOnVectorTestCase(PTOTestCase):
    """Vector consumer calls set_validshape under UP_DOWN split.

    The vector tile is halved by SplitVectorKernel, so a set_validshape whose
    row operand is the full ROWS must be localized to the halved box. Before the
    fix the full row operand exceeded the halved physical shape and PTOAS
    rejected it ('set_validshape op expects row operand <= shape dim'). Columns
    are narrowed (non-split axis) so the op is a real narrowing, not a no-op.
    """

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_setvalidshape_on_vector_updown_16x16"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [ROWS, COLS], DataType.BF16, init_value=1.0),
            TensorSpec("b", [ROWS, COLS], DataType.BF16, init_value=2.0),
            TensorSpec("output", [ROWS, COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class SetValidShapeOnVectorProgram:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def cube_producer(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ):
                c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_consumer")
                pl.aic_initialize_pipe(dir_mask=1, slot_size=SLOT_SIZE_BYTES, c2v_consumer_buf=c2v_peer)
                a_mat: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Mat] = pl.load(
                    a, [0, 0], [ROWS, COLS], target_memory=pl.MemorySpace.Mat
                )
                b_mat: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Mat] = pl.load(
                    b, [0, 0], [ROWS, COLS], target_memory=pl.MemorySpace.Mat
                )
                a_left: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Left] = pl.move(
                    a_mat, target_memory=pl.MemorySpace.Left
                )
                b_right: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Right] = pl.move(
                    b_mat, target_memory=pl.MemorySpace.Right
                )
                acc: pl.Tile[[ROWS, COLS], pl.FP32] = pl.matmul(a_left, b_right)
                # Set the producer valid_shape before tpush so the split
                # transport carries both row halves to the two AIV subblocks.
                acc_full: pl.Tile[
                    [ROWS, COLS],
                    pl.FP32,
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[ROWS, COLS]),
                ] = pl.tile.set_validshape(acc, ROWS, COLS)
                pl.tpush_to_aiv(acc_full, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def vector_consumer(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=BUFFER_SIZE_BYTES, base=0x2000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=SLOT_SIZE_BYTES, c2v_consumer_buf=c2v_buf)

                popped: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec, pl.TileView()] = pl.tpop_from_aic(split=1)
                # set_validshape requires a locally allocated source tile (PTOAS
                # rejects a raw tpop/FIFO-slot tile), so narrow a compute result.
                incremented: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec] = pl.add(popped, 1.0)
                # Row operand = full ROWS (the regression trigger); narrow columns.
                narrowed: pl.Tile[
                    [ROWS, COLS],
                    pl.FP32,
                    pl.Mem.Vec,
                    pl.TileView(valid_shape=[ROWS, SV_VALID_COLS]),
                ] = pl.tile.set_validshape(incremented, ROWS, SV_VALID_COLS)
                pl.tfree_to_aic(popped)
                # Offset [0, 0]: SplitVectorKernel adds the per-subblock row
                # offset, so lane 0 writes rows [0,8) and lane 1 writes [8,16).
                out_store: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(narrowed, [0, 0], output)
                return out_store

            @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
            def group_func(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                self.cube_producer(a, b, output)
                result = self.vector_consumer(a, b, output)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[ROWS, COLS], pl.BF16],
                b: pl.Tensor[[ROWS, COLS], pl.BF16],
                output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                result = self.group_func(a, b, output)
                return result

        return SetValidShapeOnVectorProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        matmul = torch.matmul(tensors["a"].float(), tensors["b"].float())
        # Columns >= SV_VALID_COLS are invalid (not written) -> mark NaN so
        # validate_golden skips them (runtime no longer zero-fills outputs).
        tensors["output"][:] = float("nan")
        tensors["output"][:, :SV_VALID_COLS] = matmul[:, :SV_VALID_COLS] + 1.0


class TestCrossCoreTpushValidShape:
    """a2a3-only tpush validShape runtime test."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
    def test_c2v_tpush_preserves_producer_valid_shape(self, test_runner, platform):
        result = test_runner.run(C2VTpushValidShapeTestCase(platform=platform))
        assert result.passed, f"C2V tpush validShape failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
    def test_set_validshape_on_vector_side_localized(self, test_runner, platform):
        result = test_runner.run(V2SetValidShapeOnVectorTestCase(platform=platform))
        assert result.passed, f"vector-side set_validshape under split failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
