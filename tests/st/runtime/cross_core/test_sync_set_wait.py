# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board odd-shape V2C split using explicit cross-core events.

Two AIV lanes partition a logical ``[5, K]`` tensor unevenly: lane 0 computes
rows ``[0:2]`` and lane 1 computes rows ``[2:5]``. Both write into one GM
transfer tensor and signal the same mode-2 event with ``sync_set``. One AIC
``sync_wait`` therefore completes only after both the 2-row and 3-row stores
have landed, after which Cube consumes the complete tensor. No ``tpush`` or
``tpop`` participates in the transfer or rendezvous.

The GM transfer is physically padded to 16 rows because AIC Mat/Acc tiles must
be box-aligned, while ``valid_shape`` keeps the logical payload at five rows.
The AIV and AIC signatures intentionally stay identical because they share one
mixed-kernel launch argument layout.
"""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

ROWS = 5
LANE0_ROWS = 2
LANE1_ROWS = ROWS - LANE0_ROWS
K = 16
N = 16
CUBE_PHYSICAL_ROWS = 16
V2C_EVENT_ID = 4
FFTS_WORKSPACE_ELEMENTS = 256
A2A3_BOARD_PLATFORMS = ("a2a3",)


@pl.program
class SyncSetWaitProgram:
    """Uneven 2/3-row AIV split followed by one AIC consumer."""

    @pl.function(
        type=pl.FunctionType.AIV,
        attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
    )
    def vector_producer(
        self,
        a: pl.Tensor[[ROWS, K], pl.FP32],
        b: pl.Tensor[[ROWS, K], pl.FP32],
        weight: pl.Tensor[[K, N], pl.FP32],
        transfer: pl.InOut[pl.Tensor[[CUBE_PHYSICAL_ROWS, K], pl.FP32]],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.Out[pl.Tensor[[CUBE_PHYSICAL_ROWS, N], pl.FP32]],
    ):
        pl.system.set_ffts(ffts_workspace)
        lane: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        if lane == 0:
            a_lane0: pl.Tile[[LANE0_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [LANE0_ROWS, K])
            b_lane0: pl.Tile[[LANE0_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(b, [0, 0], [LANE0_ROWS, K])
            sum_lane0: pl.Tile[[LANE0_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.add(a_lane0, b_lane0)
            pl.store(sum_lane0, [0, 0], transfer)
        else:
            a_lane1: pl.Tile[[LANE1_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(
                a, [LANE0_ROWS, 0], [LANE1_ROWS, K]
            )
            b_lane1: pl.Tile[[LANE1_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(
                b, [LANE0_ROWS, 0], [LANE1_ROWS, K]
            )
            sum_lane1: pl.Tile[[LANE1_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.add(a_lane1, b_lane1)
            pl.store(sum_lane1, [LANE0_ROWS, 0], transfer)

        # Mode 2 is a V-to-C reduction: AIC unblocks only after both AIV lanes
        # have signalled this event, so the complete five-row GM tensor is ready.
        pl.system.sync_set(V2C_EVENT_ID, pipe=pl.PipeType.MTE3)

    @pl.function(
        type=pl.FunctionType.AIC,
        attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
    )
    def cube_consumer(
        self,
        a: pl.Tensor[[ROWS, K], pl.FP32],
        b: pl.Tensor[[ROWS, K], pl.FP32],
        weight: pl.Tensor[[K, N], pl.FP32],
        transfer: pl.Tensor[[CUBE_PHYSICAL_ROWS, K], pl.FP32],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.Out[pl.Tensor[[CUBE_PHYSICAL_ROWS, N], pl.FP32]],
    ) -> pl.Tensor[[CUBE_PHYSICAL_ROWS, N], pl.FP32]:
        pl.system.set_ffts(ffts_workspace)
        pl.system.sync_wait(V2C_EVENT_ID, pipe=pl.PipeType.MTE2)
        transfer_mat: pl.Tile[
            [CUBE_PHYSICAL_ROWS, K],
            pl.FP32,
            pl.Mem.Mat,
            pl.TileView(valid_shape=[ROWS, K]),
        ] = pl.load(
            transfer,
            [0, 0],
            [CUBE_PHYSICAL_ROWS, K],
            valid_shapes=[ROWS, K],
            target_memory=pl.Mem.Mat,
        )
        weight_mat: pl.Tile[[K, N], pl.FP32, pl.Mem.Mat] = pl.load(
            weight,
            [0, 0],
            [K, N],
            target_memory=pl.Mem.Mat,
        )
        transfer_left = pl.move(transfer_mat, target_memory=pl.Mem.Left)
        weight_right = pl.move(weight_mat, target_memory=pl.Mem.Right)
        result: pl.Tile[
            [CUBE_PHYSICAL_ROWS, N],
            pl.FP32,
            pl.Mem.Acc,
            pl.TileView(valid_shape=[ROWS, N]),
        ] = pl.matmul(transfer_left, weight_right)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
    def group_func(
        self,
        a: pl.Tensor[[ROWS, K], pl.FP32],
        b: pl.Tensor[[ROWS, K], pl.FP32],
        weight: pl.Tensor[[K, N], pl.FP32],
        transfer: pl.InOut[pl.Tensor[[CUBE_PHYSICAL_ROWS, K], pl.FP32]],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.Out[pl.Tensor[[CUBE_PHYSICAL_ROWS, N], pl.FP32]],
    ) -> pl.Tensor[[CUBE_PHYSICAL_ROWS, N], pl.FP32]:
        result = self.cube_consumer(a, b, weight, transfer, ffts_workspace, output)
        self.vector_producer(a, b, weight, transfer, ffts_workspace, output)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[ROWS, K], pl.FP32],
        b: pl.Tensor[[ROWS, K], pl.FP32],
        weight: pl.Tensor[[K, N], pl.FP32],
        transfer: pl.InOut[pl.Tensor[[CUBE_PHYSICAL_ROWS, K], pl.FP32]],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.Out[pl.Tensor[[CUBE_PHYSICAL_ROWS, N], pl.FP32]],
    ) -> pl.Tensor[[CUBE_PHYSICAL_ROWS, N], pl.FP32]:
        return self.group_func(a, b, weight, transfer, ffts_workspace, output)


class SyncSetWaitTestCase(PTOTestCase):
    """Explicit-event V2C: uneven AIV split, then ``output = (a + b) @ weight``."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_sync_set_wait_odd_shape"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [ROWS, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [ROWS, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("weight", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("transfer", [CUBE_PHYSICAL_ROWS, K], DataType.FP32, init_value=torch.zeros),
            TensorSpec(
                "ffts_workspace",
                [FFTS_WORKSPACE_ELEMENTS],
                DataType.INT64,
                init_value=torch.zeros,
            ),
            TensorSpec("output", [CUBE_PHYSICAL_ROWS, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SyncSetWaitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:ROWS] = torch.matmul(tensors["a"] + tensors["b"], tensors["weight"])


class TestSyncSetWait:
    """Explicit cross-core sync event system test."""

    @pytest.mark.parametrize("platform", A2A3_BOARD_PLATFORMS)
    def test_static_event_id_on_board(self, test_runner, platform):
        """Synchronize one 2/3-row two-AIV GM write with one AIC wait."""
        result = test_runner.run(SyncSetWaitTestCase(platform=platform))
        assert result.passed, f"Cross-core sync_set/sync_wait failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
