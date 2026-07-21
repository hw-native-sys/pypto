# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board V2C test using explicit events instead of ``tpush``/``tpop``.

This reimplements ``MultiPipeNoSplitProgram`` from ``test_cross_core.py`` with
two GM transfer tensors. Vector writes ``a + b`` and ``a - b`` to GM and
signals their completion with ``sync_set`` on ``PIPE_MTE3``. Cube waits with
``sync_wait`` on ``PIPE_MTE2``, reads both tiles, and computes their matrix
product. No queue primitive participates in the transfer or rendezvous.
"""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

DIM = 16
SUM_EVENT_ID = 4
DIFF_EVENT_ID = 5
FFTS_WORKSPACE_ELEMENTS = 256
A2A3_BOARD_PLATFORMS = ("a2a3",)


@pl.program
class SyncSetWaitProgram:
    """Explicit-event version of the existing two-pipe V2C test."""

    @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
    def vector_producer(
        self,
        a: pl.Tensor[[DIM, DIM], pl.FP32],
        b: pl.Tensor[[DIM, DIM], pl.FP32],
        sum_transfer: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
        diff_transfer: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]],
    ):
        pl.system.set_ffts(ffts_workspace)
        a_tile: pl.Tile[[DIM, DIM], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [DIM, DIM])
        b_tile: pl.Tile[[DIM, DIM], pl.FP32, pl.Mem.Vec] = pl.load(b, [0, 0], [DIM, DIM])
        sum_tile: pl.Tile[[DIM, DIM], pl.FP32, pl.Mem.Vec] = pl.add(a_tile, b_tile)
        diff_tile: pl.Tile[[DIM, DIM], pl.FP32, pl.Mem.Vec] = pl.sub(a_tile, b_tile)

        pl.store(sum_tile, [0, 0], sum_transfer)
        pl.system.sync_set(SUM_EVENT_ID, pipe=pl.PipeType.MTE3)
        pl.store(diff_tile, [0, 0], diff_transfer)
        pl.system.sync_set(DIFF_EVENT_ID, pipe=pl.PipeType.MTE3)

    @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
    def cube_consumer(
        self,
        a: pl.Tensor[[DIM, DIM], pl.FP32],
        b: pl.Tensor[[DIM, DIM], pl.FP32],
        sum_transfer: pl.Tensor[[DIM, DIM], pl.FP32],
        diff_transfer: pl.Tensor[[DIM, DIM], pl.FP32],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]],
    ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
        pl.system.set_ffts(ffts_workspace)
        pl.system.sync_wait(SUM_EVENT_ID, pipe=pl.PipeType.MTE2)
        sum_tile: pl.Tile[[DIM, DIM], pl.FP32, pl.Mem.Mat] = pl.load(
            sum_transfer,
            [0, 0],
            [DIM, DIM],
            target_memory=pl.Mem.Mat,
        )
        pl.system.sync_wait(DIFF_EVENT_ID, pipe=pl.PipeType.MTE2)
        diff_tile: pl.Tile[[DIM, DIM], pl.FP32, pl.Mem.Mat] = pl.load(
            diff_transfer,
            [0, 0],
            [DIM, DIM],
            target_memory=pl.Mem.Mat,
        )
        sum_left = pl.move(sum_tile, target_memory=pl.Mem.Left)
        diff_right = pl.move(diff_tile, target_memory=pl.Mem.Right)
        result: pl.Tile[[DIM, DIM], pl.FP32] = pl.matmul(sum_left, diff_right)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
    def group_func(
        self,
        a: pl.Tensor[[DIM, DIM], pl.FP32],
        b: pl.Tensor[[DIM, DIM], pl.FP32],
        sum_transfer: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
        diff_transfer: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]],
    ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
        result = self.cube_consumer(a, b, sum_transfer, diff_transfer, ffts_workspace, output)
        self.vector_producer(a, b, sum_transfer, diff_transfer, ffts_workspace, output)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[DIM, DIM], pl.FP32],
        b: pl.Tensor[[DIM, DIM], pl.FP32],
        sum_transfer: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
        diff_transfer: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.Out[pl.Tensor[[DIM, DIM], pl.FP32]],
    ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
        return self.group_func(a, b, sum_transfer, diff_transfer, ffts_workspace, output)


class SyncSetWaitTestCase(PTOTestCase):
    """Explicit-event V2C: ``output = (a + b) @ (a - b)``."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_sync_set_wait"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [DIM, DIM], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [DIM, DIM], DataType.FP32, init_value=torch.randn),
            TensorSpec("sum_transfer", [DIM, DIM], DataType.FP32, init_value=torch.zeros),
            TensorSpec("diff_transfer", [DIM, DIM], DataType.FP32, init_value=torch.zeros),
            TensorSpec(
                "ffts_workspace",
                [FFTS_WORKSPACE_ELEMENTS],
                DataType.INT64,
                init_value=torch.zeros,
            ),
            TensorSpec("output", [DIM, DIM], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SyncSetWaitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = torch.matmul(tensors["a"] + tensors["b"], tensors["a"] - tensors["b"])


class TestSyncSetWait:
    """Explicit cross-core sync event system test."""

    @pytest.mark.parametrize("platform", A2A3_BOARD_PLATFORMS)
    def test_static_event_id_on_board(self, test_runner, platform):
        """Replace two V2C tpush/tpop queues with GM buffers and explicit events."""
        result = test_runner.run(SyncSetWaitTestCase(platform=platform))
        assert result.passed, f"Cross-core sync_set/sync_wait failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
