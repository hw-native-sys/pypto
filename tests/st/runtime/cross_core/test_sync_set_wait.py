# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board test for explicit Cube/Vector event operations.

The Cube core writes ``a @ b`` to GM, publishes the product through the
established tpush pipe, and then executes a direct ``sync_set``. The Vector
core consumes the tpush event and the matching direct ``sync_wait`` before
reading GM, adding one, and writing the final output. Dynamic event IDs remain
covered by parser and PTO code-generation unit tests; this case validates both
direct event instructions on the end-to-end device path.
"""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

DIM = 16
DIRECT_SET_EVENT_ID = 4
PIPE_EVENT_ID = 0
FFTS_WORKSPACE_ELEMENTS = 256
PIPE_SLOT_SIZE_BYTES = DIM * DIM * 4
PIPE_BUFFER_SIZE_BYTES = PIPE_SLOT_SIZE_BYTES * 8
A2A3_BOARD_PLATFORMS = ("a2a3",)


@pl.program
class SyncSetWaitProgram:
    """AIC-to-AIV handshake around a GM data transfer."""

    @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
    def cube_producer(
        self,
        a: pl.Tensor[[DIM, DIM], pl.BF16],
        b: pl.Tensor[[DIM, DIM], pl.BF16],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
    ):
        c2v_peer = pl.import_peer_buffer(name="c2v_sync_buffer", peer_func="vector_consumer")
        pl.aic_initialize_pipe(
            dir_mask=1,
            slot_size=PIPE_SLOT_SIZE_BYTES,
            c2v_consumer_buf=c2v_peer,
        )
        pl.system.set_ffts(ffts_workspace)
        a_mat: pl.Tile[[DIM, DIM], pl.BF16, pl.Mem.Mat] = pl.load(
            a,
            [0, 0],
            [DIM, DIM],
            target_memory=pl.Mem.Mat,
        )
        b_mat: pl.Tile[[DIM, DIM], pl.BF16, pl.Mem.Mat] = pl.load(
            b,
            [0, 0],
            [DIM, DIM],
            target_memory=pl.Mem.Mat,
        )
        a_left = pl.move(a_mat, target_memory=pl.Mem.Left)
        b_right = pl.move(b_mat, target_memory=pl.Mem.Right)
        product: pl.Tile[[DIM, DIM], pl.FP32] = pl.matmul(a_left, b_right)
        pl.store(product, [0, 0], output)

        # Cube's accumulator-to-GM store completes on FIX before publishing
        # the event to the paired Vector core.
        # The initialized C2V pipe uses event 0. Its signal gives sync_wait a
        # known-good hardware producer; the following independent direct event
        # is also executed on board after that ordering signal is published.
        pl.tpush_to_aiv(product, split=1)
        pl.system.sync_set(DIRECT_SET_EVENT_ID, pipe=pl.PipeType.MTE2)

    @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
    def vector_consumer(
        self,
        a: pl.Tensor[[DIM, DIM], pl.BF16],
        b: pl.Tensor[[DIM, DIM], pl.BF16],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
    ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
        c2v_buffer = pl.reserve_buffer(
            name="c2v_sync_buffer",
            size=PIPE_BUFFER_SIZE_BYTES,
            base=0x2000,
        )
        pl.aiv_initialize_pipe(
            dir_mask=1,
            slot_size=PIPE_SLOT_SIZE_BYTES,
            c2v_consumer_buf=c2v_buffer,
        )
        pl.system.set_ffts(ffts_workspace)
        # Consume the event published by Cube's tpush before reading GM.
        pl.system.sync_wait(PIPE_EVENT_ID, pipe=pl.PipeType.MTE2)
        pl.system.sync_wait(DIRECT_SET_EVENT_ID, pipe=pl.PipeType.MTE2)
        product: pl.Tile[[DIM, DIM], pl.FP32, pl.Mem.Vec] = pl.load(
            output,
            [0, 0],
            [DIM, DIM],
        )
        result: pl.Tile[[DIM, DIM], pl.FP32] = pl.add(product, 1.0)
        output = pl.store(result, [0, 0], output)

        return output

    @pl.function(type=pl.FunctionType.Group, attrs={"split": pl.SplitMode.UP_DOWN})
    def group_func(
        self,
        a: pl.Tensor[[DIM, DIM], pl.BF16],
        b: pl.Tensor[[DIM, DIM], pl.BF16],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
    ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
        self.cube_producer(a, b, ffts_workspace, output)
        result = self.vector_consumer(a, b, ffts_workspace, output)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[DIM, DIM], pl.BF16],
        b: pl.Tensor[[DIM, DIM], pl.BF16],
        ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
        output: pl.InOut[pl.Tensor[[DIM, DIM], pl.FP32]],
    ) -> pl.Tensor[[DIM, DIM], pl.FP32]:
        return self.group_func(a, b, ffts_workspace, output)


class SyncSetWaitTestCase(PTOTestCase):
    """Cube computes ``a @ b`` and Vector publishes ``a @ b + 1``."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_sync_set_wait"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [DIM, DIM], DataType.BF16, init_value=torch.randn),
            TensorSpec("b", [DIM, DIM], DataType.BF16, init_value=torch.randn),
            TensorSpec(
                "ffts_workspace",
                [FFTS_WORKSPACE_ELEMENTS],
                DataType.INT64,
                init_value=torch.zeros,
            ),
            TensorSpec("output", [DIM, DIM], DataType.FP32, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        return SyncSetWaitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = torch.matmul(tensors["a"].float(), tensors["b"].float()) + 1.0


class TestSyncSetWait:
    """Explicit cross-core sync event system test."""

    @pytest.mark.parametrize("platform", A2A3_BOARD_PLATFORMS)
    def test_static_event_id_on_board(self, test_runner, platform):
        """Run the Cube-to-Vector event handshake and verify the GM result on A2/A3."""
        result = test_runner.run(SyncSetWaitTestCase(platform=platform))
        assert result.passed, f"Cross-core sync_set/sync_wait failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
