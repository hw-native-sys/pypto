# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank broadcast — PyPTO port of ``examples/workers/l3/broadcast_distributed``.

Mirrors the 3-phase pattern of the runtime example's
``kernels/aiv/broadcast_kernel.cpp`` (simpler ``broadcast_distributed``):

* **Phase 1 (stage-in)** — root rank only: copy local ``inp`` into this rank's
  scratch slot in the window-bound ``scratch`` buffer.
* **Phase 2 (barrier)** — each rank ``AtomicAdd``s the peer's ``signal`` cell
  via ``pld.system.notify`` and ``pld.system.wait``s on its own cell until
  the peer has finished phase 1.
* **Phase 3 (broadcast)** — every rank ``pld.tile.remote_load``s the root rank's
  scratch slice and ``pl.store``s into local ``out``.

Golden: every rank's ``outputs[r]`` equals ``inputs[ROOT_RANK]`` (root tensor
broadcast to all ranks). Non-root inputs must not appear in outputs.

Driven by 2 devices via ``DistributedConfig(device_ids=device_ids[:2], ...)``,
matching the example's hardcoded 2-rank requirement.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64  # matches COUNT_PER_RANK in simpler broadcast_kernel.cpp
ROOT_RANK = 0


def _expected_broadcast(inputs: torch.Tensor, root: int = ROOT_RANK) -> torch.Tensor:
    """Root row replicated on every rank."""
    root_row = inputs[root, 0]
    return torch.stack([root_row, root_row]).unsqueeze(1)


def _build_broadcast_program():
    """Build the 2-rank broadcast program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """

    @pl.program
    class BroadcastTwoRank:
        @pl.function(type=pl.FunctionType.InCore)
        def broadcast_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
            root: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: stage-in — root only writes input into the HCCL window.
            if my_rank == root:
                local = pl.load(inp, [0, 0], [1, SIZE])
                pl.store(local, [0, 0], scratch)

            # Phase 2: barrier — AtomicAdd the peer's signal cell, then
            # wait for ours to be bumped by the rank that targets us.
            pld.system.notify(
                signal,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

            # Phase 3: broadcast — read root scratch and write local output.
            recv = pld.tile.remote_load(scratch, peer=root, offsets=[0, 0], shape=[1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
            root: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.broadcast_step(inp, out, scratch, signal, peer, my_rank, root)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            scratch_buf = pld.alloc_window_buffer(SIZE * 4)  # 1xSIZE x FP32 (4 bytes)
            signal_buf = pld.alloc_window_buffer(4)  # 1x1 x INT32

            for r in pl.range(pld.world_size()):
                scratch = pld.window(scratch_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # Ring partner for barrier: rank r notifies peer = (r + 1) % nranks.
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    scratch,
                    signal,
                    (r + 1) % pld.world_size(),
                    r,
                    ROOT_RANK,
                    device=r,
                )
            return outputs

    return BroadcastTwoRank


class TestL3Broadcast:
    """L3 distributed runtime: 2-rank broadcast via root stage-in + notify/wait + remote_load."""

    def test_broadcast(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"broadcast needs 2 devices, got {device_ids}")

        program = _build_broadcast_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # Rank 0 is root: [0, 1, …, SIZE-1]. Rank 1 holds distinct values that
        # must not appear in outputs after broadcast.
        inputs = torch.stack(
            [
                torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
                torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_broadcast(inputs)
        assert torch.allclose(outputs, expected), (
            f"broadcast mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )
        assert not torch.allclose(outputs[0], inputs[1]), "non-root input leaked into output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
