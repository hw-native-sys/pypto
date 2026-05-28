# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank allgather — PyPTO port of ``examples/workers/l3/allgather_distributed``.

Mirrors the 3-phase pattern of the runtime example's
``kernels/aiv/allgather_kernel.cpp`` (simpler ``allgather_distributed``, PR #842):

* **Phase 1 (stage-in)** — copy local ``inp`` into this rank's scratch slot in the
  window-bound ``scratch`` buffer (a plain local ``pl.store`` into the
  ``DistributedTensor``).
* **Phase 2 (barrier)** — each rank ``AtomicAdd``s the peer's ``signal`` cell
  via ``pld.system.notify`` and ``pld.system.wait``s on its own cell until
  the peer has staged its slice.
* **Phase 3 (gather)** — for each rank index ``r``, ``pld.tile.remote_load`` that
  rank's scratch slice and ``pl.store`` into ``out[r*SIZE:(r+1)*SIZE]``. For
  ``nranks=2`` both peers are read explicitly.

Golden: every rank's ``outputs[r]`` equals the rank-ordered concatenation
``[inputs[0], inputs[1]]`` (i.e. ``out[k] = inputs[k//SIZE][0, k%SIZE]``).

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

SIZE = 64  # matches COUNT_PER_RANK in simpler allgather_kernel.cpp


def _expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    """Rank-ordered concatenation; identical vector on every rank."""
    gathered = torch.cat([inputs[r, 0] for r in range(inputs.shape[0])])
    return torch.stack([gathered, gathered])


def _build_allgather_program():
    """Build the 2-rank allgather program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """

    @pl.program
    class AllGatherTwoRank:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 2 * SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
            read_peer0: pl.Scalar[pl.INT32],
            read_peer1: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 2 * SIZE], pl.FP32]:
            # Phase 1: stage-in — local input → this rank's scratch slot.
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

            # Phase 3: gather — read each rank's scratch and write rank-ordered slices.
            recv0 = pld.tile.remote_load(scratch, peer=read_peer0, offsets=[0, 0], shape=[1, SIZE])
            pl.store(recv0, [0, 0], out)
            recv1 = pld.tile.remote_load(scratch, peer=read_peer1, offsets=[0, 0], shape=[1, SIZE])
            return pl.store(recv1, [0, SIZE], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 2 * SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
            read_peer0: pl.Scalar[pl.INT32],
            read_peer1: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 2 * SIZE], pl.FP32]:
            return self.gather_step(inp, out, scratch, signal, peer, read_peer0, read_peer1)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, 2 * SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, 2 * SIZE], pl.FP32]:
            scratch_buf = pld.alloc_window_buffer(SIZE * 4)  # 1xSIZE x FP32 (4 bytes)
            signal_buf = pld.alloc_window_buffer(4)  # 1x1 x INT32

            for r in pl.range(pld.world_size()):
                scratch = pld.window(scratch_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # Ring partner: rank r notifies / reads peer = (r + 1) % nranks.
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    scratch,
                    signal,
                    (r + 1) % pld.world_size(),
                    0,
                    1,
                    device=r,
                )
            return outputs

    return AllGatherTwoRank


class TestL3AllGather:
    """L3 distributed runtime: 2-rank allgather via stage-in + notify/wait + remote_load."""

    def test_allgather(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"allgather needs 2 devices, got {device_ids}")

        program = _build_allgather_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # Per-rank input: rank 0 holds [0, 1, …, SIZE-1]; rank 1 holds
        # [100, 101, …, 100+SIZE-1]. After allgather, every rank's output
        # is the rank-ordered concatenation of both inputs.
        inputs = torch.stack(
            [
                torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
                torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, 2 * SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allgather(inputs)
        assert torch.allclose(outputs, expected), (
            f"allgather mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
