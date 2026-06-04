# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank allreduce — PyPTO port of ``runtime/examples/workers/l3/allreduce_distributed``.

Mirrors the 4-phase pattern of the runtime example's
``kernels/aiv/allreduce_kernel.cpp``:

* **Phase 1 (stage-in)** — copy local ``inp`` into this rank's slice of the
  window-bound ``data`` buffer (a plain local ``pl.store`` into the
  ``DistributedTensor``).
* **Phase 2 (barrier)** — each rank ``AtomicAdd``s the peer's ``signal`` cell
  via ``pld.system.notify`` and ``pld.system.wait``s on its own cell until
  the peer has staged its slice.
* **Phase 3 (compute)** — ``pl.load`` this rank's own slice into an
  accumulator tile, ``pld.tile.remote_load`` the peer's slice, and
  ``pl.add`` them. For ``nranks=2`` a single peer read is sufficient; the
  algorithm generalises to N peers by looping the remote-load+add.
* **Phase 4 (stage-out)** — ``pl.store`` the accumulator into local
  ``out``.

Golden: ``outputs[r] == inputs[0] + inputs[1]`` for every rank ``r``.

Driven by 2 devices via ``DistributedConfig(device_ids=device_ids[:2], ...)``,
matching the runtime example's hardcoded 2-rank requirement.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 256  # matches ALLREDUCE_COUNT in runtime allreduce_kernel.cpp


def _build_allreduce_program():
    """Build the 2-rank allreduce program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """

    @pl.program
    class AllReduceTwoRank:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: stage-in — local input → this rank's window slice.
            local = pl.load(inp, [0, 0], [1, SIZE])
            data = pl.store(local, [0, 0], data)

            # Phase 2: barrier — AtomicAdd the peer's signal cell, then
            # wait for ours to be bumped by the rank that targets us.
            # AtomicAdd on a single cell is sufficient for a 2-rank ring;
            # nranks > 2 would size the signal to nranks and let each rank
            # set a distinct slot (matching kernels/aiv/allreduce_kernel.cpp).
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

            # Phase 3: load my own slice into the accumulator, then add
            # the peer's slice via cross-rank remote_load. For nranks > 2
            # this becomes a loop over (nranks - 1) peers.
            acc = pl.load(data, [0, 0], [1, SIZE])
            recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
            acc = pl.add(acc, recv)

            # Phase 4: stage-out — reduced accumulator → local output.
            return pl.store(acc, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.reduce_step(inp, out, data, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)  # 1xSIZE x FP32 (4 bytes)
            signal_buf = pld.alloc_window_buffer(4)  # 1x1 x INT32

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # Ring partner: rank r notifies / reads peer = (r + 1) % nranks.
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    data,
                    signal,
                    (r + 1) % pld.world_size(),
                    device=r,
                )
            return outputs

    return AllReduceTwoRank


class TestL3AllReduce:
    """L3 distributed runtime: 2-rank allreduce via stage-in + notify/wait + remote_load."""

    def test_allreduce(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"allreduce needs 2 devices, got {device_ids}")

        program = _build_allreduce_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # Per-rank input: rank 0 holds [0, 1, …, SIZE-1]; rank 1 holds
        # [100, 101, …, 100+SIZE-1]. After allreduce, every rank's output
        # is inputs[0] + inputs[1].
        inputs = torch.stack(
            [
                torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
                torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        reduced = inputs[0] + inputs[1]
        expected = torch.stack([reduced, reduced])
        assert torch.allclose(outputs, expected), (
            f"allreduce mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
