# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank reduce-scatter — PyPTO port of
``examples/workers/l3/reduce_scatter_distributed``.

Mirrors the 4-phase pattern of the runtime example's
``kernels/aiv/reduce_scatter_kernel.cpp`` (simpler ``reduce_scatter_distributed``, PR #842):

* **Phase 1 (stage-in)** — copy both local chunks (``2*SIZE`` floats) into the
  window-bound ``scratch`` buffer so every peer can read the full staged input.
* **Phase 2 (barrier)** — each rank ``AtomicAdd``s the peer's ``signal`` cell
  via ``pld.system.notify`` and ``pld.system.wait``s on its own cell until
  the peer has finished staging.
* **Phase 3 (reduce)** — load this rank's chunk ``scratch[chunk_col:…]`` (``chunk_col`` is
  ``r*SIZE`` from HOST),
  ``pld.tile.remote_load`` the peer's slice at the **same chunk offset**, and
  ``pl.add`` them. For ``nranks=2`` a single peer add is sufficient.
* **Phase 4 (stage-out)** — ``pl.store`` the accumulator into local ``out``.

Golden: rank ``r`` output is the element-wise sum of chunk ``r`` across all ranks:
``outputs[r][j] = inputs[0][r*SIZE+j] + inputs[1][r*SIZE+j]``.

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

SIZE = 64  # matches COUNT_PER_RANK in simpler reduce_scatter_kernel.cpp
NRANKS = 2


def _expected_reduce_scatter(inputs: torch.Tensor) -> torch.Tensor:
    """Per-rank golden: sum of chunk ``r`` across all ranks."""
    chunks = [
        inputs[0, 0, r * SIZE : (r + 1) * SIZE] + inputs[1, 0, r * SIZE : (r + 1) * SIZE]
        for r in range(NRANKS)
    ]
    return torch.stack(chunks).reshape(NRANKS, 1, SIZE)


def _build_reduce_scatter_program():
    """Build the 2-rank reduce-scatter program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """

    @pl.program
    class ReduceScatterTwoRank:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_scatter_step(
            self,
            inp: pl.Tensor[[1, 2 * SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, 2 * SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            chunk_col: pl.Scalar[pl.INDEX],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: stage-in — copy both local chunks into scratch.
            chunk0 = pl.load(inp, [0, 0], [1, SIZE])
            pl.store(chunk0, [0, 0], scratch)
            chunk1 = pl.load(inp, [0, SIZE], [1, SIZE])
            pl.store(chunk1, [0, SIZE], scratch)

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

            # Phase 3: reduce — sum this rank's chunk (chunk_col is 0 or SIZE) across peers.
            acc = pl.load(scratch, [0, chunk_col], [1, SIZE])
            recv = pld.tile.remote_load(scratch, peer=peer, offsets=[0, chunk_col], shape=[1, SIZE])
            acc = pl.add(acc, recv)

            # Phase 4: stage-out — reduced chunk → local output.
            return pl.store(acc, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, 2 * SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, 2 * SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            chunk_col: pl.Scalar[pl.INDEX],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.reduce_scatter_step(inp, out, scratch, signal, chunk_col, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, 2 * SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            # 2xSIZE FP32 staging (2 ranks x SIZE elements x 4 bytes).
            scratch_buf = pld.alloc_window_buffer(2 * SIZE * 4)
            signal_buf = pld.alloc_window_buffer(4)  # 1x1 x INT32

            for r in pl.range(pld.world_size()):
                scratch = pld.window(scratch_buf, [1, 2 * SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                # Ring partner: rank r notifies peer = (r + 1) % nranks; chunk_col = r * SIZE.
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    scratch,
                    signal,
                    r * SIZE,
                    (r + 1) % pld.world_size(),
                    device=r,
                )
            return outputs

    return ReduceScatterTwoRank


class TestL3ReduceScatter:
    """L3 distributed runtime: 2-rank reduce-scatter via stage-in + notify/wait + remote_load."""

    def test_reduce_scatter(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"reduce-scatter needs 2 devices, got {device_ids}")

        program = _build_reduce_scatter_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # Each rank stages nranks*SIZE floats: rank r uses i + r*100 for i in 0..2*SIZE-1.
        inputs = torch.stack(
            [
                torch.arange(2 * SIZE, dtype=torch.float32).reshape(1, 2 * SIZE),
                torch.arange(100.0, 100.0 + 2 * SIZE, dtype=torch.float32).reshape(1, 2 * SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_reduce_scatter(inputs)
        assert torch.allclose(outputs, expected), (
            f"reduce-scatter mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
