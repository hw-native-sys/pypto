# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank broadcast — PyPTO port of ``examples/workers/l3/broadcast_distributed``.

Mirrors the 3-phase pattern of the runtime example's
``kernels/aiv/broadcast_kernel.cpp`` (simpler ``broadcast_distributed``):

* **Phase 1 (stage-in)** — root rank only: copy local ``inp`` into this rank's
  scratch slot in the window-bound ``scratch`` buffer.
* **Phase 2 (barrier)** — each rank ``AtomicAdd``s every peer's ``signal``
  cell via ``pld.system.notify`` and ``pld.system.wait``s on each peer slot
  until all ranks have finished phase 1 (``signal`` shape ``[NR, 1]``).
* **Phase 3 (broadcast)** — every rank ``pld.tile.remote_load``s the root rank's
  scratch slice and ``pl.store``s into local ``out``.

Golden: every rank's ``outputs[r]`` equals ``inputs[0]`` (rank-0 tensor
broadcast to all ranks). Non-root inputs must not appear in outputs.

Rank count uses ``NR = pl.dynamic("NR")`` in host tensor shapes; runtime
``inputs.shape[0]`` must match ``len(device_ids)`` / ``pld.world_size()``.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four
devices, e.g. ``--device=0,1,2,3`` or ``--device=0-3``). One program body
for both.
"""

# pyright: reportUndefinedVariable=false

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64  # matches COUNT_PER_RANK in simpler broadcast_kernel.cpp
NR = pl.dynamic("NR")


def _expected_broadcast(inputs: torch.Tensor) -> torch.Tensor:
    """Root row replicated on every rank."""
    root_row = inputs[0, 0]
    return torch.stack([root_row] * inputs.shape[0]).unsqueeze(1)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors; rank 0 is root, others must not leak into output."""
    rows = [
        torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),  # rank 0 = root
    ]
    for r in range(1, n_ranks):
        rows.append(torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE))
    return torch.stack(rows)


@pl.program
class BroadcastMesh:
    """Mesh broadcast with dynamic rank count ``NR``."""

    @pl.function(type=pl.FunctionType.InCore)
    def broadcast_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        root: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Three-phase mesh broadcast on window-bound ``scratch`` / ``signal``."""
        ctx = pld.get_comm_ctx(scratch)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # Phase 1: stage-in — root only writes input into the HCCL window.
        if my_rank == root:
            local = pl.load(inp, [0, 0], [1, SIZE])
            pl.store(local, [0, 0], scratch)

        # Phase 2: dual-loop barrier — notify every peer, wait on every peer.
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.system.notify(
                    signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(nranks):
            if src != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[src, 0],
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
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        root: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Per-device orchestration wrapper around ``broadcast_step``."""
        return self.broadcast_step(inp, out, scratch, signal, root)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        """Launch one chip orchestration per rank with shared window buffers."""
        scratch_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            scratch = pld.window(scratch_buf, [1, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            self.chip_orch(
                inputs[r],
                outputs[r],
                scratch,
                signal,
                0,  # root = rank 0
                device=r,
            )
        return outputs


class TestL3Broadcast:
    """L3 distributed runtime: N-rank broadcast via root stage-in + notify/wait + remote_load."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_broadcast(self, test_config, device_ids, n_ranks):
        """Compile and run mesh broadcast for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"broadcast P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            BroadcastMesh,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_broadcast(inputs)
        assert torch.allclose(outputs, expected), (
            f"broadcast P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )
        # Non-root inputs must not leak into outputs.
        for r in range(1, n_ranks):
            assert not torch.allclose(outputs[r], inputs[r]), f"non-root rank {r} input leaked into output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
