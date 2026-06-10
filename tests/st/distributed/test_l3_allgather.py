# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank allgather — PyPTO port of ``examples/workers/l3/allgather_distributed``.

Mirrors the 3-phase pattern of the runtime example's
``kernels/aiv/allgather_kernel.cpp`` (simpler ``allgather_distributed``, PR #842):

* **Phase 1 (stage-in)** — copy local ``inp`` into this rank's scratch slot in the
  window-bound ``scratch`` buffer (a plain local ``pl.store`` into the
  ``DistributedTensor``).
* **Phase 2 (barrier)** — each rank ``AtomicAdd``s every peer's ``signal``
  cell via ``pld.system.notify`` and ``pld.system.wait``s on each peer slot
  until all ranks have staged their slice (``signal`` shape ``[NR, 1]``).
* **Phase 3 (gather)** — for each rank ``r``, ``pld.tile.remote_load`` that
  rank's scratch slice and ``pl.store`` into ``out[r*SIZE:(r+1)*SIZE]``.

Golden: every rank's ``outputs[r]`` equals the rank-ordered concatenation of all
``inputs[*]`` (i.e. ``out[r][k] = inputs[k//SIZE][0, k%SIZE]``).

Rank count uses ``NR = pl.dynamic("NR")`` in host tensor shapes; runtime
``inputs.shape[0]`` must match ``len(device_ids)`` / ``pld.world_size()``.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four
devices, e.g. ``--device=0,1,2,3`` or ``--device=0-3``). One program body
for both.
"""

# pyright: reportUndefinedVariable=false  # NR, SIZE are used in DSL type annotations below

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64  # matches COUNT_PER_RANK in simpler allgather_kernel.cpp
NR = pl.dynamic("NR")


def _expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    """Rank-ordered concatenation, zero-padded to 4*SIZE; identical on every rank."""
    n_ranks = inputs.shape[0]
    gathered = torch.cat([inputs[r, 0] for r in range(n_ranks)])
    padded = torch.zeros(4 * SIZE, dtype=torch.float32)
    padded[: n_ranks * SIZE] = gathered
    return torch.stack([padded] * n_ranks).unsqueeze(1)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors so the golden concatenation is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


@pl.program
class AllGatherMesh:
    """Mesh allgather with dynamic rank count ``NR``."""

    @pl.function(type=pl.FunctionType.InCore)
    def gather_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, 4 * SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 4 * SIZE], pl.FP32]:
        """Three-phase mesh allgather on window-bound ``scratch`` / ``signal``."""
        ctx = pld.get_comm_ctx(scratch)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # Phase 1: stage-in — local input → this rank's scratch slot.
        local = pl.load(inp, [0, 0], [1, SIZE])
        pl.store(local, [0, 0], scratch)

        # Phase 2: barrier — notify every peer, wait on every peer slot.
        # ``alloc_window_buffer`` zero-initialises cells; AtomicAdd/Ge(1) is safe.
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

        # Phase 3: gather — read each rank's scratch, write rank-ordered slices.
        for r in pl.range(nranks):
            recv = pld.tile.remote_load(scratch, peer=r, offsets=[0, 0], shape=[1, SIZE])
            pl.store(recv, [0, r * SIZE], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, 4 * SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 4 * SIZE], pl.FP32]:
        """Per-device orchestration wrapper around ``gather_step``."""
        return self.gather_step(inp, out, scratch, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, NR * SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, NR * SIZE], pl.FP32]:
        """Launch one chip orchestration per rank with shared window buffers."""
        scratch_buf = pld.alloc_window_buffer(SIZE * 4)  # 1xSIZE x FP32 (4 bytes)
        signal_buf = pld.alloc_window_buffer(pld.world_size() * 4)  # NR x 1 x INT32

        for r in pl.range(pld.world_size()):
            scratch = pld.window(scratch_buf, [1, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            self.chip_orch(
                inputs[r],
                outputs[r],
                scratch,
                signal,
                device=r,
            )
        return outputs


class TestL3AllGather:
    """L3 distributed runtime: N-rank allgather via stage-in + notify/wait + remote_load."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allgather(self, test_config, device_ids, n_ranks):
        """Compile and run mesh allgather for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            AllGatherMesh,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, 4 * SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allgather(inputs)
        assert torch.allclose(outputs, expected), (
            f"allgather P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

        # Sanity: output must be concatenation, not any single input passed through.
        for r in range(n_ranks):
            assert not torch.allclose(outputs[r, 0, :SIZE], inputs[r, 0]), (
                f"allgather P={n_ranks}: rank {r} output[:SIZE] must not equal"
                f" its own input (concatenation required)"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
