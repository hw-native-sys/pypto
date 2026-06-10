# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank reduce-scatter — PyPTO port of
``examples/workers/l3/reduce_scatter_distributed``.

Mirrors the 4-phase pattern of the runtime example's
``kernels/aiv/reduce_scatter_kernel.cpp`` (simpler ``reduce_scatter_distributed``, PR #842):

* **Phase 1 (stage-in)** — for each chunk ``c`` in ``0..nranks-1``, copy
  ``inp[c*SIZE:(c+1)*SIZE]`` into the window-bound ``scratch`` at the
  corresponding chunk offset.
* **Phase 2 (barrier)** — each rank ``AtomicAdd``s every peer's ``signal``
  cell via ``pld.system.notify`` and ``pld.system.wait``s on each peer slot
  until all ranks have staged their slices (``signal`` shape ``[NR, 1]``).
* **Phase 3 (reduce)** — load this rank's chunk at ``scratch[my_rank*SIZE:…]``,
  then for every ``peer != my_rank`` ``pld.tile.remote_load`` the peer's
  slice at the **same chunk offset** and ``pl.add`` into the accumulator.
* **Phase 4 (stage-out)** — ``pl.store`` the accumulator into local ``out``.

Golden: rank ``r`` output is the element-wise sum of chunk ``r`` across all ranks:
``outputs[r][j] = sum(inputs[*][r*SIZE+j])``.

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

SIZE = 64  # matches COUNT_PER_RANK in simpler reduce_scatter_kernel.cpp
NR = pl.dynamic("NR")


def _expected_reduce_scatter(inputs: torch.Tensor) -> torch.Tensor:
    """Per-rank golden: element-wise sum of chunk ``r`` across all ranks."""
    n_ranks = inputs.shape[0]
    chunks = [inputs[:, 0, r * SIZE : (r + 1) * SIZE].sum(dim=0) for r in range(n_ranks)]
    return torch.stack(chunks).reshape(n_ranks, 1, SIZE)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Each rank stages up to 4*SIZE floats; unused tail is zero-padded."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + n_ranks * SIZE, dtype=torch.float32).reshape(1, n_ranks * SIZE)
        for r in range(n_ranks)
    ]
    padded = torch.zeros(n_ranks, 1, 4 * SIZE, dtype=torch.float32)
    padded[:, :, : n_ranks * SIZE] = torch.stack(rows)
    return padded


@pl.program
class ReduceScatterMesh:
    """Mesh reduce-scatter with dynamic rank count ``NR``."""

    @pl.function(type=pl.FunctionType.InCore)
    def reduce_scatter_step(
        self,
        inp: pl.Tensor[[1, 4 * SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, 4 * SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Four-phase mesh reduce-scatter on window-bound ``scratch`` / ``signal``."""
        ctx = pld.get_comm_ctx(scratch)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # Phase 1: stage-in — copy each chunk into scratch at its offset.
        for c in pl.range(nranks):
            chunk = pl.load(inp, [0, c * SIZE], [1, SIZE])
            pl.store(chunk, [0, c * SIZE], scratch)

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

        # Phase 3: reduce — sum this rank's chunk across all peers.
        acc = pl.load(scratch, [0, my_rank * SIZE], [1, SIZE])
        for peer in pl.range(nranks):
            if peer != my_rank:
                recv = pld.tile.remote_load(scratch, peer=peer, offsets=[0, my_rank * SIZE], shape=[1, SIZE])
                acc = pl.add(acc, recv)

        # Phase 4: stage-out — reduced chunk → local output.
        return pl.store(acc, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        inp: pl.Tensor[[1, 4 * SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, 4 * SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Per-device orchestration wrapper around ``reduce_scatter_step``."""
        return self.reduce_scatter_step(inp, out, scratch, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, NR * SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        """Launch one chip orchestration per rank with shared window buffers."""
        scratch_buf = pld.alloc_window_buffer(pld.world_size() * SIZE * 4)  # NR*SIZE x FP32
        signal_buf = pld.alloc_window_buffer(pld.world_size() * 4)  # NR x 1 x INT32

        for r in pl.range(pld.world_size()):
            scratch = pld.window(scratch_buf, [1, pld.world_size() * SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            self.chip_orch(
                inputs[r],
                outputs[r],
                scratch,
                signal,
                device=r,
            )
        return outputs


class TestL3ReduceScatter:
    """L3 distributed runtime: N-rank reduce-scatter via stage-in + notify/wait + remote_load."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_reduce_scatter(self, test_config, device_ids, n_ranks):
        """Compile and run mesh reduce-scatter for P=2 or P=4; skip when devices scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"reduce-scatter P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            ReduceScatterMesh,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_reduce_scatter(inputs)
        assert torch.allclose(outputs, expected), (
            f"reduce-scatter P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

        # Sanity: reduce-scatter output must be cross-rank sum, not raw local chunk.
        for r in range(n_ranks):
            local_chunk = inputs[r, 0, r * SIZE : (r + 1) * SIZE]
            assert not torch.allclose(outputs[r, 0], local_chunk), (
                f"reduce-scatter P={n_ranks}: rank {r} output must not equal"
                f" its own input chunk (reduction required)"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
