# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank all-to-all — hand-rolled 3-phase DSL port of simpler
``all_to_all_distributed`` reference.

Mirrors the 3-phase pattern of ``kernels/aiv/all_to_all_kernel.cpp``:

* **Phase 1 (stage-in)** — copy each destination chunk ``inp[dest, :]`` into
  this rank's slice of the window-bound ``data`` buffer so every peer can
  read the chunk destined for them.
* **Phase 2 (barrier)** — each rank ``Set``s every peer's ``signal`` cell
  via ``pld.system.notify`` and ``pld.system.wait``s until all ranks have
  staged.
* **Phase 3 (exchange)** — for ``src`` in 0..NR-1: read chunk ``my_rank``
  from peer ``src``'s window via ``pld.tile.get``, write to ``out[src, :]``.

Golden: ``output[rank, src, j] = src*1000 + rank*100 + j``.

Rank count uses ``NR = pl.dynamic("NR")`` in host tensor shapes; runtime
``inputs.shape[0]`` must match ``len(device_ids)`` / ``pld.world_size()``.

ST coverage: **P=2** and **P=4**.  One program body for both.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NR = pl.dynamic("NR")


def _expected_all_to_all(inputs: torch.Tensor) -> torch.Tensor:
    """Golden: output[rank, src, j] = src*1000 + rank*100 + j."""
    nranks = inputs.shape[0]
    src_idx = torch.arange(nranks, dtype=torch.float32).view(1, -1, 1)
    rank_idx = torch.arange(nranks, dtype=torch.float32).view(-1, 1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return src_idx * 1000 + rank_idx * 100 + j


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Each rank r fills input[r, d, j] = r*1000 + d*100 + j (chunk for dest d)."""
    r = torch.arange(n_ranks, dtype=torch.float32).view(-1, 1, 1)
    d = torch.arange(n_ranks, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return r * 1000 + d * 100 + j


@pl.program
class AllToAllMesh:
    """Mesh all-to-all with dynamic rank count ``NR``."""

    @pl.function(type=pl.FunctionType.InCore)
    def exchange_step(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        """Three-phase mesh all-to-all on window-bound ``data`` / ``signal``."""
        # CommContext must be acquired before any data rebinding so the
        # threading pass can trace it through all SSA uses of the window.
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # Phase 1: stage-in — copy each destination chunk into the local
        # window so every peer can read the slice destined for them.
        for dest in pl.range(nranks):
            local = pl.load(inp, [dest, 0], [1, SIZE])
            pl.store(local, [dest, 0], data)

        # Phase 2: barrier — notify every peer, wait on every peer slot.
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.system.notify(
                    signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.Set,
                )
        for src in pl.range(nranks):
            if src != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[src, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

        # Phase 3: exchange — read chunk my_rank from each peer's window.
        # Peer src staged the chunk for me at data[my_rank, :] in its
        # local window.  pld.tile.get reads it directly into out[src, :]
        # via a shared VEC staging tile (matches the C++ lowering).
        recv_stage = pl.tile.create([1, SIZE], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for src in pl.range(nranks):
            pld.tile.get(
                out,
                peer=src,
                src=data,
                stage=recv_stage,
                dst_offsets=[src, 0],
                src_offsets=[my_rank, 0],
                shape=[1, SIZE],
            )

        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        return self.exchange_step(inp, out, data, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, NR, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, NR, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(NR * SIZE * 4)
        signal_buf = pld.alloc_window_buffer(NR * 4)

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [NR, 1], dtype=pl.INT32)
            self.chip_orch(inputs[r], outputs[r], data, sig, device=r)
        return outputs


class TestL3AllToAll:
    """L3 distributed runtime: hand-rolled 3-phase mesh all-to-all.

    Validates that the raw DSL primitives (``pl.load`` / ``pl.store`` /
    ``pld.system.notify`` / ``pld.system.wait`` / ``pld.tile.get``) produce
    the correct rank-ordered personalized exchange identical to simpler's
    ``all_to_all_distributed`` reference.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_all_to_all(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"all-to-all P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            AllToAllMesh,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_all_to_all(inputs)
        assert torch.allclose(outputs, expected), (
            f"all-to-all P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
