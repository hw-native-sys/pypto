# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device repro for issue #1670 (real trigger: dsv4 models/deepseek/v4/moe_ep.py).

EP-MoE ``combine`` is split into a push+barrier ``InCore`` kernel and a separate
local-reduce kernel that reads the ``routed_y_buf`` window. ``pld.tile.remote_-
store`` writes a local tile into a region of a *peer's* ``DistributedTensor``
window, so from local dataflow the window param looks read-only; ``ConvertTensor-
ToTileOps`` left it ``ParamDirection.In`` and ``DeriveCallDirections`` propagated
``ArgDirection.Input``. The writer was therefore not seen as a *producer* of the
window, the orchestration emitted no producer->consumer (RAW) edge to the reader,
and the reader raced on an unfilled window (~97.9% wrong, run-to-run varying).

The #1670 fix classifies the ``remote_store`` target as a write, so the writer
produces the window and the reader is ordered after the push + barrier.

Note: the reader is a separate ``InCore`` kernel (not ``Inline``). An inlined
reader is *spliced* into the orchestration rather than scheduled as its own
task, so it would not exercise the cross-kernel RAW race the bug is about — the
real trigger (``comb_reduce_spmd``) is likewise a separate kernel.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

N_RANKS = 2
D = 64


def _build_repro():
    @pl.program
    class Repro:
        # writer: cross-rank push of my row into the peer's window + barrier
        @pl.function(type=pl.FunctionType.InCore)
        def push_step(
            self,
            src: pl.Tensor[[N_RANKS, D], pl.BF16],
            win: pld.DistributedTensor[[N_RANKS, D], pl.BF16],
            done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[N_RANKS, D], pl.BF16]:
            peer = (my_rank + 1) % N_RANKS
            row = pl.load(src, [my_rank, 0], [1, D])
            pld.tile.remote_store(row, target=win, peer=peer, offsets=[my_rank, 0])
            # Cross-rank barrier: signal every other rank, then wait for them.
            for p in pl.range(N_RANKS):
                if p != my_rank:
                    pld.system.notify(
                        target=done, peer=p, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
                    )
            for s in pl.range(N_RANKS):
                if s != my_rank:
                    pld.system.wait(signal=done, offsets=[s, 0], expected=1, cmp=pld.WaitCmp.Ge)
            return src

        # reader: separate kernel that reads the now-filled window
        @pl.function(type=pl.FunctionType.InCore)
        def read_step(
            self,
            win: pld.DistributedTensor[[N_RANKS, D], pl.BF16],
            out: pl.Out[pl.Tensor[[N_RANKS, D], pl.BF16]],
        ) -> pl.Tensor[[N_RANKS, D], pl.BF16]:
            recv = pl.load(win, [0, 0], [N_RANKS, D])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(
            self,
            src: pl.Tensor[[N_RANKS, D], pl.BF16],
            out: pl.Out[pl.Tensor[[N_RANKS, D], pl.BF16]],
            win: pld.DistributedTensor[[N_RANKS, D], pl.BF16],
            done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[N_RANKS, D], pl.BF16]:
            self.push_step(src, win, done, my_rank)
            # Must run AFTER push_step's barrier fills `win` — the RAW edge #1670 drops.
            return self.read_step(win, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            srcs: pl.Tensor[[N_RANKS, N_RANKS, D], pl.BF16],
            outs: pl.Out[pl.Tensor[[N_RANKS, N_RANKS, D], pl.BF16]],
        ) -> pl.Tensor[[N_RANKS, N_RANKS, D], pl.BF16]:
            win_buf = pld.alloc_window_buffer(N_RANKS * D * 2)  # BF16 bytes
            done_buf = pld.alloc_window_buffer(N_RANKS * 4)  # INT32 slots
            for r in pl.range(pld.world_size()):
                win = pld.window(win_buf, [N_RANKS, D], dtype=pl.BF16)
                done = pld.window(done_buf, [N_RANKS, 1], dtype=pl.INT32)
                self.chip(srcs[r], outs[r], win, done, r, device=r)
            return outs

    return Repro


def test_remote_store_window_raw_device(test_config, device_ids):
    """The reader must observe the window the cross-rank push filled (issue #1670).

    Rank r pushes src[r][r, :] into peer=(r+1)%N's window at row r. With the fix
    the writer produces the window, so read_step is ordered after the push +
    barrier and sees the filled value; before the fix it raced on an unfilled
    window and returned non-deterministic zeros/garbage.
    """
    if len(device_ids) < 2:
        pytest.skip(f"window RAW repro needs 2 devices, got {device_ids}")
    program = _build_repro()
    compiled = ir.compile(
        program,
        platform=test_config.platform,
        distributed_config=DistributedConfig(device_ids=device_ids[:2], num_sub_workers=0),
    )
    srcs = (
        torch.arange(N_RANKS * N_RANKS * D, dtype=torch.float32)
        .reshape(N_RANKS, N_RANKS, D)
        .to(torch.bfloat16)
    )
    outs = torch.zeros((N_RANKS, N_RANKS, D), dtype=torch.bfloat16)
    compiled(srcs, outs)
    # Rank q's window row (q+1)%N is the one a peer pushed into; it must equal
    # that peer's own row. Unwritten rows stay at the zero-initialised window.
    for q in range(N_RANKS):
        written_row = (q + 1) % N_RANKS
        assert torch.equal(outs[q][written_row], srcs[written_row][written_row]), (
            f"rank {q}: window row {written_row} not filled before read "
            "(missing RAW edge -> raced on unfilled window?)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
