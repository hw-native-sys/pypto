# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""All-reduce v3 (ring): chunked reduce-scatter + all-gather around the ring.

Concepts introduced:
  - the ring schedule: the same two-phase shape as step 09, but each rank only
    ever talks to its left neighbour — one chunk of ``N/P`` elements per
    step, for ``2 * (P-1)`` steps
  - why the ring scales: the per-step transfer stays ``N/P`` no matter how
    large P is, so the per-step size is constant as the world grows — same
    total bytes as two-phase, in smaller, constant-size rounds
  - a neighbour-ready handshake instead of a per-round barrier: the signal is
    ``[2*(P-1), P]`` — one row per round — and each rank notifies its *right*
    neighbour after a store and waits on its *left* neighbour before a
    remote_load (step 09 used ``[2, P]`` full-mesh barriers; the ring has more
    rounds but only ever synchronizes with the two adjacent ranks)
  - chunk indices rotate around the ring: in round s, the chunk you send to
    the right and receive from the left both shift by one

Same golden as steps 08/09: every rank receives the element-wise sum of all
rank slices, compared with a tolerance.

Run + walkthrough: see docs/en/user/distributed/15-allreduce_ring.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def build_ring_allreduce(nr: int):
    """Build the ring allreduce program for a compile-time rank count ``nr``.

    A rank-count factory for the same single reason as step 09:
    ``chunk = SIZE // nr`` is a **tile shape** (the ``[1, chunk]`` on every load
    below), and tile shapes must be known when the kernel is compiled. The
    ``[2*(nr-1), nr]`` signal is then written from the same constant, but a
    signal row count alone does not force it — step 08 keeps its rows dynamic
    and needs no factory at all.
    """
    if SIZE % nr != 0:
        raise ValueError(f"SIZE={SIZE} must be divisible by the rank count, got {nr}")
    total_rounds = 2 * (nr - 1)
    chunk = SIZE // nr

    @pl.program
    class RingAllreduce:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Monolithic ring allreduce: stage-in, RS loop, AG loop, stage-out.

            ``scratch`` holds ``nr`` chunks laid out flat in ``[1, SIZE]``;
            chunk *c* starts at offset ``c * chunk``. The signal carries
            ``2*(nr-1)`` rows, one per round, and is a neighbour-ready
            handshake: notify the *right* neighbour after a store, wait on the
            *left* neighbour before a ``remote_load`` (payload reads the left
            neighbour; synchronization touches only both adjacent ranks).
            ``alloc_window_buffer`` zero-initialises every cell, so per-cell
            ``AtomicAdd(0->1)`` / ``WaitGe(1)`` is safe without a reset.
            """
            ctx = pld.get_comm_ctx(scratch)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)
            left = (my_rank - 1 + nranks) % nranks
            right = (my_rank + 1) % nranks

            # Phase 1 — stage-in: copy each local input chunk into scratch,
            # then tell the right neighbour the ring is ready (signal row 0).
            for c in pl.range(nranks):
                src_tile = pl.load(x, [0, c * chunk], [1, chunk])
                scratch = pl.store(src_tile, [0, c * chunk], scratch)
            pld.system.notify(
                signal,
                peer=right,
                offsets=[0, my_rank],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

            # Phase 2 — reduce-scatter: (nr-1) ring steps. In step s the chunk
            # we add from the left and the chunk we forward both rotate.
            for s in pl.range(nranks - 1):
                step = s + 1
                recv_add_idx = (my_rank - step - 1 + nranks) % nranks
                left_send_idx = (left - step + nranks) % nranks
                rs_round = s

                # Wait for the left neighbour's round-rs_round chunk (staged by
                # its previous store, signalled on row rs_round).
                pld.system.wait(
                    signal,
                    offsets=[rs_round, left],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

                # Add the left neighbour's send chunk into our accumulator chunk.
                recv = pld.tile.remote_load(
                    scratch,
                    peer=left,
                    offsets=[0, left_send_idx * chunk],
                    shape=[1, chunk],
                )
                acc = pl.load(scratch, [0, recv_add_idx * chunk], [1, chunk])
                acc = pl.add(acc, recv)
                scratch = pl.store(acc, [0, recv_add_idx * chunk], scratch)

                # The store above stages the right neighbour's round-(rs_round+1)
                # send: signal it on the next row.
                pld.system.notify(
                    signal,
                    peer=right,
                    offsets=[rs_round + 1, my_rank],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

            # Phase 3 — all-gather: (nr-1) ring steps.
            for s in pl.range(nranks - 1):
                step = s + 1
                recv_idx = (my_rank - step + nranks) % nranks
                left_send_idx = (left - step + 1 + nranks) % nranks
                ag_round = (nranks - 1) + s

                # Wait for the left neighbour's chunk staged for this round.
                pld.system.wait(
                    signal,
                    offsets=[ag_round, left],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

                # Copy the left neighbour's send chunk into our local chunk.
                recv = pld.tile.remote_load(
                    scratch,
                    peer=left,
                    offsets=[0, left_send_idx * chunk],
                    shape=[1, chunk],
                )
                scratch = pl.store(recv, [0, recv_idx * chunk], scratch)

                # Pass the completion on to the right neighbour — except after
                # the final round, whose row would exceed the signal's
                # 2*(nr-1) rows.
                if s < nranks - 2:
                    pld.system.notify(
                        signal,
                        peer=right,
                        offsets=[ag_round + 1, my_rank],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

            # Phase 4 — stage-out: write the concatenated chunks to the output.
            for c in pl.range(nranks):
                src_tile = pl.load(scratch, [0, c * chunk], [1, chunk])
                y = pl.store(src_tile, [0, c * chunk], y)

            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Per-device orchestration: one incore call, on this device."""
            return self.ring_step(x, y, scratch, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def ring_allreduce(
            self,
            x: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            """Host orchestrator: shared scratch + signal windows, one dispatch per rank."""
            scratch_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
            signal_buf = pld.alloc_window_buffer([total_rounds, nr], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                scratch = pld.window(scratch_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [total_rounds, nr], dtype=pl.INT32)
                self.per_rank(x[r], y[r], scratch, signal, device=r)
            return y

    return RingAllreduce


def expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    """Every rank receives the element-wise sum of all rank slices."""
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="10_allreduce_ring")
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="0,1,2,3",
        help="comma-separated device ids (>= 2, dividing SIZE); P>=4 shows the constant per-step size",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    nr = len(device_ids)
    program = build_ring_allreduce(nr)

    x = torch.randn((nr, 1, SIZE), dtype=torch.float32)
    y = torch.zeros((nr, 1, SIZE), dtype=torch.float32)

    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids,
            num_sub_workers=0,
        ),
    )
    if args.compile_only:
        print(f"compile_only done: {compiled.output_dir}")
        return 0

    compiled(x, y)

    expected = expected_allreduce(x)
    assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
        f"ring allreduce P={nr} mismatch: max diff = {(y - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
