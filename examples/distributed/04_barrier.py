# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Signals only: an N-rank barrier for one rendezvous, built from notify/wait.

Concepts introduced:
  - ``pld.system.notify`` (``NotifyOp.AtomicAdd``) / ``pld.system.wait``
    (``WaitCmp.Ge``): the cross-rank signal handshake
  - an N-rank barrier for one rendezvous: every rank notifies all peers, then
    waits on every peer -- no data moves
  - why AtomicAdd + Ge: each rank owns a dedicated row (offsets=[my_rank, 0]),
    so every signal cell has a single writer and a Set would work identically;
    AtomicAdd is the canonical notify that also generalizes to shared-cell
    barriers; Ge(1) passes when every peer has arrived
  - ``pld.rank(ctx)`` / ``pld.nranks(ctx)`` derived from ``pld.get_comm_ctx``
  - the reveal: ``pld.tensor.barrier`` provides the same synchronization as
    one call (run with ``--use-builtin``)

By default (hand-rolled) each rank owns row ``my_rank`` in every peer's signal
window. After the barrier, rank r's row in its OWN window is
``[1, ..., 0, ..., 1]`` -- a 1 in every column except its own (it never
notifies itself). The example surfaces that row as the output, so the result
proves every peer arrived, on every rank.

The example runs a single rendezvous. Reusing the same window for a second
barrier needs either a cell reset or a generation-specific expected threshold:
the counters are monotonic, so ``Ge(1)`` would already be satisfied.

With ``--use-builtin`` the barrier is one ``pld.tensor.barrier`` call. The
builtin synchronizes but does not leave a tally in the signal window, so the
reveal instead proves correct ordering with data: every rank stages its slice,
barriers, then reads the next rank's slice from its window (``remote_load`` --
used here in one line to observe the ordering; step 05 teaches it properly). A
missing barrier would let the load race the peer's store; the golden
``y[r] = x[(r+1) % N]`` holds only because the barrier ordered them.

Run + walkthrough: see docs/en/user/distributed/09-barrier.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

N_RANKS = 2
SIZE = 64


@pl.jit.incore
def barrier_handrolled(
    y: pl.Out[pl.Tensor[[N_RANKS, 1], pl.INT32]],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Chip kernel: hand-rolled N-rank barrier for one rendezvous, then surface the signal row."""
    ctx = pld.get_comm_ctx(signal)
    my_rank = pld.rank(ctx)

    # Every rank notifies all peers (AtomicAdd), then waits on all peers (Ge).
    # Each rank owns a dedicated row (offsets=[my_rank, 0]), so every cell has
    # one writer and a Set would work identically; AtomicAdd is the canonical
    # notify that also generalizes to shared-cell barriers. This is a single
    # rendezvous -- reusing the window for another barrier needs a cell reset
    # or a generation-specific expected threshold.
    for peer in pl.range(N_RANKS):
        if peer != my_rank:
            pld.system.notify(
                signal,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
    for src in pl.range(N_RANKS):
        if src != my_rank:
            pld.system.wait(
                signal,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    # Surface the signal row cell-by-cell as the observable. (A tile load of
    # the [2,1] INT32 window would be rejected: its 8-byte column is below the
    # 32-byte alignment ptoas requires for a col-major tile.)
    for i in pl.range(N_RANKS):
        val = pl.read(signal, [i, 0])
        pl.write(y, [i, 0], val)
    return y


@pl.jit.incore
def barrier_builtin(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Chip kernel: the reveal -- pld.tensor.barrier orders a remote_load."""
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)

    signal = pld.tensor.barrier(signal)

    peer = (my_rank + 1) % nranks
    recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y


@pl.jit
def per_rank_hand(
    y: pl.Out[pl.Tensor[[N_RANKS, 1], pl.INT32]],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    return barrier_handrolled(y, signal)


@pl.jit
def per_rank_builtin(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    return barrier_builtin(x, y, data, signal)


@pl.jit.host
def barrier_program_hand(
    y: pl.Out[pl.Tensor[[N_RANKS, N_RANKS, 1], pl.INT32]],
):
    signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        signal = pld.window(signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        per_rank_hand(y[r], signal, device=r)


@pl.jit.host
def barrier_program_builtin(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        per_rank_builtin(x[r], y[r], data, signal, device=r)


def expected_barrier(n_ranks: int) -> torch.Tensor:
    """Rank r's signal row is 1 everywhere except its own column (never self-notifies)."""
    expected = torch.zeros((n_ranks, n_ranks, 1), dtype=torch.int32)
    for r in range(n_ranks):
        for i in range(n_ranks):
            expected[r, i, 0] = 0 if i == r else 1
    return expected


def expected_shift(inputs: torch.Tensor) -> torch.Tensor:
    """y[r] = x[(r+1) % N]."""
    n = inputs.shape[0]
    return torch.stack([inputs[(r + 1) % n] for r in range(n)])


def main() -> int:
    parser = argparse.ArgumentParser(description="04_barrier")
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
        default="0,1",
        help=f"comma-separated device ids (need exactly {N_RANKS})",
    )
    parser.add_argument("--use-builtin", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) != N_RANKS:
        raise SystemExit(f"need exactly {N_RANKS} devices, got {device_ids}")

    use_builtin = args.use_builtin
    x = torch.randn((N_RANKS, 1, SIZE), dtype=torch.float32) if use_builtin else None
    y = (
        torch.zeros((N_RANKS, 1, SIZE), dtype=torch.float32)
        if use_builtin
        else torch.zeros((N_RANKS, N_RANKS, 1), dtype=torch.int32)
    )
    program = barrier_program_builtin if use_builtin else barrier_program_hand

    config = RunConfig(
        platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids,
            num_sub_workers=0,
        ),
    )
    if use_builtin:
        compiled = program.compile(x, y, config=config)
    else:
        compiled = program.compile(y, config=config)
    if args.compile_only:
        print(f"compile_only done: {compiled.output_dir}")
        return 0

    if use_builtin:
        compiled(x, y, config=config)
    else:
        compiled(y, config=config)

    if use_builtin:
        assert x is not None
        expected = expected_shift(x)
        assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
            f"barrier_builtin mismatch: max diff = {(y - expected).abs().max().item()}"
        )
    else:
        expected = expected_barrier(N_RANKS)
        assert torch.equal(y, expected), f"barrier mismatch: got {y.tolist()} expected {expected.tolist()}"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
