# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Tile-level point-to-point: remote_load / remote_store, a one-step ring shift.

Concepts introduced:
  - ``pld.tile.remote_load``: pull a peer's window slice into a local tile
  - ``pld.tile.remote_store``: push a local tile into a peer's window slice
  - ``pld.DistributedTensor`` vs plain ``Tensor``: only the window-bound type
    is visible to other ranks
  - a one-step ring shift behind a barrier

Two modes show the same shift from the two sides of the RMA:
  - ``--mode load``  (default): stage your slice, barrier, then *pull* the next
    rank's slice -> ``y[r] = x[(r+1) % N]``
  - ``--mode store``: *push* your slice into the next rank's window, barrier,
    then read your own window (which the previous rank just wrote) ->
    ``y[r] = x[(r-1) % N]``

Run + walkthrough: see docs/en/user/distributed/10-remote_load_store.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig

N_RANKS = 2
SIZE = 64


@pl.jit.incore
def shift_by_load(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Chip kernel: stage your slice, barrier, then remote_load the next rank's."""
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


@pl.jit.incore
def shift_by_store(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Chip kernel: remote_store your slice into the next rank, then read your own."""
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    peer = (my_rank + 1) % nranks
    pld.tile.remote_store(local, data, peer=peer, offsets=[0, 0])

    signal = pld.tensor.barrier(signal)

    # Rank (r-1) just stored into OUR window: read it back locally.
    back = pl.load(data, [0, 0], [1, SIZE])
    y = pl.store(back, [0, 0], y)
    return y


@pl.jit
def per_rank_load(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    return shift_by_load(x, y, data, signal)


@pl.jit
def per_rank_store(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    return shift_by_store(x, y, data, signal)


@pl.jit.host
def ring_shift_load(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        per_rank_load(x[r], y[r], data, signal, device=r)


@pl.jit.host
def ring_shift_store(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        per_rank_store(x[r], y[r], data, signal, device=r)


def expected_shift(inputs: torch.Tensor, store_mode: bool) -> torch.Tensor:
    """load: y[r] = x[(r+1) % N]; store: y[r] = x[(r-1) % N]."""
    n = inputs.shape[0]
    if store_mode:
        return torch.stack([inputs[(r - 1) % n] for r in range(n)])
    return torch.stack([inputs[(r + 1) % n] for r in range(n)])


def main() -> int:
    parser = argparse.ArgumentParser(description="05_remote_load_store")
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
    parser.add_argument(
        "--mode",
        type=str,
        default="load",
        choices=["load", "store"],
        help="RMA direction: pull with remote_load (load) or push with remote_store (store)",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) != N_RANKS:
        raise SystemExit(f"need exactly {N_RANKS} devices, got {device_ids}")

    store_mode = args.mode == "store"
    program = ring_shift_store if store_mode else ring_shift_load

    x = torch.randn((N_RANKS, 1, SIZE), dtype=torch.float32)
    y = torch.zeros((N_RANKS, 1, SIZE), dtype=torch.float32)

    compiled = program.compile(
        x,
        y,
        config=RunConfig(
            platform=args.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
        ),
    )
    if args.compile_only:
        print(f"compile_only done: {compiled.output_dir}")
        return 0

    compiled(x, y, config=RunConfig(platform=args.platform))

    expected = expected_shift(x, store_mode)
    assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
        f"remote_{args.mode} mismatch: max diff = {(y - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
