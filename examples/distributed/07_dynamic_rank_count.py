# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Dynamic rank count: one ring-shift source, compiled for any P.

Concepts introduced:
  - ``NR = pl.dynamic("NR")``: a runtime-resolved dimension. The rank count is
    not baked into any shape — the host world tensor is ``[NR, 1, SIZE]``.
  - the SAME source compiles and runs at P=2, P=3, P=4, ... — you only change
    ``-d``; no ``N_RANKS`` constant, no edits when the world grows.
  - kernels stay rank-agnostic: loops bound by the runtime ``pld.nranks(ctx)``,
    peers by ``(my_rank +/- 1) % nranks``.
  - the golden derives from the actual rank count at runtime.

This is step 06's ring shift (put/get) made rank-count-agnostic — the bridge
between the fixed P=2 substrate steps and the P=4 collective comparisons in
steps 08+.

Two modes, one step:
  - ``--mode put`` (default): stage, ``put`` into the next rank, signal, then
    read your own dst -> ``y[r] = x[(r-1) % P]``
  - ``--mode get``: stage, signal, ``get`` the next rank's slice -> ``y[r] =
    x[(r+1) % P]``

Run + walkthrough: see docs/en/user/distributed/12-dynamic_rank_count.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig

SIZE = 64
NR = pl.dynamic("NR")


@pl.jit.incore
def put_step(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    src: pld.DistributedTensor[[1, SIZE], pl.FP32],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[1, 1], pl.INT32],
):
    """Chip kernel: push (put) this rank's src slice into the next rank's dst."""
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)

    peer = (my_rank + 1) % nranks
    pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)

    # Signal the peer we pushed to, then wait for the rank that targets us.
    pld.system.notify(
        signal,
        peer=peer,
        offsets=[0, 0],
        value=1,
        op=pld.NotifyOp.AtomicAdd,
    )
    pld.system.wait(
        signal,
        offsets=[0, 0],
        expected=1,
        cmp=pld.WaitCmp.Ge,
    )

    # Our own dst was written by rank (r-1): read it back.
    recv = pl.load(dst, [0, 0], [1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y


@pl.jit.incore
def get_step(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    src: pld.DistributedTensor[[1, SIZE], pl.FP32],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[1, 1], pl.INT32],
):
    """Chip kernel: pull (get) the next rank's src slice into this rank's dst.

    Each rank notifies the rank that reads from it (the previous rank) and
    waits for the rank it reads from (the next rank), so the wait covers the
    get source for any rank count, not just the two-rank case.
    """
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)

    get_peer = (my_rank + 1) % nranks
    # Notify the rank that reads from us (the previous rank); our wait is then
    # satisfied by exactly the rank we get from (the next rank). nranks is
    # added before the -1 so the dividend is never negative.
    pld.system.notify(
        signal,
        peer=(my_rank + nranks - 1) % nranks,
        offsets=[0, 0],
        value=1,
        op=pld.NotifyOp.AtomicAdd,
    )
    pld.system.wait(
        signal,
        offsets=[0, 0],
        expected=1,
        cmp=pld.WaitCmp.Ge,
    )

    pld.tensor.get(dst, peer=get_peer, src=src)

    recv = pl.load(dst, [0, 0], [1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y


@pl.jit
def per_rank_put(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    src: pld.DistributedTensor[[1, SIZE], pl.FP32],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[1, 1], pl.INT32],
):
    return put_step(x, y, src, dst, signal)


@pl.jit
def per_rank_get(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    src: pld.DistributedTensor[[1, SIZE], pl.FP32],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[1, 1], pl.INT32],
):
    return get_step(x, y, src, dst, signal)


@pl.jit.host
def ring_put(
    x: pl.Tensor[[NR, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
):
    """Host orchestrator: the leading world dim is the dynamic rank count."""
    src_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    dst_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
        dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
        per_rank_put(x[r], y[r], src, dst, signal, device=r)


@pl.jit.host
def ring_get(
    x: pl.Tensor[[NR, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
):
    """Host orchestrator: the leading world dim is the dynamic rank count."""
    src_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    dst_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
        dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
        per_rank_get(x[r], y[r], src, dst, signal, device=r)


def expected_ring(inputs: torch.Tensor, get_mode: bool) -> torch.Tensor:
    """put: y[r] = x[(r-1) % P]; get: y[r] = x[(r+1) % P]."""
    n = inputs.shape[0]
    if get_mode:
        return torch.stack([inputs[(r + 1) % n] for r in range(n)])
    return torch.stack([inputs[(r - 1) % n] for r in range(n)])


def main() -> int:
    parser = argparse.ArgumentParser(description="07_dynamic_rank_count")
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
        help="comma-separated device ids -- any count >= 2; the same source runs at any P",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="put",
        choices=["put", "get"],
        help="initiator side: push with put or pull with get",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    get_mode = args.mode == "get"
    program = ring_get if get_mode else ring_put

    x = torch.randn((len(device_ids), 1, SIZE), dtype=torch.float32)
    y = torch.zeros((len(device_ids), 1, SIZE), dtype=torch.float32)

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

    expected = expected_ring(x, get_mode)
    assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
        f"{args.mode} P={len(device_ids)} mismatch: max diff = {(y - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
