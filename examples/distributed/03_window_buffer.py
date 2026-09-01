# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Window memory: allocate a window buffer and read/write your own slice.

Concepts introduced:
  - ``pld.alloc_window_buffer`` / ``pld.window``: per-rank memory that the
    group can see, with a signal tail
  - a ``pld.DistributedTensor`` view of one rank's window slice
  - the 4 KB floor: a window is padded to at least 4 KiB even for tiny data
    (this rank's data is 1 KB, yet the buffer costs 4 KB)

Nothing is shared yet: every rank writes its own slice and reads it back, and
``y == x`` is the golden. The window is where all later cross-rank traffic
(barrier signals, remote_load/store, put/get) will live.

Run + walkthrough: see docs/en/user/distributed/08-window_buffer.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

N_RANKS = 2
SIZE = 256  # 1 KiB per rank -- below the 4 KiB window floor


@pl.jit.incore
def window_roundtrip(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
):
    """Chip kernel: write this rank's slice into its own window, read it back."""
    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)
    back = pl.load(data, [0, 0], [1, SIZE])
    y = pl.store(back, [0, 0], y)
    return y


@pl.jit
def per_rank(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
):
    """Per-device orchestration: one incore call, on this device."""
    return window_roundtrip(x, y, data)


@pl.jit.host
def window_program(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    """Host orchestrator: one window buffer, one view per rank, one dispatch per rank."""
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        per_rank(x[r], y[r], data, device=r)


def main() -> int:
    parser = argparse.ArgumentParser(description="03_window_buffer")
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
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) != N_RANKS:
        raise SystemExit(f"need exactly {N_RANKS} devices, got {device_ids}")

    x = torch.randn((N_RANKS, 1, SIZE), dtype=torch.float32)
    y = torch.zeros((N_RANKS, 1, SIZE), dtype=torch.float32)

    compiled = window_program.compile(
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

    assert torch.allclose(y, x, rtol=1e-5, atol=1e-5), (
        f"window_buffer mismatch: max diff = {(y - x).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
