# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""The distributed "hello world": rank identity, world_size, and per-rank dispatch.

Concepts introduced:
  - ``@pl.jit.host``: the host orchestrator that launches one child per rank
  - ``device=r``: pin each child dispatch to the rank's own NPU device
  - ``pld.world_size()``: the number of ranks this program was compiled for
  - ``DistributedConfig(device_ids=...)``: which NPU devices participate
  - the three call forms: ``@pl.jit.host`` (host) -> ``@pl.jit`` (per-device)
    -> ``@pl.jit.incore`` (chip)

Every rank computes ``y[r] = x[r] + r``, so the output row alone tells you which
rank produced it — that is the whole point: the same kernel runs on every card,
parameterised by its rank.

Note the signature order: all tensor args come before the scalar ``rank`` arg.
The distributed TaskArgs packing requires tensors-first / scalars-last — a
scalar wedged between tensors fails at runtime with "cannot add tensor after
scalar".

Run + walkthrough: see docs/en/user/distributed/06-hello_rank.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig

N_RANKS = 2
ROWS = 8
COLS = 8


@pl.jit.incore
def add_rank(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    rank: pl.Scalar[pl.INT32],
):
    """Chip kernel: ``y = x + rank``. Runs on the AICore of this rank's device."""
    tile = pl.load(x, [0, 0], [ROWS, COLS])
    rank_f32 = pl.cast(rank, target_type=pl.FP32)
    tile = pl.add(tile, rank_f32)
    y = pl.store(tile, [0, 0], y)
    return y


@pl.jit
def per_rank(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    rank: pl.Scalar[pl.INT32],
):
    """Per-device orchestration: one incore call, on this device."""
    return add_rank(x, y, rank)


@pl.jit.host
def hello_rank(
    x: pl.Tensor[[N_RANKS, ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, ROWS, COLS], pl.FP32]],
):
    """Host orchestrator: launch ``per_rank`` once per rank, pinned to ``device=r``."""
    for r in pl.range(pld.world_size()):
        per_rank(x[r], y[r], r, device=r)


def main() -> int:
    parser = argparse.ArgumentParser(description="01_hello_rank")
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

    x = torch.randn((N_RANKS, ROWS, COLS), dtype=torch.float32)
    y = torch.zeros((N_RANKS, ROWS, COLS), dtype=torch.float32)

    compiled = hello_rank.compile(
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

    expected = x + torch.arange(N_RANKS, dtype=torch.float32).view(N_RANKS, 1, 1)
    assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
        f"hello_rank mismatch: max diff = {(y - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
