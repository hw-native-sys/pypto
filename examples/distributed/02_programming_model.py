# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""The three-level model: ``@pl.jit.host`` -> ``@pl.jit`` -> ``@pl.jit.incore``.

Concepts introduced:
  - which processor runs what: ``host_orch`` (host CPU, control plane) ->
    ``device_orch`` (per-device orchestration) -> ``scale_by_rank`` (AICore)
  - the host loop owns the control plane and hands each rank its parameters;
    the chip kernel only does math on its own tile
  - per-rank parameters flow host -> device -> chip: every rank computes
    ``y[r] = x[r] * (r + 1)``

This is the same skeleton as 01_hello_rank; the point here is the labels — three
functions, three levels, one processor each.

Run + walkthrough: see docs/en/user/distributed/07-programming_model.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

N_RANKS = 2
ROWS = 8
COLS = 8


@pl.jit.incore
def scale_by_rank(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    rank: pl.Scalar[pl.INT32],
):
    """Level 3 — the AICore kernel: ``y = x * (rank + 1)``.

    Runs on the chip (AICore). It sees only its own tile and the scalar
    parameter the device orchestration forwarded to it. Tensor args precede the
    scalar (tensors-first / scalars-last TaskArgs packing rule).
    """
    tile = pl.load(x, [0, 0], [ROWS, COLS])
    # Cast the INT32 rank scalar to FP32 first (a scalar *cast* is legal; a
    # scalar *add* is not — arith.addf is marked illegal on the AICore, scalars
    # live on the AICPU). So the +1 is folded into vector ops: x*rank + x.
    rank_f32 = pl.cast(rank, target_type=pl.FP32)
    scaled = pl.mul(tile, rank_f32)  # x * rank
    result = pl.add(scaled, tile)  # x * rank + x == x * (rank + 1)
    y = pl.store(result, [0, 0], y)
    return y


@pl.jit
def device_orch(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    rank: pl.Scalar[pl.INT32],
):
    """Level 2 — per-device orchestration (AICPU): forwards to the AICore kernel."""
    return scale_by_rank(x, y, rank)


@pl.jit.host
def host_orch(
    x: pl.Tensor[[N_RANKS, ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, ROWS, COLS], pl.FP32]],
):
    """Level 1 — the host orchestrator (host CPU): control plane, one launch per rank."""
    for r in pl.range(pld.world_size()):
        device_orch(x[r], y[r], r, device=r)


def main() -> int:
    parser = argparse.ArgumentParser(description="02_programming_model")
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

    compiled = host_orch.compile(
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

    expected = x * torch.arange(1, N_RANKS + 1, dtype=torch.float32).view(N_RANKS, 1, 1)
    assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
        f"programming_model mismatch: max diff = {(y - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
