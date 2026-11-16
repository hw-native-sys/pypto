# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare Tile/SIMT and Tile/SIMD row updates on an allocated A5 device.

For columns below valid, update a seed Tile with (src + 1) * row_scale;
then add 2 to every element and store the result. A second output reads the
original seed Tile after the update. The caller supplies 0 <= valid <= 64.

Run: python examples/advanced/06_simt_row_update.py --device 0 --method simt
     python examples/advanced/06_simt_row_update.py --device 0 --method simd
"""

import argparse

import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@pl.jit.aiv
def row_update_kernel(
    src: pl.Tensor[[8, 64], pl.FP32],
    scales: pl.Tensor[[1, 8], pl.FP32],
    seed: pl.Tensor[[8, 64], pl.FP32],
    dst: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
    snapshot: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
    valid: pl.Scalar[pl.INT32],
    USE_SIMT: pl.constexpr,
):
    values = pl.add(pl.load(src, [0, 0], [8, 64]), 1.0)
    scale_row = pl.load(scales, [0, 0], [1, 8])
    previous = pl.load(seed, [0, 0], [8, 64])
    saved = previous
    if USE_SIMT:
        # The compiler preserves saved's original Tile version while rebinding
        # previous to the partially updated result at the scope boundary.
        with pl.simt_scope(dims=(64, 8, 1)):
            col = pl.simt.thread_idx(pl.simt.Axis.X)
            row = pl.simt.thread_idx(pl.simt.Axis.Y)
            if col < valid:
                value = pl.simt.load(values, (row, col))
                scale = pl.simt.load(scale_row, (0, row))
                pl.simt.store(previous, (row, col), value * scale)
    else:
        # Native Tile broadcast multiply computes the candidate. SIMD masks
        # perform the same partial update; invalid columns retain the seed.
        scale_column = pl.reshape(scale_row, [8, 1])
        candidate = pl.row_expand_mul(values, scale_column)
        flat = pl.reshape(pl.tile.move(previous, target_memory=pl.MemorySpace.Vec), [512])
        candidate_flat = pl.reshape(candidate, [512])
        with pl.reg_scope():
            columns = pl.reg.index_gen(pl.const(0, pl.INT32), lanes=64, dtype=pl.INT32)
            limit = pl.reg.broadcast(valid, lanes=64, dtype=pl.INT32)
            full = pl.reg.full_mask(lanes=64)
            mask = pl.reg.compare(columns, limit, full, predicate="lt")
            for row in pl.unroll(8):
                value = pl.reg.load(candidate_flat, offset=row * 64, lanes=64)
                flat = pl.reg.store(flat, offset=row * 64, value=value, mask=mask)
        previous = pl.reshape(flat, [8, 64])
    # This consumer must see the updated Tile, including the retained tail.
    result = pl.store(pl.add(previous, 2.0), [0, 0], dst)
    original = pl.store(saved, [0, 0], snapshot)
    return result, original


@pl.jit
def row_update(
    src: pl.Tensor[[8, 64], pl.FP32],
    scales: pl.Tensor[[1, 8], pl.FP32],
    seed: pl.Tensor[[8, 64], pl.FP32],
    dst: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
    snapshot: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
    valid: pl.Scalar[pl.INT32],
    USE_SIMT: pl.constexpr = True,
):
    return row_update_kernel(src, scales, seed, dst, snapshot, valid, USE_SIMT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--method", choices=("simt", "simd"), default="simt")
    parser.add_argument("--valid", type=int, default=33)
    args = parser.parse_args()
    if not 0 <= args.valid <= 64:
        parser.error("--valid must be in [0, 64]")
    src = torch.arange(512, dtype=torch.float32).reshape(8, 64) * 0.25
    scales = torch.tensor([[0.0, -2.0, 0.5, 4.0, -0.25, 1.0, 2.0, -4.0]])
    seed = -src - 100.0
    dst, snapshot = torch.zeros_like(src), torch.zeros_like(src)
    row_update(
        src,
        scales,
        seed,
        dst,
        snapshot,
        args.valid,
        USE_SIMT=args.method == "simt",
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            device_id=args.device,
        ),
    )
    expected = seed.clone()
    expected[:, : args.valid] = ((src + 1) * scales.reshape(8, 1))[:, : args.valid]
    torch.testing.assert_close(dst, expected + 2, rtol=0, atol=0)
    torch.testing.assert_close(snapshot, seed, rtol=0, atol=0)
    print(f"{args.method}: row update, retained tails and original Tile verified")


if __name__ == "__main__":
    main()
