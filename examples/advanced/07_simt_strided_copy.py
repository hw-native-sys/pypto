# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Copy between pitched GM subviews with a three-dimensional SIMT block.

The source, destination and thread-tag output have different physical row
strides. Only the first valid elements of each 8x256 window are updated;
all surrounding guards and inactive elements retain their initial values.

Run bare to compile without a device. On an allocated A5 device:
    python examples/advanced/07_simt_strided_copy.py --device 0 --dims 8 4 8
"""

import argparse
import math

import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@pl.jit.aiv
def strided_copy_kernel(
    src: pl.Tensor[[8, 256], pl.FP32],
    dst: pl.InOut[pl.Tensor[[8, 256], pl.FP32]],
    tags: pl.InOut[pl.Tensor[[8, 256], pl.INT32]],
    valid: pl.Scalar[pl.INT32],
    X: pl.constexpr,
    Y: pl.constexpr,
    Z: pl.constexpr,
):
    with pl.simt_scope(dims=(X, Y, Z)):
        x = pl.simt.thread_idx(pl.simt.Axis.X)
        y = pl.simt.thread_idx(pl.simt.Axis.Y)
        z = pl.simt.thread_idx(pl.simt.Axis.Z)
        nx = pl.simt.block_dim(pl.simt.Axis.X)
        ny = pl.simt.block_dim(pl.simt.Axis.Y)
        nz = pl.simt.block_dim(pl.simt.Axis.Z)
        linear = x + nx * (y + ny * z)
        threads = nx * ny * nz
        lane = pl.simt.lane_id()
        # A thread-strided loop separates logical data size from block size.
        # Physical row pitches come from the Tensor views, not the 256 columns.
        for i in pl.range(linear, valid, threads):
            row = i // 256
            col = i % 256
            value = pl.simt.load(src, (row, col))
            pl.simt.store(dst, (row, col), value)
            tag = pl.cast(z * 1000000 + y * 10000 + x * 32 + lane, pl.INT32)
            pl.simt.store(tags, (row, col), tag)
    return dst, tags


@pl.jit
def strided_copy(
    src: pl.Tensor[[10, 288], pl.FP32],
    dst: pl.InOut[pl.Tensor[[12, 320], pl.FP32]],
    tags: pl.InOut[pl.Tensor[[10, 352], pl.INT32]],
    valid: pl.Scalar[pl.INT32],
    X: pl.constexpr = 8,
    Y: pl.constexpr = 4,
    Z: pl.constexpr = 8,
):
    # Construct views outside SIMT. The runtime passes each origin once;
    # the helper must retain the parent pitches 288, 320 and 352.
    source = pl.slice(src, [8, 256], [1, 7])
    destination = pl.slice(dst, [8, 256], [2, 11])
    thread_tags = pl.slice(tags, [8, 256], [1, 13])
    updated, written_tags = strided_copy_kernel(source, destination, thread_tags, valid, X, Y, Z)
    dst = pl.assemble(dst, updated, [2, 11])
    tags = pl.assemble(tags, written_tags, [1, 13])
    return dst, tags


def inputs(repeat: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.arange(10 * 288, dtype=torch.float32).reshape(10, 288) * 0.25 + repeat
    dst = torch.full((12, 320), -777.0 - repeat)
    tags = torch.full((10, 352), -999 - repeat, dtype=torch.int32)
    return src, dst, tags


def expected_outputs(src, dst, tags, valid: int, dims: tuple[int, int, int]):
    """Independent CPU oracle includes every guard and every inactive element."""
    expected, expected_tags = dst.clone(), tags.clone()
    indices = torch.arange(valid)
    rows, columns = indices // 256, indices % 256
    expected[rows + 2, columns + 11] = src[rows + 1, columns + 7]
    thread = indices % math.prod(dims)
    x = thread % dims[0]
    y = (thread // dims[0]) % dims[1]
    z = thread // (dims[0] * dims[1])
    expected_tags[rows + 1, columns + 13] = (z * 1000000 + y * 10000 + x * 32 + thread % 32).int()
    return expected, expected_tags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, help="Allocated A5 device; omit to compile only")
    parser.add_argument("--dims", type=int, nargs=3, default=(8, 4, 8), metavar=("X", "Y", "Z"))
    parser.add_argument("--valid", type=int, default=257)
    args = parser.parse_args()
    dims = tuple(args.dims)
    if min(dims) <= 0 or math.prod(dims) > 2048 or not 0 <= args.valid <= 2048:
        parser.error("--dims must be positive with product <= 2048; --valid must be in [0, 2048]")
    config = RunConfig(
        backend_type=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device if args.device is not None else 0,
    )
    compiled = strided_copy.compile(X=dims[0], Y=dims[1], Z=dims[2], config=config)
    if args.device is None:
        print("compiled; pass --device to execute on an allocated A5 device")
        return
    src, dst, tags = inputs()
    original = src.clone()
    expected, expected_tags = expected_outputs(src, dst, tags, args.valid, dims)
    compiled(src, dst, tags, args.valid, config=config)
    torch.testing.assert_close(src, original, rtol=0, atol=0)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    torch.testing.assert_close(tags, expected_tags, rtol=0, atol=0)
    print(f"PASS dims={dims}, valid={args.valid}: copied values, thread coordinates, guards and tails")


if __name__ == "__main__":
    main()
