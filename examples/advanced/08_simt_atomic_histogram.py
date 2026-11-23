# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Count eight bins and allocate per-bin tickets with contending SIMT threads.

atomic_add returns the old count as a unique ticket. atomic_or accumulates
which of 16 input-position classes contributed to each bin. Ticket order is
not stable: the host checks a permutation of 0..count-1 for each bin.

Run bare to compile only. On an allocated A5 device:
    python examples/advanced/08_simt_atomic_histogram.py --device 0 --valid 47
The caller initializes stats to zero and supplies keys in [0, 7].
"""

import argparse

import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@pl.jit.aiv
def histogram_kernel(
    keys: pl.Tensor[[64], pl.INT32],
    stats: pl.InOut[pl.Tensor[[2, 8], pl.INT32]],
    tickets: pl.InOut[pl.Tensor[[64], pl.INT32]],
    valid: pl.Scalar[pl.INT32],
):
    with pl.simt_scope(dims=(64, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        if tid < valid:
            bucket = pl.simt.load(keys, (tid,))
            ticket = pl.simt.atomic_add(stats, (0, bucket), pl.const(1, pl.INT32))
            bit = pl.cast(1 << (tid % 16), pl.INT32)
            # The atomic remains a side effect even though its old value is unused.
            pl.simt.atomic_or(stats, (1, bucket), bit)
            pl.simt.store(tickets, (tid,), ticket)
    return stats, tickets


@pl.jit
def histogram(
    keys: pl.Tensor[[64], pl.INT32],
    stats: pl.InOut[pl.Tensor[[2, 8], pl.INT32]],
    tickets: pl.InOut[pl.Tensor[[64], pl.INT32]],
    valid: pl.Scalar[pl.INT32],
):
    return histogram_kernel(keys, stats, tickets, valid)


def make_inputs() -> tuple[torch.Tensor, ...]:
    """Create initialized counters, masks and ticket storage on the host."""
    keys = ((torch.arange(64) * 5 // 3) % 8).to(torch.int32)
    stats = torch.zeros((2, 8), dtype=torch.int32)
    tickets = torch.full((64,), -1, dtype=torch.int32)
    return keys, stats, tickets


def check_outputs(inputs: tuple[torch.Tensor, ...], original: tuple[torch.Tensor, ...], valid: int) -> None:
    """Check counts and membership, allowing any serialization of the atomics."""
    keys, stats, tickets = inputs
    original_keys = original[0]
    expected_counts = torch.bincount(original_keys[:valid].long(), minlength=8).to(torch.int32)
    expected_masks = torch.zeros(8, dtype=torch.int32)
    for tid in range(valid):
        bucket = int(original_keys[tid])
        expected_masks[bucket] |= 1 << (tid % 16)
    torch.testing.assert_close(keys, original_keys, rtol=0, atol=0)
    torch.testing.assert_close(stats[0], expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(stats[1], expected_masks, rtol=0, atol=0)
    for bucket in range(8):
        actual = tickets[:valid][original_keys[:valid] == bucket].sort().values
        expected = torch.arange(int(expected_counts[bucket]), dtype=torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(tickets[valid:], original[2][valid:], rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, help="Allocated A5 device; omit to compile only")
    parser.add_argument("--valid", type=int, default=47)
    args = parser.parse_args()
    if not 0 <= args.valid <= 64:
        parser.error("--valid must be in [0, 64]")
    config = RunConfig(
        backend_type=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device if args.device is not None else 0,
    )
    compiled = histogram.compile(config=config)
    if args.device is None:
        print("compiled; pass --device to execute on an allocated A5 device")
        return
    inputs = make_inputs()
    original = tuple(value.clone() for value in inputs)
    compiled(*inputs, args.valid, config=config)
    check_outputs(inputs, original, args.valid)
    counts = [int(count) for count in inputs[1][0]]
    print(f"PASS valid={args.valid}: counts={counts}, masks and ticket permutations verified")


if __name__ == "__main__":
    main()
