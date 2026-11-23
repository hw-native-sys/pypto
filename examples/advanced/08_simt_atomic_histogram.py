# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Count eight bins and allocate per-bin tickets with contending SIMT threads.

atomic_add returns the old count as a unique ticket. atomic_or accumulates
which of 16 input-position classes contributed to each bin. Tickets run from
0 to count-1 within each bin, in unspecified thread order.

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
    keys, stats, tickets = make_inputs()
    compiled(keys, stats, tickets, args.valid, config=config)
    print("Bin counts:", stats[0])
    print("Position masks:", stats[1])
    print("Tickets:", tickets[: args.valid])


if __name__ == "__main__":
    main()
