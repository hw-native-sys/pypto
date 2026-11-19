# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stable routing-index compaction within each of two A5 warps.

Run without arguments to compile only (no device required).
Run on an allocated A5 device:
    python examples/advanced/04_simt_compaction.py --device 0 --method ballot
    python examples/advanced/04_simt_compaction.py --device 0 --method scan --rounds 20

Each warp owns one output row. Only its first counts[warp] indices are valid;
the remaining output cells retain their initial contents. This is a building
block for routing, not a cross-warp/global dispatch implementation.
"""

import argparse

import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig, benchmark


@pl.jit.aiv
def compact_warps(
    flags: pl.Tensor[[64], pl.INT32],
    indices: pl.InOut[pl.Tensor[[2, 32], pl.INT32]],
    counts: pl.InOut[pl.Tensor[[2], pl.INT32]],
    ranks: pl.InOut[pl.Tensor[[64], pl.INT32]],
    valid: pl.Scalar[pl.INT32],
    USE_BALLOT: pl.constexpr,
):
    """Compact nonzero flags stably; the caller supplies 0 <= valid <= 64.

    ranks holds the exclusive selected-lane count within the warp for every
    physical lane, including the zero-padded tail. It can index other payloads.
    """
    with pl.simt_scope(dims=(64, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        lane = pl.simt.lane_id()
        warp = tid // 32
        selected = pl.const(0, pl.INT32)
        if tid < valid:
            selected = pl.cast(pl.simt.load(flags, (tid,)) != 0, pl.INT32)
        # Reconverge before every collective: invalid input lanes contribute
        # zero but remain active, including lane 31 used to broadcast the count.
        if USE_BALLOT:
            bits = pl.simt.vote_ballot(selected != 0)
            # Physical lower-lane bits, independent of the active mask. Bit 31
            # is never included, so the population-count shifts remain signed-safe.
            lower = pl.simt.lanemask_lt()
            packed = bits & lower
            # Parallel bit count using existing scalar IR; no native popcount
            # operation is assumed. All intermediate values are nonnegative.
            packed = packed - ((packed >> 1) & 0x55555555)
            packed = (packed & 0x33333333) + ((packed >> 2) & 0x33333333)
            packed = (packed + (packed >> 4)) & 0x0F0F0F0F
            packed = packed + (packed >> 8)
            rank = pl.cast((packed + (packed >> 16)) & 0x3F, pl.INT32)
            inclusive = pl.cast(rank + selected, pl.INT32)
        else:
            inclusive = selected
            # The shuffle executes in every lane. Only consume its result when
            # the source is in range; otherwise self-copy would double the sum.
            for stage in pl.unroll(5):
                offset = pl.cast(1 << stage, pl.INT32)
                previous = pl.simt.shuffle_up(inclusive, offset)
                if lane >= offset:
                    inclusive = pl.cast(inclusive + previous, pl.INT32)
            rank = pl.cast(inclusive - selected, pl.INT32)
        total = pl.simt.shuffle_idx(inclusive, 31)
        pl.simt.store(ranks, (tid,), rank)
        if selected != 0:
            pl.simt.store(indices, (warp, rank), tid)
        if lane == 0:
            pl.simt.store(counts, (warp,), total)
    return indices, counts, ranks


@pl.jit
def compact(
    flags: pl.Tensor[[64], pl.INT32],
    indices: pl.InOut[pl.Tensor[[2, 32], pl.INT32]],
    counts: pl.InOut[pl.Tensor[[2], pl.INT32]],
    ranks: pl.InOut[pl.Tensor[[64], pl.INT32]],
    valid: pl.Scalar[pl.INT32],
    USE_BALLOT: pl.constexpr = True,
):
    return compact_warps(flags, indices, counts, ranks, valid, USE_BALLOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, help="Run on this allocated A5 device; omit to compile only")
    parser.add_argument("--method", choices=("ballot", "scan"), default="ballot")
    parser.add_argument("--valid", type=int, default=47)
    parser.add_argument("--rounds", type=int, default=0, help="Optional measured launches after 3 warmups")
    args = parser.parse_args()
    if not 0 <= args.valid <= 64 or args.rounds < 0:
        parser.error("--valid must be in [0, 64] and --rounds must be nonnegative")
    config = RunConfig(
        backend_type=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device if args.device is not None else 0,
    )
    compiled = compact.compile(USE_BALLOT=args.method == "ballot", config=config)
    # The simulator example sweep invokes this script bare on a CPU checkout.
    # Compile the A5 helper there, but never load or run it without an opt-in.
    if args.device is None:
        print(f"{args.method}: compiled; pass --device to execute on an allocated A5 device")
        return
    flags = (torch.arange(64) % 3 == 1).to(torch.int32)
    indices = torch.full((2, 32), -1, dtype=torch.int32)
    counts = torch.full((2,), -1, dtype=torch.int32)
    ranks = torch.full((64,), -1, dtype=torch.int32)
    compiled(flags, indices, counts, ranks, args.valid, config=config)
    for warp in range(2):
        ids = torch.arange(warp * 32, (warp + 1) * 32)
        expected = ids[(ids < args.valid) & (flags[ids] != 0)].to(torch.int32)
        assert counts[warp].item() == len(expected)
        torch.testing.assert_close(indices[warp, : len(expected)], expected, rtol=0, atol=0)
        assert torch.all(indices[warp, len(expected) :] == -1)
        print(f"warp {warp}: {expected.tolist()}")
    if args.rounds:
        stats = benchmark(
            compiled,
            (flags, indices, counts, ranks, args.valid),
            rounds=args.rounds,
            warmup=3,
            config=config,
        )
        print(stats)
        print("Device wall time includes orchestration and launch; it is not isolated instruction latency.")


if __name__ == "__main__":
    main()
