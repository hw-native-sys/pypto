# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""All-reduce v2 (two-phase): reduce-scatter then all-gather.

Concepts introduced:
  - the two-phase shape of every efficient all-reduce: first reduce-scatter
    (each rank owns one chunk of the reduced result), then all-gather (every
    rank collects every chunk) — roughly half of mesh's remote traffic:
    ``2 * (P-1) / P * N`` bytes per rank vs mesh's ``(P-1) * N``
  - chunk ownership: rank r reduces chunk r and stages it in its result
    window; the all-gather then reads one chunk per peer
  - two barriers, two rounds of the ``notify``/``wait`` handshake from step 04:
    barrier A before any reduce-scatter read (all inputs staged), barrier B
    before any all-gather read (all reduced chunks staged)
  - the signal is a ``[2, nr]`` matrix — one row per barrier round (the ring
    step will generalise this to one row per round)

Same golden as step 08: every rank receives the element-wise sum of all rank
slices, compared with a tolerance. Run at P>=4 to see the traffic difference
from mesh (P=2 collapses the two algorithms into the same exchange).

Run + walkthrough: see docs/en/user/distributed/14-allreduce_two_phase.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def build_two_phase_allreduce(nr: int):
    """Build the two-phase allreduce program for a compile-time rank count ``nr``.

    A rank-count factory — the first step that needs one. Step 08 has none: its
    signal row count stays ``pl.dynamic``. What forces a compile-time ``nr``
    here is ``chunk = SIZE // nr``, used as a **tile shape** (the ``[1, chunk]``
    on every ``pl.load`` / ``remote_load`` below), and tile shapes must be known
    when the kernel is compiled. The ``[2, nr]`` signal then falls out of the
    same constant for free — but a signal shape on its own would not have
    required it. One source still serves any P that divides ``SIZE`` evenly
    (pick it with ``-d``).
    """
    if SIZE % nr != 0:
        raise ValueError(f"SIZE={SIZE} must be divisible by the rank count, got {nr}")
    chunk = SIZE // nr

    @pl.program
    class TwoPhaseAllreduce:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            result: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, nr], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Chip kernel: stage, barrier, reduce-scatter, barrier, all-gather."""
            ctx = pld.get_comm_ctx(data)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)

            # Phase 1 — stage this rank's slice into its window slot.
            local = pl.load(x, [0, 0], [1, SIZE])
            data = pl.store(local, [0, 0], data)

            # Barrier A (signal row 0) — all inputs staged before the RS reads.
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[0, my_rank],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(nranks):
                if src != my_rank:
                    pld.system.wait(
                        signal,
                        offsets=[0, src],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # Phase 2 — reduce-scatter: rank r owns chunk r of the result.
            acc = pl.load(data, [0, my_rank * chunk], [1, chunk])
            for peer in pl.range(nranks):
                if peer != my_rank:
                    recv = pld.tile.remote_load(
                        data,
                        peer=peer,
                        offsets=[0, my_rank * chunk],
                        shape=[1, chunk],
                    )
                    acc = pl.add(acc, recv)
            result = pl.store(acc, [0, my_rank * chunk], result)

            # Barrier B (signal row 1) — all reduced chunks staged before the
            # all-gather reads.
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[1, my_rank],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(nranks):
                if src != my_rank:
                    pld.system.wait(
                        signal,
                        offsets=[1, src],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # Phase 3 — all-gather: rank r reads every rank's reduced chunk.
            for c in pl.range(nranks):
                recv = pld.tile.remote_load(
                    result,
                    peer=c,
                    offsets=[0, c * chunk],
                    shape=[1, chunk],
                )
                y = pl.store(recv, [0, c * chunk], y)

            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            result: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, nr], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Per-device orchestration: one incore call, on this device."""
            return self.reduce_step(x, y, data, result, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def two_phase_allreduce(
            self,
            x: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            """Host orchestrator: shared data + result + signal windows, one dispatch per rank."""
            data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
            result_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
            signal_buf = pld.alloc_window_buffer([2, nr], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                result = pld.window(result_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [2, nr], dtype=pl.INT32)
                self.per_rank(x[r], y[r], data, result, signal, device=r)
            return y

    return TwoPhaseAllreduce


def expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    """Every rank receives the element-wise sum of all rank slices."""
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="09_allreduce_two_phase")
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
        default="0,1,2,3",
        help="comma-separated device ids (>= 2, dividing SIZE); P>=4 shows the saving vs mesh",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    nr = len(device_ids)
    program = build_two_phase_allreduce(nr)

    x = torch.randn((nr, 1, SIZE), dtype=torch.float32)
    y = torch.zeros((nr, 1, SIZE), dtype=torch.float32)

    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids,
            num_sub_workers=0,
        ),
    )
    if args.compile_only:
        print(f"compile_only done: {compiled.output_dir}")
        return 0

    compiled(x, y)

    expected = expected_allreduce(x)
    assert torch.allclose(y, expected, rtol=1e-5, atol=1e-5), (
        f"two-phase allreduce P={nr} mismatch: max diff = {(y - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
