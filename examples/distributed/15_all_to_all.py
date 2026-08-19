# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Personalized exchange: every rank sends a DISTINCT slice to every peer — then the reveal.

Concepts introduced:
  - all-to-all semantics: every rank sends a *different* ``N/P`` slice to
    every peer and receives a different slice from every peer — the most
    general of the point-to-point patterns
  - the hand-rolled pattern: push each destination chunk with ``put`` ->
    notify/wait barrier -> read back the chunk each peer sent you
  - the reveal: ``pld.tensor.all_to_all(stage, data, signal)`` is the
    push + barrier + read in one call
  - this is the pattern behind distributed MoE dispatch/combine and the
    AllGather-GEMM pipeline (see the walkthrough's pypto-lib pointers)

Both modes share one ``[nr, SIZE]`` window: rank r writes its chunk-for-dest
at row ``r`` of dest's window, then reads row ``src`` of its own window.

Two modes, one step:
  - ``--mode hand`` (default): put each dest chunk, barrier, read back ->
    ``y[r][src] = the chunk src intended for r``
  - ``--mode builtin``: ``pld.tensor.all_to_all`` -> same golden

The cost card: every rank sends a *different* ``N/P`` slice to each peer —
``(P-1)/P * N`` bytes received, and no two ranks want the same bytes.

Run + walkthrough: see docs/en/user/distributed/20-all_to_all.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def build_all_to_all(nr: int, use_builtin: bool):
    """Build the all-to-all program for a compile-time rank count ``nr``.

    A factory rather than a module-level program because the signal and data
    window shapes depend on the world size (``[nr, 1]`` / ``[nr, SIZE]``) and
    window shapes must be statically known: ``nr`` as a closure constant
    becomes a compile-time shape, so the same source serves any world size.
    """

    @pl.program
    class AllToAll:
        @pl.function(type=pl.FunctionType.InCore)
        def hand_step(
            self,
            x: pl.Tensor[[nr, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            """Chip kernel: put each dest chunk, barrier, then read back."""
            ctx = pld.get_comm_ctx(data)
            my_rank = pld.rank(ctx)

            # Phase 1 — push: write chunk-for-dest into dest's window at our
            # row (row my_rank of peer dest's window). The self-rank case is
            # handled uniformly by put's identity mapping.
            for dest in pl.range(nr):
                pld.tensor.put(data, dest, x, [my_rank, 0], [dest, 0], [1, SIZE])

            # Phase 2 — barrier: notify every peer, wait on every peer slot.
            for peer in pl.range(nr):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
            for src in pl.range(nr):
                if src != my_rank:
                    pld.system.wait(
                        signal,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # Phase 3 — read-back: row src of our window holds src's chunk for us.
            for src in pl.range(nr):
                chunk = pl.load(data, [src, 0], [1, SIZE])
                y = pl.store(chunk, [src, 0], y)
            return y

        @pl.function(type=pl.FunctionType.InCore)
        def builtin_step(
            self,
            x: pl.Tensor[[nr, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            """Chip kernel: the reveal — one call exchanges every chunk."""
            # Push + barrier + read in one call: the window becomes the
            # personalized result (row src = the chunk src sent us).
            result = pld.tensor.all_to_all(x, data, signal)

            for src in pl.range(nr):
                chunk = pl.load(result, [src, 0], [1, SIZE])
                y = pl.store(chunk, [src, 0], y)
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_hand(
            self,
            x: pl.Tensor[[nr, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            """Per-device orchestration: the hand-rolled exchange kernel."""
            return self.hand_step(x, y, data, signal)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_builtin(
            self,
            x: pl.Tensor[[nr, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            """Per-device orchestration: the reveal kernel."""
            return self.builtin_step(x, y, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, nr, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, nr, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, nr, SIZE], pl.FP32]:
            """Host orchestrator: one shared [nr, SIZE] data + [nr, 1] signal window."""
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                if use_builtin:
                    self.chip_orch_builtin(inputs[r], outputs[r], data, signal, device=r)
                else:
                    self.chip_orch_hand(inputs[r], outputs[r], data, signal, device=r)
            return outputs

    return AllToAll


def expected_all_to_all(inputs: torch.Tensor) -> torch.Tensor:
    """Golden: output[rank, src, j] = the chunk rank src intended for rank."""
    nranks = inputs.shape[0]
    src_idx = torch.arange(nranks, dtype=torch.float32).view(1, -1, 1)
    rank_idx = torch.arange(nranks, dtype=torch.float32).view(-1, 1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return src_idx * 1000 + rank_idx * 100 + j


def main() -> int:
    parser = argparse.ArgumentParser(description="15_all_to_all")
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
        help="comma-separated device ids -- any count >= 2",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hand",
        choices=["hand", "builtin"],
        help="hand-rolled put/barrier/read or the pld.tensor.all_to_all reveal",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    nr = len(device_ids)
    use_builtin = args.mode == "builtin"
    program = build_all_to_all(nr, use_builtin)

    # Each rank r fills input[r, d, j] = r*1000 + d*100 + j (chunk for dest d).
    r = torch.arange(nr, dtype=torch.float32).view(-1, 1, 1)
    d = torch.arange(nr, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    inputs = r * 1000 + d * 100 + j
    outputs = torch.zeros((nr, nr, SIZE), dtype=torch.float32)

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

    compiled(inputs, outputs)

    expected = expected_all_to_all(inputs)
    assert torch.allclose(outputs, expected, rtol=1e-5, atol=1e-5), (
        f"all_to_all {args.mode} P={nr} mismatch: max diff = {(outputs - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
