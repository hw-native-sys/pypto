# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""All-to-all slices: every rank publishes, every rank reads all — then the reveal.

Concepts introduced:
  - allgather semantics: every rank contributes one slice, every rank ends
    with the rank-ordered concatenation of ALL slices
  - the hand-rolled pattern: stage your slice -> notify/wait barrier ->
    ``remote_load`` every peer's slice into your output
  - the reveal: ``pld.tensor.allgather(stage, data, signal)`` is the stage +
    barrier + gather in one call (the push-based form)
  - this is the **all-gather half of two-phase all-reduce** (step 09): once
    every rank has every slice, reduction is a local op

Two modes, one step:
  - ``--mode hand`` (default): stage, barrier, remote_load every peer ->
    ``y[r] = concat(x[0], x[1], ..., x[P-1])`` for every r
  - ``--mode builtin``: ``pld.tensor.allgather`` -> same golden

The cost card: every rank sends ``N/P`` bytes to every peer, so each rank
receives ``(P-1)/P * N`` bytes in one round — the same bytes as the two-phase
all-reduce's gather phase.

Run + walkthrough: see docs/en/user/distributed/18-allgather.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 64


def build_allgather(nr: int, use_builtin: bool):
    """Build the allgather program for a rank count ``nr`` and a mode.

    A factory because of ``use_builtin`` — ``host_orch`` branches on it to
    pick which per-device orchestrator to dispatch, so it must be a Python
    constant at trace time. The ``nr``-shaped windows are *not* the reason:
    none of them is a tile shape, and all of them work with a dynamic rank
    count (verified in both modes). ``nr`` is folded in for readability.
    """

    @pl.program
    class AllGather:
        @pl.function(type=pl.FunctionType.InCore)
        def hand_step(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            """Chip kernel: stage your row, barrier, then pull every peer's row."""
            ctx = pld.get_comm_ctx(data)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)

            # Phase 1 — stage this rank's slice into its own row.
            local = pl.load(x, [0, 0], [1, SIZE])
            data = pl.store(local, [my_rank, 0], data)

            # Phase 2 — barrier: notify every peer, wait on every peer slot.
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(nranks):
                if src != my_rank:
                    pld.system.wait(
                        signal,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # Phase 3 — gather: pull every peer's row into the output.
            for peer in pl.range(nranks):
                recv = pld.tile.remote_load(data, peer=peer, offsets=[peer, 0], shape=[1, SIZE])
                y = pl.store(recv, [0, peer * SIZE], y)
            return y

        @pl.function(type=pl.FunctionType.InCore)
        def builtin_step(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            """Chip kernel: the reveal — one call gathers every rank's slice."""
            nranks = pld.nranks(pld.get_comm_ctx(data))
            # Stage + barrier + gather in one call: the window becomes the
            # rank-ordered [nr, SIZE] result (row src = rank src's slice).
            data = pld.tensor.allgather(x, data, signal)

            for src in pl.range(nranks):
                chunk = pl.load(data, [src, 0], [1, SIZE])
                y = pl.store(chunk, [0, src * SIZE], y)
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_hand(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            """Per-device orchestration: the hand-rolled gather kernel."""
            return self.hand_step(x, y, data, signal)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_builtin(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            """Per-device orchestration: the reveal kernel."""
            return self.builtin_step(x, y, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
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

    return AllGather


def expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    """Golden: rank-ordered concatenation; identical on every rank."""
    gathered = torch.cat([inputs[r, 0] for r in range(inputs.shape[0])])
    return torch.stack([gathered] * inputs.shape[0]).unsqueeze(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="13_allgather")
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
        help="hand-rolled stage/barrier/gather or the pld.tensor.allgather reveal",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    nr = len(device_ids)
    use_builtin = args.mode == "builtin"
    program = build_allgather(nr, use_builtin)

    # Distinct per-rank slices so the concatenation golden is non-trivial.
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE) for r in range(nr)
    ]
    inputs = torch.stack(rows)
    outputs = torch.zeros((nr, 1, nr * SIZE), dtype=torch.float32)

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

    expected = expected_allgather(inputs)
    assert torch.allclose(outputs, expected, rtol=1e-5, atol=1e-5), (
        f"allgather {args.mode} P={nr} mismatch: max diff = {(outputs - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
