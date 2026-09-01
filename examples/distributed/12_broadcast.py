# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""One-to-all: root stages its slice, every rank reads it — then the reveal.

Concepts introduced:
  - broadcast semantics: root rank's data reaches every rank (weights,
    configs, a global row)
  - the hand-rolled three-phase pattern: root stage-in -> notify/wait
    barrier -> every rank ``remote_load``s root's slice
  - the reveal: ``pld.tensor.broadcast(data, signal, root=...)`` is the
    barrier + read in one call — the stage stays yours (root must write its
    slice into the window before the call)
  - non-root inputs are ignored: only root's data may appear in the output

Two modes, one step:
  - ``--mode hand`` (default): root stages, barrier, every rank remote_loads
    root -> ``y[r] = x[0]`` for every r
  - ``--mode builtin``: root stages, ``pld.tensor.broadcast`` -> same golden

The cost card: root writes ``N`` bytes, every peer reads them — ``(P-1)*N``
total bytes in one round; the round trip, not the bytes, is what you feel.

Run + walkthrough: see docs/en/user/distributed/17-broadcast.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 64
ROOT_RANK = 0


def build_broadcast(nr: int, use_builtin: bool):
    """Build the broadcast program for a rank count ``nr`` and a mode.

    A factory because of ``use_builtin``, not because of any shape.
    ``host_orch`` branches on it to dispatch either ``chip_orch_hand`` or
    ``chip_orch_builtin``, so it has to be a Python constant when the body is
    traced. The ``[nr, 1]`` signal does *not* require one: this is step 08's
    shape, and a signal row count can stay dynamic (verified — a
    ``pl.dynamic`` rank count compiles and passes the golden in both modes).
    ``nr`` is folded in only to keep the window shapes readable.
    """

    @pl.program
    class Broadcast:
        @pl.function(type=pl.FunctionType.InCore)
        def hand_step(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            root: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Chip kernel: root stages, barrier, then every rank reads root."""
            ctx = pld.get_comm_ctx(data)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)

            # Phase 1 — stage-in: root only writes its slice into the window.
            if my_rank == root:
                local = pl.load(x, [0, 0], [1, SIZE])
                data = pl.store(local, [0, 0], data)

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

            # Phase 3 — broadcast: pull root's slice into local output.
            recv = pld.tile.remote_load(data, peer=root, offsets=[0, 0], shape=[1, SIZE])
            return pl.store(recv, [0, 0], y)

        @pl.function(type=pl.FunctionType.InCore)
        def builtin_step(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Chip kernel: the reveal — root stages, then one call broadcasts."""
            ctx = pld.get_comm_ctx(data)
            my_rank = pld.rank(ctx)

            if my_rank == ROOT_RANK:
                local = pl.load(x, [0, 0], [1, SIZE])
                data = pl.store(local, [0, 0], data)

            # Phases 2-3 in one call: barrier + every rank reads root's slice.
            data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)

            acc = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(acc, [0, 0], y)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_hand(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            root: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Per-device orchestration: the hand-rolled three-phase kernel."""
            return self.hand_step(x, y, data, signal, root)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_builtin(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Per-device orchestration: the reveal kernel."""
            return self.builtin_step(x, y, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            """Host orchestrator: one shared window buffer, one dispatch per rank."""
            data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                if use_builtin:
                    self.chip_orch_builtin(inputs[r], outputs[r], data, signal, device=r)
                else:
                    self.chip_orch_hand(inputs[r], outputs[r], data, signal, ROOT_RANK, device=r)
            return outputs

    return Broadcast


def expected_broadcast(inputs: torch.Tensor) -> torch.Tensor:
    """Golden: root row replicated on every rank."""
    root_row = inputs[ROOT_RANK, 0]
    return torch.stack([root_row] * inputs.shape[0]).unsqueeze(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="12_broadcast")
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
        help="comma-separated device ids -- any count >= 2; root is rank 0",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hand",
        choices=["hand", "builtin"],
        help="hand-rolled three-phase broadcast or the pld.tensor.broadcast reveal",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    nr = len(device_ids)
    use_builtin = args.mode == "builtin"
    program = build_broadcast(nr, use_builtin)

    # Distinct per-rank inputs; rank 0 is root and the only one that may leak.
    rows = [torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE)]
    for r in range(1, nr):
        rows.append(torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE))
    inputs = torch.stack(rows)
    outputs = torch.zeros((nr, 1, SIZE), dtype=torch.float32)

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

    expected = expected_broadcast(inputs)
    assert torch.allclose(outputs, expected, rtol=1e-5, atol=1e-5), (
        f"broadcast {args.mode} P={nr} mismatch: max diff = {(outputs - expected).abs().max().item()}"
    )
    # Non-root inputs must not leak into the output.
    for r in range(1, nr):
        assert not torch.allclose(outputs[r], inputs[r]), f"non-root rank {r} input leaked into output"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
