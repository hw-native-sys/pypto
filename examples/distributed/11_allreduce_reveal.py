# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""The reveal: ``pld.tensor.allreduce`` in one call, mesh or ring.

Concepts introduced:
  - the builtin: ``pld.tensor.allreduce(data, signal, op=..., mode=...)``
    replaces the hand-rolled steps 08-10 with one call — the same golden, the
    same signal conventions, none of the schedule
  - ``mode="mesh"`` (default) / ``mode="ring"``: the same source picks the
    algorithm; the IR diff (see the walkthrough) shows what each mode lowers
    to — the hand-rolled mesh/ring pattern you wrote in steps 08/10, or a
    better one
  - what the builtin accepts: the full ``ReduceOp`` family (``Sum``/``Max``/
    ``Min``/``Prod``) and ``FP16``/``FP32`` in both modes — ring keeps the
    ``[2*(P-1), P]`` signal (the row-per-round signal step 10 taught), mesh
    takes ``[P, 1]``
  - stage-in / stage-out stay yours: the builtin only owns the barrier +
    reduce + store-back; you still move data in and out of the window

Same golden as steps 08-10: every rank receives the element-wise sum of all
rank slices, compared with a tolerance. Run ``--mode mesh`` and ``--mode ring``
at P>=4 and compare the lowered IR in the walkthrough.

Run + walkthrough: see docs/en/user/distributed/16-allreduce_reveal.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def build_reveal_allreduce(nr: int, mode: str):
    """Build the builtin allreduce program for a rank count ``nr`` and ``mode``.

    A factory over ``(nr, mode)``, and for a different reason than steps 09/10:
    the builtin owns the chunking, so no **tile shape** here depends on the
    rank count — ``nr`` need not be compile-time at all (the ring layout, the
    more constrained of the two, compiles and passes its golden with a
    ``pl.dynamic`` rank count). What must be fixed when the kernel is traced is
    ``mode``: it picks both the lowering ``pld.tensor.allreduce`` emits and the
    signal layout the kernel is annotated with, and those are two different
    shapes, not two extents of one — mesh ``[nr, 1]``, ring ``[2*(nr-1), nr]``
    (row per round). Folding ``nr`` in beside it lets one source spell both
    layouts and build either variant (pick it with ``--mode``).
    """
    total_rounds = 2 * (nr - 1)
    if mode == "ring":
        sig_rows, sig_cols = total_rounds, nr
    elif mode == "mesh":
        sig_rows, sig_cols = nr, 1
    else:
        raise ValueError(f"unsupported mode: {mode}")

    @pl.program
    class AllreduceReveal:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Chip kernel: stage-in, one builtin call, stage-out."""

            # Phase 1 — stage this rank's slice into its window slot.
            local = pl.load(x, [0, 0], [1, SIZE])
            data = pl.store(local, [0, 0], data)

            # Phase 2 — the builtin: barrier + reduce + store-back, in one
            # call. mode is folded in by the factory (mesh default / ring).
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode=mode)

            # Phase 3 — stage-out: the reduced slice back to the local output.
            recv = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(recv, [0, 0], y)

        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Per-device orchestration: one incore call, on this device."""
            return self.reduce_step(x, y, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def allreduce_reveal(
            self,
            x: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            y: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
            """Host orchestrator: shared data + signal windows, one dispatch per rank."""
            data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
            signal_buf = pld.alloc_window_buffer([sig_rows, sig_cols], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [sig_rows, sig_cols], dtype=pl.INT32)
                self.per_rank(x[r], y[r], data, signal, device=r)
            return y

    return AllreduceReveal


def expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    """Every rank receives the element-wise sum of all rank slices."""
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="11_allreduce_reveal")
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
        help="comma-separated device ids (any count >= 2)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mesh",
        choices=["mesh", "ring"],
        help="builtin mode: mesh (default) or ring",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    nr = len(device_ids)
    program = build_reveal_allreduce(nr, args.mode)

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
        f"allreduce reveal ({args.mode}) P={nr} mismatch: max diff = {(y - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
