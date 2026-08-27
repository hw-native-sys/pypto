# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""All-reduce v1 (mesh): every rank reads every peer's slice and sums locally.

Concepts introduced:
  - all-reduce semantics: every rank contributes its slice; every rank ends
    with the element-wise reduction of ALL slices (here: Sum)
  - mesh topology: each rank ``remote_load``s every peer's slice and
    accumulates locally — O(P) reads, ``(P-1) * N`` bytes per rank, simple
    and round-heavy
  - the four phases every hand-rolled collective shares: stage-in -> barrier
    -> accumulate -> stage-out
  - the barrier before any remote read: the ``notify``/``wait`` handshake from
    step 04 guarantees no rank reads a peer that has not staged yet
  - a rank count that stays *dynamic*: the barrier signal is a per-rank row
    matrix, but its row count never becomes a compile-time constant. It is
    ``NR = pl.dynamic("NR")`` in the annotations and ``pld.world_size()`` in
    the host body, so one source serves any P — picked at run time with ``-d``,
    with no rank-count factory. This mirrors
    ``tests/st/distributed/collectives/test_l3_allreduce.py`` exactly.
  - why this step leaves the ``@pl.jit`` family of steps 01-07: ``signal`` is a
    window whose shape is the runtime expression ``[pld.world_size(), 1]``, and
    ``@pl.jit`` must infer a static shape/dtype for every parameter it passes
    to a dep — it rejects this one with "missing inferred tensor metadata for
    parameter 'signal'". The ``@pl.program`` class form has no such
    requirement, so the switch is forced by the *dynamic* signal shape, not by
    a compile-time one. Steps 09 and 10 go further and do need a genuine
    compile-time rank count, because their chunk size ``SIZE // nr`` is a
    **tile shape**; a signal row count is not.

This is the simplest of the three all-reduces (steps 08-10): one barrier, then
every rank reads every peer. Its O(P) traffic is why the two-phase and ring
variants exist. The golden is the element-wise sum over all rank slices, per
rank, compared with a tolerance — reduction order differs from torch.

Run + walkthrough: see docs/en/user/distributed/13-allreduce_mesh.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NR = pl.dynamic("NR")


@pl.program
class MeshAllreduce:
    @pl.function(type=pl.FunctionType.InCore)
    def reduce_step(
        self,
        x: pl.Tensor[[1, SIZE], pl.FP32],
        y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Chip kernel: four-phase mesh allreduce — stage, barrier, accumulate, stage-out."""
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # Phase 1 — stage this rank's slice into its window slot.
        local = pl.load(x, [0, 0], [1, SIZE])
        data = pl.store(local, [0, 0], data)

        # Phase 2 — barrier: notify every peer, wait on every peer slot.
        # Each rank owns a dedicated row (offsets=[my_rank, 0]);
        # AtomicAdd/Ge(1) means the wait only passes once every peer has
        # staged its slice.
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

        # Phase 3 — accumulate: start from our own slice, add every peer's slice.
        acc = pl.load(data, [0, 0], [1, SIZE])
        for peer in pl.range(nranks):
            if peer != my_rank:
                recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
                acc = pl.add(acc, recv)

        # Phase 4 — stage-out: the accumulated result is this rank's output.
        return pl.store(acc, [0, 0], y)

    @pl.function(type=pl.FunctionType.Orchestration)
    def per_rank(
        self,
        x: pl.Tensor[[1, SIZE], pl.FP32],
        y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Per-device orchestration: one incore call, on this device."""
        return self.reduce_step(x, y, data, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def mesh_allreduce(
        self,
        x: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        y: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        """Host orchestrator: one shared data window + one shared signal window, one dispatch per rank."""
        data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            self.per_rank(x[r], y[r], data, signal, device=r)
        return y


def expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    """Every rank receives the element-wise sum of all rank slices."""
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="08_allreduce_mesh")
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
        help="comma-separated device ids (any count >= 2); run P>=4 to see mesh's O(P) traffic",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    nr = len(device_ids)

    x = torch.randn((nr, 1, SIZE), dtype=torch.float32)
    y = torch.zeros((nr, 1, SIZE), dtype=torch.float32)

    compiled = ir.compile(
        MeshAllreduce,
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
        f"mesh allreduce P={nr} mismatch: max diff = {(y - expected).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
