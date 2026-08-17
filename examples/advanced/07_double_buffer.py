# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Double buffering: overlapping the transfer of the next tile with this tile's compute.

A loop that loads, computes and stores with a single buffer stalls on every load
— the core has nothing to work on while MTE2 fetches. Two buffers let iteration
``i+1``'s load run underneath iteration ``i``'s compute. PyPTO offers a
compiler-managed form and a hand-managed one.

Kernels:
  single_buffer  — the baseline: load, compute, store, repeat
  pipelined      — pl.pipeline(stage=2): the compiler replicates the body
  explicit_slots — pl.MemRef("ub", slots=2)[i % 2]: the buffer rotation by hand

Concepts introduced:
  - pl.pipeline(n, stage=k) and what "stage" costs in on-chip buffers
  - pl.MemRef(name, slots=N) with an index expression per iteration
  - The difference between the two: pl.pipeline is a *schedule*, slots are
    *placement* — see the note on explicit_slots
  - Why the explicit form keeps ONE slot live per iteration (see the PTOAS note)

Run:  python examples/advanced/07_double_buffer.py
      python examples/advanced/07_double_buffer.py --mode explicit_slots
Docs: docs/en/user/performance/04-incore.md
Next: examples/advanced/04_task_granularity.py — the loop this pipelines usually
      comes from moving a dispatch loop inside the kernel
"""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

NT = 8  # tiles in the loop
TR = 64  # tile rows
TC = 128  # tile cols
ROWS = NT * TR


@pl.jit
def single_buffer(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Baseline: one buffer, so each load waits for the previous compute."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="single"):
        for i in pl.range(NT):
            tile = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


@pl.jit
def pipelined(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Compiler-managed double buffering.

    ``pl.pipeline`` replicates the body ``stage`` times per outer iteration, so
    iteration ``i+1``'s load overlaps iteration ``i``'s compute. The outer loop
    then advances in strides of ``stage``, with a tail dispatch for a trip count
    that is not divisible by it.

    **Cost:** ``stage`` copies of every buffer the body stages, live at once. If
    they do not fit, the compiler says so in ``report/perf_hints.log`` (PH-MR-001,
    "requested depth 4 ... but only 2 of 4 buffers fit") rather than silently
    under-delivering. Depths of 2-4 are the usual range.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="pipelined"):
        for i in pl.pipeline(NT, stage=2):
            tile = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


@pl.jit
def explicit_slots(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Two slots of one allocation, picked by an index expression.

    ``pl.MemRef("ub", slots=2)`` reserves two equally-sized slots in a single
    allocation; ``[i % 2]`` picks one per iteration. Use the inline spelling
    rather than binding the declaration to a Python variable — ``@pl.jit``
    re-parses generated source in a fresh module namespace where such a variable
    does not exist.

    **This is placement, not a schedule — the two are not interchangeable.**
    ``pl.pipeline`` restructures the loop so a load and a compute from different
    iterations are in flight together. Alternating slots only removes the
    same-buffer hazard that would stop them overlapping; the loop is still an
    ordinary sequential ``pl.range``, and whether the MTE2 and vector pipes
    actually overlap is a question for the L0 trace, not something this spelling
    guarantees. It is the storage form a hand-rolled rotation needs — and the one
    ``LowerPipelineToSlots`` rotates a ``pl.pipeline`` body through under
    ``memory_planner=PTOAS``.

    Note the shape this is written in: **one slot live per iteration**. Two
    co-live slots work under the default ``PYPTO`` planner but are rejected at
    codegen under ``memory_planner=PTOAS``, because ptoas guards only the first
    ``multi_tile_get`` of an iteration.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="slots"):
        for i in pl.range(NT):
            tile: pl.Tile[[TR, TC], pl.FP32, pl.MemRef("ub", slots=2)[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * TR, 0], [TR, TC], target_memory=pl.Mem.Vec
            )
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


_MODES = {
    "single_buffer": single_buffer,
    "pipelined": pipelined,
    "explicit_slots": explicit_slots,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double-buffering example.")
    parser.add_argument("--mode", choices=sorted(_MODES), default="pipelined")
    parser.add_argument("-p", "--platform", default="a2a3sim")
    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.randn(ROWS, TC, dtype=torch.float32)
    out = torch.zeros(ROWS, TC, dtype=torch.float32)

    _MODES[args.mode](a, out, config=RunConfig(platform=args.platform))

    torch.testing.assert_close(out, torch.exp(a), rtol=1e-4, atol=1e-4)
    print("OK")
