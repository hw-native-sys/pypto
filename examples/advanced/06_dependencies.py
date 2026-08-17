# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Breaking false serialization: four ways to say "these tasks do not collide".

The runtime derives task dependencies from tensor arguments, and the producer
lookup is an *overlap* test over buffer addresses. Iterations that write disjoint
regions of one buffer look like a collision to it, so it chains them — correct,
but slower than it needs to be.

All five kernels below write the same disjoint row bands of one output and
produce the same answer. They differ only in how they tell the runtime so.

Kernels:
  serialized     — the baseline: a WAW chain the runtime cannot rule out
  narrow_claim   — pl.at(no_dep_args=[out]): one tensor, one task
  tensor_claim   — pl.create_tensor(manual_dep=True): one tensor, its lifetime
  region_claim   — with pl.manual_scope(): every task in a region
  sliced         — pass each task its own slice: disjoint BY CONSTRUCTION

Concepts introduced:
  - Why a sequential loop over one output buffer serializes
  - The three opt-out constructs, narrowest first
  - Slicing in orchestration as the one remedy needing no unprovable assertion
  - The cost of each: an unverifiable claim vs. longer dependency resolution

Run:  python examples/advanced/06_dependencies.py
      python examples/advanced/06_dependencies.py --mode sliced
      # see the edges themselves:
      python examples/advanced/06_dependencies.py --mode serialized --dep-gen
Docs: docs/en/user/performance/03-dependencies.md
"""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

N = 4
TILE_ROWS = 64
COLS = 128
ROWS = N * TILE_ROWS


@pl.jit
def serialized(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Baseline: N tasks writing disjoint bands of ``out``, chained anyway.

    Every iteration passes the whole ``out`` tensor, so the producer lookup sees
    an overlap it cannot disprove and adds a WAW edge on the previous iteration.
    """
    for i in pl.range(N):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="chained"):
            t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], out)
    return out


@pl.jit
def narrow_claim(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Narrowest opt-out: stop tracking ``out`` *for this task only*."""
    for i in pl.range(N):
        with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[out], name_hint="no_dep_arg"):
            t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], out)
    return out


@pl.jit
def tensor_claim(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Middle opt-out: a tensor that is never dependency-tracked at all.

    ``manual_dep=True`` applies for the tensor's whole lifetime, so it is the
    right shape when *every* task touching it writes its own region.

    **It removes every edge on that tensor, not just the ones you dislike.** The
    consumer below reads all four bands, and its RAW edges on the four writers
    are inferred from exactly the tracking that was just turned off. Without the
    explicit ``deps=`` it reads bands that have not been written yet — which
    shows up as a *partly* wrong output (here the last quarter stayed zero),
    the intermittent shape this whole page warns about.

    A ``pl.array`` of ``pl.TASK_ID`` is how a loop's worth of producers is
    collected — a Python list will not do, since the body is traced, not executed
    (``writers.append(tid)`` is rejected as an unsupported call).
    """
    scratch = pl.create_tensor([ROWS, COLS], pl.FP32, manual_dep=True)
    writers = pl.array.create(N, pl.TASK_ID)
    for i in pl.range(N):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="manual_dep") as tid:
            t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], scratch)
        writers[i] = tid
    with pl.at(level=pl.Level.CORE_GROUP, deps=[writers], name_hint="copy_out"):
        for i in pl.range(N):
            t = pl.load(scratch, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(t, [i * TILE_ROWS, 0], out)
    return out


@pl.jit
def region_claim(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Widest opt-out: inside a manual scope the runtime infers nothing.

    Fan-in computation is skipped outright for the region — creator retention and
    the producer lookup included — so every edge here would have to be one you
    wrote with ``deps=``. These iterations need none.
    """
    with pl.manual_scope():
        for i in pl.range(N):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="in_manual_scope"):
                t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
                pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], out)
    return out


@pl.jit
def sliced(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """No assertion at all: give each task its own slice of the output.

    The tasks no longer share a buffer, so the runtime *derives* that they are
    independent instead of being told. The cost is the extra orchestration-level
    tensors — more to register and walk, so dependency resolution takes longer
    per task.
    """
    for i in pl.range(N):
        band_in = pl.slice(a, [TILE_ROWS, COLS], [i * TILE_ROWS, 0])
        band_out = pl.slice(out, [TILE_ROWS, COLS], [i * TILE_ROWS, 0])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sliced"):
            t = pl.load(band_in, [0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(t, 2.0), [0, 0], band_out)
    return out


_MODES = {
    "serialized": serialized,
    "narrow_claim": narrow_claim,
    "tensor_claim": tensor_claim,
    "region_claim": region_claim,
    "sliced": sliced,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-dependency example.")
    parser.add_argument("--mode", choices=sorted(_MODES), default="sliced")
    parser.add_argument("-p", "--platform", default="a2a3sim")
    parser.add_argument(
        "--dep-gen",
        action="store_true",
        help="capture deps.json so the edges can be compared between modes",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    out = torch.zeros(ROWS, COLS, dtype=torch.float32)

    cfg = RunConfig(
        platform=args.platform,
        enable_dep_gen=args.dep_gen,
        save_kernels=args.dep_gen,
    )
    _MODES[args.mode](a, out, config=cfg)

    torch.testing.assert_close(out, a * 2.0, rtol=1e-4, atol=1e-4)
    print("OK")
