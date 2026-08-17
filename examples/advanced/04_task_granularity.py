# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Task granularity: the same work as many small tasks, then as few large ones.

Every ``pl.at`` block is a task the runtime must dispatch, and a dispatch is not
free. These four kernels compute the *same* result on the same data and differ
only in how the work is cut into tasks — so the task count is the variable and
the answer is the control.

Kernels:
  many_small_tasks  — four tasks, one per 64x128 tile (a swimlane staircase)
  larger_tiles      — the same rows in 128x128 tiles: two tasks           (a)
  loop_inside       — one task, the chunk loop moved inside it            (b)
  merged_chain      — add-then-exp as ONE task instead of two             (c)

Concepts introduced:
  - Larger tiling as a granularity knob, and what it costs on-chip
  - Moving a chunk loop inside pl.at (pl.range) to pay one dispatch
  - Merging a producer/consumer chain so the intermediate stays on chip

Run:  python examples/advanced/04_task_granularity.py
      python examples/advanced/04_task_granularity.py --mode loop_inside
Docs: docs/en/user/performance/01-task-granularity.md
Next: examples/advanced/07_double_buffer.py — where a loop-inside kernel wins the
      parallelism back
"""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

ROWS = 256
COLS = 128
SMALL = 64  # tile rows for the many-small-tasks form
LARGE = 128  # tile rows after (a)


@pl.jit
def many_small_tasks(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Four tasks, one per ``[64, 128]`` tile — the shape to avoid.

    ``pl.unroll`` is unrolled at compile time, so each iteration emits its own
    ``pl.at`` block and therefore its own dispatch.
    """
    for i in pl.unroll(ROWS // SMALL):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="small"):
            ta = pl.load(a, [i * SMALL, 0], [SMALL, COLS])
            tb = pl.load(b, [i * SMALL, 0], [SMALL, COLS])
            pl.store(pl.add(ta, tb), [i * SMALL, 0], c)
    return c


@pl.jit
def larger_tiles(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """(a) Same structure, twice the rows per tile — two tasks instead of four.

    Only the row count grows here (``COLS`` is already the full width), so this
    is a 2x reduction. Scaling both axes of a tile scales the task count by the
    same factor in each — and the on-chip footprint with it, which is why the
    cost below is quadratic in a 2D tile rather than linear.
    """
    for i in pl.unroll(ROWS // LARGE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="large"):
            ta = pl.load(a, [i * LARGE, 0], [LARGE, COLS])
            tb = pl.load(b, [i * LARGE, 0], [LARGE, COLS])
            pl.store(pl.add(ta, tb), [i * LARGE, 0], c)
    return c


@pl.jit
def loop_inside(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """(b) One task for all chunks — the loop moved inside ``pl.at``.

    ``pl.range`` is a device-side loop, so the whole thing is a single dispatch.
    The tiles keep their original size; only the offset moves per iteration.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="one_task"):
        for i in pl.range(ROWS // SMALL):
            ta = pl.load(a, [i * SMALL, 0], [SMALL, COLS])
            tb = pl.load(b, [i * SMALL, 0], [SMALL, COLS])
            pl.store(pl.add(ta, tb), [i * SMALL, 0], c)
    return c


@pl.jit
def two_tasks_via_gm(a: pl.Tensor, b: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
    """The producer/consumer chain (c) replaces: ``s`` round-trips through GM."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="producer"):
        s = pl.add(pl.load(a, [0, 0], [LARGE, COLS]), pl.load(b, [0, 0], [LARGE, COLS]))
        pl.store(s, [0, 0], scratch)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="consumer"):
        pl.store(pl.exp(pl.load(scratch, [0, 0], [LARGE, COLS])), [0, 0], out)
    return scratch, out


@pl.jit
def merged_chain(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    """(c) The same chain as ONE task — ``s`` never leaves the chip.

    One dispatch instead of two, and no GM traffic for the intermediate. The cost
    is that every intermediate is live at once inside the merged task.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="merged"):
        s = pl.add(pl.load(a, [0, 0], [LARGE, COLS]), pl.load(b, [0, 0], [LARGE, COLS]))
        pl.store(pl.exp(s), [0, 0], out)
    return out


def _run_elementwise(fn, platform):
    torch.manual_seed(0)
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    b = torch.randn(ROWS, COLS, dtype=torch.float32)
    c = torch.zeros(ROWS, COLS, dtype=torch.float32)
    fn(a, b, c, config=RunConfig(platform=platform))
    torch.testing.assert_close(c, a + b, rtol=1e-4, atol=1e-4)


def _run_chain(mode, platform):
    torch.manual_seed(0)
    a = torch.randn(LARGE, COLS, dtype=torch.float32)
    b = torch.randn(LARGE, COLS, dtype=torch.float32)
    out = torch.zeros(LARGE, COLS, dtype=torch.float32)
    cfg = RunConfig(platform=platform)
    if mode == "merged_chain":
        merged_chain(a, b, out, config=cfg)
    else:
        scratch = torch.zeros(LARGE, COLS, dtype=torch.float32)
        two_tasks_via_gm(a, b, scratch, out, config=cfg)
    torch.testing.assert_close(out, torch.exp(a + b), rtol=1e-4, atol=1e-4)


_ELEMENTWISE = {
    "many_small_tasks": many_small_tasks,
    "larger_tiles": larger_tiles,
    "loop_inside": loop_inside,
}
_CHAIN = ("two_tasks_via_gm", "merged_chain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-granularity example.")
    parser.add_argument("--mode", choices=[*sorted(_ELEMENTWISE), *_CHAIN], default="loop_inside")
    parser.add_argument("-p", "--platform", default="a2a3sim")
    args = parser.parse_args()

    if args.mode in _ELEMENTWISE:
        _run_elementwise(_ELEMENTWISE[args.mode], args.platform)
    else:
        _run_chain(args.mode, args.platform)
    print("OK")
