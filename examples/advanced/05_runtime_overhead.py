# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Cutting runtime overhead: one dispatch for many blocks, pre-staging, in-kernel sync.

The previous example made each task bigger. These three knobs leave the work
alone and make the *dispatching* of it cheaper.

Kernels:
  per_block_tasks   — one pl.at per block: N dispatches
  spmd_blocks       — the same N blocks as ONE SPMD dispatch
  early_resolve     — a producer/consumer chain with allow_early_resolve
  soft_barrier      — two phases in one kernel, joined by a soft syncall

Concepts introduced:
  - pl.spmd(n) — one dispatch that fans out to n blocks, pl.block_idx() to index
  - allow_early_resolve — let the scheduler pre-stage consumers of a producer
  - pl.system.syncall(mode="soft") — barrier inside the kernel, no AICPU round trip
  - Why the soft form (not the hard FFTS one) is what a partial launch must use

Run:  python examples/advanced/05_runtime_overhead.py
      python examples/advanced/05_runtime_overhead.py --mode soft_barrier

      The soft_barrier mode needs the pto-isa the repo pins in runtime/pto_isa.pin:
      the cacheinvalid path emits dcci(..., cache_line_t::SINGLE_CACHE_LINE), and an
      older pto-isa has no cache_line_t, so the kernel C++ compile fails rather
      than the run.
Docs: docs/en/user/performance/02-runtime-overhead.md
Next: examples/advanced/03_mixed_kernel.py — the fourth knob on that page
"""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

BLOCKS = 4
TILE_ROWS = 64
COLS = 128
ROWS = BLOCKS * TILE_ROWS


@pl.jit
def per_block_tasks(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """One task per block — ``BLOCKS`` separate dispatches for independent work."""
    for i in pl.unroll(BLOCKS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="block"):
            ta = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            tb = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.add(ta, tb), [i * TILE_ROWS, 0], c)
    return c


@pl.jit
def spmd_blocks(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """The same work as ONE dispatch fanning out to ``BLOCKS`` blocks.

    The loop variable of ``pl.spmd`` is the block index, so each block computes
    its own offset. The task count in ``deps.json`` drops from ``BLOCKS`` to one.
    """
    for i in pl.spmd(BLOCKS):
        ta = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        tb = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        pl.store(pl.add(ta, tb), [i * TILE_ROWS, 0], c)
    return c


@pl.jit
def early_resolve(a: pl.Tensor, b: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
    """A two-link chain whose producer is flagged for pre-staging.

    ``allow_early_resolve=True`` lets the scheduler stage the consumer onto a core
    before the producer finishes, releasing it with a doorbell the moment it does.
    It is a pure scheduling hint: the results are identical either way.

    Note the rule that makes it worth doing — a consumer pre-stages only when
    **all** of its producers are flagged, which is why real models carry the flag
    along a whole chain rather than on one task.
    """
    with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True, name_hint="producer"):
        s = pl.add(pl.load(a, [0, 0], [TILE_ROWS, COLS]), pl.load(b, [0, 0], [TILE_ROWS, COLS]))
        pl.store(s, [0, 0], scratch)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="consumer"):
        pl.store(pl.exp(pl.load(scratch, [0, 0], [TILE_ROWS, COLS])), [0, 0], out)
    return scratch, out


@pl.jit.incore
def _phased_add(a: pl.Tensor, ws: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Two phases separated by a barrier *inside* the kernel.

    Written as two tasks with a dependency, the join would go out to the AICPU
    scheduler and back. ``pl.system.syncall`` keeps it on the cores.

    ``mode="soft"`` polls a counter in a shared GM workspace, so it works at the
    partial occupancy this launch has (``BLOCKS`` blocks, not every physical
    core). The default ``mode="hard"`` is an FFTS barrier that requires *full*
    occupancy and deadlocks without it — PyPTO rejects that combination at
    compile time rather than letting it reach the device.

    ``ws`` must be a zero-initialized INT32 GM tensor with at least
    ``used_cores * 8`` elements, and it has to be a **kernel parameter** so every
    block polls the same buffer.

    See the module docstring for the pto-isa this mode currently needs.
    """
    offset = pl.tile.get_block_idx() * TILE_ROWS
    t = pl.load(a, [offset, 0], [TILE_ROWS, COLS])
    out = pl.store(pl.mul(t, 2.0), [offset, 0], out)

    pl.system.syncall(mode="soft", core_type="aiv_only", gm_workspace=ws, used_cores=BLOCKS)

    # Past the barrier every block's phase-1 store is visible.
    t2 = pl.load(out, [offset, 0], [TILE_ROWS, COLS])
    out = pl.store(pl.add(t2, 1.0), [offset, 0], out)
    return out


@pl.jit
def soft_barrier(a: pl.Tensor, ws: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Dispatch the barrier kernel across ``BLOCKS`` blocks."""
    with pl.spmd(BLOCKS):
        out = _phased_add(a, ws, out)
    return out


def _run(mode, platform):
    torch.manual_seed(0)
    cfg = RunConfig(platform=platform)
    a = torch.randn(ROWS, COLS, dtype=torch.float32)

    if mode in ("per_block_tasks", "spmd_blocks"):
        b = torch.randn(ROWS, COLS, dtype=torch.float32)
        c = torch.zeros(ROWS, COLS, dtype=torch.float32)
        (per_block_tasks if mode == "per_block_tasks" else spmd_blocks)(a, b, c, config=cfg)
        torch.testing.assert_close(c, a + b, rtol=1e-4, atol=1e-4)
    elif mode == "early_resolve":
        b = torch.randn(ROWS, COLS, dtype=torch.float32)
        scratch = torch.zeros(TILE_ROWS, COLS, dtype=torch.float32)
        out = torch.zeros(TILE_ROWS, COLS, dtype=torch.float32)
        early_resolve(a[:TILE_ROWS], b[:TILE_ROWS], scratch, out, config=cfg)
        torch.testing.assert_close(out, torch.exp(a[:TILE_ROWS] + b[:TILE_ROWS]), rtol=1e-4, atol=1e-4)
    else:
        ws = torch.zeros(BLOCKS * 8, dtype=torch.int32)
        out = torch.zeros(ROWS, COLS, dtype=torch.float32)
        soft_barrier(a, ws, out, config=cfg)
        torch.testing.assert_close(out, a * 2.0 + 1.0, rtol=1e-4, atol=1e-4)


_MODES = ("per_block_tasks", "spmd_blocks", "early_resolve", "soft_barrier")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runtime-overhead example.")
    parser.add_argument("--mode", choices=_MODES, default="spmd_blocks")
    parser.add_argument("-p", "--platform", default="a2a3sim")
    args = parser.parse_args()

    _run(args.mode, args.platform)
    print("OK")
