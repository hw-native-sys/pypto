# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Scope placement: which of the runtime's four rings your tasks land in.

The runtime has four independent task-resource rings, and a task's ring is picked
by its scope nesting depth — ``ring_idx = min(scope_depth, 3)``. Each ring has its
own task-slot window, output heap and dependency-edge pool, reclaimed FIFO and
independently of the others.

By default the compiler places the scopes: the whole function body, plus each
``for`` body and each ``if`` branch. That means the ring assignment is a side
effect of control-flow shape rather than a decision. ``auto_scope=False`` hands
the placement back to you.

Kernels:
  auto_placement   — the default: compiler-inserted scopes
  manual_placement — auto_scope=False, scopes placed by phase

Concepts introduced:
  - ring_idx = min(scope_depth, 3), and why depth 3+ all collapse onto ring 3
  - @pl.jit(auto_scope=False) + with pl.scope()
  - RunConfig(enable_scope_stats=True) to see per-ring peaks before resizing
  - ring_task_window / ring_heap / ring_dep_pool as 4-element lists

Run:  python examples/advanced/08_scope_placement.py
      python examples/advanced/08_scope_placement.py --mode manual_placement
      python examples/advanced/08_scope_placement.py --scope-stats   # per-ring peaks
Docs: docs/en/user/performance/05-memory.md
"""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

N = 4
TR = 64
TC = 128
ROWS = N * TR


@pl.jit
def auto_placement(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Default placement — the compiler wraps the body and the loop body.

    That is two levels: the function body is depth 0 (ring 0) and everything
    inside the loop is depth 1 (ring 1). Correct, but not a choice anyone made.
    """
    for i in pl.range(N):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="phase1"):
            t = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.mul(t, 2.0), [i * TR, 0], out)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="phase2"):
        for i in pl.range(N):
            t = pl.load(out, [i * TR, 0], [TR, TC])
            pl.store(pl.add(t, 1.0), [i * TR, 0], out)
    return out


@pl.jit(auto_scope=False)
def manual_placement(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Placement by hand: the two phases get their own rings.

    With ``auto_scope=False`` the compiler inserts **nothing**, so every scope
    here is one you wrote — including the ones it used to add for free. Phase 1's
    tasks are reclaimed on ring 1 without waiting on phase 2's.

    This is a placement decision only. An AUTO ``pl.scope()`` keeps automatic
    dependency tracking on, so rebalancing rings does not change dependency
    semantics — ``pl.manual_scope()`` would, and that is a different subject.
    """
    with pl.scope():  # depth 0 -> ring 0
        with pl.scope():  # depth 1 -> ring 1
            for i in pl.range(N):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="phase1"):
                    t = pl.load(a, [i * TR, 0], [TR, TC])
                    pl.store(pl.mul(t, 2.0), [i * TR, 0], out)
        with pl.scope():  # depth 1 again -> also ring 1, reclaimed separately
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="phase2"):
                for i in pl.range(N):
                    t = pl.load(out, [i * TR, 0], [TR, TC])
                    pl.store(pl.add(t, 1.0), [i * TR, 0], out)
    return out


_MODES = {"auto_placement": auto_placement, "manual_placement": manual_placement}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scope-placement / ring example.")
    parser.add_argument("--mode", choices=sorted(_MODES), default="manual_placement")
    parser.add_argument("-p", "--platform", default="a2a3sim")
    parser.add_argument(
        "--scope-stats",
        action="store_true",
        help="record per-scope ring peaks into dfx_outputs/scope_stats/",
    )
    parser.add_argument(
        "--size-rings",
        action="store_true",
        help="also override the per-ring sizes (4 entries, rings 0..3)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.randn(ROWS, TC, dtype=torch.float32)
    out = torch.zeros(ROWS, TC, dtype=torch.float32)

    cfg_kwargs = {
        "platform": args.platform,
        "enable_scope_stats": args.scope_stats,
        "save_kernels": args.scope_stats,
    }
    if args.size_rings:
        # One entry per ring 0..3. task_window entries are powers of two >= 4;
        # heap entries are powers of two >= 1024 BYTES. Measure with
        # --scope-stats before changing these: splitting an overflowing scope
        # beats enlarging its ring.
        cfg_kwargs["ring_task_window"] = [8192, 16384, 16384, 16384]
        cfg_kwargs["ring_heap"] = [134217728, 268435456, 268435456, 268435456]

    _MODES[args.mode](a, out, config=RunConfig(**cfg_kwargs))

    torch.testing.assert_close(out, a * 2.0 + 1.0, rtol=1e-4, atol=1e-4)
    print("OK")
