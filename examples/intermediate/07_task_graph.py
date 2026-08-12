# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Ordering between tasks: the edge the runtime infers, and the one you declare.

The runtime does not run an orchestration function statement by statement. It
builds a dependency graph and runs whatever is ready, deriving edges from the
buffers each task touches and the direction each parameter declares. Writing one
dispatch after another therefore orders nothing by itself.

Kernels:
  inferred_edge — stage 2 reads what stage 1 wrote; the edge is derived
  declared_edge — the same order, stated explicitly with ``deps=``

Concepts introduced:
  - pl.at(...) as tid — binding a region's TaskId
  - deps=[tid] — declaring an edge the inference cannot reach
  - Automatic and explicit edges compose (the wait set is their union)

Run:  python examples/intermediate/07_task_graph.py
Next: examples/advanced/03_mixed_kernel.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

ROWS = 128
COLS = 128


@pl.jit
def inferred_edge(x: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
    """Nothing is declared — the edge falls out of the buffer directions.

    Stage 1 declares ``scratch`` as an output, so the runtime records it as the
    producer. Stage 2 reads the same buffer, so a read-after-write edge is
    derived. This is the case that needs nothing from you.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1"):
        scratch = pl.assemble(scratch, pl.add(x, x), [0, 0])  # writes scratch
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2"):
        out = pl.assemble(out, pl.add(scratch, scratch), [0, 0])  # reads scratch
    return scratch, out


@pl.jit
def declared_edge(x: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
    """The same order, this time stated.

    ``as first`` binds the producing region's TaskId; ``deps=[first]`` makes the
    consumer wait on it. Note there is no ``manual_scope`` here — an explicit
    edge is added on top of the automatic tracking, not instead of it, so the
    final wait set is the union of the two.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1") as first:
        scratch = pl.assemble(scratch, pl.add(x, x), [0, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2", deps=[first]) as second:
        out = pl.assemble(out, pl.add(scratch, scratch), [0, 0])
    return scratch, out


_MODES = {"inferred": inferred_edge, "declared": declared_edge}


if __name__ == "__main__":
    cfg = RunConfig(platform="a2a3sim")
    torch.manual_seed(0)
    x = torch.randn(ROWS, COLS, dtype=torch.float32)
    expected = (x + x) + (x + x)

    for name, kernel in _MODES.items():
        scratch = torch.zeros(ROWS, COLS, dtype=torch.float32)
        out = torch.zeros(ROWS, COLS, dtype=torch.float32)
        kernel(x, scratch, out, config=cfg)
        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
            f"{name}: max diff = {(out - expected).abs().max().item()}"
        )

    print("OK")
