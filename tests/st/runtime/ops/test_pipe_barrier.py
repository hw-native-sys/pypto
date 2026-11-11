# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device load/compute/store ordered by pipeline barriers instead of flags.

A barrier drains one pipeline, so the same load -> vector -> store handoffs that
``test_set_wait_flag.py`` authors as ``set_flag`` / ``wait_flag`` pairs can be
authored as single barriers, without spending an intra-core event slot. The MTE
pipes carry the two halves of that chain: ``MTE2`` drains the GM -> UB load and
``MTE3`` the UB -> GM store.

Each test checks the numerical result, which exercises the handoffs, and then the
saved PTO, which checks that the authored barriers reached the backend with the
right pipe and in the authored order.
"""

from dataclasses import replace

import pypto.language as pl
import pytest
import torch

ROWS = 1
COLS = 256


@pl.jit
def mte_barrier_double(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Order each handoff with the barrier for the pipe that produced the data."""
    for _ in pl.spmd(1):
        loaded: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [ROWS, COLS])
        pl.system.barrier(pipe=pl.PipeType.MTE2)
        doubled: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec] = pl.add(loaded, loaded)
        pl.system.barrier(pipe=pl.PipeType.V)
        out = pl.store(doubled, [0, 0], out)
        pl.system.barrier(pipe=pl.PipeType.MTE3)
    return out


@pl.jit
def all_barrier_double(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The coarse form: one ALL barrier drains every pipe on the core."""
    for _ in pl.spmd(1):
        loaded: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [ROWS, COLS])
        pl.system.barrier(pipe=pl.PipeType.ALL)
        doubled: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec] = pl.add(loaded, loaded)
        pl.system.barrier(pipe=pl.PipeType.ALL)
        out = pl.store(doubled, [0, 0], out)
    return out


def _saved_pto(tmp_path) -> str:
    pto_sources = [path.read_text() for path in tmp_path.rglob("*.pto")]
    assert pto_sources, "compilation did not save the PTO kernel"
    return "\n".join(pto_sources)


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
def test_mte_barriers_order_load_compute_store_on_device(test_config, tmp_path):
    """Per-pipe barriers preserve data across the load and store handoffs."""
    config = replace(test_config, save_kernels=True, save_kernels_dir=str(tmp_path))
    x = torch.randn(ROWS, COLS, dtype=torch.float32)
    out = torch.zeros_like(x)

    mte_barrier_double(x, out, config=config)

    assert torch.equal(out, x + x)
    pto = _saved_pto(tmp_path)
    # Authored order: drain the load, then the vector work, then the store.
    barriers = (
        "pto.barrier <PIPE_MTE2>",
        "pto.barrier <PIPE_V>",
        "pto.barrier <PIPE_MTE3>",
    )
    positions = [pto.find(barrier) for barrier in barriers]
    assert all(position >= 0 for position in positions), (
        f"authored barrier missing from PTO output; found positions {positions} for {barriers}"
    )
    assert positions == sorted(positions), "barriers crossed their authored order"


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
def test_all_barrier_orders_load_compute_store_on_device(test_config, tmp_path):
    """The ALL form covers the same chain with one barrier per handoff."""
    config = replace(test_config, save_kernels=True, save_kernels_dir=str(tmp_path))
    x = torch.randn(ROWS, COLS, dtype=torch.float32)
    out = torch.zeros_like(x)

    all_barrier_double(x, out, config=config)

    assert torch.equal(out, x + x)
    assert _saved_pto(tmp_path).count("pto.barrier <PIPE_ALL>") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
