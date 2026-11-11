# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device coverage for authored pipeline barriers.

A barrier drains one hardware pipeline: ``MTE2`` the GM -> UB loads, ``MTE3`` the
UB -> GM stores, ``FIX`` the L0C writeback. These kernels place one at each phase
boundary of a load -> vector -> store chain and check that the result is still
correct and that each barrier reached the backend with the right pipe, in the
authored order, on the intended lane.

**Scope.** These compile with PTOAS automatic synchronization enabled (the
default), so they do NOT establish that the barriers alone order the handoffs --
PTOAS inserts the flag pairs that do. A barrier is not a substitute for a
``set_flag`` / ``wait_flag`` pair on a cross-pipeline handoff; see
``test_set_wait_flag.py`` for the flag-authored form, which disables automatic
synchronization. What is guarded here is placement and lane assignment.
"""

import re
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
    """One barrier per phase boundary, on the pipe that produced the data."""
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
    """Each authored barrier reaches the backend with the right pipe, in order."""
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


M_TILE = 64
N_TILE = 128
K_TILE = 256


@pl.jit
def cube_pinned_fix_barrier(
    x: pl.Tensor[[M_TILE, K_TILE], pl.FP16],
    w: pl.Tensor[[N_TILE, K_TILE], pl.FP16],
    out: pl.Out[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
):
    """A mixed kernel: cube matmul into L0C, vector epilogue after the C2V crossing.

    FIX is the L0C-writeback pipe, which only the cube lane owns, so the barrier
    is pinned to AIC. Without ``core_type`` it is SHARED and ExpandMixedKernel
    copies it into the AIV kernel too.
    """
    for _ in pl.spmd(1):
        acc = pl.matmul(x, w, b_trans=True, out_dtype=pl.FP32)
        pl.system.barrier(pipe=pl.PipeType.FIX, core_type=pl.KernelType.AIC)
        out[:] = pl.add(acc, acc)
    return out


def _barriers_per_kernel(tmp_path, pipe: str) -> dict[str, int]:
    """Count `pto.barrier <pipe>` in each generated func, keyed by function name."""
    counts: dict[str, int] = {}
    for path in tmp_path.rglob("*.pto"):
        for chunk in re.split(r"(?=func\.func @)", path.read_text()):
            name = re.match(r"func\.func @(\w+)", chunk)
            if name:
                counts[name.group(1)] = chunk.count(f"pto.barrier <{pipe}>")
    assert counts, "compilation did not save the PTO kernel"
    return counts


@pytest.mark.platforms("a2a3", "a2a3sim", "a5", "a5sim")
def test_cube_pinned_barrier_stays_off_the_vector_kernel(test_config, tmp_path):
    """core_type keeps a cube-side barrier out of the AIV half of a mixed kernel."""
    config = replace(test_config, save_kernels=True, save_kernels_dir=str(tmp_path))
    x = torch.randn(M_TILE, K_TILE, dtype=torch.float16)
    w = torch.randn(N_TILE, K_TILE, dtype=torch.float16)
    out = torch.zeros(M_TILE, N_TILE, dtype=torch.float32)

    cube_pinned_fix_barrier(x, w, out, config=config)

    expected = (x.to(torch.float32) @ w.to(torch.float32).T) * 2
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

    counts = _barriers_per_kernel(tmp_path, "PIPE_FIX")
    aic = {name: n for name, n in counts.items() if name.endswith("_aic")}
    aiv = {name: n for name, n in counts.items() if name.endswith("_aiv")}
    assert aic and aiv, f"expected a mixed kernel pair, got {sorted(counts)}"
    assert all(n >= 1 for n in aic.values()), f"cube lane lost its FIX barrier: {aic}"
    assert all(n == 0 for n in aiv.values()), f"FIX barrier leaked onto the vector lane: {aiv}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
