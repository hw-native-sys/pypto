# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Intra-core multi-buffer lowering of a pipelined tile carry (ptoas multi_tile_buf).

Under the PTOAS memory planner, a `pl.pipeline(stage=F)` loop carrying a tile is
lowered to a rolled loop over an F-slot `pto.alloc_multi_tile`: iteration k reads
slot (k-1) mod F (or the init for the peeled k==0) and writes slot k mod F, and
ptoas plans/rotates the physical slots. Under the PyPTO planner the same loop
keeps the existing unroll-based ping-pong (no multi_tile_buf).
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes

from pypto import ir as _ir


@pl.program
class PipeTileCarry:
    """`acc` starts as a loaded tile and is incremented once per pipelined
    iteration (stage=2), so the loop-carried tile rotates through 2 slots."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        acc: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        for i, (a,) in pl.pipeline(0, 4, 1, stage=2, init_values=(acc,)):  # noqa: B007
            a = pl.add(a, 1.0)
            a = pl.yield_(a)
        return pl.store(a, [0, 0], out)


def _pto(program, planner) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    emit_addr = planner == passes.MemoryPlanner.PYPTO
    return codegen.PTOCodegen().generate(
        _ir.Program([func], "kernel", optimized.span), emit_tile_addr=emit_addr
    )


def test_ptoas_planner_lowers_tile_carry_to_multi_tile_buf():
    """PTOAS planner: rolled loop + one alloc_multi_tile(count=2) + multi_tile_get slots."""
    mlir = _pto(PipeTileCarry, passes.MemoryPlanner.PTOAS)

    # Exactly one multi-buffer allocation, declaring count = stage = 2.
    allocs = [ln for ln in mlir.splitlines() if "pto.alloc_multi_tile" in ln]
    assert len(allocs) == 1, f"expected one alloc_multi_tile, got {len(allocs)}:\n{mlir}"
    assert "count = 2" in allocs[0], f"multi_tile_buf must carry count = 2:\n{allocs[0]}"

    # Slots are selected via multi_tile_get (peeled slot 0, loop prev/cur, final res).
    gets = [ln for ln in mlir.splitlines() if "pto.multi_tile_get" in ln]
    assert len(gets) >= 3, f"expected >=3 multi_tile_get (s0, sp/sc, res), got {len(gets)}:\n{mlir}"

    # The carry producers write into a slot handle (not a plain per-var alloc_tile),
    # and the loop stays rolled (step 1, not the unrolled step 2).
    adds = [ln for ln in mlir.splitlines() if "pto.tadds" in ln]
    assert adds, f"expected tadds producers:\n{mlir}"
    for ln in adds:
        assert "outs(%__mtb" in ln, f"producer must write a multi-buffer slot:\n{ln}"
    assert " step %c1_index" in mlir, f"multi-buffer loop must stay rolled (step 1):\n{mlir}"


def test_pypto_planner_keeps_unrolled_ping_pong():
    """PyPTO planner: existing unroll path — no multi_tile_buf is emitted."""
    mlir = _pto(PipeTileCarry, passes.MemoryPlanner.PYPTO)
    assert "multi_tile" not in mlir, f"PyPTO planner must not emit multi_tile_buf:\n{mlir}"
    # Body replicated F=2 times -> two increment producers, baked addresses.
    assert mlir.count("pto.tadds") == 2, f"expected the F=2 unrolled ping-pong:\n{mlir}"
    assert "pto.alloc_tile addr" in mlir, f"PyPTO planner bakes tile addresses:\n{mlir}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
