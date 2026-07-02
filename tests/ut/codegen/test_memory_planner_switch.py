# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the ``memory_planner`` switch (PyPTO vs ptoas PlanMemory).

Two coupled behaviours are exercised:

1. ``MemoryPlanner.PTOAS`` makes ``PassManager`` skip the PyPTO on-chip
   allocation passes (``MemoryReuse`` + ``AllocateMemoryAddr``) so the ptoas
   ``PlanMemory`` pass owns allocation instead.
2. ``PTOCodegen.generate(..., emit_tile_addr=False)`` omits the physical
   ``addr`` operand on ``pto.alloc_tile`` (required at ptoas
   ``--pto-level=level2``, which rejects any ``addr`` operand).

The default (``MemoryPlanner.PYPTO``) preserves the pre-existing behaviour:
both passes run and codegen bakes ``addr`` for ``--pto-level=level3``.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pl.program
class ElementwiseAdd:
    """Minimal InCore kernel: load two tiles, add, store — allocates tiles."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        ta: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        tb: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        tc: pl.Tile[[64, 64], pl.FP32] = pl.add(ta, tb)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(tc, [0, 0], output)
        return out


def _run_pipeline(memory_planner: passes.MemoryPlanner) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(ElementwiseAdd)
    return optimized, list(pm.pass_names)


def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)


# ---------------------------------------------------------------------------
# PassContext round-trip
# ---------------------------------------------------------------------------


def test_pass_context_default_planner_is_pypto():
    ctx = passes.PassContext([])
    assert ctx.get_memory_planner() == passes.MemoryPlanner.PYPTO


def test_pass_context_planner_round_trip():
    ctx = passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS)
    assert ctx.get_memory_planner() == passes.MemoryPlanner.PTOAS


# ---------------------------------------------------------------------------
# Pipeline: PTOAS skips the allocation passes; PYPTO keeps them
# ---------------------------------------------------------------------------


def test_pypto_pipeline_runs_allocation_passes():
    _, pass_names = _run_pipeline(passes.MemoryPlanner.PYPTO)
    assert "MemoryReuse" in pass_names
    assert "AllocateMemoryAddr" in pass_names
    assert "InitMemRef" in pass_names


def test_ptoas_pipeline_skips_allocation_passes():
    _, pass_names = _run_pipeline(passes.MemoryPlanner.PTOAS)
    # InitMemRef still runs (creates the MemRefs / alloc ops ptoas plans over).
    assert "InitMemRef" in pass_names
    # The two PyPTO allocation passes are handed off to ptoas PlanMemory.
    assert "MemoryReuse" not in pass_names
    assert "AllocateMemoryAddr" not in pass_names


# ---------------------------------------------------------------------------
# Codegen: emit_tile_addr controls the physical addr operand
# ---------------------------------------------------------------------------


def test_pypto_codegen_emits_alloc_tile_addr():
    optimized, _ = _run_pipeline(passes.MemoryPlanner.PYPTO)
    mlir = _codegen(optimized, emit_tile_addr=True)
    alloc_lines = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert alloc_lines, f"expected at least one pto.alloc_tile:\n{mlir}"
    assert any("addr =" in line for line in alloc_lines), (
        f"PYPTO mode must bake a physical addr on pto.alloc_tile:\n{mlir}"
    )


def test_ptoas_codegen_omits_alloc_tile_addr():
    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS)
    mlir = _codegen(optimized, emit_tile_addr=False)
    alloc_lines = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert alloc_lines, f"expected at least one pto.alloc_tile:\n{mlir}"
    assert all("addr =" not in line for line in alloc_lines), (
        f"PTOAS mode must not emit an addr operand (ptoas --pto-level=level2 rejects it):\n{mlir}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
