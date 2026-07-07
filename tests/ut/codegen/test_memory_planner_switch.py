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
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes

from pypto import backend, ir


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


@pl.program
class LoopCarriedAdd:
    """Loop-carried accumulator — the must-alias case: acc must stay one buffer."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[128, 64], pl.FP32],
        n: pl.Scalar[pl.INDEX],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        acc: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        for i, (acc_i,) in pl.range(n, init_values=(acc,)):
            t: pl.Tile[[64, 64], pl.FP32] = pl.load(
                a, [i * 64, 0], [64, 64], target_memory=pl.MemorySpace.Vec
            )
            acc_next: pl.Tile[[64, 64], pl.FP32] = pl.add(acc_i, t)
            r = pl.yield_(acc_next)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(r, [0, 0], output)
        return out


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program = ElementwiseAdd
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
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
    assert "MaterializeSemanticAliases" in pass_names


def test_ptoas_pipeline_skips_reuse_keeps_semantic_aliases():
    _, pass_names = _run_pipeline(passes.MemoryPlanner.PTOAS)
    # InitMemRef still runs (creates the MemRefs / alloc ops ptoas plans over).
    assert "InitMemRef" in pass_names
    # Semantics-required aliasing is preserved; only opportunistic reuse + addr
    # assignment are handed off to ptoas PlanMemory.
    assert "MaterializeSemanticAliases" in pass_names
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


# ---------------------------------------------------------------------------
# Loop-carried accumulator: must-alias must survive in PTOAS mode as an
# in-place tadd on a single shared handle (no addr, MemoryReuse skipped).
# ---------------------------------------------------------------------------


def test_ptoas_loop_carry_is_in_place_single_handle():
    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, LoopCarriedAdd)
    mlir = _codegen(optimized, emit_tile_addr=False)

    tadd = next((ln for ln in mlir.splitlines() if "pto.tadd" in ln), None)
    assert tadd is not None, f"expected a pto.tadd:\n{mlir}"
    # The accumulator's out handle must be one of the in handles (in-place),
    # i.e. MaterializeSemanticAliases retargeted acc_next onto acc's buffer even
    # though MemoryReuse did not run.
    ins = tadd.split("ins(")[1].split(")")[0]
    out = tadd.split("outs(")[1].split(")")[0]
    out_handle = out.split(":")[0].strip()
    in_handles = [tok.strip() for tok in ins.split(":")[0].split(",")]
    assert out_handle in in_handles, (
        f"PTOAS loop-carry must be in-place (out handle {out_handle} should alias an input "
        f"{in_handles}):\n{tadd}"
    )

    # The accumulator alloc appears once, not once per SSA name (acc == acc_next).
    acc_allocs = [ln for ln in mlir.splitlines() if "pto.alloc_tile" in ln and out_handle in ln]
    assert len(acc_allocs) == 1, f"accumulator buffer must have exactly one alloc_tile:\n{mlir}"


def test_pypto_loop_carry_uses_shared_addr():
    optimized, _ = _run_pipeline(passes.MemoryPlanner.PYPTO, LoopCarriedAdd)
    mlir = _codegen(optimized, emit_tile_addr=True)
    # Since #1956 the loop-carry accumulator is a single structural alloc_tile
    # handle (the same must-alias collapse as the PTOAS path — see
    # test_ptoas_loop_carry_is_in_place_single_handle — rather than the pre-#1956
    # pair of per-var allocs sharing one addr). The tadd writes in place onto that
    # handle, and PYPTO bakes a physical addr on it (the level3 distinction).
    tadd = next((ln for ln in mlir.splitlines() if "pto.tadd" in ln), None)
    assert tadd is not None, f"expected a pto.tadd:\n{mlir}"
    ins = tadd.split("ins(")[1].split(")")[0]
    out_handle = tadd.split("outs(")[1].split(")")[0].split(":")[0].strip()
    in_handles = [tok.strip() for tok in ins.split(":")[0].split(",")]
    assert out_handle in in_handles, (
        f"PYPTO loop-carry must be in-place (out handle {out_handle} should alias an input "
        f"{in_handles}):\n{tadd}"
    )
    acc_allocs = [ln for ln in mlir.splitlines() if "pto.alloc_tile" in ln and out_handle in ln]
    assert len(acc_allocs) == 1, f"accumulator buffer must have exactly one alloc_tile:\n{mlir}"
    assert "addr =" in acc_allocs[0], (
        f"PYPTO mode must bake a physical addr on the accumulator alloc_tile:\n{acc_allocs[0]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
