# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the PyPTO, standalone DSA, and ptoas memory planners.

Two coupled behaviours are exercised:

1. ``MemoryPlanner.PTOAS`` makes ``PassManager`` skip the PyPTO on-chip
   allocation passes (``MemoryReuse`` + ``AllocateMemoryAddr``) so the ptoas
   ``PlanMemory`` pass owns allocation instead.
2. ``PTOCodegen.generate(..., emit_tile_addr=False)`` omits the physical
   ``addr`` operand on ``pto.alloc_tile`` (required at ptoas
   ``--pto-level=level2``, which rejects any ``addr`` operand).

The default (``MemoryPlanner.PYPTO``) preserves the pre-existing behaviour:
both passes run and codegen bakes ``addr`` for ``--pto-level=level3``.
``MemoryPlanner.DSA`` skips only ``MemoryReuse``: ``AllocateMemoryAddr`` exports
the unmerged problem, invokes the standalone solver, validates, and writes
physical addresses for the same level3 codegen contract.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import json

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.compile_profiling import CompileProfiler
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes

requires_dsa = pytest.mark.skipif(
    not passes.is_dsa_solver_available(), reason="PyPTO was built without PYPTO_ENABLE_DSA_SOLVER"
)


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


@pl.program
class ColVecIfPhiCarry:
    """Col-vector if-phi whose loop carry needs an explicit write-back move."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[16, 1], pl.FP32],
        y: pl.Tensor[[16, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        acc: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        m: pl.Tile[[16, 1], pl.FP32] = pl.load(x, [0, 0], [16, 1])
        s: pl.Tile[[16, 1], pl.FP32] = pl.load(x, [0, 0], [16, 1])
        for i in pl.range(4):
            c: pl.Tile[[16, 1], pl.FP32] = pl.load(y, [0, 0], [16, 1])
            if i == 0:
                m_new = pl.maximum(m, c)
                alpha = pl.exp(pl.sub(m, m_new))
                s = pl.mul(s, alpha)
                m = m_new
            else:
                m_new = pl.maximum(m, c)
                alpha = pl.exp(pl.sub(m, m_new))
                beta = pl.exp(pl.sub(c, m_new))
                s = pl.add(pl.mul(s, alpha), beta)
                m = m_new
        out = pl.store(m, [0, 0], out)
        acc = pl.store(s, [0, 0], acc)
        return out


def _run_pipeline(
    memory_planner: passes.MemoryPlanner,
    program: ir.Program = ElementwiseAdd,
    *,
    dsa_export_dir: str | None = None,
    dsa_solution_dir: str | None = None,
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext(
        [],
        memory_planner=memory_planner,
        dsa_export_dir=dsa_export_dir,
        dsa_solution_dir=dsa_solution_dir,
    ):
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


def test_pass_context_dsa_settings_round_trip(tmp_path):
    solution_dir = tmp_path / "solutions"
    ctx = passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.DSA,
        dsa_export_dir=str(tmp_path),
        dsa_solution_dir=str(solution_dir),
    )
    assert ctx.get_memory_planner() == passes.MemoryPlanner.DSA
    assert ctx.get_dsa_export_dir() == str(tmp_path)
    assert ctx.get_dsa_solution_dir() == str(solution_dir)


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


@requires_dsa
def test_dsa_pipeline_skips_reuse_but_runs_writeback():
    _, pass_names = _run_pipeline(passes.MemoryPlanner.DSA)
    assert "InitMemRef" in pass_names
    assert "MaterializeSemanticAliases" in pass_names
    assert "MemoryReuse" not in pass_names
    assert "AllocateMemoryAddr" in pass_names


@requires_dsa
def test_dsa_context_survives_dump_pipeline_and_exports(tmp_path):
    """The dump wrapper must preserve DSA selection and corpus destination."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    export_dir = tmp_path / "corpus"
    dump_dir = tmp_path / "ir"
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA, dsa_export_dir=str(export_dir)):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(ElementwiseAdd, dump_ir=True, output_dir=str(dump_dir))

    assert list(export_dir.glob("*.dsa.json"))
    mlir = _codegen(optimized, emit_tile_addr=True)
    assert "pto.alloc_tile" in mlir and "addr =" in mlir


@requires_dsa
def test_dsa_context_survives_profiled_pipeline_and_exports(tmp_path):
    """The profiling wrapper must preserve DSA selection and corpus destination."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    export_dir = tmp_path / "corpus"
    with (
        passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA, dsa_export_dir=str(export_dir)),
        CompileProfiler(),
    ):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        pm.run_passes(ElementwiseAdd)

    assert list(export_dir.glob("*.dsa.json"))


@requires_dsa
def test_dsa_solution_replay_survives_dump_pipeline_context(tmp_path):
    """Nested dump contexts must preserve the selected replay directory."""
    artifact_dir = tmp_path / "artifacts"
    solved, _ = _run_pipeline(
        passes.MemoryPlanner.DSA,
        dsa_export_dir=str(artifact_dir),
    )
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext(
        [],
        memory_planner=passes.MemoryPlanner.DSA,
        dsa_solution_dir=str(artifact_dir),
    ):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        replayed = pm.run_passes(
            ElementwiseAdd,
            dump_ir=True,
            output_dir=str(tmp_path / "ir"),
        )
    ir.assert_structural_equal(solved, replayed)


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


@requires_dsa
def test_dsa_codegen_emits_validated_alloc_tile_addr():
    optimized, _ = _run_pipeline(passes.MemoryPlanner.DSA)
    mlir = _codegen(optimized, emit_tile_addr=True)
    alloc_lines = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert alloc_lines, f"expected at least one pto.alloc_tile:\n{mlir}"
    assert all("addr =" in line for line in alloc_lines), (
        f"DSA mode writes validated addresses for ptoas --pto-level=level3:\n{mlir}"
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
    alloc_lines = [ln for ln in mlir.splitlines() if "pto.alloc_tile" in ln]
    addrs = [ln.split("addr =")[1].split()[0] for ln in alloc_lines if "addr =" in ln]
    # In level3 the loop-carry aliasing is carried by two allocs sharing an addr.
    assert len(addrs) != len(set(addrs)), f"PYPTO mode must alias the accumulator via a shared addr:\n{mlir}"


@requires_dsa
def test_dsa_colvec_loop_carry_runs_external_planner_fixup():
    """DSA must materialize both branch-phi and loop-carry write-backs."""
    dsa_optimized, _ = _run_pipeline(passes.MemoryPlanner.DSA, ColVecIfPhiCarry)
    ptoas_optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ColVecIfPhiCarry)

    dsa_ir = ir.python_print(dsa_optimized)
    ptoas_ir = ir.python_print(ptoas_optimized)
    dsa_moves = dsa_ir.count("tile.move")
    ptoas_moves = ptoas_ir.count("tile.move")
    assert ptoas_moves > 0, f"expected the col-vector carry to need a write-back move:\n{ptoas_ir}"
    # PTOAS materializes only the loop moves in IR; its addr-less codegen copies
    # branch yields into the if-phi handle. DSA emits explicit addresses, so it
    # must materialize those branch copies before lifetime export as additional
    # tile.move operations.
    assert dsa_moves > ptoas_moves, (
        f"DSA skipped its explicit-address if-phi write-backs: DSA={dsa_moves}, PTOAS={ptoas_moves}\n{dsa_ir}"
    )

    dsa_pto = _codegen(dsa_optimized, emit_tile_addr=True)
    assert dsa_pto.count("pto.tmov") == dsa_moves, (
        "DSA codegen dropped an IR-level branch-phi or loop-carry write-back: "
        f"IR tile.move={dsa_moves}, PTO pto.tmov={dsa_pto.count('pto.tmov')}\n{dsa_pto}"
    )


@requires_dsa
def test_dsa_mandatory_aliases_export_allocation_lifetime_hulls(tmp_path):
    """SSA-member gaps must not become unproved physical dead regions.

    Mandatory aliases share one physical allocation across loop carries and
    branch phis. An individual SSA member can stop being referenced before a
    later member reads the carried value, so only the allocation-level hull is
    sound solver input. This guards the DeepSeek-v4 ratio-4 softmax-pool
    regression where scratch occupied such a gap and corrupted the accumulator.
    """
    export_dir = tmp_path / "corpus"
    _run_pipeline(
        passes.MemoryPlanner.DSA,
        ColVecIfPhiCarry,
        dsa_export_dir=str(export_dir),
    )

    documents = [json.loads(path.read_text()) for path in export_dir.glob("*.dsa.json")]
    assert documents, "expected the DSA pipeline to export a structured problem"

    saw_mandatory_alias = False
    for document in documents:
        problem = document["problem"]
        aliases = {
            alias_class["buffer"]: alias_class["members"]
            for alias_class in problem["pypto_structure"]["alias_classes"]
        }
        for buffer in problem["buffers"]:
            if len(aliases[buffer["id"]]) > 1:
                saw_mandatory_alias = True
            assert len(buffer["live_intervals"]) == 1, (
                "DSA exported an unproved hole inside allocation "
                f"{buffer['name']}: {buffer['live_intervals']}"
            )

    assert saw_mandatory_alias, "test fixture no longer exercises mandatory alias classes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
