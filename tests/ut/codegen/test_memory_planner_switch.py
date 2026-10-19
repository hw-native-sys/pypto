# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the ``memory_planner`` switch.

The three product modes exercise different ownership boundaries:

1. ``MemoryPlanner.DSA_RP`` keeps planning in PyPTO, skips legacy
   ``MemoryReuse``, and runs the in-tree DSA-RP allocator.
2. ``MemoryPlanner.PTOAS`` makes ``PassManager`` skip the PyPTO on-chip
   allocation passes (``MemoryReuse`` + ``AllocateMemoryAddr``) so the ptoas
   ``PlanMemory`` pass owns allocation instead.
3. ``PTOCodegen.generate(..., emit_tile_addr=False)`` omits the physical
   ``addr`` operand on ``pto.alloc_tile`` (required at ptoas
   ``--pto-level=level2``, which rejects any ``addr`` operand).

The default (``MemoryPlanner.DSA_RP``) uses the in-tree capacity-constrained
planner. ``MemoryPlanner.PYPTO`` preserves the legacy behaviour explicitly:
both passes run and codegen bakes ``addr`` for ``--pto-level=level3``.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import re

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir
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
class ConsecutiveLoopCarriedAdd:
    """Two update loops whose return values share one carried allocation."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        seed: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        for _i, (first_i,) in pl.range(2, init_values=(seed,)):
            first_next: pl.Tile[[64, 64], pl.FP32] = pl.add(first_i, first_i)
            first_result = pl.yield_(first_next)
        for _j, (second_i,) in pl.range(2, init_values=(first_result,)):
            second_next: pl.Tile[[64, 64], pl.FP32] = pl.add(second_i, second_i)
            second_result = pl.yield_(second_next)
        residual: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        final: pl.Tile[[64, 64], pl.FP32] = pl.add(second_result, residual)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(final, [0, 0], output)
        return out


@pl.program
class StationaryMatmulLoop:
    """Tensor-level QK loop whose LHS should become one resident Mat panel."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[16, 128], pl.BF16],
        rhs: pl.Tensor[[128, 256], pl.BF16],
        output: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
    ) -> pl.Tensor[[16, 256], pl.FP32]:
        for n, (acc,) in pl.range(0, 256, 128, init_values=(output,)):
            rhs_n: pl.Tensor[[128, 128], pl.BF16] = pl.slice(rhs, [128, 128], [0, n])
            c_n: pl.Tensor[[16, 128], pl.FP32] = pl.matmul(lhs, rhs_n, out_dtype=pl.FP32)
            acc_next: pl.Tensor[[16, 256], pl.FP32] = pl.assemble(acc, c_n, [0, n])
            result = pl.yield_(acc_next)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        rhs: pl.Tensor[[128, 256], pl.BF16],
        output: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
    ) -> pl.Tensor[[16, 256], pl.FP32]:
        fresh_lhs = pl.create_tensor([16, 128], dtype=pl.BF16)
        result = self.kernel(fresh_lhs, rhs, output)
        return result


@pl.program
class StationaryMatmulPipelinedK:
    """A larger K creates a nested AutoTile pipeline under the user loop."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[16, 512], pl.BF16],
        rhs: pl.Tensor[[512, 256], pl.BF16],
        output: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
    ) -> pl.Tensor[[16, 256], pl.FP32]:
        for n, (acc,) in pl.range(0, 256, 128, init_values=(output,)):
            rhs_n: pl.Tensor[[512, 128], pl.BF16] = pl.slice(rhs, [512, 128], [0, n])
            c_n: pl.Tensor[[16, 128], pl.FP32] = pl.matmul(lhs, rhs_n, out_dtype=pl.FP32)
            acc_next: pl.Tensor[[16, 256], pl.FP32] = pl.assemble(acc, c_n, [0, n])
            result = pl.yield_(acc_next)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        rhs: pl.Tensor[[512, 256], pl.BF16],
        output: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
    ) -> pl.Tensor[[16, 256], pl.FP32]:
        fresh_lhs = pl.create_tensor([16, 512], dtype=pl.BF16)
        result = self.kernel(fresh_lhs, rhs, output)
        return result


@pl.program
class ConditionalGather:
    """Two conditional writes followed by a differently typed Mat view (#3016)."""

    @pl.function(type=pl.FunctionType.AIC)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP16],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
    ) -> pl.Tensor[[64, 16], pl.FP32]:
        base = pl.tile.create([32, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
        if flag != 0:
            first = pl.tile.gather_row(base, x, [0, 0], [0, 0], [16, 64])
        else:
            first = pl.tile.gather_row(base, x, [0, 0], [16, 0], [16, 64])
        if flag != 0:
            second = pl.tile.gather_row(first, x, [16, 0], [32, 0], [16, 64])
        else:
            second = pl.tile.gather_row(first, x, [16, 0], [48, 0], [16, 64])
        transposed = pl.tile.transpose_view(second)
        rhs = pl.tile.load(x, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        left = pl.tile.move(transposed, target_memory=pl.Mem.Left)
        right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
        product = pl.tile.matmul(left, right)
        output = pl.tile.store(product, [0, 0], out)
        return output


def _collect_tile_phi_storage(program: ir.Program):
    """Collect phi/arm MemRefs and named tile definitions without comparing SSA names."""

    class Collector(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.phis = []
            self.tiles = {}

        def visit_assign_stmt(self, op):
            if isinstance(op.var.type, ir.TileType):
                self.tiles[op.var.name_hint] = op.var
            super().visit_assign_stmt(op)

        def visit_if_stmt(self, op):
            arms = []
            for body in (op.then_body, op.else_body):
                statements = body.stmts if isinstance(body, ir.SeqStmts) else [body]
                yielded = next(stmt for stmt in statements if isinstance(stmt, ir.YieldStmt))
                tile_type = yielded.value[0].type
                assert isinstance(tile_type, ir.TileType)
                arms.append(tile_type.memref)
            self.phis.append((op.return_vars[0], arms))
            super().visit_if_stmt(op)

    collector = Collector()
    for function in program.functions.values():
        collector.visit_stmt(function.body)
    return collector.phis, collector.tiles


def _assert_same_tile_window(actual: ir.MemRef, expected: ir.MemRef):
    assert ir.MemRef.same_allocation(actual, expected)
    assert ir.structural_equal(actual.byte_offset_, expected.byte_offset_)
    assert actual.size_ == expected.size_
    assert actual.slot_count_ == expected.slot_count_
    if expected.slot_index_ is None:
        assert actual.slot_index_ is None
    else:
        assert actual.slot_index_ is not None
        assert ir.structural_equal(actual.slot_index_, expected.slot_index_)


def _calls_with_loop_depth(program):
    class Collector(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.depth = 0
            self.calls = []

        def visit_iter_arg(self, op):
            pass

        def visit_call(self, op):
            self.calls.append((op, self.depth))
            super().visit_call(op)

    collector = Collector()
    for function in program.functions.values():
        pending = [(function.body, 0)]
        while pending:
            statement, depth = pending.pop()
            if isinstance(statement, ir.SeqStmts):
                pending.extend((child, depth) for child in reversed(statement.stmts))
            elif isinstance(statement, ir.ForStmt):
                pending.append((statement.body, depth + 1))
            elif isinstance(statement, ir.IfStmt):
                if statement.else_body is not None:
                    pending.append((statement.else_body, depth))
                pending.append((statement.then_body, depth))
            else:
                collector.depth = depth
                collector.visit_stmt(statement)
    return collector.calls


def _run_pipeline(
    memory_planner: passes.MemoryPlanner,
    program: ir.Program = ElementwiseAdd,
    *,
    enable_buffer_ir: bool = True,
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=memory_planner, enable_buffer_ir=enable_buffer_ir):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)


def _run_default_pipeline(program: ir.Program = ElementwiseAdd) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline without overriding PassContext's planner."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([]):
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


def test_pass_context_default_planner_is_dsa_rp():
    assert passes.get_default_memory_planner() == passes.MemoryPlanner.DSA_RP
    ctx = passes.PassContext([])
    assert ctx.get_memory_planner() == passes.MemoryPlanner.DSA_RP


def test_default_pipeline_matches_explicit_dsa_rp():
    default_program, default_pass_names = _run_default_pipeline()
    explicit_program, explicit_pass_names = _run_pipeline(passes.MemoryPlanner.DSA_RP)

    assert default_pass_names == explicit_pass_names
    assert "MemoryReuse" not in default_pass_names
    assert "AllocateMemoryAddr" in default_pass_names
    assert ir.python_print(default_program) == ir.python_print(explicit_program)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS])
def test_pass_context_planner_round_trip(planner):
    ctx = passes.PassContext([], memory_planner=planner)
    assert ctx.get_memory_planner() == planner


# ---------------------------------------------------------------------------
# Pipeline: PTOAS skips the allocation passes; PYPTO keeps them
# ---------------------------------------------------------------------------


def test_pypto_pipeline_runs_allocation_passes():
    _, pass_names = _run_pipeline(passes.MemoryPlanner.PYPTO)
    assert "MemoryReuse" in pass_names
    assert "AllocateMemoryAddr" in pass_names
    assert "InitMemRef" in pass_names
    assert "MaterializeSemanticAliases" in pass_names


def test_dsa_rp_pipeline_replaces_legacy_reuse_but_assigns_addresses():
    _, pass_names = _run_pipeline(passes.MemoryPlanner.DSA_RP)
    assert "InitMemRef" in pass_names
    assert "MaterializeSemanticAliases" in pass_names
    assert "MemoryReuse" not in pass_names
    assert "AllocateMemoryAddr" in pass_names


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


def test_dsa_rp_codegen_emits_alloc_tile_addr():
    optimized, _ = _run_pipeline(passes.MemoryPlanner.DSA_RP)
    mlir = _codegen(optimized, emit_tile_addr=True)
    alloc_lines = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert alloc_lines, f"expected at least one pto.alloc_tile:\n{mlir}"
    assert all("addr =" in line for line in alloc_lines), (
        f"DSA_RP mode must bake every physical address:\n{mlir}"
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


def test_ptoas_conditional_gather_phis_share_the_written_mat_window():
    """The second update and transpose must observe the first phi's repaired storage."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        initialized = passes.init_mem_ref()(passes.convert_to_ssa()(ConditionalGather))
        after = passes.materialize_semantic_aliases()(initialized)

    phis, tiles = _collect_tile_phi_storage(after)
    assert len(phis) == 2
    base = next(var for name, var in tiles.items() if name.startswith("base")).type.memref
    assert base is not None
    for phi, arms in phis:
        assert phi.type.memref is not None
        _assert_same_tile_window(phi.type.memref, base)
        for arm in arms:
            assert arm is not None
            _assert_same_tile_window(arm, base)
    transposed = next(var for name, var in tiles.items() if name.startswith("transposed"))
    _assert_same_tile_window(transposed.type.memref, base)


@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_ptoas_conditional_gather_preserves_one_mat_buffer(enable_buffer_ir):
    """Both branch updates must reach one Mat buffer without an illegal Mat->Mat move."""
    optimized, _ = _run_pipeline(
        passes.MemoryPlanner.PTOAS, ConditionalGather, enable_buffer_ir=enable_buffer_ir
    )
    mlir = _codegen(optimized, emit_tile_addr=False)
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert not any(line.count("loc=mat") == 2 for line in moves), mlir
    bases = {
        line.split("pto.subview ", 1)[1].split("[", 1)[0]
        for line in mlir.splitlines()
        if "pto.subview " in line
    }
    assert len(bases) == 1, mlir


def test_ptoas_conditional_inplace_vec_updates_reuse_mixed_view_handle():
    """A Vec phi with a later reshape preserves its existing in-place destination."""

    @pl.program
    class ConditionalVecUpdate:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            x: pl.Tensor[[16, 32], pl.FP32],
            src: pl.Tensor[[8, 32], pl.FP32],
            indexes: pl.Tensor[[4, 32], pl.INT32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
        ) -> pl.Tensor[[8, 64], pl.FP32]:
            base = pl.tile.load(x, [0, 0], [16, 32], target_memory=pl.Mem.Vec)
            first_src = pl.tile.load(src, [0, 0], [4, 32], target_memory=pl.Mem.Vec)
            second_src = pl.tile.load(src, [4, 0], [4, 32], target_memory=pl.Mem.Vec)
            index = pl.tile.load(indexes, [0, 0], [4, 32], target_memory=pl.Mem.Vec)
            if flag != 0:
                selected = pl.tile.scatter(base, first_src, index)
            else:
                selected = pl.tile.scatter(base, second_src, index)
            reshaped = pl.tile.reshape(selected, [8, 64])
            output = pl.tile.store(reshaped, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ConditionalVecUpdate, enable_buffer_ir=False)
    mlir = _codegen(optimized, emit_tile_addr=False)
    lines = mlir.splitlines()
    loads = [line for line in lines if "pto.tload" in line]
    assert len(loads) == 4, mlir
    base = loads[0].split("outs(")[1].split(":")[0].strip()
    updates = [line for line in lines if "pto.tscatter" in line]
    assert len(updates) == 2, mlir
    assert all(line.split("outs(")[1].split(":")[0].strip() == base for line in updates), mlir
    # A differently shaped view makes this storage identity mixed-type. Both
    # branch writes must still target the initialized Vec buffer and allocate it once.
    allocations = [line for line in lines if "pto.alloc_tile" in line and "loc=vec" in line]
    assert len(allocations) == 4, mlir  # The base, two source tiles, and flattened indexes.
    assert not any("pto.tmov" in line and line.count("loc=vec") == 2 for line in lines), mlir


def test_ptoas_distinct_mat_phi_preserves_independently_live_mixed_view_input():
    """A differently typed view must not let an else load overwrite a live then input."""

    @pl.program
    class ConditionalMatLoad:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[128, 16], pl.FP32]],
        ) -> pl.Tensor[[128, 16], pl.FP32]:
            base = pl.tile.load(x, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
            if flag != 0:
                selected = base
            else:
                selected = pl.tile.load(x, [32, 0], [32, 64], target_memory=pl.Mem.Mat)
            base_transposed = pl.tile.transpose_view(base)
            selected_transposed = pl.tile.transpose_view(selected)
            rhs = pl.tile.load(x, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
            base_left = pl.tile.move(base_transposed, target_memory=pl.Mem.Left)
            right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
            base_product = pl.tile.matmul(base_left, right)
            first_out = pl.tile.store(base_product, [0, 0], out)
            selected_left = pl.tile.move(selected_transposed, target_memory=pl.Mem.Left)
            selected_product = pl.tile.matmul(selected_left, right)
            output = pl.tile.store(selected_product, [64, 0], first_out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ConditionalMatLoad, enable_buffer_ir=False)
    mlir = _codegen(optimized, emit_tile_addr=False)
    loads = [line for line in mlir.splitlines() if "pto.tload" in line]
    assert len(loads) >= 2, mlir
    base_handle, else_handle = [line.split("outs(")[1].split(":")[0].strip() for line in loads[:2]]
    # The base is still read after the merge, independently of the selected value.
    # Redirecting the else producer into it would corrupt the first output block.
    # Distinct windows must keep their existing reconciliation fallback.
    assert base_handle != else_handle, mlir


@pytest.mark.parametrize("identity_merge", [False, True])
def test_ptoas_nested_mat_phi_does_not_reuse_sibling_branch_declaration(identity_merge):
    """A previously emitted sibling Mat handle does not dominate a nested phi."""

    @pl.program
    class NestedMatPhi:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            outer_flag: pl.Scalar[pl.INT32],
            inner_flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            rhs = pl.tile.load(x, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
            right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
            if outer_flag != 0:
                first = pl.tile.load(x, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
                first_view = pl.tile.transpose_view(first)
                first_left = pl.tile.move(first_view, target_memory=pl.Mem.Left)
                first_product = pl.tile.matmul(first_left, right)
                output = pl.tile.store(first_product, [0, 0], out)
            else:
                if inner_flag != 0:
                    second = pl.tile.load(x, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
                    selected = pl.yield_(second)
                else:
                    third = pl.tile.load(x, [32, 0], [32, 64], target_memory=pl.Mem.Mat)
                    selected = pl.yield_(third)
                selected_view = pl.tile.transpose_view(selected)
                selected_left = pl.tile.move(selected_view, target_memory=pl.Mem.Left)
                selected_product = pl.tile.matmul(selected_left, right)
                output = pl.tile.store(selected_product, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, NestedMatPhi, enable_buffer_ir=False)

    if identity_merge:
        # Insert after Simplify: source-level identity branches would disappear
        # before reaching the codegen boundary. Both arms yield the first phi,
        # whose actual handle becomes a head allocation after rejecting the
        # canonical handle declared only in the sibling outer branch.
        class InsertIdentityMerge(ir.IRMutator):
            def __init__(self):
                super().__init__()
                self.replacements = {}
                self.inserted = 0

            def visit_var(self, op):
                return self.replacements.get(op.unique_id, op)

            def rewrite_body(self, op):
                # Re-enter the Python override explicitly: the native trampoline
                # suppresses recursive calls to an already active override.
                if isinstance(op, ir.IfStmt):
                    return self.visit_if_stmt(op)
                if isinstance(op, ir.SeqStmts):
                    return ir.SeqStmts([self.rewrite_body(stmt) for stmt in op.stmts], op.span)
                return self.visit_stmt(op)

            def visit_if_stmt(self, op):
                result = ir.IfStmt(
                    self.visit_expr(op.condition),
                    self.rewrite_body(op.then_body),
                    self.rewrite_body(op.else_body) if op.else_body is not None else None,
                    op.return_vars,
                    op.span,
                )
                if len(result.return_vars) != 1 or not isinstance(result.return_vars[0].type, ir.TileType):
                    return result
                phi = result.return_vars[0]
                merged = ir.Var("identity_merged", phi.type, phi.span)
                following = ir.IfStmt(
                    result.condition,
                    ir.YieldStmt([phi], phi.span),
                    ir.YieldStmt([phi], phi.span),
                    [merged],
                    phi.span,
                )
                self.replacements[phi.unique_id] = merged
                self.inserted += 1
                return ir.SeqStmts([result, following], result.span)

        insert = InsertIdentityMerge()
        optimized = insert.visit_program(optimized)
        assert insert.inserted == 1

    # Model coalesced storage across mutually exclusive branches directly in the
    # lowered IR. PTOAS rejects author-pinned single-slot allocations, so a DSL
    # MemRef declaration would stop before reaching the codegen boundary here.
    class CoalesceBranchPanels(ir.IRMutator):
        def __init__(self):
            super().__init__()
            self.memref = None
            self.variables = {}

        def visit_var(self, op):
            tile_type = op.type
            if (
                not isinstance(tile_type, ir.TileType)
                or tile_type.memory_space != ir.MemorySpace.Mat
                or tile_type.memref is None
                or tile_type.memref.size_ != 4096
            ):
                return op
            if self.memref is None:
                self.memref = tile_type.memref
            if op.unique_id not in self.variables:
                self.variables[op.unique_id] = ir.Var(
                    op.name_hint,
                    ir.TileType(
                        tile_type.shape,
                        tile_type.dtype,
                        self.memref,
                        tile_type.tile_view,
                        tile_type.memory_space,
                    ),
                    op.span,
                )
            return self.variables[op.unique_id]

    optimized = CoalesceBranchPanels().visit_program(optimized)
    mlir = _codegen(optimized, emit_tile_addr=False)
    lines = mlir.splitlines()
    panel_loads = [line for line in lines if "pto.tload" in line and "rows=32, cols=64" in line]
    assert len(panel_loads) == 3, mlir
    nested_handles = [line.split("outs(")[1].split(":")[0].strip() for line in panel_loads[1:]]
    assert nested_handles[0] == nested_handles[1], mlir
    declarations = [i for i, line in enumerate(lines) if f"{nested_handles[0]} = pto.alloc_tile" in line]
    assert len(declarations) == 1, mlir
    outer_if = next(i for i, line in enumerate(lines) if "scf.if" in line)
    # A function-global "already emitted" set includes allocations inside the
    # sibling then branch. The nested phi needs a head declaration that dominates
    # both of its producers, regardless of which outer branch executes.
    assert declarations[0] < outer_if, mlir
    panel_allocations = [
        line
        for line in lines
        if "pto.alloc_tile" in line and "loc=mat" in line and "rows=32, cols=64" in line
    ]
    assert len(panel_allocations) == 2, mlir  # Sibling-local storage and one shared head phi.
    assert not any("pto.tmov" in line and line.count("loc=mat") == 2 for line in lines), mlir
    transposes = [line for line in lines if "pto.treshape" in line]
    assert transposes and f"pto.treshape {nested_handles[0]} :" in transposes[-1], mlir


def test_ptoas_nested_mat_phi_reuses_dominating_enclosing_branch_declaration():
    """An inline Mat declaration in the active outer branch dominates its nested if."""

    @pl.program
    class NestedGather:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            outer_flag: pl.Scalar[pl.INT32],
            inner_flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            if outer_flag != 0:
                output = out
            else:
                base = pl.tile.load(x, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
                if inner_flag != 0:
                    selected = pl.tile.gather_row(base, x, [0, 0], [0, 0], [16, 64])
                else:
                    selected = pl.tile.gather_row(base, x, [0, 0], [32, 0], [16, 64])
                transposed = pl.tile.transpose_view(selected)
                rhs = pl.tile.load(x, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
                left = pl.tile.move(transposed, target_memory=pl.Mem.Left)
                right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
                product = pl.tile.matmul(left, right)
                output = pl.tile.store(product, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, NestedGather, enable_buffer_ir=False)
    mlir = _codegen(optimized, emit_tile_addr=False)
    lines = mlir.splitlines()
    base_load = next(line for line in lines if "pto.tload" in line)
    base = base_load.split("outs(")[1].split(":")[0].strip()
    subview_bases = {
        line.split("pto.subview ", 1)[1].split("[", 1)[0] for line in lines if "pto.subview " in line
    }
    assert subview_bases == {base}, mlir
    assert not any("pto.tmov" in line and line.count("loc=mat") == 2 for line in lines), mlir
    declarations = [i for i, line in enumerate(lines) if f"{base} = pto.alloc_tile" in line]
    if_indices = [i for i, line in enumerate(lines) if "scf.if" in line]
    assert len(declarations) == 1 and len(if_indices) == 2, mlir
    # Keep useful reuse inside an active ancestor scope. Requiring every shared
    # Mat to be head-declared would reintroduce the forbidden Mat-to-Mat copy.
    assert if_indices[0] < declarations[0] < if_indices[1], mlir


def test_ptoas_repaired_mat_phi_preserves_partial_view_and_gather_storage():
    """Repair a descendant slice descriptor without changing its relative window or slot."""
    span = ir.Span.unknown()
    allocation = ir.Var("allocation", ir.PtrType(), span)
    stale_allocation = ir.Var("stale_allocation", ir.PtrType(), span)
    slot = ir.ConstInt(1, DataType.INDEX, span)
    full_offset = ir.ConstInt(4096, DataType.INDEX, span)
    slice_offset = ir.ConstInt(6144, DataType.INDEX, span)
    source = ir.MemRef(allocation, full_offset, 4096, slots=2, slot=slot)
    stale = ir.MemRef(stale_allocation, full_offset, 4096, slots=2, slot=slot)
    old_slice = ir.MemRef(stale_allocation, slice_offset, 2048, slots=2, slot=slot)
    full_type = ir.TileType([32, 64], DataType.FP16, source, memory_space=ir.MemorySpace.Mat)
    base = ir.Var("base", full_type, span)
    phi = ir.Var("merged", ir.TileType([32, 64], DataType.FP16, stale, memory_space=ir.MemorySpace.Mat), span)
    slice_type = ir.TileType([16, 64], DataType.FP16, old_slice, memory_space=ir.MemorySpace.Mat)
    sliced = ir.Var("sliced", slice_type, span)
    gathered = ir.Var("gathered", slice_type, span)
    condition = ir.Var("condition", ir.ScalarType(DataType.BOOL), span)
    x = ir.Var("x", ir.TensorType([64, 64], DataType.FP16), span)

    def dims(*values):
        return ir.MakeTuple([ir.ConstInt(value, DataType.INDEX, span) for value in values], span)

    body = ir.SeqStmts(
        [
            ir.IfStmt(condition, ir.YieldStmt([base], span), ir.YieldStmt([base], span), [phi], span),
            ir.AssignStmt(
                sliced,
                ir.Call(ir.get_op("tile.slice"), [phi, dims(16, 64), dims(16, 0)], slice_type, span),
                span,
            ),
            ir.AssignStmt(
                gathered,
                ir.Call(
                    ir.get_op("tile.gather_row"),
                    [sliced, x, dims(0, 0), dims(0, 0), dims(16, 64)],
                    slice_type,
                    span,
                ),
                span,
            ),
            ir.ReturnStmt([gathered], span),
        ],
        span,
    )
    function = ir.Function(
        "kernel",
        [allocation, stale_allocation, base, x, condition],
        [ir.TileType([16, 64], DataType.FP16, memory_space=ir.MemorySpace.Mat)],
        body,
        span,
        type=ir.FunctionType.AIC,
    )
    before = ir.Program([function], "Before", span)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(before)
    phis, tiles = _collect_tile_phi_storage(after)
    _assert_same_tile_window(phis[0][0].type.memref, source)
    expected = ir.MemRef(allocation, slice_offset, 2048, slots=2, slot=slot)
    for name in ("sliced", "gathered"):
        tile_type = tiles[name].type
        assert isinstance(tile_type, ir.TileType) and tile_type.memref is not None
        _assert_same_tile_window(tile_type.memref, expected)
        assert tile_type.tile_view == slice_type.tile_view


def test_ptoas_loop_mat_identity_phi_gather_writes_nonzero_row_subview():
    """An IterArg passed through an if still supplies the gather destination subview."""

    @pl.program
    class LoopGather:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
        ) -> pl.Tensor[[48, 16], pl.FP32]:
            base = pl.tile.load(x, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
            rhs = pl.tile.load(x, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
            right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
            for i, (carried,) in pl.range(2, init_values=(base,)):
                window = pl.tile.slice(carried, [16, 64], [16, 0])
                updated = pl.tile.gather_row(window, x, [0, 0], [16 + i * 16, 0], [16, 64])
                partial_left = pl.tile.move(updated, target_memory=pl.Mem.Left)
                partial_product = pl.tile.matmul(partial_left, right)
                _partial_out = pl.tile.store(partial_product, [0, 0], out)
                full_update = pl.tile.gather_row(carried, x, [0, 0], [i * 16, 0], [16, 64])
                result = pl.yield_(full_update)
            left = pl.tile.move(result, target_memory=pl.Mem.Left)
            product = pl.tile.matmul(left, right)
            output = pl.tile.store(product, [16, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, LoopGather, enable_buffer_ir=False)

    class InsertCarryIdentityMerge(ir.IRMutator):
        def __init__(self):
            super().__init__()
            self.loop_vars = []
            self.inserted = 0

        def visit_for_stmt(self, op):
            self.loop_vars.append(op.loop_var)
            result = super().visit_for_stmt(op)
            self.loop_vars.pop()
            return result

        def visit_assign_stmt(self, op):
            result = super().visit_assign_stmt(op)
            if (
                not isinstance(result, ir.AssignStmt)
                or not isinstance(result.value, ir.Call)
                or result.value.op.name != ir.get_op("tile.slice").name
                or not isinstance(result.value.args[0], ir.IterArg)
            ):
                return result
            carried = result.value.args[0]
            phi = ir.Var("selected_carry", carried.type, result.span)
            condition = ir.Gt(
                self.loop_vars[-1], ir.ConstInt(0, DataType.INDEX, result.span), DataType.BOOL, result.span
            )
            merge = ir.IfStmt(
                condition,
                ir.YieldStmt([carried], result.span),
                ir.YieldStmt([carried], result.span),
                [phi],
                result.span,
            )
            call = result.value
            sliced = ir.AssignStmt(
                result.var,
                ir.Call(call.op, [phi, *call.args[1:]], dict(call.kwargs), call.type, call.span),
                result.span,
            )
            self.inserted += 1
            return ir.SeqStmts([merge, sliced], result.span)

    insert = InsertCarryIdentityMerge()
    optimized = insert.visit_program(optimized)
    assert insert.inserted == 1  # Keep the identity if after Simplify has finished.
    mlir = _codegen(optimized, emit_tile_addr=False)
    lines = mlir.splitlines()
    base_load = next(line for line in lines if "pto.tload" in line)
    base = base_load.split("outs(")[1].split(":")[0].strip()
    views = {
        line.split("=", 1)[0].strip(): line.split("pto.subview ", 1)[1].split("[", 1)[0]
        for line in lines
        if "pto.subview " in line
    }
    lower_views = {
        line.split("=", 1)[0].strip() for line in lines if "pto.subview " in line and "[%c16_index," in line
    }
    gather_loads = [line for line in lines if "pto.tload" in line and "rows=16, cols=64" in line]
    assert len(gather_loads) == 2 and lower_views, mlir
    writes_lower_window = False
    for line in gather_loads:
        handle = line.split("outs(")[1].split(":")[0].strip()
        path = {handle}
        while handle in views:
            handle = views[handle]
            path.add(handle)
        assert handle == base, mlir
        writes_lower_window |= bool(path & lower_views)
    assert writes_lower_window, mlir
    allocations = [line for line in lines if "pto.alloc_tile" in line and "loc=mat" in line]
    references = re.findall(r"%[\w.$-]+", mlir)
    live_allocations = {
        line.split("=", 1)[0].strip()
        for line in allocations
        if references.count(line.split("=", 1)[0].strip()) > 1
    }
    initial_loads = [line for line in lines if "pto.tload" in line][:2]
    assert live_allocations == {line.split("outs(")[1].split(":")[0].strip() for line in initial_loads}, mlir
    assert len(live_allocations) == 2, mlir  # Only the full carried panel and RHS supply live storage.
    # Slice lowering may reserve an unused window allocation. It must never
    # supply either gather write; the subview chains above prove both target base.
    assert all(
        line.split("=", 1)[0].strip() in live_allocations or "rows=16, cols=64" in line
        for line in allocations
    ), mlir
    assert not any("pto.tmov" in line and line.count("loc=mat") == 2 for line in lines), mlir


@pytest.mark.parametrize(
    "memory_space",
    [
        ir.MemorySpace.Mat,
        ir.MemorySpace.Vec,
        ir.MemorySpace.Left,
        ir.MemorySpace.Right,
        ir.MemorySpace.Acc,
        ir.MemorySpace.Bias,
        ir.MemorySpace.LeftScale,
        ir.MemorySpace.RightScale,
    ],
    ids=lambda space: space.name,
)
@pytest.mark.parametrize(
    "difference",
    [
        "same_window",
        "offset",
        "size",
        "slot",
        "slot_count",
        "allocation",
        "cross_space",
        "missing_storage",
        "unknown_space",
        "result_space",
        "missing_result_storage",
        "unknown_result_space",
        "all_unknown",
    ],
)
def test_ptoas_if_alias_fixup_only_coalesces_same_tile_window(memory_space, difference):
    """Reuse requires an exact window and a known, matching space on both yields and phi."""
    span = ir.Span.unknown()
    allocation = ir.Var("allocation", ir.PtrType(), span)
    other_allocation = ir.Var("other_allocation", ir.PtrType(), span)
    destination_allocation = ir.Var("destination_allocation", ir.PtrType(), span)
    zero = ir.ConstInt(0, DataType.INDEX, span)
    one = ir.ConstInt(1, DataType.INDEX, span)
    a_memref = ir.MemRef(allocation, zero, 1024, slots=2, slot=zero)
    b_memref = a_memref
    if difference == "offset":
        b_memref = ir.MemRef(allocation, ir.ConstInt(1024, DataType.INDEX, span), 1024, slots=2, slot=zero)
    elif difference == "size":
        b_memref = ir.MemRef(allocation, zero, 2048, slots=2, slot=zero)
    elif difference == "slot":
        b_memref = ir.MemRef(allocation, zero, 1024, slots=2, slot=one)
    elif difference == "slot_count":
        b_memref = ir.MemRef(allocation, zero, 1024, slots=3, slot=zero)
    elif difference == "allocation":
        b_memref = ir.MemRef(other_allocation, zero, 1024, slots=2, slot=zero)
    destination = ir.MemRef(destination_allocation, zero, 1024, slots=2, slot=zero)
    other_space = ir.MemorySpace.Mat if memory_space == ir.MemorySpace.Vec else ir.MemorySpace.Vec
    b_space = other_space if difference == "cross_space" else memory_space
    phi_space = other_space if difference == "result_space" else memory_space
    a_space = memory_space
    if difference == "unknown_space":
        b_space = None
    elif difference == "unknown_result_space":
        phi_space = None
    elif difference == "all_unknown":
        a_space = b_space = phi_space = None
    dtype = DataType.FP32 if memory_space in (ir.MemorySpace.Acc, ir.MemorySpace.Bias) else DataType.FP16
    if memory_space in (ir.MemorySpace.LeftScale, ir.MemorySpace.RightScale):
        dtype = DataType.FP8E8M0
    source_view = ir.TileView(valid_shape=[16, 16])
    view = ir.TileView(valid_shape=[4, 32], stride=[32, 1], start_offset=0, pad=ir.PadValue.zero)
    # TileType rejects unknown spaces together with a physical MemRef. An
    # unknown-space value therefore also has no initialized storage to reuse.
    # Known-space values without a MemRef exercise the separate missing-storage guard.
    a_storage = a_memref if a_space is not None else None
    b_storage = b_memref if b_space is not None and difference != "missing_storage" else None
    a = ir.Var("a", ir.TileType([16, 16], dtype, a_storage, source_view, a_space), span)
    b = ir.Var("b", ir.TileType([16, 16], dtype, b_storage, source_view, b_space), span)
    phi_memref = destination if phi_space is not None and difference != "missing_result_storage" else None
    phi_type = ir.TileType([16, 16], dtype, phi_memref, source_view, phi_space)
    phi = ir.Var("merged", phi_type, span)
    view_type = ir.TileType([8, 32], dtype, phi_memref, view, phi_space)
    reshaped = ir.Var("reshaped", view_type, span)
    alias = ir.Var("alias", view_type, span)
    condition = ir.Var("condition", ir.ScalarType(DataType.BOOL), span)
    shape = ir.MakeTuple([ir.ConstInt(8, DataType.INDEX, span), ir.ConstInt(32, DataType.INDEX, span)], span)
    reshape = ir.Call(ir.get_op("tile.reshape"), [phi, shape], view_type, span)
    body = ir.SeqStmts(
        [
            ir.IfStmt(condition, ir.YieldStmt([a], span), ir.YieldStmt([b], span), [phi], span),
            ir.AssignStmt(reshaped, reshape, span),
            ir.AssignStmt(alias, reshaped, span),
            ir.ReturnStmt([alias], span),
        ],
        span,
    )
    # Native initialized IR keeps this storage-only test independent of which
    # producer/load operations a backend supports for Bias and MX scale tiles.
    # An independent old phi destination makes every attempted repair observable.
    function = ir.Function(
        "kernel",
        [allocation, other_allocation, destination_allocation, a, b, condition],
        [ir.TileType([8, 32], dtype, tile_view=view, memory_space=phi_space)],
        body,
        span,
        type=ir.FunctionType.InCore,
    )
    before = ir.Program([function], "Before", span)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(before)

    if difference != "same_window":
        # Sharing a pointer does not authorize cross-space or uncertain aliases,
        # and the false switch continues to leave distinct windows to codegen.
        ir.assert_structural_equal(after, before)
        return

    phis, tiles = _collect_tile_phi_storage(after)
    new_phi, arms = phis[0]
    _assert_same_tile_window(new_phi.type.memref, a_memref)
    assert new_phi.type.memory_space == memory_space
    assert new_phi.type.tile_view == phi_type.tile_view
    for arm in arms:
        _assert_same_tile_window(arm, a_memref)
    for name in ("reshaped", "alias"):
        tile_type = tiles[name].type
        assert isinstance(tile_type, ir.TileType) and tile_type.memref is not None
        _assert_same_tile_window(tile_type.memref, a_memref)
        assert tile_type.memory_space == memory_space and tile_type.dtype == dtype
        assert len(tile_type.shape) == 2
        for dimension, extent in zip(tile_type.shape, [8, 32]):
            assert isinstance(dimension, ir.ConstInt) and dimension.value == extent
        assert tile_type.tile_view == view_type.tile_view
    new_function = next(iter(after.functions.values()))
    assert isinstance(new_function.body, ir.SeqStmts)
    returned = new_function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    assert returned.value[0].same_as(tiles["alias"])


@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_pypto_loop_carry_uses_shared_addr(enable_buffer_ir):
    optimized, _ = _run_pipeline(
        passes.MemoryPlanner.PYPTO, LoopCarriedAdd, enable_buffer_ir=enable_buffer_ir
    )
    mlir = _codegen(optimized, emit_tile_addr=True)
    alloc_lines = [ln for ln in mlir.splitlines() if "pto.alloc_tile" in ln]
    addrs = [ln.split("addr =")[1].split()[0] for ln in alloc_lines if "addr =" in ln]
    if enable_buffer_ir:
        assert len(addrs) == 2 and len(set(addrs)) == 2, mlir
        tadd = next(line for line in mlir.splitlines() if "pto.tadd" in line)
        inputs = tadd.split("ins(")[1].split(":")[0].split(",")
        output = tadd.split("outs(")[1].split(":")[0].strip()
        assert output in [value.strip() for value in inputs], tadd
    else:
        assert len(addrs) != len(set(addrs)), f"Tile IR must alias the carry via a shared addr:\n{mlir}"


def test_dsa_rp_loop_carry_is_in_place_with_resolved_address():
    optimized, _ = _run_pipeline(passes.MemoryPlanner.DSA_RP, LoopCarriedAdd)
    mlir = _codegen(optimized, emit_tile_addr=True)
    tadd = next((ln for ln in mlir.splitlines() if "pto.tadd" in ln), None)
    assert tadd is not None, f"expected a pto.tadd:\n{mlir}"
    ins = tadd.split("ins(")[1].split(")")[0]
    out = tadd.split("outs(")[1].split(")")[0]
    out_handle = out.split(":")[0].strip()
    in_handles = [tok.strip() for tok in ins.split(":")[0].split(",")]
    alloc_lines = [ln for ln in mlir.splitlines() if "pto.alloc_tile" in ln]
    assert alloc_lines and all("addr =" in ln for ln in alloc_lines)
    addr_by_handle = {line.split("=")[0].strip(): line.split("addr =")[1].split()[0] for line in alloc_lines}
    assert out_handle in addr_by_handle
    assert any(
        handle in addr_by_handle and addr_by_handle[handle] == addr_by_handle[out_handle]
        for handle in in_handles
    ), f"DSA_RP must place the loop-carried input and output at one physical address:\n{mlir}"


@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_dsa_rp_consecutive_loop_return_keeps_carried_buffer_live(enable_buffer_ir):
    """A later load must not overwrite a carry returned through two loops."""

    optimized, _ = _run_pipeline(
        passes.MemoryPlanner.DSA_RP, ConsecutiveLoopCarriedAdd, enable_buffer_ir=enable_buffer_ir
    )
    mlir = _codegen(optimized, emit_tile_addr=True)
    alloc_lines = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    # Buffer lowering names handles after allocation roots rather than source
    # tiles. Follow the two loads' real destination operands in either IR mode.
    loads = [line for line in mlir.splitlines() if "pto.tload " in line]
    assert len(loads) == 2, mlir
    load_handles = [line.split("outs(")[1].split(":")[0].strip() for line in loads]
    constants = dict(re.findall(r"(%\w+) = arith.constant (\d+) : i64", mlir))
    addresses = {
        line.split("=")[0].strip(): int(constants[line.split("addr =")[1].split()[0]]) for line in alloc_lines
    }
    seed_addr, residual_addr = (addresses[handle] for handle in load_handles)
    assert abs(seed_addr - residual_addr) >= 64 * 64 * 4, (
        f"the load after the second loop must not clobber its still-live carried result:\n{mlir}"
    )


@pytest.mark.parametrize(
    "planner",
    [
        passes.MemoryPlanner.PYPTO,
        passes.MemoryPlanner.DSA_RP,
        passes.MemoryPlanner.PTOAS,
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_stationary_matmul_loop_allocates_with_one_lhs_load(planner, enabled):
    """All planners accept the extended Mat/L0A lifetime and emit one LHS load."""
    optimized, _ = _run_pipeline(planner, StationaryMatmulLoop, enable_buffer_ir=enabled)
    calls = _calls_with_loop_depth(optimized)
    load_name = ir.get_op("buffer.load" if enabled else "tile.load").name
    copy_name = ir.get_op("buffer.copy" if enabled else "tile.move").name
    lhs_loads = [
        (call, depth)
        for call, depth in calls
        if call.op.name == load_name
        and isinstance(call.args[0], ir.Var)
        and call.args[0].name_hint.startswith("lhs")
    ]
    rhs_loads = [
        (call, depth)
        for call, depth in calls
        if call.op.name == load_name
        and isinstance(call.args[0], ir.Var)
        and call.args[0].name_hint.startswith("rhs")
    ]
    entry_copies = [call for call, depth in calls if call.op.name == copy_name and depth == 0]
    assert len(lhs_loads) == len(rhs_loads) == len(entry_copies) == 1
    assert lhs_loads[0][1] == 0 and rhs_loads[0][1] > 0

    # Codegen is the end-to-end allocation gate. PyPTO must emit resolved
    # addresses; PTOAS must retain auto-placement allocs for PlanMemory.
    emits_addresses = planner != passes.MemoryPlanner.PTOAS
    mlir = _codegen(optimized, emit_tile_addr=emits_addresses)
    alloc_lines = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert alloc_lines
    if emits_addresses:
        assert all("addr =" in line for line in alloc_lines)
    else:
        assert all("addr =" not in line for line in alloc_lines)


@pytest.mark.parametrize(
    "planner",
    [
        passes.MemoryPlanner.PYPTO,
        passes.MemoryPlanner.DSA_RP,
        passes.MemoryPlanner.PTOAS,
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_nested_autotile_pipeline_allocates_safely(planner, enabled):
    """Resident Mat panel plus loop-local ping/pong allocates under either planner."""
    optimized, _ = _run_pipeline(planner, StationaryMatmulPipelinedK, enable_buffer_ir=enabled)
    calls = _calls_with_loop_depth(optimized)
    load_name = ir.get_op("buffer.load" if enabled else "tile.load").name
    extract_name = ir.get_op("buffer.extract" if enabled else "tile.extract").name
    lhs_loads = [
        (call, depth)
        for call, depth in calls
        if call.op.name == load_name
        and isinstance(call.args[0], ir.Var)
        and call.args[0].name_hint.startswith("lhs")
    ]
    lhs_extracts = [
        (call, depth)
        for call, depth in calls
        if call.op.name == extract_name
        and (call.args[-1].type if enabled else call.type).memory_space == ir.MemorySpace.Left
    ]
    assert len(lhs_loads) == 1 and lhs_loads[0][1] == 0
    assert len(lhs_extracts) == 2 and all(depth >= 2 for _, depth in lhs_extracts)

    emits_addresses = planner != passes.MemoryPlanner.PTOAS
    mlir = _codegen(optimized, emit_tile_addr=emits_addresses)
    alloc_lines = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    assert alloc_lines
    if emits_addresses:
        assert all("addr =" in line for line in alloc_lines)
    else:
        assert all("addr =" not in line for line in alloc_lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
