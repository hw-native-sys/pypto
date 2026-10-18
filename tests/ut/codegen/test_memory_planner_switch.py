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


def _collect_mat_phi_storage(program: ir.Program):
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


def _assert_same_mat_window(actual: ir.MemRef, expected: ir.MemRef):
    assert ir.MemRef.same_allocation(actual, expected)
    assert ir.structural_equal(actual.byte_offset_, expected.byte_offset_)
    assert actual.size_ == expected.size_


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
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        initialized = passes.init_mem_ref()(passes.convert_to_ssa()(ConditionalGather))
        after = passes.materialize_semantic_aliases()(initialized)

    phis, tiles = _collect_mat_phi_storage(after)
    assert len(phis) == 2
    base = next(var for name, var in tiles.items() if name.startswith("base")).type.memref
    assert base is not None
    for phi, arms in phis:
        assert phi.type.memref is not None
        _assert_same_mat_window(phi.type.memref, base)
        for arm in arms:
            assert arm is not None
            _assert_same_mat_window(arm, base)
    transposed = next(var for name, var in tiles.items() if name.startswith("transposed"))
    _assert_same_mat_window(transposed.type.memref, base)


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


@pytest.mark.parametrize(
    "difference", ["same_window", "offset", "size", "slot", "allocation", "memory_space"]
)
def test_ptoas_if_alias_fixup_only_coalesces_same_mat_window(difference):
    """Only matching Mat windows are normalized; other yields retain PTOAS handling."""
    storage = pl.MemRef(slots=2)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            flag: pl.Scalar[pl.INT32],
        ) -> pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat]:
            a: pl.Tile[[16, 16], pl.FP16, storage[0], pl.Mem.Mat] = pl.tile.load(
                x, [0, 0], [16, 16], target_memory=pl.Mem.Mat
            )
            b = pl.tile.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            _destination = pl.tile.create([16, 16], dtype=pl.FP16, target_memory=pl.Mem.Mat)
            if flag != 0:
                merged = pl.yield_(a)
            else:
                merged = pl.yield_(b)
            alias = pl.tile.reshape(merged, [16, 16])
            return alias

    program = Before
    if difference == "memory_space":

        @pl.program
        class VectorBefore:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 16], pl.FP16],
                flag: pl.Scalar[pl.INT32],
            ) -> pl.Tile[[16, 16], pl.FP16, pl.Mem.Vec]:
                a = pl.tile.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
                b = pl.tile.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
                _destination = pl.tile.create([16, 16], dtype=pl.FP16, target_memory=pl.Mem.Vec)
                if flag != 0:
                    merged = pl.yield_(a)
                else:
                    merged = pl.yield_(b)
                alias = pl.tile.reshape(merged, [16, 16])
                return alias

        program = VectorBefore

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        initialized = passes.init_mem_ref()(program)
        phis, tiles = _collect_mat_phi_storage(initialized)
        phi = phis[0][0]
        a_memref = tiles["a"].type.memref
        b_memref = tiles["b"].type.memref
        destination = tiles["_destination"].type.memref
        assert a_memref is not None and b_memref is not None and destination is not None
        if difference == "offset":
            b_memref = ir.MemRef(
                a_memref.base_,
                ir.ConstInt(a_memref.size_, pl.INDEX, a_memref.span),
                a_memref.size_,
                slots=a_memref.slot_count_,
                slot=a_memref.slot_index_,
            )
        elif difference == "size":
            b_memref = ir.MemRef(
                a_memref.base_,
                a_memref.byte_offset_,
                2 * a_memref.size_,
                slots=a_memref.slot_count_,
                slot=a_memref.slot_index_,
            )
        elif difference == "slot":
            # Equal byte offsets alone do not prove equal pipeline slot metadata.
            b_memref = ir.MemRef(
                a_memref.base_,
                a_memref.byte_offset_,
                a_memref.size_,
                slots=a_memref.slot_count_,
                slot=ir.ConstInt(1, pl.INDEX, a_memref.span),
            )
        elif difference in ("same_window", "memory_space"):
            b_memref = a_memref

        # InitMemRef initially inherits the then yield. Give the phi an independent
        # existing destination so an accidental coalesce becomes observable. Both
        # offset/size variants fit in the declared two-slot allocation.
        replacements = {}
        for old, memref in ((tiles["b"], b_memref), (phi, destination), (tiles["alias"], destination)):
            tile_type = old.type
            replacements[old.unique_id] = ir.Var(
                old.name_hint,
                ir.TileType(
                    tile_type.shape,
                    tile_type.dtype,
                    memref,
                    tile_type.tile_view,
                    tile_type.memory_space,
                ),
                old.span,
            )

        class SetWindows(ir.IRMutator):
            def visit_var(self, op):
                return replacements.get(op.unique_id, op)

        before = SetWindows().visit_program(initialized)
        after = passes.materialize_semantic_aliases()(before)

    if difference == "same_window":
        phis, tiles = _collect_mat_phi_storage(after)
        phi, arms = phis[0]
        _assert_same_mat_window(phi.type.memref, a_memref)
        for arm in arms:
            _assert_same_mat_window(arm, a_memref)
        _assert_same_mat_window(tiles["alias"].type.memref, a_memref)
        function = next(iter(after.functions.values()))
        assert isinstance(function.body, ir.SeqStmts)
        returned = function.body.stmts[-1]
        assert isinstance(returned, ir.ReturnStmt)
        assert returned.value[0].same_as(tiles["alias"])
    else:
        # The PTOAS false switch still leaves other branch storage to codegen.
        ir.assert_structural_equal(after, before)


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
