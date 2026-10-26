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

import re

import pypto.language as pl
import pytest
from pypto import DataType, InternalError, backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes

_STORAGE_PLANNER_MODES = [(planner, True) for planner in passes.MemoryPlanner.__members__.values()] + [
    (passes.MemoryPlanner.DSA_RP, False)
]


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
    instruments: tuple[passes.PassInstrument, ...] = (),
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext(
        list(instruments), memory_planner=memory_planner, enable_buffer_ir=enable_buffer_ir
    ):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)


def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)


def _run_pipeline_storage_stages(planner, program, *, enable_buffer_ir):
    """Retain Tile storage before lowering so the same assertions cover both IR modes."""
    stages: dict[str, ir.Program] = {}

    def after(pass_object, candidate):
        if pass_object.get_name() == "MaterializeSemanticAliases":
            stages["legalized"] = candidate

    def before(pass_object, candidate):
        if pass_object.get_name() == "LowerTileToBuffer":
            stages["planned"] = candidate

    instrument = passes.CallbackInstrument(before_pass=before, after_pass=after)
    optimized, _ = _run_pipeline(
        planner, program, enable_buffer_ir=enable_buffer_ir, instruments=(instrument,)
    )
    assert "legalized" in stages
    if enable_buffer_ir:
        assert "planned" in stages
    return stages["legalized"], stages.get("planned", optimized), optimized


def _tile_control_flow(program):
    class Collector(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.branches = []
            self.loops = []
            self.assignments = []

        def visit_if_stmt(self, op):
            self.branches.append(op)
            super().visit_if_stmt(op)

        def visit_for_stmt(self, op):
            self.loops.append(op)
            super().visit_for_stmt(op)

        def visit_while_stmt(self, op):
            self.loops.append(op)
            super().visit_while_stmt(op)

        def visit_assign_stmt(self, op):
            self.assignments.append(op)
            super().visit_assign_stmt(op)

    collector = Collector()
    for function in program.functions.values():
        collector.visit_stmt(function.body)
    return collector.branches, collector.loops, collector.assignments


def _tile_memory(value):
    tile_type = value.type
    assert isinstance(tile_type, ir.TileType) and tile_type.memref is not None
    return tile_type.memref


def _assert_disjoint_tile_windows(left, right, *, physical):
    left_memory, right_memory = _tile_memory(left), _tile_memory(right)
    assert not ir.MemRef.may_alias(left_memory, right_memory)
    if physical:
        assert isinstance(left_memory.byte_offset_, ir.ConstInt)
        assert isinstance(right_memory.byte_offset_, ir.ConstInt)
        assert (
            left_memory.byte_offset_.value + left_memory.size_ <= right_memory.byte_offset_.value
            or right_memory.byte_offset_.value + right_memory.size_ <= left_memory.byte_offset_.value
        ), "simultaneously live Vec values must occupy disjoint physical windows"


def _assert_tile_valid_rows(value, rows):
    tile_type = value.type
    assert isinstance(tile_type, ir.TileType)
    assert [dim.value for dim in tile_type.shape if isinstance(dim, ir.ConstInt)] == [16, 32]
    view = tile_type.tile_view
    valid = view.valid_shape if view is not None and view.valid_shape else tile_type.shape
    assert [dim.value for dim in valid if isinstance(dim, ir.ConstInt)] == [rows, 32]


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


@pytest.mark.parametrize("yield_aliases", [False, True], ids=["same_value", "same_descriptor_aliases"])
def test_dsa_rp_legacy_duplicate_if_slots_share_one_reconciling_transfer(yield_aliases):
    """Two result slots receiving the same value need one write to their shared destination."""
    aliases = "        a_alias = a\n        b_alias = b\n" if yield_aliases else ""
    before = pl.parse_program(f"""
@pl.program
class DuplicateBranchResults:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 32], pl.FP32], flag: pl.Scalar[pl.BOOL],
               output: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        a = pl.tile.load(x, [0, 0], [16, 32], target_memory=pl.Mem.Vec)
        b = pl.tile.load(x, [16, 0], [16, 32], target_memory=pl.Mem.Vec)
{aliases}        if flag:
            left, right = pl.yield_(a, {"a_alias" if yield_aliases else "a"})
        else:
            left, right = pl.yield_(b, {"b_alias" if yield_aliases else "b"})
        left_alias = left
        right_alias = right
        combined = pl.tile.add(left_alias, right_alias)
        return pl.tile.store(combined, [0, 0], output)
""")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False):
        initialized = passes.init_mem_ref()(before)
        _, initial_tiles = _collect_tile_phi_storage(initialized)
        a_type, b_type = initial_tiles["a"].type, initial_tiles["b"].type
        assert isinstance(a_type, ir.TileType) and a_type.memref is not None
        assert isinstance(b_type, ir.TileType) and b_type.memref is not None
        assert not ir.MemRef.may_alias(a_type.memref, b_type.memref), "the arms must require reconciliation"
        legalized = passes.materialize_semantic_aliases()(initialized)

    function = next(iter(legalized.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    branch = next(statement for statement in function.body.stmts if isinstance(statement, ir.IfStmt))
    assert len(branch.return_vars) == 2
    destination_type = branch.return_vars[0].type
    assert isinstance(destination_type, ir.TileType) and destination_type.memref is not None
    destination = destination_type.memref
    for result in branch.return_vars:
        result_type = result.type
        assert isinstance(result_type, ir.TileType) and result_type.memref is not None
        _assert_same_tile_window(result_type.memref, destination)
    for body in (branch.then_body, branch.else_body):
        assert body is not None
        yielded = ir.get_last_yield_stmt(body)
        assert yielded is not None and len(yielded.value) == 2
        for value in yielded.value:
            value_type = value.type
            assert isinstance(value_type, ir.TileType) and value_type.memref is not None
            _assert_same_tile_window(value_type.memref, destination)

    moves = [
        call for call, _ in _calls_with_loop_depth(legalized) if call.op.name == ir.get_op("tile.move").name
    ]
    assert len(moves) == 1, "duplicate result slots must not emit duplicate writes to their shared window"
    _, tiles = _collect_tile_phi_storage(legalized)
    for name in ("left_alias", "right_alias"):
        alias_type = tiles[name].type
        assert isinstance(alias_type, ir.TileType) and alias_type.memref is not None
        _assert_same_tile_window(alias_type.memref, destination)

    optimized, pass_names = _run_pipeline(passes.MemoryPlanner.DSA_RP, before, enable_buffer_ir=False)
    assert "MemoryReuse" not in pass_names and "AllocateMemoryAddr" in pass_names
    mlir = _codegen(optimized, emit_tile_addr=True)
    assert len([line for line in mlir.splitlines() if "pto.tmov" in line]) == 1, mlir
    assert "pto.tadd" in mlir, "both result slots must remain live at their common consumer"


def test_if_yielding_acc_iter_arg_resolves_its_request_without_an_acc_copy():
    """An independent result request must resolve to the unchanged carry without an Acc-to-Acc copy."""
    before = pl.parse_program("""
@pl.program
class IdentityCarryBranch:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, initial: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc],
               flag: pl.Scalar[pl.BOOL]) -> pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc]:
        for index, (carried,) in pl.range(2, init_values=(initial,)):
            if flag:
                selected = pl.yield_(carried)
            else:
                selected = pl.yield_(carried)
            observed = pl.tile.move(selected, target_memory=pl.Mem.Vec)
            result = pl.yield_(carried)
        return result
""")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False):
        initialized = passes.init_mem_ref()(before)
        legalized = passes.materialize_semantic_aliases()(initialized)
    for program, materialized in ((initialized, False), (legalized, True)):
        branches, loops, assignments = _tile_control_flow(program)
        assert len(branches) == 1 and len(loops) == 1
        loop, branch = loops[0], branches[0]
        carry = _tile_memory(loop.iter_args[0])
        _assert_same_tile_window(_tile_memory(loop.iter_args[0].initValue), carry)
        if materialized:
            _assert_same_tile_window(_tile_memory(branch.return_vars[0]), carry)
        else:
            _assert_disjoint_tile_windows(branch.return_vars[0], loop.iter_args[0], physical=False)
        for body in (branch.then_body, branch.else_body):
            assert body is not None
            edge = ir.get_last_yield_stmt(body)
            assert edge is not None and isinstance(edge.value[0], ir.IterArg)
            _assert_same_tile_window(_tile_memory(edge.value[0]), carry)
        loop_yield = ir.get_last_yield_stmt(loop.body)
        assert loop_yield is not None and isinstance(loop_yield.value[0], ir.IterArg)
        _assert_same_tile_window(_tile_memory(loop_yield.value[0]), carry)
        moves = [
            op
            for op in assignments
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.move").name
        ]
        assert len(moves) == 1, "only the explicit Acc-to-Vec observation needs a transfer"
        move = moves[0]
        assert isinstance(move.value, ir.Call)
        _assert_same_tile_window(_tile_memory(move.value.args[0]), _tile_memory(branch.return_vars[0]))
        assert isinstance(move.var.type, ir.TileType) and move.var.type.memory_space == ir.MemorySpace.Vec


@pytest.mark.parametrize("planner", list(passes.MemoryPlanner.__members__.values()))
@pytest.mark.parametrize("short_first", [False, True], ids=["full_short", "short_full"])
@pytest.mark.parametrize("readonly_forward", [False, True], ids=["direct", "readonly_forward"])
@pytest.mark.parametrize("observe_short", [False, True], ids=["mutable_short", "isolated_loop_input"])
def test_buffer_ir_branch_local_typed_results_preserve_full_value_across_short_loop(
    planner, short_first, readonly_forward, observe_short
):
    """Retargeting a local producer must preserve the other result descriptor's storage."""
    names = "short, full" if short_first else "full, short"
    then_values = "a_short, a" if short_first else "a, a_short"
    else_values = "b_short, b" if short_first else "b, b_short"
    forwarding = (
        """        for forward_index, (unchanged,) in pl.range(forward_count, init_values=(short,)):
            forwarded = pl.yield_(unchanged)
"""
        if readonly_forward
        else ""
    )
    seed = "forwarded" if readonly_forward else "short"
    output_rows = 48 if observe_short else 32
    observation = "        old_saved = pl.tile.store(short, [32, 0], saved)\n" if observe_short else ""
    saved = "old_saved" if observe_short else "saved"
    before = pl.parse_program(f"""
@pl.program
class LocalTypedBranchResults:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 32], pl.FP32], flag: pl.Scalar[pl.BOOL],
               forward_count: pl.Scalar[pl.INDEX],
               output: pl.Out[pl.Tensor[[{output_rows}, 32], pl.FP32]]
               ) -> pl.Tensor[[{output_rows}, 32], pl.FP32]:
        if flag:
            a = pl.tile.load(x, [0, 0], [16, 32], target_memory=pl.Mem.Vec)
            a_short = pl.tile.set_validshape(a, 8, 32)
            {names} = pl.yield_({then_values})
        else:
            b = pl.tile.load(x, [16, 0], [16, 32], target_memory=pl.Mem.Vec)
            b_short = pl.tile.set_validshape(b, 8, 32)
            {names} = pl.yield_({else_values})
{forwarding}        for index, (carried,) in pl.range(2, init_values=({seed},)):
            updated = pl.tile.add(carried, 1.0)
            loop_value = pl.yield_(updated)
        saved = pl.tile.store(full, [0, 0], output)
{observation}        return pl.tile.store(loop_value, [16, 0], {saved})
""")
    legalized, planned, optimized = _run_pipeline_storage_stages(planner, before, enable_buffer_ir=True)
    kernel = next(function for function in planned.functions.values() if function.name == "kernel")
    selected = bool(kernel.attrs.get("buffer_ir_selected", False))
    for program, physical, after_planning in (
        (legalized, False, False),
        (planned, planner != passes.MemoryPlanner.PTOAS, True),
    ):
        branches, loops, assignments = _tile_control_flow(program)
        assert len(branches) == 1 and len(loops) == 1 + readonly_forward
        results = branches[0].return_vars
        short, full = (results[0], results[1]) if short_first else (results[1], results[0])
        writers = [
            op.var
            for op in assignments
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.adds").name
        ]
        assert len(writers) == 1
        loop = loops[-1]
        yielded = ir.get_last_yield_stmt(loop.body)
        assert yielded is not None
        _assert_tile_valid_rows(full, 16)
        _assert_tile_valid_rows(short, 8)
        mutable_values = (loop.iter_args[0], writers[0], yielded.value[0], loop.return_vars[0])
        for value in mutable_values:
            _assert_tile_valid_rows(value, 8)
            _assert_disjoint_tile_windows(full, value, physical=physical)
        if observe_short:
            # Entry isolation is a real copy, so its later writes must not mark
            # the original readonly result mutable through an SSA-only link.
            _assert_same_tile_window(_tile_memory(full), _tile_memory(short))
            for value in mutable_values:
                _assert_disjoint_tile_windows(short, value, physical=physical)
            entry = next(op for op in assignments if op.var.same_as(loop.iter_args[0].initValue))
            assert isinstance(entry.value, ir.Call) and entry.value.op.name == ir.get_op("tile.move").name
            _assert_same_tile_window(_tile_memory(entry.value.args[0]), _tile_memory(short))
        else:
            # A readonly forwarding loop does not stop a *later* mutation from
            # making the short result mutable; its peer full stays independent.
            _assert_disjoint_tile_windows(full, short, physical=physical)
            _assert_same_tile_window(_tile_memory(loop.iter_args[0]), _tile_memory(short))
        if readonly_forward:
            forwarding_loop = loops[0]
            forwarding_yield = ir.get_last_yield_stmt(forwarding_loop.body)
            assert forwarding_yield is not None
            for value in (
                forwarding_loop.iter_args[0],
                forwarding_yield.value[0],
                forwarding_loop.return_vars[0],
            ):
                _assert_same_tile_window(_tile_memory(value), _tile_memory(short))
        _assert_same_tile_window(_tile_memory(yielded.value[0]), _tile_memory(loop.return_vars[0]))
        owns_edges = (
            selected
            or planner == passes.MemoryPlanner.DSA_RP
            or (after_planning and planner == passes.MemoryPlanner.PYPTO)
        )
        if owns_edges:
            for body in (branches[0].then_body, branches[0].else_body):
                assert body is not None
                edge = ir.get_last_yield_stmt(body)
                assert edge is not None
                for result, value in zip(results, edge.value, strict=True):
                    _assert_same_tile_window(_tile_memory(value), _tile_memory(result))
    mlir = _codegen(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    assert "pto.tadds" in mlir, mlir
    assert len([line for line in mlir.splitlines() if "pto.tstore" in line]) == 2 + observe_short, mlir


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    _STORAGE_PLANNER_MODES,
)
@pytest.mark.parametrize("swap_arms", [False, True], ids=["repeated_then", "repeated_else"])
def test_if_duplicate_incoming_slot_splits_when_other_arm_values_differ(planner, enable_buffer_ir, swap_arms):
    """Repeated incoming values need independent results while their original value remains observable."""
    then_values, else_values = ("b, c", "a, a") if swap_arms else ("a, a", "b, c")
    before = pl.parse_program(f"""
@pl.program
class PartiallyDuplicateBranchResults:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[48, 32], pl.FP32], flag: pl.Scalar[pl.BOOL],
               output: pl.Out[pl.Tensor[[48, 32], pl.FP32]]) -> pl.Tensor[[48, 32], pl.FP32]:
        a = pl.tile.load(x, [0, 0], [16, 32], target_memory=pl.Mem.Vec)
        b = pl.tile.load(x, [16, 0], [16, 32], target_memory=pl.Mem.Vec)
        c = pl.tile.load(x, [32, 0], [16, 32], target_memory=pl.Mem.Vec)
        if flag:
            first, second = pl.yield_({then_values})
        else:
            first, second = pl.yield_({else_values})
        saved = pl.tile.store(first, [0, 0], output)
        saved_second = pl.tile.store(second, [16, 0], saved)
        return pl.tile.store(a, [32, 0], saved_second)
""")
    legalized, planned, optimized = _run_pipeline_storage_stages(
        planner, before, enable_buffer_ir=enable_buffer_ir
    )
    for program, physical in ((legalized, False), (planned, planner != passes.MemoryPlanner.PTOAS)):
        branches, loops, _ = _tile_control_flow(program)
        assert len(branches) == 1 and not loops
        first, second = branches[0].return_vars
        _assert_disjoint_tile_windows(first, second, physical=physical)
        stores = [
            call
            for call, _ in _calls_with_loop_depth(program)
            if call.op.name == ir.get_op("tile.store").name
        ]
        assert len(stores) == 3
        original_a = stores[2].args[0]
        _assert_tile_valid_rows(original_a, 16)
        for value in (first, second):
            _assert_tile_valid_rows(value, 16)
            _assert_disjoint_tile_windows(original_a, value, physical=physical)
        for body in (branches[0].then_body, branches[0].else_body):
            assert body is not None
            edge = ir.get_last_yield_stmt(body)
            assert edge is not None
            for result, value in zip(branches[0].return_vars, edge.value, strict=True):
                _assert_same_tile_window(_tile_memory(value), _tile_memory(result))
    mlir = _codegen(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    assert len([line for line in mlir.splitlines() if "pto.tstore" in line]) == 3, mlir


@pytest.mark.parametrize("planner,enable_buffer_ir", _STORAGE_PLANNER_MODES)
@pytest.mark.parametrize("swap_arms", [False, True], ids=["views_then", "views_else"])
def test_if_overlapping_inherited_views_receive_independent_result_windows(
    planner, enable_buffer_ir, swap_arms
):
    """Inherited slice windows may overlap, while independently live results need separate storage."""
    then_values, else_values = ("b, c", "left, right") if swap_arms else ("left, right", "b, c")
    before = pl.parse_program(f"""
@pl.program
class OverlappingIncomingViews:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[1, 128], pl.FP32], flag: pl.Scalar[pl.BOOL],
               output: pl.Out[pl.Tensor[[3, 128], pl.FP32]]) -> pl.Tensor[[3, 128], pl.FP32]:
        whole = pl.tile.load(x, [0, 0], [1, 128], target_memory=pl.Mem.Vec)
        left = pl.tile.slice(whole, [1, 64], [0, 0])
        right = pl.tile.slice(whole, [1, 64], [0, 32])
        b = pl.tile.load(x, [0, 0], [1, 64], target_memory=pl.Mem.Vec)
        c = pl.tile.load(x, [0, 64], [1, 64], target_memory=pl.Mem.Vec)
        if flag:
            first, second = pl.yield_({then_values})
        else:
            first, second = pl.yield_({else_values})
        saved = pl.tile.store(first, [0, 0], output)
        saved_second = pl.tile.store(second, [1, 0], saved)
        return pl.tile.store(whole, [2, 0], saved_second)
""")
    legalized, planned, optimized = _run_pipeline_storage_stages(
        planner, before, enable_buffer_ir=enable_buffer_ir
    )
    for program, physical in (
        (legalized, False),
        (planned, planner != passes.MemoryPlanner.PTOAS),
    ):
        branches, loops, assignments = _tile_control_flow(program)
        assert len(branches) == 1 and not loops
        # Every planner consumes the same explicit If-edge transfers, including
        # legacy Tile PTOAS when window yields prevent Buffer IR selection.
        stores = [
            op.value
            for op in assignments
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.store").name
        ]
        assert len(stores) == 3
        whole = stores[2].args[0]
        first, second = branches[0].return_vars
        _assert_disjoint_tile_windows(first, second, physical=physical)
        extracts = [
            op
            for op in assignments
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.extract").name
        ]
        for result in (first, second):
            _assert_disjoint_tile_windows(whole, result, physical=physical)
            assert any(
                ir.MemRef.same_allocation(_tile_memory(op.var), _tile_memory(result))
                and ir.structural_equal(_tile_memory(op.var).byte_offset_, _tile_memory(result).byte_offset_)
                and _tile_memory(op.var).size_ == _tile_memory(result).size_
                for op in extracts
            ), "each incoming view needs a window transfer into its result destination"
        for extract in extracts:
            assert isinstance(extract.value, ir.Call)
            source_type = extract.value.args[0].type
            assert isinstance(source_type, ir.TileType)
            assert [dim.value for dim in source_type.shape if isinstance(dim, ir.ConstInt)] == [1, 128]
        for body in (branches[0].then_body, branches[0].else_body):
            assert body is not None
            edge = ir.get_last_yield_stmt(body)
            assert edge is not None
            for result, value in zip(branches[0].return_vars, edge.value, strict=True):
                _assert_same_tile_window(_tile_memory(value), _tile_memory(result))
    mlir = _codegen(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    lines = mlir.splitlines()
    stores = [line for line in lines if "pto.tstore" in line]
    assert len(stores) == 3, mlir
    extracts = [line for line in lines if "pto.textract" in line]
    assert len(extracts) == 2, mlir
    allocations = {line.split("=", 1)[0].strip(): line for line in lines if "= pto.alloc_tile" in line}

    def storage_key(handle):
        allocation = allocations[handle]
        return allocation.split("addr =", 1)[1].split()[0] if "addr =" in allocation else handle

    stored = [line.split("ins(", 1)[1].split(" :", 1)[0].strip() for line in stores]
    assert storage_key(stored[0]) != storage_key(stored[1]), mlir
    assert all(storage_key(value) != storage_key(stored[2]) for value in stored[:2]), mlir
    constants = {
        name: int(value)
        for name, value in re.findall(r"(%[A-Za-z0-9_]+) = arith.constant (\d+) : index", mlir)
    }
    if_index = next(i for i, line in enumerate(lines) if "scf.if " in line)
    else_index = next(i for i in range(if_index + 1, len(lines)) if lines[i].strip() == "} else {")
    for transfer, output, column in zip(extracts, stored[:2], (0, 32), strict=True):
        operands = [operand.strip() for operand in transfer.split("ins(", 1)[1].split(" :", 1)[0].split(",")]
        destination = transfer.split("outs(", 1)[1].split(" :", 1)[0].strip()
        assert operands[0] == stored[2], "extract must read the preserved full allocation"
        assert [constants[operand] for operand in operands[1:]] == [0, column], transfer
        assert storage_key(destination) == storage_key(output), transfer
        transfer_index = lines.index(transfer)
        assert lines.index(allocations[output]) < if_index, "merged storage must dominate both arms"
        assert lines.index(allocations[destination]) < transfer_index, transfer
        assert (transfer_index > else_index) if swap_arms else (if_index < transfer_index < else_index)


@pytest.mark.parametrize("planner,enable_buffer_ir", _STORAGE_PLANNER_MODES)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
def test_nonzero_offset_loop_carry_move_preserves_the_full_allocation(planner, enable_buffer_ir, while_loop):
    """A carried subwindow is not the whole allocation when deciding whether a move is redundant."""
    header = (
        "for (carried, index) in pl.while_(init_values=(window, 0)):\n            pl.cond(index < count)"
        if while_loop
        else "for index, (carried,) in pl.range(count, init_values=(window,)):"
    )
    loop_yield = (
        "result, _counter = pl.yield_(carried, index + 1)" if while_loop else "result = pl.yield_(carried)"
    )
    before = pl.parse_program(f"""
@pl.program
class CarriedOffsetWindow:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[1, 128], pl.FP32], count: pl.Scalar[pl.INDEX],
               output: pl.Out[pl.Tensor[[2, 128], pl.FP32]]) -> pl.Tensor[[2, 128], pl.FP32]:
        whole = pl.tile.load(x, [0, 0], [1, 128], target_memory=pl.Mem.Vec)
        window = pl.tile.slice(whole, [1, 64], [0, 32])
        saved = pl.tile.store(whole, [1, 0], output)
        {header}
            {loop_yield}
        copied = pl.tile.move(result, target_memory=pl.Mem.Vec)
        return pl.tile.store(copied, [0, 0], saved)
""")
    legalized, planned, optimized = _run_pipeline_storage_stages(
        planner, before, enable_buffer_ir=enable_buffer_ir
    )
    for program, physical in ((legalized, False), (planned, planner != passes.MemoryPlanner.PTOAS)):
        branches, loops, assignments = _tile_control_flow(program)
        assert not branches and len(loops) == 1
        stores = [
            op.value
            for op in assignments
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.store").name
        ]
        assert len(stores) == 2
        whole, copied = stores[0].args[0], stores[1].args[0]
        loop = loops[0]
        _assert_same_tile_window(_tile_memory(loop.return_vars[0]), _tile_memory(loop.iter_args[0]))
        assert ir.MemRef.may_alias(_tile_memory(whole), _tile_memory(loop.return_vars[0]))
        assert _tile_memory(loop.return_vars[0]).size_ < _tile_memory(whole).size_
        assert not ir.structural_equal(
            _tile_memory(loop.return_vars[0]).byte_offset_, _tile_memory(whole).byte_offset_
        ), "the carried value must remain a nonzero-offset view of the observed full allocation"
        _assert_disjoint_tile_windows(whole, copied, physical=physical)
        _assert_disjoint_tile_windows(loop.return_vars[0], copied, physical=physical)
        transfers = [
            op
            for op in assignments
            if isinstance(op.value, ir.Call)
            and op.value.op.name in {ir.get_op("tile.move").name, ir.get_op("tile.extract").name}
        ]
        assert transfers, "a nonzero-offset window needs a real transfer to the independent destination"
        assert any(
            ir.MemRef.same_allocation(_tile_memory(op.var), _tile_memory(copied))
            and ir.structural_equal(_tile_memory(op.var).byte_offset_, _tile_memory(copied).byte_offset_)
            and _tile_memory(op.var).size_ == _tile_memory(copied).size_
            for op in transfers
        )
    mlir = _codegen(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    assert "pto.textract" in mlir, mlir
    assert len([line for line in mlir.splitlines() if "pto.tstore" in line]) == 2, mlir


@pytest.mark.parametrize(
    "valid_rows",
    [(16, 8), (8, 16), (16, 8, 16), (8, 16, 8)],
    ids=["full_short", "short_full", "full_short_full", "short_full_short"],
)
def test_dsa_rp_legacy_duplicate_if_windows_preserve_each_result_descriptor(valid_rows):
    """Descriptor groups retain valid extents and selected data without redundant per-slot transfers."""
    result_names = [f"result_{i}" for i in range(len(valid_rows))]
    names = ", ".join(result_names)
    then_values = ", ".join("a" if rows == 16 else "a_short" for rows in valid_rows)
    else_values = ", ".join("b" if rows == 16 else "b_short" for rows in valid_rows)
    stores = "\n".join(
        f"        saved_{i} = pl.tile.store({name}, [{i * 16}, 0], "
        f"{'output' if i == 0 else f'saved_{i - 1}'})"
        for i, name in enumerate(result_names)
    )
    output_rows = len(valid_rows) * 16
    before = pl.parse_program(f"""
@pl.program
class TypedDuplicateResults:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 32], pl.FP32], flag: pl.Scalar[pl.BOOL],
               output: pl.Out[pl.Tensor[[{output_rows}, 32], pl.FP32]]) -> pl.Tensor[
                   [{output_rows}, 32], pl.FP32]:
        a = pl.tile.load(x, [0, 0], [16, 32], target_memory=pl.Mem.Vec)
        b = pl.tile.load(x, [16, 0], [16, 32], target_memory=pl.Mem.Vec)
        a_short = pl.tile.set_validshape(a, 8, 32)
        b_short = pl.tile.set_validshape(b, 8, 32)
        if flag:
            {names} = pl.yield_({then_values})
        else:
            {names} = pl.yield_({else_values})
{stores}
        return saved_{len(valid_rows) - 1}
""")

    def check_results(program, *, physical):
        branches: list[ir.IfStmt] = []

        class BranchCollector(ir.IRVisitor):
            def visit_if_stmt(self, op):
                branches.append(op)
                super().visit_if_stmt(op)

        for function in program.functions.values():
            BranchCollector().visit_stmt(function.body)
        assert len(branches) == 1
        branch = branches[0]
        assert len(branch.return_vars) == len(valid_rows)
        groups: dict[int, ir.MemRef] = {}
        for result, rows in zip(branch.return_vars, valid_rows, strict=True):
            tile_type = result.type
            assert isinstance(tile_type, ir.TileType) and tile_type.memref is not None
            assert len(tile_type.shape) == 2
            assert [dim.value for dim in tile_type.shape if isinstance(dim, ir.ConstInt)] == [16, 32]
            view = tile_type.tile_view
            valid = view.valid_shape if view is not None and view.valid_shape else tile_type.shape
            assert len(valid) == 2
            assert [dim.value for dim in valid if isinstance(dim, ir.ConstInt)] == [rows, 32]
            if rows in groups:
                _assert_same_tile_window(tile_type.memref, groups[rows])
            else:
                groups[rows] = tile_type.memref
        assert len(groups) == 2
        full, short = groups[16], groups[8]
        assert not ir.MemRef.may_alias(full, short), (
            "different result descriptors must not share a transfer target"
        )
        if physical:
            assert isinstance(full.byte_offset_, ir.ConstInt) and isinstance(short.byte_offset_, ir.ConstInt)
            assert (
                full.byte_offset_.value + full.size_ <= short.byte_offset_.value
                or short.byte_offset_.value + short.size_ <= full.byte_offset_.value
            ), "the simultaneously live typed results must occupy disjoint physical windows"
        for body in (branch.then_body, branch.else_body):
            assert body is not None
            yielded = ir.get_last_yield_stmt(body)
            assert yielded is not None and len(yielded.value) == len(valid_rows)
            for result, value in zip(branch.return_vars, yielded.value, strict=True):
                result_type, value_type = result.type, value.type
                assert isinstance(result_type, ir.TileType) and result_type.memref is not None
                assert isinstance(value_type, ir.TileType) and value_type.memref is not None
                _assert_same_tile_window(value_type.memref, result_type.memref)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False):
        legalized = passes.materialize_semantic_aliases()(passes.init_mem_ref()(before))
    check_results(legalized, physical=False)
    _, tiles = _collect_tile_phi_storage(legalized)

    def window(value):
        memory = _tile_memory(value)
        assert isinstance(memory.byte_offset_, ir.ConstInt)
        return memory.base_.unique_id, memory.byte_offset_.value, memory.size_

    origins = {window(tiles["a"]): "a", window(tiles["b"]): "b"}
    function = next(iter(legalized.functions.values()))
    branch = next(op for op in ir.flatten_to_stmts(function.body) if isinstance(op, ir.IfStmt))
    for body, expected in ((branch.then_body, "a"), (branch.else_body, "b")):
        assert body is not None
        memory = dict(origins)
        written = set()
        for op in ir.flatten_to_stmts(body):
            if not isinstance(op, ir.AssignStmt):
                continue
            assert isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.move").name
            source, target = window(op.value.args[0]), window(op.var)
            assert source in memory, "every transfer must read initialized incoming data"
            assert target not in written, "nonadjacent duplicate slots must share their transfer"
            written.add(target)
            memory[target] = memory[source]
        edge = ir.get_last_yield_stmt(body)
        assert edge is not None
        assert [memory[window(value)] for value in edge.value] == [expected] * len(valid_rows)
    allocated = {
        op.var.unique_id
        for op in ir.flatten_to_stmts(function.body)
        if isinstance(op, ir.AssignStmt)
        and isinstance(op.value, ir.Call)
        and op.value.op.name == ir.get_op("tile.alloc").name
    }
    assert allocated == {_tile_memory(value).base_.unique_id for value in tiles.values()}, (
        "every retained allocation must supply an input or a descriptor-group destination"
    )
    optimized, pass_names = _run_pipeline(passes.MemoryPlanner.DSA_RP, before, enable_buffer_ir=False)
    assert "MemoryReuse" not in pass_names and "AllocateMemoryAddr" in pass_names
    check_results(optimized, physical=True)
    mlir = _codegen(optimized, emit_tile_addr=True)
    assert len([line for line in mlir.splitlines() if "pto.tstore" in line]) == len(valid_rows), mlir


@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
def test_dsa_rp_legacy_typed_result_storage_follows_alias_loop_and_later_phi(while_loop):
    """Splitting storage must follow loop results through aliases, views, and later boundaries."""
    header = (
        "for (carried, index) in pl.while_(init_values=(short_alias, 0)):\n            pl.cond(index < 2)"
        if while_loop
        else "for index, (carried,) in pl.range(2, init_values=(short_alias,)):"
    )
    loop_yield = (
        "loop_value, _counter = pl.yield_(updated, index + 1)"
        if while_loop
        else "loop_value = pl.yield_(updated)"
    )
    before = pl.parse_program(f"""
@pl.program
class ComposedTypedResults:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[48, 32], pl.FP32], flag: pl.Scalar[pl.BOOL],
               other_flag: pl.Scalar[pl.BOOL],
               output: pl.Out[pl.Tensor[[48, 32], pl.FP32]]) -> pl.Tensor[[48, 32], pl.FP32]:
        a = pl.tile.load(x, [0, 0], [16, 32], target_memory=pl.Mem.Vec)
        b = pl.tile.load(x, [16, 0], [16, 32], target_memory=pl.Mem.Vec)
        c = pl.tile.load(x, [32, 0], [16, 32], target_memory=pl.Mem.Vec)
        a_short = pl.tile.set_validshape(a, 8, 32)
        b_short = pl.tile.set_validshape(b, 8, 32)
        c_short = pl.tile.set_validshape(c, 8, 32)
        if flag:
            full, short = pl.yield_(a, a_short)
        else:
            full, short = pl.yield_(b, b_short)
        short_alias = short
        {header}
            if other_flag:
                step_input = pl.yield_(carried)
            else:
                step_input = pl.yield_(c_short)
            updated = pl.tile.add(step_input, 1.0)
            {loop_yield}
        loop_alias = loop_value
        loop_view = pl.tile.reshape(loop_alias, [16, 32])
        if other_flag:
            chosen = pl.yield_(loop_view)
        else:
            chosen = pl.yield_(c_short)
        saved = pl.tile.store(full, [0, 0], output)
        observed = pl.tile.store(c, [16, 0], saved)
        return pl.tile.store(chosen, [32, 0], observed)
""")

    def memory(value):
        tile_type = value.type
        assert isinstance(tile_type, ir.TileType) and tile_type.memref is not None
        return tile_type.memref

    def check_composed_storage(program, *, physical=False):
        branches: list[ir.IfStmt] = []
        loops: list[ir.ForStmt | ir.WhileStmt] = []

        class BoundaryCollector(ir.IRVisitor):
            def visit_if_stmt(self, op):
                branches.append(op)
                super().visit_if_stmt(op)

            def visit_for_stmt(self, op):
                loops.append(op)
                super().visit_for_stmt(op)

            def visit_while_stmt(self, op):
                loops.append(op)
                super().visit_while_stmt(op)

        for function in program.functions.values():
            BoundaryCollector().visit_stmt(function.body)
        assert len(branches) == 3 and len(loops) == 1
        full, short = branches[0].return_vars
        short_destination = memory(short)
        assert not ir.MemRef.may_alias(memory(full), short_destination)
        _, _, assignments = _tile_control_flow(program)

        def definition(predicate, description):
            matches = [op.var for op in assignments if predicate(op.value)]
            assert len(matches) == 1, f"expected one {description}, found {len(matches)}"
            return matches[0]

        def aliases(value, source):
            return isinstance(value, (ir.Var, ir.IterArg)) and value.unique_id == source.unique_id

        def calls(value, name):
            return isinstance(value, ir.Call) and value.op.name == ir.get_op(name).name

        short_alias = definition(lambda value: aliases(value, short), "bare alias of the short phi")
        updated = definition(lambda value: calls(value, "tile.adds"), "loop scalar-add producer")
        stores = [call for call, _ in _calls_with_loop_depth(program) if calls(call, "tile.store")]
        assert len(stores) == 3
        assert isinstance(stores[1], ir.Call)
        original_c = stores[1].args[0]
        loop = loops[0]
        loop_alias = definition(
            lambda value: aliases(value, loop.return_vars[0]), "bare alias of the loop result"
        )
        # FoldNoOpReshape may turn this identity view into a same-source rename.
        loop_view = definition(
            lambda value: aliases(value, loop_alias)
            or (
                calls(value, "tile.reshape")
                and isinstance(value, ir.Call)
                and aliases(value.args[0], loop_alias)
            ),
            "view or folded alias of the loop-result alias",
        )
        yielded = ir.get_last_yield_stmt(loop.body)
        assert yielded is not None
        short_values = [
            short,
            short_alias,
            loop.iter_args[0].initValue,
            loop.iter_args[0],
            yielded.value[0],
            loop.return_vars[0],
            loop_alias,
            loop_view,
        ]
        for value in short_values:
            _assert_same_tile_window(memory(value), short_destination)
            tile_type = value.type
            assert isinstance(tile_type, ir.TileType) and tile_type.tile_view is not None
            valid = tile_type.tile_view.valid_shape
            assert len(valid) == 2
            assert [dim.value for dim in valid if isinstance(dim, ir.ConstInt)] == [8, 32]
        step_input, chosen = branches[1].return_vars[0], branches[2].return_vars[0]
        for value in (short, step_input, updated, chosen):
            for observer in (full, original_c):
                assert not ir.MemRef.may_alias(memory(value), memory(observer)), (
                    "loop and branch writes must preserve the independently observed full and c values"
                )
        if physical:
            for observer in (full, original_c):
                original = memory(observer)
                assert isinstance(original.byte_offset_, ir.ConstInt)
                for value in (short, step_input, updated, chosen):
                    other = memory(value)
                    assert isinstance(other.byte_offset_, ir.ConstInt)
                    assert (
                        original.byte_offset_.value + original.size_ <= other.byte_offset_.value
                        or other.byte_offset_.value + other.size_ <= original.byte_offset_.value
                    ), "loop and branch writes must not overlap either independently live old value"
        for branch in branches[1:]:
            target = memory(branch.return_vars[0])
            for body in (branch.then_body, branch.else_body):
                assert body is not None
                edge = ir.get_last_yield_stmt(body)
                assert edge is not None
                _assert_same_tile_window(memory(edge.value[0]), target)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False):
        legalized = passes.materialize_semantic_aliases()(passes.init_mem_ref()(before))
    check_composed_storage(legalized)
    optimized, pass_names = _run_pipeline(passes.MemoryPlanner.DSA_RP, before, enable_buffer_ir=False)
    assert "MemoryReuse" not in pass_names and "AllocateMemoryAddr" in pass_names
    check_composed_storage(optimized, physical=True)
    mlir = _codegen(optimized, emit_tile_addr=True)
    assert "pto.tadds" in mlir, mlir
    assert len([line for line in mlir.splitlines() if "pto.tstore" in line]) == 3, mlir


def test_dsa_rp_legacy_distinct_if_values_cannot_share_a_destination():
    """Transfer deduplication must still reject conflicting values targeting the same bytes."""
    span = ir.Span.unknown()
    a_base = ir.Var("a_base", ir.PtrType(), span)
    b_base = ir.Var("b_base", ir.PtrType(), span)
    destination_base = ir.Var("destination_base", ir.PtrType(), span)
    a_memref = ir.MemRef(a_base, 0, 2048, span)
    b_memref = ir.MemRef(b_base, 0, 2048, span)
    destination = ir.MemRef(destination_base, 0, 2048, span, is_pinned=True)
    a_type = ir.TileType([16, 32], DataType.FP32, a_memref, memory_space=ir.MemorySpace.Vec)
    b_type = ir.TileType([16, 32], DataType.FP32, b_memref, memory_space=ir.MemorySpace.Vec)
    result_type = ir.TileType([16, 32], DataType.FP32, destination, memory_space=ir.MemorySpace.Vec)
    a, b = ir.Var("a", a_type, span), ir.Var("b", b_type, span)
    left, right = ir.Var("left", result_type, span), ir.Var("right", result_type, span)
    flag = ir.Var("flag", ir.ScalarType(DataType.BOOL), span)
    branch = ir.IfStmt(flag, ir.YieldStmt([a, b], span), ir.YieldStmt([a, b], span), [left, right], span)
    body = ir.SeqStmts([branch, ir.ReturnStmt([left, right], span)], span)
    function = ir.Function(
        "kernel",
        [a_base, b_base, destination_base, a, b, flag],
        [result_type, result_type],
        body,
        span,
        type=ir.FunctionType.InCore,
    )
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False):
        with pytest.raises(ValueError, match="independent destination windows"):
            passes.materialize_semantic_aliases()(ir.Program([function], "ConflictingBranchResults", span))


def _nested_if_window_program(write_nested=False, *, constrained=False):
    span = ir.Span.unknown()
    shared = ir.Var("shared", ir.PtrType(), span)
    separate = ir.Var("separate", ir.PtrType(), span)
    windows = [
        ir.MemRef(shared, 0, 128, span, is_pinned=constrained),
        ir.MemRef(shared, 32, 32, span, is_pinned=constrained),
        ir.MemRef(shared, 96, 16, span, is_pinned=constrained),
    ]
    types = [
        ir.TileType([1, columns], DataType.FP32, window, memory_space=ir.MemorySpace.Vec)
        for columns, window in zip((32, 8, 4), windows, strict=True)
    ]
    source = ir.Var("source", ir.PtrType(), span)
    input_types = (
        [
            ir.TileType(
                [1, columns],
                DataType.FP32,
                ir.MemRef(source, offset, size, span),
                memory_space=ir.MemorySpace.Vec,
            )
            for columns, offset, size in ((32, 0, 128), (8, 128, 32), (4, 160, 16))
        ]
        if constrained
        else types
    )
    inputs = [ir.Var(f"input_{i}", tile_type, span) for i, tile_type in enumerate(input_types)]
    results = [ir.Var(f"result_{i}", tile_type, span) for i, tile_type in enumerate(types)]
    replacement_type = ir.TileType(
        [1, 4], DataType.FP32, ir.MemRef(separate, 0, 16, span), memory_space=ir.MemorySpace.Vec
    )
    replacement = ir.Var("replacement", replacement_type, span)
    flag = ir.Var("flag", ir.ScalarType(DataType.BOOL), span)
    then_values: list[ir.Expr] = [inputs[0], inputs[1], replacement if write_nested else inputs[2]]
    branch = ir.IfStmt(flag, ir.YieldStmt(then_values, span), ir.YieldStmt([*inputs], span), results, span)
    statements: list[ir.Stmt] = [branch, ir.ReturnStmt([*results], span)]
    body = ir.SeqStmts(statements, span)
    function = ir.Function(
        "kernel",
        [shared, separate, *([source] if constrained else []), *inputs, replacement, flag],
        types,
        body,
        span,
        type=ir.FunctionType.InCore,
    )
    return ir.Program([function], "NestedResultWindows", span)


@pytest.mark.parametrize("write_nested", [False, True], ids=["all_no_copy", "inherited_write"])
def test_dsa_rp_legacy_inherited_nested_if_windows_preserve_old_inputs(write_nested):
    """Readonly nested windows stay in place while the written result uses private storage."""
    before = _nested_if_window_program(write_nested)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(before)
    if not write_nested:
        ir.assert_structural_equal(after, before)
        return
    original = next(iter(before.functions.values()))
    rewritten = next(iter(after.functions.values()))
    old_inputs = [value for value in original.params if isinstance(value.type, ir.TileType)]

    def descriptor(value):
        tile_type = value.type
        assert isinstance(tile_type, ir.TileType)
        return ir.TileType(
            tile_type.shape, tile_type.dtype, None, tile_type.tile_view, tile_type.memory_space
        )

    for old, new in zip(original.params, rewritten.params, strict=True):
        if isinstance(old.type, ir.TileType):
            _assert_same_tile_window(_tile_memory(old), _tile_memory(new))
            ir.assert_structural_equal(descriptor(new), descriptor(old))
    branches, loops, _ = _tile_control_flow(after)
    assert len(branches) == 1 and not loops
    results = branches[0].return_vars
    assert len(results) == 3
    for result, old in zip(results[:2], old_inputs[:2], strict=True):
        _assert_same_tile_window(_tile_memory(result), _tile_memory(old))
        ir.assert_structural_equal(descriptor(result), descriptor(old))
    for old in old_inputs:
        assert not ir.MemRef.same_allocation(_tile_memory(results[2]), _tile_memory(old))
        _assert_disjoint_tile_windows(results[2], old, physical=False)
    for readonly in results[:2]:
        _assert_disjoint_tile_windows(results[2], readonly, physical=False)
    assert isinstance(original.body, ir.SeqStmts)
    original_branch = next(stmt for stmt in original.body.stmts if isinstance(stmt, ir.IfStmt))

    class CopyCollector(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.copies = []

        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.move").name:
                self.copies.append(op)
            super().visit_assign_stmt(op)

    for body, old_body in zip(
        (branches[0].then_body, branches[0].else_body),
        (original_branch.then_body, original_branch.else_body),
        strict=True,
    ):
        assert body is not None and old_body is not None
        edge = ir.get_last_yield_stmt(body)
        old_edge = ir.get_last_yield_stmt(old_body)
        assert edge is not None and old_edge is not None
        for result, value, old_value in zip(results, edge.value, old_edge.value, strict=True):
            _assert_same_tile_window(_tile_memory(value), _tile_memory(result))
            ir.assert_structural_equal(descriptor(value), descriptor(old_value))
            ir.assert_structural_equal(descriptor(result), descriptor(old_value))
        collector = CopyCollector()
        collector.visit_stmt(body)
        assert collector.copies, "the differing slot-2 input must reach its fresh result window"
        for copy in collector.copies:
            assert isinstance(copy.value, ir.Call)
            _assert_same_tile_window(_tile_memory(copy.var), _tile_memory(results[2]))
            _assert_same_tile_window(_tile_memory(copy.value.args[0]), _tile_memory(old_edge.value[2]))
            ir.assert_structural_equal(descriptor(copy.value.args[0]), descriptor(old_edge.value[2]))


def test_dsa_rp_legacy_constrained_partial_if_destinations_reject_overlapping_writes():
    """Partial-overlap destinations independent of both incoming arms cannot be reassigned."""
    before = _nested_if_window_program(constrained=True)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False):
        with pytest.raises(ValueError, match="independent destination windows"):
            passes.materialize_semantic_aliases()(before)


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


def test_ptoas_distinct_mat_phi_selects_an_independently_live_input():
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

    # The base is still read after the merge, independently of the selected value.
    # Redirecting the else producer into it would corrupt the first output block.
    # Select a handle instead of copying payload into a fixed phi allocation.
    # Both the selected value and the old base remain readable after the merge.
    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ConditionalMatLoad, enable_buffer_ir=False)
    mlir = _codegen(optimized, emit_tile_addr=False)
    assert any(" = scf.if " in line and "!pto.tile_buf<" in line for line in mlir.splitlines()), mlir
    assert not any("pto.tmov" in line and line.count("loc=mat") == 2 for line in mlir.splitlines()), mlir


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
        [allocation, stale_allocation, (base, ir.ParamDirection.InOut), x, condition],
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
    repaired_function = after.get_function("kernel")
    assert repaired_function is not None and repaired_function.param_directions[2] == ir.ParamDirection.InOut
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


def _if_window_fixture(memory_space, difference):
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
    result_view = source_view
    if difference == "result_space" and phi_space == ir.MemorySpace.Mat:
        # Vec -> Mat produces Mat's native boxed layout rather than preserving
        # the source Vec layout. Keep the request compatible with that move.
        result_view = ir.TileView(
            valid_shape=[16, 16], blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major
        )
    phi_type = ir.TileType([16, 16], dtype, phi_memref, result_view, phi_space)
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
    return before, a, b, phi_type, view_type


def _assert_reference_if_window_edges(after, a, b, phi_type, view_type):
    phis, tiles = _collect_tile_phi_storage(after)
    assert len(phis) == 1
    result = phis[0][0]
    result_type = result.type
    assert isinstance(result_type, ir.TileType) and result_type.memref is None
    assert result_type.memory_space == phi_type.memory_space
    assert result_type.get_effective_tile_view() == phi_type.get_effective_tile_view()
    function = next(iter(after.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    branch = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.IfStmt))
    actual_inputs = {value.name_hint: value for value in function.params}
    for source, body in zip((a, b), (branch.then_body, branch.else_body), strict=True):
        assert body is not None
        edge = ir.get_last_yield_stmt(body)
        assert edge is not None and len(edge.value) == 1
        actual = actual_inputs[source.name_hint]
        assert edge.value[0].same_as(actual)
        assert isinstance(actual.type, ir.TileType) and isinstance(source.type, ir.TileType)
        if source.type.memref is None:
            assert actual.type.memref is None
        else:
            assert actual.type.memref is not None
            _assert_same_tile_window(actual.type.memref, source.type.memref)
        statements = body.stmts if isinstance(body, ir.SeqStmts) else [body]
        assert all(isinstance(stmt, ir.YieldStmt) for stmt in statements)
    definitions = {
        stmt.var.name_hint: stmt for stmt in function.body.stmts if isinstance(stmt, ir.AssignStmt)
    }
    reshape = definitions["reshaped"].value
    assert isinstance(reshape, ir.Call) and reshape.args[0].same_as(result)
    assert definitions["alias"].value.same_as(tiles["reshaped"])
    for name in ("reshaped", "alias"):
        descriptor = tiles[name].type
        assert isinstance(descriptor, ir.TileType) and descriptor.memref is None
        assert descriptor.memory_space == phi_type.memory_space and descriptor.dtype == phi_type.dtype
        assert len(descriptor.shape) == len(view_type.shape)
        assert all(
            ir.structural_equal(actual, expected)
            for actual, expected in zip(descriptor.shape, view_type.shape, strict=True)
        )
        assert descriptor.tile_view == view_type.tile_view
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt) and returned.value[0].same_as(tiles["alias"])


def _assert_if_window_edges(after, a, b, phi_type, view_type, difference):
    assert isinstance(a.type, ir.TileType)
    a_memref = a.type.memref
    assert a_memref is not None
    phi_space, dtype = phi_type.memory_space, phi_type.dtype
    phis, tiles = _collect_tile_phi_storage(after)
    assert len(phis) == 1
    new_phi, arms = phis[0]
    new_phi_type = new_phi.type
    assert isinstance(new_phi_type, ir.TileType)
    result_window = new_phi_type.memref
    if result_window is None:
        _assert_reference_if_window_edges(after, a, b, phi_type, view_type)
        return
    assert new_phi_type.memory_space == phi_space
    assert new_phi_type.tile_view == phi_type.tile_view
    if difference == "same_window":
        _assert_same_tile_window(result_window, a_memref)
    else:
        # Tile parameters are caller-owned. A different result may copy them,
        # but cannot silently redefine either input's bytes or its descriptor.
        for source in (a, b):
            source_type = source.type
            assert isinstance(source_type, ir.TileType)
            source_window = source_type.memref
            if source_window is not None:
                assert not ir.MemRef.may_alias(result_window, source_window)
    new_function = next(iter(after.functions.values()))
    assert isinstance(new_function.body, ir.SeqStmts)
    branch = next(stmt for stmt in new_function.body.stmts if isinstance(stmt, ir.IfStmt))
    for source, arm, arm_window in zip((a, b), (branch.then_body, branch.else_body), arms, strict=True):
        source_type = source.type
        assert isinstance(source_type, ir.TileType)
        assert arm is not None
        edge = ir.get_last_yield_stmt(arm)
        assert edge is not None and len(edge.value) == 1
        statements = arm.stmts if isinstance(arm, ir.SeqStmts) else [arm]
        copies = [
            stmt
            for stmt in statements
            if isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == ir.get_op("tile.move").name
        ]
        if source_type.memref is None:
            assert arm_window is None and edge.value[0].same_as(source) and not copies
        else:
            assert arm_window is not None
            _assert_same_tile_window(arm_window, result_window)
            if difference == "same_window":
                assert edge.value[0].same_as(source) and not copies
            else:
                assert len(copies) == 1
                copy = copies[0]
                assert isinstance(copy.value, ir.Call)
                assert copy.value.args[0].same_as(source) and edge.value[0].same_as(copy.var)
        edge_type = edge.value[0].type
        assert isinstance(edge_type, ir.TileType)
        assert edge_type.shape == source_type.shape
        assert edge_type.dtype == source_type.dtype
        # Cross-space moves can change layout; their destination descriptor is
        # the planned result. Compare effective views so absent default views
        # do not invoke TileView.__eq__ with None.
        expected = source_type if edge_type.memory_space == source_type.memory_space else new_phi_type
        assert edge_type.get_effective_tile_view() == expected.get_effective_tile_view()
    for name in ("reshaped", "alias"):
        tile_type = tiles[name].type
        assert isinstance(tile_type, ir.TileType) and tile_type.memref is not None
        _assert_same_tile_window(tile_type.memref, result_window)
        assert tile_type.memory_space == phi_space and tile_type.dtype == dtype
        assert len(tile_type.shape) == 2
        for dimension, extent in zip(tile_type.shape, [8, 32]):
            assert isinstance(dimension, ir.ConstInt) and dimension.value == extent
        assert tile_type.tile_view == view_type.tile_view
    returned = new_function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    assert returned.value[0].same_as(tiles["alias"])


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
    """Exact windows share; other known edges require legal transfers before any planner."""
    before, a, b, phi_type, view_type = _if_window_fixture(memory_space, difference)
    phi_space = phi_type.memory_space
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        opaque_result = difference in {"missing_result_storage", "unknown_result_space", "all_unknown"}
        supported_copy = memory_space == ir.MemorySpace.Vec
        if difference == "cross_space":
            supported_copy = False  # At least one arm has an unsupported space pair.
        elif difference == "result_space":
            # This pass checks the target-independent ISA union. Vec -> Mat
            # is A5-only; this fixture does not claim Ascend910 codegen support.
            supported_copy = memory_space in {ir.MemorySpace.Vec, ir.MemorySpace.Acc}
        reference_selection = memory_space != ir.MemorySpace.Vec and difference in {
            "offset",
            "size",
            "slot",
            "slot_count",
            "allocation",
            "missing_storage",
        }
        if (
            difference != "same_window"
            and not opaque_result
            and not supported_copy
            and not reference_selection
        ):
            if phi_space == ir.MemorySpace.Acc:
                error, message = InternalError, "cannot reconcile divergent L0C accumulator buffers"
            elif phi_space in {ir.MemorySpace.LeftScale, ir.MemorySpace.RightScale}:
                error, message = ValueError, "requires the input tile to be in Mat memory"
            else:
                error, message = ValueError, "unsupported tile.move memory-space pair"
            with pytest.raises(error, match=message):
                passes.materialize_semantic_aliases()(before)
            return
        after = passes.materialize_semantic_aliases()(before)

    if opaque_result:
        # No physical result storage was supplied. Legacy handle lowering must
        # resolve it; neither a shared window nor source coordinates are invented.
        ir.assert_structural_equal(after, before)
        return

    _assert_if_window_edges(after, a, b, phi_type, view_type, difference)
    if reference_selection:
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.TileStorageLegalized)
        assert passes.PropertyVerifierRegistry.verify(properties, after) == []


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
