# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Canonical branch and carry storage established before all three memory planners."""

from textwrap import indent

import pypto.language as pl
import pytest
from pypto import ir, passes

from .buffer_test_utils import statements

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
_TILE = "pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec]"
_MOVE = ir.get_op("tile.move").name
_ALLOC = ir.get_op("tile.alloc").name


def _program(then_body: str, else_body: str, merge: str = "") -> ir.Program:
    # Both original inputs are read after the branch. A transfer onto either
    # original input would therefore change an independently observable value.
    return pl.parse_program(
        f"""
@pl.program
class BranchStorage:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a_gm: pl.Tensor[[4, 8], pl.FP32],
               b_gm: pl.Tensor[[4, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
               flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[4, 8], pl.FP32]:
        a: {_TILE} = pl.tile.load(a_gm, [0, 0], [4, 8])
        b: {_TILE} = pl.tile.load(b_gm, [0, 0], [4, 8])
        if flag:
{indent(then_body, "            ")}
        else:
{indent(else_body, "            ")}
{indent(merge, "        ")}
        sum_a: {_TILE} = pl.tile.add(a, chosen)
        sum_b: {_TILE} = pl.tile.add(b, sum_a)
        result: pl.Tensor[[4, 8], pl.FP32] = pl.tile.store(sum_b, [0, 0], out)
        return result
"""
    )


def _legalize(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        return passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))


class _Storage:
    def __init__(self, program: ir.Program):
        self.assigns: dict[str, ir.AssignStmt] = {}
        self.statements: list[ir.AssignStmt] = []
        self.branches: list[ir.IfStmt] = []
        self.loops: list[ir.ForStmt | ir.WhileStmt] = []
        for statement in statements(program):
            if isinstance(statement, ir.AssignStmt):
                self.assigns[statement.var.name_hint] = statement
                self.statements.append(statement)
            elif isinstance(statement, ir.IfStmt):
                self.branches.append(statement)
            elif isinstance(statement, (ir.ForStmt, ir.WhileStmt)):
                self.loops.append(statement)

    def calls(self, name: str) -> list[ir.AssignStmt]:
        return [op for op in self.statements if isinstance(op.value, ir.Call) and op.value.op.name == name]


def _memref(expr: ir.Expr) -> ir.MemRef:
    tile = expr.type
    assert isinstance(tile, ir.TileType) and tile.memref is not None
    return tile.memref


def _base(expr: ir.Expr) -> int:
    return _memref(expr).base_.unique_id


def _yield(body: ir.Stmt) -> ir.YieldStmt:
    if isinstance(body, ir.YieldStmt):
        return body
    assert isinstance(body, ir.SeqStmts)
    result = body.stmts[-1]
    assert isinstance(result, ir.YieldStmt)
    return result


def _assert_canonical_arms(storage: _Storage) -> None:
    assert storage.branches
    for branch in storage.branches:
        assert branch.else_body is not None
        for body in [branch.then_body, branch.else_body]:
            values = _yield(body).value
            for result, value in zip(branch.return_vars, values, strict=True):
                if isinstance(result.type, ir.TileType):
                    assert _base(result) == _base(value)
                    assert ir.structural_equal(_memref(result).byte_offset_, _memref(value).byte_offset_)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_external_branch_inputs_keep_their_storage(planner):
    before = _program(f"chosen: {_TILE} = pl.yield_(a)", f"chosen: {_TILE} = pl.yield_(b)")
    after = _legalize(before, planner)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    phi = storage.branches[0].return_vars[0]
    assert _base(phi) not in {_base(storage.assigns[name].var) for name in ("a", "b")}
    moves = storage.calls(_MOVE)
    assert len(moves) == 2
    assert {_base(move.value.args[0]) for move in moves if isinstance(move.value, ir.Call)} == {
        _base(storage.assigns[name].var) for name in ("a", "b")
    }
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        ir.assert_structural_equal(passes.materialize_semantic_aliases()(after), after)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_branch_local_fresh_producers_write_the_canonical_destination(planner):
    before = _program(
        f"then_value: {_TILE} = pl.tile.add(a, 1.0)\nchosen: {_TILE} = pl.yield_(then_value)",
        f"else_value: {_TILE} = pl.tile.mul(b, 2.0)\nchosen: {_TILE} = pl.yield_(else_value)",
    )
    storage = _Storage(_legalize(before, planner))
    _assert_canonical_arms(storage)
    assert not storage.calls(_MOVE)
    assert _base(storage.assigns["then_value"].var) == _base(storage.assigns["else_value"].var)
    # Two inputs, one branch destination, and the two post-branch additions.
    # The discarded branch-producer allocations must not reach PTOAS planning.
    assert len(storage.calls(_ALLOC)) == 5


@pytest.mark.parametrize("planner", _PLANNERS)
def test_branch_swap_snapshots_into_independent_destinations(planner):
    before = _program(
        "left, right = pl.yield_(a, b)",
        "left, right = pl.yield_(b, a)",
        f"chosen: {_TILE} = pl.tile.add(left, right)",
    )
    storage = _Storage(_legalize(before, planner))
    _assert_canonical_arms(storage)
    inputs = {_base(storage.assigns[name].var) for name in ("a", "b")}
    destinations = {_base(var) for var in storage.branches[0].return_vars}
    assert len(destinations) == 2 and inputs.isdisjoint(destinations)
    assert len(storage.calls(_MOVE)) == 4


@pytest.mark.parametrize("planner", _PLANNERS)
def test_branch_view_copies_without_retargeting_its_live_input(planner):
    before = _program(
        f"view: {_TILE} = pl.tile.reshape(a, [4, 8])\nchosen: {_TILE} = pl.yield_(view)",
        f"chosen: {_TILE} = pl.yield_(b)",
    )
    storage = _Storage(_legalize(before, planner))
    _assert_canonical_arms(storage)
    assert _base(storage.assigns["view"].var) == _base(storage.assigns["a"].var)
    assert _base(storage.branches[0].return_vars[0]) != _base(storage.assigns["a"].var)
    assert len(storage.calls(_MOVE)) == 2


@pytest.mark.parametrize("planner", _PLANNERS)
def test_identical_branch_inputs_need_no_transfer(planner):
    before = _program(f"chosen: {_TILE} = pl.yield_(a)", f"chosen: {_TILE} = pl.yield_(a)")
    storage = _Storage(_legalize(before, planner))
    _assert_canonical_arms(storage)
    assert _base(storage.branches[0].return_vars[0]) == _base(storage.assigns["a"].var)
    assert not storage.calls(_MOVE)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_declared_branch_producer_keeps_its_pinned_output(planner):
    pinned = 'pl.Tile[[4, 8], pl.FP32, pl.MemRef("then_slot"), pl.Mem.Vec]'
    before = _program(
        f"then_value: {pinned} = pl.tile.add(a, 1.0)\nchosen: {_TILE} = pl.yield_(then_value)",
        f"else_value: {_TILE} = pl.tile.mul(b, 2.0)\nchosen: {_TILE} = pl.yield_(else_value)",
    )
    storage = _Storage(_legalize(before, planner))
    _assert_canonical_arms(storage)
    assert _base(storage.assigns["then_value"].var) != _base(storage.branches[0].return_vars[0])
    assert _memref(storage.assigns["then_value"].var).base_.name_hint == "then_slot"
    assert len(storage.calls(_MOVE)) == 1


def test_pypto_reuse_preserves_legalized_branch_transfers():
    before = _program(f"chosen: {_TILE} = pl.yield_(a)", f"chosen: {_TILE} = pl.yield_(b)")
    legalized = _legalize(before, passes.MemoryPlanner.PYPTO)
    with passes.PassContext([], enable_buffer_ir=True):
        after = passes.memory_reuse()(legalized)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    assert _base(storage.branches[0].return_vars[0]) not in {
        _base(storage.assigns[name].var) for name in ("a", "b")
    }


def test_ptoas_legacy_branch_path_remains_the_default():
    before = _program(f"chosen: {_TILE} = pl.yield_(a)", f"chosen: {_TILE} = pl.yield_(b)")
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(before))
    storage = _Storage(after)
    assert not storage.calls(_MOVE)
    assert _base(storage.branches[0].return_vars[0]) == _base(storage.assigns["a"].var)


def _loop_program(
    body, *, while_loop=False, initializers=("a", "b"), observe_initial=False, view_result=False, prelude=""
):
    carries = ["left", "right", "third"][: len(initializers)]
    values = ", ".join(initializers)
    names = ", ".join(carries)
    header = (
        f"for ({names},) in pl.while_(init_values=({values},)):\n            pl.cond(flag)"
        if while_loop
        else f"for _i, ({names},) in pl.range(0, 3, init_values=({values},)):"
    )
    view = f"viewed: {_TILE} = pl.tile.reshape(r_left, [4, 8])" if view_result else ""
    left_result = "viewed" if view_result else "r_left"
    observe = (
        f"observed: {_TILE} = pl.tile.add(combined, a)"
        if observe_initial
        else f"observed: {_TILE} = combined"
    )
    return pl.parse_program(f"""
@pl.program
class LoopStorage:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a_gm: pl.Tensor[[4, 8], pl.FP32],
               b_gm: pl.Tensor[[4, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
               flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[4, 8], pl.FP32]:
        a: {_TILE} = pl.tile.load(a_gm, [0, 0], [4, 8])
        b: {_TILE} = pl.tile.load(b_gm, [0, 0], [4, 8])
{indent(prelude, "        ")}
        {header}
{indent(body, "            ")}
        {view}
        combined: {_TILE} = pl.tile.add({left_result}, r_right)
        {observe}
        result: pl.Tensor[[4, 8], pl.FP32] = pl.tile.store(observed, [0, 0], out)
        return result
""")


def _verify_storage(program, *, physical=False):
    properties = passes.IRPropertySet()
    properties.insert(
        passes.IRProperty.TileStorageAllocated if physical else passes.IRProperty.TileStorageLegalized
    )
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []


def _simulate_parallel_moves(loop, expected):
    """Execute the explicit transfer writes using distinct old carry payloads."""
    memory = {_base(argument): i for i, argument in enumerate(loop.iter_args)}
    assert len(memory) == len(loop.iter_args)
    statements = loop.body.stmts if isinstance(loop.body, ir.SeqStmts) else [loop.body]
    moves = []
    for statement in statements:
        if not isinstance(statement, ir.AssignStmt):
            continue
        if not isinstance(statement.var.type, ir.TileType):
            continue
        assert isinstance(statement.value, ir.Call) and statement.value.op.name == _MOVE
        source = _base(statement.value.args[0])
        assert source in memory, "a scratch read must be preceded by its explicit snapshot"
        memory[_base(statement.var)] = memory[source]
        moves.append(statement)
    assert moves, "the permutation must exercise explicit writes"
    assert [memory[_base(value)] for value in _yield(loop.body).value] == expected
    assert all(
        _base(result) == _base(argument)
        for result, argument in zip(loop.return_vars, loop.iter_args, strict=True)
    )


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("while_loop", [False, True])
def test_live_loop_input_is_copied_before_any_iteration(planner, while_loop):
    program = _loop_program(
        f"next_left: {_TILE} = pl.tile.add(left, a)\nr_left, r_right = pl.yield_(next_left, right)",
        while_loop=while_loop,
        observe_initial=True,
    )
    after = _legalize(program, planner)
    storage = _Storage(after)
    loop = storage.loops[0]
    original = _base(storage.assigns["a"].var)
    assert _base(loop.iter_args[0]) != original
    initial = loop.iter_args[0].initValue
    assert isinstance(initial, ir.Var)
    entry = next(
        statement for statement in storage.statements if statement.var.unique_id == initial.unique_id
    )
    assert isinstance(entry.value, ir.Call) and entry.value.op.name == _MOVE
    assert _base(entry.value.args[0]) == original
    # The entry copy is outside the body, so a zero-trip For/While still has the
    # original carried value. Neither a body producer nor a writeback may clobber a.
    body_statements = loop.body.stmts if isinstance(loop.body, ir.SeqStmts) else [loop.body]
    assert all(statement is not entry for statement in body_statements)
    assert all(
        _base(statement.var) != original
        for statement in body_statements
        if isinstance(statement, ir.AssignStmt) and isinstance(statement.var.type, ir.TileType)
    )
    _verify_storage(after)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        ir.assert_structural_equal(passes.materialize_semantic_aliases()(after), after)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("while_loop", [False, True])
@pytest.mark.parametrize("fanout", [False, True])
def test_carry_cycles_and_fanout_preserve_parallel_values(planner, while_loop, fanout):
    body = (
        "keep_third: pl.Tensor[[4, 8], pl.FP32] = pl.tile.store(third, [0, 0], out)\n"
        "r_left, r_right, r_third = pl.yield_(right, left, left)"
        if fanout
        else "r_left, r_right = pl.yield_(right, left)"
    )
    initializers = ("a", "b", "a") if fanout else ("a", "b")
    after = _legalize(_loop_program(body, while_loop=while_loop, initializers=initializers), planner)
    storage = _Storage(after)
    loop = storage.loops[0]
    _simulate_parallel_moves(loop, [1, 0, 0] if fanout else [1, 0])
    _verify_storage(after)
    # Scratch and entry allocations are already visible before any planner.
    function = next(iter(after.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    allocations = storage.calls(_ALLOC)
    assert allocations and all(
        any(statement is allocation for statement in function.body.stmts) for allocation in allocations
    )
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        if planner == passes.MemoryPlanner.PYPTO:
            after = passes.memory_reuse()(after)
        _verify_storage(after)
        if planner != passes.MemoryPlanner.PTOAS:
            allocation_count = len(_Storage(after).calls(_ALLOC))
            addressed = passes.allocate_memory_addr()(after)
            assert len(_Storage(addressed).calls(_ALLOC)) == allocation_count
            _verify_storage(addressed, physical=True)


def test_repeated_initializer_gets_two_independent_entry_copies():
    after = _legalize(
        _loop_program("r_left, r_right = pl.yield_(right, left)", initializers=("a", "a")),
        passes.MemoryPlanner.PTOAS,
    )
    storage = _Storage(after)
    loop = storage.loops[0]
    assert _base(loop.iter_args[0]) != _base(loop.iter_args[1])
    entry_ids = set()
    for argument in loop.iter_args:
        initial = argument.initValue
        assert isinstance(initial, ir.Var)
        entry_ids.add(initial.unique_id)
    entries = [statement for statement in storage.statements if statement.var.unique_id in entry_ids]
    assert len(entries) == 2 and len({statement.var.name_hint for statement in entries}) == 2
    assert all(
        isinstance(statement.value, ir.Call)
        and _base(statement.value.args[0]) == _base(storage.assigns["a"].var)
        for statement in entries
    )
    _simulate_parallel_moves(loop, [1, 0])
    _verify_storage(after)


def test_while_result_views_follow_the_canonical_carry():
    program = _loop_program("r_left, r_right = pl.yield_(right, left)", while_loop=True, view_result=True)
    # The view must follow the While result after InitMemRef originally gave
    # that result fresh storage.
    after = _legalize(program, passes.MemoryPlanner.PTOAS)
    storage = _Storage(after)
    loop = storage.loops[0]
    _verify_storage(after)
    assert all(
        _base(result) == _base(argument)
        for result, argument in zip(loop.return_vars, loop.iter_args, strict=True)
    )
    combined = storage.assigns["combined"].value
    assert isinstance(combined, ir.Call)
    assert [_base(value) for value in combined.args] == [_base(value) for value in loop.return_vars]
    assert _base(storage.assigns["viewed"].var) == _base(loop.return_vars[0])


def test_nested_loop_preserves_input_read_on_the_next_outer_iteration():
    program = pl.parse_program(f"""
@pl.program
class NestedStorage:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a_gm: pl.Tensor[[4, 8], pl.FP32],
               b_gm: pl.Tensor[[4, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[4, 8], pl.FP32]]) -> pl.Tensor[[4, 8], pl.FP32]:
        a: {_TILE} = pl.tile.load(a_gm, [0, 0], [4, 8])
        b: {_TILE} = pl.tile.load(b_gm, [0, 0], [4, 8])
        for _i, (outer,) in pl.range(0, 2, init_values=(b,)):
            observed: {_TILE} = pl.tile.add(a, outer)
            for _j, (inner,) in pl.range(0, 2, init_values=(a,)):
                next_inner: {_TILE} = pl.tile.add(inner, 1.0)
                inner_result = pl.yield_(next_inner)
            next_outer: {_TILE} = pl.tile.add(observed, inner_result)
            outer_result = pl.yield_(next_outer)
        result: pl.Tensor[[4, 8], pl.FP32] = pl.tile.store(outer_result, [0, 0], out)
        return result
""")
    storage = _Storage(_legalize(program, passes.MemoryPlanner.PTOAS))
    assert len(storage.loops) == 2
    inner = storage.loops[1]
    assert _base(inner.iter_args[0]) != _base(storage.assigns["a"].var)
    assert _base(storage.assigns["next_inner"].var) != _base(storage.assigns["a"].var)


def test_metadata_only_preloop_view_does_not_force_an_entry_copy():
    program = _loop_program(
        f"next_left: {_TILE} = pl.tile.add(left, right)\nr_left, r_right = pl.yield_(next_left, right)",
        initializers=("alias_seed", "b"),
        prelude=f"alias_seed: {_TILE} = pl.tile.reshape(a, [4, 8])",
    )
    storage = _Storage(_legalize(program, passes.MemoryPlanner.PTOAS))
    assert _base(storage.loops[0].iter_args[0]) == _base(storage.assigns["a"].var)
    assert not any(statement.var.name_hint.startswith("carry_input_") for statement in storage.statements)


def test_view_created_inside_loop_still_observes_old_input_data():
    program = _loop_program(
        f"old_view: {_TILE} = pl.tile.reshape(a, [4, 8])\n"
        f"next_left: {_TILE} = pl.tile.add(left, old_view)\n"
        "r_left, r_right = pl.yield_(next_left, right)",
    )
    storage = _Storage(_legalize(program, passes.MemoryPlanner.PTOAS))
    original = _base(storage.assigns["a"].var)
    assert _base(storage.loops[0].iter_args[0]) != original
    assert _base(storage.assigns["old_view"].var) == original


def test_prior_read_does_not_isolate_a_local_top_level_accumulator():
    program = pl.parse_program("""
@pl.program
class AccStorage:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, left: pl.Tile[[16, 16], pl.FP16, pl.Mem.Left],
               right: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        acc: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(left, right)
        previous: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.move(acc, target_memory=pl.Mem.Vec)
        saved: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(previous, [0, 0], out)
        for _i, (current,) in pl.range(0, 2, init_values=(acc,)):
            updated: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(current, left, right)
            accumulated = pl.yield_(updated)
        final: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.move(accumulated, target_memory=pl.Mem.Vec)
        result: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(final, [0, 0], saved)
        return result
""")
    after = _legalize(program, passes.MemoryPlanner.PTOAS)
    storage = _Storage(after)
    assert _base(storage.loops[0].iter_args[0]) == _base(storage.assigns["acc"].var)
    assert not any(statement.var.name_hint.startswith("carry_input_") for statement in storage.statements)
    _verify_storage(after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
