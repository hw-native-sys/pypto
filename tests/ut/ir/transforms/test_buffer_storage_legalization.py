# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Branch storage contracts established before all three memory planners."""

from textwrap import indent

import pypto.language as pl
import pytest
from pypto import ir, passes

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


class _Storage(ir.IRVisitor):
    def __init__(self, program: ir.Program):
        super().__init__()
        self.assigns: dict[str, ir.AssignStmt] = {}
        self.statements: list[ir.AssignStmt] = []
        self.branches: list[ir.IfStmt] = []
        for function in program.functions.values():
            self.visit_stmt(function.body)

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        self.assigns[op.var.name_hint] = op
        self.statements.append(op)
        super().visit_assign_stmt(op)

    def visit_if_stmt(self, op: ir.IfStmt) -> None:
        self.branches.append(op)
        super().visit_if_stmt(op)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
