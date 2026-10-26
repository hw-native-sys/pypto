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
from pypto.ir.op import tile as tile_ops

from .buffer_test_utils import TileHandleOrigins, statements

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
_STORAGE_MODES = [
    (passes.MemoryPlanner.PYPTO, False),
    (passes.MemoryPlanner.DSA_RP, False),
    (passes.MemoryPlanner.PYPTO, True),
]
_STORAGE_MODE_IDS = ["legacy_pypto", "legacy_dsa", "buffer_pypto"]
_TILE = "pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec]"
_MOVE = ir.get_op("tile.move").name
_ALLOC = ir.get_op("tile.alloc").name
_LOOP_INPUT_CONFLICT = (
    "independently live loop input without a same-space copy|independently observed older value version"
    "|readonly caller buffer"
)


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


def _legalize(
    program: ir.Program, planner: passes.MemoryPlanner, *, enable_buffer_ir: bool = True
) -> ir.Program:
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
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
    result = ir.get_last_yield_stmt(body)
    assert result is not None
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


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("swap_arms", [False, True], ids=["original_arms", "reversed_arms"])
@pytest.mark.parametrize("swap_slots", [False, True], ids=["varying_first", "common_first"])
def test_common_result_window_is_reserved_before_other_slots_borrow(
    planner, enable_buffer_ir, swap_arms, swap_slots
):
    """The common a result must survive a peer selecting either a or b in the same if."""
    names = ["varying", "common"]
    then_values, else_values = ["a", "a"], ["b", "a"]
    if swap_slots:
        names.reverse()
        then_values.reverse()
        else_values.reverse()
    if swap_arms:
        then_values, else_values = else_values, then_values
    condition = "not flag" if swap_arms else "flag"
    before = pl.parse_program(f"""
@pl.program
class CommonResultStorage:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a_gm: pl.Tensor[[4, 8], pl.FP32], b_gm: pl.Tensor[[4, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[8, 8], pl.FP32]], flag: pl.Scalar[pl.BOOL]
               ) -> pl.Tensor[[8, 8], pl.FP32]:
        a = pl.tile.load(a_gm, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        b = pl.tile.load(b_gm, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        if {condition}:
            {", ".join(names)} = pl.yield_({", ".join(then_values)})
        else:
            {", ".join(names)} = pl.yield_({", ".join(else_values)})
        first_saved = pl.tile.store(varying, [0, 0], out)
        second_saved = pl.tile.store(common, [4, 0], first_saved)
        return second_saved
""")
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.branches) == 1
    branch = storage.branches[0]
    _assert_canonical_arms(storage)
    by_name = dict(zip(names, branch.return_vars, strict=True))

    def window(value):
        memory = _memref(value)
        assert isinstance(memory.byte_offset_, ir.ConstInt)
        return _base(value), memory.byte_offset_.value, memory.size_

    a, b = storage.assigns["a"].var, storage.assigns["b"].var
    assert not ir.MemRef.may_alias(_memref(a), _memref(b))
    assert window(by_name["common"]) == window(a), "the common result reserves the unchanged a window"
    assert not ir.MemRef.may_alias(_memref(by_name["varying"]), _memref(by_name["common"]))
    for name, result in (("first_saved", by_name["varying"]), ("second_saved", by_name["common"])):
        store = storage.assigns[name].value
        assert isinstance(store, ir.Call) and store.op.name == ir.get_op("tile.store").name
        assert window(store.args[0]) == window(result)
    for result in branch.return_vars:
        tile_type = result.type
        assert isinstance(tile_type, ir.TileType)
        assert tile_type.shape == [4, 8] and tile_type.dtype == ir.DataType.FP32
        assert tile_type.memory_space == ir.MemorySpace.Vec
        view = tile_type.tile_view
        valid = view.valid_shape if view is not None and view.valid_shape else tile_type.shape
        assert len(valid) == 2
        assert [dim.value for dim in valid if isinstance(dim, ir.ConstInt)] == [4, 8]
    origins = {window(a): "a", window(b): "b"}

    class ArmDefinitions(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.assignments = []

        def visit_assign_stmt(self, op):
            if isinstance(op.var.type, ir.TileType):
                self.assignments.append(op)
            super().visit_assign_stmt(op)

    for body, expected in zip((branch.then_body, branch.else_body), (then_values, else_values), strict=True):
        assert body is not None
        definitions = ArmDefinitions()
        definitions.visit_stmt(body)
        memory = dict(origins)
        for op in definitions.assignments:
            source = op.value
            if isinstance(source, ir.Call):
                assert source.op.name == _MOVE, "reconciliation must preserve the incoming tile value"
                source = source.args[0]
            assert isinstance(source, (ir.Var, ir.IterArg))
            assert window(source) in memory, "a snapshot must be initialized before its read"
            memory[window(op.var)] = memory[window(source)]
        assert [memory[window(value)] for value in _yield(body).value] == expected
    _verify_storage(after)


def test_duplicate_results_with_different_last_uses_preserve_the_later_old_value():
    """An early duplicate may share a, but a later duplicate must survive an intervening write to a."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a_gm: pl.Tensor[[4, 8], pl.FP32],
            replacement_gm: pl.Tensor[[4, 8], pl.FP32],
            flag: pl.Scalar[pl.BOOL],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tensor[[8, 8], pl.FP32]:
            a: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(a_gm, [0, 0], [4, 8])
            replacement: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(replacement_gm, [0, 0], [4, 8])
            if flag:
                early, late = pl.yield_(a, a)
            else:
                early, late = pl.yield_(a, a)
            first_saved = pl.tile.store(early, [0, 0], out)
            _written = pl.tile.assemble(a, replacement, [0, 0])
            second_saved = pl.tile.store(late, [4, 0], first_saved)
            return second_saved

    after = _legalize(Before, passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    assert len(storage.branches) == 1
    early, late = storage.branches[0].return_vars
    writers = storage.calls(ir.get_op("tile.assemble").name)
    assert len(writers) == 1
    writer = writers[0]
    assert isinstance(writer.value, ir.Call)
    assert writer.value.args[0].same_as(storage.assigns["a"].var), "the writer must target the old a origin"
    assert _base(writer.var) == _base(writer.value.args[0])
    first_store, second_store = storage.assigns["first_saved"], storage.assigns["second_saved"]
    assert storage.statements.index(first_store) < storage.statements.index(writer)
    assert storage.statements.index(writer) < storage.statements.index(second_store)
    assert isinstance(second_store.value, ir.Call)
    reader = second_store.value.args[0]
    assert not ir.MemRef.may_alias(_memref(reader), _memref(writer.value.args[0])), (
        "the later old-value read must not observe the intervening write to a"
    )
    if _base(reader) != _base(late):
        snapshots = [op for op in storage.statements if op.var.same_as(reader)]
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert isinstance(snapshot.value, ir.Call) and snapshot.value.op.name == _MOVE
        assert storage.statements.index(snapshot) < storage.statements.index(writer)
        reader = snapshot.value.args[0]
    assert _base(reader) == _base(late)
    assert ir.structural_equal(_memref(reader).byte_offset_, _memref(late).byte_offset_)
    assert _memref(reader).size_ == _memref(late).size_
    for value in (early, late, second_store.value.args[0]):
        tile_type = value.type
        assert isinstance(tile_type, ir.TileType)
        assert tile_type.shape == [4, 8] and tile_type.dtype == ir.DataType.FP32
        assert tile_type.memory_space == ir.MemorySpace.Vec
        view = tile_type.tile_view
        valid = view.valid_shape if view is not None and view.valid_shape else tile_type.shape
        assert len(valid) == 2
        assert [dim.value for dim in valid if isinstance(dim, ir.ConstInt)] == [4, 8]
    _verify_storage(after)


def test_later_fixed_mat_write_does_not_block_an_earlier_disjoint_window():
    """A later write in A1 must not prevent an earlier identity result from sharing A0."""
    span = ir.Span.unknown()
    shared = ir.Var("shared", ir.PtrType(), span)
    gm_base = ir.Var("gm_base", ir.PtrType(), span)
    gm = ir.Var("gm", ir.TensorType([32, 32], ir.DataType.FP16, ir.MemRef(gm_base, 0, 2048, span)), span)
    first_flag = ir.Var("first_flag", ir.ScalarType(ir.DataType.BOOL), span)
    second_flag = ir.Var("second_flag", ir.ScalarType(ir.DataType.BOOL), span)
    load_then = tile_ops.load(gm, [0, 0], [16, 32], target_memory=ir.MemorySpace.Mat)
    load_else = tile_ops.load(gm, [16, 0], [16, 32], target_memory=ir.MemorySpace.Mat)
    load_type = load_then.type
    assert isinstance(load_type, ir.TileType)
    allocations: list[ir.Stmt] = []

    def allocation(name, space):
        base = ir.Var(name, ir.PtrType(), span)
        args: list[ir.Expr] = [
            ir.ConstInt(space.value, ir.DataType.INDEX, span),
            ir.ConstInt(1024, ir.DataType.INDEX, span),
        ]
        allocations.append(ir.AssignStmt(base, ir.Call(ir.get_op("tile.alloc"), args, base.type, span), span))
        return ir.MemRef(base, 0, 1024, span)

    def mat_type(memory):
        return ir.TileType([16, 32], ir.DataType.FP16, memory, load_type.tile_view, ir.MemorySpace.Mat)

    a0 = ir.Var("a0", mat_type(ir.MemRef(shared, 0, 1024, span)), span)
    earlier = ir.Var("earlier", mat_type(allocation("earlier_request", ir.MemorySpace.Mat)), span)
    later = ir.Var("later", mat_type(ir.MemRef(shared, 1024, 1024, span, is_pinned=True)), span)
    then_value = ir.Var("then_value", mat_type(allocation("then_root", ir.MemorySpace.Mat)), span)
    else_value = ir.Var("else_value", mat_type(allocation("else_root", ir.MemorySpace.Mat)), span)
    first = ir.IfStmt(first_flag, ir.YieldStmt([a0], span), ir.YieldStmt([a0], span), [earlier], span)
    second = ir.IfStmt(
        second_flag,
        ir.SeqStmts([ir.AssignStmt(then_value, load_then, span), ir.YieldStmt([then_value], span)], span),
        ir.SeqStmts([ir.AssignStmt(else_value, load_else, span), ir.YieldStmt([else_value], span)], span),
        [later],
        span,
    )
    read = tile_ops.move(earlier, ir.MemorySpace.Left)
    read_type = read.type
    assert isinstance(read_type, ir.TileType)
    observed = ir.Var(
        "observed",
        ir.TileType(
            read_type.shape,
            read_type.dtype,
            allocation("observed_root", ir.MemorySpace.Left),
            read_type.tile_view,
            ir.MemorySpace.Left,
        ),
        span,
    )
    body = ir.SeqStmts(
        [
            *allocations,
            first,
            second,
            ir.AssignStmt(observed, read, span),
            ir.ReturnStmt([observed, later], span),
        ],
        span,
    )
    function = ir.Function(
        "kernel",
        [shared, gm_base, gm, a0, first_flag, second_flag],
        [observed.type, later.type],
        body,
        span,
        type=ir.FunctionType.InCore,
    )
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP, enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(ir.Program([function], "DisjointMatWrites", span))
    storage = _Storage(after)
    assert len(storage.branches) == 2
    _assert_canonical_arms(storage)
    early_memory = _memref(storage.branches[0].return_vars[0])
    late_memory = _memref(storage.branches[1].return_vars[0])
    assert ir.MemRef.same_allocation(early_memory, _memref(a0))
    assert ir.structural_equal(early_memory.byte_offset_, _memref(a0).byte_offset_)
    assert ir.MemRef.same_allocation(early_memory, late_memory)
    assert isinstance(late_memory.byte_offset_, ir.ConstInt) and late_memory.byte_offset_.value == 1024
    assert early_memory.size_ == late_memory.size_ == 1024
    assert not ir.MemRef.may_alias(early_memory, late_memory)
    observation = storage.assigns["observed"].value
    assert isinstance(observation, ir.Call)
    assert _base(observation.args[0]) == _base(a0)
    assert ir.structural_equal(_memref(observation.args[0]).byte_offset_, early_memory.byte_offset_)
    for move in storage.calls(_MOVE):
        assert isinstance(move.value, ir.Call) and isinstance(move.var.type, ir.TileType)
        source_type = move.value.args[0].type
        assert isinstance(source_type, ir.TileType)
        assert (source_type.memory_space, move.var.type.memory_space) != (
            ir.MemorySpace.Mat,
            ir.MemorySpace.Mat,
        )
    _verify_storage(after)


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


def test_ptoas_legacy_branch_path_with_explicit_off():
    """Legacy PTOAS receives closed If edges while both old inputs stay observable."""
    before = _program(f"chosen: {_TILE} = pl.yield_(a)", f"chosen: {_TILE} = pl.yield_(b)")
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(before))
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    branch = storage.branches[0]
    chosen = branch.return_vars[0]
    for body, name in ((branch.then_body, "a"), (branch.else_body, "b")):
        assert body is not None
        source = storage.assigns[name].var
        assert not ir.MemRef.may_alias(_memref(chosen), _memref(source))
        yielded = _yield(body).value[0]
        transfer = next(op for op in storage.statements if op.var.same_as(yielded))
        assert isinstance(transfer.value, ir.Call) and transfer.value.op.name == _MOVE
        assert transfer.value.args[0].same_as(source), "each arm must copy its own incoming value"
        source_type, target_type = source.type, transfer.var.type
        assert isinstance(source_type, ir.TileType) and isinstance(target_type, ir.TileType)
        assert target_type.shape == source_type.shape == [4, 8]
        assert target_type.dtype == source_type.dtype == ir.DataType.FP32
        assert target_type.memory_space == source_type.memory_space == ir.MemorySpace.Vec
        assert target_type.tile_view == source_type.tile_view
    for name, old_input in (("sum_a", "a"), ("sum_b", "b")):
        observer = storage.assigns[name].value
        assert isinstance(observer, ir.Call) and observer.op.name == ir.get_op("tile.add").name
        assert observer.args[0].same_as(storage.assigns[old_input].var)
    _verify_storage(after)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("scope_kind", ["split", "runtime", "nested"])
@pytest.mark.parametrize("fresh_producer", [False, True])
def test_branch_yields_inside_transparent_scopes(planner, scope_kind, fresh_producer):
    if fresh_producer:
        before = _program(
            f"then_value: {_TILE} = pl.tile.add(a, 1.0)\nchosen: {_TILE} = pl.yield_(then_value)",
            f"else_value: {_TILE} = pl.tile.mul(b, 2.0)\nchosen: {_TILE} = pl.yield_(else_value)",
        )
    else:
        before = _program(f"chosen: {_TILE} = pl.yield_(a)", f"chosen: {_TILE} = pl.yield_(b)")

    class WrapArms(ir.IRMutator):
        def visit_if_stmt(self, op: ir.IfStmt) -> ir.Stmt:
            def wrap(body: ir.Stmt) -> ir.Stmt:
                # An unrelated region's yields must not receive the outer copies.
                nested = ir.IfStmt(
                    op.condition, ir.YieldStmt([], op.span), ir.YieldStmt([], op.span), [], op.span
                )
                body = ir.SeqStmts([nested, body], op.span)
                if scope_kind in ("split", "nested"):
                    body = ir.SplitAivScopeStmt(
                        ir.SplitMode.UP_DOWN, count=2, name_hint="split_arm", body=body, span=op.span
                    )
                if scope_kind in ("runtime", "nested"):
                    body = ir.RuntimeScopeStmt(manual=True, name_hint="runtime_arm", body=body, span=op.span)
                return ir.SeqStmts([body], op.span)

            assert op.else_body is not None
            return ir.IfStmt(op.condition, wrap(op.then_body), wrap(op.else_body), op.return_vars, op.span)

    after = _legalize(WrapArms().visit_program(before), planner)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    branch = storage.branches[0]
    assert _base(branch.return_vars[0]) not in {_base(storage.assigns[name].var) for name in ("a", "b")}
    assert len(storage.calls(_MOVE)) == (0 if fresh_producer else 2)
    if fresh_producer:
        assert _base(storage.assigns["then_value"].var) == _base(branch.return_vars[0])
        assert _base(storage.assigns["else_value"].var) == _base(branch.return_vars[0])

    nested_branches = 0
    for body in (branch.then_body, branch.else_body):
        assert body is not None
        scopes = []
        while isinstance(body, (ir.SeqStmts, ir.SplitAivScopeStmt, ir.RuntimeScopeStmt)):
            if isinstance(body, ir.SeqStmts):
                for child in body.stmts:
                    if isinstance(child, ir.IfStmt):
                        assert not _yield(child.then_body).value
                        assert child.else_body is not None and not _yield(child.else_body).value
                        nested_branches += 1
                if not fresh_producer and isinstance(body.stmts[-1], ir.YieldStmt):
                    move = body.stmts[-2]
                    assert isinstance(move, ir.AssignStmt) and isinstance(move.value, ir.Call)
                    assert move.value.op.name == _MOVE
                    assert _yield(body).value[0].same_as(move.var)
                    assert scopes  # Transfer remains inside the branch's transparent scope.
                body = body.stmts[-1]
            else:
                if isinstance(body, ir.SplitAivScopeStmt):
                    assert body.split == ir.SplitMode.UP_DOWN and body.count == 2
                    assert body.name_hint == "split_arm"
                else:
                    assert body.manual and body.name_hint == "runtime_arm"
                scopes.append(type(body))
                body = body.body
        assert len(scopes) == (2 if scope_kind == "nested" else 1)
        assert isinstance(body, ir.YieldStmt)
    assert nested_branches == 2
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        ir.assert_structural_equal(passes.materialize_semantic_aliases()(after), after)


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


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
def test_partial_phi_write_isolates_an_observed_loop_initializer(planner, enable_buffer_ir, while_loop):
    """A partial phi-view write must update the carry while preserving its observed initializer."""
    header = (
        "for (carry, index) in pl.while_(init_values=(initial, 0)):\n            pl.cond(index < count)"
        if while_loop
        else "for index, (carry,) in pl.range(count, init_values=(initial,)):"
    )
    yielded = (
        "result, _counter = pl.yield_(selected, index + 1)" if while_loop else "result = pl.yield_(selected)"
    )
    program = pl.parse_program(f"""
@pl.program
class PartialPhiWrite:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a_gm: pl.Tensor[[4, 8], pl.FP32], replacement_gm: pl.Tensor[[1, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[8, 8], pl.FP32]], count: pl.Scalar[pl.INDEX],
               flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[8, 8], pl.FP32]:
        initial = pl.tile.load(a_gm, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        replacement = pl.tile.load(replacement_gm, [0, 0], [1, 8], target_memory=pl.Mem.Vec)
        {header}
            if flag:
                selected = pl.yield_(carry)
            else:
                selected = pl.yield_(carry)
            row = pl.tile.slice(selected, [1, 8], [1, 0])
            written = pl.tile.assemble(row, replacement, [0, 0])
            {yielded}
        old_saved = pl.tile.store(initial, [0, 0], out)
        updated_saved = pl.tile.store(result, [4, 0], old_saved)
        return updated_saved
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        initialized = passes.init_mem_ref()(program)
        after = passes.materialize_semantic_aliases()(initialized)
        if planner == passes.MemoryPlanner.PYPTO and not enable_buffer_ir:
            # Legacy PYPTO closes loop edges after packing. Check the entry
            # isolation here before advancing to that boundary-owning phase.
            materialized = _Storage(after)
            old = _memref(materialized.assigns["initial"].var)
            for value in (materialized.loops[0].iter_args[0], materialized.assigns["written"].var):
                assert not ir.MemRef.may_alias(old, _memref(value))
            after = passes.memory_reuse()(after)
    original = _Storage(initialized)
    storage = _Storage(after)
    assert len(storage.loops) == 1 and len(storage.branches) == 1
    _assert_canonical_arms(storage)
    loop = storage.loops[0]
    initial = storage.assigns["initial"].var
    selected = storage.branches[0].return_vars[0]
    row = storage.assigns["row"].var
    writers = storage.calls(ir.get_op("tile.assemble").name)
    assert len(writers) == 1
    writer = writers[0]
    assert isinstance(writer.value, ir.Call)
    assert writer.value.args[0].same_as(row)
    assert writer.value.args[1].same_as(storage.assigns["replacement"].var)

    def window(value):
        memory = _memref(value)
        assert isinstance(memory.byte_offset_, ir.ConstInt)
        return _base(value), memory.byte_offset_.value, memory.size_

    carry = loop.iter_args[0]
    assert window(carry)[2] == window(selected)[2] == 128
    for value in (carry.initValue, carry, loop.return_vars[0], _yield(loop.body).value[0]):
        assert window(value) == window(carry)
        assert not ir.MemRef.may_alias(_memref(initial), _memref(value))
    for value in (selected, row, writer.var):
        assert not ir.MemRef.may_alias(_memref(initial), _memref(value)), (
            "the partial write must not reach the independently observed initializer"
        )
    assert window(row) == (_base(selected), window(selected)[1] + 32, 32)
    assert window(writer.var) == window(row)
    for value, original_value in (
        (row, original.assigns["row"].var),
        (writer.var, original.assigns["written"].var),
    ):
        memory, parent = _memref(value), _memref(selected)
        assert memory.slot_count_ == parent.slot_count_
        assert (memory.slot_index_ is None) == (parent.slot_index_ is None)
        if memory.slot_index_ is not None:
            assert parent.slot_index_ is not None
            assert ir.structural_equal(memory.slot_index_, parent.slot_index_)
        before_type, after_type = original_value.type, value.type
        assert isinstance(before_type, ir.TileType) and isinstance(after_type, ir.TileType)
        assert after_type.shape == [1, 8] and after_type.dtype == before_type.dtype == ir.DataType.FP32
        assert after_type.memory_space == before_type.memory_space == ir.MemorySpace.Vec
        assert after_type.tile_view == before_type.tile_view

    entry = next(op for op in storage.statements if op.var.same_as(carry.initValue))
    assert isinstance(entry.value, ir.Call) and entry.value.op.name == _MOVE
    assert window(entry.value.args[0]) == window(initial), "carry initialization must read the old data"
    if window(selected) != window(carry):
        assert any(
            window(op.var) == window(carry)
            and isinstance(op.value, ir.Call)
            and window(op.value.args[0]) == window(selected)
            and storage.statements.index(op) > storage.statements.index(writer)
            for op in storage.calls(_MOVE)
        ), "a separate selected window must be copied back after its partial write"
    for name, source in (("old_saved", initial), ("updated_saved", loop.return_vars[0])):
        saved = storage.assigns[name].value
        assert isinstance(saved, ir.Call) and saved.op.name == ir.get_op("tile.store").name
        assert window(saved.args[0]) == window(source)
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    _STORAGE_MODES,
    ids=_STORAGE_MODE_IDS,
)
def test_planned_carry_writer_preserves_a_phi_read_after_the_producer(planner, enable_buffer_ir):
    """The old phi value must survive a producer that will write the carry's storage."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a_gm: pl.Tensor[[4, 8], pl.FP32],
            flag: pl.Scalar[pl.BOOL],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tensor[[8, 8], pl.FP32]:
            initial: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(a_gm, [0, 0], [4, 8])
            for _i, (carry,) in pl.range(2, init_values=(initial,)):
                if flag:
                    chosen = pl.yield_(carry)
                else:
                    chosen = pl.yield_(carry)
                updated: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(chosen, 1.0)
                _old_saved = pl.tile.store(chosen, [0, 0], out)
                result = pl.yield_(updated)
            return pl.tile.store(result, [4, 0], out)

    after = _legalize(Before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    assert len(storage.branches) == 1 and len(storage.loops) == 1
    phi = storage.branches[0].return_vars[0]
    writers = storage.calls(ir.get_op("tile.adds").name)
    assert len(writers) == 1
    writer = writers[0]
    old_store = storage.assigns["_old_saved"]
    assert isinstance(old_store.value, ir.Call) and old_store.value.op.name == ir.get_op("tile.store").name
    reader = old_store.value.args[0]
    assert not ir.MemRef.may_alias(_memref(reader), _memref(writer.var)), (
        "the planned carry write must not overwrite the old value before its store"
    )
    assert storage.statements.index(writer) < storage.statements.index(old_store)
    if _base(reader) != _base(phi):
        snapshots = [op for op in storage.statements if op.var.same_as(reader)]
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert isinstance(snapshot.value, ir.Call) and snapshot.value.op.name == _MOVE
        assert storage.statements.index(snapshot) < storage.statements.index(writer)
        reader = snapshot.value.args[0]
    assert _base(reader) == _base(phi), "the old-value store must read the phi or its preceding snapshot"
    assert ir.structural_equal(_memref(reader).byte_offset_, _memref(phi).byte_offset_)
    assert _memref(reader).size_ == _memref(phi).size_
    _verify_storage(after)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["legacy_storage", "buffer_ir"])
def test_phi_view_loop_initializer_preserves_the_view_read_after_loop(
    planner, enable_buffer_ir, pass_verification_instruments
):
    """A phi rebase propagates through aliases without losing an isolated loop copy."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a_gm: pl.Tensor[[4, 8], pl.FP32],
            b_gm: pl.Tensor[[4, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            a: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(a_gm, [0, 0], [4, 8])
            b: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(b_gm, [0, 0], [4, 8])
            if flag:
                then_value: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(a, 1.0)
                chosen = pl.yield_(then_value)
            else:
                else_value: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.mul(b, 2.0)
                chosen = pl.yield_(else_value)
            viewed: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(chosen, [4, 8])
            renamed: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = viewed
            for _i, (current,) in pl.range(0, 3, init_values=(renamed,)):
                updated: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(current, 1.0)
                carried = pl.yield_(updated)
            observed: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(viewed, carried)
            return pl.tile.store(observed, [0, 0], out)

    with passes.PassContext(
        pass_verification_instruments, memory_planner=planner, enable_buffer_ir=enable_buffer_ir
    ):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before))
        ir.assert_structural_equal(passes.materialize_semantic_aliases()(after), after)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    assert len(storage.loops) == 1
    loop = storage.loops[0]
    phi = storage.branches[0].return_vars[0]
    assert _base(storage.assigns["viewed"].var) == _base(phi)
    assert _base(storage.assigns["renamed"].var) == _base(phi)
    assert _base(loop.iter_args[0]) != _base(phi), "the loop clobbered its independently live initializer"
    entry_value = loop.iter_args[0].initValue
    assert isinstance(entry_value, ir.Var)
    entry = next(stmt for stmt in storage.statements if stmt.var.unique_id == entry_value.unique_id)
    assert isinstance(entry.value, ir.Call) and entry.value.op.name == _MOVE
    assert _base(entry.value.args[0]) == _base(phi), "the entry copy reads the phi's abandoned buffer"
    assert _base(storage.assigns["updated"].var) == _base(loop.iter_args[0])
    observed = storage.assigns["observed"].value
    assert isinstance(observed, ir.Call)
    assert [_base(value) for value in observed.args] == [_base(phi), _base(loop.return_vars[0])]
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
@pytest.mark.parametrize("memory,dtype,target", [("Mat", "FP16", "Left"), ("Acc", "FP32", "Vec")])
@pytest.mark.parametrize("phi_before_loop", [False, True], ids=["phi_in_body", "phi_initializer"])
def test_readonly_matrix_carry_keeps_its_observed_initializer(
    planner, enable_buffer_ir, while_loop, memory, dtype, target, phi_before_loop
):
    """A readonly phi at entry or in the body does not make a loop write its input."""
    tile = f"pl.Tile[[16, 16], pl.{dtype}, pl.Mem.{memory}]"
    read_tile = f"pl.Tile[[16, 16], pl.{dtype}, pl.Mem.{target}]"
    initializer = "selected" if phi_before_loop else "initial"
    header = (
        f"for (current, index) in pl.while_(init_values=({initializer}, 0)):\n"
        "            pl.cond(index < count)"
        if while_loop
        else f"for index, (current,) in pl.range(count, init_values=({initializer},)):"
    )
    entry = (
        """        if flag:
            selected = pl.yield_(initial)
        else:
            selected = pl.yield_(initial)"""
        if phi_before_loop
        else ""
    )
    body = (
        ""
        if phi_before_loop
        else """            if flag:
                selected = pl.yield_(alias)
            else:
                selected = pl.yield_(current)"""
    )
    source = "alias" if phi_before_loop else "selected"
    yielded = (
        f"result, _counter = pl.yield_({source}, index + 1)"
        if while_loop
        else f"result = pl.yield_({source})"
    )
    program = pl.parse_program(f"""
@pl.program
class ReadonlyMatrixCarry:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, initial: {tile}, count: pl.Scalar[pl.INDEX], flag: pl.Scalar[pl.BOOL]
               ) -> tuple[{read_tile}, {read_tile}]:
{entry}
        {header}
            alias = current
{body}
            {yielded}
        old_read = pl.tile.move(initial, target_memory=pl.Mem.{target})
        result_read = pl.tile.move(result, target_memory=pl.Mem.{target})
        return old_read, result_read
""")
    after = _legalize(program, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.loops) == 1 and len(storage.branches) == 1
    function = next(iter(after.functions.values()))
    initial = function.params[0]
    loop = storage.loops[0]
    selected = storage.branches[0].return_vars[0]
    values = [
        loop.iter_args[0].initValue,
        loop.iter_args[0],
        loop.return_vars[0],
        storage.assigns["alias"].var,
        selected,
        _yield(loop.body).value[0],
    ]
    for value in values:
        assert _base(value) == _base(initial)
        assert ir.structural_equal(_memref(value).byte_offset_, _memref(initial).byte_offset_)
        assert _memref(value).size_ == _memref(initial).size_
    _assert_canonical_arms(storage)
    moves = storage.calls(_MOVE)
    assert len(moves) == 2, "only the two explicit post-loop observations need copies"
    for move in moves:
        assert isinstance(move.value, ir.Call)
        assert _base(move.value.args[0]) == _base(initial)
        assert ir.structural_equal(_memref(move.value.args[0]).byte_offset_, _memref(initial).byte_offset_)
    expected_allocations = {
        _base(initial),
        _base(storage.assigns["old_read"].var),
        _base(storage.assigns["result_read"].var),
    }
    assert {op.var.unique_id for op in storage.calls(_ALLOC)} == expected_allocations, (
        "a readonly carry must not allocate entry isolation or snapshot storage"
    )
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
def test_sequential_conditional_accumulators_share_their_required_window(planner, enable_buffer_ir):
    """A later pass-through arm does not make the preceding accumulator independently observable."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            left: pl.Tile[[16, 16], pl.FP16, pl.Mem.Left],
            right: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            flag1: pl.Scalar[pl.BOOL],
            flag2: pl.Scalar[pl.BOOL],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            seed: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(left, right)
            if flag1:
                updated: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(seed, left, right)
                first = pl.yield_(updated)
            else:
                first = pl.yield_(seed)
            if flag2:
                updated2: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(first, left, right)
                second = pl.yield_(updated2)
            else:
                second = pl.yield_(first)
            observed = pl.tile.move(second, target_memory=pl.Mem.Vec)
            saved = pl.tile.store(observed, [0, 0], out)
            return saved

    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        initialized = passes.init_mem_ref()(Before)
        after = passes.materialize_semantic_aliases()(initialized)
    original, storage = _Storage(initialized), _Storage(after)
    assert len(storage.branches) == 2 and not storage.loops
    _assert_canonical_arms(storage)
    seed = storage.assigns["seed"].var
    pairs = [
        (storage.assigns[name].var, original.assigns[name].var) for name in ("seed", "updated", "updated2")
    ]
    pairs.extend(
        (branch.return_vars[0], before_branch.return_vars[0])
        for branch, before_branch in zip(storage.branches, original.branches, strict=True)
    )
    for value, before in pairs:
        assert _base(value) == _base(seed)
        assert ir.structural_equal(_memref(value).byte_offset_, _memref(seed).byte_offset_)
        assert _memref(value).size_ == _memref(seed).size_ == 1024
        tile, before_tile = value.type, before.type
        assert isinstance(tile, ir.TileType) and isinstance(before_tile, ir.TileType)
        assert tile.shape == before_tile.shape == [16, 16]
        assert tile.dtype == before_tile.dtype == ir.DataType.FP32
        assert tile.memory_space == before_tile.memory_space == ir.MemorySpace.Acc
        assert tile.tile_view == before_tile.tile_view
    for move in storage.calls(_MOVE):
        assert isinstance(move.value, ir.Call)
        source_type, target_type = move.value.args[0].type, move.var.type
        assert isinstance(source_type, ir.TileType) and isinstance(target_type, ir.TileType)
        assert (source_type.memory_space, target_type.memory_space) != (
            ir.MemorySpace.Acc,
            ir.MemorySpace.Acc,
        )
    observed = storage.assigns["observed"].value
    assert isinstance(observed, ir.Call) and observed.op.name == _MOVE
    assert observed.args[0].same_as(storage.branches[1].return_vars[0])
    _verify_storage(after)


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


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("while_loop", [False, True])
@pytest.mark.parametrize("memory", ["Mat", "Left", "Right", "Acc"])
def test_readonly_same_space_carry_swap_selects_existing_handles(planner, while_loop, memory):
    tile = f"pl.Tile[[16, 16], pl.FP16, pl.Mem.{memory}]"
    if memory == "Acc":
        tile = "pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc]"
        producer = "pl.tile.matmul(left, right)"
    elif memory == "Mat":
        producer = "pl.tile.load(gm, [0, 0], [16, 16], target_memory=pl.Mem.Mat)"
    else:
        producer = f"pl.tile.move(mat, target_memory=pl.Mem.{memory})"
    header = (
        "for (a, b) in pl.while_(init_values=(first, second)):\n            pl.cond(flag)"
        if while_loop
        else "for _i, (a, b) in pl.range(0, 2, init_values=(first, second)):"
    )
    program = pl.parse_program(f"""
@pl.program
class ReadonlySwap:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, gm: pl.Tensor[[16, 16], pl.FP16],
               left: pl.Tile[[16, 16], pl.FP16, pl.Mem.Left],
               right: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right],
               flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[16, 16], pl.FP16]:
        mat: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tile.load(
            gm, [0, 0], [16, 16], target_memory=pl.Mem.Mat
        )
        first: {tile} = {producer}
        second: {tile} = {producer}
        {header}
            r_a, r_b = pl.yield_(b, a)
        return gm
""")
    after = _legalize(program, planner)
    storage = _Storage(after)
    loop = storage.loops[0]
    first, second = (storage.assigns[name].var for name in ("first", "second"))
    assert not ir.MemRef.may_alias(_memref(first), _memref(second))
    for carry, initial in zip(loop.iter_args, (first, second), strict=True):
        assert isinstance(carry.type, ir.TileType) and carry.type.memref is None
        assert carry.initValue.same_as(initial)
    for result in loop.return_vars:
        assert isinstance(result.type, ir.TileType) and result.type.memref is None
    backedge = _yield(loop.body).value
    assert backedge[0].same_as(loop.iter_args[1])
    assert backedge[1].same_as(loop.iter_args[0])
    for move in storage.calls(_MOVE):
        assert isinstance(move.value, ir.Call)
        source_type, target_type = move.value.args[0].type, move.var.type
        assert isinstance(source_type, ir.TileType) and isinstance(target_type, ir.TileType)
        assert source_type.memory_space != target_type.memory_space
    _verify_storage(after)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("while_loop", [False, True])
@pytest.mark.parametrize("loop_in_then", [False, True])
def test_sibling_branch_read_does_not_isolate_local_accumulator(planner, while_loop, loop_in_then):
    header = (
        "for (current,) in pl.while_(init_values=(acc,)):\n    pl.cond(flag)"
        if while_loop
        else "for _i, (current,) in pl.range(0, 2, init_values=(acc,)):"
    )
    loop_arm = f"""{header}
    updated: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(current, left, right)
    accumulated = pl.yield_(updated)
final: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.move(accumulated, target_memory=pl.Mem.Vec)
stored: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(final, [0, 0], out)
result = pl.yield_(stored)"""
    read_arm = """\
previous: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.move(acc, target_memory=pl.Mem.Vec)
saved: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(previous, [0, 0], out)
result = pl.yield_(saved)"""
    then_arm, else_arm = (loop_arm, read_arm) if loop_in_then else (read_arm, loop_arm)
    program = pl.parse_program(f"""
@pl.program
class SiblingRead:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, left: pl.Tile[[16, 16], pl.FP16, pl.Mem.Left],
               right: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
               flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[16, 16], pl.FP32]:
        acc: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(left, right)
        if flag:
{indent(then_arm, "            ")}
        else:
{indent(else_arm, "            ")}
        return result
""")
    after = _legalize(program, planner)
    storage = _Storage(after)
    assert _base(storage.loops[0].iter_args[0]) == _base(storage.assigns["acc"].var)
    assert not any(statement.var.name_hint.startswith("carry_input_") for statement in storage.statements)
    _verify_storage(after)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("outer_while", [False, True])
@pytest.mark.parametrize("inner_while", [False, True])
@pytest.mark.parametrize("local_seed", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_nested_accumulator_isolation_respects_definition_scope(
    planner, outer_while, inner_while, local_seed, alias
):
    """A recreated seed is reusable; a seed surviving outer iterations is live."""
    acc_type = "pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc]"
    seed = f"acc: {acc_type} = pl.tile.matmul(left, right)"
    outer_header = (
        "for (dst,) in pl.while_(init_values=(out,)):\n            pl.cond(flag)"
        if outer_while
        else "for _outer, (dst,) in pl.range(0, 2, init_values=(out,)):"
    )
    inner_header = (
        "for (current,) in pl.while_(init_values=(initial,)):\n                pl.cond(flag)"
        if inner_while
        else "for _inner, (current,) in pl.range(0, 2, init_values=(initial,)):"
    )
    initial = "pl.tile.reshape(acc, [16, 16])" if alias else "acc"
    program = pl.parse_program(f"""
@pl.program
class NestedAccumulator:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, left: pl.Tile[[16, 16], pl.FP16, pl.Mem.Left],
               right: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
               flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[16, 16], pl.FP32]:
{indent(seed if not local_seed else "", "        ")}
        {outer_header}
{indent(seed if local_seed else "", "            ")}
            initial: {acc_type} = {initial}
            previous: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.move(acc, target_memory=pl.Mem.Vec)
            saved: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(previous, [0, 0], dst)
            {inner_header}
                updated: {acc_type} = pl.tile.matmul_acc(current, left, right)
                accumulated = pl.yield_(updated)
            final: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.move(
                accumulated, target_memory=pl.Mem.Vec
            )
            stored: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(final, [0, 0], saved)
            result = pl.yield_(stored)
        return result
""")
    if not local_seed:
        with pytest.raises(ValueError, match=_LOOP_INPUT_CONFLICT):
            _legalize(program, planner)
        return
    after = _legalize(program, planner)
    storage = _Storage(after)
    assert _base(storage.loops[1].iter_args[0]) == _base(storage.assigns["acc"].var)
    assert not any(statement.var.name_hint.startswith("carry_input_") for statement in storage.statements)
    _verify_storage(after)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        ir.assert_structural_equal(passes.materialize_semantic_aliases()(after), after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("inner_while", [False, True], ids=["inner_for", "inner_while"])
def test_nested_loop_phi_initializer_preserves_the_outer_carry_observer(
    planner, enable_buffer_ir, inner_while
):
    """A provisional phi address cannot hide the old outer carry from inner writes."""
    header = (
        "for (p, index) in pl.while_(init_values=(selected, 0)):\n                pl.cond(index < count)"
        if inner_while
        else "for index, (p,) in pl.range(count, init_values=(selected,)):"
    )
    yielded = (
        "inner_result, _counter = pl.yield_(inner_phi, index + 1)"
        if inner_while
        else "inner_result = pl.yield_(inner_phi)"
    )
    before = pl.parse_program(f"""
@pl.program
class NestedPhiInitializer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a_gm: pl.Tensor[[4, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[8, 8], pl.FP32]], count: pl.Scalar[pl.INDEX],
               select_flag: pl.Scalar[pl.BOOL], update_flag: pl.Scalar[pl.BOOL]
               ) -> pl.Tensor[[8, 8], pl.FP32]:
        initial = pl.tile.load(a_gm, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        for _outer, (q,) in pl.range(2, init_values=(initial,)):
            if select_flag:
                selected = pl.yield_(q)
            else:
                selected = pl.yield_(q)
            {header}
                if update_flag:
                    changed = pl.tile.add(p, 1.0)
                    inner_phi = pl.yield_(changed)
                else:
                    inner_phi = pl.yield_(p)
                {yielded}
            old_saved = pl.tile.store(q, [0, 0], out)
            outer_result = pl.yield_(inner_result)
        return pl.tile.store(outer_result, [4, 0], out)
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        initialized = passes.init_mem_ref()(before)
        legalized = passes.materialize_semantic_aliases()(initialized)
        # Legacy PYPTO schedules remaining loop-boundary copies after packing.
        after = passes.memory_reuse()(legalized)
    storage = _Storage(after)
    assert len(storage.loops) == len(storage.branches) == 2
    outer, inner = storage.loops
    common, updated = storage.branches
    assert isinstance(outer, ir.ForStmt)
    assert isinstance(inner, ir.WhileStmt if inner_while else ir.ForStmt)
    q, p = outer.iter_args[0], inner.iter_args[0]
    selected = common.return_vars[0]

    def same_window(left, right):
        lhs, rhs = _memref(left), _memref(right)
        assert ir.MemRef.same_allocation(lhs, rhs)
        assert ir.structural_equal(lhs.byte_offset_, rhs.byte_offset_)
        assert lhs.size_ == rhs.size_
        assert lhs.slot_count_ == rhs.slot_count_
        assert (lhs.slot_index_ is None) == (rhs.slot_index_ is None)
        if lhs.slot_index_ is not None:
            assert rhs.slot_index_ is not None
            assert ir.structural_equal(lhs.slot_index_, rhs.slot_index_)

    _assert_canonical_arms(storage)
    same_window(selected, q)
    assert not ir.MemRef.may_alias(_memref(p), _memref(q)), "inner updates must preserve the old outer value"
    copies = [op for op in storage.calls(_MOVE) if op.var.same_as(p.initValue)]
    assert len(copies) == 1
    entry = copies[0]
    assert isinstance(entry.value, ir.Call) and entry.value.args[0].same_as(selected)
    outer_body = ir.flatten_to_stmts(outer.body)
    assert outer_body.index(entry) < outer_body.index(inner), (
        "isolation must execute before any inner iteration"
    )
    same_window(entry.var, p)
    same_window(inner.return_vars[0], p)
    same_window(_yield(inner.body).value[0], p)
    same_window(updated.return_vars[0], p)
    same_window(_yield(outer.body).value[0], q)
    old_store = storage.assigns["old_saved"].value
    assert isinstance(old_store, ir.Call) and old_store.op.name == ir.get_op("tile.store").name
    assert old_store.args[0].same_as(q), "the post-inner observation must read the original outer carry"

    original_tile = _Storage(initialized).assigns["initial"].var.type
    assert isinstance(original_tile, ir.TileType)
    for value in (q, selected, entry.var, p, inner.return_vars[0], updated.return_vars[0]):
        tile = value.type
        assert isinstance(tile, ir.TileType)
        assert tile.shape == original_tile.shape == [4, 8]
        assert tile.dtype == original_tile.dtype == ir.DataType.FP32
        assert tile.memory_space == original_tile.memory_space == ir.MemorySpace.Vec
        assert tile.tile_view == original_tile.tile_view
        assert _memref(value).size_ == 128
        assert _memref(value).slot_count_ == 1 and _memref(value).slot_index_ is None
    # count=0 takes no body edge: the inner result must still name the entry
    # snapshot of selected, rather than an uninitialized branch-produced value.
    same_window(inner.return_vars[0], p.initValue)
    if isinstance(inner, ir.WhileStmt):
        assert isinstance(inner.iter_args[1].initValue, ir.ConstInt)
        assert inner.iter_args[1].initValue.value == 0
    else:
        assert isinstance(inner, ir.ForStmt)
        assert isinstance(inner.start, ir.ConstInt) and inner.start.value == 0
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [
        (passes.MemoryPlanner.PYPTO, True),
        (passes.MemoryPlanner.DSA_RP, True),
        (passes.MemoryPlanner.PTOAS, True),
        (passes.MemoryPlanner.PTOAS, False),
    ],
    ids=["buffer_pypto", "buffer_dsa", "buffer_ptoas", "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True])
@pytest.mark.parametrize("producer_in_then", [False, True])
def test_if_edge_copy_waits_for_a_live_old_carry_observer(
    planner, enable_buffer_ir, while_loop, producer_in_then
):
    """An in-loop old-value read must precede the carry's actual writeback."""
    header = (
        "for (current,) in pl.while_(init_values=(initial,)):\n            pl.cond(flag)"
        if while_loop
        else "for _i, (current,) in pl.range(0, 2, init_values=(initial,)):"
    )
    producer = f"updated: {_TILE} = pl.tile.add(current, 1.0)\nselected: {_TILE} = pl.yield_(updated)"
    passthrough = f"selected: {_TILE} = pl.yield_(current)"
    then_arm, else_arm = (producer, passthrough) if producer_in_then else (passthrough, producer)
    program = pl.parse_program(f"""
@pl.program
class LateCarryWriteback:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, gm: pl.Tensor[[4, 8], pl.FP32],
               out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
               flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[4, 8], pl.FP32]:
        initial: {_TILE} = pl.tile.load(gm, [0, 0], [4, 8])
        {header}
            if flag:
{indent(then_arm, "                ")}
            else:
{indent(else_arm, "                ")}
            old_saved: pl.Tensor[[4, 8], pl.FP32] = pl.tile.store(current, [0, 0], out)
            accumulated = pl.yield_(selected)
        result: pl.Tensor[[4, 8], pl.FP32] = pl.tile.store(accumulated, [0, 0], out)
        return result
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        initialized = passes.init_mem_ref()(program)
        after = passes.materialize_semantic_aliases()(initialized)
    before_storage, storage = _Storage(initialized), _Storage(after)
    assert len(storage.loops) == 1 and len(storage.branches) == 1
    loop, branch = storage.loops[0], storage.branches[0]
    current, selected = loop.iter_args[0], branch.return_vars[0]
    assert not ir.MemRef.may_alias(_memref(selected), _memref(current))
    assert not ir.MemRef.may_alias(_memref(storage.assigns["updated"].var), _memref(current))
    _assert_canonical_arms(storage)

    observer = storage.assigns["old_saved"]
    assert isinstance(observer.value, ir.Call) and observer.value.op.name == ir.get_op("tile.store").name
    assert observer.value.args[0].same_as(current)
    assert isinstance(loop.body, ir.SeqStmts)
    body = list(loop.body.stmts)
    observer_index = next(
        i for i, stmt in enumerate(body) if isinstance(stmt, ir.AssignStmt) and stmt.var.same_as(observer.var)
    )
    carried = _yield(loop.body).value[0]
    late_index, late_copy = next(
        (i, stmt)
        for i, stmt in enumerate(body)
        if isinstance(stmt, ir.AssignStmt) and stmt.var.same_as(carried)
    )
    assert observer_index < late_index
    assert isinstance(late_copy.value, ir.Call) and late_copy.value.op.name == _MOVE
    assert late_copy.value.args[0].same_as(selected)
    for value in (late_copy.var, carried, loop.return_vars[0]):
        assert _base(value) == _base(current)
        assert ir.structural_equal(_memref(value).byte_offset_, _memref(current).byte_offset_)
        assert _memref(value).size_ == _memref(current).size_

    original_type = before_storage.branches[0].return_vars[0].type
    assert isinstance(original_type, ir.TileType) and isinstance(selected.type, ir.TileType)
    assert selected.type.shape == original_type.shape == [4, 8]
    assert selected.type.dtype == original_type.dtype == ir.DataType.FP32
    assert selected.type.memory_space == original_type.memory_space == ir.MemorySpace.Vec
    assert selected.type.tile_view == original_type.tile_view
    _verify_storage(after)


@pytest.mark.parametrize("shared_ddr_window", [False, True], ids=["disjoint_vec", "shared_ddr"])
def test_known_tensor_write_cannot_hide_a_shared_ddr_tile_clobber(shared_ddr_window):
    """Known GM writes participate in storage proofs without poisoning unrelated Vec storage."""
    span = ir.Span.unknown()
    shared = ir.Var("shared", ir.PtrType(), span)
    vec_base = ir.Var("vec_base", ir.PtrType(), span)
    request = ir.Var("request", ir.PtrType(), span)
    gm_window = ir.MemRef(shared, 0, 128, span)
    tile_window = gm_window if shared_ddr_window else ir.MemRef(vec_base, 0, 128, span)
    space = ir.MemorySpace.DDR if shared_ddr_window else ir.MemorySpace.Vec
    gm_type = ir.TensorType([4, 8], ir.DataType.FP32, gm_window)
    source_type = ir.TileType([4, 8], ir.DataType.FP32, tile_window, memory_space=space)
    result_type = ir.TileType([4, 8], ir.DataType.FP32, ir.MemRef(request, 0, 128, span), memory_space=space)
    tensor = ir.Var("tensor", gm_type, span)
    source = ir.Var("source", source_type, span)
    result = ir.Var("selected", result_type, span)
    flag = ir.Var("flag", ir.ScalarType(ir.DataType.BOOL), span)
    branch = ir.IfStmt(flag, ir.YieldStmt([source], span), ir.YieldStmt([source], span), [result], span)
    indices = ir.MakeTuple(
        [ir.ConstInt(0, ir.DataType.INDEX, span), ir.ConstInt(0, ir.DataType.INDEX, span)], span
    )
    write = ir.Call(
        ir.get_op("tensor.write"),
        [tensor, indices, ir.ConstFloat(1.0, ir.DataType.FP32, span)],
        gm_type,
        span,
    )
    function = ir.Function(
        "kernel",
        [shared, vec_base, (tensor, ir.ParamDirection.InOut), source, flag],
        [ir.TileType([4, 8], ir.DataType.FP32, memory_space=space)],
        ir.SeqStmts([branch, ir.EvalStmt(write, span), ir.ReturnStmt([result], span)], span),
        span,
        type=ir.FunctionType.InCore,
    )
    before = ir.Program([function], "KnownGMWrite", span)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        if shared_ddr_window:
            # Tile DDR is legal in initialized typed IR, but DDR -> DDR has
            # no tile.move implementation to preserve this independently live r.
            with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
                passes.materialize_semantic_aliases()(before)
            return
        after = passes.materialize_semantic_aliases()(before)

    storage = _Storage(after)
    assert len(storage.branches) == 1
    _assert_canonical_arms(storage)
    selected = storage.branches[0].return_vars[0]
    assert ir.MemRef.same_allocation(_memref(selected), tile_window)
    assert ir.structural_equal(_memref(selected).byte_offset_, tile_window.byte_offset_)
    assert _memref(selected).size_ == tile_window.size_ == 128
    assert not storage.calls(_MOVE), "a known, disjoint GM write must not force a Vec snapshot"
    updated_function = next(iter(after.functions.values()))
    assert isinstance(updated_function.body, ir.SeqStmts)
    effects = [stmt for stmt in updated_function.body.stmts if isinstance(stmt, ir.EvalStmt)]
    assert len(effects) == 1
    assert isinstance(effects[0].expr, ir.Call) and effects[0].expr.op.name == ir.get_op("tensor.write").name
    assert effects[0].expr.args[0].same_as(tensor)
    returned = updated_function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt) and returned.value[0].same_as(selected)


# Mutable common-window continuations must preserve old SSA observations.
@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
def test_three_conditional_mat_gathers_keep_their_native_destination(planner, enable_buffer_ir):
    """The three updates model an unrolled row-gather chain without Mat copies."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            pool: pl.Tensor[[16, 16], pl.FP16],
            flag1: pl.Scalar[pl.BOOL],
            flag2: pl.Scalar[pl.BOOL],
            flag3: pl.Scalar[pl.BOOL],
        ) -> pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat]:
            initial = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            if flag1:
                update1 = pl.tile.gather_row(initial, pool, [0, 0], [1, 0], [1, 16])
                first = pl.yield_(update1)
            else:
                first = pl.yield_(initial)
            if flag2:
                update2 = pl.tile.gather_row(first, pool, [1, 0], [2, 0], [1, 16])
                second = pl.yield_(update2)
            else:
                second = pl.yield_(first)
            if flag3:
                update3 = pl.tile.gather_row(second, pool, [2, 0], [3, 0], [1, 16])
                third = pl.yield_(update3)
            else:
                third = pl.yield_(second)
            return third

    after = _legalize(Before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.branches) == 3 and not storage.loops
    _assert_canonical_arms(storage)
    target = _memref(storage.assigns["initial"].var)
    writers = storage.calls(ir.get_op("tile.gather_row").name)
    assert len(writers) == 3 and not storage.calls(_MOVE)
    for writer in writers:
        assert isinstance(writer.value, ir.Call)
        for value in (writer.var, writer.value.args[0]):
            memory = _memref(value)
            assert ir.MemRef.same_allocation(memory, target)
            assert ir.structural_equal(memory.byte_offset_, target.byte_offset_)
            assert memory.size_ == target.size_ == 512
            assert memory.slot_count_ == target.slot_count_
    for branch in storage.branches:
        result = branch.return_vars[0]
        assert ir.MemRef.same_allocation(_memref(result), target)
        tile = result.type
        assert isinstance(tile, ir.TileType)
        assert tile.shape == [16, 16] and tile.dtype == ir.DataType.FP16
        assert tile.memory_space == ir.MemorySpace.Mat
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("memory", ["Vec", "Mat"])
def test_mutable_common_result_preserves_an_independently_observed_source(planner, enable_buffer_ir, memory):
    """Selecting a/a does not authorize a later DPS update to overwrite old a."""
    before = pl.parse_program(f"""
@pl.program
class CommonResultObserver:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[16, 16], pl.FP16], flag: pl.Scalar[pl.BOOL]
               ) -> tuple[pl.Tile[[16, 16], pl.FP16, pl.Mem.{memory}],
                          pl.Tile[[16, 16], pl.FP16, pl.Mem.{memory}]]:
        initial = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.{memory})
        if flag:
            selected = pl.yield_(initial)
        else:
            selected = pl.yield_(initial)
        changed = pl.tile.gather_row(selected, pool, [0, 0], [1, 0], [1, 16])
        return changed, initial
""")
    if memory == "Mat":
        with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
            _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
        return
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    initial = storage.assigns["initial"].var
    selected = storage.branches[0].return_vars[0]
    assert not ir.MemRef.may_alias(_memref(initial), _memref(selected))
    writer = storage.assigns["changed"]
    assert isinstance(writer.value, ir.Call) and writer.value.args[0].same_as(selected)
    assert ir.MemRef.same_allocation(_memref(writer.var), _memref(selected))
    moves = storage.calls(_MOVE)
    assert len(moves) == 2
    for move in moves:
        assert isinstance(move.value, ir.Call) and move.value.args[0].same_as(initial)
        assert ir.MemRef.same_allocation(_memref(move.var), _memref(selected))
    function = next(iter(after.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt) and returned.value[-1].same_as(initial)
    _verify_storage(after)


def _accumulator_row_writeback_program(while_loop, old_observer):
    header = (
        "for (current, index) in pl.while_(init_values=(row, 0)):\n            pl.cond(index < count)"
        if while_loop
        else "for index, (current,) in pl.range(count, init_values=(row,)):"
    )
    yielded = (
        "accumulated, _counter = pl.yield_(updated, index + 1)"
        if while_loop
        else "accumulated = pl.yield_(updated)"
    )
    observation = "old_saved = out"
    if old_observer != "none":
        original = "row" if old_observer == "row" else "initial"
        observation = (
            f"old_vec = pl.tile.move({original}, target_memory=pl.Mem.Vec)\n"
            "old_saved = pl.tile.store(old_vec, [0, 0], out)"
        )
    return pl.parse_program(f"""
@pl.program
class AccumulatorRowWriteback:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, left: pl.Tile[[64, 32], pl.FP16, pl.Mem.Left],
               row_left: pl.Tile[[16, 32], pl.FP16, pl.Mem.Left],
               right: pl.Tile[[32, 16], pl.FP16, pl.Mem.Right],
               out: pl.Out[pl.Tensor[[128, 16], pl.FP32]],
               count: pl.Scalar[pl.INDEX]) -> pl.Tensor[[128, 16], pl.FP32]:
        initial = pl.tile.matmul(left, right)
        row = pl.tile.slice(initial, [16, 16], [16, 0])
        {header}
            updated = pl.tile.matmul_acc(current, row_left, right)
            {yielded}
{indent(observation, "        ")}
        assembled = pl.tile.assemble(initial, accumulated, [16, 0])
        observed = pl.tile.move(assembled, target_memory=pl.Mem.Vec)
        return pl.tile.store(observed, [64, 0], old_saved)
""")


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [(passes.MemoryPlanner.PYPTO, True), (passes.MemoryPlanner.PTOAS, False)],
    ids=["buffer_pypto", "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
@pytest.mark.parametrize("old_observer", ["none", "row", "whole"])
def test_row_carry_writeback_masks_only_its_preserved_target_read(
    planner, enable_buffer_ir, while_loop, old_observer
):
    """Assemble continuation is safe; separately reading the old row or full tile is not."""
    before = _accumulator_row_writeback_program(while_loop, old_observer)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        initialized = passes.init_mem_ref()(before)
        if old_observer != "none":
            with pytest.raises(
                ValueError,
                match=f"{_LOOP_INPUT_CONFLICT}|requires its output to be disjoint",
            ):
                passes.materialize_semantic_aliases()(initialized)
            return
        after = passes.materialize_semantic_aliases()(initialized)

    original, storage = _Storage(initialized), _Storage(after)
    assert len(storage.loops) == 1
    loop = storage.loops[0]
    row = storage.assigns["row"].var
    for value in (loop.iter_args[0], loop.return_vars[0], _yield(loop.body).value[0]):
        assert ir.MemRef.same_allocation(_memref(value), _memref(row))
        assert ir.structural_equal(_memref(value).byte_offset_, _memref(row).byte_offset_)
        assert _memref(value).size_ == _memref(row).size_
        tile, row_type = value.type, row.type
        assert isinstance(tile, ir.TileType) and isinstance(row_type, ir.TileType)
        assert tile.shape == row_type.shape == [16, 16]
        assert tile.dtype == row_type.dtype == ir.DataType.FP32
        assert tile.memory_space == row_type.memory_space == ir.MemorySpace.Acc
    assembled = storage.assigns["assembled"]
    assert isinstance(assembled.value, ir.Call)
    assert assembled.value.op.name == ir.get_op("tile.assemble").name
    assert assembled.value.args[0].same_as(storage.assigns["initial"].var)
    assert assembled.value.args[1].same_as(loop.return_vars[0])
    assembled_type = assembled.var.type
    original_assembled_type = original.assigns["assembled"].var.type
    assert isinstance(assembled_type, ir.TileType) and isinstance(original_assembled_type, ir.TileType)
    assert assembled_type.tile_view == original_assembled_type.tile_view
    assert not any(op.var.name_hint.startswith("carry_input_") for op in storage.statements)
    for transfer in storage.calls(_MOVE):
        assert isinstance(transfer.value, ir.Call)
        source_type, target_type = transfer.value.args[0].type, transfer.var.type
        assert isinstance(source_type, ir.TileType) and isinstance(target_type, ir.TileType)
        assert (source_type.memory_space, target_type.memory_space) != (
            ir.MemorySpace.Acc,
            ir.MemorySpace.Acc,
        )
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("readonly", [False, True], ids=["inout_caller", "readonly_caller"])
def test_common_mat_continuation_respects_the_caller_mutation_contract(planner, enable_buffer_ir, readonly):
    tile = "pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat]"
    parameter = tile if readonly else f"pl.InOut[{tile}]"
    before = pl.parse_program(f"""
@pl.program
class CallerMatContinuation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[16, 16], pl.FP16], initial: {parameter},
               flag1: pl.Scalar[pl.BOOL], flag2: pl.Scalar[pl.BOOL]) -> {tile}:
        if flag1:
            selected = pl.yield_(initial)
        else:
            selected = pl.yield_(initial)
        if flag2:
            update1 = pl.tile.gather_row(selected, pool, [0, 0], [1, 0], [1, 16])
            first = pl.yield_(update1)
        else:
            first = pl.yield_(selected)
        update2 = pl.tile.gather_row(first, pool, [1, 0], [2, 0], [1, 16])
        return update2
""")
    if readonly:
        with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
            _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
        return
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    function = next(iter(after.functions.values()))
    initial = next(value for value in function.params if value.name_hint == "initial")
    target = _memref(initial)
    assert not storage.calls(_MOVE)
    assert len(storage.calls(ir.get_op("tile.gather_row").name)) == 2
    values = [branch.return_vars[0] for branch in storage.branches]
    values.extend(storage.assigns[name].var for name in ("update1", "update2"))
    for value in values:
        memory = _memref(value)
        assert ir.MemRef.same_allocation(memory, target)
        assert ir.structural_equal(memory.byte_offset_, target.byte_offset_)
        assert memory.size_ == target.size_ == 512
        assert memory.slot_count_ == target.slot_count_
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("memory", ["Vec", "Mat"])
@pytest.mark.parametrize("swap_slots", [False, True], ids=["mutable_first", "mutable_last"])
@pytest.mark.parametrize("overlapping_peer", [True, False], ids=["same_window", "disjoint_slot"])
def test_mutable_common_output_preserves_a_live_peer_result(
    planner, enable_buffer_ir, memory, swap_slots, overlapping_peer
):
    """A native update must preserve the other result's original payload."""
    names = ["mutable_result", "preserved"]
    values = ["initial", "initial" if overlapping_peer else "other"]
    if swap_slots:
        names.reverse()
        values.reverse()
    tile = f"pl.Tile[[16, 16], pl.FP16, pl.Mem.{memory}]"
    before = pl.parse_program(f"""
@pl.program
class ParallelCommonResults:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[32, 16], pl.FP16], flag: pl.Scalar[pl.BOOL]
               ) -> tuple[{tile}, {tile}]:
        initial: pl.Tile[[16, 16], pl.FP16, pl.MemRef("peer_arena", slots=2)[0], pl.Mem.{memory}] = (
            pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.{memory}))
        other: pl.Tile[[16, 16], pl.FP16, pl.MemRef("peer_arena", slots=2)[1], pl.Mem.{memory}] = (
            pl.tile.load(pool, [16, 0], [16, 16], target_memory=pl.Mem.{memory}))
        if flag:
            {", ".join(names)} = pl.yield_({", ".join(values)})
        else:
            {", ".join(names)} = pl.yield_({", ".join(values)})
        changed = pl.tile.gather_row(mutable_result, pool, [0, 0], [1, 0], [1, 16])
        return changed, preserved
""")
    if memory == "Mat" and overlapping_peer:
        with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
            _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
        return
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.branches) == 1
    _assert_canonical_arms(storage)
    results = dict(zip(names, storage.branches[0].return_vars, strict=True))
    initial = storage.assigns["initial"].var
    other = storage.assigns["other"].var
    old_peer = initial if overlapping_peer else other
    preserved, mutable = results["preserved"], results["mutable_result"]

    def window(value):
        memory = _memref(value)
        assert isinstance(memory.byte_offset_, ir.ConstInt) and memory.size_ == 512
        return memory.base_.unique_id, memory.byte_offset_.value, memory.size_

    for slot, source in enumerate((initial, other)):
        memory = _memref(source)
        assert memory.slot_count_ == 2 and isinstance(memory.slot_index_, ir.ConstInt)
        assert memory.slot_index_.value == slot and window(source)[1] == slot * 512
    assert not ir.MemRef.may_alias(_memref(mutable), _memref(preserved))
    writer = storage.assigns["changed"]
    assert isinstance(writer.value, ir.Call) and writer.value.args[0].same_as(mutable)
    assert window(writer.var) == window(mutable)
    assert not ir.MemRef.may_alias(_memref(writer.var), _memref(preserved))
    for value in (mutable, preserved, writer.var):
        tile_type = value.type
        initial_type = initial.type
        assert isinstance(tile_type, ir.TileType) and isinstance(initial_type, ir.TileType)
        assert tile_type.shape == initial_type.shape == [16, 16]
        assert tile_type.dtype == initial_type.dtype == ir.DataType.FP16
        assert tile_type.memory_space == initial_type.memory_space
        assert tile_type.tile_view == initial_type.tile_view

    # Either readonly result may keep its old window or receive a snapshot.
    # Execute the arm's transfers by storage window so both choices prove the
    # same old payload, including when result slot order is reversed.
    expected = {"mutable_result": "initial", "preserved": "initial" if overlapping_peer else "other"}
    branch = storage.branches[0]
    for body in (branch.then_body, branch.else_body):
        assert body is not None
        payloads = {window(initial): "initial", window(other): "other"}
        arm_statements = body.stmts if isinstance(body, ir.SeqStmts) else [body]
        for statement in arm_statements:
            if not isinstance(statement, ir.AssignStmt):
                continue
            source = statement.value
            if isinstance(source, ir.Call):
                assert source.op.name == _MOVE
                source = source.args[0]
                assert not ir.MemRef.may_alias(_memref(source), _memref(statement.var))
            else:
                assert isinstance(source, (ir.Var, ir.IterArg))
                assert window(source) == window(statement.var)
            assert window(source) in payloads, "a snapshot must read an initialized old value"
            payloads[window(statement.var)] = payloads[window(source)]
        for name, value in zip(names, _yield(body).value, strict=True):
            assert payloads[window(value)] == expected[name]

    copies = storage.calls(_MOVE)
    if overlapping_peer:
        assert copies, "the update must be isolated from the independently returned old value"
    else:
        assert not copies, "different pipeline slots must not force a snapshot"
        assert window(mutable) == window(initial)
        assert window(preserved) == window(old_peer)
        preserved_memory, peer_memory = _memref(preserved), _memref(old_peer)
        assert preserved_memory.slot_count_ == peer_memory.slot_count_ == 2
        preserved_slot, peer_slot = preserved_memory.slot_index_, peer_memory.slot_index_
        assert preserved_slot is not None and peer_slot is not None
        assert ir.structural_equal(preserved_slot, peer_slot)
    function = next(iter(after.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    returned = function.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt) and returned.value[0].same_as(writer.var)
    assert returned.value[-1].same_as(preserved)
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint_caller_windows", "readonly_overlap"])
def test_common_output_window_respects_only_overlapping_readonly_caller_storage(
    planner, enable_buffer_ir, overlap
):
    """An unused In window protects its bytes, without closing a disjoint Out window."""
    span = ir.Span.unknown()
    allocation = ir.Var("caller_storage", ir.PtrType(), span)
    request = ir.Var("result_request", ir.PtrType(), span)
    readonly_window = ir.MemRef(allocation, 0, 512, span)
    output_window = ir.MemRef(allocation, 0 if overlap else 512, 512, span)
    readonly_type = ir.TileType([16, 16], ir.DataType.FP16, readonly_window, memory_space=ir.MemorySpace.Mat)
    output_type = ir.TileType([16, 16], ir.DataType.FP16, output_window, memory_space=ir.MemorySpace.Mat)
    result_type = ir.TileType(
        [16, 16], ir.DataType.FP16, ir.MemRef(request, 0, 512, span), memory_space=ir.MemorySpace.Mat
    )
    readonly = ir.Var("readonly", readonly_type, span)
    output = ir.Var("output", output_type, span)
    result = ir.Var("selected", result_type, span)
    changed = ir.Var("changed", result_type, span)
    pool = ir.Var("pool", ir.TensorType([16, 16], ir.DataType.FP16), span)
    flag = ir.Var("flag", ir.ScalarType(ir.DataType.BOOL), span)
    branch = ir.IfStmt(flag, ir.YieldStmt([output], span), ir.YieldStmt([output], span), [result], span)

    def coordinates(first, second):
        return ir.MakeTuple(
            [ir.ConstInt(first, ir.DataType.INDEX, span), ir.ConstInt(second, ir.DataType.INDEX, span)], span
        )

    gather = ir.Call(
        ir.get_op("tile.gather_row"),
        [result, pool, coordinates(0, 0), coordinates(1, 0), coordinates(1, 16)],
        result_type,
        span,
    )
    function = ir.Function(
        "kernel",
        [allocation, (readonly, ir.ParamDirection.In), (output, ir.ParamDirection.Out), pool, flag],
        [ir.TileType([16, 16], ir.DataType.FP16, memory_space=ir.MemorySpace.Mat)],
        ir.SeqStmts([branch, ir.AssignStmt(changed, gather, span), ir.ReturnStmt([changed], span)], span),
        span,
        type=ir.FunctionType.InCore,
    )
    before = ir.Program([function], "CallerWindowContract", span)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        if overlap:
            # The caller's In contract applies even without a local read of it.
            with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
                passes.materialize_semantic_aliases()(before)
            return
        after = passes.materialize_semantic_aliases()(before)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    assert len(storage.branches) == 1 and not storage.calls(_MOVE)
    values = [storage.branches[0].return_vars[0], storage.assigns["changed"].var]
    for value in values:
        memory = _memref(value)
        assert ir.MemRef.same_allocation(memory, output_window)
        assert ir.structural_equal(memory.byte_offset_, output_window.byte_offset_)
        assert memory.size_ == output_window.size_ == 512
        assert not ir.MemRef.may_alias(memory, readonly_window)
    rewritten = next(iter(after.functions.values()))
    for old, new in zip(function.params, rewritten.params, strict=True):
        if isinstance(old.type, ir.TileType):
            assert ir.structural_equal(_memref(old), _memref(new))
    _verify_storage(after)


# Old-value observations must be reachable in both branch orderings.
@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("work_in_else", [False, True], ids=["work_then", "work_else"])
@pytest.mark.parametrize("old_observer", ["sibling_only", "same_arm", "after_merge"])
def test_nested_mat_continuations_ignore_only_unreachable_old_reads(
    planner, enable_buffer_ir, work_in_else, old_observer
):
    work = """if flag1:
    update1 = pl.tile.gather_row(initial, pool, [0, 0], [1, 0], [1, 16])
    first = pl.yield_(update1)
else:
    first = pl.yield_(initial)
if flag2:
    update2 = pl.tile.gather_row(first, pool, [1, 0], [2, 0], [1, 16])
    second = pl.yield_(update2)
else:
    second = pl.yield_(first)
if flag3:
    update3 = pl.tile.gather_row(second, pool, [2, 0], [3, 0], [1, 16])
    third = pl.yield_(update3)
else:
    third = pl.yield_(second)
"""
    result_type = "pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat]"
    returned = "updated"
    if old_observer == "same_arm":
        work += "old_left = pl.tile.move(initial, target_memory=pl.Mem.Left)\n"
        work += "selected, saved = pl.yield_(third, old_left)"
        sibling = (
            "old_other = pl.tile.move(initial, target_memory=pl.Mem.Left)\n"
            "selected, saved = pl.yield_(initial, old_other)"
        )
        result_type = f"tuple[{result_type}, pl.Tile[[16, 16], pl.FP16, pl.Mem.Left]]"
        returned += ", saved"
    else:
        work += "selected = pl.yield_(third)"
        sibling = "selected = pl.yield_(initial)"
    after_merge = ""
    if old_observer == "after_merge":
        after_merge = "old_after = pl.tile.move(initial, target_memory=pl.Mem.Left)"
        result_type = f"tuple[{result_type}, pl.Tile[[16, 16], pl.FP16, pl.Mem.Left]]"
        returned += ", old_after"
    then_arm, else_arm = (sibling, work) if work_in_else else (work, sibling)
    before = pl.parse_program(f"""
@pl.program
class NestedMatContinuation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[16, 16], pl.FP16], outer: pl.Scalar[pl.BOOL],
               flag1: pl.Scalar[pl.BOOL], flag2: pl.Scalar[pl.BOOL],
               flag3: pl.Scalar[pl.BOOL]) -> {result_type}:
        initial = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        if outer:
{indent(then_arm, "            ")}
        else:
{indent(else_arm, "            ")}
        updated = pl.tile.gather_row(selected, pool, [3, 0], [4, 0], [1, 16])
{indent(after_merge, "        ")}
        return {returned}
""")
    if old_observer != "sibling_only":
        # Neither reference selection nor a canonical destination can preserve
        # the old contents after this in-place writer without a real copy.
        with pytest.raises(
            ValueError,
            match="independently observed older value version|unsupported tile.move memory-space pair",
        ):
            _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
        return
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.branches) == 4 and not storage.loops
    origins = TileHandleOrigins(after)
    original_values = TileHandleOrigins(before).values
    target = _memref(storage.assigns["initial"].var)
    writers = storage.calls(ir.get_op("tile.gather_row").name)
    assert len(writers) == 4 and not storage.calls(_MOVE)
    required_inputs = {
        "update1": "initial",
        "update2": "first",
        "update3": "second",
        "updated": "selected",
    }
    for branch in storage.branches:
        assert branch.else_body is not None
        for result in branch.return_vars:
            origins.assert_window(result, target)
            origins.assert_descriptor(result, original_values[result.name_hint])
        for arm in [branch.then_body, branch.else_body]:
            for value in _yield(arm).value:
                origins.assert_window(value, target)
    for writer in writers:
        assert isinstance(writer.value, ir.Call)
        source = writer.value.args[0]
        assert isinstance(source, (ir.Var, ir.IterArg))
        assert source.unique_id == origins.values[required_inputs[writer.var.name_hint]].unique_id
        for value in (writer.var, writer.value.args[0]):
            origins.assert_window(value, target)
    assert target.size_ == 512
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("recreate_seed", [False, True], ids=["external_seed", "per_iteration_seed"])
def test_mutable_common_result_preserves_the_next_iterations_old_seed(
    planner, enable_buffer_ir, recreate_seed
):
    load = "initial = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.Mat)"
    outside = "" if recreate_seed else load
    inside = load if recreate_seed else ""
    before = pl.parse_program(f"""
@pl.program
class RepeatedMatContinuation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[16, 16], pl.FP16],
               right: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right],
               out: pl.Out[pl.Tensor[[32, 16], pl.FP32]], flag: pl.Scalar[pl.BOOL]
               ) -> pl.Tensor[[32, 16], pl.FP32]:
{indent(outside, "        ")}
        for _index, (current_out,) in pl.range(2, init_values=(out,)):
{indent(inside, "            ")}
            old_left = pl.tile.move(initial, target_memory=pl.Mem.Left)
            old_acc = pl.tile.matmul(old_left, right)
            old_vec = pl.tile.move(old_acc, target_memory=pl.Mem.Vec)
            old_saved = pl.tile.store(old_vec, [0, 0], current_out)
            if flag:
                selected = pl.yield_(initial)
            else:
                selected = pl.yield_(initial)
            changed = pl.tile.gather_row(selected, pool, [0, 0], [1, 0], [1, 16])
            new_left = pl.tile.move(changed, target_memory=pl.Mem.Left)
            new_acc = pl.tile.matmul(new_left, right)
            new_vec = pl.tile.move(new_acc, target_memory=pl.Mem.Vec)
            new_saved = pl.tile.store(new_vec, [16, 0], old_saved)
            result = pl.yield_(new_saved)
        return result
""")
    if not recreate_seed:
        with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
            _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
        return
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.loops) == 1 and len(storage.branches) == 1
    _assert_canonical_arms(storage)
    initial = storage.assigns["initial"].var
    selected = storage.branches[0].return_vars[0]
    changed = storage.assigns["changed"].var
    for value in (selected, changed):
        assert ir.MemRef.same_allocation(_memref(value), _memref(initial))
        assert ir.structural_equal(_memref(value).byte_offset_, _memref(initial).byte_offset_)
        assert _memref(value).size_ == _memref(initial).size_ == 512
    for transfer in storage.calls(_MOVE):
        assert isinstance(transfer.value, ir.Call)
        source, target = transfer.value.args[0].type, transfer.var.type
        assert isinstance(source, ir.TileType) and isinstance(target, ir.TileType)
        assert (source.memory_space, target.memory_space) != (ir.MemorySpace.Mat, ir.MemorySpace.Mat)
    _verify_storage(after)


# A partial-valid insertion cannot discard the target's other physical rows.
@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
def test_partial_valid_row_writeback_keeps_full_loop_writes_private(planner, enable_buffer_ir, while_loop):
    """A full load followed by narrowing must not overwrite the other 15 target rows."""
    header = (
        "for (current, index) in pl.while_(init_values=(row, 0)):\n            pl.cond(index < count)"
        if while_loop
        else "for index, (current,) in pl.range(count, init_values=(row,)):"
    )
    yielded = (
        "accumulated, _counter = pl.yield_(updated, index + 1)"
        if while_loop
        else "accumulated = pl.yield_(updated)"
    )
    before = pl.parse_program(f"""
@pl.program
class PartialValidRowWriteback:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, original: pl.Tensor[[32, 16], pl.FP32],
               replacement: pl.Tensor[[16, 16], pl.FP32],
               out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
               count: pl.Scalar[pl.INDEX]) -> pl.Tensor[[32, 16], pl.FP32]:
        initial = pl.tile.load(original, [0, 0], [32, 16], target_memory=pl.Mem.Vec)
        row = pl.tile.slice(initial, [16, 16], [16, 0], valid_shape=[1, 16])
        {header}
            full = pl.tile.load(replacement, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            updated = pl.tile.set_validshape(full, 1, 16)
            {yielded}
        assembled = pl.tile.assemble(initial, accumulated, [16, 0])
        return pl.tile.store(assembled, [0, 0], out)
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(before))
        materialized = _Storage(after)
        original_window = _memref(materialized.assigns["initial"].var)
        assert not ir.MemRef.may_alias(original_window, _memref(materialized.loops[0].iter_args[0]))
        if planner == passes.MemoryPlanner.PYPTO and not enable_buffer_ir:
            after = passes.memory_reuse()(after)

    storage = _Storage(after)
    assert len(storage.loops) == 1 and not storage.branches
    loop = storage.loops[0]
    initial = storage.assigns["initial"].var
    initial_window = _memref(initial)
    carry = loop.iter_args[0]
    # This separation, rather than the eventual narrow metadata, protects rows
    # 17..31 when the loop's load writes all sixteen physical rows first.
    for value in (
        carry.initValue,
        carry,
        loop.return_vars[0],
        _yield(loop.body).value[0],
        storage.assigns["full"].var,
        storage.assigns["updated"].var,
    ):
        assert not ir.MemRef.may_alias(initial_window, _memref(value))
    for value in (carry.initValue, carry, loop.return_vars[0], _yield(loop.body).value[0]):
        assert ir.MemRef.same_allocation(_memref(value), _memref(carry))
        assert ir.structural_equal(_memref(value).byte_offset_, _memref(carry).byte_offset_)
        tile = value.type
        assert isinstance(tile, ir.TileType) and tile.shape == [16, 16]
        assert tile.tile_view is not None and tile.tile_view.valid_shape == [1, 16]
    full_type = storage.assigns["full"].var.type
    assert isinstance(full_type, ir.TileType) and full_type.shape == [16, 16]
    assert full_type.tile_view is None or full_type.tile_view.valid_shape in ([], [16, 16])
    assembled = storage.assigns["assembled"]
    assert isinstance(assembled.value, ir.Call)
    assert assembled.value.args[0].same_as(initial)
    assert assembled.value.args[1].same_as(loop.return_vars[0])
    assert isinstance(assembled.value.args[2], ir.MakeTuple)
    offsets = assembled.value.args[2].elements
    for offset, expected in zip(offsets, (16, 0), strict=True):
        assert isinstance(offset, ir.ConstInt) and offset.value == expected
    assert ir.MemRef.same_allocation(_memref(assembled.var), initial_window)
    _verify_storage(after)


# Frozen old-read windows keep byte and slot identity despite unused root handles.
@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("separation", ["bytes", "slots"])
@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint_old_read", "overlapping_old_read"])
@pytest.mark.parametrize("earlier_phi", [False, True], ids=["original_handle", "relocated_phi"])
def test_original_old_read_windows_are_not_joined_by_an_unused_root(
    planner, enable_buffer_ir, separation, overlap, earlier_phi
):
    span = ir.Span.unknown()
    arena = ir.Var("arena", ir.PtrType(), span)
    request = ir.Var("request", ir.PtrType(), span)
    prior_request = ir.Var("prior_request", ir.PtrType(), span)
    left_base = ir.Var("left", ir.PtrType(), span)
    slots = 2 if separation == "slots" else 1
    source_slot = ir.ConstInt(0, ir.DataType.INDEX, span) if slots == 2 else None
    observer_slot = ir.ConstInt(0 if overlap else 1, ir.DataType.INDEX, span) if slots == 2 else None
    source_window = ir.MemRef(arena, 0, 512, span, slots=slots, slot=source_slot)
    # InitMemRef resolves a static slot into its physical byte offset too.
    observer_offset = 0 if overlap else 512
    observer_window = ir.MemRef(arena, observer_offset, 512, span, slots=slots, slot=observer_slot)
    root_window = ir.MemRef(arena, 0, 1024, span, slots=slots)
    pool = ir.Var("pool", ir.TensorType([32, 16], ir.DataType.FP16), span)
    flag = ir.Var("flag", ir.ScalarType(ir.DataType.BOOL), span)

    def binding(name, call, window):
        tile = call.type
        assert isinstance(tile, ir.TileType)
        value = ir.Var(
            name, ir.TileType(tile.shape, tile.dtype, window, tile.tile_view, tile.memory_space), span
        )
        return value, ir.AssignStmt(value, call, span)

    _, bridge = binding(
        "_unused_whole_root",
        tile_ops.load(pool, [0, 0], [32, 16], target_memory=ir.MemorySpace.Mat, span=span),
        root_window,
    )
    initial, initialize = binding(
        "initial",
        tile_ops.load(pool, [0, 0], [16, 16], target_memory=ir.MemorySpace.Mat, span=span),
        source_window,
    )
    observed, observe_initialize = binding(
        "observed",
        tile_ops.load(pool, [16, 0], [16, 16], target_memory=ir.MemorySpace.Mat, span=span),
        observer_window,
    )
    initial_type = initial.type
    assert isinstance(initial_type, ir.TileType)
    result_type = ir.TileType(
        initial_type.shape,
        initial_type.dtype,
        ir.MemRef(request, 0, 512, span),
        initial_type.tile_view,
        ir.MemorySpace.Mat,
    )
    prefix: list[ir.Stmt] = [bridge, initialize, observe_initialize]
    if earlier_phi:
        observed_type = observed.type
        assert isinstance(observed_type, ir.TileType)
        prior = ir.Var(
            "prior",
            ir.TileType(
                observed_type.shape,
                observed_type.dtype,
                ir.MemRef(prior_request, 0, 512, span),
                observed_type.tile_view,
                ir.MemorySpace.Mat,
            ),
            span,
        )
        prefix.append(
            ir.IfStmt(flag, ir.YieldStmt([observed], span), ir.YieldStmt([observed], span), [prior], span)
        )
        observation_source = prior
    else:
        observation_source = observed
    selected = ir.Var("selected", result_type, span)
    branch = ir.IfStmt(flag, ir.YieldStmt([initial], span), ir.YieldStmt([initial], span), [selected], span)
    changed = ir.Var("changed", result_type, span)
    update = ir.AssignStmt(
        changed, tile_ops.gather_row(selected, pool, [0, 0], [1, 0], [1, 16], span=span), span
    )
    old_left, observation = binding(
        "old_left",
        tile_ops.move(observation_source, ir.MemorySpace.Left, span=span),
        ir.MemRef(left_base, 0, 512, span),
    )
    function = ir.Function(
        "kernel",
        [arena, request, prior_request, left_base, pool, flag],
        [result_type, old_left.type],
        ir.SeqStmts(
            [
                *prefix,
                branch,
                update,
                observation,
                ir.ReturnStmt([changed, old_left], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.InCore,
    )
    before = ir.Program([function], "OriginalReadWindow", span)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        if overlap:
            with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
                passes.materialize_semantic_aliases()(before)
            return
        after = passes.materialize_semantic_aliases()(before)
    storage = _Storage(after)
    _assert_canonical_arms(storage)
    for value in (storage.branches[-1].return_vars[0], storage.assigns["changed"].var):
        window = _memref(value)
        assert ir.MemRef.same_allocation(window, source_window)
        assert ir.structural_equal(window.byte_offset_, source_window.byte_offset_)
        assert window.size_ == source_window.size_ == 512
        assert window.slot_count_ == source_window.slot_count_ == slots
        if source_slot is not None:
            assert window.slot_index_ is not None
            assert ir.structural_equal(window.slot_index_, source_slot)
    copies = storage.calls(_MOVE)
    assert len(copies) == 1
    assert isinstance(copies[0].value, ir.Call)
    if earlier_phi:
        prior_result = storage.branches[0].return_vars[0]
        assert copies[0].value.args[0].same_as(prior_result)
        assert ir.MemRef.same_allocation(_memref(prior_result), observer_window)
        assert ir.structural_equal(_memref(prior_result).byte_offset_, observer_window.byte_offset_)
        assert _memref(prior_result).slot_count_ == observer_window.slot_count_
    else:
        assert copies[0].value.args[0].same_as(storage.assigns["observed"].var)
    assert _base(copies[0].var) == _base(old_left)
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("alias_first", [False, True], ids=["producer_first", "alias_first"])
def test_later_readonly_result_reuses_newly_common_mat_producer_windows(
    planner, enable_buffer_ir, alias_first
):
    incoming = "alias, loaded" if alias_first else "loaded, alias"
    before = pl.parse_program(f"""
@pl.program
class NewlyCommonReadonlyResults:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[32, 16], pl.FP16], flag: pl.Scalar[pl.BOOL]
               ) -> tuple[pl.Tile[[16, 16], pl.FP16, pl.Mem.Left],
                          pl.Tile[[16, 16], pl.FP16, pl.Mem.Left]]:
        if flag:
            loaded = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            alias = pl.tile.reshape(loaded, [16, 16])
            first, second = pl.yield_({incoming})
        else:
            loaded = pl.tile.load(pool, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
            alias = pl.tile.reshape(loaded, [16, 16])
            first, second = pl.yield_({incoming})
        first_left = pl.tile.move(first, target_memory=pl.Mem.Left)
        second_left = pl.tile.move(second, target_memory=pl.Mem.Left)
        return first_left, second_left
""")
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.branches) == 1
    _assert_canonical_arms(storage)
    first, second = storage.branches[0].return_vars
    first_window, second_window = _memref(first), _memref(second)
    assert ir.MemRef.same_allocation(first_window, second_window)
    assert ir.structural_equal(first_window.byte_offset_, second_window.byte_offset_)
    assert first_window.size_ == second_window.size_ == 512
    transfers = storage.calls(_MOVE)
    assert len(transfers) == 2
    for transfer in transfers:
        assert isinstance(transfer.value, ir.Call)
        assert isinstance(transfer.var.type, ir.TileType)
        assert transfer.var.type.memory_space == ir.MemorySpace.Left
    _verify_storage(after)


# Loop arguments must register their reads after following a relocated initializer.
@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
@pytest.mark.parametrize("observe_old", [False, True], ids=["continuation_only", "old_carry_observer"])
def test_relocated_initializer_registers_the_loop_arguments_old_reads(
    planner, enable_buffer_ir, while_loop, observe_old
):
    """An old carry read remains visible after its initializer's phi moves to another allocation."""
    header = (
        "for (carry, index) in pl.while_(init_values=(seeded, 0)):\n            pl.cond(index < 2)"
        if while_loop
        else "for index, (carry,) in pl.range(2, init_values=(seeded,)):"
    )
    yielded = (
        "_loop_result, _counter = pl.yield_(next_value, index + 1)"
        if while_loop
        else "_loop_result = pl.yield_(next_value)"
    )
    old_read = "            _old_saved = pl.tile.store(carry, [4, 0], _selected_saved)" if observe_old else ""
    output_rows = 8 if observe_old else 4
    before = pl.parse_program(f"""
@pl.program
class RelocatedCarryObserver:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[4, 8], pl.FP32], y: pl.Tensor[[4, 8], pl.FP32],
               initial_flag: pl.Scalar[pl.BOOL], inner_flag: pl.Scalar[pl.BOOL],
               out: pl.Out[pl.Tensor[[{output_rows}, 8], pl.FP32]]) -> pl.Tensor[[{output_rows}, 8], pl.FP32]:
        initial = pl.tile.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        replacement = pl.tile.load(y, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        if initial_flag:
            seeded = pl.yield_(initial)
        else:
            seeded = pl.yield_(initial)
        {header}
            if inner_flag:
                selected = pl.yield_(carry)
            else:
                selected = pl.yield_(carry)
            written = pl.tile.assemble(selected, replacement, [0, 0])
            _selected_saved = pl.tile.store(written, [0, 0], out)
{old_read}
            next_value = pl.tile.full([4, 8], dtype=pl.FP32, value=2.0)
            {yielded}
        return out
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        initialized = passes.init_mem_ref()(before)
        original = _Storage(initialized)
        request = _memref(original.branches[0].return_vars[0])
        assert not ir.MemRef.may_alias(request, _memref(original.assigns["initial"].var))
        after = passes.materialize_semantic_aliases()(initialized)
        storage = _Storage(after)
        assert len(storage.branches) == 2 and len(storage.loops) == 1
        _assert_canonical_arms(storage)
        initial = storage.assigns["initial"].var
        seeded = storage.branches[0].return_vars[0]
        inner = storage.branches[1]
        selected = inner.return_vars[0]
        carry = storage.loops[0].iter_args[0]

        def same_window(value, expected):
            assert _base(value) == _base(expected)
            assert ir.structural_equal(_memref(value).byte_offset_, _memref(expected).byte_offset_)
            assert _memref(value).size_ == _memref(expected).size_ == 128
            value_type, expected_type = value.type, expected.type
            assert isinstance(value_type, ir.TileType) and isinstance(expected_type, ir.TileType)
            assert value_type.shape == expected_type.shape == [4, 8]
            assert value_type.dtype == expected_type.dtype == ir.DataType.FP32
            assert value_type.memory_space == expected_type.memory_space == ir.MemorySpace.Vec
            assert value_type.get_effective_tile_view() == expected_type.get_effective_tile_view()

        same_window(seeded, initial)
        same_window(carry, seeded)
        assert not ir.MemRef.may_alias(request, _memref(carry)), (
            "the regression requires an actual relocation away from the frozen observer index"
        )
        writer = storage.assigns["written"]
        assert isinstance(writer.value, ir.Call) and writer.value.op.name == ir.get_op("tile.assemble").name
        assert writer.value.args[0].same_as(selected)
        assert writer.value.args[1].same_as(storage.assigns["replacement"].var)
        same_window(writer.var, selected)
        saved = storage.assigns["_selected_saved"].value
        assert isinstance(saved, ir.Call) and saved.args[0].same_as(writer.var)
        if observe_old:
            assert not ir.MemRef.may_alias(_memref(selected), _memref(carry)), (
                "a planned mutable phi must not overwrite the old carry read after it"
            )
            old_store = storage.assigns["_old_saved"]
            assert isinstance(old_store.value, ir.Call) and old_store.value.args[0].same_as(carry)
            assert storage.statements.index(writer) < storage.statements.index(old_store)
            assert storage.statements.index(old_store) < storage.statements.index(
                storage.assigns["next_value"]
            )
            for body in (inner.then_body, inner.else_body):
                assert body is not None
                value = _yield(body).value[0]
                copy = next(op for op in storage.statements if op.var.same_as(value))
                assert isinstance(copy.value, ir.Call) and copy.value.op.name == _MOVE
                assert copy.value.args[0].same_as(carry)
        else:
            same_window(selected, carry)
        if planner == passes.MemoryPlanner.PYPTO and not enable_buffer_ir:
            after = passes.memory_reuse()(after)
        _verify_storage(after)


# Native continuations retain their pipeline clone, without relaxing hazards.
@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
def test_pipeline_common_results_retain_their_separate_native_clone_windows(planner, enable_buffer_ir):
    """Each mutable result stays on its own clone; no cross-stage allocation is merged."""
    before = pl.parse_program("""
@pl.program
class NativePipelineCommon:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[32, 16], pl.FP16], flag: pl.Scalar[pl.BOOL]
               ) -> tuple[pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat],
                          pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat]]:
        first = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.Mat,
                             attrs={"pipeline_membership": "9:0"})
        second = pl.tile.load(pool, [16, 0], [16, 16], target_memory=pl.Mem.Mat,
                              attrs={"pipeline_membership": "9:1"})
        if flag:
            selected_first, selected_second = pl.yield_(first, second)
        else:
            selected_first, selected_second = pl.yield_(first, second)
        changed_first = pl.tile.gather_row(selected_first, pool, [0, 0], [0, 0], [1, 16],
                                           attrs={"pipeline_membership": "9:0"})
        changed_second = pl.tile.gather_row(selected_second, pool, [0, 0], [16, 0], [1, 16],
                                            attrs={"pipeline_membership": "9:1"})
        return changed_first, changed_second
""")
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.branches) == 1 and not storage.calls(_MOVE)
    _assert_canonical_arms(storage)
    first, second = (storage.assigns[name].var for name in ("first", "second"))
    assert not ir.MemRef.may_alias(_memref(first), _memref(second))
    for index, (original, name) in enumerate(((first, "changed_first"), (second, "changed_second"))):
        selected = storage.branches[0].return_vars[index]
        writer = storage.assigns[name]
        assert isinstance(writer.value, ir.Call) and writer.value.args[0].same_as(selected)
        for value in (selected, writer.var):
            assert ir.MemRef.same_allocation(_memref(value), _memref(original))
            assert ir.structural_equal(_memref(value).byte_offset_, _memref(original).byte_offset_)
            assert _memref(value).size_ == _memref(original).size_
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("protection", ["split_load_tpop", "written_workspace"])
def test_mutable_common_results_do_not_relax_native_hardware_storage_hazards(
    planner, enable_buffer_ir, protection, ascend_backend
):
    """A native common result cannot turn the old source into a hazardous write target."""
    if protection == "split_load_tpop":
        before = pl.parse_program("""
@pl.program
class CommonLoadTpop:
    @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
    def kernel(self, original: pl.Tensor[[8, 128], pl.FP32], flag: pl.Scalar[pl.BOOL],
               out: pl.Out[pl.Tensor[[8, 128], pl.FP32]]) -> pl.Tensor[[8, 128], pl.FP32]:
        pipe_storage: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        initial = pl.tile.load(original, [0, 0], [8, 128], target_memory=pl.Mem.Vec)
        chunk: pl.Tile[[8, 128], pl.FP32, pl.MemRef(pipe_storage, 0, 4096), pl.Mem.Vec] = (
            pl.tile.tpop_from_aic(split=1))
        if flag:
            selected = pl.yield_(initial)
        else:
            selected = pl.yield_(initial)
        changed = pl.tile.assemble(selected, chunk, [0, 0])
        return pl.tile.store(changed, [0, 0], out)
""")
        # Cross-core slots intentionally have no tile MemRef. Establish the
        # same legacy selection as the full pipeline before storage mutation;
        # an absent selection attribute instead requests low-level Buffer IR.
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
            before = passes.select_buffer_ir_functions()(before)
        function = next(iter(before.functions.values()))
        assert function.attrs["buffer_ir_selected"] is False
    else:
        before = pl.parse_program("""
@pl.program
class CommonWrittenWorkspace:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, original: pl.Tensor[[16, 16], pl.FP32], other: pl.Tensor[[16, 16], pl.FP32],
               flag: pl.Scalar[pl.BOOL], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
               ) -> pl.Tensor[[16, 16], pl.FP32]:
        initial = pl.tile.load(original, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        source = pl.tile.load(other, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        mask = pl.tile.cmps(initial, 0.0, cmp_type=4)
        if flag:
            selected = pl.yield_(initial)
        else:
            selected = pl.yield_(initial)
        changed = pl.tile.sels(mask, source, selected, -1.0)
        return pl.tile.store(changed, [0, 0], out)
""")
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.branches) == 1
    _assert_canonical_arms(storage)
    initial = storage.assigns["initial"].var
    selected = storage.branches[0].return_vars[0]
    assert not ir.MemRef.may_alias(_memref(initial), _memref(selected))
    writer = storage.assigns["changed"]
    assert isinstance(writer.value, ir.Call)
    argument = 0 if protection == "split_load_tpop" else 2
    assert writer.value.args[argument].same_as(selected)
    if protection == "split_load_tpop":
        chunk_type = storage.assigns["chunk"].var.type
        assert isinstance(chunk_type, ir.TileType) and chunk_type.memref is None
        assert not ir.MemRef.may_alias(_memref(initial), _memref(writer.var))
    else:
        # The last old read precedes the if. Only the workspace's independent
        # storage contract prevents its hidden SCALAR write racing that read.
        mask = storage.assigns["mask"]
        assert isinstance(mask.value, ir.Call) and mask.value.args[0].same_as(initial)
    copies = storage.calls(_MOVE)
    assert len(copies) == 2
    for copy in copies:
        assert isinstance(copy.value, ir.Call) and copy.value.args[0].same_as(initial)
        assert ir.MemRef.same_allocation(_memref(copy.var), _memref(selected))
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
def test_nested_canonical_workspace_cannot_move_into_an_outer_live_in_window(
    planner, enable_buffer_ir, ascend_backend
):
    """An outer result may copy scratch late, but cannot move its hidden write into old data."""
    before = pl.parse_program("""
@pl.program
class NestedWorkspaceDestination:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, original: pl.Tensor[[16, 16], pl.FP32], choices: pl.Tensor[[32, 16], pl.FP32],
               outer_flag: pl.Scalar[pl.BOOL], inner_flag: pl.Scalar[pl.BOOL],
               out: pl.Out[pl.Tensor[[32, 16], pl.FP32]]) -> pl.Tensor[[32, 16], pl.FP32]:
        initial = pl.tile.load(original, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        left = pl.tile.load(choices, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        right = pl.tile.load(choices, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
        mask = pl.tile.cmps(initial, 0.0, cmp_type=4)
        if outer_flag:
            if inner_flag:
                workspace = pl.yield_(left)
            else:
                workspace = pl.yield_(right)
            changed = pl.tile.sels(mask, left, workspace, -1.0)
            _saved = pl.tile.store(changed, [0, 0], out)
            selected = pl.yield_(workspace)
        else:
            selected = pl.yield_(initial)
        return pl.tile.store(selected, [16, 0], out)
""")
    after = _legalize(before, planner, enable_buffer_ir=enable_buffer_ir)
    storage = _Storage(after)
    assert len(storage.branches) == 2
    _assert_canonical_arms(storage)
    outer = next(branch for branch in storage.branches if branch.return_vars[0].name_hint == "selected")
    inner = next(branch for branch in storage.branches if branch.return_vars[0].name_hint == "workspace")
    initial = storage.assigns["initial"].var
    workspace = inner.return_vars[0]
    assert not ir.MemRef.may_alias(_memref(initial), _memref(workspace)), (
        "canonical forwarding must not move the workspace's hidden SCALAR write into the old input"
    )
    writer = storage.assigns["changed"]
    assert isinstance(writer.value, ir.Call) and writer.value.args[2].same_as(workspace)
    mask = storage.assigns["mask"]
    assert isinstance(mask.value, ir.Call) and mask.value.args[0].same_as(initial)
    saved = storage.assigns["_saved"]
    assert isinstance(saved.value, ir.Call) and saved.value.args[0].same_as(writer.var)
    yielded = _yield(outer.then_body).value[0]
    copy = next(statement for statement in storage.statements if statement.var.same_as(yielded))
    assert isinstance(copy.value, ir.Call) and copy.value.op.name == _MOVE
    assert copy.value.args[0].same_as(workspace)
    assert storage.statements.index(writer) < storage.statements.index(saved) < storage.statements.index(copy)
    assert _base(copy.var) == _base(outer.return_vars[0])
    _verify_storage(after)


# Required destinations follow the current SSA carry owner, not its old request.
@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
@pytest.mark.parametrize("update_else", [False, True], ids=["update_then", "update_else"])
@pytest.mark.parametrize("fixed_result", [False, True], ids=["planned_result", "fixed_result"])
def test_required_nested_if_targets_follow_their_rebound_loop_storage_owner(
    planner, enable_buffer_ir, while_loop, update_else, fixed_result
):
    # This is the initialized input of MaterializeSemanticAliases. The native
    # writeback results inherit their operand windows; only logical If results
    # have independent requests. No user binding is attached to a gather result.
    span = ir.Span.unknown()
    allocations: list[ir.Stmt] = []

    def allocation(name, size, space=ir.MemorySpace.Mat):
        base = ir.Var(name, ir.PtrType(), span)
        arguments = [
            ir.ConstInt(space.value, ir.DataType.INDEX, span),
            ir.ConstInt(size, ir.DataType.INDEX, span),
        ]
        allocations.append(
            ir.AssignStmt(base, ir.Call(ir.get_op("tile.alloc"), arguments, base.type, span), span)
        )
        return base

    gm_base = ir.Var("gm_base", ir.PtrType(), span)
    pool = ir.Var("pool", ir.TensorType([16, 16], ir.DataType.FP16, ir.MemRef(gm_base, 0, 512, span)), span)
    flags = [ir.Var(name, ir.ScalarType(ir.DataType.BOOL), span) for name in ("entry", "outer", "inner")]
    entry, outer, inner = flags
    count = ir.Var("count", ir.ScalarType(ir.DataType.INDEX), span)
    one = ir.ConstInt(1, ir.DataType.INDEX, span)
    zero = ir.ConstInt(0, ir.DataType.INDEX, span)
    slot = ir.ConstInt(1, ir.DataType.INDEX, span)
    seed_window = ir.MemRef(allocation("seed_arena", 1024), 512, 512, span, slots=2, slot=slot)
    initial_window = ir.MemRef(
        allocation("initial_request", 1024 if fixed_result else 512),
        512 if fixed_result else 0,
        512,
        span,
        is_pinned=fixed_result,
        slots=2 if fixed_result else 1,
        slot=slot if fixed_result else None,
    )

    def binding(name, call, window=None):
        tile = call.type
        assert isinstance(tile, ir.TileType)
        if window is not None:
            tile = ir.TileType(tile.shape, tile.dtype, window, tile.tile_view, tile.memory_space)
        value = ir.Var(name, tile, span)
        typed_call = ir.Call(call.op, list(call.args), dict(call.kwargs), dict(call.attrs), tile, span)
        return value, ir.AssignStmt(value, typed_call, span)

    seed, load = binding(
        "seed", tile_ops.load(pool, [0, 0], [16, 16], target_memory=ir.MemorySpace.Mat), seed_window
    )
    seed_type = seed.type
    assert isinstance(seed_type, ir.TileType)

    def result_type(window):
        return ir.TileType(seed_type.shape, seed_type.dtype, window, seed_type.tile_view, ir.MemorySpace.Mat)

    initial_type = result_type(initial_window)
    initial = ir.Var("initial", initial_type, span)
    entry_update, entry_write = binding(
        "entry_update", tile_ops.gather_row(seed, pool, [0, 0], [1, 0], [1, 16])
    )
    update_arm = ir.SeqStmts([entry_write, ir.YieldStmt([entry_update], span)], span)
    pass_arm = ir.YieldStmt([seed], span)
    then_arm, else_arm = (pass_arm, update_arm) if update_else else (update_arm, pass_arm)
    entry_if = ir.IfStmt(entry, then_arm, else_arm, [initial], span)
    current = ir.IterArg("current", initial_type, initial, span)
    left_update, left_write = binding(
        "left_update", tile_ops.gather_row(current, pool, [1, 0], [2, 0], [1, 16])
    )
    right_update, right_write = binding(
        "right_update", tile_ops.gather_row(current, pool, [2, 0], [3, 0], [1, 16])
    )
    nested = ir.Var("nested", result_type(ir.MemRef(allocation("nested_request", 512), 0, 512, span)), span)
    nested_if = ir.IfStmt(
        inner,
        ir.SeqStmts([left_write, ir.YieldStmt([left_update], span)], span),
        ir.SeqStmts([right_write, ir.YieldStmt([right_update], span)], span),
        [nested],
        span,
    )
    fallback, fallback_write = binding(
        "fallback", tile_ops.gather_row(current, pool, [3, 0], [4, 0], [1, 16])
    )
    selected = ir.Var(
        "selected", result_type(ir.MemRef(allocation("selected_request", 512), 0, 512, span)), span
    )
    outer_if = ir.IfStmt(
        outer,
        ir.SeqStmts([nested_if, ir.YieldStmt([nested], span)], span),
        ir.SeqStmts([fallback_write, ir.YieldStmt([fallback], span)], span),
        [selected],
        span,
    )
    result = ir.Var("result", initial_type, span)
    loop: ir.ForStmt | ir.WhileStmt
    if while_loop:
        index = ir.IterArg("index", ir.ScalarType(ir.DataType.INDEX), zero, span)
        final_index = ir.Var("_final_index", index.type, span)
        loop = ir.WhileStmt(
            ir.Lt(index, count, ir.DataType.BOOL, span),
            [current, index],
            ir.SeqStmts(
                [outer_if, ir.YieldStmt([selected, ir.Add(index, one, ir.DataType.INDEX, span)], span)], span
            ),
            [result, final_index],
            span,
        )
    else:
        loop = ir.ForStmt(
            ir.Var("_index", ir.ScalarType(ir.DataType.INDEX), span),
            zero,
            count,
            one,
            [current],
            ir.SeqStmts([outer_if, ir.YieldStmt([selected], span)], span),
            [result],
            span,
        )
    left_result, move = binding(
        "left_result",
        tile_ops.move(result, ir.MemorySpace.Left),
        ir.MemRef(allocation("left_result_storage", 512, ir.MemorySpace.Left), 0, 512, span),
    )
    function = ir.Function(
        "kernel",
        [gm_base, pool, *flags, count],
        [ir.TileType([16, 16], ir.DataType.FP16, memory_space=ir.MemorySpace.Left)],
        ir.SeqStmts([*allocations, load, entry_if, loop, move, ir.ReturnStmt([left_result], span)], span),
        span,
        type=ir.FunctionType.InCore,
    )
    initialized = ir.Program([function], "RequiredCarryStorageOwner", span)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        original = _Storage(initialized)
        for write in original.calls(ir.get_op("tile.gather_row").name):
            assert isinstance(write.value, ir.Call)
            assert ir.structural_equal(write.var.type, write.value.type)
            assert ir.structural_equal(write.var.type, write.value.args[0].type)
        # The dependency is captured while the carry still names the independent
        # initial-If request. Its eventual slot-one seed window is different.
        assert not ir.MemRef.same_allocation(
            _memref(original.loops[0].iter_args[0]), _memref(original.assigns["seed"].var)
        )
        assert _memref(original.branches[0].return_vars[0]).is_pinned_ == fixed_result
        if fixed_result:
            with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
                passes.materialize_semantic_aliases()(initialized)
            return
        after = passes.materialize_semantic_aliases()(initialized)

    storage = _Storage(after)
    assert len(storage.loops) == 1 and len(storage.branches) == 3
    origins = TileHandleOrigins(after)
    original_values = TileHandleOrigins(initialized).values
    loop = storage.loops[0]
    seed = storage.assigns["seed"].var
    seed_window = _memref(seed)
    assert seed_window.slot_count_ == 2 and seed_window.slot_index_ is not None
    assert isinstance(seed_window.slot_index_, ir.ConstInt) and seed_window.slot_index_.value == 1
    assert isinstance(seed_window.byte_offset_, ir.ConstInt) and seed_window.byte_offset_.value == 512
    values = [loop.iter_args[0], loop.return_vars[0], _yield(loop.body).value[0]]
    values.extend(branch.return_vars[0] for branch in storage.branches)
    values.extend(
        storage.assigns[name].var for name in ("entry_update", "left_update", "right_update", "fallback")
    )
    for value in values:
        origins.assert_window(value, seed_window)
        assert isinstance(value, (ir.Var, ir.IterArg))
        origins.assert_descriptor(value, original_values[value.name_hint])
    for branch in storage.branches:
        assert branch.else_body is not None
        for arm in [branch.then_body, branch.else_body]:
            for value in _yield(arm).value:
                origins.assert_window(value, seed_window)
    for name in ("entry_update", "left_update", "right_update", "fallback"):
        writer = storage.assigns[name].value
        assert isinstance(writer, ir.Call)
        operand = writer.args[0]
        assert isinstance(operand, (ir.Var, ir.IterArg))
        expected = seed if name == "entry_update" else loop.iter_args[0]
        assert operand.unique_id == expected.unique_id
    assert isinstance(loop.iter_args[0].initValue, (ir.Var, ir.IterArg))
    assert loop.iter_args[0].initValue.unique_id == origins.values["initial"].unique_id
    backedge = _yield(loop.body).value[0]
    assert isinstance(backedge, (ir.Var, ir.IterArg))
    assert backedge.unique_id == origins.values["selected"].unique_id
    for transfer in storage.calls(_MOVE):
        assert isinstance(transfer.var.type, ir.TileType)
        assert transfer.var.type.memory_space == ir.MemorySpace.Left
    _verify_storage(after)


# Loop isolation must preserve the old read's byte and slot window too.
def _loop_old_window_suffix(observer_kind, loop, result, observed, count, params, span, outer_current):
    def integer(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    initial_type = result.type
    assert isinstance(initial_type, ir.TileType)
    space = initial_type.memory_space
    caller = observer_kind == "caller"
    if observer_kind == "recurring":
        # The old read precedes this iteration's inner write but repeats after it.
        # Carry the updated value between outer iterations: reusing the fixed
        # initial SSA value would itself require preserving another old value.
        assert isinstance(outer_current, ir.IterArg)
        outer_result = ir.Var("outer_result", initial_type, span)
        outer = ir.ForStmt(
            ir.Var("outer_index", count.type, span),
            integer(0),
            integer(2),
            integer(1),
            [outer_current],
            ir.SeqStmts([ir.EvalStmt(observed, span), loop, ir.YieldStmt([result], span)], span),
            [outer_result],
            span,
        )
        suffix: list[ir.Stmt] = [outer, ir.ReturnStmt([outer_result], span)]
        returns: list[ir.Type] = [
            ir.TileType(initial_type.shape, initial_type.dtype, None, initial_type.tile_view, space)
        ]
    elif observer_kind in ("sibling", "nested_sibling"):
        # The initializer read is at the loop header. Combining its reverse
        # coordinate with a later sibling's primary coordinate invents a read
        # reachable after this loop; both coordinates must belong to one event.
        flag = ir.Var("branch_flag", ir.ScalarType(ir.DataType.BOOL), span)
        params.append(flag)
        update_arm = ir.SeqStmts([loop, ir.EvalStmt(result, span), ir.YieldStmt([], span)], span)

        def read_arm():
            return ir.SeqStmts([ir.EvalStmt(observed, span), ir.YieldStmt([], span)], span)

        branch = ir.IfStmt(flag, update_arm, read_arm(), [], span)
        if observer_kind == "nested_sibling":
            outer_flag = ir.Var("outer_flag", flag.type, span)
            params.append(outer_flag)
            branch = ir.IfStmt(
                outer_flag, read_arm(), ir.SeqStmts([branch, ir.YieldStmt([], span)], span), [], span
            )
        suffix = [branch, ir.ReturnStmt(span)]
        returns = []
    else:
        values = [result] if caller else [result, observed]
        suffix = [loop, ir.ReturnStmt(values, span)]
        result_type = ir.TileType(initial_type.shape, initial_type.dtype, None, initial_type.tile_view, space)
        returns = [result_type] * len(values)
    return suffix, returns


def _loop_old_window_program(memory, separation, overlap, while_loop, observer_kind):
    span = ir.Span.unknown()
    space = getattr(ir.MemorySpace, memory)
    size = 1024 if memory == "Acc" else 512
    allocations: list[ir.Stmt] = []

    def integer(value):
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    def allocation(name, extent, memory_space):
        base = ir.Var(name, ir.PtrType(), span)
        allocations.append(
            ir.AssignStmt(
                base,
                ir.Call(
                    ir.get_op("tile.alloc"), [integer(memory_space.value), integer(extent)], base.type, span
                ),
                span,
            )
        )
        return base

    def binding(name, call, window):
        tile = call.type
        assert isinstance(tile, ir.TileType)
        tile = ir.TileType(tile.shape, tile.dtype, window, tile.tile_view, tile.memory_space)
        value = ir.Var(name, tile, span)
        typed = ir.Call(call.op, list(call.args), dict(call.kwargs), dict(call.attrs), tile, span)
        return value, ir.AssignStmt(value, typed, span)

    caller = observer_kind == "caller"
    arena = ir.Var("arena", ir.PtrType(), span) if caller else allocation("arena", size * 2, space)
    slots = 2 if separation == "slots" else 1
    source_slot = integer(0) if slots == 2 else None
    old_slot = integer(0 if overlap else 1) if slots == 2 else None
    source_window = ir.MemRef(arena, 0, size, span, slots=slots, slot=source_slot)
    # These are initialized MemRefs, where slot one starts one slot-size later.
    old_offset = 0 if overlap else size
    old_window = ir.MemRef(arena, old_offset, size, span, slots=slots, slot=old_slot)
    gm_base = ir.Var("gm_base", ir.PtrType(), span)
    pool = ir.Var("pool", ir.TensorType([32, 16], ir.DataType.FP16, ir.MemRef(gm_base, 0, 1024, span)), span)
    count = ir.Var("count", ir.ScalarType(ir.DataType.INDEX), span)
    params: list[ir.Var | tuple[ir.Var, ir.ParamDirection]] = [gm_base, pool, count]
    if caller:
        params.append(arena)
    operands: list[ir.Var] = []
    if memory == "Acc":
        for name, rows, operand_space in (
            ("left", 16, ir.MemorySpace.Left),
            ("wide_left", 32, ir.MemorySpace.Left),
            ("right", 16, ir.MemorySpace.Right),
        ):
            base = ir.Var(name + "_base", ir.PtrType(), span)
            operand = ir.Var(
                name,
                ir.TileType(
                    [rows, 16], ir.DataType.FP16, ir.MemRef(base, 0, rows * 32, span), None, operand_space
                ),
                span,
            )
            params.extend([base, operand])
            operands.append(operand)
        left, wide_left, right = operands
        seed_call = tile_ops.matmul(left, right, span=span)
        bridge_call = tile_ops.matmul(wide_left, right, span=span)
    else:
        seed_call = tile_ops.load(pool, [0, 0], [16, 16], target_memory=space, span=span)
        bridge_call = tile_ops.load(pool, [0, 0], [32, 16], target_memory=space, span=span)
    initial, initialize = binding("initial", seed_call, source_window)
    initial_type = initial.type
    assert isinstance(initial_type, ir.TileType)
    prefix: list[ir.Stmt] = []
    if not caller:
        # This unused wide handle must not join the two independently read windows.
        _, bridge = binding("_unused_whole_root", bridge_call, ir.MemRef(arena, 0, size * 2, span))
        prefix.append(bridge)
    if caller:
        observed = ir.Var(
            "observed",
            ir.TileType(initial_type.shape, initial_type.dtype, old_window, initial_type.tile_view, space),
            span,
        )
        params.append((observed, ir.ParamDirection.In))
        if overlap:
            # Preserve the caller's original payload before loop entry. A load
            # into this window would already violate the readonly parameter.
            initialize = ir.AssignStmt(initial, observed, span)
        prefix.append(initialize)
    elif overlap:
        # A genuine old alias, rather than two incompatible initial writes.
        observed = ir.Var("observed", initial_type, span)
        prefix.extend([initialize, ir.AssignStmt(observed, initial, span)])
    else:
        observed, initialize_old = binding("observed", seed_call, old_window)
        prefix.extend([initialize, initialize_old])
    outer_current = (
        ir.IterArg("outer_current", initial_type, initial, span) if observer_kind == "recurring" else None
    )
    current = ir.IterArg(
        "current", initial_type, outer_current if outer_current is not None else initial, span
    )
    if memory == "Acc":
        update_call = tile_ops.matmul_acc(current, operands[0], operands[2], span=span)
    elif memory == "Mat":
        update_call = tile_ops.gather_row(current, pool, [0, 0], [1, 0], [1, 16], span=span)
    else:
        replacement_window = ir.MemRef(allocation("replacement_storage", size, space), 0, size, span)
        replacement, load = binding("replacement", seed_call, replacement_window)
        prefix.append(load)
        update_call = tile_ops.assemble(current, replacement, [0, 0], span=span)
    changed, update = binding("changed", update_call, source_window)
    result = ir.Var("loop_result", initial_type, span)
    if while_loop:
        index = ir.IterArg("index", count.type, integer(0), span)
        final_index = ir.Var("_final_index", index.type, span)
        loop = ir.WhileStmt(
            ir.Lt(index, count, ir.DataType.BOOL, span),
            [current, index],
            ir.SeqStmts(
                [update, ir.YieldStmt([changed, ir.Add(index, integer(1), ir.DataType.INDEX, span)], span)],
                span,
            ),
            [result, final_index],
            span,
        )
    else:
        loop = ir.ForStmt(
            ir.Var("index", count.type, span),
            integer(0),
            count,
            integer(1),
            [current],
            ir.SeqStmts([update, ir.YieldStmt([changed], span)], span),
            [result],
            span,
        )
    suffix, returns = _loop_old_window_suffix(
        observer_kind, loop, result, observed, count, params, span, outer_current
    )
    function = ir.Function(
        "kernel",
        params,
        returns,
        ir.SeqStmts([*allocations, *prefix, *suffix], span),
        span,
        type=ir.FunctionType.InCore,
    )
    return ir.Program([function], "LoopOldReadWindows", span)


def _check_loop_old_windows(before, planner, enable_buffer_ir, memory, overlap, observer_kind):
    original = _Storage(before)
    old_function = next(iter(before.functions.values()))
    old_observed = (
        next(value for value in old_function.params if value.name_hint == "observed")
        if observer_kind == "caller"
        else original.assigns["observed"].var
    )
    source = original.assigns["initial"].var
    exclusive = observer_kind in ("sibling", "nested_sibling")
    must_isolate = overlap and not exclusive
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        if must_isolate and memory != "Vec":
            with pytest.raises(ValueError, match=_LOOP_INPUT_CONFLICT):
                passes.materialize_semantic_aliases()(before)
            return
        after = passes.materialize_semantic_aliases()(before)
    storage = _Storage(after)
    loop = next(loop for loop in storage.loops if loop.iter_args and loop.iter_args[0].name_hint == "current")
    carry = loop.iter_args[0]
    writer = storage.assigns["changed"]
    assert isinstance(writer.value, ir.Call) and writer.value.args[0].same_as(carry)
    target = _memref(carry)
    if observer_kind == "recurring":
        outer = next(loop for loop in storage.loops if loop.iter_args[0].name_hint == "outer_current")
        assert carry.initValue.same_as(outer.iter_args[0])
        for value in (outer.iter_args[0], outer.return_vars[0], _yield(outer.body).value[0]):
            assert ir.structural_equal(_memref(value), target)
    for value in (carry.initValue, loop.return_vars[0], _yield(loop.body).value[0], writer.var):
        window = _memref(value)
        assert ir.MemRef.same_allocation(window, target)
        assert ir.structural_equal(window.byte_offset_, target.byte_offset_)
        assert window.size_ == target.size_ and window.slot_count_ == target.slot_count_
        assert (window.slot_index_ is None) == (target.slot_index_ is None)
        if target.slot_index_ is not None:
            assert window.slot_index_ is not None
            assert ir.structural_equal(window.slot_index_, target.slot_index_)
        tile = value.type
        assert isinstance(tile, ir.TileType) and isinstance(source.type, ir.TileType)
        assert tile.shape == source.type.shape and tile.dtype == source.type.dtype
        assert tile.memory_space == source.type.memory_space and tile.tile_view == source.type.tile_view
    function = next(iter(after.functions.values()))
    observed = (
        next(value for value in function.params if value.name_hint == "observed")
        if observer_kind == "caller"
        else storage.assigns["observed"].var
    )
    assert ir.structural_equal(_memref(observed), _memref(old_observed))
    copies = storage.calls(_MOVE)
    if must_isolate:
        assert not ir.MemRef.may_alias(target, _memref(observed))
        assert not ir.MemRef.may_alias(_memref(writer.var), _memref(observed))
        entry = next(statement for statement in copies if statement.var.same_as(carry.initValue))
        assert isinstance(entry.value, ir.Call) and entry.value.args[0].same_as(
            storage.assigns["initial"].var
        )
        assert storage.statements.index(entry) < storage.statements.index(writer)
    else:
        assert ir.structural_equal(target, _memref(source))
        if exclusive:
            assert ir.structural_equal(target, _memref(observed))
        else:
            assert not ir.MemRef.may_alias(target, _memref(observed))
        assert not copies, "disjoint windows or unreachable old reads must not require an entry snapshot"
    _verify_storage(after)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
@pytest.mark.parametrize(
    "memory,separation,overlap,observer_kind",
    [
        ("Mat", "bytes", False, "local"),
        ("Mat", "slots", False, "local"),
        ("Mat", "bytes", True, "local"),
        ("Acc", "bytes", False, "local"),
        ("Acc", "bytes", True, "local"),
        ("Vec", "bytes", True, "local"),
        ("Mat", "slots", False, "caller"),
        ("Mat", "slots", True, "caller"),
    ],
)
def test_loop_input_isolation_respects_old_read_and_caller_windows(
    planner, enable_buffer_ir, while_loop, memory, separation, overlap, observer_kind
):
    before = _loop_old_window_program(memory, separation, overlap, while_loop, observer_kind)
    _check_loop_old_windows(before, planner, enable_buffer_ir, memory, overlap, observer_kind)


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize(
    "overlap", [False, True], ids=["disjoint_recurring_read", "overlapping_recurring_read"]
)
def test_loop_input_isolation_respects_the_window_of_an_outer_recurring_read(
    planner, enable_buffer_ir, overlap
):
    before = _loop_old_window_program("Mat", "bytes", overlap, False, "recurring")
    _check_loop_old_windows(before, planner, enable_buffer_ir, "Mat", overlap, "recurring")


@pytest.mark.parametrize(
    "planner,enable_buffer_ir",
    [*_STORAGE_MODES, (passes.MemoryPlanner.PTOAS, False)],
    ids=[*_STORAGE_MODE_IDS, "legacy_ptoas"],
)
@pytest.mark.parametrize("while_loop", [False, True], ids=["for_loop", "while_loop"])
@pytest.mark.parametrize("observer_kind", ["sibling", "nested_sibling"])
def test_loop_input_isolation_keeps_initializer_and_sibling_read_coordinates_together(
    planner, enable_buffer_ir, while_loop, observer_kind
):
    before = _loop_old_window_program("Mat", "bytes", True, while_loop, observer_kind)
    _check_loop_old_windows(before, planner, enable_buffer_ir, "Mat", True, observer_kind)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
