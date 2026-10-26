# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native writes preserve old SSA versions behind selected storage references."""

import pytest
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import tile as tile_ops


def _tile_type(value: ir.Var) -> ir.TileType:
    descriptor = value.type
    assert isinstance(descriptor, ir.TileType)
    return descriptor


class _Effects:
    """Already initialized storage isolates reference effects from parser policy.

    Vec references are legal normalized IR even though automatic normalization
    retains the established Vec destination-planning optimization. The same
    version rules apply to copyless Mat references.
    """

    def __init__(self, *, matrix=False):
        self.span = ir.Span.unknown()
        self.space = ir.MemorySpace.Mat if matrix else ir.MemorySpace.Vec
        self.dtype = DataType.FP16 if matrix else DataType.FP32
        self.shape = [16, 16] if matrix else [8, 8]
        self.size = 512 if matrix else 256
        self.gm = ir.Var("gm", ir.TensorType([2 * self.shape[0], self.shape[1]], self.dtype), self.span)
        self.flag = ir.Var("flag", ir.ScalarType(DataType.BOOL), self.span)
        self.parameters: list[ir.Var | tuple[ir.Var, ir.ParamDirection]] = [self.gm, self.flag]
        self.allocations: list[ir.Stmt] = []
        self.statements: list[ir.Stmt] = []
        self.a, define_a = self.owner("a", 0)
        self.b, define_b = self.owner("b", self.shape[0])
        self.statements.extend([define_a, define_b])
        self.reference_type = ir.TileType(
            self.shape, self.dtype, None, _tile_type(self.a).tile_view, self.space
        )

    def integer(self, value):
        return ir.ConstInt(value, DataType.INDEX, self.span)

    def bind(self, name, call, memory=None):
        source = call.type
        assert isinstance(source, ir.TileType)
        value_type = ir.TileType(source.shape, source.dtype, memory, source.tile_view, source.memory_space)
        value = ir.Var(name, value_type, self.span)
        lowered = ir.Call(
            call.op, list(call.args), dict(call.kwargs), dict(call.attrs), value_type, self.span
        )
        return value, ir.AssignStmt(value, lowered, self.span)

    def owner(self, name, row, memory=None):
        if memory is None:
            base = ir.Var(f"{name}_base", ir.PtrType(), self.span)
            allocation = ir.Call(
                ir.get_op("tile.alloc"),
                [self.integer(self.space.value), self.integer(self.size)],
                base.type,
                self.span,
            )
            self.allocations.append(ir.AssignStmt(base, allocation, self.span))
            memory = ir.MemRef(base, 0, self.size, self.span)
        call = tile_ops.load(self.gm, [row, 0], self.shape, target_memory=self.space)
        return self.bind(name, call, memory)

    def select(self, name, left=None, right=None):
        value = ir.Var(name, self.reference_type, self.span)
        statement = ir.IfStmt(
            self.flag,
            ir.YieldStmt([self.a if left is None else left], self.span),
            ir.YieldStmt([self.b if right is None else right], self.span),
            [value],
            self.span,
        )
        return value, statement

    def write(self, name, target):
        call = tile_ops.write(target, [0, 0], ir.ConstFloat(7.0, self.dtype, self.span))
        return self.bind(name, call, _tile_type(target).memref)

    def read(self, value):
        return ir.EvalStmt(tile_ops.read(value, [0, 0]), self.span)

    def program(self, suffix=(), returns=()):
        body = ir.SeqStmts(
            [*self.allocations, *self.statements, *suffix, ir.ReturnStmt(list(returns), self.span)], self.span
        )
        function = ir.Function(
            "kernel",
            self.parameters,
            [value.type for value in returns],
            body,
            self.span,
            type=ir.FunctionType.InCore,
        )
        return ir.Program([function], "ReferenceEffects", self.span)


def _materialize(program, planner):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
            return passes.materialize_semantic_aliases()(program)
    finally:
        backend.reset_for_testing()


_PLANNERS = list(passes.MemoryPlanner.__members__.values())


@pytest.mark.parametrize("planner", _PLANNERS)
def test_fresh_selected_sources_do_not_alias_each_other(planner):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    # Initializing b happens after a's definition, but b cannot overwrite a.
    # A may-alias component must not turn their selection into physical aliasing.
    _materialize(fixture.program([choose, fixture.read(selected)]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("late_alias", [False, True], ids=["old-source", "late-old-alias"])
def test_selected_write_rejects_later_old_version_observation(planner, late_alias):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    _, write = fixture.write("updated", selected)
    suffix: list[ir.Stmt] = [choose, write]
    observed = fixture.a
    if late_alias:
        # A later binding still observes a's old logical value, rather than a
        # fresh version merely because the alias is defined after the writer.
        observed = ir.Var("late_alias", fixture.a.type, fixture.span)
        suffix.append(ir.AssignStmt(observed, fixture.a, fixture.span))
    suffix.append(fixture.read(observed))
    with pytest.raises(ValueError, match="older value version"):
        _materialize(fixture.program(suffix), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_opaque_operation_keeps_conservative_selected_write_effects(planner):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    # This intentionally unregistered callable has no functional schema from
    # which to prove that its Tile operand is readonly.
    call = ir.Call(ir.Op("opaque_external"), [selected], ir.ScalarType(DataType.INDEX), fixture.span)
    with pytest.raises(ValueError, match="older value version"):
        _materialize(
            fixture.program([choose, ir.EvalStmt(call, fixture.span), fixture.read(fixture.a)]), planner
        )


@pytest.mark.parametrize("planner", _PLANNERS)
def test_exclusive_selected_write_preserves_the_fresh_result(planner):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    updated, write = fixture.write("updated", selected)
    # The only later observation is the writer's new version; treating every
    # reference as permanently readonly would incorrectly reject this program.
    _materialize(fixture.program([choose, write, fixture.read(updated)]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("write_in_else", [False, True])
def test_selected_update_and_unchanged_arms_preserve_correlated_versions(planner, write_in_else):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    updated, write = fixture.write("updated", selected)
    joined = ir.Var("joined", fixture.reference_type, fixture.span)
    changed = ir.SeqStmts([write, ir.YieldStmt([updated], fixture.span)], fixture.span)
    unchanged = ir.YieldStmt([selected], fixture.span)
    branch = ir.IfStmt(
        fixture.flag,
        unchanged if write_in_else else changed,
        changed if write_in_else else unchanged,
        [joined],
        fixture.span,
    )
    _materialize(fixture.program([choose, branch, fixture.read(joined)]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_demanded_yield_cannot_forward_a_prewrite_version(planner):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    _, write = fixture.write("updated", selected)
    joined = ir.Var("joined", fixture.reference_type, fixture.span)
    branch = ir.IfStmt(
        fixture.flag,
        ir.SeqStmts([write, ir.YieldStmt([selected], fixture.span)], fixture.span),
        ir.YieldStmt([selected], fixture.span),
        [joined],
        fixture.span,
    )
    with pytest.raises(ValueError, match="older value version"):
        _materialize(fixture.program([choose, branch, fixture.read(joined)]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_concrete_candidate_write_rejects_later_selected_old_value(planner):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    _, overwrite = fixture.owner("replacement", fixture.shape[0], _tile_type(fixture.a).memref)
    with pytest.raises(ValueError, match="older value version"):
        _materialize(fixture.program([choose, overwrite, fixture.read(selected)]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_concrete_candidates_in_one_selector_remain_physically_independent(planner):
    fixture = _Effects()
    _, choose = fixture.select("selected")
    updated, write = fixture.write("updated", fixture.a)
    # The shared selector component is only an alias overapproximation. A
    # concrete write to a cannot invalidate b's independently allocated bytes.
    _materialize(fixture.program([choose, write, fixture.read(fixture.b), fixture.read(updated)]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_selected_operand_retains_native_no_alias_contract(planner):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    # TTRANS is not in-place safe even if no old value is read afterwards.
    # Its concrete destination is explicitly bound to a possible selected root.
    transpose = tile_ops.transpose(selected, 0, 1)
    _, write = fixture.bind("transposed", transpose, _tile_type(fixture.a).memref)
    with pytest.raises(ValueError, match="disjoint from a possible selected storage source"):
        _materialize(fixture.program([choose, write]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_repeated_source_definition_preserves_prior_carried_version(planner):
    fixture = _Effects()
    # This definition is executed each iteration in the same concrete storage.
    # After the first backedge, carried may refer to the previous local value.
    local, define_local = fixture.owner("local", fixture.shape[0])
    carried = ir.IterArg("carried", fixture.reference_type, fixture.a, fixture.span)
    result = ir.Var("result", fixture.reference_type, fixture.span)
    body = ir.SeqStmts(
        [define_local, fixture.read(carried), ir.YieldStmt([local], fixture.span)], fixture.span
    )
    loop = ir.ForStmt(
        ir.Var("i", ir.ScalarType(DataType.INDEX), fixture.span),
        fixture.integer(0),
        fixture.integer(2),
        fixture.integer(1),
        [carried],
        body,
        [result],
        fixture.span,
    )
    with pytest.raises(ValueError, match="older value version"):
        _materialize(fixture.program([loop]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_unused_forwarded_handle_is_not_a_recurring_payload_observation(planner):
    fixture = _Effects()
    carried = ir.IterArg("carried", fixture.reference_type, fixture.b, fixture.span)
    unused = ir.Var("unused", fixture.reference_type, fixture.span)
    inner = ir.ForStmt(
        ir.Var("inner", ir.ScalarType(DataType.INDEX), fixture.span),
        fixture.integer(0),
        fixture.integer(1),
        fixture.integer(1),
        [carried],
        ir.YieldStmt([fixture.a], fixture.span),
        [unused],
        fixture.span,
    )
    updated, write = fixture.owner("updated", 0, _tile_type(fixture.a).memref)
    outer = ir.ForStmt(
        ir.Var("outer", ir.ScalarType(DataType.INDEX), fixture.span),
        fixture.integer(0),
        fixture.integer(2),
        fixture.integer(1),
        [],
        ir.SeqStmts([inner, write, fixture.read(updated), ir.YieldStmt([], fixture.span)], fixture.span),
        [],
        fixture.span,
    )
    # The inner loop only forwards a descriptor to a dead result. Re-entering
    # its outer loop cannot turn that forwarding into a read of a's old bytes.
    _materialize(fixture.program([outer]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_readonly_mat_swap_uses_reference_carries(planner):
    fixture = _Effects(matrix=True)
    carried = ir.IterArg("carried", fixture.reference_type, fixture.a, fixture.span)
    spare = ir.IterArg("spare", fixture.reference_type, fixture.b, fixture.span)
    first = ir.Var("first", fixture.reference_type, fixture.span)
    second = ir.Var("second", fixture.reference_type, fixture.span)
    loop = ir.ForStmt(
        ir.Var("i", ir.ScalarType(DataType.INDEX), fixture.span),
        fixture.integer(0),
        fixture.integer(3),
        fixture.integer(1),
        [carried, spare],
        ir.YieldStmt([spare, carried], fixture.span),
        [first, second],
        fixture.span,
    )
    after = _materialize(fixture.program([loop], [first, second]), planner)
    function = next(iter(after.functions.values()))
    loops = [
        statement for statement in ir.flatten_to_stmts(function.body) if isinstance(statement, ir.ForStmt)
    ]
    assert len(loops) == 1
    assert all(_tile_type(argument).memref is None for argument in loops[0].iter_args)
    assert all(_tile_type(result).memref is None for result in loops[0].return_vars)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_reference_validity_follows_a_rebuilt_scalar_iterarg(planner):
    fixture = _Effects()
    selected, choose = fixture.select("selected")
    index_type = ir.ScalarType(DataType.INDEX)
    initial = ir.Cast(tile_ops.read(selected, [0, 0]), DataType.INDEX, fixture.span)
    rows = ir.IterArg("rows", index_type, initial, fixture.span)
    result = ir.Var("result", index_type, fixture.span)
    view_type = ir.TileType(
        fixture.shape,
        fixture.dtype,
        None,
        ir.TileView(valid_shape=[rows, fixture.integer(fixture.shape[1])]),
        fixture.space,
    )
    view = ir.Var("view", view_type, fixture.span)
    reshape = ir.Call(
        ir.get_op("tile.reshape"),
        [selected, ir.MakeTuple([fixture.integer(extent) for extent in fixture.shape], fixture.span)],
        view_type,
        fixture.span,
    )
    loop = ir.ForStmt(
        ir.Var("i", index_type, fixture.span),
        fixture.integer(0),
        fixture.integer(2),
        fixture.integer(1),
        [rows],
        ir.SeqStmts(
            [
                ir.AssignStmt(view, reshape, fixture.span),
                fixture.read(view),
                ir.YieldStmt([rows], fixture.span),
            ],
            fixture.span,
        ),
        [result],
        fixture.span,
    )
    after = _materialize(fixture.program([choose, loop]), planner)
    function = next(iter(after.functions.values()))
    updated_loop = next(stmt for stmt in ir.flatten_to_stmts(function.body) if isinstance(stmt, ir.ForStmt))
    updated_view = next(
        stmt
        for stmt in ir.flatten_to_stmts(updated_loop.body)
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint == "view"
    )
    descriptor = _tile_type(updated_view.var)
    assert descriptor.tile_view is not None
    valid_rows = descriptor.tile_view.valid_shape[0]
    assert isinstance(valid_rows, ir.IterArg)
    assert updated_loop.iter_args[0].unique_id != rows.unique_id
    assert valid_rows.unique_id == updated_loop.iter_args[0].unique_id


@pytest.mark.parametrize("planner", _PLANNERS)
def test_selected_write_respects_readonly_caller_storage(planner):
    fixture = _Effects()
    fixture.allocations.clear()
    fixture.statements.clear()
    fixture.parameters = [fixture.flag, fixture.a, fixture.b]
    selected, choose = fixture.select("selected")
    updated, write = fixture.write("updated", selected)
    with pytest.raises(ValueError, match="readonly caller buffer"):
        _materialize(fixture.program([choose, write, fixture.read(updated)]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_input_pointer_pins_ownership_without_readonly_pointee_contract(planner):
    fixture = _Effects()
    pointer = ir.Var("caller_pointer", ir.PtrType(), fixture.span)
    window = ir.MemRef(pointer, 0, fixture.size, fixture.span)
    fixture.a, define_a = fixture.owner("caller_a", 0, window)
    fixture.statements[0] = define_a
    fixture.parameters.append(pointer)
    selected, choose = fixture.select("selected")
    updated, write = fixture.write("updated", selected)
    # An In Ptr parameter passes an address. Readonly payload ownership is a
    # typed Tile/Tensor direction contract, not a property of the raw pointer.
    _materialize(fixture.program([choose, write, fixture.read(updated)]), planner)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_concrete_out_window_is_disjoint_from_readonly_caller_window(planner):
    fixture = _Effects()
    base = ir.Var("shared_caller_base", ir.PtrType(), fixture.span)
    readonly = ir.MemRef(base, 0, fixture.size, fixture.span)
    writable = ir.MemRef(base, fixture.size, fixture.size, fixture.span)
    descriptor = _tile_type(fixture.a)
    fixture.a = ir.Var(
        "readonly",
        ir.TileType(descriptor.shape, descriptor.dtype, readonly, descriptor.tile_view, fixture.space),
        fixture.span,
    )
    fixture.b = ir.Var(
        "writable",
        ir.TileType(descriptor.shape, descriptor.dtype, writable, descriptor.tile_view, fixture.space),
        fixture.span,
    )
    fixture.allocations.clear()
    fixture.statements.clear()
    fixture.parameters = [fixture.flag, fixture.a, (fixture.b, ir.ParamDirection.InOut)]
    _, choose = fixture.select("selected")
    updated, write = fixture.bind(
        "updated",
        tile_ops.write(fixture.b, [0, 0], ir.ConstFloat(7.0, fixture.dtype, fixture.span)),
        writable,
    )
    # Both windows occur in one selector's protection component, while only
    # the first window is readonly. A concrete write checks actual byte overlap.
    _materialize(fixture.program([choose, write, fixture.read(fixture.a), fixture.read(updated)]), planner)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
