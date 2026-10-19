# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dual-AIV replay assignments retain the substituted value's type."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.pypto_core import ir as _ir_core


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    yield
    backend.reset_for_testing()


def _verify_assignments(program):
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.AssignTypeSymmetry)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, program)
    assert not diagnostics, "\n".join(d.message for d in diagnostics)


def _lanes(program):
    function = program.get_function("main_aiv")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    branch = function.body.stmts[-1]
    assert isinstance(branch, ir.IfStmt)
    assert isinstance(branch.then_body, ir.SeqStmts)
    assert isinstance(branch.else_body, ir.SeqStmts)
    return branch.then_body, branch.else_body


def _assert_empty_tile(type_):
    assert isinstance(type_, ir.TileType)
    assert type_.tile_view is not None
    assert len(type_.tile_view.valid_shape) == 2
    for dim in type_.tile_view.valid_shape:
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == 0


def test_replay_tile_alias_chain_inherits_zero_valid_shape():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
        def main_aiv(self, out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
            slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
            tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            alias = tile
            alias2 = alias
            result = pl.mul(alias2, 2.0)
            pl.tfree_to_aic(tile)
            updated = pl.store(result, [0, 0], out)
            return updated

    before = passes.convert_to_ssa()(Before)
    _verify_assignments(before)
    after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    _, replay = _lanes(after)
    aliases = [s for s in replay.stmts if isinstance(s, ir.AssignStmt) and "alias" in s.var.name_hint]
    assert len(aliases) == 2
    for assign in aliases:
        _assert_empty_tile(assign.var.type)
        ir.assert_structural_equal(assign.var.type, assign.value.type)


def test_store_bypass_uses_destination_shape_symbols():
    before = pl.parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
    def main_aiv(
        self,
        outer_n: pl.Scalar[pl.INDEX],
        inner_n: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[outer_n, 16], pl.FP32]],
    ) -> pl.Tensor[[outer_n, 16], pl.FP32]:
        slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
        tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        pl.tfree_to_aic(tile)
        updated = pl.store(tile, [0, 0], out)
        return updated
""")
    function = before.get_function("main_aiv")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    stmts = list(function.body.stmts)
    assign = stmts[-2]
    assert isinstance(assign, ir.AssignStmt)
    call = assign.value
    assert isinstance(call, ir.Call)
    # Extracted kernels can retain a callee-local symbol on an explicitly typed
    # store result while the destination still carries its caller-local symbol.
    destination_type = function.params[2].type
    assert isinstance(destination_type, ir.TensorType)
    store_type = ir.TensorType([function.params[1], destination_type.shape[1]], pl.FP32)
    value = ir.Call(call.op, call.args, call.kwargs, store_type, call.span)
    result = ir.Var(assign.var.name_hint, store_type, assign.var.span)
    stmts[-2] = ir.AssignStmt(result, value, assign.span)
    stmts[-1] = ir.ReturnStmt([result], function.span)
    function = ir.Function(
        function.name,
        list(zip(function.params, function.param_directions)),
        [store_type],
        ir.SeqStmts(stmts, function.span),
        function.span,
        type=function.func_type,
        attrs=function.attrs,
    )
    before = ir.Program([function], "StoreSymbols", function.span)
    _verify_assignments(before)
    # This explicit intermediate call type is not expressible by the DSL
    # printer: parsing pl.store would infer the destination type instead.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    function = after.get_function("main_aiv")
    assert function is not None
    _, replay = _lanes(after)
    alias = replay.stmts[-2]
    assert isinstance(alias, ir.AssignStmt)
    assert alias.value is function.params[2]
    ir.assert_structural_equal(alias.var.type, function.params[2].type)


@pytest.mark.parametrize("op", ["fmods", "rems"])
def test_scalar_remainder_replay_materializes_empty_tile(op):
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def main_aiv(self, out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        pl.func_attr({{"dual_aiv_dispatch": True}})
        slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
        tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        tmp = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        remainder = pl.tile.{op}(tile, 16.0{", tmp" if op == "rems" else ""})
        pl.tfree_to_aic(tile)
        updated = pl.store(remainder, [0, 0], out)
        return updated
""")
    before = passes.convert_to_ssa()(before)
    after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    lane0, lane1 = _lanes(after)
    for body, expected_op in [(lane0, f"tile.{op}"), (lane1, "tile.create")]:
        remainder = next(
            s for s in body.stmts if isinstance(s, ir.AssignStmt) and "remainder" in s.var.name_hint
        )
        assert isinstance(remainder.value, ir.Call)
        assert remainder.value.op.name == expected_op
        if expected_op == "tile.create":
            _assert_empty_tile(remainder.var.type)


@pytest.mark.parametrize("nested", [False, True])
def test_replay_if_result_inherits_yielded_tile_type(nested):
    inner = "alias = tile"
    if nested:
        inner = (
            "if flag:\n                alias = tile"
            "\n            else:\n                alias = pl.mul(tile, 2.0)"
        )
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def main_aiv(self, flag: pl.Scalar[pl.BOOL],
                 out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        pl.func_attr({{"dual_aiv_dispatch": True}})
        slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
        tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        if flag:
            {inner}
        else:
            alias = pl.mul(tile, 3.0)
        result = pl.mul(alias, 4.0)
        pl.tfree_to_aic(tile)
        updated = pl.store(result, [0, 0], out)
        return updated
""")
    before = passes.convert_to_ssa()(before)
    after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    _, replay = _lanes(after)
    branch = next(stmt for stmt in replay.stmts if isinstance(stmt, ir.IfStmt))
    assert len(branch.return_vars) == 1
    merged = branch.return_vars[0]
    _assert_empty_tile(merged.type)
    for body in [branch.then_body, branch.else_body]:
        assert isinstance(body, ir.SeqStmts)
        yielded = body.stmts[-1]
        assert isinstance(yielded, ir.YieldStmt)
        ir.assert_structural_equal(merged.type, yielded.value[0].type)
    result = next(
        stmt
        for stmt in replay.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("result")
    )
    assert isinstance(result.value, ir.Call)
    assert result.value.args[0] is merged


def test_replay_if_preserves_join_metadata_with_different_branch_memrefs():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
        def main_aiv(
            self, flag: pl.Scalar[pl.BOOL], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
            tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            if flag:
                alias = pl.mul(tile, 2.0)
            else:
                alias = pl.tile.full([16, 16], dtype=pl.FP32, value=0.0)
            result = pl.mul(alias, 4.0)
            pl.tfree_to_aic(tile)
            updated = pl.store(result, [0, 0], out)
            return updated

    before = passes.convert_to_ssa()(Before)
    function = before.get_function("main_aiv")
    assert function is not None and isinstance(function.body, ir.SeqStmts)
    original = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.IfStmt))
    # Branch-local storage is not part of the join type. Model an explicit
    # producer allocation below the DSL parser; type verification stays enabled.
    assert isinstance(original.then_body, ir.SeqStmts)
    producer = original.then_body.stmts[0]
    assert isinstance(producer, ir.AssignStmt) and isinstance(producer.value, ir.Call)
    allocated_type = ir.TileType(
        [16, 16], pl.FP32, memref=ir.MemRef("branch_storage", producer.span), memory_space=ir.MemorySpace.Vec
    )
    allocated = ir.Var(producer.var.name_hint, allocated_type, producer.span)
    call = producer.value
    then_body = ir.SeqStmts(
        [
            ir.AssignStmt(
                allocated, ir.Call(call.op, call.args, call.kwargs, allocated_type, call.span), producer.span
            ),
            ir.YieldStmt([allocated], original.span),
        ],
        original.span,
    )
    branch = ir.IfStmt(original.condition, then_body, original.else_body, original.return_vars, original.span)
    body = ir.SeqStmts([branch if stmt is original else stmt for stmt in function.body.stmts], function.span)
    function = ir.Function(
        function.name,
        list(zip(function.params, function.param_directions)),
        function.return_types,
        body,
        function.span,
        type=function.func_type,
        attrs=function.attrs,
    )
    before = ir.Program([function], "BranchStorage", function.span)
    _verify_assignments(before)
    after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    _, replay = _lanes(after)
    branch = next(stmt for stmt in replay.stmts if isinstance(stmt, ir.IfStmt))
    merged = branch.return_vars[0]
    _assert_empty_tile(merged.type)
    assert isinstance(merged.type, ir.TileType)
    assert isinstance(original.return_vars[0].type, ir.TileType)
    assert merged.type.memref is original.return_vars[0].type.memref
    assert isinstance(branch.then_body, ir.SeqStmts)
    assert isinstance(branch.else_body, ir.SeqStmts)
    then_yield = branch.then_body.stmts[-1]
    else_yield = branch.else_body.stmts[-1]
    assert isinstance(then_yield, ir.YieldStmt) and isinstance(else_yield, ir.YieldStmt)
    then_type = then_yield.value[0].type
    else_type = else_yield.value[0].type
    assert isinstance(then_type, ir.TileType) and isinstance(else_type, ir.TileType)
    assert then_type.memref is not else_type.memref


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("same_destination", [False, True])
@pytest.mark.parametrize("scope_kind", ["plain", "runtime", "split", "nested"])
def test_replay_tensor_join_preserves_type_after_store_bypass(symbolic, same_destination, scope_kind):
    before = pl.parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
    def main_aiv(self, flag: pl.Scalar[pl.BOOL], rows: pl.Scalar[pl.INDEX],
                 a: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                 b: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
        tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        if flag:
            updated = pl.store(tile, [0, 0], a)
        else:
            updated = pl.store(tile, [0, 0], b)
        pl.tfree_to_aic(tile)
        return updated
""")
    before = passes.convert_to_ssa()(before)
    function = before.get_function("main_aiv")
    assert function is not None and isinstance(function.body, ir.SeqStmts)
    original = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.IfStmt))
    assert isinstance(original.else_body, ir.SeqStmts)
    store = original.else_body.stmts[0]
    assert isinstance(store, ir.AssignStmt) and isinstance(store.value, ir.Call)
    # Explicit store results retain the common join type, while the destination
    # can carry a different caller-local symbol or physical extent.
    extent = function.params[1] if symbolic else ir.ConstInt(32, pl.INDEX, function.span)
    destination = ir.Var(
        "b",
        ir.TensorType([extent, ir.ConstInt(16, pl.INDEX, function.span)], pl.FP32),
        function.params[3].span,
    )
    call = store.value
    args = list(call.args)
    args[2] = destination
    else_body = ir.SeqStmts(
        [
            ir.AssignStmt(store.var, ir.Call(call.op, args, call.kwargs, call.type, call.span), store.span),
            *original.else_body.stmts[1:],
        ],
        original.span,
    )
    branch = ir.IfStmt(original.condition, original.then_body, else_body, original.return_vars, original.span)
    if same_destination:
        assert isinstance(original.then_body, ir.SeqStmts)
        then_store = original.then_body.stmts[0]
        assert isinstance(then_store, ir.AssignStmt) and isinstance(then_store.value, ir.Call)
        then_call = then_store.value
        then_args = list(then_call.args)
        then_args[2] = destination
        then_body = ir.SeqStmts(
            [
                ir.AssignStmt(
                    then_store.var,
                    ir.Call(then_call.op, then_args, then_call.kwargs, then_call.type, then_call.span),
                    then_store.span,
                ),
                *original.then_body.stmts[1:],
            ],
            original.span,
        )
        branch = ir.IfStmt(original.condition, then_body, else_body, original.return_vars, original.span)
    if scope_kind != "plain":
        marker_call = _ir_core.create_op_call("tile.get_block_idx", [], {}, original.span)
        marker_result = ir.Var("marker_join", marker_call.type, original.span)

        def wrap_scope(replay_body):
            assert isinstance(replay_body, ir.SeqStmts)
            yielded = replay_body.stmts[-1]
            assert isinstance(yielded, ir.YieldStmt)
            marker = ir.Var("branch_marker", marker_call.type, original.span)
            tail: ir.Stmt = ir.SeqStmts(
                [
                    ir.AssignStmt(marker, marker_call, original.span),
                    ir.YieldStmt([*yielded.value, marker], yielded.span),
                ],
                original.span,
            )
            if scope_kind in ("split", "nested"):
                tail = ir.SplitAivScopeStmt(
                    pl.SplitMode.NONE, name_hint="normalization_scope", body=tail, span=original.span
                )
            if scope_kind in ("runtime", "nested"):
                tail = ir.RuntimeScopeStmt(
                    manual=True, name_hint="normalization_scope", body=tail, span=original.span
                )
            return ir.SeqStmts([*replay_body.stmts[:-1], tail], original.span)

        branch = ir.IfStmt(
            original.condition,
            wrap_scope(branch.then_body),
            wrap_scope(branch.else_body),
            [*original.return_vars, marker_result],
            original.span,
        )
    body = ir.SeqStmts([branch if stmt is original else stmt for stmt in function.body.stmts], function.span)
    params = list(zip(function.params, function.param_directions))
    params[3] = (destination, function.param_directions[3])
    function = ir.Function(
        function.name,
        params,
        function.return_types,
        body,
        function.span,
        type=function.func_type,
        attrs=function.attrs,
    )
    before = ir.Program([function], "TensorJoin", function.span)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    # These explicitly typed intermediate stores cannot round-trip through the
    # DSL: parsing pl.store would infer the destination type instead.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = passes.split_vector_kernel()(before)
    for program in [before, after]:
        _verify_assignments(program)
        diagnostics = passes.PropertyVerifierRegistry.verify(properties, program)
        assert not diagnostics, "\n".join(d.message for d in diagnostics)
    _, replay = _lanes(program)
    branch = next(stmt for stmt in replay.stmts if isinstance(stmt, ir.IfStmt))

    def terminal_body(replay_body):
        assert isinstance(replay_body, ir.SeqStmts)
        if scope_kind == "plain":
            return replay_body
        scope = replay_body.stmts[-1]
        if scope_kind in ("runtime", "nested"):
            assert isinstance(scope, ir.RuntimeScopeStmt)
            assert scope.manual and scope.name_hint == "normalization_scope"
            scope = scope.body
        if scope_kind in ("split", "nested"):
            assert isinstance(scope, ir.SplitAivScopeStmt)
            assert scope.split == pl.SplitMode.NONE and scope.count == 2
            assert scope.name_hint == "normalization_scope"
            scope = scope.body
        assert isinstance(scope, ir.SeqStmts)
        marker = scope.stmts[0]
        yielded = scope.stmts[-1]
        assert isinstance(marker, ir.AssignStmt) and isinstance(marker.value, ir.Call)
        assert marker.value.op.name == _ir_core.get_op("tile.get_block_idx").name
        assert isinstance(yielded, ir.YieldStmt) and yielded.value[1] is marker.var
        return scope

    then_body = terminal_body(branch.then_body)
    else_body = terminal_body(branch.else_body)
    if same_destination:
        ir.assert_structural_equal(branch.return_vars[0].type, destination.type)
        for replay_body in [then_body, else_body]:
            assert isinstance(replay_body, ir.SeqStmts)
            assert not any(
                isinstance(stmt, ir.AssignStmt)
                and isinstance(stmt.value, ir.Call)
                and stmt.value.op.name == _ir_core.get_op("tensor.view").name
                for stmt in replay_body.stmts
            )
        return
    ir.assert_structural_equal(branch.return_vars[0].type, original.return_vars[0].type)
    view = else_body.stmts[-2]
    assert isinstance(view, ir.AssignStmt) and isinstance(view.value, ir.Call)
    assert view.value.op.name == _ir_core.get_op("tensor.view").name
    replay_function = program.get_function("main_aiv")
    assert replay_function is not None
    assert view.value.args[0] is replay_function.params[3]
    yielded = else_body.stmts[-1]
    assert isinstance(yielded, ir.YieldStmt)
    assert yielded.value[0] is view.var


@pytest.mark.parametrize("kind", ["tile", "tensor"])
def test_replay_join_type_uses_lane_local_scalar_definitions(kind):
    then_value = "pl.mul(tile, 2.0)" if kind == "tile" else "pl.store(tile, [0, 0], a)"
    else_value = "pl.mul(tile, 3.0)" if kind == "tile" else "pl.store(tile, [0, 0], b)"
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def main_aiv(self, flag: pl.Scalar[pl.BOOL], rows: pl.Scalar[pl.INDEX],
                 a: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                 b: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
        pl.func_attr({{"dual_aiv_dispatch": True}})
        slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
        local_extent = rows + 1
        tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        if flag:
            merged = {then_value}
        else:
            merged = {else_value}
        alias = merged
        pl.tfree_to_aic(tile)
""")
    before = passes.convert_to_ssa()(before)
    function = before.get_function("main_aiv")
    assert function is not None and isinstance(function.body, ir.SeqStmts)
    stmts = list(function.body.stmts)
    extent = next(
        stmt.var
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("local_extent")
    )
    original = next(stmt for stmt in stmts if isinstance(stmt, ir.IfStmt))
    params = list(zip(function.params, function.param_directions))
    if kind == "tile":
        join_type = ir.TileType(
            [16, 16], pl.FP32, tile_view=ir.TileView(stride=[extent, 1]), memory_space=ir.MemorySpace.Vec
        )

        def rebuild_tile(body):
            assert isinstance(body, ir.SeqStmts)
            producer = body.stmts[0]
            assert isinstance(producer, ir.AssignStmt) and isinstance(producer.value, ir.Call)
            value = ir.Var(producer.var.name_hint, join_type, producer.span)
            call = producer.value
            return ir.SeqStmts(
                [
                    ir.AssignStmt(
                        value, ir.Call(call.op, call.args, call.kwargs, join_type, call.span), producer.span
                    ),
                    ir.YieldStmt([value], original.span),
                ],
                original.span,
            )

        then_body, else_body = rebuild_tile(original.then_body), rebuild_tile(original.else_body)
    else:
        join_type = ir.TensorType([extent, ir.ConstInt(16, pl.INDEX, function.span)], pl.FP32)
        destination = ir.Var("b", ir.TensorType([32, 16], pl.FP32), function.span)
        params[3] = (destination, function.param_directions[3])

        def rebuild_store(body, destination):
            assert isinstance(body, ir.SeqStmts)
            store = body.stmts[0]
            assert isinstance(store, ir.AssignStmt) and isinstance(store.value, ir.Call)
            value = ir.Var(store.var.name_hint, join_type, store.span)
            call = store.value
            args = [*call.args[:2], destination]
            return ir.SeqStmts(
                [
                    ir.AssignStmt(
                        value, ir.Call(call.op, args, call.kwargs, join_type, call.span), store.span
                    ),
                    ir.YieldStmt([value], original.span),
                ],
                original.span,
            )

        then_body = rebuild_store(original.then_body, function.params[2])
        else_body = rebuild_store(original.else_body, destination)
    merged = ir.Var(original.return_vars[0].name_hint, join_type, original.span)
    branch = ir.IfStmt(original.condition, then_body, else_body, [merged], original.span)
    branch_index = stmts.index(original)
    tail = [stmt for stmt in stmts[branch_index + 1 :] if isinstance(stmt, ir.EvalStmt)]
    body = ir.SeqStmts(
        [*stmts[:branch_index], branch, *tail, ir.ReturnStmt([], function.span)], function.span
    )
    function = ir.Function(
        function.name, params, [], body, function.span, type=function.func_type, attrs=function.attrs
    )
    before = ir.Program([function], "DynamicJoin", function.span)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, before)
    assert not diagnostics, "\n".join(d.message for d in diagnostics)
    # Explicit join metadata is legal intermediate IR, but is not preserved by
    # DSL inference. Keep property verification active without text roundtrip.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.UseAfterDef)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, after)
    assert not diagnostics, "\n".join(d.message for d in diagnostics)
    lane0, lane1 = _lanes(after)

    def local_extent(body):
        return next(
            stmt.var
            for stmt in body.stmts
            if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("local_extent")
        )

    replay_join = next(stmt for stmt in lane1.stmts if isinstance(stmt, ir.IfStmt))
    result_type = replay_join.return_vars[0].type
    if isinstance(result_type, ir.TileType):
        assert result_type.tile_view is not None
        actual = result_type.tile_view.stride[0]
    else:
        assert isinstance(result_type, ir.TensorType)
        actual = result_type.shape[0]
    assert actual.same_as(local_extent(lane1))
    assert not actual.same_as(local_extent(lane0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
