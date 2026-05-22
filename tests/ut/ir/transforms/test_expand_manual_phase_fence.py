# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from pypto import DataType, ir
from pypto.pypto_core import passes

S = ir.Span.unknown()
TASK_ID = ir.ScalarType(DataType.TASK_ID)


def _const(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, S)


def _consumer(name: str, dep_edges: list[ir.Var]) -> ir.AssignStmt:
    call = ir.Call(
        ir.GlobalVar("kernel"),
        [],
        {},
        [
            ("arg_directions", []),
            ("manual_dep_edges", dep_edges),
        ],
        TASK_ID,
        S,
    )
    return ir.AssignStmt(ir.Var(name, TASK_ID, S), call, S)


def _get_array_slot(name: str, array: ir.Var) -> ir.AssignStmt:
    call = ir.create_op_call("array.get_element", [array, _const(0)], {}, S)
    return ir.AssignStmt(ir.Var(name, TASK_ID, S), call, S)


def _program_with_loop(loop_body: ir.Stmt, *, kind: ir.ForKind = ir.ForKind.Parallel) -> ir.Program:
    loop = ir.ForStmt(
        ir.Var("p", ir.ScalarType(DataType.INDEX), S),
        _const(0),
        _const(4),
        _const(1),
        [],
        loop_body,
        [],
        S,
        kind=kind,
    )
    scope = ir.RuntimeScopeStmt(True, "manual", loop, S)
    orch = ir.Function("main", [], [], scope, S, type=ir.FunctionType.Orchestration)
    kernel = ir.Function("kernel", [], [TASK_ID], ir.ReturnStmt([ir.Var("ret_tid", TASK_ID, S)], S), S)
    return ir.Program([orch, kernel], "test_expand_manual_phase_fence", S)


def _manual_scope_body(program: ir.Program) -> ir.Stmt:
    main = program.get_function("main")
    assert main is not None
    scope = cast(ir.RuntimeScopeStmt, main.body)
    return scope.body


def _main_loop(program: ir.Program) -> ir.ForStmt:
    body = _manual_scope_body(program)
    if isinstance(body, ir.SeqStmts):
        return cast(ir.ForStmt, body.stmts[-1])
    return cast(ir.ForStmt, body)


def _manual_scope_stmts(program: ir.Program) -> list[ir.Stmt]:
    body = _manual_scope_body(program)
    if isinstance(body, ir.SeqStmts):
        return list(body.stmts)
    return [body]


def _loop_body_stmts(program: ir.Program) -> list[ir.Stmt]:
    body = _main_loop(program).body
    if isinstance(body, ir.SeqStmts):
        return list(body.stmts)
    return [body]


def _run(program: ir.Program) -> ir.Program:
    with passes.PassContext([], passes.VerificationLevel.NONE):
        return passes.expand_manual_phase_fence()(program)


def test_profitable_parallel_array_dep_inserts_dummy_and_rewrites_consumers():
    tids = ir.Var("tids", ir.ArrayType(DataType.TASK_ID, 4), S)
    before = _program_with_loop(
        ir.SeqStmts(
            [
                _consumer("a", [tids]),
                _consumer("b", [tids]),
            ],
            S,
        )
    )

    after = _run(before)
    scope_stmts = _manual_scope_stmts(after)

    barrier = cast(ir.AssignStmt, scope_stmts[0])
    barrier_call = cast(ir.Call, barrier.value)
    assert barrier_call.op.name == "system.task_dummy"
    assert barrier_call.attrs["dummy_task"] is True
    assert barrier_call.attrs["manual_dep_edges"] == [tids]

    for stmt in _loop_body_stmts(after):
        call = cast(ir.Call, cast(ir.AssignStmt, stmt).value)
        assert call.attrs["manual_dep_edges"] == [barrier.var]


def test_parallel_iter_arg_barrier_uses_visible_init_value():
    tids = ir.Var("tids", ir.ArrayType(DataType.TASK_ID, 4), S)
    iter_tids = ir.IterArg("tids_iter", tids.type, tids, S)
    return_tids = ir.Var("tids_rv", tids.type, S)
    loop = ir.ForStmt(
        ir.Var("p", ir.ScalarType(DataType.INDEX), S),
        _const(0),
        _const(4),
        _const(1),
        [iter_tids],
        ir.SeqStmts(
            [
                _consumer("a", [iter_tids]),
                _consumer("b", [iter_tids]),
                ir.YieldStmt([iter_tids], S),
            ],
            S,
        ),
        [return_tids],
        S,
        kind=ir.ForKind.Parallel,
    )
    scope = ir.RuntimeScopeStmt(True, "manual", loop, S)
    orch = ir.Function("main", [], [], scope, S, type=ir.FunctionType.Orchestration)
    kernel = ir.Function("kernel", [], [TASK_ID], ir.ReturnStmt([ir.Var("ret_tid", TASK_ID, S)], S), S)

    after = _run(ir.Program([orch, kernel], "iter_arg_source", S))
    scope_stmts = _manual_scope_stmts(after)

    barrier = cast(ir.AssignStmt, scope_stmts[0])
    barrier_call = cast(ir.Call, barrier.value)
    assert barrier_call.op.name == "system.task_dummy"
    assert barrier_call.attrs["manual_dep_edges"] == [tids]
    assert all(
        cast(ir.Call, cast(ir.AssignStmt, stmt).value).attrs["manual_dep_edges"] == [barrier.var]
        for stmt in _loop_body_stmts(after)[:2]
    )


def test_non_orchestration_function_is_ignored():
    tids = ir.Var("tids", ir.ArrayType(DataType.TASK_ID, 4), S)
    loop = _main_loop(_program_with_loop(ir.SeqStmts([_consumer("a", [tids]), _consumer("b", [tids])], S)))
    scope = ir.RuntimeScopeStmt(True, "manual", loop, S)
    worker = ir.Function("worker", [], [], scope, S, type=ir.FunctionType.AIV)
    kernel = ir.Function("kernel", [], [TASK_ID], ir.ReturnStmt([ir.Var("ret_tid", TASK_ID, S)], S), S)

    after = _run(ir.Program([worker, kernel], "ignore_worker", S))
    worker_after = after.get_function("worker")
    assert worker_after is not None
    scope_after = cast(ir.RuntimeScopeStmt, worker_after.body)
    assert isinstance(scope_after.body, ir.ForStmt)


def test_scalar_dep_does_not_insert_dummy():
    tid = ir.Var("tid", TASK_ID, S)
    before = _program_with_loop(
        ir.SeqStmts(
            [
                _consumer("a", [tid]),
                _consumer("b", [tid]),
            ],
            S,
        )
    )

    after = _run(before)
    assert all(
        cast(ir.Call, cast(ir.AssignStmt, stmt).value).op.name != "system.task_dummy"
        for stmt in _loop_body_stmts(after)
    )


def test_mixed_array_scalar_deps_do_not_insert_dummy():
    tids = ir.Var("tids", ir.ArrayType(DataType.TASK_ID, 4), S)
    tid = ir.Var("tid", TASK_ID, S)
    before = _program_with_loop(
        ir.SeqStmts(
            [
                _consumer("a", [tids, tid]),
                _consumer("b", [tids, tid]),
            ],
            S,
        )
    )

    after = _run(before)
    assert all(
        cast(ir.Call, cast(ir.AssignStmt, stmt).value).op.name != "system.task_dummy"
        for stmt in _loop_body_stmts(after)
    )


def test_partial_slot_dep_does_not_insert_dummy():
    tids = ir.Var("tids", ir.ArrayType(DataType.TASK_ID, 4), S)
    slot = _get_array_slot("slot", tids)
    before = _program_with_loop(
        ir.SeqStmts(
            [
                slot,
                _consumer("a", [slot.var]),
                _consumer("b", [slot.var]),
            ],
            S,
        )
    )

    after = _run(before)
    assert all(
        cast(ir.Call, cast(ir.AssignStmt, stmt).value).op.name != "system.task_dummy"
        for stmt in _loop_body_stmts(after)
    )


def test_two_by_two_low_benefit_does_not_insert_dummy():
    tids = ir.Var("tids", ir.ArrayType(DataType.TASK_ID, 2), S)
    loop = ir.ForStmt(
        ir.Var("p", ir.ScalarType(DataType.INDEX), S),
        _const(0),
        _const(2),
        _const(1),
        [],
        _consumer("a", [tids]),
        [],
        S,
        kind=ir.ForKind.Parallel,
    )
    scope = ir.RuntimeScopeStmt(True, "manual", loop, S)
    orch = ir.Function("main", [], [], scope, S, type=ir.FunctionType.Orchestration)
    kernel = ir.Function("kernel", [], [TASK_ID], ir.ReturnStmt([ir.Var("ret_tid", TASK_ID, S)], S), S)

    after = _run(ir.Program([orch, kernel], "low_benefit", S))
    first_call = cast(ir.Call, cast(ir.AssignStmt, _loop_body_stmts(after)[0]).value)
    assert first_call.op.name != "system.task_dummy"


def test_three_by_three_min_profitable_inserts_dummy():
    tids = ir.Var("tids", ir.ArrayType(DataType.TASK_ID, 3), S)
    loop = ir.ForStmt(
        ir.Var("p", ir.ScalarType(DataType.INDEX), S),
        _const(0),
        _const(3),
        _const(1),
        [],
        ir.SeqStmts(
            [
                _consumer("a", [tids]),
                _consumer("b", [tids]),
                _consumer("c", [tids]),
            ],
            S,
        ),
        [],
        S,
        kind=ir.ForKind.Parallel,
    )
    scope = ir.RuntimeScopeStmt(True, "manual", loop, S)
    orch = ir.Function("main", [], [], scope, S, type=ir.FunctionType.Orchestration)
    kernel = ir.Function("kernel", [], [TASK_ID], ir.ReturnStmt([ir.Var("ret_tid", TASK_ID, S)], S), S)

    after = _run(ir.Program([orch, kernel], "min_profitable", S))
    barrier = cast(ir.AssignStmt, _manual_scope_stmts(after)[0])
    barrier_call = cast(ir.Call, barrier.value)
    assert barrier_call.op.name == "system.task_dummy"
    assert barrier_call.attrs["manual_dep_edges"] == [tids]
    assert all(
        cast(ir.Call, cast(ir.AssignStmt, stmt).value).attrs["manual_dep_edges"] == [barrier.var]
        for stmt in _loop_body_stmts(after)
    )


def test_pure_range_consumer_does_not_insert_dummy():
    tids = ir.Var("tids", ir.ArrayType(DataType.TASK_ID, 4), S)
    before = _program_with_loop(_consumer("a", [tids]), kind=ir.ForKind.Sequential)

    after = _run(before)
    first_call = cast(ir.Call, cast(ir.AssignStmt, _loop_body_stmts(after)[0]).value)
    assert first_call.op.name != "system.task_dummy"
