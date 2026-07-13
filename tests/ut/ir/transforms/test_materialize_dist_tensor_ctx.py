# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Tests for ``MaterializeDistTensorCtx``."""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType
from pypto.pypto_core import ir, passes


@pytest.fixture(autouse=True)
def _basic_verification_context():
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _collect_assign_stmts(stmt: ir.Stmt) -> list[ir.AssignStmt]:
    found: list[ir.AssignStmt] = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.AssignStmt):
            found.append(s)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk(child)
        if isinstance(s, ir.ForStmt):
            walk(s.body)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)

    walk(stmt)
    return found


def _collect_calls(stmt: ir.Stmt, op_name: str) -> list[ir.Call]:
    calls: list[ir.Call] = []

    def visit_expr(expr: ir.Expr) -> None:
        if isinstance(expr, ir.Call):
            if expr.op.name == op_name:
                calls.append(expr)
            for arg in expr.args:
                visit_expr(arg)

    for assign in _collect_assign_stmts(stmt):
        visit_expr(assign.value)

    def walk_eval(s: ir.Stmt) -> None:
        if isinstance(s, ir.EvalStmt):
            visit_expr(s.expr)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk_eval(child)
        if isinstance(s, ir.ForStmt):
            walk_eval(s.body)
        if isinstance(s, ir.ScopeStmt):
            walk_eval(s.body)

    walk_eval(stmt)
    return calls


def _span() -> ir.Span:
    return ir.Span("test_materialize_dist_tensor_ctx.py", 1, 1)


def _dist_ty() -> ir.DistributedTensorType:
    return ir.DistributedTensorType([4], pl.FP32)


def _apply(program: ir.Program) -> ir.Program:
    program = passes.materialize_comm_domain_scopes()(program)
    program = passes.lower_host_tensor_collectives()(program)
    program = passes.derive_call_directions()(program)
    return passes.materialize_dist_tensor_ctx()(program)


def test_host_dispatch_materializes_comm_ctx_args():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            data: pld.DistributedTensor[[256], pl.FP32],
            signal: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            return 0

    result = _apply(P)
    chip = _get_func(result, "chip_orch")
    host = _get_func(result, "host_orch")

    assert [type(param.type) for param in chip.params][-2:] == [ir.CommCtxType, ir.CommCtxType]
    assert list(chip.param_directions)[-2:] == [ir.ParamDirection.In, ir.ParamDirection.In]

    calls = _collect_calls(host.body, "chip_orch")
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 4
    assert isinstance(call.args[2].type, ir.CommCtxType)
    assert isinstance(call.args[3].type, ir.CommCtxType)
    assert list(call.arg_directions)[-2:] == [ir.ArgDirection.Scalar, ir.ArgDirection.Scalar]

    get_ctx_calls = _collect_calls(host.body, "pld.system.get_comm_ctx")
    assert len(get_ctx_calls) == 2
    assert get_ctx_calls[0].args[0] is call.args[0]
    assert get_ctx_calls[1].args[0] is call.args[1]
    ctx_assigns = [
        assign for assign in _collect_assign_stmts(host.body) if isinstance(assign.var.type, ir.CommCtxType)
    ]
    assert [assign.var.name_hint for assign in ctx_assigns] == ["data_ctx", "signal_ctx"]


def _call_with_dirs(op_name: str, args: list[ir.Expr], span: ir.Span) -> ir.Call:
    return ir.Call(
        ir.GlobalVar(op_name),
        args,
        {},
        {"arg_directions": [ir.ArgDirection.Input for _ in args]},
        ir.TupleType([]),
        span,
    )


def _manual_wrapper_program(wrapper_type: ir.FunctionType) -> ir.Program:
    span = _span()
    data_ty = _dist_ty()

    inner_data = ir.Var("data", data_ty, span)
    inner = ir.Function(
        "inner",
        [(inner_data, ir.ParamDirection.In)],
        [],
        ir.ReturnStmt(span),
        span,
        ir.FunctionType.InCore,
    )

    wrapper_data = ir.Var("data", data_ty, span)
    wrapper_call = _call_with_dirs("inner", [wrapper_data], span)
    wrapper = ir.Function(
        "wrapper",
        [(wrapper_data, ir.ParamDirection.In)],
        [],
        ir.EvalStmt(wrapper_call, span),
        span,
        wrapper_type,
    )

    main_data = ir.Var("data", data_ty, span)
    main_call = _call_with_dirs("wrapper", [main_data], span)
    main = ir.Function(
        "main",
        [(main_data, ir.ParamDirection.In)],
        [],
        ir.EvalStmt(main_call, span),
        span,
        ir.FunctionType.Orchestration,
    )
    return ir.Program([inner, wrapper, main], f"manual_{wrapper_type.name.lower()}_wrapper", span)


def _expected_manual_wrapper_program(wrapper_type: ir.FunctionType) -> ir.Program:
    span = _span()
    data_ty = _dist_ty()
    ctx_ty = ir.CommCtxType.get()

    inner_data = ir.Var("data", data_ty, span)
    inner_ctx = ir.Var("data_ctx", ctx_ty, span)
    inner = ir.Function(
        "inner",
        [(inner_data, ir.ParamDirection.In), (inner_ctx, ir.ParamDirection.In)],
        [],
        ir.ReturnStmt(span),
        span,
        ir.FunctionType.InCore,
    )

    wrapper_data = ir.Var("data", data_ty, span)
    wrapper_ctx = ir.Var("data_ctx", ctx_ty, span)
    wrapper_call = ir.Call(
        ir.GlobalVar("inner"),
        [wrapper_data, wrapper_ctx],
        {},
        {"arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.Scalar]},
        ir.TupleType([]),
        span,
    )
    wrapper = ir.Function(
        "wrapper",
        [(wrapper_data, ir.ParamDirection.In), (wrapper_ctx, ir.ParamDirection.In)],
        [],
        ir.EvalStmt(wrapper_call, span),
        span,
        wrapper_type,
    )

    main_data = ir.Var("data", data_ty, span)
    main_ctx = ir.Var("data_ctx", ctx_ty, span)
    main_call = ir.Call(
        ir.GlobalVar("wrapper"),
        [main_data, main_ctx],
        {},
        {"arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.Scalar]},
        ir.TupleType([]),
        span,
    )
    main = ir.Function(
        "main",
        [(main_data, ir.ParamDirection.In), (main_ctx, ir.ParamDirection.In)],
        [],
        ir.EvalStmt(main_call, span),
        span,
        ir.FunctionType.Orchestration,
    )
    return ir.Program([inner, wrapper, main], f"manual_{wrapper_type.name.lower()}_wrapper", span)


@pytest.mark.parametrize("wrapper_type", [ir.FunctionType.Spmd, ir.FunctionType.Group])
def test_wrapper_calls_forward_materialized_comm_ctx_params(wrapper_type: ir.FunctionType):
    result = passes.materialize_dist_tensor_ctx()(_manual_wrapper_program(wrapper_type))
    ir.assert_structural_equal(result, _expected_manual_wrapper_program(wrapper_type))

    inner = _get_func(result, "inner")
    wrapper = _get_func(result, "wrapper")
    main = _get_func(result, "main")

    assert isinstance(inner.params[-1].type, ir.CommCtxType)
    assert isinstance(wrapper.params[-1].type, ir.CommCtxType)
    assert isinstance(main.params[-1].type, ir.CommCtxType)

    wrapper_call = _collect_calls(wrapper.body, "inner")[0]
    main_call = _collect_calls(main.body, "wrapper")[0]

    assert wrapper_call.args[-1] is wrapper.params[-1]
    assert main_call.args[-1] is main.params[-1]
    assert list(wrapper_call.arg_directions)[-1] == ir.ArgDirection.Scalar
    assert list(main_call.arg_directions)[-1] == ir.ArgDirection.Scalar


def test_materialized_comm_ctx_param_name_avoids_existing_param():
    span = _span()
    data_ty = _dist_ty()
    scalar_ty = ir.ScalarType(DataType.INDEX)

    kernel_data = ir.Var("data", data_ty, span)
    kernel_data_ctx = ir.Var("data_ctx", scalar_ty, span)
    kernel = ir.Function(
        "kernel",
        [(kernel_data, ir.ParamDirection.In), (kernel_data_ctx, ir.ParamDirection.In)],
        [],
        ir.ReturnStmt(span),
        span,
        ir.FunctionType.InCore,
    )

    main_data = ir.Var("data", data_ty, span)
    main_data_ctx = ir.Var("data_ctx", scalar_ty, span)
    main_call = _call_with_dirs("kernel", [main_data, main_data_ctx], span)
    main = ir.Function(
        "main",
        [(main_data, ir.ParamDirection.In), (main_data_ctx, ir.ParamDirection.In)],
        [],
        ir.EvalStmt(main_call, span),
        span,
        ir.FunctionType.Orchestration,
    )

    result = passes.materialize_dist_tensor_ctx()(ir.Program([kernel, main], "ctx_name_collision", span))
    kernel_after = _get_func(result, "kernel")
    main_after = _get_func(result, "main")
    call_after = _collect_calls(main_after.body, "kernel")[0]

    assert [param.name_hint for param in kernel_after.params] == ["data", "data_ctx", "data_ctx_1"]
    assert [param.name_hint for param in main_after.params] == ["data", "data_ctx", "data_ctx_1"]
    assert isinstance(kernel_after.params[-1].type, ir.CommCtxType)
    assert isinstance(main_after.params[-1].type, ir.CommCtxType)
    assert call_after.args[-1] is main_after.params[-1]
    assert list(call_after.arg_directions) == [
        ir.ArgDirection.Input,
        ir.ArgDirection.Input,
        ir.ArgDirection.Scalar,
    ]


def test_materialized_local_comm_ctx_name_avoids_existing_local():
    span = _span()
    data_ty = _dist_ty()
    scalar_ty = ir.ScalarType(DataType.INDEX)

    data = ir.Var("data", data_ty, span)
    source_data = ir.Var("source_data", data_ty, span)
    producer = ir.Function(
        "producer",
        [(source_data, ir.ParamDirection.In)],
        [data_ty],
        ir.ReturnStmt([source_data], span),
        span,
        ir.FunctionType.InCore,
    )
    callee = ir.Function(
        "callee",
        [(data, ir.ParamDirection.In)],
        [],
        ir.ReturnStmt(span),
        span,
        ir.FunctionType.InCore,
    )

    main_source_data = ir.Var("source_data", data_ty, span)
    local_data = ir.Var("data", data_ty, span)
    existing_local = ir.Var("data_ctx", scalar_ty, span)
    producer_call = _call_with_dirs("producer", [main_source_data], span)
    call = _call_with_dirs("callee", [local_data], span)
    main = ir.Function(
        "main",
        [(main_source_data, ir.ParamDirection.In)],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(existing_local, ir.ConstInt(0, DataType.INDEX, span), span),
                ir.AssignStmt(local_data, producer_call, span),
                ir.EvalStmt(call, span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )

    result = passes.materialize_dist_tensor_ctx()(
        ir.Program([producer, callee, main], "local_ctx_name_collision", span)
    )
    main_after = _get_func(result, "main")
    call_after = _collect_calls(main_after.body, "callee")[0]
    ctx_assigns = [
        assign
        for assign in _collect_assign_stmts(main_after.body)
        if isinstance(assign.var.type, ir.CommCtxType)
    ]

    assert [assign.var.name_hint for assign in ctx_assigns] == ["data_ctx_1"]
    assert call_after.args[-1] is ctx_assigns[0].var


def test_materialized_comm_ctx_param_name_avoids_existing_local():
    span = _span()
    data_ty = _dist_ty()
    scalar_ty = ir.ScalarType(DataType.INDEX)

    data = ir.Var("data", data_ty, span)
    existing_local = ir.Var("data_ctx", scalar_ty, span)
    kernel = ir.Function(
        "kernel",
        [(data, ir.ParamDirection.In)],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(existing_local, ir.ConstInt(0, DataType.INDEX, span), span),
                ir.ReturnStmt(span),
            ],
            span,
        ),
        span,
        ir.FunctionType.InCore,
    )

    result = passes.materialize_dist_tensor_ctx()(ir.Program([kernel], "ctx_param_local_collision", span))
    kernel_after = _get_func(result, "kernel")

    assert [param.name_hint for param in kernel_after.params] == ["data", "data_ctx_1"]
    assert isinstance(kernel_after.params[-1].type, ir.CommCtxType)


def test_param_alias_forwards_materialized_comm_ctx_param():
    span = _span()
    data_ty = _dist_ty()

    kernel_data = ir.Var("data", data_ty, span)
    kernel = ir.Function(
        "kernel",
        [(kernel_data, ir.ParamDirection.In)],
        [],
        ir.ReturnStmt(span),
        span,
        ir.FunctionType.InCore,
    )

    main_data = ir.Var("data", data_ty, span)
    alias = ir.Var("alias", data_ty, span)
    call = _call_with_dirs("kernel", [alias], span)
    main = ir.Function(
        "main",
        [(main_data, ir.ParamDirection.In)],
        [],
        ir.SeqStmts([ir.AssignStmt(alias, main_data, span), ir.EvalStmt(call, span)], span),
        span,
        ir.FunctionType.Orchestration,
    )

    result = passes.materialize_dist_tensor_ctx()(ir.Program([kernel, main], "ctx_param_alias", span))
    main_after = _get_func(result, "main")
    call_after = _collect_calls(main_after.body, "kernel")[0]
    ctx_assigns = [
        assign
        for assign in _collect_assign_stmts(main_after.body)
        if isinstance(assign.var.type, ir.CommCtxType)
    ]

    assert [param.name_hint for param in main_after.params] == ["data", "data_ctx"]
    assert ctx_assigns == []
    assert call_after.args[-1] is main_after.params[-1]


def test_unsupported_expression_context_rejects_synthesized_prefix():
    span = _span()
    data_ty = _dist_ty()
    bool_ty = ir.ScalarType(DataType.BOOL)

    producer_data = ir.Var("data", data_ty, span)
    producer = ir.Function(
        "producer",
        [(producer_data, ir.ParamDirection.In)],
        [data_ty],
        ir.ReturnStmt([producer_data], span),
        span,
        ir.FunctionType.InCore,
    )

    predicate_data = ir.Var("data", data_ty, span)
    predicate = ir.Function(
        "predicate",
        [(predicate_data, ir.ParamDirection.In)],
        [bool_ty],
        ir.ReturnStmt([ir.ConstBool(True, span)], span),
        span,
        ir.FunctionType.InCore,
    )

    source_data = ir.Var("source_data", data_ty, span)
    local_data = ir.Var("local_data", data_ty, span)
    producer_call = _call_with_dirs("producer", [source_data], span)
    predicate_call = ir.Call(
        ir.GlobalVar("predicate"),
        [local_data],
        {},
        {"arg_directions": [ir.ArgDirection.Input]},
        bool_ty,
        span,
    )
    main = ir.Function(
        "main",
        [(source_data, ir.ParamDirection.In)],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(local_data, producer_call, span),
                ir.IfStmt(predicate_call, ir.ReturnStmt(span), None, [], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )

    program = ir.Program([producer, predicate, main], "unsupported_prefix_context", span)
    with pytest.raises(RuntimeError, match="cannot synthesize get_comm_ctx prefix"):
        passes.materialize_dist_tensor_ctx()(program)


def test_submit_prefix_runtime_out_keeps_ctx_after_passed_args():
    span = _span()
    data_ty = _dist_ty()
    scratch_ty = ir.TensorType([4], pl.FP32)
    submit_ty = ir.TupleType([scratch_ty, ir.ScalarType(DataType.TASK_ID)])

    data = ir.Var("data", data_ty, span)
    scratch = ir.Var("scratch", scratch_ty, span)
    stage = ir.Function(
        "stage",
        [(data, ir.ParamDirection.In), (scratch, ir.ParamDirection.Out)],
        [scratch_ty],
        ir.ReturnStmt([scratch], span),
        span,
        ir.FunctionType.InCore,
    )

    main_data = ir.Var("data", data_ty, span)
    submit_result = ir.Var("submit_result", submit_ty, span)
    submit = ir.Submit(
        ir.GlobalVar("stage"),
        [main_data],
        [],
        {},
        {"arg_directions": [ir.ArgDirection.Input]},
        submit_ty,
        span,
    )
    main = ir.Function(
        "main",
        [(main_data, ir.ParamDirection.In)],
        [submit_ty],
        ir.SeqStmts([ir.AssignStmt(submit_result, submit, span), ir.ReturnStmt([submit_result], span)], span),
        span,
        ir.FunctionType.Orchestration,
    )

    result = passes.materialize_dist_tensor_ctx()(ir.Program([stage, main], "submit_prefix_ctx", span))
    stage_after = _get_func(result, "stage")
    main_after = _get_func(result, "main")
    assigns = _collect_assign_stmts(main_after.body)
    assert len(assigns) == 1
    submit_after = assigns[0].value

    assert isinstance(submit_after, ir.Submit)
    assert len(stage_after.params) == 3
    assert stage_after.param_directions[1] == ir.ParamDirection.Out
    assert isinstance(stage_after.params[2].type, ir.CommCtxType)
    assert list(submit_after.args) == [main_after.params[0], main_after.params[1]]
    assert list(submit_after.arg_directions) == [ir.ArgDirection.Input, ir.ArgDirection.Scalar]


def test_return_call_materializes_local_ctx_prefix():
    span = _span()
    data_ty = _dist_ty()
    ret_ty = ir.ScalarType(DataType.INDEX)

    data = ir.Var("data", data_ty, span)
    source_data = ir.Var("source_data", data_ty, span)
    producer = ir.Function(
        "producer",
        [(source_data, ir.ParamDirection.In)],
        [data_ty],
        ir.ReturnStmt([source_data], span),
        span,
        ir.FunctionType.InCore,
    )
    callee = ir.Function(
        "callee",
        [(data, ir.ParamDirection.In)],
        [ret_ty],
        ir.ReturnStmt([ir.ConstInt(0, DataType.INDEX, span)], span),
        span,
        ir.FunctionType.InCore,
    )
    main_source_data = ir.Var("source_data", data_ty, span)
    local_data = ir.Var("data", data_ty, span)
    producer_call = _call_with_dirs("producer", [main_source_data], span)
    call = ir.Call(
        ir.GlobalVar("callee"),
        [local_data],
        {},
        {"arg_directions": [ir.ArgDirection.Input]},
        ret_ty,
        span,
    )
    main = ir.Function(
        "main",
        [(main_source_data, ir.ParamDirection.In)],
        [ret_ty],
        ir.SeqStmts([ir.AssignStmt(local_data, producer_call, span), ir.ReturnStmt([call], span)], span),
        span,
        ir.FunctionType.Orchestration,
    )

    exp_source_data = ir.Var("source_data", data_ty, span)
    exp_source_ctx_param = ir.Var("source_data_ctx", ir.CommCtxType.get(), span)
    exp_producer = ir.Function(
        "producer",
        [(exp_source_data, ir.ParamDirection.In), (exp_source_ctx_param, ir.ParamDirection.In)],
        [data_ty],
        ir.ReturnStmt([exp_source_data], span),
        span,
        ir.FunctionType.InCore,
    )
    exp_data = ir.Var("data", data_ty, span)
    exp_ctx_param = ir.Var("data_ctx", ir.CommCtxType.get(), span)
    exp_callee = ir.Function(
        "callee",
        [(exp_data, ir.ParamDirection.In), (exp_ctx_param, ir.ParamDirection.In)],
        [ret_ty],
        ir.ReturnStmt([ir.ConstInt(0, DataType.INDEX, span)], span),
        span,
        ir.FunctionType.InCore,
    )
    exp_main_source_data = ir.Var("source_data", data_ty, span)
    exp_main_source_ctx_param = ir.Var("source_data_ctx", ir.CommCtxType.get(), span)
    exp_local_data = ir.Var("data", data_ty, span)
    exp_local_ctx = ir.Var("data_ctx", ir.CommCtxType.get(), span)
    exp_producer_call_ty = ir.TupleType([])
    exp_producer_call = ir.Call(
        ir.GlobalVar("producer"),
        [exp_main_source_data, exp_main_source_ctx_param],
        {},
        {"arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.Scalar]},
        exp_producer_call_ty,
        span,
    )
    exp_get_ctx = ir.create_op_call("pld.system.get_comm_ctx", [exp_local_data], {}, span)
    exp_call = ir.Call(
        ir.GlobalVar("callee"),
        [exp_local_data, exp_local_ctx],
        {},
        {"arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.Scalar]},
        ret_ty,
        span,
    )
    exp_main = ir.Function(
        "main",
        [(exp_main_source_data, ir.ParamDirection.In), (exp_main_source_ctx_param, ir.ParamDirection.In)],
        [ret_ty],
        ir.SeqStmts(
            [
                ir.AssignStmt(exp_local_data, exp_producer_call, span),
                ir.AssignStmt(exp_local_ctx, exp_get_ctx, span),
                ir.ReturnStmt([exp_call], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )

    result = passes.materialize_dist_tensor_ctx()(
        ir.Program([producer, callee, main], "return_call_ctx", span)
    )
    expected = ir.Program([exp_producer, exp_callee, exp_main], "return_call_ctx", span)
    ir.assert_structural_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
