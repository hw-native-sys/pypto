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


def _alloc_window_buffer(span: ir.Span) -> ir.Expr:
    """A window-buffer allocation, the origin of every DistributedTensor."""
    return ir.create_op_call(
        "pld.tensor.alloc_window_buffer",
        [ir.ConstInt(16, DataType.INDEX, span)],
        {"name": "buf"},
        span,
    )


def _window(buffer: ir.Var, span: ir.Span) -> ir.Expr:
    """A fresh DistributedTensor over @p buffer — it inherits no context."""
    return ir.create_op_call(
        "pld.tensor.window",
        [buffer, ir.MakeTuple([ir.ConstInt(4, DataType.INDEX, span)], span)],
        {"dtype": pl.FP32},
        span,
    )


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
    callee = ir.Function(
        "callee",
        [(data, ir.ParamDirection.In)],
        [],
        ir.ReturnStmt(span),
        span,
        ir.FunctionType.InCore,
    )

    # A window allocated in the host body: a genuine new DistributedTensor with
    # no context to inherit, so host orchestration synthesizes the query — and
    # must not reuse the `data_ctx` name already taken by a local.
    buffer = ir.Var("buffer", ir.PtrType.get(), span)
    local_data = ir.Var("data", data_ty, span)
    existing_local = ir.Var("data_ctx", scalar_ty, span)
    call = _call_with_dirs("callee", [local_data], span)
    main = ir.Function(
        "main",
        [],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(existing_local, ir.ConstInt(0, DataType.INDEX, span), span),
                ir.AssignStmt(buffer, _alloc_window_buffer(span), span),
                ir.AssignStmt(local_data, _window(buffer, span), span),
                ir.EvalStmt(call, span),
            ],
            span,
        ),
        span,
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )

    result = passes.materialize_dist_tensor_ctx()(
        ir.Program([callee, main], "local_ctx_name_collision", span)
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


def test_returned_mixed_values_use_reordered_distributed_param_contexts():
    """Return positions, rather than a tail heuristic, select each CommCtx."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reorder(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            marker: pl.Scalar[pl.INT32],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
        ) -> tuple[
            pl.Scalar[pl.INT32],
            pld.DistributedTensor[[4], pl.FP32],
            pld.DistributedTensor[[4], pl.FP32],
        ]:
            return marker, second, first

        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32]):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            marker: pl.Scalar[pl.INT32],
        ):
            result = self.reorder(first, marker, second)
            returned_second = result[1]
            returned_first = result[2]
            self.consume(returned_second)
            self.consume(returned_first)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def reorder(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            marker: pl.Scalar[pl.INT32],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            first_ctx: pld.CommCtx,
            second_ctx: pld.CommCtx,
        ) -> tuple[
            pl.Scalar[pl.INT32],
            pld.DistributedTensor[[4], pl.FP32],
            pld.DistributedTensor[[4], pl.FP32],
        ]:
            return marker, second, first

        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def main(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            marker: pl.Scalar[pl.INT32],
            first_ctx: pld.CommCtx,
            second_ctx: pld.CommCtx,
        ):
            result = self.reorder(
                first,
                marker,
                second,
                first_ctx,
                second_ctx,
                attrs={
                    "arg_directions": [
                        pl.adir.inout,
                        pl.adir.scalar,
                        pl.adir.inout,
                        pl.adir.scalar,
                        pl.adir.scalar,
                    ]
                },
            )
            returned_second = result[1]
            returned_first = result[2]
            # `reorder` returns (marker, second, first), so position 1 carries
            # `second`'s context and position 2 carries `first`'s — a tail
            # heuristic would swap them.
            self.consume(
                returned_second, second_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]}
            )
            self.consume(returned_first, first_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    ir.assert_structural_equal(_apply(passes.convert_to_ssa()(Before)), Expected)


def test_loop_carried_returned_distributed_tensors_reuse_contexts():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def comm(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2], pl.INT32]],
        ) -> tuple[pld.DistributedTensor[[4], pl.FP32], pld.DistributedTensor[[2], pl.INT32]]:
            return data, signal

        @pl.function(type=pl.FunctionType.InCore)
        def compute(self, value: pl.InOut[pl.Tensor[[4], pl.FP32]]) -> pl.Tensor[[4], pl.FP32]:
            return value

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2], pl.INT32]],
            value: pl.InOut[pl.Tensor[[4], pl.FP32]],
        ) -> pl.Tensor[[4], pl.FP32]:
            for _layer in pl.range(2):
                data, signal = self.comm(data, signal)
                value = self.compute(value)
            data, signal = self.comm(data, signal)
            return value

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def comm(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2], pl.INT32]],
            data_ctx: pld.CommCtx,
            signal_ctx: pld.CommCtx,
        ) -> tuple[pld.DistributedTensor[[4], pl.FP32], pld.DistributedTensor[[2], pl.INT32]]:
            return data, signal

        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def compute(self, value: pl.InOut[pl.Tensor[[4], pl.FP32]]) -> pl.Tensor[[4], pl.FP32]:
            return value

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def main(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2], pl.INT32]],
            value: pl.InOut[pl.Tensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
            signal_ctx: pld.CommCtx,
        ) -> pl.Tensor[[4], pl.FP32]:
            # Both call sites forward the caller's ctx params: the loop carry is
            # seeded from them and the yield keeps naming the same windows.
            for _layer, (data_i, signal_i, value_i) in pl.range(2, init_values=(data, signal, value)):
                carried = self.comm(
                    data_i,
                    signal_i,
                    data_ctx,
                    signal_ctx,
                    attrs={
                        "arg_directions": [
                            pl.adir.inout,
                            pl.adir.inout,
                            pl.adir.scalar,
                            pl.adir.scalar,
                        ]
                    },
                )
                data_next = carried[0]
                signal_next = carried[1]
                value_next = self.compute(value_i, attrs={"arg_directions": [pl.adir.inout]})
                data_out, signal_out, value_out = pl.yield_(data_next, signal_next, value_next)
            tail = self.comm(
                data_out,
                signal_out,
                data_ctx,
                signal_ctx,
                attrs={
                    "arg_directions": [
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.scalar,
                        pl.adir.scalar,
                    ]
                },
            )
            _tail_data = tail[0]
            _tail_signal = tail[1]
            return value_out

    ir.assert_structural_equal(_apply(passes.convert_to_ssa()(Before)), Expected)


def test_buffer_aliasing_view_inherits_source_context():
    """A zero-copy view keeps the source window, so it forwards the source ctx."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, data: pld.DistributedTensor[[2, 2], pl.FP32]):
            pld.system.wait(data, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]]):
            view = pl.tensor.view(data, [2, 2])
            self.consume(view)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def consume(self, data: pld.DistributedTensor[[2, 2], pl.FP32], data_ctx: pld.CommCtx):
            pld.system.wait(data, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def main(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
        ):
            view = pl.tensor.view(data, [2, 2])
            # `view` is a fresh Var, but it names `data`'s window, so the call
            # forwards `data_ctx` rather than querying a context for it.
            self.consume(view, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    ir.assert_structural_equal(_apply(Before), Expected)


def test_loop_rebinding_carry_to_another_distributed_tensor_is_rejected():
    """A carry seeded from its init must still yield back the same context."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def pick(self, data: pld.DistributedTensor[[4], pl.FP32]) -> pld.DistributedTensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32]):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            first: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            second: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
        ):
            data = first
            for _step in pl.range(2):
                self.consume(data)
                # Rebinds the carry to `second`: it would enter the loop with
                # `first`'s context and leave with `second`'s.
                data = self.pick(second)
            self.consume(data)

    with pytest.raises(ValueError, match="Rebinding loop-carried DistributedTensor"):
        _apply(passes.convert_to_ssa()(P))


def test_submit_result_forwards_returned_param_ctx():
    """A Submit result position maps back to the arg it writes back, like Call."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def stage(
            self, data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]]
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32]):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]]):
            produced, _tid = pl.submit(self.stage, data)
            self.consume(produced)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def stage(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
        ) -> pld.DistributedTensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def consume(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def main(
            self,
            data: pl.InOut[pld.DistributedTensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
        ):
            # The submit gets its ctx arg like any Call, and `produced` — result
            # position 0, ahead of the trailing TASK_ID — resolves back through
            # `stage`'s return to `data`, so `consume` reuses the same ctx.
            produced, _tid = pl.submit(
                self.stage, data, data_ctx, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
            )
            self.consume(produced, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]})

    ir.assert_structural_equal(_apply(Before), Expected)


def test_distributed_if_keeps_current_no_context_yield_behavior():
    """The `if` is left alone: no CommCtxType is added to its return vars."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            cond: pl.Scalar[pl.BOOL],
        ):
            result = data
            if cond:
                result = data
            ctx = pld.system.get_comm_ctx(result)
            _rank = pld.system.rank(ctx)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def kernel(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            cond: pl.Scalar[pl.BOOL],
            data_ctx: pld.CommCtx,
        ):
            # `_result` keeps the `if` return var the Before program has; the
            # pass adds no CommCtxType yield alongside it, and the query below
            # resolves straight to the materialized parameter. Underscored
            # because nothing reads it once `get_comm_ctx` is gone.
            _result = data
            if cond:
                _result = data
            ctx = data_ctx
            _rank = pld.system.rank(ctx)

    ir.assert_structural_equal(_apply(Before), Expected)


def test_device_get_comm_ctx_is_replaced_by_materialized_context():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[4], pl.FP32]):
            ctx = pld.system.get_comm_ctx(data)
            _rank = pld.system.rank(ctx)
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self):
            buffer = pld.alloc_window_buffer(4 * pl.FP32.get_byte())
            data = pld.window(buffer, [4], dtype=pl.FP32)
            self.chip_orch(data, device=0)

    @pl.program
    class ExpectedChipOrch:
        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip_orch(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            # The query is gone: `ctx` is now a plain alias of the materialized
            # parameter, which the final Simplify folds away.
            ctx = data_ctx
            _rank = pld.system.rank(ctx)
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

    result = _apply(Before)
    ir.assert_structural_equal(_get_func(result, "chip_orch"), _get_func(ExpectedChipOrch, "chip_orch"))

    # `host` is compared by assertion rather than structurally: its body is
    # wrapped in a CommDomainScopeStmt by MaterializeCommDomainScopes, and that
    # node has no DSL surface to spell out in an Expected program.
    host_queries = _collect_calls(_get_func(result, "host").body, "pld.system.get_comm_ctx")
    assert len(host_queries) == 1, "host orchestration must keep the runtime query"


def test_device_call_without_materialized_context_rejects_synthesized_prefix():
    span = _span()
    data_ty = _dist_ty()
    bool_ty = ir.ScalarType(DataType.BOOL)

    predicate_data = ir.Var("data", data_ty, span)
    predicate = ir.Function(
        "predicate",
        [(predicate_data, ir.ParamDirection.In)],
        [bool_ty],
        ir.ReturnStmt([ir.ConstBool(True, span)], span),
        span,
        ir.FunctionType.InCore,
    )

    # `local_data` is a window created inside a chip-orchestration function, so
    # it is a genuine new DistributedTensor rather than a view of a parameter —
    # there is no context for it to inherit, and outside host orchestration the
    # pass cannot query one either.
    buffer = ir.Var("buffer", ir.PtrType.get(), span)
    local_data = ir.Var("local_data", data_ty, span)
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
        [],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(buffer, _alloc_window_buffer(span), span),
                ir.AssignStmt(local_data, _window(buffer, span), span),
                ir.EvalStmt(predicate_call, span),
            ],
            span,
        ),
        span,
        ir.FunctionType.Orchestration,
    )

    program = ir.Program([predicate, main], "unsupported_prefix_context", span)
    # A user-facing limitation, not an internal invariant: the message must name
    # the tensor and the function, and say what to change.
    with pytest.raises(ValueError, match="Cannot determine the communication context"):
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


def test_return_call_reuses_returned_param_ctx():
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
    exp_producer_call_ty = ir.TupleType([])
    exp_producer_call = ir.Call(
        ir.GlobalVar("producer"),
        [exp_main_source_data, exp_main_source_ctx_param],
        {},
        {"arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.Scalar]},
        exp_producer_call_ty,
        span,
    )
    exp_call = ir.Call(
        ir.GlobalVar("callee"),
        [exp_local_data, exp_main_source_ctx_param],
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
