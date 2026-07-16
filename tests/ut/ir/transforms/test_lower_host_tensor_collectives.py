# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Tests for ``LowerHostTensorCollectives``."""

from typing import cast

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import DataType, ir, passes


@pytest.fixture(autouse=True)
def _basic_verification_context():
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _as_call(expr: ir.Expr) -> ir.Call:
    assert isinstance(expr, ir.Call)
    return expr


def _as_var(expr: ir.Expr) -> ir.Var:
    assert isinstance(expr, ir.Var)
    return expr


def _eval_call(stmt: ir.Stmt) -> ir.Call:
    assert isinstance(stmt, ir.EvalStmt)
    return _as_call(stmt.expr)


def _collect_for_stmts(stmt: ir.Stmt) -> list[ir.ForStmt]:
    found: list[ir.ForStmt] = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.ForStmt):
            found.append(s)
            walk(s.body)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk(child)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)

    walk(stmt)
    return found


def _collect_assign_stmts(stmt: ir.Stmt) -> list[ir.AssignStmt]:
    found: list[ir.AssignStmt] = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.AssignStmt):
            found.append(s)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk(child)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)

    walk(stmt)
    return found


def _collect_return_stmts(stmt: ir.Stmt) -> list[ir.ReturnStmt]:
    found: list[ir.ReturnStmt] = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.ReturnStmt):
            found.append(s)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk(child)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)

    walk(stmt)
    return found


def _last_stmt(stmt: ir.Stmt) -> ir.Stmt:
    if isinstance(stmt, ir.ScopeStmt):
        return _last_stmt(stmt.body)
    if isinstance(stmt, ir.SeqStmts):
        assert len(stmt.stmts) > 0
        return _last_stmt(stmt.stmts[-1])
    return stmt


def _assert_alias_keeps_window_buffer(alias: ir.AssignStmt) -> None:
    lhs_type = alias.var.type
    rhs = alias.value
    assert isinstance(rhs, ir.Var)
    rhs_type = rhs.type
    assert isinstance(lhs_type, ir.DistributedTensorType)
    assert isinstance(rhs_type, ir.DistributedTensorType)
    assert lhs_type.window_buffer is not None
    assert lhs_type.window_buffer is rhs_type.window_buffer


def test_host_allreduce_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = cast(ir.Program, passes.materialize_comm_domain_scopes()(P))
    result = cast(ir.Program, passes.lower_host_tensor_collectives()(program))
    host = _get_func(result, "host_orch")

    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.allreduce"
    ]
    assert len(builtin_loops) == 1

    body = builtin_loops[0].body
    assert isinstance(body, ir.EvalStmt)
    call = body.expr
    assert isinstance(call, ir.Call)
    assert call.kwargs["op"] == int(pld.ReduceOp.Sum)
    assert call.kwargs["dtype"] == pl.FP32
    assert call.attrs["op"] == int(pld.ReduceOp.Sum)
    assert call.attrs["dtype"] == pl.FP32
    assert call.attrs["device"] is builtin_loops[0].loop_var
    assert list(call.arg_directions) == [ir.ArgDirection.InOut, ir.ArgDirection.InOut]


def test_implicit_host_allreduce_synthesizes_signal_then_lowers():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    program = passes.synthesize_allreduce_signals()(P)
    program = passes.materialize_comm_domain_scopes()(program)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")

    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.allreduce"
    ]
    assert len(builtin_loops) == 1

    call = _eval_call(builtin_loops[0].body)
    assert len(call.args) == 2
    assert _as_var(call.args[1]).name_hint == "__allreduce_signal_0"
    assert list(call.arg_directions) == [ir.ArgDirection.InOut, ir.ArgDirection.InOut]


def test_return_implicit_host_allreduce_synthesizes_signal_then_lowers():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)

    program = cast(ir.Program, passes.synthesize_allreduce_signals()(P))
    program = cast(ir.Program, passes.materialize_comm_domain_scopes()(program))
    result = cast(ir.Program, passes.lower_host_tensor_collectives()(program))
    host = _get_func(result, "host_orch")

    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.allreduce"
    ]
    returns = [
        stmt
        for stmt in _collect_assign_stmts(host.body)
        if isinstance(stmt.value, ir.Var) and stmt.var.name_hint.startswith("__allreduce_result_")
    ]
    return_stmts = _collect_return_stmts(host.body)

    assert len(builtin_loops) == 1
    call = _eval_call(builtin_loops[0].body)
    assert len(call.args) == 2
    assert _as_var(call.args[1]).name_hint == "__allreduce_signal_0"
    assert list(call.arg_directions) == [ir.ArgDirection.InOut, ir.ArgDirection.InOut]
    assert len(returns) == 1
    _assert_alias_keeps_window_buffer(returns[0])
    assert len(return_stmts) == 1
    assert isinstance(return_stmts[0].value[0], ir.Var)
    assert return_stmts[0].value[0] is returns[0].var
    assert _last_stmt(host.body) is return_stmts[0]


def test_return_explicit_host_allreduce_lowers_with_user_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)

    program = cast(ir.Program, passes.synthesize_allreduce_signals()(P))
    program = cast(ir.Program, passes.materialize_comm_domain_scopes()(program))
    result = cast(ir.Program, passes.lower_host_tensor_collectives()(program))
    host = _get_func(result, "host_orch")

    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.allreduce"
    ]
    returns = [
        stmt
        for stmt in _collect_assign_stmts(host.body)
        if isinstance(stmt.value, ir.Var) and stmt.var.name_hint.startswith("__allreduce_result_")
    ]
    return_stmts = _collect_return_stmts(host.body)

    assert len(builtin_loops) == 1
    call = _eval_call(builtin_loops[0].body)
    assert len(call.args) == 2
    assert _as_var(call.args[1]).name_hint == "signal"
    assert list(call.arg_directions) == [ir.ArgDirection.InOut, ir.ArgDirection.InOut]
    assert len(returns) == 1
    _assert_alias_keeps_window_buffer(returns[0])
    assert len(return_stmts) == 1
    assert isinstance(return_stmts[0].value[0], ir.Var)
    assert return_stmts[0].value[0] is returns[0].var
    assert _last_stmt(host.body) is return_stmts[0]


def test_host_allreduce_assign_result_var_carries_window_buffer():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            self.chip_orch(data, device=0)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")

    allreduce_aliases = [
        stmt
        for stmt in _collect_assign_stmts(host.body)
        if isinstance(stmt.value, ir.Var)
        and isinstance(stmt.var.type, ir.DistributedTensorType)
        and isinstance(stmt.value.type, ir.DistributedTensorType)
        and stmt.var.name_hint == "data"
    ]
    assert len(allreduce_aliases) == 1
    _assert_alias_keeps_window_buffer(allreduce_aliases[0])


def test_host_allreduce_chained_assign_uses_remapped_result_var():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            signal_buf_1 = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            signal_1 = pld.window(signal_buf_1, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            data = pld.tensor.allreduce(data, signal_1, op=pld.ReduceOp.Sum)
            self.chip_orch(data, device=0)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")

    allreduce_aliases = [
        stmt
        for stmt in _collect_assign_stmts(host.body)
        if isinstance(stmt.value, ir.Var)
        and isinstance(stmt.var.type, ir.DistributedTensorType)
        and isinstance(stmt.value.type, ir.DistributedTensorType)
        and stmt.var.name_hint == "data"
    ]
    assert len(allreduce_aliases) == 2
    for alias in allreduce_aliases:
        _assert_alias_keeps_window_buffer(alias)


def test_host_allreduce_resolves_non_innermost_comm_domain_scope():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(type=pl.FunctionType.Orchestration)
        def other_chip_orch(self, data: pld.DistributedTensor[[128], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            other_buf = pld.alloc_window_buffer(128 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            other = pld.window(other_buf, [128], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            self.other_chip_orch(other, device=0)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")

    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.allreduce"
    ]
    assert len(builtin_loops) == 1
    assert isinstance(builtin_loops[0].stop, ir.Call)
    assert builtin_loops[0].stop.op.name == "pld.system.world_size"


def test_host_allreduce_rejects_static_signal_smaller_than_explicit_device_count():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [1], dtype=pl.INT32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(Exception, match=r"signal shape\[0\].*participating device count"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_rejects_rank2_signal_with_dynamic_second_extent():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            width = pld.world_size()
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), width], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(Exception, match=r"rank-2 signal shape\[1\] must be the constant 1"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_rejects_unsupported_builtin_dtype_variant():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP16]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP16.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP16)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(Exception, match="currently supports only"):
        passes.lower_host_tensor_collectives()(program)


def test_host_barrier_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
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
            pld.tensor.barrier(signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")
    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.barrier"
    ]
    assert len(builtin_loops) == 1
    body = builtin_loops[0].body
    assert isinstance(body, ir.EvalStmt)
    call = body.expr
    assert isinstance(call, ir.Call)
    assert list(call.arg_directions) == [ir.ArgDirection.InOut]


def test_host_broadcast_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
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
            pld.tensor.broadcast(data, signal, root=0)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")
    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.broadcast"
    ]
    assert len(builtin_loops) == 1
    body = builtin_loops[0].body
    assert isinstance(body, ir.EvalStmt)
    call = body.expr
    assert isinstance(call, ir.Call)
    assert list(call.arg_directions) == [ir.ArgDirection.InOut, ir.ArgDirection.InOut]
    assert call.kwargs["root"] == 0


def test_host_reduce_scatter_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[4, 256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")
    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.reduce_scatter"
    ]
    assert len(builtin_loops) == 1
    body = builtin_loops[0].body
    assert isinstance(body, ir.EvalStmt)
    call = body.expr
    assert isinstance(call, ir.Call)
    assert list(call.arg_directions) == [ir.ArgDirection.InOut, ir.ArgDirection.InOut]
    assert call.kwargs["op"] == int(pld.ReduceOp.Sum)


def test_host_allgather_lowers_to_namesake_builtin():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[1, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            # `stage_buf` ([1, SIZE] TPUT source, this rank's chunk) and
            # `data_buf` ([NR, SIZE] TPUT destination / result) must be two
            # DISTINCT windows — same constraint as all_to_all (see allgather
            # kernel.cpp.in).
            stage_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            stage = pld.window(stage_buf, [1, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.allgather(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")
    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.allgather"
    ]
    assert len(builtin_loops) == 1
    body = builtin_loops[0].body
    assert isinstance(body, ir.EvalStmt)
    call = body.expr
    assert isinstance(call, ir.Call)
    assert list(call.arg_directions) == [
        ir.ArgDirection.Input,
        ir.ArgDirection.InOut,
        ir.ArgDirection.InOut,
    ]
    assert call.kwargs["dtype"] == DataType.FP32


def test_host_allgather_rejects_aliased_input_target_windows():
    """Two pld.window views over one alloc must fail at host lowering."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[1, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            stage = pld.window(buf, [1, 256], dtype=pl.FP32)
            data = pld.window(buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.allgather(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(Exception, match=r"different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_rejects_aliased_input_target_windows():
    """Two pld.window views over one alloc must fail at host lowering."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[4, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            stage = pld.window(buf, [4, 256], dtype=pl.FP32)
            data = pld.window(buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.all_to_all(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(Exception, match=r"different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_lowers_to_namesake_builtin():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[4, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            # `stage_buf` (TPUT source) and `data_buf` (TPUT destination /
            # result) must be two DISTINCT windows — reusing one buffer for
            # both is a genuine cross-process data race (see
            # python/pypto/runtime/builtins/collectives/all_to_all/templates
            # /kernel.cpp.in for the full explanation).
            stage_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            stage = pld.window(stage_buf, [4, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.all_to_all(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")
    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == "builtin.tensor.all_to_all"
    ]
    assert len(builtin_loops) == 1
    body = builtin_loops[0].body
    assert isinstance(body, ir.EvalStmt)
    call = body.expr
    assert isinstance(call, ir.Call)
    assert list(call.arg_directions) == [
        ir.ArgDirection.Input,
        ir.ArgDirection.InOut,
        ir.ArgDirection.InOut,
    ]
    assert call.kwargs["dtype"] == DataType.FP32
