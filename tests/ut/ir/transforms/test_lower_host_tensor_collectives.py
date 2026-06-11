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

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir, passes


@pytest.fixture(autouse=True)
def _basic_verification_context():
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


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


def test_host_allreduce_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(1024)
            signal_buf = pld.alloc_window_buffer(16)
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
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
            data_buf = pld.alloc_window_buffer(1024)
            signal_buf = pld.alloc_window_buffer(16)
            other_buf = pld.alloc_window_buffer(512)
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
            data_buf = pld.alloc_window_buffer(1024)
            signal_buf = pld.alloc_window_buffer(4)
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [1], dtype=pl.INT32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(Exception, match=r"signal shape\[0\].*participating device count"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_rejects_unsupported_builtin_dtype_variant():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP16]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(512)
            signal_buf = pld.alloc_window_buffer(16)
            data = pld.window(data_buf, [256], dtype=pl.FP16)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(Exception, match="currently supports only"):
        passes.lower_host_tensor_collectives()(program)
