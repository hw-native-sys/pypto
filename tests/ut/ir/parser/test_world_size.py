# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.world_size()``.

``pld.world_size()`` is the host-only entry point for the distributed world
size. The parser lifts it to ``ir.OpExpr('pld.world_size')`` returning a
scalar INT64. Host-only enforcement happens at parse time: invocations
outside a ``level=pl.Level.HOST`` function body are rejected with a clear
error message.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType
from pypto.pypto_core import ir


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _find_world_size_calls(func: ir.Function) -> list[ir.Call]:
    """Return every ``pld.world_size()`` call appearing anywhere in ``func``.

    Walks both statements and expression subtrees so we catch calls nested
    inside another call's argument list (e.g.
    ``pld.alloc_window_buffer(pld.world_size() * 4)``).
    """
    found: list[ir.Call] = []

    def visit_expr(expr: ir.Expr | None) -> None:
        if expr is None or not isinstance(expr, ir.Call):
            return
        if expr.op.name == "pld.world_size":
            found.append(expr)
        for sub in expr.args:
            visit_expr(sub)

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            visit_expr(stmt.value)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            visit_expr(stmt.start)
            visit_expr(stmt.stop)
            visit_expr(stmt.step)
            walk(stmt.body)

    walk(func.body)
    return found


def test_world_size_call_returns_int64_scalar():
    """A bare ``pld.world_size()`` call lifts to a ScalarType(INT64) IR call."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            n = pld.world_size()
            return n

    func = _get_func(P, "host_orch")
    calls = _find_world_size_calls(func)
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.INT64
    assert call.args == []
    assert call.kwargs == {}


def test_world_size_can_drive_pl_range_bound():
    """``pl.range(pld.world_size())`` produces a for-loop whose stop bound is
    the IR call (no DynVar / sentinel rewrite)."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            for _ in pl.range(pld.world_size()):  # type: ignore[arg-type]
                pass
            return 0

    func = _get_func(P, "host_orch")

    def find_for(stmt: ir.Stmt) -> ir.ForStmt | None:
        if isinstance(stmt, ir.ForStmt):
            return stmt
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                hit = find_for(s)
                if hit is not None:
                    return hit
        return None

    for_stmt = find_for(func.body)
    assert for_stmt is not None
    # The stop bound is the result of pld.world_size() — a Var assigned from
    # the world_size call, since the parser threads expressions through SSA.
    stop = for_stmt.stop
    assert isinstance(stop.type, ir.ScalarType)
    assert stop.type.dtype == DataType.INT64


def test_world_size_rejects_positional_args():
    with pytest.raises(Exception, match="no positional arguments"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                n = pld.world_size(4)  # type: ignore[call-arg]  # noqa: F841
                return 0


def test_world_size_rejects_kwargs():
    with pytest.raises(Exception, match="does not accept keyword arguments"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                n = pld.world_size(rank=0)  # type: ignore[call-arg]  # noqa: F841
                return 0


def test_world_size_rejected_outside_host_function():
    """``pld.world_size()`` is host-only — calling it from a CORE_GROUP-level
    function body is a parse error."""
    with pytest.raises(Exception, match="HOST"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    n = pld.world_size()  # noqa: F841
                return x


def test_world_size_rejected_in_nested_device_scope_within_host_function():
    """Even inside a HOST orchestrator, ``pld.world_size()`` must be rejected
    when nested inside a device-side scope (InCore / AutoInCore / SPMD), since
    the call is not lowerable there."""
    with pytest.raises(Exception, match="InCore"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                with pl.at(level=pl.Level.CORE_GROUP):
                    n = pld.world_size()  # noqa: F841
                return 0


def test_world_size_call_used_as_size_in_alloc():
    """``pld.world_size()`` can flow into ``pld.alloc_window_buffer(size)`` as
    a per-rank byte-size operand."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(pld.world_size())
            return buf

    func = _get_func(P, "host_orch")
    calls = _find_world_size_calls(func)
    assert len(calls) == 1
    # The alloc op consumed the world_size call as its size argument.
    body = func.body
    assert isinstance(body, ir.SeqStmts)
    alloc_call = next(
        c
        for stmt in body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pld.alloc_window_buffer"
        for c in [stmt.value]
    )
    assert alloc_call.args[0] is calls[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
