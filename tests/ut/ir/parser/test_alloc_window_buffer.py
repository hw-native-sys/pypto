# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.alloc_window_buffer``.

After the MemRef-mirror redesign, the alloc op is a pure-allocation primitive
(parallel to ``tile.alloc(memspace, size)``):

* ``size`` is a scalar **byte** count; no shape, no dtype on the alloc.
* The op returns the singleton :class:`PtrType`; the parser binds the LHS as
  a plain :class:`ir.Var` of type :class:`ir.PtrType`. The
  comm-collection pass later wraps the Ptr in an :class:`ir.WindowBuffer` Var
  subclass and registers it on ``Program.comm_groups``.
* The LHS variable name flows through ``Var.name_hint`` (and is also injected
  as the op's ``name`` kwarg so the comm-collection pass can find it).
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir


def _get_host_orch(program: ir.Program, name: str = "host_orch") -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None, f"Function '{name}' not found in program"
    return program.functions[gvar]


def _find_alloc_assignment(func: ir.Function) -> ir.AssignStmt:
    """Return the first AssignStmt whose RHS is a ``pld.alloc_window_buffer`` Call."""

    def walk(stmt: ir.Stmt) -> ir.AssignStmt | None:
        if isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "pld.alloc_window_buffer":
                return stmt
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                hit = walk(s)
                if hit is not None:
                    return hit
        return None

    hit = walk(func.body)
    assert hit is not None, "no pld.alloc_window_buffer assignment found in function body"
    return hit


def test_alloc_window_buffer_lhs_is_plain_ptr_var():
    """The LHS variable is a plain ``ir.Var`` of type ``ir.PtrType`` —
    no specialised ``WindowBuffer`` Var subclass at parse time."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(1024)
            return buf  # noqa: RET504

    func = _get_host_orch(P)
    stmt = _find_alloc_assignment(func)
    var = stmt.var
    # Plain Var with PtrType — exact mirror of `mem_vec_7: Ptr = tile.alloc(...)`.
    assert isinstance(var, ir.Var)
    assert not isinstance(var, ir.WindowBuffer)
    assert isinstance(var.type, ir.PtrType)
    # The buffer's runtime-unique identifier comes from the LHS variable name
    # via Var.name_hint.
    assert var.name_hint == "buf"


def test_alloc_window_buffer_call_carries_name_kwarg():
    """The op call's kwargs carry the LHS-injected name. No dtype kwarg."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(512)
            return data_buf

    func = _get_host_orch(P)
    stmt = _find_alloc_assignment(func)
    assert isinstance(stmt.value, ir.Call)
    call = stmt.value
    assert call.kwargs["name"] == "data_buf"
    assert "dtype" not in call.kwargs
    assert len(call.args) == 1
    assert isinstance(call.args[0], ir.ConstInt)
    assert call.args[0].value == 512


def test_alloc_window_buffer_returns_singleton_ptr_type():
    """Different alloc sites all return the SAME singleton PtrType."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf_a = pld.alloc_window_buffer(8)
            buf_b = pld.alloc_window_buffer(16)  # noqa: F841
            return buf_a

    func = _get_host_orch(P)
    allocs: list[ir.AssignStmt] = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            if stmt.value.op.name == "pld.alloc_window_buffer":
                allocs.append(stmt)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)

    walk(func.body)
    assert len(allocs) == 2
    type_a = allocs[0].value.type
    type_b = allocs[1].value.type
    assert isinstance(type_a, ir.PtrType)
    assert isinstance(type_b, ir.PtrType)
    assert ir.structural_equal(type_a, type_b)


def test_alloc_window_buffer_rejects_non_name_lhs():
    """Tuple-unpacking / subscript / attribute LHS is rejected — name must be a bare identifier."""
    with pytest.raises(Exception, match="must appear as the RHS of a simple assignment"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                a, b = pld.alloc_window_buffer(8), 0  # noqa: F841
                return a


def test_alloc_window_buffer_rejects_duplicate_names():
    with pytest.raises(Exception, match="already declared"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer(8)
                buf = pld.alloc_window_buffer(8)  # noqa: F841
                return buf


def test_alloc_window_buffer_rejects_user_kwargs():
    """The user-facing alloc takes only a positional size — no kwargs are allowed."""
    with pytest.raises(Exception, match="does not accept user-supplied kwargs"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer(8, dtype=pl.FP32)  # noqa: F841
                return buf


def test_alloc_window_buffer_rejects_bare_call_outside_assignment():
    """Without an assignment LHS there is no globally-unique name to bind to."""
    with pytest.raises(Exception, match="must appear as the RHS of a simple assignment"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                pld.alloc_window_buffer(8)
                return 0


def test_alloc_window_buffer_rejects_list_for_size():
    """The redesigned signature takes a scalar byte size, not a shape list."""
    with pytest.raises(Exception, match="size must be a scalar"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer([256])  # pyright: ignore[reportArgumentType] # noqa: F841
                return buf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
