# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.tensor.alloc_window_buffer`` (and its
``pld.alloc_window_buffer`` short form).

The alloc op supports two forms:

* **Canonical byte form:** ``alloc_window_buffer(size)`` — ``size`` is a scalar
  **byte** count; no shape, no dtype on the alloc.
* **Shape+dtype convenience overload:** ``alloc_window_buffer(shape, *, dtype=...)``
  — ``shape`` is a list / tuple of per-rank dimensions. The byte size is computed
  automatically as ``product(shape) × dtype.get_byte()`` and the call normalizes
  to the canonical byte form in IR.

The op returns the singleton :class:`PtrType`; the parser binds the LHS as
a plain :class:`ir.Var` of type :class:`ir.PtrType`. The
comm-collection pass later wraps the Ptr in an :class:`ir.WindowBuffer` Var
subclass and registers it on ``CommDomainScopeStmt wrappers in each host_orch body``.
The LHS variable name flows through ``Var.name_hint`` (and is also injected
as the op's ``name`` kwarg so the comm-collection pass can find it).
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType
from pypto.pypto_core import ir


def _get_host_orch(program: ir.Program, name: str = "host_orch") -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None, f"Function '{name}' not found in program"
    return program.functions[gvar]


def _find_alloc_assignment(func: ir.Function) -> ir.AssignStmt:
    """Return the first AssignStmt whose RHS is a ``pld.tensor.alloc_window_buffer`` Call."""

    def walk(stmt: ir.Stmt) -> ir.AssignStmt | None:
        if isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "pld.tensor.alloc_window_buffer":
                return stmt
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                hit = walk(s)
                if hit is not None:
                    return hit
        return None

    hit = walk(func.body)
    assert hit is not None, "no pld.tensor.alloc_window_buffer assignment found in function body"
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
            if stmt.value.op.name == "pld.tensor.alloc_window_buffer":
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


def test_alloc_window_buffer_long_form():
    """``pld.tensor.alloc_window_buffer(N)`` (canonical 3-segment form) parses
    to the same registered op as the unified short form."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.tensor.alloc_window_buffer(64)
            return buf

    func = _get_host_orch(P)
    stmt = _find_alloc_assignment(func)
    assert isinstance(stmt.value, ir.Call)
    assert stmt.value.op.name == "pld.tensor.alloc_window_buffer"
    assert stmt.value.kwargs["name"] == "buf"


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
    """``dtype=`` is rejected on the scalar byte form — it is only valid with the shape form."""
    with pytest.raises(Exception, match="dtype= is only valid when the first argument is a shape"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer(8, dtype=pl.FP32)  # pyright: ignore[reportArgumentType] # noqa: F841
                return buf


def test_alloc_window_buffer_rejects_explicit_name_kwarg():
    """``name`` is parser-injected from the LHS and can't be passed explicitly."""
    with pytest.raises(Exception, match="'name' kwarg cannot be passed explicitly"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer(8, name="other")  # noqa: F841
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


def test_alloc_window_buffer_rejects_list_without_dtype():
    """A list/tuple without ``dtype=`` is rejected — the shape form requires dtype."""
    with pytest.raises(Exception, match="requires dtype="):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer([256])  # pyright: ignore[reportArgumentType] # noqa: F841
                return buf


def test_alloc_window_buffer_shaped_static():
    """Shape+dtype form with static int-literal dimensions folds to a single ``ConstInt``
    carrying the total byte size (not a Mul chain)."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer([64, 128], dtype=pl.FP32)
            return buf

    func = _get_host_orch(P)
    stmt = _find_alloc_assignment(func)
    assert isinstance(stmt.value, ir.Call)
    call = stmt.value
    assert call.op.name == "pld.tensor.alloc_window_buffer"
    assert call.kwargs["name"] == "buf"
    assert "dtype" not in call.kwargs
    assert len(call.args) == 1
    # Static shape must fold to a single ConstInt byte-size arg.
    assert isinstance(call.args[0], ir.ConstInt)
    assert call.args[0].value == 64 * 128 * 4  # FP32 = 4 bytes
    assert call.args[0].dtype == DataType.INT64


def test_alloc_window_buffer_shaped_long_form():
    """Shape+dtype form works with the 3-segment ``pld.tensor.alloc_window_buffer`` form
    and folds static dims to a ConstInt byte-size arg."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.tensor.alloc_window_buffer([32, 16], dtype=pl.INT32)
            return buf

    func = _get_host_orch(P)
    stmt = _find_alloc_assignment(func)
    assert isinstance(stmt.value, ir.Call)
    call = stmt.value
    assert call.op.name == "pld.tensor.alloc_window_buffer"
    assert call.kwargs["name"] == "buf"
    assert len(call.args) == 1
    # Static shape must fold to a single ConstInt byte-size arg.
    assert isinstance(call.args[0], ir.ConstInt)
    assert call.args[0].value == 32 * 16 * 4  # INT32 = 4 bytes
    assert call.args[0].dtype == DataType.INT64


def test_alloc_window_buffer_rejects_empty_shape():
    """An empty shape list is rejected with a clear error."""
    with pytest.raises(Exception, match="shape must be non-empty"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer([], dtype=pl.FP32)  # noqa: F841
                return buf


def test_alloc_window_buffer_rejects_non_positive_static_dim():
    """Zero and negative static dimensions are rejected with a clear error."""
    with pytest.raises(Exception, match="all dimensions must be positive"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer([0, 128], dtype=pl.FP32)  # noqa: F841
                return buf

    with pytest.raises(Exception, match="all dimensions must be positive"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer([64, -1], dtype=pl.FP32)  # noqa: F841
                return buf


def test_alloc_window_buffer_shaped_dynamic():
    """Shape+dtype form with a dynamic dim (``pld.world_size()``) produces a Mul chain
    (not a single ConstInt) and parses successfully."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer([64, pld.world_size()], dtype=pl.FP32)
            return buf

    func = _get_host_orch(P)
    stmt = _find_alloc_assignment(func)
    assert isinstance(stmt.value, ir.Call)
    call = stmt.value
    assert call.op.name == "pld.tensor.alloc_window_buffer"
    assert call.kwargs["name"] == "buf"
    assert "dtype" not in call.kwargs
    assert len(call.args) == 1
    # Dynamic shape — the byte size is a Mul chain, not a ConstInt.
    byte_size = call.args[0]
    assert isinstance(byte_size, ir.Mul)
    assert isinstance(byte_size.type, ir.ScalarType)
    assert byte_size.type.dtype == DataType.INT64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
