# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.window``.

After the MemRef-mirror redesign:

* ``pld.window(buf, shape, dtype=...)`` consumes a ``Ptr``-typed Var
  (the LHS of ``pld.alloc_window_buffer``) plus an explicit ``shape`` list
  and a ``dtype`` kwarg.
* Returns a :class:`ir.DistributedTensorType` carrying shape and dtype.
  ``window_buffer`` back-reference is **None** at parse time — the
  comm-collection pass populates it later.
"""

from typing import cast

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir


def _get_host_orch(program: ir.Program) -> ir.Function:
    gvar = program.get_global_var("host_orch")
    assert gvar is not None
    return program.functions[gvar]


def _find_call(func: ir.Function, op_name: str) -> ir.Call:
    found: list[ir.Call] = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            if isinstance(stmt.value, ir.Call) and stmt.value.op.name == op_name:
                found.append(stmt.value)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)

    walk(func.body)
    assert found, f"no {op_name} call found in function body"
    return found[0]


def _find_alloc_var(func: ir.Function) -> ir.Var:
    """Return the LHS Var of the first pld.alloc_window_buffer assignment."""

    def walk(stmt: ir.Stmt) -> ir.Var | None:
        if (
            isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == "pld.alloc_window_buffer"
        ):
            return stmt.var
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                hit = walk(s)
                if hit is not None:
                    return hit
        return None

    hit = walk(func.body)
    assert hit is not None
    return hit


def test_window_returns_distributed_tensor_type_no_buffer_yet():
    """Parse-time: result type carries shape + dtype; window_buffer is None."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(1024)
            data = pld.window(buf, [256], dtype=pl.FP32)
            return data

    func = _get_host_orch(P)
    win_call = _find_call(func, "pld.window")
    assert isinstance(win_call.type, ir.DistributedTensorType)
    assert win_call.type.dtype == pl.FP32
    shape = win_call.type.shape
    assert len(shape) == 1
    assert isinstance(shape[0], ir.ConstInt)
    assert shape[0].value == 256
    # window_buffer is filled in by the comm-collection pass; not yet at parse
    # time. Mirrors how TensorType.memref starts as None until InitMemRef runs.
    assert win_call.type.window_buffer is None


def test_window_input_is_alloc_ptr_var():
    """The window op's first input is the plain Ptr Var bound by alloc."""

    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(64)
            data = pld.window(buf, [16], dtype=pl.FP32)
            return data

    func = _get_host_orch(P)
    win_call = _find_call(func, "pld.window")
    buf_var = _find_alloc_var(func)
    assert len(win_call.args) == 2
    buf_arg = win_call.args[0]
    # The window op receives the same Var instance bound by alloc — there's
    # no second-binding indirection (this is the disconnect the redesign
    # fixed).
    assert buf_arg is buf_var
    assert isinstance(buf_arg, ir.Var)
    assert isinstance(buf_arg.type, ir.PtrType)
    assert buf_arg.name_hint == "buf"


def test_window_propagates_multi_dim_shape():
    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(2048)
            data = pld.window(buf, [8, 64], dtype=pl.FP16)
            return data

    func = _get_host_orch(P)
    call = _find_call(func, "pld.window")
    dt = call.type
    assert isinstance(dt, ir.DistributedTensorType)
    assert dt.dtype == pl.FP16
    assert all(isinstance(d, ir.ConstInt) for d in dt.shape)
    assert [int(cast(ir.ConstInt, d).value) for d in dt.shape] == [8, 64]


def test_window_rejects_non_ptr_arg():
    """A non-Ptr-typed Var cannot stand in for a Ptr handle."""
    with pytest.raises(Exception, match="Ptr handle"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]):
                data = pld.window(x, [16], dtype=pl.FP32)  # type: ignore[arg-type]  # noqa: F841
                return 0


def test_window_rejects_unknown_kwarg():
    with pytest.raises(Exception, match="does not accept kwarg"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer(8)
                data = pld.window(buf, [8], dtype=pl.FP32, target_memory=pl.Mem.DDR)  # noqa: F841
                return 0


def test_window_rejects_missing_shape_arg():
    with pytest.raises(Exception, match="2 positional"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer(8)
                data = pld.window(buf, dtype=pl.FP32)  # noqa: F841
                return 0


def test_window_rejects_missing_dtype_kwarg():
    with pytest.raises(Exception, match="dtype"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer(8)
                data = pld.window(buf, [8])  # noqa: F841
                return 0


def test_window_can_be_called_inside_for_loop():
    @pl.program
    class P:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(64)
            for _ in pl.range(0, 4):
                data = pld.window(buf, [16], dtype=pl.FP32)  # noqa: F841
            return 0

    func = _get_host_orch(P)
    win_call = _find_call(func, "pld.window")
    assert isinstance(win_call.type, ir.DistributedTensorType)


def test_alloc_names_globally_unique_across_functions():
    """A second function in the same @pl.program cannot reuse a buffer name."""
    with pytest.raises(Exception, match="already declared"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                buf = pld.alloc_window_buffer(8)
                return buf

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch_2(self):
                buf = pld.alloc_window_buffer(8)
                return buf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
