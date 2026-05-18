# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.get_comm_ctx`` / ``pld.comm_ctx.rank`` / ``pld.comm_ctx.nranks``.

These ops are called explicitly (no attribute-access sugar). Dispatch
mirrors the rest of the ``pld.*`` surface:

* ``pld.get_comm_ctx(data)`` — 2-segment, routed through
  :meth:`_parse_pld_op` / :func:`_dispatch_op` to the DSL wrapper in
  :mod:`pypto.language.distributed.op.system_ops`.
* ``pld.comm_ctx.rank(ctx)`` / ``pld.comm_ctx.nranks(ctx)`` — 3-segment,
  routed through :meth:`_parse_pld_comm_ctx_op` to the DSL wrappers in
  :mod:`pypto.language.distributed.op.comm_ctx_ops`.

Verifier-level negatives (plain ``pl.Tensor`` into ``pld.get_comm_ctx``,
non-CommCtx into ``pld.comm_ctx.rank``) come from the C++ op definitions
in :file:`src/ir/op/distributed/get_comm_ctx.cpp`.
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


def _find_calls_in_func(func: ir.Function, op_name: str) -> list[ir.Call]:
    found: list[ir.Call] = []

    def visit(expr: ir.Expr | None) -> None:
        if expr is None:
            return
        if isinstance(expr, ir.Call):
            if expr.op.name == op_name:
                found.append(expr)
            for a in expr.args:
                visit(a)
        elif isinstance(expr, ir.BinaryExpr):
            visit(expr.left)
            visit(expr.right)

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            visit(stmt.value)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ReturnStmt):
            for v in stmt.value:
                visit(v)

    walk(func.body)
    return found


def test_get_comm_ctx_returns_comm_ctx_typed_call():
    """``pld.get_comm_ctx(data)`` parses to a Call of type CommCtxType."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.get_comm_ctx(data)
            return ctx

    func = _get_func(P, "worker")
    calls = _find_calls_in_func(func, "pld.get_comm_ctx")
    assert len(calls) == 1
    assert isinstance(calls[0].type, ir.CommCtxType)
    assert len(calls[0].args) == 1
    assert isinstance(calls[0].args[0].type, ir.DistributedTensorType)


def test_comm_ctx_rank_explicit_call():
    """``pld.comm_ctx.rank(ctx)`` parses to the rank op with an INT32 result."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.get_comm_ctx(data)
            return pld.comm_ctx.rank(ctx)

    func = _get_func(P, "worker")
    rank_calls = _find_calls_in_func(func, "pld.comm_ctx.rank")
    assert len(rank_calls) == 1
    assert isinstance(rank_calls[0].type, ir.ScalarType)
    assert rank_calls[0].type.dtype == DataType.INT32


def test_comm_ctx_nranks_explicit_call():
    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.get_comm_ctx(data)
            return pld.comm_ctx.nranks(ctx)

    func = _get_func(P, "worker")
    nranks_calls = _find_calls_in_func(func, "pld.comm_ctx.nranks")
    assert len(nranks_calls) == 1
    assert isinstance(nranks_calls[0].type, ir.ScalarType)
    assert nranks_calls[0].type.dtype == DataType.INT32


def test_comm_ctx_rank_inline_nested_get_comm_ctx():
    """``pld.comm_ctx.rank(pld.get_comm_ctx(data))`` parses to the nested Call form."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            return pld.comm_ctx.rank(pld.get_comm_ctx(data))

    func = _get_func(P, "worker")
    assert len(_find_calls_in_func(func, "pld.comm_ctx.rank")) == 1
    assert len(_find_calls_in_func(func, "pld.get_comm_ctx")) == 1


def test_comm_rank_and_nranks_compose_in_expression():
    """rank + nranks composes through arithmetic; both Calls survive in IR."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.get_comm_ctx(data)
            return pld.comm_ctx.rank(ctx) + pld.comm_ctx.nranks(ctx)

    func = _get_func(P, "worker")
    assert len(_find_calls_in_func(func, "pld.comm_ctx.rank")) == 1
    assert len(_find_calls_in_func(func, "pld.comm_ctx.nranks")) == 1


def test_get_comm_ctx_rejects_plain_tensor():
    """The C++ verifier refuses a plain ``pl.Tensor`` — precise ObjectKind match."""
    with pytest.raises(Exception, match="DistributedTensor"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def worker(self, x: pl.Tensor[[64], pl.FP32]):
                return pld.get_comm_ctx(x)  # type: ignore[arg-type]


def test_comm_ctx_rank_rejects_non_comm_ctx_arg():
    """The C++ verifier refuses any non-CommCtx argument to comm_ctx.rank."""
    with pytest.raises(Exception, match="CommCtx"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
                return pld.comm_ctx.rank(data)  # type: ignore[arg-type]


def test_unknown_comm_ctx_op_rejected():
    """Unknown 3-segment ``pld.comm_ctx.<foo>`` produces a clear parser error."""
    with pytest.raises(Exception, match=r"pld\.comm_ctx\.foo"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
                ctx = pld.get_comm_ctx(data)
                return pld.comm_ctx.foo(ctx)  # type: ignore[attr-defined]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
