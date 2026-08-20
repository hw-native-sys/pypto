# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.system.get_comm_ctx`` / ``pld.system.rank`` /
``pld.system.nranks`` (and their unified short forms) plus the canonical
``pld.system.defer_wait`` form.

These ops are called explicitly (no attribute-access sugar). Dispatch mirrors
the rest of the ``pld.*`` surface — the canonical 3-segment form and the
unified 2-segment short form both lift to the same registered IR op:

* 3-segment: ``pld.system.<op>(...)`` is routed through
  :meth:`_parse_pld_category_op` to the DSL wrappers in
  :mod:`pypto.language.distributed.op.system_ops`.
* 2-segment: ``pld.<op>(...)`` is routed through :meth:`_parse_pld_op` to the
  same wrappers via :mod:`pypto.language.distributed.op.unified_ops`.

Verifier-level negatives (plain ``pl.Tensor`` into ``pld.system.get_comm_ctx``,
non-CommCtx into ``pld.system.rank``) come from the C++ op definitions in
:file:`src/ir/op/distributed/get_comm_ctx.cpp`.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType
from pypto.language.parser.diagnostics import InvalidOperationError
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
        if isinstance(stmt, ir.EvalStmt):
            visit(stmt.expr)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)
        if isinstance(stmt, ir.ReturnStmt):
            for v in stmt.value:
                visit(v)

    walk(func.body)
    return found


def test_comm_ctx_param_accepts_wrapper_and_raw_type_spelling():
    """``pld.CommCtx`` and ``pld.CommCtxType`` name the same parameter type.

    A materialized communication-context parameter should be spelled with the
    DSL wrapper (mirroring what ``pld.get_comm_ctx`` hands back); the raw IR
    type name stays accepted for the text the printer emits today.
    """

    @pl.program
    class Wrapper:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32], ctx: pld.CommCtx):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

    @pl.program
    class RawType:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32], ctx: pld.CommCtxType):
            pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

    assert isinstance(_get_func(Wrapper, "worker").params[-1].type, ir.CommCtxType)
    ir.assert_structural_equal(Wrapper, RawType)


def test_comm_ctx_param_feeds_rank_like_a_queried_handle():
    """A materialized ctx parameter is usable wherever a queried handle is.

    This is the shape ``MaterializeDistTensorCtx`` produces, and it only type
    checks with the wrapper spelling — ``pld.system.rank`` takes a ``CommCtx``.
    """

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32], ctx: pld.CommCtx):
            return pld.system.rank(ctx)

    func = _get_func(P, "worker")
    assert isinstance(func.params[-1].type, ir.CommCtxType)
    assert len(_find_calls_in_func(func, "pld.system.rank")) == 1


def test_get_comm_ctx_returns_comm_ctx_typed_call():
    """``pld.get_comm_ctx(data)`` parses to a Call of type CommCtxType."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.get_comm_ctx(data)
            return ctx

    func = _get_func(P, "worker")
    calls = _find_calls_in_func(func, "pld.system.get_comm_ctx")
    assert len(calls) == 1
    assert isinstance(calls[0].type, ir.CommCtxType)
    assert len(calls[0].args) == 1
    assert isinstance(calls[0].args[0].type, ir.DistributedTensorType)


def test_rank_short_form():
    """``pld.rank(ctx)`` (short form) parses to the rank op with a UINT32 result."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.get_comm_ctx(data)
            return pld.rank(ctx)

    func = _get_func(P, "worker")
    rank_calls = _find_calls_in_func(func, "pld.system.rank")
    assert len(rank_calls) == 1
    assert isinstance(rank_calls[0].type, ir.ScalarType)
    assert rank_calls[0].type.dtype == DataType.INT32


def test_nranks_short_form():
    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.get_comm_ctx(data)
            return pld.nranks(ctx)

    func = _get_func(P, "worker")
    nranks_calls = _find_calls_in_func(func, "pld.system.nranks")
    assert len(nranks_calls) == 1
    assert isinstance(nranks_calls[0].type, ir.ScalarType)
    assert nranks_calls[0].type.dtype == DataType.INT32


def test_long_form_system_ops():
    """``pld.system.rank`` / ``pld.system.nranks`` (canonical 3-segment) parse
    to the same registered IR op as the short form."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.system.get_comm_ctx(data)
            return pld.system.rank(ctx) + pld.system.nranks(ctx)

    func = _get_func(P, "worker")
    assert len(_find_calls_in_func(func, "pld.system.get_comm_ctx")) == 1
    assert len(_find_calls_in_func(func, "pld.system.rank")) == 1
    assert len(_find_calls_in_func(func, "pld.system.nranks")) == 1


def test_defer_wait_long_form_parses_and_round_trips():
    """The canonical deferred-wait spelling creates the registered IR op."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def worker(self, signal: pld.DistributedTensor[[4], pl.INT32]):
            pld.system.defer_wait(signal, offsets=[0], expected=1, cmp=pld.WaitCmp.Ge)

    func = _get_func(P, "worker")
    calls = _find_calls_in_func(func, ir.get_op("pld.system.defer_wait").name)
    assert len(calls) == 1
    assert isinstance(calls[0].type, ir.UnknownType)
    assert calls[0].kwargs["cmp"] == int(ir.WaitCmp.Ge)

    reparsed = pl.parse_program(str(P))
    ir.assert_structural_equal(P, reparsed)


def test_defer_wait_accepts_index_expected_and_round_trips():
    """A ``pl.range`` induction variable is a valid INDEX threshold."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def worker(self, signal: pld.DistributedTensor[[4], pl.INT32]):
            for expected in pl.range(1, 4):
                pld.system.defer_wait(signal, offsets=[0], expected=expected, cmp=pld.WaitCmp.Ge)

    func = _get_func(P, "worker")
    calls = _find_calls_in_func(func, ir.get_op("pld.system.defer_wait").name)
    assert len(calls) == 1
    assert isinstance(calls[0].args[2].type, ir.ScalarType)
    assert calls[0].args[2].type.dtype == DataType.INDEX

    reparsed = pl.parse_program(str(P))
    ir.assert_structural_equal(P, reparsed)


def test_defer_wait_rejects_float_offset():
    """The parser surfaces the integer/index coordinate contract."""
    with pytest.raises(InvalidOperationError, match="offset 0 must be an integer or index scalar"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(type=pl.FunctionType.InCore)
            def worker(self, signal: pld.DistributedTensor[[4], pl.INT32]):
                pld.system.defer_wait(
                    signal,
                    offsets=[0.5],  # pyright: ignore[reportArgumentType]  # invalid input under test
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )


def test_defer_wait_rejects_float_expected():
    """The parser rejects a non-integer dynamic threshold before lowering."""
    with pytest.raises(InvalidOperationError, match="expected must be an integer or index scalar"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(type=pl.FunctionType.InCore)
            def worker(
                self,
                signal: pld.DistributedTensor[[4], pl.INT32],
                expected: pl.Scalar[pl.FP32],
            ):
                pld.system.defer_wait(signal, offsets=[0], expected=expected, cmp=pld.WaitCmp.Ge)


def test_defer_wait_rejects_eq_comparison():
    """The parser surfaces the verifier's monotonic-comparison contract."""
    with pytest.raises(InvalidOperationError, match=r"only supports WaitCmp\.Ge"):

        @pl.program
        class P:  # noqa: F841
            @pl.function(type=pl.FunctionType.InCore)
            def worker(self, signal: pld.DistributedTensor[[4], pl.INT32]):
                pld.system.defer_wait(signal, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)


def test_rank_inline_nested_get_comm_ctx():
    """``pld.rank(pld.get_comm_ctx(data))`` parses to the nested Call form."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            return pld.rank(pld.get_comm_ctx(data))

    func = _get_func(P, "worker")
    assert len(_find_calls_in_func(func, "pld.system.rank")) == 1
    assert len(_find_calls_in_func(func, "pld.system.get_comm_ctx")) == 1


def test_rank_and_nranks_compose_in_expression():
    """rank + nranks composes through arithmetic; both Calls survive in IR."""

    @pl.program
    class P:
        @pl.function
        def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
            ctx = pld.get_comm_ctx(data)
            return pld.rank(ctx) + pld.nranks(ctx)

    func = _get_func(P, "worker")
    assert len(_find_calls_in_func(func, "pld.system.rank")) == 1
    assert len(_find_calls_in_func(func, "pld.system.nranks")) == 1


def test_get_comm_ctx_rejects_plain_tensor():
    """The C++ verifier refuses a plain ``pl.Tensor`` — precise ObjectKind match."""
    with pytest.raises(InvalidOperationError, match="DistributedTensor"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def worker(self, x: pl.Tensor[[64], pl.FP32]):
                return pld.get_comm_ctx(x)  # type: ignore[arg-type]


def test_rank_rejects_non_comm_ctx_arg():
    """The C++ verifier refuses any non-CommCtx argument to pld.system.rank."""
    with pytest.raises(InvalidOperationError, match="CommCtx"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
                return pld.rank(data)  # type: ignore[arg-type]


def test_unknown_system_op_rejected():
    """Unknown 3-segment ``pld.system.<foo>`` produces a clear parser error."""
    with pytest.raises(InvalidOperationError, match=r"pld\.system\.foo"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def worker(self, data: pld.DistributedTensor[[64], pl.FP32]):
                ctx = pld.system.get_comm_ctx(data)
                return pld.system.foo(ctx)  # type: ignore[attr-defined]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
