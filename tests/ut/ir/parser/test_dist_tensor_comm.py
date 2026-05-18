# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for the ``DistributedTensor.comm.rank/.nranks`` desugar.

The parser lifts:

    dist_t.comm          → pld.get_comm_ctx(dist_t)
    ctx.rank   / .nranks → pld.comm_ctx.{rank,nranks}(ctx)

so user code reads as a plain Python attribute chain (``data.comm.rank``)
while the IR sees explicit op calls.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _find_call(func: ir.Function, op_name: str) -> ir.Call:
    found: list[ir.Call] = []

    def visit_expr(expr: ir.Expr | None) -> None:
        if expr is None or not isinstance(expr, ir.Call):
            return
        if expr.op.name == op_name:
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
            walk(stmt.body)
        if isinstance(stmt, ir.IfStmt):
            walk(stmt.then_body)
            if stmt.else_body is not None:
                walk(stmt.else_body)

    walk(func.body)
    assert found, f"no {op_name} call found in function body"
    return found[0]


# ---------------------------------------------------------------------------
# Positive: chained dist_t.comm.rank / .nranks
# ---------------------------------------------------------------------------


def test_dist_tensor_comm_rank_lifts_to_op_chain():
    @pl.program
    class P:
        @pl.function
        def kernel(self, data: pld.DistributedTensor[[64], pl.FP32]) -> pl.Scalar[pl.INT32]:
            r = data.comm.rank  # type: ignore[attr-defined]
            return r  # type: ignore[return-value]

    func = _get_func(P, "kernel")
    rank_call = _find_call(func, "pld.comm_ctx.rank")
    assert isinstance(rank_call.type, ir.ScalarType)
    assert len(rank_call.args) == 1
    ctx_arg = rank_call.args[0]
    assert isinstance(ctx_arg, ir.Call)
    assert ctx_arg.op.name == "pld.get_comm_ctx"
    assert isinstance(ctx_arg.type, ir.CommCtxType)
    assert isinstance(ctx_arg.args[0], ir.Var)
    assert ctx_arg.args[0].name_hint == "data"


def test_dist_tensor_comm_nranks_lifts_to_op_chain():
    @pl.program
    class P:
        @pl.function
        def kernel(self, data: pld.DistributedTensor[[64], pl.FP32]) -> pl.Scalar[pl.INT32]:
            n = data.comm.nranks  # type: ignore[attr-defined]
            return n  # type: ignore[return-value]

    func = _get_func(P, "kernel")
    nranks_call = _find_call(func, "pld.comm_ctx.nranks")
    assert isinstance(nranks_call.type, ir.ScalarType)
    inner = nranks_call.args[0]
    assert isinstance(inner, ir.Call) and inner.op.name == "pld.get_comm_ctx"


def test_dist_tensor_comm_can_be_used_in_expr():
    """The chain composes with arithmetic — verifies the result is a usable IR Expr."""

    @pl.program
    class P:
        @pl.function
        def kernel(self, data: pld.DistributedTensor[[64], pl.FP32]) -> pl.Scalar[pl.INT32]:
            doubled = data.comm.rank + data.comm.rank  # type: ignore[attr-defined]
            return doubled  # type: ignore[return-value]

    func = _get_func(P, "kernel")
    # Recursively count pld.comm_ctx.rank occurrences anywhere in the IR.
    found_rank = 0

    def walk_expr(e: ir.Expr | None) -> None:
        nonlocal found_rank
        if e is None:
            return
        if isinstance(e, ir.Call):
            if e.op.name == "pld.comm_ctx.rank":
                found_rank += 1
            for a in e.args:
                walk_expr(a)
        else:
            # Walk through children for BinOp etc. (Add has left/right).
            for attr in ("left", "right", "lhs", "rhs", "operand", "value"):
                child = getattr(e, attr, None)
                if isinstance(child, ir.Expr):
                    walk_expr(child)

    def walk_stmt(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt):
            walk_expr(stmt.value)
        if isinstance(stmt, ir.ReturnStmt):
            for v in getattr(stmt, "values", []):
                walk_expr(v)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk_stmt(s)

    walk_stmt(func.body)
    assert found_rank == 2, f"expected 2 pld.comm_ctx.rank calls, found {found_rank}"


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_plain_tensor_comm_attribute_rejected():
    with pytest.raises(Exception, match="DistributedTensor|attribute"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(self, data: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.INT32]:
                return data.comm.rank  # type: ignore[attr-defined,return-value]


def test_dist_tensor_unknown_attribute_rejected():
    with pytest.raises(Exception, match="DistributedTensor has no attribute"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(self, data: pld.DistributedTensor[[64], pl.FP32]) -> pl.Scalar[pl.INT32]:
                return data.zorch  # type: ignore[attr-defined,return-value]


def test_comm_ctx_unknown_attribute_rejected():
    with pytest.raises(Exception, match="CommCtx has no attribute"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(self, data: pld.DistributedTensor[[64], pl.FP32]) -> pl.Scalar[pl.INT32]:
                return data.comm.zoinks  # type: ignore[attr-defined,return-value]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
