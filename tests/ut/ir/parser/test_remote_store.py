# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.tile.remote_store``.

``pld.tile.remote_store(src_tile, target=..., peer=..., offsets=[...])`` is the
cross-rank tile store: it writes a local tile into a sub-region of the peer
rank's window-bound distributed tensor. The op is side-effect-only (no SSA
result for downstream consumers).

The parser dispatches via the generic 3-segment ``pld.<category>.<op>`` path
(``ast_parser.py:_parse_pld_category_op``); these tests cover both the
positive DSL→IR lifting and the parser-level rejection of malformed call
sites.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.language.parser.diagnostics import InvalidOperationError
from pypto.pypto_core import ir


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _find_call(func: ir.Function, op_name: str) -> ir.Call:
    """Return the first ``op_name`` call found in ``func``'s body."""
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
        if isinstance(stmt, ir.EvalStmt):
            visit_expr(stmt.expr)
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)

    walk(func.body)
    assert found, f"no {op_name} call found in function body"
    return found[0]


# ---------------------------------------------------------------------------
# Positive: DSL lifts to ir.Call('pld.tile.remote_store', ...)
# ---------------------------------------------------------------------------


def test_remote_store_lifts_to_op_call_with_unknown_return_type():
    """``pld.tile.remote_store`` parses to an IR call whose return type is
    UnknownType (side-effect only)."""

    @pl.program
    class P:
        @pl.function
        def kernel(
            self,
            data: pld.DistributedTensor[[64, 32], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[32, 16])
            pld.tile.remote_store(tile, target=data, peer=peer, offsets=[0, 0])

    func = _get_func(P, "kernel")
    call = _find_call(func, "pld.tile.remote_store")
    assert isinstance(call.type, ir.UnknownType)


def test_remote_store_threads_args_positionally():
    """The parser packs ``[src_tile, target, peer, offsets]`` as positional args
    on the IR call (kwargs collapsed)."""

    @pl.program
    class P:
        @pl.function
        def kernel(
            self,
            data: pld.DistributedTensor[[64, 32], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 8])
            pld.tile.remote_store(tile, target=data, peer=peer, offsets=[0, 0])

    func = _get_func(P, "kernel")
    call = _find_call(func, "pld.tile.remote_store")
    assert call.kwargs == {}
    assert len(call.args) == 4
    src_arg, target_arg, peer_arg, offsets_arg = call.args
    assert isinstance(src_arg.type, ir.TileType)
    assert isinstance(target_arg, ir.Var)
    assert isinstance(target_arg.type, ir.DistributedTensorType)
    assert target_arg.name_hint == "data"
    assert isinstance(peer_arg, ir.Var)
    assert isinstance(peer_arg.type, ir.ScalarType)
    assert isinstance(offsets_arg, ir.MakeTuple)


def test_remote_store_round_trips_through_printer():
    """Printed IR round-trips through the parser: print → parse → print yields
    the same text for the kernel containing ``remote_store``."""

    @pl.program
    class Before:
        @pl.function
        def kernel(
            self,
            data: pld.DistributedTensor[[64, 32], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[32, 16])
            pld.tile.remote_store(tile, target=data, peer=peer, offsets=[0, 0])

    text1 = ir.python_print(Before)
    After = pl.parse_program(text1)
    text2 = ir.python_print(After)
    assert text1 == text2, f"round-trip mismatch:\n--- before:\n{text1}\n--- after:\n{text2}"
    # The call still lifts to the same op name post-parse.
    after_func = _get_func(After, "kernel")
    after_call = _find_call(after_func, "pld.tile.remote_store")
    assert after_call.op.name == ir.get_op("pld.tile.remote_store").name


# ---------------------------------------------------------------------------
# Negative: shape / target / offsets mistakes surface at parse or verify time
# ---------------------------------------------------------------------------


def test_remote_store_rejects_plain_tensor_target():
    """The parser refuses a ``pl.Tensor`` target — must be window-bound."""
    with pytest.raises(InvalidOperationError, match="DistributedTensor"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pl.Tensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ):
                tile = pl.tile.load(data, [0], [32])
                pld.tile.remote_store(tile, target=data, peer=peer, offsets=[0])  # type: ignore[arg-type]


def test_remote_store_rejects_unknown_subop():
    """``pld.tile.<other>`` is rejected at 3-segment dispatch."""
    with pytest.raises(InvalidOperationError, match=r"pld\.tile"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ):
                tile = pld.tile.remote_load(data, peer=peer, offsets=[0], shape=[32])
                pld.tile.no_such_op(tile, target=data, peer=peer, offsets=[0])  # type: ignore[attr-defined]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
