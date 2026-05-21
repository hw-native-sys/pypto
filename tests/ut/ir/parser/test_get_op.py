# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.tensor.get`` — the synchronous cross-rank bulk read (TGET).

``get`` is a tensor-level op: both ``dst`` and ``src`` are window-bound
:class:`pld.DistributedTensor` (GM) views and the VEC staging tile is
synthesised at codegen, so it lives in the ``pld.tensor`` namespace next to
``alloc_window_buffer`` / ``window`` rather than the tile-producing
``pld.tile.remote_load``. The op is called explicitly (3-segment form only —
no unified short form). Verifier-level negatives (plain ``pl.Tensor`` into
``dst`` / ``src``, dtype / shape mismatch) come from the C++ op definition in
:file:`src/ir/op/distributed/get.cpp` and are exercised in
:file:`tests/ut/ir/test_distributed_ops.py`.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _iter_stmts(stmt: ir.Stmt):
    """Yield ``stmt`` and every nested statement (flattening ``SeqStmts``)."""
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            yield from _iter_stmts(s)


def test_get_parses_to_side_effect_op():
    """``pld.tensor.get`` parses to the registered side-effect-only op."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.get(dst, peer=peer, src=src)

    func = _get_func(P, "kernel")
    # get is side-effect-only — it parses to a bare EvalStmt wrapping the Call.
    get_calls = [
        stmt.expr
        for stmt in _iter_stmts(func.body)
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.op.name == "pld.tensor.get"
    ]

    assert len(get_calls) == 1
    call = get_calls[0]
    assert isinstance(call.type, ir.UnknownType)
    # Positional args: (dst, peer, src).
    assert len(call.args) == 3
    assert isinstance(call.args[0].type, ir.DistributedTensorType)
    assert isinstance(call.args[2].type, ir.DistributedTensorType)
    assert call.kwargs == {}


def test_get_round_trips_through_printer_and_parser():
    """Printed ``pld.tensor.get`` IR re-parses to a structurally-equal program."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[8], pl.FP32],
            src: pld.DistributedTensor[[8], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.get(dst, peer=peer, src=src)

    reparsed = pl.parse_program(str(P))

    ir.assert_structural_equal(P, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
