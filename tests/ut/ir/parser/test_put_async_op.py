# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.tensor.put_async`` / ``get_async``.

The async URMA ops return ``dst`` (the result aliases argument 0) and carry a
compile-time ``engine`` kwarg. The printer restores the enum spelling
``pld.CommEngine.URMA`` so a printed program re-parses to the same IR without
losing the engine or the destination alias.
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


def _find_async_call(program: ir.Program, op_name: str) -> ir.Call:
    func = _get_func(program, "kernel")
    matches = [
        stmt.value
        for stmt in _iter_stmts(func.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op(op_name).name
    ]
    assert len(matches) == 1, f"expected exactly one {op_name} call, got {len(matches)}"
    return matches[0]


def test_pld_exports_comm_engine():
    """The contract spelling ``pld.CommEngine.URMA`` resolves to the IR enum."""
    assert pld.CommEngine.URMA == ir.CommEngine.URMA
    assert {"put_async", "get_async"} <= set(pld.tensor.__all__)


def test_put_async_round_trips_engine_and_alias():
    """Printed ``put_async`` re-parses with its engine and dst alias intact."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            _out = pld.tensor.put_async(dst, peer=peer, src=src, engine=pld.CommEngine.URMA)

    call = _find_async_call(P, "pld.tensor.put_async")
    assert call.kwargs["engine"] == int(ir.CommEngine.URMA)
    # The result aliases dst: its type is the destination's type.
    func = _get_func(P, "kernel")
    ir.assert_structural_equal(call.type, func.params[0].type)

    printed = str(P)
    assert "pld.tensor.put_async" in printed
    assert "pld.CommEngine.URMA" in printed

    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(P, reparsed)


def test_get_async_round_trips_engine():
    """Printed ``get_async`` re-parses with its engine and dst alias intact."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            _out = pld.tensor.get_async(dst, peer=peer, src=src, engine=pld.CommEngine.URMA)

    call = _find_async_call(P, "pld.tensor.get_async")
    assert call.kwargs["engine"] == int(ir.CommEngine.URMA)
    func = _get_func(P, "kernel")
    ir.assert_structural_equal(call.type, func.params[0].type)

    printed = str(P)
    assert "pld.tensor.get_async" in printed
    assert "pld.CommEngine.URMA" in printed

    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(P, reparsed)


def _subregion_program(op_name: str):
    """Build a one-kernel program using the requested async op with a subregion."""

    if op_name == "put_async":

        @pl.program
        class PutProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                dst: pld.DistributedTensor[[16, 64], pl.FP16],
                src: pld.DistributedTensor[[16, 64], pl.FP16],
                peer: pl.Scalar[pl.INT32],
            ):
                _out = pld.tensor.put_async(
                    dst,
                    peer=peer,
                    src=src,
                    dst_offsets=[0, 0],
                    src_offsets=[0, 0],
                    shape=[1, 32],
                    engine=pld.CommEngine.URMA,
                )

        return PutProgram

    @pl.program
    class GetProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            _out = pld.tensor.get_async(
                dst,
                peer=peer,
                src=src,
                dst_offsets=[0, 0],
                src_offsets=[0, 0],
                shape=[1, 32],
                engine=pld.CommEngine.URMA,
            )

    return GetProgram


@pytest.mark.parametrize("op_name", ["put_async", "get_async"])
def test_async_subregion_round_trips(op_name):
    """The subregion form (offsets + shape) prints and reparses losslessly."""
    program = _subregion_program(op_name)
    call = _find_async_call(program, f"pld.tensor.{op_name}")
    assert call.kwargs["engine"] == int(ir.CommEngine.URMA)
    printed = str(program)
    assert "pld.CommEngine.URMA" in printed
    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(program, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
