# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.tile.remote_load``.

``pld.tile.remote_load(target, peer=..., offsets=[...], shape=[...])`` is the
cross-rank tile load: it reads a sub-region from a peer rank's window-bound
distributed tensor and returns a local Tile of the requested shape and the
target's dtype.

The parser dispatches via the 3-segment ``pld.tile.<op>`` path
(``ast_parser.py:_parse_pld_tile_op``); these tests cover both the positive
DSL→IR lifting and the parser-level rejection of malformed call sites.
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
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
        if isinstance(stmt, ir.ForStmt):
            walk(stmt.body)

    walk(func.body)
    assert found, f"no {op_name} call found in function body"
    return found[0]


# ---------------------------------------------------------------------------
# Positive: DSL lifts to ir.Call('pld.tile.remote_load', ...)
# ---------------------------------------------------------------------------


def test_remote_load_lifts_to_op_call_with_tile_return_type():
    """``pld.tile.remote_load`` on a DistributedTensor param parses to an IR
    call whose return type is ``TileType(shape=[32], dtype=target.dtype)``."""

    @pl.program
    class P:
        @pl.function
        def kernel(
            self,
            data: pld.DistributedTensor[[64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[64], pl.FP16]:
            t = pld.tile.remote_load(data, peer=peer, offsets=[0], shape=[32])
            return t  # type: ignore[return-value]

    func = _get_func(P, "kernel")
    call = _find_call(func, "pld.tile.remote_load")
    assert isinstance(call.type, ir.TileType)
    assert call.type.dtype == pl.FP16
    assert len(call.type.shape) == 1
    assert isinstance(call.type.shape[0], ir.ConstInt)
    assert call.type.shape[0].value == 32


def test_remote_load_threads_target_and_peer_through_args():
    """The parser packs ``[target, peer, offsets, shape]`` as positional args
    on the IR call (kwargs collapsed)."""

    @pl.program
    class P:
        @pl.function
        def kernel(
            self,
            data: pld.DistributedTensor[[64], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[64], pl.FP32]:
            t = pld.tile.remote_load(data, peer=peer, offsets=[0], shape=[16])
            return t  # type: ignore[return-value]

    func = _get_func(P, "kernel")
    call = _find_call(func, "pld.tile.remote_load")
    assert call.kwargs == {}
    assert len(call.args) == 4
    target_arg, peer_arg, offsets_arg, shape_arg = call.args
    assert isinstance(target_arg, ir.Var)
    assert isinstance(target_arg.type, ir.DistributedTensorType)
    assert target_arg.name_hint == "data"
    assert isinstance(peer_arg, ir.Var)
    assert isinstance(peer_arg.type, ir.ScalarType)
    assert isinstance(offsets_arg, ir.MakeTuple)
    assert isinstance(shape_arg, ir.MakeTuple)


def test_remote_load_handles_multi_dim_shape():
    """2-D target → 2-D offsets/shape — ranks must match."""

    @pl.program
    class P:
        @pl.function
        def kernel(
            self,
            data: pld.DistributedTensor[[64, 32], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[64, 32], pl.FP16]:
            t = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 8])
            return t  # type: ignore[return-value]

    func = _get_func(P, "kernel")
    call = _find_call(func, "pld.tile.remote_load")
    assert isinstance(call.type, ir.TileType)
    assert len(call.type.shape) == 2
    assert [int(d.value) for d in call.type.shape] == [16, 8]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Negative: positional / kwarg shape mistakes
# ---------------------------------------------------------------------------


def test_remote_load_rejects_zero_positional():
    with pytest.raises(Exception, match="1 positional"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.remote_load(peer=peer, offsets=[0], shape=[32])  # type: ignore[call-arg]  # noqa: F841
                return data  # type: ignore[return-value]


def test_remote_load_rejects_extra_positional():
    with pytest.raises(Exception, match="1 positional"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.remote_load(data, peer, offsets=[0], shape=[32])  # type: ignore[call-arg]  # noqa: F841
                return data  # type: ignore[return-value]


def test_remote_load_rejects_missing_peer():
    with pytest.raises(Exception, match="missing required kwarg"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.remote_load(data, offsets=[0], shape=[32])  # type: ignore[call-arg]  # noqa: F841
                return data  # type: ignore[return-value]


def test_remote_load_rejects_missing_offsets():
    with pytest.raises(Exception, match="missing required kwarg"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.remote_load(data, peer=peer, shape=[32])  # type: ignore[call-arg]  # noqa: F841
                return data  # type: ignore[return-value]


def test_remote_load_rejects_missing_shape():
    with pytest.raises(Exception, match="missing required kwarg"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.remote_load(data, peer=peer, offsets=[0])  # type: ignore[call-arg]  # noqa: F841
                return data  # type: ignore[return-value]


def test_remote_load_rejects_unknown_kwarg():
    with pytest.raises(Exception, match="does not accept kwarg"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.remote_load(  # type: ignore[call-arg]  # noqa: F841
                    data, peer=peer, offsets=[0], shape=[32], foo=1
                )
                return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Negative: target / offsets type rejections at parse time
# ---------------------------------------------------------------------------


def test_remote_load_rejects_plain_tensor_target():
    """The parser refuses a ``pl.Tensor`` target — must be window-bound."""
    with pytest.raises(Exception, match="DistributedTensor"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pl.Tensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.remote_load(data, peer=peer, offsets=[0], shape=[32])  # type: ignore[arg-type]  # noqa: F841
                return data


def test_remote_load_rejects_non_list_offsets():
    """A scalar in place of ``offsets=[...]`` is rejected at parse time."""
    with pytest.raises(Exception, match="offsets"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.remote_load(data, peer=peer, offsets=0, shape=[32])  # type: ignore[arg-type]  # noqa: F841
                return data  # type: ignore[return-value]


def test_remote_load_rejects_unknown_subop():
    """``pld.tile.<other>`` is rejected at 3-segment dispatch."""
    with pytest.raises(Exception, match="pld.tile"):

        @pl.program
        class P:  # noqa: F841
            @pl.function
            def kernel(
                self,
                data: pld.DistributedTensor[[64], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pld.tile.no_such_op(data, peer=peer, offsets=[0], shape=[32])  # type: ignore[attr-defined]  # noqa: F841
                return data  # type: ignore[return-value]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
