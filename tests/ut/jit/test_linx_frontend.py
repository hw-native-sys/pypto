# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Linx frontend acceptance through the public JIT lowering entry point."""

from pathlib import Path
from typing import cast

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.language.diagnostics import ParserError
from pypto.pypto_core import ir as core_ir
from pypto.runtime import RunConfig


def _scope_statements(program: ir.Program, entry: str):
    fn = program.get_function(entry)
    assert fn is not None and isinstance(fn.body, ir.SeqStmts)
    (scope,) = (stmt for stmt in fn.body.stmts if isinstance(stmt, ir.ScopeStmt))
    assert isinstance(scope.body, ir.SeqStmts)
    return scope.body.stmts


def _check_assignment(stmt: ir.Stmt, op: str, space: ir.MemorySpace) -> ir.AssignStmt:
    assert isinstance(stmt, ir.AssignStmt)
    assert isinstance(stmt.value, ir.Call) and stmt.value.op.name == core_ir.get_op(op).name
    assert isinstance(stmt.var.type, ir.TileType) and stmt.var.type.memory_space == space
    return stmt


def _check_store(stmt: ir.Stmt, value: ir.Var):
    assert isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    assert stmt.expr.op.name == core_ir.get_op("tile.store").name
    assert stmt.expr.args[0].same_as(value)


def test_local_add():
    @pl.jit
    def add(
        a: pl.Tensor[[8, 32], pl.FP32],
        b: pl.Tensor[[8, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            lhs = pl.load(a, [0, 0], [8, 32], target_memory=pl.MemorySpace.Local)
            rhs = pl.load(b, [0, 0], [8, 32], target_memory=pl.MemorySpace.Local)
            total = pl.add(lhs, rhs)
            pl.store(total, [0, 0], out)

    program = add.lower(config=RunConfig(backend_type=BackendType.Linx))
    lhs, rhs, total, store = _scope_statements(program, "add")
    local = ir.MemorySpace.Local
    lhs = _check_assignment(lhs, "tile.load", local)
    rhs = _check_assignment(rhs, "tile.load", local)
    total = _check_assignment(total, "tile.add", local)
    assert isinstance(total.value, ir.Call)
    assert total.value.args[0].same_as(lhs.var) and total.value.args[1].same_as(rhs.var)
    _check_store(store, total.var)


def test_shared_roundtrip():
    @pl.jit
    def exchange(a: pl.Tensor[[8, 32], pl.FP32], out: pl.Out[pl.Tensor[[8, 32], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            value = pl.load(a, [0, 0], [8, 32], valid_shape=[7, 23], target_memory=pl.MemorySpace.Local)
            published = pl.tile.publish_shared(value)
            local = pl.tile.broadcast_shared(published)
            pl.store(local, [0, 0], out)

    program = exchange.lower(config=RunConfig(backend_type=BackendType.Linx))
    load, publish, broadcast, store = _scope_statements(program, "exchange")
    load = _check_assignment(load, "tile.load", ir.MemorySpace.Local)
    publish = _check_assignment(publish, "tile.publish_shared", ir.MemorySpace.Shared)
    broadcast = _check_assignment(broadcast, "tile.broadcast_shared", ir.MemorySpace.Local)
    assert isinstance(publish.value, ir.Call) and publish.value.args[0].same_as(load.var)
    assert isinstance(broadcast.value, ir.Call) and broadcast.value.args[0].same_as(publish.var)
    for stmt in (load, publish, broadcast):
        tile = stmt.var.type
        assert isinstance(tile, ir.TileType) and tile.tile_view is not None
        assert [cast(ir.ConstInt, dim).value for dim in tile.shape] == [8, 32]
        assert [cast(ir.ConstInt, dim).value for dim in tile.tile_view.valid_shape] == [7, 23]
    _check_store(store, broadcast.var)


def test_shared_add_reports_user_source():
    @pl.jit
    def invalid(a: pl.Tensor[[8, 32], pl.FP32], out: pl.Out[pl.Tensor[[8, 32], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            local = pl.load(a, [0, 0], [8, 32], target_memory=pl.MemorySpace.Local)
            shared = pl.tile.publish_shared(local)
            result = pl.add(shared, local)
            pl.store(result, [0, 0], out)

    with pytest.raises(ParserError, match="two Local tile operands") as caught:
        invalid.lower(config=RunConfig(backend_type=BackendType.Linx))
    span = caught.value.span
    assert span is not None
    filename, line = span["filename"], span["line"]
    assert isinstance(filename, str) and Path(filename).resolve() == Path(__file__).resolve()
    assert isinstance(line, int) and line > 0
    assert Path(filename).read_text().splitlines()[line - 1].strip() == "result = pl.add(shared, local)"
