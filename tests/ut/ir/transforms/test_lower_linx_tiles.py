# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Linx placement after shared GM-to-tile conversion."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.pypto_core import ir as core_ir


def test_tensor_add_writes_and_returns_original_out():
    @pl.program
    class Add:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[8, 32], pl.FP32],
            b: pl.Tensor[[8, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
        ) -> pl.Tensor[[8, 32], pl.FP32]:
            total = pl.add(a, b)
            out = pl.assemble(out, total, [0, 0])
            return out

    kernel = ir.lower(Add, backend_type=BackendType.Linx).get_function("main")
    assert kernel is not None and isinstance(kernel.body, ir.SeqStmts)
    a, b, out = kernel.params
    lhs, rhs, total, store, ret = kernel.body.stmts
    for stmt, param in ((lhs, a), (rhs, b)):
        assert isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
        assert stmt.value.op.name == core_ir.get_op("tile.load").name
        assert stmt.value.args[0].same_as(param)
        assert isinstance(stmt.var.type, ir.TileType) and stmt.var.type.memory_space == ir.MemorySpace.Local
    assert isinstance(total, ir.AssignStmt) and isinstance(total.value, ir.Call)
    assert total.value.op.name == core_ir.get_op("tile.add").name
    assert isinstance(total.var.type, ir.TileType) and total.var.type.memory_space == ir.MemorySpace.Local
    assert isinstance(lhs, ir.AssignStmt) and total.value.args[0].same_as(lhs.var)
    assert isinstance(rhs, ir.AssignStmt) and total.value.args[1].same_as(rhs.var)
    assert isinstance(store, ir.AssignStmt) and isinstance(store.value, ir.Call)
    assert store.value.op.name == core_ir.get_op("tile.store").name
    assert store.value.args[0].same_as(total.var) and store.value.args[2].same_as(out)
    assert isinstance(ret, ir.ReturnStmt) and ret.value[0].same_as(store.var)


def test_conditional_result_is_local_at_store():
    @pl.program
    class Conditional:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[8, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
            choose_sum: pl.Scalar[pl.BOOL],
        ):
            value = pl.load(a, [0, 0], [8, 32])
            if choose_sum:
                value = pl.add(value, value)
            pl.store(value, [0, 0], out)

    kernel = ir.lower(Conditional, backend_type=BackendType.Linx).get_function("main")
    assert kernel is not None and isinstance(kernel.body, ir.SeqStmts)
    (branch,) = (stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.IfStmt))
    (result,) = branch.return_vars
    assert isinstance(result.type, ir.TileType) and result.type.memory_space == ir.MemorySpace.Local
    (store_stmt,) = (stmt for stmt in kernel.body.stmts if isinstance(stmt, ir.EvalStmt))
    assert isinstance(store_stmt, ir.EvalStmt) and isinstance(store_stmt.expr, ir.Call)
    store = store_stmt.expr
    assert store.op.name == core_ir.get_op("tile.store").name
    assert store.args[0].same_as(result)


def test_explicit_ascend_placement_is_diagnosed():
    @pl.program
    class ExplicitVec:
        @pl.function
        def main(self, a: pl.Tensor[[8, 32], pl.FP32], out: pl.Out[pl.Tensor[[8, 32], pl.FP32]]):
            tile = pl.load(a, [0, 0], [8, 32], target_memory=pl.Mem.Vec)
            pl.store(tile, [0, 0], out)

    with pytest.raises(ValueError, match="Linx tile.load requires Local"):
        ir.lower(ExplicitVec, backend_type=BackendType.Linx)


def test_direct_shared_store_is_diagnosed():
    @pl.program
    class DirectStore:
        @pl.function
        def main(self, a: pl.Tensor[[8, 32], pl.FP32], out: pl.Out[pl.Tensor[[8, 32], pl.FP32]]):
            tile = pl.load(a, [0, 0], [8, 32], target_memory=pl.Mem.Local)
            shared = pl.tile.publish_shared(tile)
            pl.store(shared, [0, 0], out)

    with pytest.raises(ValueError, match="Linx tile.store requires Local"):
        ir.lower(DirectStore, backend_type=BackendType.Linx)
