# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Linx tile types, operand roles, and Shared publication at the IR boundary."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir
from pypto.ir.op import tile
from pypto.pypto_core import passes


def test_single_pe_add_lowers_with_local_result():
    @pl.program
    class SinglePE:
        @pl.function
        def main(
            a: pl.Tensor[[8, 32], pl.FP32],
            b: pl.Tensor[[8, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
        ):
            lhs = pl.tile.load(a, [0, 0], [8, 32], target_memory=pl.MemorySpace.Local)
            rhs = pl.tile.load(b, [0, 0], [8, 32], target_memory=pl.MemorySpace.Local)
            result = pl.tile.add(lhs, rhs)
            pl.tile.store(result, [0, 0], out)

    lowered = ir.lower(SinglePE, backend_type=backend.BackendType.Linx)
    fn = lowered.get_function("main")
    assert fn is not None and isinstance(fn.body, ir.SeqStmts)
    result = fn.body.stmts[2]
    assert isinstance(result, ir.AssignStmt) and isinstance(result.var.type, ir.TileType)
    assert result.var.type.memory_space == ir.MemorySpace.Local


def test_published_tile_roundtrips_with_def_use_and_valid_region():
    @pl.program
    class Exchange:
        @pl.function
        def main(a: pl.Tensor[[8, 32], pl.FP32], out: pl.Out[pl.Tensor[[8, 32], pl.FP32]]):
            lhs = pl.tile.load(a, [0, 0], [8, 32], [7, 23], target_memory=pl.MemorySpace.Local)
            total = pl.tile.add(lhs, lhs)
            published = pl.tile.publish_shared(total)
            restored = pl.tile.broadcast_shared(published)
            pl.tile.store(restored, [0, 0], out)

    for program in (Exchange, ir.deserialize(ir.serialize(Exchange)), pl.parse(ir.python_print(Exchange))):
        assert isinstance(program, ir.Program)
        ir.assert_structural_equal(Exchange, program, enable_auto_mapping=True)
        fn = program.get_function("main")
        assert fn is not None and isinstance(fn.body, ir.SeqStmts)
        load, add, publish, consume = (cast(ir.AssignStmt, stmt) for stmt in fn.body.stmts[:4])
        local, shared = ir.MemorySpace.Local, ir.MemorySpace.Shared
        for stmt, space in zip((load, add, publish, consume), (local, local, shared, local), strict=True):
            typ = stmt.var.type
            assert isinstance(typ, ir.TileType) and typ.memory_space == space
            assert typ.dtype == DataType.FP32
            assert [cast(ir.ConstInt, dim).value for dim in typ.shape] == [8, 32]
            assert typ.tile_view is not None
            assert [cast(ir.ConstInt, dim).value for dim in typ.tile_view.valid_shape] == [7, 23]
        assert isinstance(publish.value, ir.Call) and publish.value.args[0].same_as(add.var)
        assert isinstance(consume.value, ir.Call) and consume.value.args[0].same_as(publish.var)


def test_roles_belong_to_uses():
    role = ir.TileOperandRole
    assert ir.get_tile_operand_role("tile.add", 0) == role.Vec
    assert ir.get_tile_operand_role("tile.add", 1) == role.Vec
    assert ir.get_tile_operand_role("tile.matmul", 0) == role.Left
    assert ir.get_tile_operand_role("tile.matmul", 1) == role.Right
    assert ir.get_tile_operand_role("tile.matmul_acc", 0) == role.Acc
    assert ir.get_tile_operand_role("tile.publish_shared", 0) is None


def _var(space, *, shape=(8, 32), dtype=DataType.FP32, view=None):
    return ir.Var("x", ir.TileType(list(shape), dtype, tile_view=view, memory_space=space), ir.Span.unknown())


def test_shared_transfer_directions_and_compute_inputs():
    local = _var(ir.MemorySpace.Local)
    shared = tile.publish_shared(local)
    with pytest.raises(ValueError):
        tile.publish_shared(shared)
    with pytest.raises(ValueError):
        tile.broadcast_shared(local)
    with pytest.raises(ValueError, match="Local"):
        tile.add(local, shared)
    with pytest.raises(ValueError, match="Local"):
        tile.add(_var(ir.MemorySpace.Vec), local)
    with pytest.raises(ValueError, match="publish"):
        tile.move(local, ir.MemorySpace.Shared)
    with pytest.raises(ValueError, match="publish"):
        tile.create([8, 32], DataType.FP32, target_memory=ir.MemorySpace.Shared)
    created = tile.create([8, 32], DataType.FP32, target_memory=ir.MemorySpace.Local).type
    assert isinstance(created, ir.TileType) and created.memory_space == ir.MemorySpace.Local


@pytest.mark.parametrize(
    ("shape", "dtype", "view"),
    [
        ((8, 32), DataType.INT32, None),
        ((1, 32), DataType.FP32, None),
        ((8, 32), DataType.FP32, ir.TileView(valid_shape=[7, 32])),
        ((8, 32), DataType.FP32, ir.TileView(blayout=ir.TileLayout.col_major)),
    ],
)
def test_local_add_checks_descriptor(shape, dtype, view):
    space = ir.MemorySpace.Local
    with pytest.raises(ValueError, match="matching"):
        tile.add(_var(space), _var(space, shape=shape, dtype=dtype, view=view))


def test_dynamic_valid_region_survives_publication():
    rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    col = ir.ConstInt(32, DataType.INDEX, ir.Span.unknown())
    value = _var(ir.MemorySpace.Local, view=ir.TileView(valid_shape=[rows, col]))
    result = tile.broadcast_shared(tile.publish_shared(value))
    typ = result.type
    assert isinstance(typ, ir.TileType) and typ.tile_view is not None
    assert typ.tile_view.valid_shape[0].same_as(rows)
    invalid = _var(ir.MemorySpace.Local, view=ir.TileView(valid_shape=[9, 32]))
    span = ir.Span.unknown()
    fn = ir.Function("invalid", [invalid], [], ir.ReturnStmt([], span), span)
    assert any("physical shape" in d.message for d in _type_diagnostics(fn))


def _type_diagnostics(fn: ir.Function) -> list[passes.Diagnostic]:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.TypeChecked)
    return passes.PropertyVerifierRegistry.verify(props, ir.Program([fn], "test", ir.Span.unknown()))


@pytest.mark.parametrize("inline", [True, False])
def test_shared_handles_in_function_signatures(inline):
    span = ir.Span.unknown()
    shared = _var(ir.MemorySpace.Shared)
    signature = ir.TupleType([shared.type])
    param = ir.Var("packet", signature, span)
    fn = ir.Function(
        "exchange",
        [param],
        [signature],
        ir.ReturnStmt([param], span),
        span,
        type=ir.FunctionType.Inline if inline else ir.FunctionType.InCore,
    )
    messages = [d.message for d in _type_diagnostics(fn)]
    if inline:
        assert not messages
    else:
        assert len(messages) == 2
        assert all("Shared handle across a non-inline function boundary" in message for message in messages)


def test_shared_value_cannot_be_disguised_as_a_local_return_or_inline_argument():
    span = ir.Span.unknown()
    local = _var(ir.MemorySpace.Local)
    shared = tile.publish_shared(local)
    callee = ir.Function(
        "identity", [local], [local.type], ir.ReturnStmt([local], span), span, type=ir.FunctionType.Inline
    )
    call = ir.Call(ir.GlobalVar("identity"), [shared], local.type, span)
    body = ir.SeqStmts([ir.EvalStmt(call, span), ir.ReturnStmt([shared], span)], span)
    caller = ir.Function("entry", [local], [local.type], body, span)
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.TypeChecked)
    messages = [
        d.message
        for d in passes.PropertyVerifierRegistry.verify(props, ir.Program([callee, caller], "test", span))
    ]
    assert any("argument 0 of call to 'identity'" in message for message in messages)
    assert any("return value of 'entry'" in message for message in messages)
