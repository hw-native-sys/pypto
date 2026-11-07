# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Indexed Buffer contracts: explicit scratch, bounds and destructive overlap."""

import pytest
from pypto import DataType, ir, passes
from pypto.backend import BackendType
from pypto.pypto_core import codegen
from pypto.pypto_core import ir as _ir

SPAN = ir.Span.unknown()


def _buffer(name, shape=(4, 32), dtype=DataType.FP32, valid=None):
    return ir.Var(name, ir.BufferType(list(shape), dtype, ir.MemorySpace.Vec, list(valid or shape)), SPAN)


def _operands(kind):
    source = _buffer("source")
    indices = _buffer("indices", dtype=DataType.INT32)
    destination = _buffer("destination", (8, 64) if kind == "scatter" else (4, 32))
    if kind == "gather":
        return [source, indices, _buffer("scratch", dtype=DataType.INT32), destination]
    return [source, indices, destination]


def _call(kind, args):
    return _ir._create_internal_op_call(f"buffer.{kind}", args, {}, SPAN)


@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_indexed_effects_and_serialization(kind):
    args = _operands(kind)
    call = _call(kind, args)
    for index in range(len(args)):
        scratch = kind == "gather" and index == 2
        effect = ir.get_op_buffer_arg_effect(call.op.name, index)
        assert effect.data == (
            ir.BufferAccess.Write if scratch or index == len(args) - 1 else ir.BufferAccess.Read
        )
        assert effect.metadata == ir.BufferAccess.Read
        assert ir.op_arg_is_workspace(call.op.name, index) == scratch
    statement = ir.EvalStmt(call, SPAN)
    ir.assert_structural_equal(statement, ir.deserialize(ir.serialize(statement)), enable_auto_mapping=True)


@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("dtype", [DataType.FP32, DataType.FP16])
def test_indexed_requires_int32_indices(kind, dtype):
    args = _operands(kind)
    args[1] = _buffer("bad_index", dtype=dtype)
    with pytest.raises(ValueError, match="INT32 indices"):
        _call(kind, args)


@pytest.mark.parametrize("valid", [(3, 32), (4, 31), (-1, 32), (0, 32)])
def test_gather_rejects_undefined_index_tails(valid):
    args = _operands("gather")
    args[1] = _buffer("partial_index", dtype=DataType.INT32, valid=valid)
    with pytest.raises(ValueError, match="full-valid gather|positive static valid"):
        _call("gather", args)


@pytest.mark.parametrize("shape,dtype", [((4, 64), DataType.INT32), ((4, 32), DataType.FP32)])
def test_gather_scratch_matches_indices(shape, dtype):
    args = _operands("gather")
    args[2] = _buffer("bad_scratch", shape, dtype)
    with pytest.raises(ValueError, match="INT32 scratch"):
        _call("gather", args)


def test_scatter_rejects_a_partial_destination():
    args = _operands("scatter")
    args[-1] = _buffer("partial_destination", (8, 64), valid=(7, 61))
    with pytest.raises(ValueError, match="full-valid destination"):
        _call("scatter", args)


def _program(kind, addresses):
    args = _operands(kind)
    statements = []
    for arg, address in zip(args, addresses):
        allocation = ir.Call(
            ir.get_op("buffer.alloc"),
            [ir.MakeTuple([], SPAN), ir.ConstInt(address, DataType.INDEX, SPAN)],
            {},
            None,
            arg.type,
            SPAN,
        )
        statements.append(ir.AssignStmt(arg, allocation, SPAN))
    statements.append(ir.EvalStmt(_call(kind, args), SPAN))
    function = ir.Function(
        "kernel",
        [],
        [],
        ir.SeqStmts(statements, SPAN),
        SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return ir.Program([function], "Indexed", SPAN)


@pytest.mark.parametrize(
    "kind,addresses",
    [
        ("gather", (0, 4096, 8192, 0)),
        ("gather", (0, 4096, 8192, 4096)),
        ("gather", (0, 4096, 8192, 8192)),
        ("gather", (0, 4096, 0, 8192)),
        ("gather", (0, 4096, 4096, 8192)),
        ("scatter", (0, 4096, 0)),
        ("scatter", (0, 4096, 4096)),
    ],
)
def test_indexed_rejects_destructive_storage_overlap(kind, addresses):
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, _program(kind, addresses))
    assert diagnostics and any("disjoint" in d.message for d in diagnostics)


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_direct_indexed_operations_require_a2a3(ascend_backend, kind):
    assert ascend_backend == BackendType.Ascend950
    with pytest.raises(ValueError, match="supports A2/A3 only"):
        codegen.PTOCodegen().generate(_program(kind, (0, 4096, 8192, 12288)))


@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_indexed_operations_reject_inherited_strided_pitch(kind):
    args = _operands(kind)
    root = _buffer("root", (8, 64))
    offsets = ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, SPAN)] * 2, SPAN)
    allocations: list[ir.Stmt] = []
    for index, arg in enumerate([root, *args[1:]]):
        allocation = ir.Call(
            ir.get_op("buffer.alloc"),
            [ir.MakeTuple([], SPAN), ir.ConstInt(index * 4096, DataType.INDEX, SPAN)],
            {},
            None,
            arg.type,
            SPAN,
        )
        allocations.append(ir.AssignStmt(arg, allocation, SPAN))
    # The source descriptor is [4,32], but its physical row pitch is still 64.
    window = _ir._create_internal_op_call("buffer.subview", [root, offsets], {}, args[0].type, SPAN)
    statements: list[ir.Stmt] = [
        *allocations,
        ir.AssignStmt(args[0], window, SPAN),
        ir.EvalStmt(_call(kind, args), SPAN),
    ]
    function = ir.Function(
        "kernel",
        [],
        [],
        ir.SeqStmts(statements, SPAN),
        SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, ir.Program([function], "Strided", SPAN))
    assert any("dense physical row pitch" in diagnostic.message for diagnostic in diagnostics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
