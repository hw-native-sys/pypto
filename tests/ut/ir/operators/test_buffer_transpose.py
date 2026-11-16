# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Transpose scratch is a write-only live allocation, never an implicit temporary."""

import pytest
from pypto import DataType, ir, passes
from pypto.pypto_core import ir as _ir

_SPAN = ir.Span.unknown()


def _buffer(name, shape=(16, 32), **kwargs):
    return ir.Var(name, ir.BufferType(list(shape), DataType.FP32, ir.MemorySpace.Vec, **kwargs), _SPAN)


def _call(arguments):
    return _ir._create_internal_op_call("buffer.transpose", arguments, {}, _SPAN)


def _alloc(value, address):
    return ir.AssignStmt(
        value,
        ir.Call(
            ir.get_op("buffer.alloc"),
            [ir.MakeTuple([], _SPAN), ir.ConstInt(address, DataType.INDEX, _SPAN)],
            {},
            None,
            value.type,
            _SPAN,
        ),
        _SPAN,
    )


def _verify(statements):
    program = ir.Program(
        [
            ir.Function(
                "kernel",
                [],
                [],
                ir.SeqStmts(statements, _SPAN),
                _SPAN,
                type=ir.FunctionType.InCore,
                ir_stage=ir.FunctionIRStage.Buffer,
            )
        ],
        "Transpose",
        _SPAN,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    return passes.PropertyVerifierRegistry.verify(properties, program)


def test_transpose_scratch_effect_survives_serialization():
    call = _call([_buffer("src"), _buffer("tmp"), _buffer("dst", (32, 16))])
    assert ir.op_arg_is_workspace(call.op.name, 1)
    assert ir.get_op_buffer_arg_effect(call.op.name, 1).data == ir.BufferAccess.Write
    assert ir.get_op_buffer_arg_effect(call.op.name, 2).data == ir.BufferAccess.Write
    assert ir.get_op_buffer_arg_effect(call.op.name, 0).data == ir.BufferAccess.Read
    statement = ir.EvalStmt(call, _SPAN)
    ir.assert_structural_equal(statement, ir.deserialize(ir.serialize(statement)), enable_auto_mapping=True)


@pytest.mark.parametrize("addresses", [(0, 0, 4096), (0, 2048, 0), (0, 2048, 2048)])
def test_transpose_rejects_each_overlapping_live_pair(addresses):
    arguments = [_buffer("src"), _buffer("tmp"), _buffer("dst", (32, 16))]
    diagnostics = _verify(
        [
            *[_alloc(arg, addr) for arg, addr in zip(arguments, addresses)],
            ir.EvalStmt(_call(arguments), _SPAN),
        ]
    )
    assert diagnostics and any("disjoint" in item.message for item in diagnostics)


@pytest.mark.parametrize(
    "index,shape,valid,message",
    [
        (1, (8, 32), None, "scratch matching"),
        (1, (32, 32), None, "scratch matching"),
        (2, (16, 32), None, "swap the source"),
        (0, (16, 32), [15, 32], "full-valid"),
        (1, (16, 32), [-1, -1], "full-valid"),
    ],
)
def test_transpose_rejects_descriptor_mismatches(index, shape, valid, message):
    arguments = [_buffer("src"), _buffer("tmp"), _buffer("dst", (32, 16))]
    arguments[index] = _buffer("changed", shape, **({"valid_shape": valid} if valid else {}))
    with pytest.raises(ValueError, match=message):
        _call(arguments)


def test_transpose_rejects_a_strided_source_window():
    parent = _buffer("parent", (16, 64))
    source, scratch, destination = _buffer("window"), _buffer("tmp"), _buffer("dst", (32, 16))
    offsets = ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, _SPAN)] * 2, _SPAN)
    window = ir.AssignStmt(
        source, ir.Call(ir.get_op("buffer.subview"), [parent, offsets], {}, None, source.type, _SPAN), _SPAN
    )
    diagnostics = _verify(
        [
            _alloc(parent, 0),
            _alloc(scratch, 4096),
            _alloc(destination, 6144),
            window,
            ir.EvalStmt(_call([source, scratch, destination]), _SPAN),
        ]
    )
    assert diagnostics and any("dense physical row pitch" in item.message for item in diagnostics)
