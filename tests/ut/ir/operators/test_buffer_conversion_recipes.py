# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Heterogeneous conversion contracts, packed masks, and writable scratch."""

import pytest
from pypto import DataType, ir, passes
from pypto.backend import BackendType
from pypto.pypto_core import codegen
from pypto.pypto_core import ir as _ir

SPAN = ir.Span.unknown()
KINDS = ("cast", "cmp", "cmps", "sel", "sels")


def _buffer(name, shape=(8, 64), dtype=DataType.FP32, valid=None):
    return ir.Var(name, ir.BufferType(list(shape), dtype, ir.MemorySpace.Vec, list(valid or shape)), SPAN)


def _operands(kind):
    source = _buffer("source")
    scalar = ir.ConstFloat(0.0, DataType.FP32, SPAN)
    mask = _buffer("mask", (8, 32), DataType.UINT8, (8, 8))
    if kind == "cast":
        return [source, _buffer("dst", dtype=DataType.FP16)], {"mode": 1, "saturation_mode": 0}
    if kind == "cmp":
        return [source, _buffer("other"), mask], {"cmp_type": 4}
    if kind == "cmps":
        return [source, scalar, mask], {"cmp_type": 4}
    if kind == "sel":
        return [mask, source, _buffer("other"), _buffer("tmp", (1, 16), DataType.UINT32), _buffer("dst")], {}
    return [mask, source, _buffer("tmp", (1, 64)), scalar, _buffer("dst")], {}


def _call(kind, args, kwargs):
    return _ir._create_internal_op_call(f"buffer.{kind}", args, kwargs, SPAN)


@pytest.mark.parametrize("kind", KINDS)
def test_conversion_effects_and_serialization(kind):
    args, kwargs = _operands(kind)
    call = _call(kind, args, kwargs)
    scratch = {"sel": 3, "sels": 2}.get(kind)
    for index, arg in enumerate(args):
        effect = ir.get_op_buffer_arg_effect(call.op.name, index)
        if isinstance(arg.type, ir.BufferType):
            writes = index == len(args) - 1 or index == scratch
            assert effect.data == (ir.BufferAccess.Write if writes else ir.BufferAccess.Read)
            assert effect.metadata == ir.BufferAccess.Read
        else:
            assert effect.data == ir.BufferAccess.None_
        assert ir.op_arg_is_workspace(call.op.name, index) == (index == scratch)
    statement = ir.EvalStmt(call, SPAN)
    ir.assert_structural_equal(statement, ir.deserialize(ir.serialize(statement)), enable_auto_mapping=True)


@pytest.mark.parametrize("kind", KINDS)
def test_conversion_rejects_runtime_valid_state(kind):
    args, kwargs = _operands(kind)
    index = 1 if kind in ("sel", "sels") else 0
    args[index] = _buffer("dynamic", valid=(-1, -1))
    with pytest.raises(ValueError, match="positive static valid"):
        _call(kind, args, kwargs)


@pytest.mark.parametrize("kind", ["cmp", "cmps", "sel", "sels"])
@pytest.mark.parametrize("valid", [(8, 7), (7, 8)])
def test_predicate_geometry_must_match_values(kind, valid):
    args, kwargs = _operands(kind)
    index = -1 if kind in ("cmp", "cmps") else 0
    args[index] = _buffer("bad_mask", (8, 32), DataType.UINT8, valid)
    with pytest.raises(ValueError, match="packed UINT8 predicate"):
        _call(kind, args, kwargs)


@pytest.mark.parametrize("kind", ["sel", "sels"])
def test_selection_requires_explicit_correct_scratch(kind):
    args, kwargs = _operands(kind)
    args[3 if kind == "sel" else 2] = _buffer("wrong_tmp", (1, 8))
    with pytest.raises(ValueError, match="full-valid A2/A3 scratch"):
        _call(kind, args, kwargs)


@pytest.mark.parametrize("kwargs", [{"mode": 3}, {"mode": 1, "saturation_mode": 1}])
def test_cast_does_not_silently_drop_unsupported_modes(kwargs):
    args, _ = _operands("cast")
    with pytest.raises(ValueError, match="unsupported conversion attributes"):
        _call("cast", args, kwargs)


@pytest.mark.parametrize("kind", ["cmp", "cmps"])
def test_comparison_does_not_admit_integer_ordering(kind):
    args, kwargs = _operands(kind)
    args[0] = _buffer("integer", dtype=DataType.INT32)
    with pytest.raises(ValueError, match="FP32 inputs"):
        _call(kind, args, kwargs)


def _program(kind, addresses):
    args, kwargs = _operands(kind)
    statements = []
    for arg, address in zip(args, addresses):
        if not isinstance(arg.type, ir.BufferType):
            continue
        assert isinstance(arg, ir.Var)
        allocation = ir.Call(
            ir.get_op("buffer.alloc"),
            [ir.MakeTuple([], SPAN), ir.ConstInt(address, DataType.INDEX, SPAN)],
            {},
            None,
            arg.type,
            SPAN,
        )
        statements.append(ir.AssignStmt(arg, allocation, SPAN))
    statements.append(ir.EvalStmt(_call(kind, args, kwargs), SPAN))
    function = ir.Function(
        "kernel",
        [],
        [],
        ir.SeqStmts(statements, SPAN),
        SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return ir.Program([function], "Conversion", SPAN)


def _verify(kind, addresses):
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    return passes.PropertyVerifierRegistry.verify(properties, _program(kind, addresses))


@pytest.mark.parametrize(
    "kind,addresses",
    [
        ("cast", (0, 0)),
        ("cmp", (0, 4096, 0)),
        ("cmps", (0, None, 0)),
        ("sel", (0, 4096, 8192, 12288, 0)),
        ("sel", (0, 4096, 8192, 4096, 16384)),
        ("sels", (0, 4096, 8192, None, 8192)),
    ],
)
def test_conversion_rejects_destructive_storage_overlap(kind, addresses):
    diagnostics = _verify(kind, addresses)
    assert diagnostics and any("disjoint" in d.message for d in diagnostics)


@pytest.mark.parametrize(
    "kind,addresses",
    [
        ("sel", (0, 4096, 8192, 12288, 4096)),
        ("sels", (0, 4096, 8192, None, 4096)),
    ],
)
def test_selection_may_reuse_a_value_input(kind, addresses):
    assert _verify(kind, addresses) == []


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
def test_direct_conversion_requires_a2a3(ascend_backend):
    assert ascend_backend == BackendType.Ascend950
    with pytest.raises(ValueError, match="supports A2/A3 only"):
        codegen.PTOCodegen().generate(_program("cast", (0, 4096)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
