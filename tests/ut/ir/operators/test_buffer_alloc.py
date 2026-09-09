# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Allocation keeps physical descriptors in types and runtime values in operands."""

from collections.abc import Sequence
from typing import Any

import pytest
from pypto import DataType, ir
from pypto.pypto_core import ir as _ir


def descriptor(**overrides: Any) -> ir.BufferType:
    options: dict[str, Any] = dict(shape=[16, 32], dtype=DataType.FP32, memory_space=ir.MemorySpace.Vec)
    return ir.BufferType(**(options | overrides))


def integer(value: int, dtype: DataType = DataType.INDEX) -> ir.ConstInt:
    return ir.ConstInt(value, dtype, ir.Span.unknown())


def scalar(name: str, dtype: DataType = DataType.INDEX) -> ir.Var:
    return ir.Var(name, ir.ScalarType(dtype), ir.Span.unknown())


def allocate(
    result_type: ir.Type,
    valid: Sequence[ir.Expr] = (),
    address: ir.Expr | None = None,
    **kwargs: Any,
) -> ir.Call:
    span = ir.Span.unknown()
    args: list[ir.Expr] = [ir.MakeTuple(valid, span)]
    if address is not None:
        args.append(address)
    return _ir._create_internal_op_call("buffer.alloc", args, kwargs, result_type, span)


def test_addressless_allocation_has_no_address_operand_or_descriptor_kwargs():
    type_ = descriptor()
    call = allocate(type_)
    assert isinstance(call.type, ir.BufferType)
    ir.assert_structural_equal(call.type, type_)
    assert call.kwargs == {}
    assert len(call.args) == 1
    assert isinstance(call.args[0], ir.MakeTuple)
    assert len(call.args[0].elements) == 0
    assert ir.get_op_ir_stage("buffer.alloc") == ir.OpIRStage.Buffer
    assert ir.get_op_output_arity("buffer.alloc") == 1
    assert ir.get_op_buffer_result_spec("buffer.alloc").behavior == ir.BufferResultBehavior.Allocate
    assert ir.get_op_buffer_result_spec("buffer.alloc").alias_arg is None
    for index in (0, 1):
        effect = ir.get_op_buffer_arg_effect("buffer.alloc", index)
        assert effect.non_memory
        assert effect.data == ir.BufferAccess.None_
        assert effect.metadata == ir.BufferAccess.None_


@pytest.mark.parametrize("address", [0, 64, 4096])
def test_effective_address_is_preserved_exactly(address):
    call = allocate(descriptor(), address=integer(address))
    assert len(call.args) == 2
    assert isinstance(call.args[1], ir.ConstInt)
    assert call.args[1].value == address
    assert not ir.structural_equal(call, allocate(descriptor()))


@pytest.mark.parametrize("valid_shape,valid", [([-1, 32], [8]), ([16, -1], [24]), ([-1, -1], [0, 32])])
def test_dynamic_descriptor_extents_are_supplied_in_axis_order(valid_shape, valid):
    call = allocate(descriptor(valid_shape=valid_shape), [integer(value) for value in valid])
    assert isinstance(call.type, ir.BufferType)
    assert call.type.valid_shape == valid_shape
    assert isinstance(call.args[0], ir.MakeTuple)
    for actual, expected in zip(call.args[0].elements, valid, strict=True):
        assert isinstance(actual, ir.ConstInt)
        assert actual.value == expected


def test_address_and_valid_extents_are_visible_to_substitution_and_serialization():
    valid = scalar("valid")
    address = scalar("effective_address")
    next_valid = scalar("next_valid")
    next_address = scalar("next_effective_address")
    call = allocate(descriptor(valid_shape=[-1, 32]), [valid], address)
    rewritten = ir.substitute_expr(call, [(valid, next_valid), (address, next_address)])
    assert isinstance(rewritten, ir.Call)
    assert isinstance(rewritten.args[0], ir.MakeTuple)
    assert rewritten.args[0].elements[0].same_as(next_valid)
    assert rewritten.args[1].same_as(next_address)
    ir.assert_structural_equal(rewritten.type, call.type)
    restored = ir.deserialize(ir.serialize(rewritten))
    ir.assert_structural_equal(rewritten, restored, enable_auto_mapping=True)


@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.UINT32, DataType.INT64, DataType.INDEX])
def test_integer_runtime_values_are_accepted(dtype):
    value = scalar("runtime", dtype)
    call = allocate(descriptor(valid_shape=[16, -1]), [value], value)
    assert call.args[1].same_as(value)


@pytest.mark.parametrize("dtype", [DataType.FP32, DataType.BOOL, DataType.TASK_ID])
@pytest.mark.parametrize("operand", ["valid", "address"])
def test_non_integer_runtime_values_are_rejected(dtype, operand):
    value = scalar("invalid", dtype)
    with pytest.raises(ValueError, match="integer or INDEX scalar"):
        if operand == "valid":
            allocate(descriptor(valid_shape=[-1, 32]), [value])
        else:
            allocate(descriptor(), address=value)


@pytest.mark.parametrize("value", [-1, -64])
def test_negative_address_is_not_an_addressless_sentinel(value):
    with pytest.raises(ValueError, match="effective address must be nonnegative"):
        allocate(descriptor(), address=integer(value))


@pytest.mark.parametrize("valid_shape,valid", [([-1, 32], [-1]), ([-1, 32], [17]), ([16, -1], [33])])
def test_constant_valid_extents_must_fit_their_physical_axes(valid_shape, valid):
    with pytest.raises(ValueError, match="valid extent for dimension .* must be between"):
        allocate(descriptor(valid_shape=valid_shape), [integer(value) for value in valid])


@pytest.mark.parametrize("valid_shape,valid", [([16, 32], [16]), ([-1, 32], []), ([-1, -1], [16])])
def test_valid_operand_count_matches_dynamic_dimensions_only(valid_shape, valid):
    with pytest.raises(ValueError, match="one per dynamic descriptor dimension"):
        allocate(descriptor(valid_shape=valid_shape), [integer(value) for value in valid])


@pytest.mark.parametrize("arg_count", [0, 3])
def test_allocation_operand_arity_is_checked(arg_count):
    with pytest.raises(ValueError, match="runtime-valid tuple and an optional effective address"):
        _ir._create_internal_op_call(
            "buffer.alloc", [integer(0)] * arg_count, {}, descriptor(), ir.Span.unknown()
        )


def test_runtime_valid_extents_require_an_explicit_tuple():
    tuple_value = ir.Var("dims", ir.TupleType([ir.ScalarType(DataType.INDEX)]), ir.Span.unknown())
    with pytest.raises(ValueError, match="runtime valid extents must be a MakeTuple"):
        _ir._create_internal_op_call(
            "buffer.alloc", [tuple_value], {}, descriptor(valid_shape=[-1, 32]), ir.Span.unknown()
        )


@pytest.mark.parametrize(
    "result_type",
    [
        ir.UnknownType(),
        ir.VoidType(),
        ir.TileType([16, 32], DataType.FP32),
        ir.ScalarType(DataType.INDEX),
        ir.MultiBufferType(descriptor(), 2),
        ir.TupleType([descriptor()]),
    ],
)
def test_allocation_requires_one_buffer_descriptor(result_type):
    with pytest.raises(ValueError, match="result must have BufferType"):
        allocate(result_type)


@pytest.mark.parametrize("kwarg", ["shape", "base", "byte_offset", "dtype", "address"])
def test_descriptor_and_address_cannot_be_duplicated_in_kwargs(kwarg):
    with pytest.raises(ValueError, match=f"Unknown kwarg '{kwarg}'"):
        _ir._create_internal_op_call(
            "buffer.alloc", [ir.MakeTuple([], ir.Span.unknown())], {kwarg: 0}, descriptor(), ir.Span.unknown()
        )


def test_allocation_requires_the_internal_explicit_type_builder():
    args = [ir.MakeTuple([], ir.Span.unknown())]
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call("buffer.alloc", args, ir.Span.unknown())
    with pytest.raises(ValueError, match="explicit result type"):
        _ir._create_internal_op_call("buffer.alloc", args, {}, ir.Span.unknown())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
