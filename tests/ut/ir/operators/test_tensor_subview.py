# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Affine GM subviews: parent-window containment and descriptor-only typing."""

from collections.abc import Sequence

import pytest
from pypto import DataType, ir


def values(expressions: Sequence[ir.Expr]) -> list[int]:
    result = []
    for expr in expressions:
        assert isinstance(expr, ir.ConstInt)
        result.append(expr.value)
    return result


def tensor_type(expr: ir.Expr) -> ir.TensorType:
    result = expr.type
    assert isinstance(result, ir.TensorType)
    return result


def tensor_view(expr: ir.Expr):
    result = tensor_type(expr).tensor_view
    assert result is not None
    return result


def source(shape=(8, 512)):
    return ir.Var("cache", ir.TensorType(list(shape), DataType.BF16), ir.Span.unknown())


def test_subview_preserves_parent_pitch_and_has_no_allocation():
    call = ir.op.tensor.subview(source(), [2, 512], [1, 0], steps=[3, 1])
    assert call.op.name == "tensor.subview"
    assert values(tensor_type(call).shape) == [2, 512]
    assert values(tensor_view(call).stride) == [1536, 1]
    assert tensor_view(call).addressing_kind == ir.TensorAddressingKind.AffineSubview
    assert tensor_type(call).memref is None


@pytest.mark.parametrize(
    "shape,offsets,steps",
    [
        ([8, 512], [0, 0], [1, 1]),
        ([2, 512], [5, 0], [3, 1]),
        ([2, 512], [-1, 0], [1, 1]),
        ([0, 512], [0, 0], [1, 1]),
        ([2, 512], [0, 0], [0, 1]),
        ([2], [0], [1]),
    ],
)
def test_subview_rejects_invalid_constant_geometry(shape, offsets, steps):
    with pytest.raises(ValueError, match="subview"):
        ir.op.tensor.subview(source(), shape, offsets, steps=steps)


def test_nested_subview_composes_immediate_parent_stride():
    first = ir.op.tensor.subview(source(), [3, 512], [0, 0], steps=[2, 1])
    parent = ir.Var("first", first.type, ir.Span.unknown())
    child = ir.op.tensor.subview(parent, [1, 248], [1, 16], steps=[1, 2])
    assert values(tensor_view(child).stride) == [1024, 2]


def test_subview_serialization_preserves_affine_addressing():
    call = ir.op.tensor.subview(source(), [2, 512], [1, 0], steps=[3, 1])
    restored = ir.deserialize(ir.serialize(call))
    assert isinstance(restored, ir.Call)
    ir.assert_structural_equal(call.type, restored.type)
    assert restored.op.name == call.op.name
    assert tensor_view(restored).addressing_kind == ir.TensorAddressingKind.AffineSubview


@pytest.mark.parametrize("value", [True, 1.5])
def test_subview_geometry_is_integer_and_not_boolean(value):
    with pytest.raises(ValueError, match="subview"):
        ir.op.tensor.subview(source(), [2, 512], [value, 0])


def test_subview_rejects_tile_source():
    tile = ir.Var("tile", ir.TileType([8, 512], DataType.BF16), ir.Span.unknown())
    with pytest.raises(ValueError, match="subview.*ordinary Tensor"):
        ir.op.tensor.subview(tile, [2, 512], [0, 0])


def test_subview_rejects_distributed_source():
    distributed = ir.Var("distributed", ir.DistributedTensorType([8, 512], DataType.BF16), ir.Span.unknown())
    with pytest.raises(ValueError, match="subview.*ordinary Tensor"):
        ir.op.tensor.subview(distributed, [2, 512], [0, 0])


def test_subview_view_cannot_flatten_a_row_gap():
    selected = ir.op.tensor.subview(source(), [2, 512], [1, 0], steps=[3, 1])
    alias = ir.Var("selected", selected.type, ir.Span.unknown())
    with pytest.raises(ValueError, match="view"):
        ir.op.tensor.view(alias, [1024])


@pytest.mark.parametrize(
    "shape,new_shape,offsets,steps,expected",
    [([8], [2], [1], [3], [3]), ([4, 8, 16], [2, 3, 8], [1, 0, 0], [1, 2, 2], [128, 32, 2])],
)
def test_subview_generic_rank_and_inner_step_semantics(shape, new_shape, offsets, steps, expected):
    call = ir.op.tensor.subview(source(shape), new_shape, offsets, steps=steps)
    assert values(tensor_view(call).stride) == expected


def test_subview_containment_uses_parent_valid_window():
    span = ir.Span.unknown()
    tensor = ir.Var(
        "partial",
        ir.TensorType(
            [8, 512], DataType.BF16, tensor_view=ir.TensorView([512, 1], ir.TensorLayout.ND, [5, 512])
        ),
        span,
    )
    accepted = ir.op.tensor.subview(tensor, [2, 512], [1, 0], steps=[3, 1])
    assert values(tensor_type(accepted).shape) == [2, 512]
    with pytest.raises(ValueError, match="subview.*parent window"):
        ir.op.tensor.subview(tensor, [2, 512], [2, 0], steps=[3, 1])


def test_subview_device_integer_geometry_is_widened_before_arithmetic():
    span = ir.Span.unknown()
    offset = ir.Var("offset", ir.ScalarType(DataType.INT32), span)
    call = ir.op.tensor.subview(source(), [2, 512], [offset, 0])
    offsets = call.args[2]
    assert isinstance(offsets, ir.MakeTuple)
    assert isinstance(offsets.elements[0], ir.Cast)
    scalar_type = offsets.elements[0].type
    assert isinstance(scalar_type, ir.ScalarType)
    assert scalar_type.dtype == DataType.INDEX


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
