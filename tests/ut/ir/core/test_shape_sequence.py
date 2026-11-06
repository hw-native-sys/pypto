# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the mixed-extent shape overloads of ir.TensorType / ir.TileType.

``shape: Sequence[Expr | int]`` is the declared contract; the raw constructors
normalize a literal extent into a ``ConstInt``. These tests pin the acceptance
boundary: anything that is not an ``Expr`` or an ``int`` — ``bool`` included,
since ``PyNumber_Index`` would silently convert it to 0/1 — must be rejected
with a ``TypeError`` naming the offending element index.
"""

import pytest
from pypto import DataType, ir


@pytest.fixture
def symbolic_dim():
    span = ir.Span.unknown()
    return ir.Var("n", ir.ScalarType(DataType.INT32), span)


class TestTensorTypeMixedShape:
    def test_mixed_expr_and_constant_accepted(self, symbolic_dim):
        tensor_type = ir.TensorType([symbolic_dim, 16], DataType.FP32)
        assert len(tensor_type.shape) == 2
        assert isinstance(tensor_type.shape[0], ir.Var)
        assert isinstance(tensor_type.shape[1], ir.ConstInt)

    def test_homogeneous_constant_shape_still_accepted(self):
        tensor_type = ir.TensorType([32, 64], DataType.FP32)
        assert len(tensor_type.shape) == 2

    def test_homogeneous_expr_shape_still_accepted(self, symbolic_dim):
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span), symbolic_dim]
        tensor_type = ir.TensorType(shape, DataType.FP32)
        assert isinstance(tensor_type.shape[0], ir.ConstInt)

    @pytest.mark.parametrize("bad", [True, "16", 2.5])
    def test_non_int_element_rejected_with_index(self, symbolic_dim, bad):
        with pytest.raises(TypeError, match=r"index 1"):
            ir.TensorType([symbolic_dim, bad], DataType.FP32)

    def test_bool_element_reported_as_bool(self, symbolic_dim):
        with pytest.raises(TypeError, match="bool"):
            ir.TensorType([symbolic_dim, True], DataType.FP32)

    def test_out_of_range_int_element_rejected_with_index(self, symbolic_dim):
        # A Python int that does not fit int64 must not surface as a context-free
        # `RuntimeError: std::bad_cast` from nanobind's caster.
        with pytest.raises(ValueError, match=r"index 1.*int64"):
            ir.TensorType([symbolic_dim, 2**70], DataType.FP32)


class TestTileTypeMixedShape:
    def test_mixed_expr_and_constant_accepted(self, symbolic_dim):
        tile_type = ir.TileType([16, symbolic_dim], DataType.FP32)
        assert len(tile_type.shape) == 2
        assert isinstance(tile_type.shape[1], ir.Var)

    def test_homogeneous_constant_shape_still_accepted(self):
        tile_type = ir.TileType([16, 64], DataType.FP32)
        assert len(tile_type.shape) == 2

    @pytest.mark.parametrize("bad", [True, "16", 2.5])
    def test_non_int_element_rejected_with_index(self, bad):
        with pytest.raises(TypeError, match=r"index 1"):
            ir.TileType([16, bad], DataType.FP32)

    def test_out_of_range_int_element_rejected_with_index(self):
        with pytest.raises(ValueError, match=r"index 0.*int64"):
            ir.TileType([2**70, 16], DataType.FP32)


@pytest.mark.parametrize("factory", [ir.TensorType, ir.TileType], ids=["tensor", "tile"])
@pytest.mark.parametrize("bad", [True, False, None, "16", 2.5])
@pytest.mark.parametrize("shape_kind", ["constant", "symbolic", "single"])
def test_invalid_extent_uses_the_same_validation(factory, symbolic_dim, bad, shape_kind):
    """Neither homogeneous nor mixed shapes may bypass element validation.

    The old integer overload accepted [16, True] and [True], while the mixed
    overload rejected [n, True]. All forms must report the offending index.
    """
    if shape_kind == "single":
        shape, index = [bad], 0
    else:
        first = symbolic_dim if shape_kind == "symbolic" else 16
        shape, index = [first, bad], 1
    with pytest.raises(TypeError, match=rf"index {index}.*{type(bad).__name__}"):
        factory(shape, DataType.FP32)


@pytest.mark.parametrize("factory", [ir.TensorType, ir.TileType], ids=["tensor", "tile"])
@pytest.mark.parametrize("sequence", [list, tuple], ids=["list", "tuple"])
@pytest.mark.parametrize("shape_kind", ["empty", "constant", "symbolic", "mixed"])
def test_valid_shapes_preserve_extents(factory, sequence, symbolic_dim, shape_kind):
    """The unified constructor preserves values and symbolic Var identities."""
    if shape_kind == "empty":
        shape = []
    elif shape_kind == "constant":
        shape = [16, 32]
    elif shape_kind == "symbolic":
        shape = [symbolic_dim, ir.ConstInt(16, DataType.INT32, ir.Span.unknown())]
    else:
        shape = [16, symbolic_dim]
    result = factory(sequence(shape), DataType.FP32)
    assert result.dtype == DataType.FP32
    assert len(result.shape) == len(shape)
    for original, actual in zip(shape, result.shape, strict=True):
        if isinstance(original, int):
            assert isinstance(actual, ir.ConstInt)
            assert actual.value == original
            assert actual.dtype == DataType.INDEX
        else:
            assert actual.same_as(original)


@pytest.mark.parametrize("keyword_args", [False, True], ids=["positional", "keyword"])
def test_tensor_metadata_is_preserved(keyword_args):
    memref = ir.MemRef("storage", 0, 2048, ir.Span.unknown())
    view = ir.TensorView(stride=[32, 1], layout=ir.TensorLayout.ND, valid_shape=[8, 32])
    if keyword_args:
        result = ir.TensorType([16, 32], DataType.FP32, memref=memref, tensor_view=view)
    else:
        result = ir.TensorType([16, 32], DataType.FP32, memref, view)
    assert result.memref.same_as(memref)
    assert [extent.value for extent in result.tensor_view.valid_shape] == [8, 32]
    assert [stride.value for stride in result.tensor_view.stride] == [32, 1]


@pytest.mark.parametrize("keyword_args", [False, True], ids=["positional", "keyword"])
def test_tile_metadata_is_preserved(keyword_args):
    memref = ir.MemRef("storage", 0, 2048, ir.Span.unknown())
    view = ir.TileView(valid_shape=[8, 32])
    if keyword_args:
        result = ir.TileType(
            [16, 32], DataType.FP32, memref=memref, tile_view=view, memory_space=ir.MemorySpace.Vec
        )
    else:
        result = ir.TileType([16, 32], DataType.FP32, memref, view, ir.MemorySpace.Vec)
    assert result.memref.same_as(memref)
    assert [extent.value for extent in result.tile_view.valid_shape] == [8, 32]
    assert result.memory_space == ir.MemorySpace.Vec


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
