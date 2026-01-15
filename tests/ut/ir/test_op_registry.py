# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comprehensive tests for the operator registration system.

Tests cover:
- TileType construction and validation
- TensorAdd and TileAdd operations
- Type deduction for various input combinations
- Broadcasting behavior
- Dynamic dimension handling
- Error cases
"""

import pytest
from pypto.pypto_core import DataType, ir


def test_tile_type_valid_dimensions():
    """Test TileType with valid dimensions (0, 1, 2)."""
    span = ir.Span.unknown()

    # Scalar (0 dimensions)
    tile0 = ir.TileType(DataType.FP32, [])
    assert len(tile0.shape) == 0

    # 1D tile
    dim1 = ir.ConstInt(16, DataType.INT32, span)
    tile1 = ir.TileType(DataType.FP16, [dim1])
    assert len(tile1.shape) == 1

    # 2D tile
    dim2 = ir.ConstInt(8, DataType.INT32, span)
    tile2 = ir.TileType(DataType.INT8, [dim1, dim2])
    assert len(tile2.shape) == 2


def test_tile_type_invalid_dimensions():
    """Test TileType rejects more than 2 dimensions."""
    span = ir.Span.unknown()
    dim1 = ir.ConstInt(16, DataType.INT32, span)
    dim2 = ir.ConstInt(8, DataType.INT32, span)
    dim3 = ir.ConstInt(4, DataType.INT32, span)

    with pytest.raises(Exception):  # Should throw std::invalid_argument
        ir.TileType(DataType.FP32, [dim1, dim2, dim3])


def test_dynamic_dimension_constant():
    """Test dynamic dimension constant."""
    # Check that DYNAMIC_DIM is -1
    assert ir.DYNAMIC_DIM == -1

    # Can be used in dimension expressions
    span = ir.Span.unknown()
    dynamic_dim = ir.ConstInt(ir.DYNAMIC_DIM, DataType.INT32, span)
    assert dynamic_dim.value == -1


def test_tensor_add_same_shape():
    """Test TensorAdd with identical shapes."""
    span = ir.Span.unknown()

    # Create shape [4, 8]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape = [dim4, dim8]

    # Create two tensor variables with same shape
    tensor_type = ir.TensorType(DataType.FP32, shape)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Create tensor add operation
    call = ir.create_op_call("tensor.add", [var_a, var_b], span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_add_broadcasting():
    """Test TensorAdd with broadcasting."""
    span = ir.Span.unknown()

    # Tensor A: [4, 8]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape_a = [dim4, dim8]
    type_a = ir.TensorType(DataType.FP32, shape_a)
    var_a = ir.Var("a", type_a, span)

    # Tensor B: [8] (should broadcast to [4, 8])
    shape_b = [dim8]
    type_b = ir.TensorType(DataType.FP32, shape_b)
    var_b = ir.Var("b", type_b, span)

    # Create tensor add operation
    call = ir.create_op_call("tensor.add", [var_a, var_b], span)

    # Check result type - should be [4, 8]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert len(result_type.shape) == 2


def test_tensor_add_broadcasting_with_one():
    """Test TensorAdd broadcasting with dimension of size 1."""
    span = ir.Span.unknown()

    # Tensor A: [4, 1]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape_a = [dim4, dim1]
    type_a = ir.TensorType(DataType.FP32, shape_a)
    var_a = ir.Var("a", type_a, span)

    # Tensor B: [8]
    shape_b = [dim8]
    type_b = ir.TensorType(DataType.FP32, shape_b)
    var_b = ir.Var("b", type_b, span)

    # Create tensor add operation
    call = ir.create_op_call("tensor.add", [var_a, var_b], span)

    # Check result type - should be [4, 8]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert len(result_type.shape) == 2


def test_tensor_add_type_promotion():
    """Test TensorAdd with different data types."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape = [dim8]

    # INT32 + FP32 should promote to FP32
    type_int = ir.TensorType(DataType.INT32, shape)
    type_float = ir.TensorType(DataType.FP32, shape)
    var_int = ir.Var("a", type_int, span)
    var_float = ir.Var("b", type_float, span)

    call = ir.create_op_call("tensor.add", [var_int, var_float], span)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tile_add_same_shape():
    """Test TileAdd with identical 2D shapes."""
    span = ir.Span.unknown()

    # Create shape [16, 16]
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    shape = [dim16, dim16]

    # Create two tile variables with same shape
    tile_type = ir.TileType(DataType.FP16, shape)
    var_a = ir.Var("t1", tile_type, span)
    var_b = ir.Var("t2", tile_type, span)

    # Create tile add operation
    call = ir.create_op_call("tile.add", [var_a, var_b], span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tile_add_broadcasting():
    """Test TileAdd with 2D broadcasting."""
    span = ir.Span.unknown()

    # Tile A: [16, 16]
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    shape_a = [dim16, dim16]
    type_a = ir.TileType(DataType.FP16, shape_a)
    var_a = ir.Var("t1", type_a, span)

    # Tile B: [16] (1D, should broadcast to [16, 16])
    shape_b = [dim16]
    type_b = ir.TileType(DataType.FP16, shape_b)
    var_b = ir.Var("t2", type_b, span)

    # Create tile add operation
    call = ir.create_op_call("tile.add", [var_a, var_b], span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert len(result_type.shape) == 2


def test_tile_add_broadcasting_with_one():
    """Test TileAdd broadcasting with dimension of size 1."""
    span = ir.Span.unknown()

    # Tile A: [1, 16]
    dim1 = ir.ConstInt(1, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    shape_a = [dim1, dim16]
    type_a = ir.TileType(DataType.FP16, shape_a)
    var_a = ir.Var("t1", type_a, span)

    # Tile B: [16, 16]
    shape_b = [dim16, dim16]
    type_b = ir.TileType(DataType.FP16, shape_b)
    var_b = ir.Var("t2", type_b, span)

    # Create tile add operation
    call = ir.create_op_call("tile.add", [var_a, var_b], span)

    # Check result type - should be [16, 16]
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert len(result_type.shape) == 2


def test_tensor_add_wrong_arg_count():
    """Test TensorAdd with wrong number of arguments."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType(DataType.FP32, [dim8])
    var_a = ir.Var("a", tensor_type, span)

    # Too few arguments
    with pytest.raises(Exception):
        ir.create_op_call("tensor.add", [var_a], span)

    # Too many arguments
    var_b = ir.Var("b", tensor_type, span)
    var_c = ir.Var("c", tensor_type, span)
    with pytest.raises(Exception):
        ir.create_op_call("tensor.add", [var_a, var_b, var_c], span)


def test_tensor_add_wrong_type():
    """Test TensorAdd with non-tensor arguments."""
    span = ir.Span.unknown()

    # Scalar type instead of tensor
    scalar_type = ir.ScalarType(DataType.FP32)
    var_scalar = ir.Var("s", scalar_type, span)

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType(DataType.FP32, [dim8])
    var_tensor = ir.Var("t", tensor_type, span)

    with pytest.raises(Exception):
        ir.create_op_call("tensor.add", [var_scalar, var_tensor], span)


def test_tile_add_wrong_type():
    """Test TileAdd with non-tile arguments."""
    span = ir.Span.unknown()

    # Tensor type instead of tile
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType(DataType.FP32, [dim8])
    var_tensor = ir.Var("t", tensor_type, span)

    tile_type = ir.TileType(DataType.FP32, [dim8])
    var_tile = ir.Var("tile", tile_type, span)

    with pytest.raises(Exception):
        ir.create_op_call("tile.add", [var_tensor, var_tile], span)


def test_operator_registration_status():
    """Test operator registration queries."""
    # Check that our operators are registered
    assert ir.is_op_registered("tensor.add")
    assert ir.is_op_registered("tensor.sub")
    assert ir.is_op_registered("tensor.mul")
    assert ir.is_op_registered("tensor.div")
    assert ir.is_op_registered("tile.add")
    assert ir.is_op_registered("tile.sub")
    assert ir.is_op_registered("tile.mul")
    assert ir.is_op_registered("tile.div")

    # Check that a non-existent operator is not registered
    assert not ir.is_op_registered("nonexistent.op")


def test_get_op():
    """Test getting operator instances."""
    tensor_add_op = ir.get_op("tensor.add")
    assert tensor_add_op.name == "tensor.add"

    tile_mul_op = ir.get_op("tile.mul")
    assert tile_mul_op.name == "tile.mul"

    # Non-existent operator should raise exception
    with pytest.raises(Exception):
        ir.get_op("nonexistent.op")


def test_test_op_int_attribute():
    """Test that test.op has int_attr set to 10."""
    test_op = ir.get_op("test.op")

    # Check attribute exists
    assert test_op.has_attr("int_attr")

    # Check attribute value
    int_attr = test_op.get_attr("int_attr")
    assert int_attr == 10
    assert isinstance(int_attr, int)


def test_test_op_string_attribute():
    """Test that test.op has string_attr set to 'test'."""
    test_op = ir.get_op("test.op")

    # Check attribute exists
    assert test_op.has_attr("string_attr")

    # Check attribute value
    string_attr = test_op.get_attr("string_attr")
    assert string_attr == "test"
    assert isinstance(string_attr, str)


def test_test_op_bool_attribute():
    """Test that test.op has bool_attr set to True."""
    test_op = ir.get_op("test.op")

    # Check attribute exists
    assert test_op.has_attr("bool_attr")

    # Check attribute value
    bool_attr = test_op.get_attr("bool_attr")
    assert bool_attr is True
    assert isinstance(bool_attr, bool)


def test_test_op_all_attributes():
    """Test all attributes of test.op at once."""
    test_op = ir.get_op("test.op")

    # Get all attribute keys
    keys = test_op.get_attr_keys()

    # Check all expected attributes are present
    assert "int_attr" in keys
    assert "string_attr" in keys
    assert "bool_attr" in keys

    # Verify we have exactly 3 attributes
    assert len(keys) == 3


def test_test_op_nonexistent_attribute():
    """Test accessing non-existent attributes."""
    test_op = ir.get_op("test.op")

    # Check that non-existent attribute is not present
    assert not test_op.has_attr("nonexistent")
    assert not test_op.has_attr("device")
    assert not test_op.has_attr("priority")

    # Accessing non-existent attribute should raise exception
    with pytest.raises(Exception):
        test_op.get_attr("nonexistent")


def test_test_op_attribute_isolation():
    """Test that test.op attributes are isolated from other operators."""
    test_op = ir.get_op("test.op")
    tensor_add_op = ir.get_op("tensor.add")

    # test.op should have int_attr, string_attr, bool_attr
    assert test_op.has_attr("int_attr")
    assert test_op.has_attr("string_attr")
    assert test_op.has_attr("bool_attr")

    # tensor.add should NOT have these attributes
    assert not tensor_add_op.has_attr("int_attr")
    assert not tensor_add_op.has_attr("string_attr")
    assert not tensor_add_op.has_attr("bool_attr")

    # tensor.add has its own attributes
    if tensor_add_op.has_attr("device"):
        # test.op should not have tensor.add's attributes
        assert not test_op.has_attr("device")


def test_test_op_attribute_types():
    """Test that attribute types are correctly preserved."""
    test_op = ir.get_op("test.op")

    # Get all attributes
    int_attr = test_op.get_attr("int_attr")
    string_attr = test_op.get_attr("string_attr")
    bool_attr = test_op.get_attr("bool_attr")

    # Verify types
    assert type(int_attr) is int
    assert type(string_attr) is str
    assert type(bool_attr) is bool

    # Verify values
    assert int_attr == 10
    assert string_attr == "test"
    assert bool_attr is True


def test_tensor_sub_mul_div():
    """Test other tensor operations (sub, mul, div)."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape = [dim8]
    tensor_type = ir.TensorType(DataType.FP32, shape)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Test sub
    call_sub = ir.create_op_call("tensor.sub", [var_a, var_b], span)
    assert isinstance(call_sub.type, ir.TensorType)

    # Test mul
    call_mul = ir.create_op_call("tensor.mul", [var_a, var_b], span)
    assert isinstance(call_mul.type, ir.TensorType)

    # Test div
    call_div = ir.create_op_call("tensor.div", [var_a, var_b], span)
    assert isinstance(call_div.type, ir.TensorType)


def test_tile_sub_mul_div():
    """Test other tile operations (sub, mul, div)."""
    span = ir.Span.unknown()

    dim16 = ir.ConstInt(16, DataType.INT32, span)
    shape = [dim16, dim16]
    tile_type = ir.TileType(DataType.FP16, shape)
    var_a = ir.Var("t1", tile_type, span)
    var_b = ir.Var("t2", tile_type, span)

    # Test sub
    call_sub = ir.create_op_call("tile.sub", [var_a, var_b], span)
    assert isinstance(call_sub.type, ir.TileType)

    # Test mul
    call_mul = ir.create_op_call("tile.mul", [var_a, var_b], span)
    assert isinstance(call_mul.type, ir.TileType)

    # Test div
    call_div = ir.create_op_call("tile.div", [var_a, var_b], span)
    assert isinstance(call_div.type, ir.TileType)


def test_call_with_explicit_type():
    """Test Call constructor with explicit type parameter."""
    span = ir.Span.unknown()

    # Create a simple operation
    op = ir.get_op("tensor.add")

    # Create arguments
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType(DataType.FP32, [dim8])
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Create call with explicit type
    result_type = ir.TensorType(DataType.FP32, [dim8])
    call = ir.Call(op, [var_a, var_b], result_type, span)

    # Verify type is set correctly
    assert isinstance(call.type, ir.TensorType)
    assert call.type.dtype == DataType.FP32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
