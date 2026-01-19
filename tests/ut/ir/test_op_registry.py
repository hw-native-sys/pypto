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
    tile0 = ir.TileType([], DataType.FP32)
    assert len(tile0.shape) == 0

    # 1D tile
    dim1 = ir.ConstInt(16, DataType.INT32, span)
    tile1 = ir.TileType([dim1], DataType.FP16)
    assert len(tile1.shape) == 1

    # 2D tile
    dim2 = ir.ConstInt(8, DataType.INT32, span)
    tile2 = ir.TileType([dim1, dim2], DataType.INT8)
    assert len(tile2.shape) == 2


def test_tile_type_invalid_dimensions():
    """Test TileType rejects more than 2 dimensions."""
    span = ir.Span.unknown()
    dim1 = ir.ConstInt(16, DataType.INT32, span)
    dim2 = ir.ConstInt(8, DataType.INT32, span)
    dim3 = ir.ConstInt(4, DataType.INT32, span)

    with pytest.raises(Exception):  # Should throw std::invalid_argument
        ir.TileType([dim1, dim2, dim3], DataType.FP32)


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
    tensor_type = ir.TensorType(shape, DataType.FP32)
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
    type_a = ir.TensorType(shape_a, DataType.FP32)
    var_a = ir.Var("a", type_a, span)

    # Tensor B: [8] (should broadcast to [4, 8])
    shape_b = [dim8]
    type_b = ir.TensorType(shape_b, DataType.FP32)
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
    type_a = ir.TensorType(shape_a, DataType.FP32)
    var_a = ir.Var("a", type_a, span)

    # Tensor B: [8]
    shape_b = [dim8]
    type_b = ir.TensorType(shape_b, DataType.FP32)
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
    type_int = ir.TensorType(shape, DataType.INT32)
    type_float = ir.TensorType(shape, DataType.FP32)
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
    tile_type = ir.TileType(shape, DataType.FP16)
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
    type_a = ir.TileType(shape_a, DataType.FP16)
    var_a = ir.Var("t1", type_a, span)

    # Tile B: [16] (1D, should broadcast to [16, 16])
    shape_b = [dim16]
    type_b = ir.TileType(shape_b, DataType.FP16)
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
    type_a = ir.TileType(shape_a, DataType.FP16)
    var_a = ir.Var("t1", type_a, span)

    # Tile B: [16, 16]
    shape_b = [dim16, dim16]
    type_b = ir.TileType(shape_b, DataType.FP16)
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
    tensor_type = ir.TensorType([dim8], DataType.FP32)
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
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_tensor = ir.Var("t", tensor_type, span)

    with pytest.raises(Exception):
        ir.create_op_call("tensor.add", [var_scalar, var_tensor], span)


def test_tile_add_wrong_type():
    """Test TileAdd with non-tile arguments."""
    span = ir.Span.unknown()

    # Tensor type instead of tile
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_tensor = ir.Var("t", tensor_type, span)

    tile_type = ir.TileType([dim8], DataType.FP32)
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


def test_test_op_kwarg_schema():
    """Test that test.op has kwarg schema defined."""
    test_op = ir.get_op("test.op")

    # Check kwarg keys exist in schema
    assert test_op.has_attr("int_attr")
    assert test_op.has_attr("string_attr")
    assert test_op.has_attr("bool_attr")


def test_test_op_all_kwarg_keys():
    """Test all kwarg keys of test.op."""
    test_op = ir.get_op("test.op")

    # Get all kwarg keys from schema
    keys = test_op.get_attr_keys()

    # Check all expected kwargs are present
    assert "int_attr" in keys
    assert "string_attr" in keys
    assert "bool_attr" in keys

    # Verify we have exactly 3 kwargs
    assert len(keys) == 3


def test_test_op_nonexistent_kwarg():
    """Test checking non-existent kwargs."""
    test_op = ir.get_op("test.op")

    # Check that non-existent kwarg is not in schema
    assert not test_op.has_attr("nonexistent")
    assert not test_op.has_attr("device")
    assert not test_op.has_attr("priority")


def test_test_op_kwarg_isolation():
    """Test that test.op kwarg schema is isolated from other operators."""
    test_op = ir.get_op("test.op")
    tensor_add_op = ir.get_op("tensor.add")

    # test.op should have int_attr, string_attr, bool_attr in schema
    assert test_op.has_attr("int_attr")
    assert test_op.has_attr("string_attr")
    assert test_op.has_attr("bool_attr")

    # tensor.add should NOT have these in its schema
    assert not tensor_add_op.has_attr("int_attr")
    assert not tensor_add_op.has_attr("string_attr")
    assert not tensor_add_op.has_attr("bool_attr")


def test_tensor_sub_mul_div():
    """Test other tensor operations (sub, mul, div)."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    shape = [dim8]
    tensor_type = ir.TensorType(shape, DataType.FP32)
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
    tile_type = ir.TileType(shape, DataType.FP16)
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
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Create call with explicit type
    result_type = ir.TensorType([dim8], DataType.FP32)
    call = ir.Call(op, [var_a, var_b], {}, result_type, span)

    # Verify type is set correctly
    assert isinstance(call.type, ir.TensorType)
    assert call.type.dtype == DataType.FP32


def test_matmul_with_valid_kwargs():
    """Test tensor.matmul with valid kwargs."""
    span = ir.Span.unknown()

    # Create two matrices
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)

    type_a = ir.TensorType([dim64, dim128], DataType.FP16)
    type_b = ir.TensorType([dim128, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_b, span)

    # Test with DataType kwarg (passed directly)
    kwargs = {"out_dtype": DataType.FP32, "a_trans": False, "b_trans": False}
    call = ir.create_op_call("tensor.matmul", [var_a, var_b], kwargs, span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_matmul_with_transpose_kwargs():
    """Test tensor.matmul with transpose kwargs."""
    span = ir.Span.unknown()

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)

    type_a = ir.TensorType([dim128, dim64], DataType.FP16)  # Will be transposed
    type_b = ir.TensorType([dim128, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_b, span)

    # Test with a_trans=True
    kwargs = {"a_trans": True, "b_trans": False}
    call = ir.create_op_call("tensor.matmul", [var_a, var_b], kwargs, span)

    # Should work without error
    assert isinstance(call.type, ir.TensorType)


def test_matmul_with_unknown_kwarg():
    """Test tensor.matmul with unknown kwarg should raise error."""
    span = ir.Span.unknown()

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    type_a = ir.TensorType([dim64, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_a, span)

    # Unknown kwarg should raise ValueError
    kwargs = {"unknown_param": 123, "a_trans": False}

    with pytest.raises(Exception) as exc_info:
        ir.create_op_call("tensor.matmul", [var_a, var_b], kwargs, span)

    # Check error message contains "unknown"
    assert "unknown" in str(exc_info.value).lower() or "Unknown" in str(exc_info.value)


def test_matmul_with_wrong_type_kwarg():
    """Test tensor.matmul with wrong type kwarg should raise error."""
    span = ir.Span.unknown()

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    type_a = ir.TensorType([dim64, dim64], DataType.FP16)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_a, span)

    # Wrong type for bool kwarg (passing string instead of bool)
    kwargs = {
        "a_trans": "true"  # Should be bool, not string
    }

    with pytest.raises(Exception) as exc_info:
        ir.create_op_call("tensor.matmul", [var_a, var_b], kwargs, span)

    # Check error message indicates type mismatch
    error_msg = str(exc_info.value).lower()
    assert "type" in error_msg or "incompatible" in error_msg


def test_cast_with_datatype_kwarg():
    """Test tensor.cast with DataType kwarg."""
    span = ir.Span.unknown()

    dim8 = ir.ConstInt(8, DataType.INT32, span)
    type_fp16 = ir.TensorType([dim8], DataType.FP16)
    var_a = ir.Var("a", type_fp16, span)

    # Cast from FP16 to FP32
    kwargs = {"target_type": DataType.FP32}
    call = ir.create_op_call("tensor.cast", [var_a], kwargs, span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_reduction_with_kwargs():
    """Test tensor reduction operations with kwargs."""
    span = ir.Span.unknown()

    # Create a 2D tensor
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim4, dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)

    # Test row_max with axis and keep_dim kwargs
    kwargs = {"axis": -1, "keep_dim": True}
    call = ir.create_op_call("tensor.row_max", [var_a], kwargs, span)

    # Should work without error
    assert isinstance(call.type, ir.TensorType)


def test_matmul_kwarg_schema():
    """Test that tensor.matmul has correct kwarg schema."""
    matmul_op = ir.get_op("tensor.matmul")

    # Check that expected kwargs are in schema
    assert matmul_op.has_attr("out_dtype")
    assert matmul_op.has_attr("a_trans")
    assert matmul_op.has_attr("b_trans")
    assert matmul_op.has_attr("c_matrix_nz")

    # Get all kwarg keys
    keys = matmul_op.get_attr_keys()
    assert "out_dtype" in keys
    assert "a_trans" in keys
    assert "b_trans" in keys


def test_cast_kwarg_schema():
    """Test that tensor.cast has correct kwarg schema."""
    cast_op = ir.get_op("tensor.cast")

    # Check that expected kwargs are in schema
    assert cast_op.has_attr("target_type")
    assert cast_op.has_attr("mode")


def test_reduction_kwarg_schema():
    """Test that tensor reduction ops have correct kwarg schema."""
    row_max_op = ir.get_op("tensor.row_max")
    row_sum_op = ir.get_op("tensor.row_sum")

    # Check that expected kwargs are in schema
    assert row_max_op.has_attr("axis")
    assert row_max_op.has_attr("keep_dim")
    assert row_sum_op.has_attr("axis")
    assert row_sum_op.has_attr("keep_dim")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
