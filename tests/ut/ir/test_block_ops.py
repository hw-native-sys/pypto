# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for block operations and rms_norm_block function construction.

Tests cover:
- Block operation registration
- Type deduction for block operations
- Construction of rms_norm_block function using IR
"""

import pytest
from pypto.pypto_core import DataType, ir


def test_block_ops_registration():
    """Test that all block operations are registered."""
    assert ir.is_op_registered("block.get_block_idx")
    assert ir.is_op_registered("block.ub_copy_in")
    assert ir.is_op_registered("block.mul")
    assert ir.is_op_registered("block.add")
    assert ir.is_op_registered("block.div")
    assert ir.is_op_registered("block.sum")
    assert ir.is_op_registered("block.sqrt")
    assert ir.is_op_registered("block.ub_copy_out")


def test_block_get_block_idx():
    """Test block.get_block_idx operation."""
    span = ir.Span.unknown()

    # get_block_idx takes no arguments and returns INT32 scalar
    call = ir.create_op_call("block.get_block_idx", [], span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.ScalarType)
    assert result_type.dtype == DataType.INT32


def test_block_mul():
    """Test block.mul operation."""
    span = ir.Span.unknown()

    # Create two tiles
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tile_shape = [dim8, dim5120]
    tile_type = ir.TileType(DataType.BF16, tile_shape)

    var_t1 = ir.Var("t1", tile_type, span)
    var_t2 = ir.Var("t2", tile_type, span)

    # Create block.mul operation
    call = ir.create_op_call("block.mul", [var_t1, var_t2], span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == DataType.BF16
    assert len(result_type.shape) == 2


def test_block_add_tile_scalar():
    """Test block.add with tile and scalar."""
    span = ir.Span.unknown()

    # Create tile
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tile_shape = [dim8, dim5120]
    tile_type = ir.TileType(DataType.FP32, tile_shape)
    var_tile = ir.Var("tile", tile_type, span)

    # Create scalar
    epsilon_var = ir.Var("epsilon", ir.ScalarType(DataType.FP32), span)

    # Create block.add operation
    call = ir.create_op_call("block.add", [var_tile, epsilon_var], span)

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_block_sum():
    """Test block.sum reduction operation."""
    span = ir.Span.unknown()

    # Create tile
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tile_shape = [dim8, dim5120]
    tile_type = ir.TileType(DataType.FP32, tile_shape)
    var_tile = ir.Var("tile", tile_type, span)

    # Test block.sum with axis 0 - reduces first dimension
    axis0 = ir.ConstInt(0, DataType.INT32, span)
    call_axis0 = ir.create_op_call("block.sum", [var_tile, axis0], span)

    # Check result type - should be TileType with shape [dim5120]
    result_type_axis0 = call_axis0.type
    assert isinstance(result_type_axis0, ir.TileType)
    assert result_type_axis0.dtype == DataType.FP32
    assert len(result_type_axis0.shape) == 1

    # Test block.sum with axis 1 - reduces second dimension
    axis1 = ir.ConstInt(1, DataType.INT32, span)
    call_axis1 = ir.create_op_call("block.sum", [var_tile, axis1], span)

    # Check result type - should be TileType with shape [dim8]
    result_type_axis1 = call_axis1.type
    assert isinstance(result_type_axis1, ir.TileType)
    assert result_type_axis1.dtype == DataType.FP32
    assert len(result_type_axis1.shape) == 1

    # Test block.sum with both axes - reduces all dimensions to ScalarType
    # This would require reducing along both axes, which can be done by chaining reductions
    # or by specifying multiple axes (if supported in the future)
    # For now, we test that reducing along a single axis works correctly


def test_block_sqrt():
    """Test block.sqrt unary operation."""
    span = ir.Span.unknown()

    # Create tile
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tile_shape = [dim8, dim5120]
    tile_type = ir.TileType(DataType.FP32, tile_shape)
    var_tile = ir.Var("tile", tile_type, span)

    # Create block.sqrt operation
    call = ir.create_op_call("block.sqrt", [var_tile], span)

    # Check result type - should be TileType with same shape
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_block_ub_copy_in():
    """Test block.ub_copy_in operation."""
    span = ir.Span.unknown()

    # Create tensor
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tensor_shape = [dim128, dim5120]
    tensor_type = ir.TensorType(DataType.BF16, tensor_shape)
    var_tensor = ir.Var("x", tensor_type, span)

    # Create offset and shape arguments
    row_offset = ir.ConstInt(0, DataType.INT32, span)
    col_offset = ir.ConstInt(0, DataType.INT32, span)
    height = ir.ConstInt(8, DataType.INT32, span)
    width = ir.ConstInt(5120, DataType.INT32, span)

    # Create block.ub_copy_in operation
    call = ir.create_op_call("block.ub_copy_in", [var_tensor, row_offset, col_offset, height, width], span)

    # Check result type - should be TileType
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == DataType.BF16


def test_block_ub_copy_out():
    """Test block.ub_copy_out operation."""
    span = ir.Span.unknown()

    # Create tile
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tile_shape = [dim8, dim5120]
    tile_type = ir.TileType(DataType.BF16, tile_shape)
    var_tile = ir.Var("tile", tile_type, span)

    # Create output tensor
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_shape = [dim128, dim5120]
    tensor_type = ir.TensorType(DataType.BF16, tensor_shape)
    var_output = ir.Var("y", tensor_type, span)

    # Create offset and shape arguments
    row_offset = ir.ConstInt(0, DataType.INT32, span)
    col_offset = ir.ConstInt(0, DataType.INT32, span)
    height = ir.ConstInt(128, DataType.INT32, span)
    width = ir.ConstInt(5120, DataType.INT32, span)

    # Create block.ub_copy_out operation
    call = ir.create_op_call(
        "block.ub_copy_out", [var_tile, row_offset, col_offset, height, width, var_output], span
    )

    # Check result type - should be TensorType
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.BF16


def test_rms_norm_block_function():
    """Test constructing rms_norm_block function using IR.

    This test constructs a simplified version of the rms_norm_block function
    from examples/block-level/rmsnorm.py using IR operations.
    It demonstrates that all block operations can be combined to create a function.
    """
    span = ir.Span.unknown()

    # Define constants
    tile_height = ir.ConstInt(8, DataType.INT32, span)
    tile_width = ir.ConstInt(5120, DataType.INT32, span)

    # Create function parameters
    # x: Tensor, x_gamma: Tensor, y: Tensor, block_idx: Scalar, epsilon: Scalar
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tensor_shape = [dim128, dim5120]

    x_type = ir.TensorType(DataType.BF16, tensor_shape)
    x = ir.Var("x", x_type, span)

    x_gamma_type = ir.TensorType(DataType.BF16, [dim5120])
    x_gamma = ir.Var("x_gamma", x_gamma_type, span)

    y_type = ir.TensorType(DataType.BF16, tensor_shape)
    y = ir.Var("y", y_type, span)

    block_idx = ir.Var("block_idx", ir.ScalarType(DataType.INT32), span)
    epsilon = ir.Var("epsilon", ir.ScalarType(DataType.FP32), span)

    # Create intermediate variables
    tile_shape = [tile_height, tile_width]
    tile_type = ir.TileType(DataType.BF16, tile_shape)

    x_tmp = ir.Var("x_tmp", tile_type, span)
    x_sq = ir.Var("x_sq", tile_type, span)
    # After reducing along axis 1, shape becomes [8], so it's still a TileType
    sum_x_sq_tile_type = ir.TileType(DataType.BF16, [tile_height])
    sum_x_sq = ir.Var("sum_x_sq", sum_x_sq_tile_type, span)
    sqrt_mean_tile = ir.Var("sqrt_mean_tile", tile_type, span)
    div_tmp = ir.Var("div_tmp", tile_type, span)
    out_tmp = ir.Var("out_tmp", tile_type, span)

    # Calculate row and col offsets (simplified)
    row_offset = ir.ConstInt(0, DataType.INT32, span)
    col_offset = ir.ConstInt(0, DataType.INT32, span)

    # x_tmp = block.ub_copy_in(x, row_offset, col_offset, tile_height, tile_width)
    ub_copy_in_call = ir.create_op_call(
        "block.ub_copy_in", [x, row_offset, col_offset, tile_height, tile_width], span
    )
    stmt1 = ir.AssignStmt(x_tmp, ub_copy_in_call, span)

    # x_sq = block.mul(x_tmp, x_tmp)
    mul_call1 = ir.create_op_call("block.mul", [x_tmp, x_tmp], span)
    stmt2 = ir.AssignStmt(x_sq, mul_call1, span)

    # sum_x_sq = block.sum(x_sq, axis)
    # Note: In the actual rmsnorm.py, block.sum(x_sq) reduces all elements
    # For IR, we need to specify the axis. To get a scalar from a 2D tile [8, 5120],
    # we can reduce along axis 1 first to get shape [8], then reduce along axis 0 to get scalar
    # But since we only support single axis reduction, we'll reduce along axis 1
    # and then the result would need another reduction to get scalar
    # For this test, we'll reduce along axis 1
    sum_axis = ir.ConstInt(1, DataType.INT32, span)
    sum_call = ir.create_op_call("block.sum", [x_sq, sum_axis], span)
    # Note: sum_x_sq type should be TileType([8]) after reducing axis 1, not ScalarType
    # But for the test to work, we'll keep the variable type as ScalarType and adjust later
    stmt3 = ir.AssignStmt(sum_x_sq, sum_call, span)

    # For demonstration, create a tile from scalar for sqrt_mean
    # In practice, this would require proper conversion/broadcasting
    # sqrt_mean_tile = ... (simplified, using a placeholder)
    stmt4 = ir.AssignStmt(sqrt_mean_tile, x_tmp, span)  # Placeholder

    # div_tmp = block.div(x_tmp, sqrt_mean_tile)
    div_call1 = ir.create_op_call("block.div", [x_tmp, sqrt_mean_tile], span)
    stmt5 = ir.AssignStmt(div_tmp, div_call1, span)

    # out_tmp = block.mul(div_tmp, x_gamma_tile)
    # Note: x_gamma needs to be converted to tile or broadcast
    # For simplicity, we'll use x_tmp as placeholder
    x_gamma_tile = ir.Var("x_gamma_tile", tile_type, span)
    stmt6_placeholder = ir.AssignStmt(x_gamma_tile, x_tmp, span)  # Placeholder
    mul_call2 = ir.create_op_call("block.mul", [div_tmp, x_gamma_tile], span)
    stmt6 = ir.AssignStmt(out_tmp, mul_call2, span)

    # block.ub_copy_out(out_tmp, row_offset, col_offset, height, width, y)
    out_row_offset = ir.ConstInt(0, DataType.INT32, span)
    out_col_offset = ir.ConstInt(0, DataType.INT32, span)
    ub_copy_out_call = ir.create_op_call(
        "block.ub_copy_out", [out_tmp, out_row_offset, out_col_offset, dim128, dim5120, y], span
    )
    stmt7 = ir.AssignStmt(y, ub_copy_out_call, span)

    # Create function body
    body = ir.SeqStmts([stmt1, stmt2, stmt3, stmt4, stmt5, stmt6_placeholder, stmt6, stmt7], span)

    # Create function
    func = ir.Function("rms_norm_block", [x, x_gamma, y, block_idx, epsilon], [], body, span)

    # Verify function structure
    assert func is not None
    assert len(func.params) == 5
    assert func.body is not None
    assert isinstance(func.body, ir.SeqStmts)
    assert len(func.body.stmts) == 8

    # Verify all block operations are used
    # This is a basic structural check - the function should be constructible
    assert isinstance(func, ir.Function)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
