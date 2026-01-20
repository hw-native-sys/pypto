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


def test_block_ops_pipe():
    """Test that block operators have the correct pipe property."""

    # MTE2 ops
    op = ir.get_op("block.ub_copy_in")
    assert op.pipe == ir.PipeType.MTE2

    # MTE3 ops
    op = ir.get_op("block.ub_copy_out")
    assert op.pipe == ir.PipeType.MTE3

    # Vector ops
    vector_ops = ["block.mul", "block.add", "block.div", "block.sum", "block.sqrt"]
    for op_name in vector_ops:
        op = ir.get_op(op_name)
        assert op.pipe == ir.PipeType.V

    # Scalar ops
    op = ir.get_op("block.get_block_idx")
    assert op.pipe == ir.PipeType.S


def test_block_mul():
    """Test block.mul operation."""
    span = ir.Span.unknown()

    # Create two tiles
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tile_shape = [dim8, dim5120]
    tile_type = ir.TileType(tile_shape, DataType.BF16)

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
    tile_type = ir.TileType(tile_shape, DataType.FP32)
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
    tile_type = ir.TileType(tile_shape, DataType.FP32)
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
    tile_type = ir.TileType(tile_shape, DataType.FP32)
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
    tensor_type = ir.TensorType(tensor_shape, DataType.BF16)
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
    tile_type = ir.TileType(tile_shape, DataType.BF16)
    var_tile = ir.Var("tile", tile_type, span)

    # Create output tensor
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_shape = [dim128, dim5120]
    tensor_type = ir.TensorType(tensor_shape, DataType.BF16)
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

    This test constructs the rms_norm_block function from examples/block-level/rmsnorm.py
    using IR operations, matching the actual implementation logic.
    """
    span = ir.Span.unknown()

    # Define constants matching rmsnorm.py
    cut_height = ir.ConstInt(128, DataType.INT32, span)
    tile_height = ir.ConstInt(8, DataType.INT32, span)
    tile_width = ir.ConstInt(5120, DataType.INT32, span)

    # Create function parameters
    # x: Tensor, x_gamma: Tensor, y: Tensor, block_idx: Scalar, epsilon: Scalar
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim5120 = ir.ConstInt(5120, DataType.INT32, span)
    tensor_shape = [dim128, dim5120]

    x_type = ir.TensorType(tensor_shape, DataType.BF16)
    x = ir.Var("x", x_type, span)

    # x_gamma is reshaped to [1, 5120] in rms_norm, so we use 2D tensor
    x_gamma_shape = [ir.ConstInt(1, DataType.INT32, span), dim5120]
    x_gamma_type = ir.TensorType(x_gamma_shape, DataType.BF16)
    x_gamma = ir.Var("x_gamma", x_gamma_type, span)

    y_type = ir.TensorType(tensor_shape, DataType.BF16)
    y = ir.Var("y", y_type, span)

    block_idx = ir.Var("block_idx", ir.ScalarType(DataType.INT32), span)
    epsilon = ir.Var("epsilon", ir.ScalarType(DataType.FP32), span)

    # Create intermediate variables
    tile_shape = [tile_height, tile_width]  # [8, 5120]
    tile_type = ir.TileType(tile_shape, DataType.BF16)
    x_gamma_tile_shape = [ir.ConstInt(1, DataType.INT32, span), tile_width]  # [1, 5120]
    x_gamma_tile_type = ir.TileType(x_gamma_tile_shape, DataType.BF16)

    x_tmp = ir.Var("x_tmp", tile_type, span)
    x_sq = ir.Var("x_sq", tile_type, span)
    # sum_x_sq = block.sum(x_sq, -1, keepdim=True) reduces along axis 1 with keepdim, result is [8, 1]
    sum_x_sq_shape = [tile_height, ir.ConstInt(1, DataType.INT32, span)]  # [8, 1]
    sum_x_sq_type = ir.TileType(sum_x_sq_shape, DataType.BF16)
    sum_x_sq = ir.Var("sum_x_sq", sum_x_sq_type, span)  # After reducing axis 1 with keepdim: [8, 1]
    # mean_x_sq, mean_x_sq_eps, sqrt_mean are all tiles (block operations return tiles)
    # mean_x_sq has shape [8, 1] (BF16)
    # mean_x_sq_eps and sqrt_mean have shape [8, 1] (FP32) after type promotion with epsilon
    mean_x_sq = ir.Var("mean_x_sq", sum_x_sq_type, span)  # [8, 1] BF16
    mean_x_sq_eps_type = ir.TileType(sum_x_sq_shape, DataType.FP32)  # [8, 1] FP32
    mean_x_sq_eps = ir.Var("mean_x_sq_eps", mean_x_sq_eps_type, span)  # [8, 1] FP32
    sqrt_mean = ir.Var("sqrt_mean", mean_x_sq_eps_type, span)  # [8, 1] FP32
    div_tmp = ir.Var("div_tmp", tile_type, span)
    x_gamma_tile = ir.Var("x_gamma_tile", x_gamma_tile_type, span)  # [1, 5120]
    out_tmp = ir.Var("out_tmp", tile_type, span)
    y1 = ir.Var("y1", y_type, span)  # Result of ub_copy_out

    # Calculate row_size = cut[0] / tile[0]
    row_size = ir.Var("row_size", ir.ScalarType(DataType.INT32), span)
    row_size_calc = ir.FloorDiv(cut_height, tile_height, DataType.INT32, span)
    stmt_row_size = ir.AssignStmt(row_size, row_size_calc, span)

    # Get block_idx from get_block_idx()
    current_block_idx = ir.Var("current_block_idx", ir.ScalarType(DataType.INT32), span)
    get_block_idx_call = ir.create_op_call("block.get_block_idx", [], span)
    stmt_get_block_idx = ir.AssignStmt(current_block_idx, get_block_idx_call, span)

    # Calculate row = get_block_idx() / row_size
    row = ir.Var("row", ir.ScalarType(DataType.INT32), span)
    row_calc = ir.FloorDiv(current_block_idx, row_size, DataType.INT32, span)
    stmt_row = ir.AssignStmt(row, row_calc, span)

    # Calculate col = get_block_idx() % row_size
    col = ir.Var("col", ir.ScalarType(DataType.INT32), span)
    col_calc = ir.FloorMod(current_block_idx, row_size, DataType.INT32, span)
    stmt_col = ir.AssignStmt(col, col_calc, span)

    # Calculate row_offset = row * tile[0] + block_idx
    row_tile = ir.Var("row_tile", ir.ScalarType(DataType.INT32), span)
    row_tile_calc = ir.Mul(row, tile_height, DataType.INT32, span)
    stmt_row_tile = ir.AssignStmt(row_tile, row_tile_calc, span)
    row_offset = ir.Var("row_offset", ir.ScalarType(DataType.INT32), span)
    row_offset_calc = ir.Add(row_tile, block_idx, DataType.INT32, span)
    stmt_row_offset = ir.AssignStmt(row_offset, row_offset_calc, span)

    # Calculate col_offset = col * tile[1]
    col_offset = ir.Var("col_offset", ir.ScalarType(DataType.INT32), span)
    col_offset_calc = ir.Mul(col, tile_width, DataType.INT32, span)
    stmt_col_offset = ir.AssignStmt(col_offset, col_offset_calc, span)

    # x_tmp = block.ub_copy_in(x, row_offset, col_offset, tile_height, tile_width)
    ub_copy_in_call = ir.create_op_call(
        "block.ub_copy_in", [x, row_offset, col_offset, tile_height, tile_width], span
    )
    stmt1 = ir.AssignStmt(x_tmp, ub_copy_in_call, span)

    # x_sq = block.mul(x_tmp, x_tmp)
    mul_call1 = ir.create_op_call("block.mul", [x_tmp, x_tmp], span)
    stmt2 = ir.AssignStmt(x_sq, mul_call1, span)

    sum_axis1 = ir.ConstInt(-1, DataType.INT32, span)
    keepdim = ir.ConstInt(1, DataType.INT32, span)
    sum_call = ir.create_op_call("block.sum", [x_sq, sum_axis1, keepdim], span)
    stmt3 = ir.AssignStmt(sum_x_sq, sum_call, span)

    # mean_x_sq = block.div(sum_x_sq, x_tmp.size)
    # sum_x_sq is tile [8, 1], x_tmp.size is scalar, result is tile [8, 1]
    tile_size = ir.ConstInt(8 * 5120, DataType.INT32, span)
    mean_x_sq_calc = ir.create_op_call("block.div", [sum_x_sq, tile_size], span)
    stmt5 = ir.AssignStmt(mean_x_sq, mean_x_sq_calc, span)

    # mean_x_sq_eps = block.add(mean_x_sq, epsilon)
    # block.add(tile, scalar) returns tile with same shape
    mean_x_sq_eps_calc = ir.create_op_call("block.add", [mean_x_sq, epsilon], span)
    stmt6 = ir.AssignStmt(mean_x_sq_eps, mean_x_sq_eps_calc, span)

    # sqrt_mean = block.sqrt(mean_x_sq_eps)
    # block.sqrt(tile) returns tile with same shape [8, 1]
    sqrt_call = ir.create_op_call("block.sqrt", [mean_x_sq_eps], span)
    stmt7 = ir.AssignStmt(sqrt_mean, sqrt_call, span)

    # div_tmp = block.div(x_tmp, sqrt_mean)
    # x_tmp is [8, 5120], sqrt_mean is [8, 1], broadcasting results in [8, 5120]
    div_call1 = ir.create_op_call("block.div", [x_tmp, sqrt_mean], span)
    stmt8 = ir.AssignStmt(div_tmp, div_call1, span)

    # x_gamma_tile = block.ub_copy_in(x_gamma, 0, 0, 1, tile_width)
    # Load x_gamma [1, 5120] as a tile [1, 5120]
    x_gamma_height = ir.ConstInt(1, DataType.INT32, span)
    x_gamma_row_offset = ir.ConstInt(0, DataType.INT32, span)
    x_gamma_col_offset = ir.ConstInt(0, DataType.INT32, span)
    ub_copy_in_gamma = ir.create_op_call(
        "block.ub_copy_in",
        [x_gamma, x_gamma_row_offset, x_gamma_col_offset, x_gamma_height, tile_width],
        span,
    )
    stmt9 = ir.AssignStmt(x_gamma_tile, ub_copy_in_gamma, span)

    # out_tmp = block.mul(div_tmp, x_gamma_tile) - broadcasting [8, 5120] * [1, 5120]
    mul_call2 = ir.create_op_call("block.mul", [div_tmp, x_gamma_tile], span)
    stmt10 = ir.AssignStmt(out_tmp, mul_call2, span)

    # Calculate output row_offset = block_idx + row * tile[0]
    out_row_offset = ir.Var("out_row_offset", ir.ScalarType(DataType.INT32), span)
    out_row_offset_calc = ir.Add(block_idx, row_tile, DataType.INT32, span)
    stmt_out_row_offset = ir.AssignStmt(out_row_offset, out_row_offset_calc, span)

    # block.ub_copy_out(out_tmp, out_row_offset, 0, y.shape[0], y.shape[1], y)
    out_col_offset = ir.ConstInt(0, DataType.INT32, span)
    ub_copy_out_call = ir.create_op_call(
        "block.ub_copy_out", [out_tmp, out_row_offset, out_col_offset, dim128, dim5120, y], span
    )
    stmt11 = ir.AssignStmt(y1, ub_copy_out_call, span)

    # Create function body with all statements in correct order
    body = ir.SeqStmts(
        [
            stmt_row_size,
            stmt_get_block_idx,
            stmt_row,
            stmt_col,
            stmt_row_tile,
            stmt_row_offset,
            stmt_col_offset,
            stmt1,  # ub_copy_in x
            stmt2,  # mul x_sq
            stmt3,  # sum axis 1 with keepdim
            stmt5,  # div mean
            stmt6,  # add epsilon
            stmt7,  # sqrt (simplified placeholder)
            stmt8,  # div x_tmp / sqrt_mean
            stmt9,  # ub_copy_in x_gamma
            stmt10,  # mul div_tmp * x_gamma_tile
            stmt_out_row_offset,
            stmt11,  # ub_copy_out
        ],
        span,
    )

    # Create function
    func = ir.Function("rms_norm_block", [x, x_gamma, y, block_idx, epsilon], [], body, span)

    # Print the generated IR using PythonPrinter (via __str__ method)
    ir_str = str(func)
    print("\n" + "=" * 80)
    print("Generated IR for rms_norm_block function:")
    print("=" * 80)
    print(ir_str)
    print("=" * 80 + "\n")

    # Verify function structure
    assert func is not None
    assert len(func.params) == 5
    assert func.body is not None
    assert isinstance(func.body, ir.SeqStmts)
    assert len(func.body.stmts) == 18

    # Verify all block operations are used
    # This is a basic structural check - the function should be constructible
    assert isinstance(func, ir.Function)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
