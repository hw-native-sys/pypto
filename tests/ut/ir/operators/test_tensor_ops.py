# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comprehensive tests for tensor operations.

Tests cover:
- Memory operations (create, slice, assemble)
- Matrix multiplication (matmul)
- Reduction operations (row_max, row_sum)
- Unary operations (exp, cast)
- Binary operations (maximum)
- Python helper functions
"""

import pytest
from pypto import DataType, ir
from pypto.ir.op import tensor


def test_tensor_create():
    """Test tensor.create operation."""
    # Create a 2D tensor [4, 8] with FP32
    call = ir.op.tensor.create([4, 8], DataType.FP32)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.create"

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_slice():
    """Test tensor.slice operation."""
    span = ir.Span.unknown()

    # Create a tensor variable [16, 32]
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim16, dim32], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Create a slice [8, 16]
    call = ir.op.tensor.slice(tensor_var, [8, 16], [0, 0])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.slice"

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16


def test_tensor_matmul():
    """Test tensor.matmul operation."""
    span = ir.Span.unknown()

    # Create two tensors [4, 8] and [8, 16]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)

    lhs_type = ir.TensorType([dim4, dim8], DataType.FP32)
    rhs_type = ir.TensorType([dim8, dim16], DataType.FP32)

    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)

    # Perform matmul
    call = ir.op.tensor.matmul(lhs, rhs, out_dtype=DataType.FP32)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.matmul"

    # Check result type - should be [4, 16]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_matmul_with_transpose():
    """Test tensor.matmul with transpose flags."""
    span = ir.Span.unknown()

    # Create tensors
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)

    lhs_type = ir.TensorType([dim8, dim4], DataType.FP16)  # [8, 4]
    rhs_type = ir.TensorType([dim8, dim4], DataType.FP16)  # [8, 4]

    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)

    # Transpose lhs: [8, 4]^T x [8, 4] -> [4, 4]
    call = ir.op.tensor.matmul(lhs, rhs, out_dtype=DataType.FP16, a_trans=True, b_trans=False)

    assert isinstance(call, ir.Call)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_matmul_acc():
    """Test tensor.matmul_acc operation."""
    span = ir.Span.unknown()

    # acc[4, 16] FP32 += lhs[4, 8] FP32 @ rhs[8, 16] FP32
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)

    acc_type = ir.TensorType([dim4, dim16], DataType.FP32)
    lhs_type = ir.TensorType([dim4, dim8], DataType.FP32)
    rhs_type = ir.TensorType([dim8, dim16], DataType.FP32)

    acc = ir.Var("acc", acc_type, span)
    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)

    call = ir.op.tensor.matmul_acc(acc, lhs, rhs)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.matmul_acc"

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_matmul_acc_with_transpose():
    """Test tensor.matmul_acc with a_trans=True."""
    span = ir.Span.unknown()

    # acc[4, 16] += lhs[8, 4]^T @ rhs[8, 16]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)

    acc_type = ir.TensorType([dim4, dim16], DataType.FP32)
    lhs_type = ir.TensorType([dim8, dim4], DataType.FP32)  # [8, 4], transposed to [4, 8]
    rhs_type = ir.TensorType([dim8, dim16], DataType.FP32)

    acc = ir.Var("acc", acc_type, span)
    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)

    call = ir.op.tensor.matmul_acc(acc, lhs, rhs, a_trans=True)

    assert isinstance(call, ir.Call)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_row_max():
    """Test tensor.row_max reduction."""
    span = ir.Span.unknown()

    # Create a tensor [64, 128]
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Row max reduction (reduce last axis)
    call = ir.op.tensor.row_max(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_max"

    # Check result type - should be [64, 1]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_row_sum():
    """Test tensor.row_sum reduction."""
    span = ir.Span.unknown()

    # Create a tensor [64, 128]
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Row sum reduction (reduce last axis)
    call = ir.op.tensor.row_sum(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_sum"

    # Check result type - should be [64, 1]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_exp():
    """Test tensor.exp operation."""
    span = ir.Span.unknown()

    # Create a tensor [64, 128]
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Apply exp
    call = ir.op.tensor.exp(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.exp"

    # Check result type - should preserve shape and dtype
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


# =============================================================================
# Tensor neg tests
# =============================================================================


def test_tensor_neg():
    """Test tensor.neg operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.neg(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.neg"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_neg_int_dtype():
    """Test tensor.neg preserves integer dtype (no float promotion)."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.INT32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.neg(tensor_var)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.INT32


# =============================================================================
# Tensor recip tests
# =============================================================================


def test_tensor_recip():
    """Test tensor.recip operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.recip(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.recip"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_recip_int_promotes_to_fp32():
    """Test tensor.recip promotes integer dtype to FP32."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.INT32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.recip(tensor_var)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_sqrt():
    """Test tensor.sqrt operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.sqrt(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.sqrt"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_sqrt_int_promotion():
    """Test tensor.sqrt promotes integer dtype to FP32."""
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.INT32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.sqrt(tensor_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_sqrt_wrong_type():
    """Test tensor.sqrt rejects non-TensorType inputs."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64, 128], DataType.FP16)
    tile_var = ir.Var("t", tile_type, span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.op.tensor.sqrt(tile_var)


def test_tensor_rsqrt():
    """Test tensor.rsqrt operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.rsqrt(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.rsqrt"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_rsqrt_int_promotion():
    """Test tensor.rsqrt promotes integer dtype to FP32."""
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.INT32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.rsqrt(tensor_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_rsqrt_wrong_type():
    """Test tensor.rsqrt rejects non-TensorType inputs."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64, 128], DataType.FP16)
    tile_var = ir.Var("t", tile_type, span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.op.tensor.rsqrt(tile_var)


def test_tensor_cast():
    """Test tensor.cast operation."""
    span = ir.Span.unknown()

    # Create a FP16 tensor
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Cast to FP32
    call = ir.op.tensor.cast(tensor_var, DataType.FP32)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.cast"

    # Check result type - should preserve shape but change dtype
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_assemble():
    """Test tensor.assemble operation."""
    span = ir.Span.unknown()

    # Create target and source tensors
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    target_type = ir.TensorType([dim64, dim128], DataType.FP32)
    source_type = ir.TensorType([dim64, dim128], DataType.FP32)

    target = ir.Var("target", target_type, span)
    source = ir.Var("source", source_type, span)

    # Assemble at offset [0, 0]
    call = ir.op.tensor.assemble(target, source, [0, 0])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.assemble"

    # Check result type - should be target type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_row_expand_mul():
    """Test tensor.row_expand_mul operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand_mul(tensor_var, row_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_expand_mul"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_row_expand_mul_dtype_promotion():
    """Test tensor.row_expand_mul promotes data types."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand_mul(tensor_var, row_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_row_expand_mul_wrong_type():
    """Test tensor.row_expand_mul rejects non-TensorType inputs."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64, 128], DataType.FP16)
    tile_var = ir.Var("t", tile_type, span)

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    row_var = ir.Var("rv", row_type, span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.op.tensor.row_expand_mul(tile_var, row_var)


def test_tensor_row_expand_div():
    """Test tensor.row_expand_div operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand_div(tensor_var, row_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_expand_div"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_row_expand_div_dtype_promotion():
    """Test tensor.row_expand_div promotes data types."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand_div(tensor_var, row_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_row_expand_div_wrong_type():
    """Test tensor.row_expand_div rejects non-TensorType inputs."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64, 128], DataType.FP16)
    tile_var = ir.Var("t", tile_type, span)

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    row_var = ir.Var("rv", row_type, span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.op.tensor.row_expand_div(tile_var, row_var)


def test_tensor_col_expand_mul():
    """Test tensor.col_expand_mul operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    col_type = ir.TensorType([dim1, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    col_var = ir.Var("cv", col_type, span)

    call = ir.op.tensor.col_expand_mul(tensor_var, col_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.col_expand_mul"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_col_expand_mul_dtype_promotion():
    """Test tensor.col_expand_mul promotes data types."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    col_type = ir.TensorType([dim1, dim128], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    col_var = ir.Var("cv", col_type, span)

    call = ir.op.tensor.col_expand_mul(tensor_var, col_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_col_expand_mul_wrong_type():
    """Test tensor.col_expand_mul rejects non-TensorType inputs."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64, 128], DataType.FP16)
    tile_var = ir.Var("t", tile_type, span)

    dim1 = ir.ConstInt(1, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    col_type = ir.TensorType([dim1, dim128], DataType.FP16)
    col_var = ir.Var("cv", col_type, span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.op.tensor.col_expand_mul(tile_var, col_var)


def test_tensor_maximum():
    """Test tensor.maximum operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    type_a = ir.TensorType([dim64, dim1], DataType.FP32)
    type_b = ir.TensorType([dim64, dim1], DataType.FP32)

    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_b, span)

    # Element-wise maximum
    call = ir.op.tensor.maximum(var_a, var_b)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.maximum"

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_mul():
    """Test tensor.mul operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Create a second tensor for multiplication (broadcasting: scalar tensor)
    scalar_tensor_type = ir.TensorType([], DataType.FP32)  # 0-D tensor (scalar)
    scalar_tensor_var = ir.Var("s", scalar_tensor_type, span)
    call = ir.op.tensor.mul(tensor_var, scalar_tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.mul"


def test_tensor_add():
    """Test tensor.add operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Add
    call = ir.op.tensor.add(var_a, var_b)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.add"


def test_tensor_sub():
    """Test tensor.sub operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Subtract
    call = ir.op.tensor.sub(var_a, var_b)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.sub"


def test_tensor_div():
    """Test tensor.div operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Divide
    call = ir.op.tensor.div(var_a, var_b)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.div"


def test_const_float():
    """Test ConstFloat expression creation and usage."""
    span = ir.Span.unknown()

    # Create a ConstFloat with FP32
    const_float = ir.ConstFloat(3.14, DataType.FP32, span)
    assert isinstance(const_float, ir.ConstFloat)
    assert const_float.value == 3.14
    assert const_float.dtype == DataType.FP32

    # Create a ConstFloat with FP16
    const_float_fp16 = ir.ConstFloat(2.718, DataType.FP16, span)
    assert isinstance(const_float_fp16, ir.ConstFloat)
    assert const_float_fp16.value == 2.718
    assert const_float_fp16.dtype == DataType.FP16

    # Test with negative value
    const_float_neg = ir.ConstFloat(-1.5, DataType.FP32, span)
    assert const_float_neg.value == -1.5

    # Test with zero
    const_float_zero = ir.ConstFloat(0.0, DataType.FP32, span)
    assert const_float_zero.value == 0.0


def test_tensor_read():
    """Test tensor.read operation."""
    span = ir.Span.unknown()

    # Create a 2D tensor [4, 8] with FP32
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim4, dim8], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    # Read at indices [2, 3]
    call = ir.op.tensor.read(tensor_var, [2, 3])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.read"

    # Result should be ScalarType with tensor's element dtype
    result_type = call.type
    assert isinstance(result_type, ir.ScalarType)
    assert result_type.dtype == DataType.FP32


def test_tensor_read_with_expr_indices():
    """Test tensor.read with expression indices."""
    span = ir.Span.unknown()

    # Create a 1D tensor [64] with FP16
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Read at a variable index
    idx_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    call = ir.op.tensor.read(tensor_var, [idx_var])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.read"
    result_type = call.type
    assert isinstance(result_type, ir.ScalarType)
    assert result_type.dtype == DataType.FP16


def test_tensor_dim():
    """Test tensor.dim operation extracts shape dimension as scalar."""
    span = ir.Span.unknown()

    # Create a 3D tensor [4, 8, 16]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    tensor_type = ir.TensorType([dim4, dim8, dim16], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    # Extract dimension at axis 1
    call = ir.op.tensor.dim(tensor_var, 1)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.dim"

    # Result should be ScalarType(INDEX) — tensor.dim returns machine-word index type
    result_type = call.type
    assert isinstance(result_type, ir.ScalarType)
    assert result_type.dtype == DataType.INDEX


def test_tensor_dim_negative_axis():
    """Test tensor.dim with negative axis indexing."""
    span = ir.Span.unknown()

    # Create a 2D tensor [32, 64]
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    tensor_type = ir.TensorType([dim32, dim64], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Extract last dimension using negative index
    call = ir.op.tensor.dim(tensor_var, -1)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.dim"
    result_type = call.type
    assert isinstance(result_type, ir.ScalarType)
    assert result_type.dtype == DataType.INDEX


def test_tensor_create_dynamic_shape():
    """Test tensor.create with dynamic (Expr) shape dimensions."""
    span = ir.Span.unknown()

    # Create with a mix of int and Expr dimensions
    dim_n = ir.Var("n", ir.ScalarType(DataType.UINT64), span)
    call = ir.op.tensor.create([dim_n, 128], DataType.FP32)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.create"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_operator_registration():
    """Test that all new operators are registered."""
    # Check that our new operators are registered
    assert ir.is_op_registered("tensor.create")
    assert ir.is_op_registered("tensor.read")
    assert ir.is_op_registered("tensor.write")
    assert ir.is_op_registered("tensor.slice")
    assert ir.is_op_registered("tensor.matmul")
    assert ir.is_op_registered("tensor.row_max")
    assert ir.is_op_registered("tensor.row_sum")
    assert ir.is_op_registered("tensor.exp")
    assert ir.is_op_registered("tensor.sqrt")
    assert ir.is_op_registered("tensor.rsqrt")
    assert ir.is_op_registered("tensor.cast")
    assert ir.is_op_registered("tensor.assemble")
    assert ir.is_op_registered("tensor.fillpad")
    assert ir.is_op_registered("tensor.maximum")
    assert ir.is_op_registered("tensor.row_expand_mul")
    assert ir.is_op_registered("tensor.row_expand_div")
    assert ir.is_op_registered("tensor.col_expand_mul")
    assert ir.is_op_registered("tensor.dim")
    # Check transform operators
    assert ir.is_op_registered("tensor.reshape")
    assert ir.is_op_registered("tensor.transpose")


def test_tensor_reshape():
    """Test tensor.reshape operation."""
    span = ir.Span.unknown()

    # Create a tensor variable [4, 8]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim4, dim8], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    # Reshape to [32] (flatten)
    call = ir.op.tensor.reshape(tensor_var, [32])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.reshape"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 1

    # Reshape to [2, 16]
    call2 = ir.op.tensor.reshape(tensor_var, [2, 16])
    result_type2 = call2.type
    assert isinstance(result_type2, ir.TensorType)
    assert len(result_type2.shape) == 2


def test_tensor_reshape_dynamic():
    """Test tensor.reshape with dynamic shapes."""
    span = ir.Span.unknown()

    # Create a tensor with dynamic dimensions
    dim_n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
    dim_m = ir.Var("m", ir.ScalarType(DataType.INT64), span)
    tensor_type = ir.TensorType([dim_n, dim_m], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Reshape with dynamic shape (cannot verify element count at compile time)
    dim_k = ir.Var("k", ir.ScalarType(DataType.INT64), span)
    call = ir.op.tensor.reshape(tensor_var, [dim_k])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.reshape"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_transpose():
    """Test tensor.transpose operation."""
    span = ir.Span.unknown()

    # Create a 3D tensor [2, 3, 4]
    dim2 = ir.ConstInt(2, DataType.INT32, span)
    dim3 = ir.ConstInt(3, DataType.INT32, span)
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    tensor_type = ir.TensorType([dim2, dim3, dim4], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    # Transpose by swapping axis 0 and 2: [2, 3, 4] -> [4, 3, 2]
    call = ir.op.tensor.transpose(tensor_var, 0, 2)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.transpose"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 3


def test_tensor_transpose_negative_axis():
    """Test tensor.transpose with negative axis indices."""
    span = ir.Span.unknown()

    # Create a 2D tensor [8, 16]
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8, dim16], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Transpose using negative indices: axis1=-2 (0), axis2=-1 (1)
    # [8, 16] -> [16, 8]
    call = ir.op.tensor.transpose(tensor_var, -2, -1)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.transpose"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_get_new_ops():
    """Test getting new operator instances."""
    matmul_op = ir.get_op("tensor.matmul")
    assert matmul_op.name == "tensor.matmul"

    exp_op = ir.get_op("tensor.exp")
    assert exp_op.name == "tensor.exp"

    cast_op = ir.get_op("tensor.cast")
    assert cast_op.name == "tensor.cast"


def test_tensor_slice_with_valid_shape():
    """Test tensor.slice with valid_shape parameter."""
    span = ir.Span.unknown()
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim16, dim32], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.slice(tensor_var, [8, 16], [0, 0], valid_shape=[4, 8])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.slice"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(call.args) == 4
    assert result_type.tensor_view is not None
    assert len(result_type.tensor_view.valid_shape) == 2


def test_tensor_fillpad_clears_valid_shape():
    """Test tensor.fillpad materializes a full-valid tensor view."""
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    tensor_view = ir.TensorView(
        stride=[],
        layout=ir.TensorLayout.ND,
        valid_shape=[dim8, ir.ConstInt(4, DataType.INT32, span)],
    )
    tensor_type = ir.TensorType([dim8, dim16], DataType.FP32, None, tensor_view)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.fillpad(tensor_var, pad_value=ir.PadValue.min)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.fillpad"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert result_type.tensor_view is not None
    assert result_type.tensor_view.layout == ir.TensorLayout.ND
    assert len(result_type.tensor_view.valid_shape) == 2
    assert result_type.tensor_view.valid_shape[0] == dim8
    assert result_type.tensor_view.valid_shape[1] == dim16


def test_tensor_reshape_with_valid_shape():
    """Test tensor.reshape with valid_shape parameter."""
    span = ir.Span.unknown()
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim4, dim8], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.reshape(tensor_var, [32], valid_shape=[16])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.reshape"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(call.args) == 3
    assert result_type.tensor_view is not None
    assert len(result_type.tensor_view.valid_shape) == 1


def test_tensor_transpose_with_valid_shape():
    """Test tensor.transpose with valid_shape parameter."""
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8, dim16], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.transpose(tensor_var, 0, 1, valid_shape=[16, 8])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.transpose"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(call.args) == 4
    assert result_type.tensor_view is not None
    assert len(result_type.tensor_view.valid_shape) == 2


class TestTensorScalarMemoryOps:
    """Test suite for tensor-level scalar memory operations (tensor.read / tensor.write)."""

    def test_read_write_exported(self):
        """Test tensor.read and tensor.write are exported from tensor_ops."""
        assert hasattr(tensor, "read")
        assert hasattr(tensor, "write")

    def test_read_return_type(self):
        """Test tensor.read returns a Call with ScalarType matching tensor dtype."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(64, DataType.INT32, span)
        tensor_type = ir.TensorType([dim], DataType.FP32)
        tensor_var = ir.Var("t", tensor_type, span)
        idx = ir.ConstInt(0, DataType.INT64, span)

        call = tensor.read(tensor_var, [idx])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tensor.read"
        assert isinstance(call.type, ir.ScalarType)
        assert call.type.dtype == DataType.FP32

    def test_read_2d(self):
        """Test tensor.read with 2D indices."""
        span = ir.Span.unknown()
        d0 = ir.ConstInt(4, DataType.INT32, span)
        d1 = ir.ConstInt(8, DataType.INT32, span)
        tensor_type = ir.TensorType([d0, d1], DataType.FP32)
        tensor_var = ir.Var("t", tensor_type, span)
        i = ir.ConstInt(1, DataType.INT64, span)
        j = ir.ConstInt(3, DataType.INT64, span)

        call = tensor.read(tensor_var, [i, j])

        assert call.op.name == "tensor.read"
        assert isinstance(call.type, ir.ScalarType)
        assert call.type.dtype == DataType.FP32

    def test_write_basic(self):
        """Test tensor.write returns a Call with correct op name."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(64, DataType.INT32, span)
        tensor_type = ir.TensorType([dim], DataType.FP32)
        tensor_var = ir.Var("t", tensor_type, span)
        value = ir.Var("v", ir.ScalarType(DataType.FP32), span)
        idx = ir.ConstInt(0, DataType.INT64, span)

        call = tensor.write(tensor_var, [idx], value)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tensor.write"

    def test_write_2d(self):
        """Test tensor.write with 2D indices."""
        span = ir.Span.unknown()
        d0 = ir.ConstInt(4, DataType.INT32, span)
        d1 = ir.ConstInt(8, DataType.INT32, span)
        tensor_type = ir.TensorType([d0, d1], DataType.FP32)
        tensor_var = ir.Var("t", tensor_type, span)
        value = ir.Var("v", ir.ScalarType(DataType.FP32), span)
        i = ir.ConstInt(1, DataType.INT64, span)
        j = ir.ConstInt(3, DataType.INT64, span)

        call = tensor.write(tensor_var, [i, j], value)

        assert call.op.name == "tensor.write"

    def test_read_type_mismatch(self):
        """Test tensor.read with wrong argument types raises error."""
        span = ir.Span.unknown()
        # First arg must be TensorType, not TileType
        tile_type = ir.TileType([32, 32], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)
        idx = ir.ConstInt(0, DataType.INT64, span)

        with pytest.raises(ValueError, match="TensorType"):
            tensor.read(tile_var, [idx])


# =============================================================================
# Tensor row_min tests
# =============================================================================


def test_tensor_row_min():
    """Test tensor.row_min reduction."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.row_min(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_min"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


# =============================================================================
# Tensor row_expand tests
# =============================================================================


def test_tensor_row_expand():
    """Test tensor.row_expand operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.row_expand(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_expand"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


# =============================================================================
# Tensor row_expand_add tests
# =============================================================================


def test_tensor_row_expand_add():
    """Test tensor.row_expand_add operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand_add(tensor_var, row_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_expand_add"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_row_expand_add_dtype_promotion():
    """Test tensor.row_expand_add promotes data types."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand_add(tensor_var, row_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_row_expand_add_wrong_type():
    """Test tensor.row_expand_add rejects non-TensorType inputs."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64, 128], DataType.FP16)
    tile_var = ir.Var("t", tile_type, span)

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    row_var = ir.Var("rv", row_type, span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.op.tensor.row_expand_add(tile_var, row_var)


# =============================================================================
# Tensor row_expand_sub tests
# =============================================================================


def test_tensor_row_expand_sub():
    """Test tensor.row_expand_sub operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand_sub(tensor_var, row_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_expand_sub"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_row_expand_sub_dtype_promotion():
    """Test tensor.row_expand_sub promotes data types."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand_sub(tensor_var, row_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_row_expand_sub_wrong_type():
    """Test tensor.row_expand_sub rejects non-TensorType inputs."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64, 128], DataType.FP16)
    tile_var = ir.Var("t", tile_type, span)

    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    row_var = ir.Var("rv", row_type, span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.op.tensor.row_expand_sub(tile_var, row_var)


# =============================================================================
# Tensor col_expand tests
# =============================================================================


def test_tensor_col_expand():
    """Test tensor.col_expand operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    col_type = ir.TensorType([dim1, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    col_var = ir.Var("cv", col_type, span)

    call = ir.op.tensor.col_expand(tensor_var, col_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.col_expand"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_col_expand_dtype_promotion():
    """Test tensor.col_expand promotes data types."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    col_type = ir.TensorType([dim1, dim128], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    col_var = ir.Var("cv", col_type, span)

    call = ir.op.tensor.col_expand(tensor_var, col_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_col_expand_wrong_type():
    """Test tensor.col_expand rejects non-TensorType inputs."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([64, 128], DataType.FP16)
    tile_var = ir.Var("t", tile_type, span)

    dim1 = ir.ConstInt(1, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    col_type = ir.TensorType([dim1, dim128], DataType.FP16)
    col_var = ir.Var("cv", col_type, span)

    with pytest.raises(ValueError, match="TensorType"):
        ir.op.tensor.col_expand(tile_var, col_var)


# =============================================================================
# Tensor col_expand_div tests
# =============================================================================


def test_tensor_col_expand_div():
    """Test tensor.col_expand_div operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    col_type = ir.TensorType([dim1, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    col_var = ir.Var("cv", col_type, span)

    call = ir.op.tensor.col_expand_div(tensor_var, col_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.col_expand_div"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_col_expand_div_dtype_promotion():
    """Test tensor.col_expand_div promotes data types."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    col_type = ir.TensorType([dim1, dim128], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    col_var = ir.Var("cv", col_type, span)

    call = ir.op.tensor.col_expand_div(tensor_var, col_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


# =============================================================================
# Tensor col_expand_sub tests
# =============================================================================


def test_tensor_col_expand_sub():
    """Test tensor.col_expand_sub operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    col_type = ir.TensorType([dim1, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    col_var = ir.Var("cv", col_type, span)

    call = ir.op.tensor.col_expand_sub(tensor_var, col_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.col_expand_sub"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_col_expand_sub_dtype_promotion():
    """Test tensor.col_expand_sub promotes data types."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    col_type = ir.TensorType([dim1, dim128], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    col_var = ir.Var("cv", col_type, span)

    call = ir.op.tensor.col_expand_sub(tensor_var, col_var)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


# =============================================================================
# Tensor expands tests
# =============================================================================


def test_tensor_expands():
    """Test tensor.expands operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)

    tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    scalar_type = ir.ScalarType(DataType.FP32)
    scalar_var = ir.Var("s", scalar_type, span)

    call = ir.op.tensor.expands(tensor_var, scalar_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.expands"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_concat():
    """Test tensor.concat - column-wise concatenation."""
    span = ir.Span.unknown()
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    t0_type = ir.TensorType([dim32, dim16], DataType.FP32)
    t1_type = ir.TensorType([dim32, dim16], DataType.FP32)
    t0_var = ir.Var("src0", t0_type, span)
    t1_var = ir.Var("src1", t1_type, span)

    call = tensor.concat(t0_var, t1_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.concat"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2
    assert isinstance(result_type.shape[1], ir.ConstInt)
    assert result_type.shape[1].value == 32


def test_tensor_concat_dtype_mismatch():
    """Test tensor.concat rejects mismatched dtypes."""
    span = ir.Span.unknown()
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    t0_type = ir.TensorType([dim32, dim16], DataType.FP32)
    t1_type = ir.TensorType([dim32, dim16], DataType.FP16)
    t0_var = ir.Var("src0", t0_type, span)
    t1_var = ir.Var("src1", t1_type, span)

    with pytest.raises(ValueError, match="same dtype"):
        tensor.concat(t0_var, t1_var)


def test_tensor_concat_row_mismatch():
    """Test tensor.concat rejects mismatched row counts."""
    span = ir.Span.unknown()
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    t0_type = ir.TensorType([dim32, dim16], DataType.FP32)
    t1_type = ir.TensorType([dim8, dim16], DataType.FP32)
    t0_var = ir.Var("src0", t0_type, span)
    t1_var = ir.Var("src1", t1_type, span)

    with pytest.raises(ValueError, match="row count must match"):
        tensor.concat(t0_var, t1_var)


def test_tensor_scatter_update_2d():
    """Test tensor.scatter_update with 2D input and src."""
    span = ir.Span.unknown()

    rows = ir.ConstInt(16, DataType.INT32, span)
    d = ir.ConstInt(64, DataType.INT32, span)
    b = ir.ConstInt(2, DataType.INT32, span)
    s = ir.ConstInt(4, DataType.INT32, span)
    bs = ir.ConstInt(8, DataType.INT32, span)

    input_type = ir.TensorType([rows, d], DataType.FP16)
    index_type = ir.TensorType([b, s], DataType.INT32)
    src_type = ir.TensorType([bs, d], DataType.FP16)

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)
    src_var = ir.Var("src", src_type, span)

    call = ir.op.tensor.scatter_update(input_var, -2, index_var, src_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.scatter_update"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_scatter_update_4d():
    """Test tensor.scatter_update with 4D input and src."""
    span = ir.Span.unknown()

    block_num = ir.ConstInt(4, DataType.INT32, span)
    block_size = ir.ConstInt(4, DataType.INT32, span)
    one = ir.ConstInt(1, DataType.INT32, span)
    d = ir.ConstInt(64, DataType.INT32, span)
    b = ir.ConstInt(2, DataType.INT32, span)
    s = ir.ConstInt(4, DataType.INT32, span)

    input_type = ir.TensorType([block_num, block_size, one, d], DataType.BF16)
    index_type = ir.TensorType([b, s], DataType.INT32)
    src_type = ir.TensorType([b, s, one, d], DataType.BF16)

    input_var = ir.Var("kv_cache", input_type, span)
    index_var = ir.Var("block_table", index_type, span)
    src_var = ir.Var("new_kv", src_type, span)

    call = ir.op.tensor.scatter_update(input_var, -2, index_var, src_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.scatter_update"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.BF16
    assert len(result_type.shape) == 4


def test_tensor_scatter_update_dtype_mismatch():
    """Test tensor.scatter_update rejects mismatched dtypes."""
    span = ir.Span.unknown()

    rows = ir.ConstInt(16, DataType.INT32, span)
    d = ir.ConstInt(64, DataType.INT32, span)
    b = ir.ConstInt(2, DataType.INT32, span)
    s = ir.ConstInt(4, DataType.INT32, span)
    bs = ir.ConstInt(8, DataType.INT32, span)

    input_type = ir.TensorType([rows, d], DataType.FP16)
    index_type = ir.TensorType([b, s], DataType.INT32)
    src_type = ir.TensorType([bs, d], DataType.FP32)  # wrong dtype

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)
    src_var = ir.Var("src", src_type, span)

    with pytest.raises(ValueError, match="src dtype"):
        ir.op.tensor.scatter_update(input_var, -2, index_var, src_var)


def test_tensor_scatter_update_invalid_dim():
    """Test tensor.scatter_update rejects dim values other than -2."""
    span = ir.Span.unknown()

    rows = ir.ConstInt(16, DataType.INT32, span)
    d = ir.ConstInt(64, DataType.INT32, span)
    b = ir.ConstInt(2, DataType.INT32, span)
    s = ir.ConstInt(4, DataType.INT32, span)
    bs = ir.ConstInt(8, DataType.INT32, span)

    input_type = ir.TensorType([rows, d], DataType.FP16)
    index_type = ir.TensorType([b, s], DataType.INT32)
    src_type = ir.TensorType([bs, d], DataType.FP16)

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)
    src_var = ir.Var("src", src_type, span)

    with pytest.raises(ValueError, match="dim=-2"):
        ir.op.tensor.scatter_update(input_var, 0, index_var, src_var)


# ── tensor.gather tests ──────────────────────────────────────────────────


def test_tensor_gather_2d_dim0():
    """Test tensor.gather with 2D tensors and dim=0."""
    span = ir.Span.unknown()

    rows = ir.ConstInt(8, DataType.INT32, span)
    cols = ir.ConstInt(4, DataType.INT32, span)
    idx_rows = ir.ConstInt(3, DataType.INT32, span)

    input_type = ir.TensorType([rows, cols], DataType.FP32)
    index_type = ir.TensorType([idx_rows, cols], DataType.INT32)

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)

    call = ir.op.tensor.gather(input_var, 0, index_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.gather"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2
    # Output shape == index shape
    assert isinstance(result_type.shape[0], ir.ConstInt)
    assert isinstance(result_type.shape[1], ir.ConstInt)
    assert result_type.shape[0].value == 3
    assert result_type.shape[1].value == 4


def test_tensor_gather_2d_dim1():
    """Test tensor.gather with 2D tensors and dim=1 (MoE use case)."""
    span = ir.Span.unknown()

    batch = ir.ConstInt(16, DataType.INT32, span)
    num_experts = ir.ConstInt(64, DataType.INT32, span)
    topk = ir.ConstInt(4, DataType.INT32, span)

    input_type = ir.TensorType([batch, num_experts], DataType.FP16)
    index_type = ir.TensorType([batch, topk], DataType.INT32)

    input_var = ir.Var("topk_weights", input_type, span)
    index_var = ir.Var("topk_ids", index_type, span)

    call = ir.op.tensor.gather(input_var, 1, index_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.gather"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2
    assert isinstance(result_type.shape[0], ir.ConstInt)
    assert isinstance(result_type.shape[1], ir.ConstInt)
    assert result_type.shape[0].value == 16
    assert result_type.shape[1].value == 4


def test_tensor_gather_3d():
    """Test tensor.gather with 3D tensors and dim=1."""
    span = ir.Span.unknown()

    d0 = ir.ConstInt(4, DataType.INT32, span)
    d1 = ir.ConstInt(8, DataType.INT32, span)
    d2 = ir.ConstInt(16, DataType.INT32, span)
    idx_d1 = ir.ConstInt(3, DataType.INT32, span)

    input_type = ir.TensorType([d0, d1, d2], DataType.FP32)
    index_type = ir.TensorType([d0, idx_d1, d2], DataType.INT64)

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)

    call = ir.op.tensor.gather(input_var, 1, index_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.gather"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 3
    assert isinstance(result_type.shape[1], ir.ConstInt)
    assert result_type.shape[1].value == 3


def test_tensor_gather_negative_dim():
    """Test tensor.gather with negative dim value."""
    span = ir.Span.unknown()

    d0 = ir.ConstInt(4, DataType.INT32, span)
    d1 = ir.ConstInt(8, DataType.INT32, span)

    input_type = ir.TensorType([d0, d1], DataType.FP32)
    index_type = ir.TensorType([d0, d1], DataType.INT32)

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)

    # dim=-1 should be equivalent to dim=1 for a 2D tensor
    call = ir.op.tensor.gather(input_var, -1, index_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.gather"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32


def test_tensor_gather_rank_mismatch():
    """Test tensor.gather rejects rank mismatch between input and index."""
    span = ir.Span.unknown()

    d0 = ir.ConstInt(4, DataType.INT32, span)
    d1 = ir.ConstInt(8, DataType.INT32, span)
    d2 = ir.ConstInt(16, DataType.INT32, span)

    input_type = ir.TensorType([d0, d1, d2], DataType.FP32)  # 3D
    index_type = ir.TensorType([d0, d1], DataType.INT32)  # 2D

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)

    with pytest.raises(ValueError, match="index rank"):
        ir.op.tensor.gather(input_var, 0, index_var)


def test_tensor_gather_non_integer_index():
    """Test tensor.gather rejects non-integer index dtype."""
    span = ir.Span.unknown()

    d0 = ir.ConstInt(4, DataType.INT32, span)
    d1 = ir.ConstInt(8, DataType.INT32, span)

    input_type = ir.TensorType([d0, d1], DataType.FP32)
    index_type = ir.TensorType([d0, d1], DataType.FP32)  # float, not int

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)

    with pytest.raises(ValueError, match="index dtype must be integer"):
        ir.op.tensor.gather(input_var, 0, index_var)


def test_tensor_gather_dim_out_of_range():
    """Test tensor.gather rejects dim out of range."""
    span = ir.Span.unknown()

    d0 = ir.ConstInt(4, DataType.INT32, span)
    d1 = ir.ConstInt(8, DataType.INT32, span)

    input_type = ir.TensorType([d0, d1], DataType.FP32)
    index_type = ir.TensorType([d0, d1], DataType.INT32)

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)

    with pytest.raises(ValueError, match="dim.*out of range"):
        ir.op.tensor.gather(input_var, 2, index_var)


def test_tensor_gather_index_shape_exceeds_input():
    """Test tensor.gather rejects index shape larger than input on non-gather dims."""
    span = ir.Span.unknown()

    d0 = ir.ConstInt(4, DataType.INT32, span)
    d1 = ir.ConstInt(8, DataType.INT32, span)
    d1_big = ir.ConstInt(16, DataType.INT32, span)  # bigger than input's d1

    input_type = ir.TensorType([d0, d1], DataType.FP32)
    index_type = ir.TensorType([d0, d1_big], DataType.INT32)

    input_var = ir.Var("inp", input_type, span)
    index_var = ir.Var("idx", index_type, span)

    # dim=0, so d1 (index 1) is a non-gather dim and index.shape[1]=16 > input.shape[1]=8
    with pytest.raises(ValueError, match="must be <="):
        ir.op.tensor.gather(input_var, 0, index_var)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
