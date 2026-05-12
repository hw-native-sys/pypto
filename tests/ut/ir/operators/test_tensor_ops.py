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

import math

import pypto.language as pl
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


def test_tensor_matmul_acc_nd_batch_broadcast():
    """tensor.matmul_acc accepts ND inputs and broadcasts lhs/rhs batch dims to acc batch."""
    span = ir.Span.unknown()

    def cd(v: int) -> ir.ConstInt:
        return ir.ConstInt(v, DataType.INT32, span)

    # acc[1, 16, 64] FP32 += lhs[16, 32] BF16 @ rhs[1, 64, 32]^T BF16   (b_trans=True)
    # lhs is 2D (batch=[]), rhs is 3D (batch=[1]), broadcast batch=[1] == acc batch.
    acc_type = ir.TensorType([cd(1), cd(16), cd(64)], DataType.FP32)
    lhs_type = ir.TensorType([cd(16), cd(32)], DataType.BF16)
    rhs_type = ir.TensorType([cd(1), cd(64), cd(32)], DataType.BF16)
    acc = ir.Var("acc", acc_type, span)
    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)

    call = ir.op.tensor.matmul_acc(acc, lhs, rhs, b_trans=True)

    assert isinstance(call, ir.Call)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    const_dims = [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)]
    assert const_dims == [1, 16, 64]


def test_tensor_matmul_acc_nd_acc_batch_mismatch_fails():
    """tensor.matmul_acc rejects acc batch dims that disagree with broadcast(lhs, rhs)."""
    span = ir.Span.unknown()

    def cd(v: int) -> ir.ConstInt:
        return ir.ConstInt(v, DataType.INT32, span)

    # acc batch [3] but broadcast(lhs[2], rhs[1]) batch is [2] — should fail.
    acc_type = ir.TensorType([cd(3), cd(16), cd(64)], DataType.FP32)
    lhs_type = ir.TensorType([cd(2), cd(16), cd(32)], DataType.BF16)
    rhs_type = ir.TensorType([cd(1), cd(32), cd(64)], DataType.BF16)
    acc = ir.Var("acc", acc_type, span)
    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)

    with pytest.raises(ValueError, match="acc batch dim"):
        ir.op.tensor.matmul_acc(acc, lhs, rhs)


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
# Tensor sin/cos tests (FP32-only)
# =============================================================================


def test_tensor_sin_creates_call():
    """tensor.sin on an FP32 tensor produces a Call with FP32 output of the same shape."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
    tensor_var = ir.Var("x", tensor_type, span)

    call = ir.op.tensor.sin(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.sin"

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_cos_creates_call():
    """tensor.cos on an FP32 tensor produces a Call with FP32 output of the same shape."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
    tensor_var = ir.Var("x", tensor_type, span)

    call = ir.op.tensor.cos(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.cos"

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_sin_rejects_integer_input():
    """tensor.sin must reject INT32 input with an error mentioning the op name and FP32."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.INT32)
    tensor_var = ir.Var("x", tensor_type, span)

    with pytest.raises(ValueError, match=r"tensor\.sin.*FP32"):
        ir.op.tensor.sin(tensor_var)


def test_tensor_sin_rejects_fp16_input():
    """tensor.sin must reject FP16 input with an FP32-mentioning error."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("x", tensor_type, span)

    with pytest.raises(ValueError, match=r"(?i)FP32"):
        ir.op.tensor.sin(tensor_var)


def test_tensor_cos_rejects_bf16_input():
    """tensor.cos must reject BF16 input with an error mentioning the op name and FP32."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.BF16)
    tensor_var = ir.Var("x", tensor_type, span)

    with pytest.raises(ValueError, match=r"tensor\.cos.*FP32"):
        ir.op.tensor.cos(tensor_var)


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
# Tensor abs tests
# =============================================================================


def test_tensor_abs():
    """Test tensor.abs operation."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.BF16)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.abs(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.abs"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.BF16
    assert len(result_type.shape) == 2


def test_tensor_abs_int_dtype():
    """Test tensor.abs preserves integer dtype."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64], DataType.INT32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.abs(tensor_var)
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


def test_tensor_rsqrt_high_precision_kwarg():
    """tensor.rsqrt carries the high_precision kwarg when requested."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.rsqrt(tensor_var, high_precision=True)

    assert call.op.name == "tensor.rsqrt"
    kwargs = dict(call.kwargs)
    assert kwargs.get("high_precision") is True


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


def test_tensor_maximum_scalar():
    """Test tensor.maximum with scalar rhs."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    type_a = ir.TensorType([dim64], DataType.FP32)
    var_a = ir.Var("a", type_a, span)

    call = ir.op.tensor.maximum(var_a, 0.5)
    assert call.op.name == "tensor.maximum"


def test_tensor_minimum():
    """Test tensor.minimum operation (tensor-tensor and tensor-scalar)."""
    span = ir.Span.unknown()
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    type_a = ir.TensorType([dim64], DataType.FP32)
    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_a, span)

    call_tt = ir.op.tensor.minimum(var_a, var_b)
    assert call_tt.op.name == "tensor.minimum"

    call_ts = ir.op.tensor.minimum(var_a, 1.0)
    assert call_ts.op.name == "tensor.minimum"


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
    assert ir.is_op_registered("tensor.set_validshape")
    assert ir.is_op_registered("tensor.maximum")
    assert ir.is_op_registered("tensor.minimum")
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


def _const_int_values(exprs) -> list[int]:
    """Extract values from a sequence of ConstInt exprs (asserting the type)."""
    out: list[int] = []
    for e in exprs:
        assert isinstance(e, ir.ConstInt)
        out.append(e.value)
    return out


def test_tensor_transpose_2d_records_swapped_strides_and_dn():
    """tensor.transpose on a 2D tensor records swapped physical strides and
    toggles the layout from ND to DN.

    Regression test for #1209: codegen needs the explicit strides to emit
    a make_tensor_view that matches the source's actual (row-major) memory
    layout — synthesizing strides from the DN tag alone gave wrong addresses
    (column-major reinterpretation of row-major data). The DN tag is still
    toggled because PTOAS expects it on the kernel boundary.
    """
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    tensor_var = ir.Var("t", ir.TensorType([dim8, dim16], DataType.FP32), span)

    call = tensor.transpose(tensor_var, 0, 1)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert _const_int_values(result_type.shape) == [16, 8]
    assert result_type.tensor_view is not None
    assert result_type.tensor_view.layout == ir.TensorLayout.DN
    # Input row-major strides [16, 1] swapped at (0, 1) -> [1, 16].
    assert _const_int_values(result_type.tensor_view.stride) == [1, 16]


def test_tensor_transpose_3d_trailing_axes_records_swapped_strides_and_dn():
    """tensor.transpose 3D at the trailing axes (1, 2) records swapped
    strides and toggles to DN.

    Input row-major strides for [2, 3, 4]: [12, 4, 1]. Swap at (1, 2) ->
    [12, 1, 4]. The DN tag covers "trailing two dimensions swapped".
    """
    span = ir.Span.unknown()
    dim2 = ir.ConstInt(2, DataType.INT32, span)
    dim3 = ir.ConstInt(3, DataType.INT32, span)
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    tensor_var = ir.Var("t", ir.TensorType([dim2, dim3, dim4], DataType.FP32), span)

    call = tensor.transpose(tensor_var, 1, 2)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert _const_int_values(result_type.shape) == [2, 4, 3]
    assert result_type.tensor_view is not None
    assert result_type.tensor_view.layout == ir.TensorLayout.DN
    assert _const_int_values(result_type.tensor_view.stride) == [12, 1, 4]


def test_tensor_transpose_non_trailing_axes_records_strides_no_dn():
    """Non-trailing transpose records swapped strides; layout stays ND.

    ND/DN only capture trailing-two-dim swaps, so non-trailing axes cannot
    be described by the layout tag alone.
    Non-trailing transposes fall back to the legacy "no metadata" path
    Explicit strides handle this: strides are reordered at the swap axes;
    layout stays ND because ND/DN cannot encode arbitrary outer-dim swaps.
    Codegen lowers via the explicit-stride path of EmitMakeTensorViews.
    """
    span = ir.Span.unknown()
    dim2 = ir.ConstInt(2, DataType.INT32, span)
    dim3 = ir.ConstInt(3, DataType.INT32, span)
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    tensor_var = ir.Var("t", ir.TensorType([dim2, dim3, dim4], DataType.FP32), span)

    call = tensor.transpose(tensor_var, 0, 1)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert _const_int_values(result_type.shape) == [3, 2, 4]
    assert result_type.tensor_view is not None
    assert result_type.tensor_view.layout == ir.TensorLayout.ND
    # Input row-major strides [12, 4, 1] swapped at (0, 1) -> [4, 12, 1].
    assert _const_int_values(result_type.tensor_view.stride) == [4, 12, 1]


def test_tensor_transpose_idempotent_layout():
    """transpose(transpose(x, 0, 1), 0, 1) collapses back to a bare TensorType.

    Strides round-trip through both swaps to the canonical row-major
    pattern, layout flips ND -> DN -> ND, and valid_shape/pad stay default,
    so the result type drops its TensorView entirely.
    """
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    tensor_var = ir.Var("t", ir.TensorType([dim8, dim16], DataType.FP32), span)

    once = tensor.transpose(tensor_var, 0, 1)
    intermediate = ir.Var("xt", once.type, span)
    twice = tensor.transpose(intermediate, 0, 1)

    result_type = twice.type
    assert isinstance(result_type, ir.TensorType)
    assert _const_int_values(result_type.shape) == [8, 16]
    assert result_type.tensor_view is None


def test_tensor_transpose_dynamic_shape_records_symbolic_strides():
    """Dynamic input shapes get symbolic swapped strides plus the DN tag.

    Row-major strides for [M, N] are [N, 1]; swap at (0, 1) -> [1, N].
    The N-stride is a Var, not a ConstInt — codegen emits it via
    EmitCastToIndex on the explicit-strides path.
    """
    span = ir.Span.unknown()
    m = ir.Var("M", ir.ScalarType(DataType.INDEX), span)
    n = ir.Var("N", ir.ScalarType(DataType.INDEX), span)
    tensor_var = ir.Var("t", ir.TensorType([m, n], DataType.FP32), span)

    call = tensor.transpose(tensor_var, 0, 1)

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert len(result_type.shape) == 2
    assert result_type.tensor_view is not None
    assert result_type.tensor_view.layout == ir.TensorLayout.DN
    # strides[0] is the (folded) ConstInt(1) from the row-major identity;
    # strides[1] is the symbolic dim N — value-compared rather than identity-
    # compared to stay robust against MakeIndexMul folding-rule changes.
    strides = result_type.tensor_view.stride
    assert len(strides) == 2
    assert isinstance(strides[0], ir.ConstInt) and strides[0].value == 1
    assert strides[1] == n


def test_tensor_transpose_explicit_valid_shape_not_swapped():
    """User-supplied valid_shape (4th arg) is in the OUTPUT coordinate system."""
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    tensor_var = ir.Var("t", ir.TensorType([dim8, dim16], DataType.FP32), span)

    # User supplies valid_shape in output's coord order: [16, 8] for the
    # transposed tensor's [16, 8] shape — must NOT be swapped.
    call = tensor.transpose(tensor_var, 0, 1, valid_shape=[16, 8])

    rt = call.type
    assert isinstance(rt, ir.TensorType)
    assert rt.tensor_view is not None
    assert _const_int_values(rt.tensor_view.valid_shape) == [16, 8]
    # Layout is still toggled to DN (trailing-two-dim transpose), and the
    # explicit-strides path also records swapped row-major strides.
    assert rt.tensor_view.layout == ir.TensorLayout.DN
    assert _const_int_values(rt.tensor_view.stride) == [1, 16]


def test_tensor_transpose_valid_shape_rank_mismatch_rejected():
    """A 4th-arg valid_shape with the wrong rank raises a clear error."""
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    tensor_var = ir.Var("t", ir.TensorType([dim8, dim16], DataType.FP32), span)

    with pytest.raises(Exception, match="valid_shape rank"):
        tensor.transpose(tensor_var, 0, 1, valid_shape=[16])


def test_tensor_transpose_input_explicit_strides_propagated_swapped():
    """If the input already carries explicit strides, those take precedence
    over the row-major default and get swapped at the same axes."""
    span = ir.Span.unknown()
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    s10 = ir.ConstInt(10, DataType.INDEX, span)  # non-default outer stride (e.g. from a slice)
    s1 = ir.ConstInt(1, DataType.INDEX, span)
    input_view = ir.TensorView([s10, s1], ir.TensorLayout.ND)
    tensor_var = ir.Var("t", ir.TensorType([dim8, dim16], DataType.FP32, tensor_view=input_view), span)

    call = tensor.transpose(tensor_var, 0, 1)

    rt = call.type
    assert isinstance(rt, ir.TensorType)
    assert rt.tensor_view is not None
    # Input strides [10, 1] swapped -> [1, 10]; layout toggled ND -> DN.
    assert _const_int_values(rt.tensor_view.stride) == [1, 10]
    assert rt.tensor_view.layout == ir.TensorLayout.DN


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


def test_tensor_slice_drop_dims_rank_reduces():
    """tensor.slice drop_dims erases the listed unit axes from the result type."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([64, 64, 64, 64], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.slice(tensor_var, [1, 1, 64, 64], [3, 5, 0, 0], drop_dims=[0, 1])

    assert call.op.name == "tensor.slice"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [64, 64]
    # shape / offset stay full-rank; drop_dims is the 5th operand (empty valid_shape 4th).
    assert len(call.args) == 5


def test_tensor_slice_drop_dims_drops_valid_shape_axes():
    """drop_dims removes the same axes from a supplied valid_shape."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("t", ir.TensorType([64, 64, 64], DataType.FP32), span)

    call = ir.op.tensor.slice(tensor_var, [1, 8, 64], [2, 0, 0], valid_shape=[1, 4, 64], drop_dims=[0])
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [8, 64]
    assert result_type.tensor_view is not None
    assert [d.value for d in result_type.tensor_view.valid_shape if isinstance(d, ir.ConstInt)] == [4, 64]


def test_tensor_slice_drop_dims_rejects_non_unit_dim():
    """drop_dims may only erase statically size-1 dimensions."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("t", ir.TensorType([64, 64], DataType.FP32), span)
    with pytest.raises(ValueError, match="static unit dimension"):
        ir.op.tensor.slice(tensor_var, [8, 64], [0, 0], drop_dims=[0])


def test_tensor_slice_drop_dims_rejects_out_of_range():
    """drop_dims indices must be within the slice rank."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("t", ir.TensorType([64, 64], DataType.FP32), span)
    with pytest.raises(ValueError, match="out of range"):
        ir.op.tensor.slice(tensor_var, [1, 64], [0, 0], drop_dims=[2])


def test_tensor_slice_empty_drop_dims_is_backward_compatible():
    """drop_dims=None / [] keeps the legacy 3-arg result type (no tensor_view)."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("t", ir.TensorType([64, 64], DataType.FP32), span)
    call_none = ir.op.tensor.slice(tensor_var, [8, 16], [0, 0])
    call_empty = ir.op.tensor.slice(tensor_var, [8, 16], [0, 0], drop_dims=[])
    for call in (call_none, call_empty):
        assert len(call.args) == 3
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert result_type.tensor_view is None
        assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [8, 16]


def test_tensor_slice_drop_dims_print_parse_roundtrip():
    """A drop_dims slice survives python_print -> pl.parse -> python_print."""
    src = (
        "import pypto.language as pl\n\n"
        "@pl.program\n"
        "class P:\n"
        "    @pl.function\n"
        "    def main(self, x: pl.Tensor[[64, 64, 64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:\n"
        "        y: pl.Tensor[[64, 64], pl.FP32] = "
        "pl.tensor.slice(x, [1, 1, 64, 64], [3, 5, 0, 0], drop_dims=[0, 1])\n"
        "        return y\n"
    )
    prog = pl.parse(src)
    reparsed = pl.parse(ir.python_print(prog))
    ir.assert_structural_equal(reparsed, prog)


def _make_slice_tensor_var():
    """Build a [16, 32] FP16 tensor Var for slice pad_value tests."""
    span = ir.Span.unknown()
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim16, dim32], DataType.FP16)
    return ir.Var("t", tensor_type, span)


def test_tensor_slice_with_pad_value():
    """tensor.slice writes pad_value=zero to the output tensor_view.pad."""
    tensor_var = _make_slice_tensor_var()
    call = tensor.slice(tensor_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=ir.PadValue.zero)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.slice"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.tensor_view is not None
    assert result_type.tensor_view.pad == ir.PadValue.zero
    assert len(result_type.tensor_view.valid_shape) == 2

    # Sanity-check min/max variants reach the same field.
    for pad in (ir.PadValue.min, ir.PadValue.max):
        call_p = tensor.slice(tensor_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=pad)
        result_type_p = call_p.type
        assert isinstance(result_type_p, ir.TensorType)
        assert result_type_p.tensor_view is not None
        assert result_type_p.tensor_view.pad == pad


def test_tensor_slice_default_pad_is_null():
    """tensor.slice without pad_value defaults to PadValue.null (backward compat)."""
    tensor_var = _make_slice_tensor_var()

    # No tensor_view created when both valid_shape and pad_value are absent.
    call = tensor.slice(tensor_var, [8, 16], [0, 0])
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.tensor_view is None

    # With only valid_shape provided, tensor_view is present and pad defaults to null.
    call_vs = tensor.slice(tensor_var, [8, 16], [0, 0], valid_shape=[8, 4])
    result_type_vs = call_vs.type
    assert isinstance(result_type_vs, ir.TensorType)
    assert result_type_vs.tensor_view is not None
    assert result_type_vs.tensor_view.pad == ir.PadValue.null


def test_tensor_slice_rejects_bad_pad_value():
    """tensor.slice rejects a non-PadValue pad_value kwarg via registry validation."""
    tensor_var = _make_slice_tensor_var()
    span = tensor_var.span
    shape_tuple = ir.MakeTuple(
        [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)], span
    )
    offset_tuple = ir.MakeTuple(
        [ir.ConstInt(0, DataType.INT32, span), ir.ConstInt(0, DataType.INT32, span)], span
    )
    valid_shape_tuple = ir.MakeTuple(
        [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(4, DataType.INT32, span)], span
    )
    with pytest.raises(TypeError, match="'pad_value'.*incompatible type"):
        ir.create_op_call(
            "tensor.slice",
            [tensor_var, shape_tuple, offset_tuple, valid_shape_tuple],
            {"pad_value": 5},
            span,
        )


def test_tensor_slice_accepts_numeric_sugar_pad_value():
    """tensor.slice maps 0 / math.inf / -math.inf onto PadValue zero/max/min."""
    tensor_var = _make_slice_tensor_var()
    for literal, expected_pad in [
        (0, ir.PadValue.zero),
        (math.inf, ir.PadValue.max),
        (-math.inf, ir.PadValue.min),
    ]:
        call = tensor.slice(tensor_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=literal)
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert result_type.tensor_view is not None
        assert result_type.tensor_view.pad == expected_pad


def test_tensor_slice_pad_without_valid_shape_warns():
    """DSL emits a UserWarning when pad_value is set but valid_shape is None."""
    span = ir.Span.unknown()
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim16, dim32], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    tensor_arg = pl.Tensor(expr=tensor_var)
    with pytest.warns(UserWarning, match="pad_value has no effect"):
        pl.tensor.slice(tensor_arg, [8, 16], [0, 0], pad_value=pl.PadValue.zero)


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


def test_tensor_set_validshape():
    """Test tensor.set_validshape sets valid-shape metadata on a 2D tensor."""
    span = ir.Span.unknown()
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim32, dim32], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.set_validshape(tensor_var, 16, 24)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.set_validshape"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert result_type.tensor_view is not None
    assert len(result_type.tensor_view.valid_shape) == 2


def test_tensor_set_validshape_dynamic():
    """Test tensor.set_validshape with dynamic scalar arguments."""
    span = ir.Span.unknown()
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim32, dim32], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)
    vr = ir.Var("vr", ir.ScalarType(DataType.INDEX), span)
    vc = ir.Var("vc", ir.ScalarType(DataType.INDEX), span)

    call = ir.op.tensor.set_validshape(tensor_var, vr, vc)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.set_validshape"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.tensor_view is not None
    assert len(result_type.tensor_view.valid_shape) == 2


def test_tensor_set_validshape_rejects_negative():
    """Test tensor.set_validshape rejects negative constant bounds."""
    span = ir.Span.unknown()
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim32, dim32], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    with pytest.raises(Exception, match="must be >= 0"):
        ir.op.tensor.set_validshape(tensor_var, -1, 16)


def test_tensor_set_validshape_rejects_exceeding_bound():
    """Test tensor.set_validshape rejects bounds exceeding physical shape."""
    span = ir.Span.unknown()
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim32, dim32], DataType.FP32)
    tensor_var = ir.Var("t", tensor_type, span)

    with pytest.raises(Exception, match="exceeds tensor bound"):
        ir.op.tensor.set_validshape(tensor_var, 16, 64)


def test_tensor_set_validshape_preserves_existing_view():
    """Test tensor.set_validshape preserves existing TensorView stride and layout."""
    span = ir.Span.unknown()
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    existing_view = ir.TensorView(
        stride=[ir.ConstInt(64, DataType.INT32, span), ir.ConstInt(1, DataType.INT32, span)],
        layout=ir.TensorLayout.ND,
        valid_shape=[dim32, dim32],
    )
    tensor_type = ir.TensorType([dim32, dim32], DataType.FP32, None, existing_view)
    tensor_var = ir.Var("t", tensor_type, span)

    call = ir.op.tensor.set_validshape(tensor_var, 16, 24)

    assert isinstance(call, ir.Call)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.tensor_view is not None
    assert result_type.tensor_view.layout == ir.TensorLayout.ND
    assert len(result_type.tensor_view.stride) == 2
    stride_0 = result_type.tensor_view.stride[0]
    stride_1 = result_type.tensor_view.stride[1]
    assert isinstance(stride_0, ir.ConstInt)
    assert isinstance(stride_1, ir.ConstInt)
    assert stride_0.value == 64
    assert stride_1.value == 1
    assert len(result_type.tensor_view.valid_shape) == 2


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
    dim1 = ir.ConstInt(1, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    row_type = ir.TensorType([dim64, dim1], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)
    row_var = ir.Var("rv", row_type, span)

    call = ir.op.tensor.row_expand(tensor_var, row_var)

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


# =============================================================================
# Tensor expand_clone tests
# =============================================================================


def test_tensor_expand_clone_dim0():
    """Test tensor.expand_clone broadcasts dim0."""
    span = ir.Span.unknown()

    dim1 = ir.ConstInt(1, DataType.INT32, span)
    dim2 = ir.ConstInt(2, DataType.INT32, span)
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)

    input_type = ir.TensorType([dim1, dim4, dim8], DataType.FP32)
    target_type = ir.TensorType([dim2, dim4, dim8], DataType.FP32)
    input_var = ir.Var("src", input_type, span)
    target_var = ir.Var("dst", target_type, span)

    call = ir.op.tensor.expand_clone(input_var, target_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.expand_clone"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 3
    for dim, expected in zip(result_type.shape, [2, 4, 8]):
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == expected


def test_tensor_expand_clone_dim1():
    """Test tensor.expand_clone broadcasts dim1."""
    span = ir.Span.unknown()

    dim1 = ir.ConstInt(1, DataType.INT32, span)
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)

    input_type = ir.TensorType([dim4, dim1, dim8], DataType.FP32)
    target_type = ir.TensorType([dim4, dim16, dim8], DataType.FP32)
    input_var = ir.Var("src", input_type, span)
    target_var = ir.Var("dst", target_type, span)

    call = ir.op.tensor.expand_clone(input_var, target_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.expand_clone"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 3
    for dim, expected in zip(result_type.shape, [4, 16, 8]):
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == expected


def test_tensor_expand_clone_dim2():
    """Test tensor.expand_clone broadcasts dim2."""
    span = ir.Span.unknown()

    dim1 = ir.ConstInt(1, DataType.INT32, span)
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)

    input_type = ir.TensorType([dim4, dim8, dim1], DataType.FP32)
    target_type = ir.TensorType([dim4, dim8, dim16], DataType.FP32)
    input_var = ir.Var("src", input_type, span)
    target_var = ir.Var("dst", target_type, span)

    call = ir.op.tensor.expand_clone(input_var, target_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.expand_clone"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 3
    for dim, expected in zip(result_type.shape, [4, 8, 16]):
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == expected


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


class TestTensorFormatShapeError:
    """Regression tests for issue #824: FormatShape prints readable shapes, not pointer addresses."""

    def test_tensor_add_shape_mismatch_shows_readable_dims(self):
        """Test that tensor shape mismatch errors show readable dimensions."""
        span = ir.Span.unknown()

        dim4 = ir.ConstInt(4, DataType.DEFAULT_CONST_INT, span)
        dim8 = ir.ConstInt(8, DataType.DEFAULT_CONST_INT, span)
        dim3 = ir.ConstInt(3, DataType.DEFAULT_CONST_INT, span)

        tensor_type1 = ir.TensorType([dim4, dim8], DataType.FP32)
        tensor_type2 = ir.TensorType([dim3, dim8], DataType.FP32)

        tensor_a = ir.Var("a", tensor_type1, span)
        tensor_b = ir.Var("b", tensor_type2, span)

        with pytest.raises(ValueError, match=r"\[4, 8\].*\[3, 8\]"):
            ir.op.tensor.add(tensor_a, tensor_b)

    def test_tensor_add_symbolic_shape_mismatch_shows_var_names(self):
        """Test that symbolic tensor shape mismatch errors show variable names."""
        span = ir.Span.unknown()

        sym_m = ir.Var("M", ir.ScalarType(DataType.INT32), span)
        sym_n = ir.Var("N", ir.ScalarType(DataType.INT32), span)
        dim8 = ir.ConstInt(8, DataType.DEFAULT_CONST_INT, span)

        tensor_type1 = ir.TensorType([sym_m, dim8], DataType.FP32)
        tensor_type2 = ir.TensorType([sym_n, dim8], DataType.FP32)

        tensor_a = ir.Var("a", tensor_type1, span)
        tensor_b = ir.Var("b", tensor_type2, span)

        with pytest.raises(ValueError, match=r"\[M, 8\].*\[N, 8\]"):
            ir.op.tensor.add(tensor_a, tensor_b)


def test_tensor_sort32():
    """tensor.sort32 doubles the last dim and preserves dtype."""
    span = ir.Span.unknown()
    d8 = ir.ConstInt(8, DataType.INT32, span)
    d32 = ir.ConstInt(32, DataType.INT32, span)
    src = ir.Var("src", ir.TensorType([d8, d32], DataType.FP32), span)
    idx = ir.Var("idx", ir.TensorType([d8, d32], DataType.UINT32), span)

    call = ir.op.tensor.sort32(src, idx)
    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.sort32"

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2
    assert isinstance(result_type.shape[1], ir.ConstInt)
    assert result_type.shape[1].value == 64


def test_tensor_sort32_wrong_dtype():
    """tensor.sort32 rejects non-FP src dtype."""
    span = ir.Span.unknown()
    d8 = ir.ConstInt(8, DataType.INT32, span)
    d32 = ir.ConstInt(32, DataType.INT32, span)
    src = ir.Var("src", ir.TensorType([d8, d32], DataType.INT32), span)
    idx = ir.Var("idx", ir.TensorType([d8, d32], DataType.INT32), span)

    with pytest.raises(Exception, match=r"FP16 or FP32"):
        ir.op.tensor.sort32(src, idx)


def test_tensor_mrgsort_format1():
    """tensor.mrgsort(block_len=...) emits tensor.mrgsort_format1 with src shape."""
    span = ir.Span.unknown()
    d1 = ir.ConstInt(1, DataType.INT32, span)
    d128 = ir.ConstInt(128, DataType.INT32, span)
    src = ir.Var("src", ir.TensorType([d1, d128], DataType.FP32), span)

    call = ir.op.tensor.mrgsort(src, block_len=64)
    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.mrgsort_format1"

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert isinstance(result_type.shape[1], ir.ConstInt)
    assert result_type.shape[1].value == 128


def test_tensor_mrgsort_format1_invalid_block_len():
    """tensor.mrgsort_format1 rejects block_len that is not a multiple of 64."""
    span = ir.Span.unknown()
    d1 = ir.ConstInt(1, DataType.INT32, span)
    d128 = ir.ConstInt(128, DataType.INT32, span)
    src = ir.Var("src", ir.TensorType([d1, d128], DataType.FP32), span)

    with pytest.raises(Exception, match=r"multiple of 64"):
        ir.op.tensor.mrgsort(src, block_len=63)


def test_tensor_mrgsort_format2():
    """tensor.mrgsort(src0..src3) emits tensor.mrgsort_format2 with summed last-dim shape."""
    span = ir.Span.unknown()
    d1 = ir.ConstInt(1, DataType.INT32, span)
    d128 = ir.ConstInt(128, DataType.INT32, span)
    src_t = ir.TensorType([d1, d128], DataType.FP32)

    srcs = [ir.Var(f"s{i}", src_t, span) for i in range(4)]

    call = ir.op.tensor.mrgsort(*srcs)
    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.mrgsort_format2"

    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    # Output shape: last dim = sum of all src last dims (4 * 128 = 512)
    assert isinstance(result_type.shape[1], ir.ConstInt)
    assert result_type.shape[1].value == 512


def test_tensor_mrgsort_format2_dtype_mismatch():
    """tensor.mrgsort_format2 rejects mismatched src dtypes."""
    span = ir.Span.unknown()
    d1 = ir.ConstInt(1, DataType.INT32, span)
    d128 = ir.ConstInt(128, DataType.INT32, span)
    src_fp32 = ir.TensorType([d1, d128], DataType.FP32)
    src_fp16 = ir.TensorType([d1, d128], DataType.FP16)

    s0 = ir.Var("s0", src_fp32, span)
    s1 = ir.Var("s1", src_fp16, span)
    s2 = ir.Var("s2", src_fp32, span)
    s3 = ir.Var("s3", src_fp32, span)

    with pytest.raises(Exception, match=r"matching dtype"):
        ir.op.tensor.mrgsort(s0, s1, s2, s3)


def test_tensor_mrgsort_mixed_args_rejected():
    """mrgsort cannot mix block_len with format2 positional args."""
    span = ir.Span.unknown()
    d1 = ir.ConstInt(1, DataType.INT32, span)
    d128 = ir.ConstInt(128, DataType.INT32, span)
    s0 = ir.Var("s0", ir.TensorType([d1, d128], DataType.FP32), span)
    s1 = ir.Var("s1", ir.TensorType([d1, d128], DataType.FP32), span)

    with pytest.raises(ValueError, match=r"mutually exclusive"):
        ir.op.tensor.mrgsort(s0, s1, block_len=64)


# Tensor gather tests


def _make_gather_inputs(src_dtype=DataType.FP32, idx_dtype=DataType.INT32, b=4, n=16, k=3):
    span = ir.Span.unknown()
    B = ir.ConstInt(b, DataType.INT32, span)
    N = ir.ConstInt(n, DataType.INT32, span)
    K = ir.ConstInt(k, DataType.INT32, span)
    inp = ir.Var("inp", ir.TensorType([B, N], src_dtype), span)
    idx = ir.Var("idx", ir.TensorType([B, K], idx_dtype), span)
    return inp, idx


def test_tensor_gather_basic():
    """tensor.gather output has index shape and input dtype."""
    inp, idx = _make_gather_inputs()
    call = ir.op.tensor.gather(inp, dim=-1, index=idx)
    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.gather"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2
    assert isinstance(result_type.shape[0], ir.ConstInt) and result_type.shape[0].value == 4
    assert isinstance(result_type.shape[1], ir.ConstInt) and result_type.shape[1].value == 3


def test_tensor_gather_dim_last_axis_positive():
    """dim=rank-1 is accepted as an alias for dim=-1."""
    inp, idx = _make_gather_inputs()
    call = ir.op.tensor.gather(inp, dim=1, index=idx)
    assert call.op.name == "tensor.gather"


def test_tensor_gather_rejects_bad_dim():
    inp, idx = _make_gather_inputs()
    # rank=2, valid dims are -2..1. dim=2 is out of range.
    with pytest.raises(Exception, match=r"dim"):
        ir.op.tensor.gather(inp, dim=2, index=idx)


def test_tensor_gather_rejects_non_int32_index():
    inp, idx = _make_gather_inputs(idx_dtype=DataType.INT16)
    with pytest.raises(Exception, match=r"index dtype to be INT32"):
        ir.op.tensor.gather(inp, dim=-1, index=idx)


def test_tensor_gather_rejects_unsupported_input_dtype():
    inp, idx = _make_gather_inputs(src_dtype=DataType.UINT32)
    with pytest.raises(Exception, match=r"FP16, FP32, INT16, or INT32"):
        ir.op.tensor.gather(inp, dim=-1, index=idx)


def test_tensor_gather_rejects_rank_mismatch():
    span = ir.Span.unknown()
    B = ir.ConstInt(4, DataType.INT32, span)
    N = ir.ConstInt(16, DataType.INT32, span)
    K = ir.ConstInt(3, DataType.INT32, span)
    inp = ir.Var("inp", ir.TensorType([B, N], DataType.FP32), span)
    idx = ir.Var("idx", ir.TensorType([K], DataType.INT32), span)
    with pytest.raises(Exception, match=r"rank"):
        ir.op.tensor.gather(inp, dim=-1, index=idx)


def test_tensor_gather_rejects_non_matching_outer_dim():
    span = ir.Span.unknown()
    B = ir.ConstInt(4, DataType.INT32, span)
    B2 = ir.ConstInt(5, DataType.INT32, span)
    N = ir.ConstInt(16, DataType.INT32, span)
    K = ir.ConstInt(3, DataType.INT32, span)
    inp = ir.Var("inp", ir.TensorType([B, N], DataType.FP32), span)
    idx = ir.Var("idx", ir.TensorType([B2, K], DataType.INT32), span)
    with pytest.raises(Exception, match=r"non-gather axes"):
        ir.op.tensor.gather(inp, dim=-1, index=idx)


# ---- tensor.gather_mask (mask-pattern form) -----------------------------------


def _make_gather_mask_input(rows: int = 8, cols: int = 64, dtype: DataType = DataType.FP32):
    span = ir.Span.unknown()
    R = ir.ConstInt(rows, DataType.INT32, span)
    C = ir.ConstInt(cols, DataType.INT32, span)
    return ir.Var("inp", ir.TensorType([R, C], dtype), span)


def test_tensor_gather_mask_p0101_halves_last_dim():
    """tensor.gather(input, mask_pattern=1) emits tensor.gather_mask, last dim /= 2."""
    inp = _make_gather_mask_input(rows=8, cols=64)
    call = ir.op.tensor.gather(inp, mask_pattern=1)
    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.gather_mask"
    rt = call.type
    assert isinstance(rt, ir.TensorType)
    assert rt.dtype == DataType.FP32
    assert isinstance(rt.shape[0], ir.ConstInt) and rt.shape[0].value == 8
    assert isinstance(rt.shape[1], ir.ConstInt) and rt.shape[1].value == 32


def test_tensor_gather_mask_p0001_quarters_last_dim():
    """Patterns 3..6 produce a /4 shrink."""
    inp = _make_gather_mask_input(rows=4, cols=64)
    call = ir.op.tensor.gather(inp, mask_pattern=3)
    rt = call.type
    assert isinstance(rt, ir.TensorType)
    assert isinstance(rt.shape[1], ir.ConstInt)
    assert rt.shape[1].value == 16


def test_tensor_gather_mask_p1111_keeps_last_dim():
    inp = _make_gather_mask_input(rows=4, cols=64)
    call = ir.op.tensor.gather(inp, mask_pattern=7)
    rt = call.type
    assert isinstance(rt, ir.TensorType)
    assert isinstance(rt.shape[1], ir.ConstInt)
    assert rt.shape[1].value == 64


def test_tensor_gather_mask_output_dtype_reinterpret():
    """output_dtype reinterprets bits to a same-bit-width dtype."""
    inp = _make_gather_mask_input(rows=2, cols=32, dtype=DataType.FP32)
    call = ir.op.tensor.gather(inp, mask_pattern=2, output_dtype=DataType.UINT32)
    assert call.op.name == "tensor.gather_mask"
    rt = call.type
    assert isinstance(rt, ir.TensorType)
    assert rt.dtype == DataType.UINT32


def test_tensor_gather_mask_rejects_bad_pattern():
    inp = _make_gather_mask_input()
    with pytest.raises(Exception, match=r"mask_pattern in range"):
        ir.op.tensor.gather(inp, mask_pattern=0)


def test_tensor_gather_mask_rejects_indivisible_cols():
    inp = _make_gather_mask_input(rows=2, cols=33)
    with pytest.raises(Exception, match=r"divisible by 2"):
        ir.op.tensor.gather(inp, mask_pattern=1)


def test_tensor_gather_mask_rejects_dtype_width_mismatch():
    inp = _make_gather_mask_input(rows=2, cols=32, dtype=DataType.FP16)
    with pytest.raises(Exception, match=r"same bit width"):
        ir.op.tensor.gather(inp, mask_pattern=1, output_dtype=DataType.FP32)


def test_tensor_gather_rejects_mixed_index_and_mask():
    inp, idx = _make_gather_inputs()
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        ir.op.tensor.gather(inp, dim=-1, index=idx, mask_pattern=1)


class TestTensorCiOp:
    """Tests for tensor.ci (contiguous integer sequence)."""

    def test_tensor_ci_ascending(self):
        call = tensor.ci(0, [1, 32], dtype=DataType.INT32)
        t = call.type
        assert isinstance(t, ir.TensorType)
        assert t.dtype == DataType.INT32
        assert len(t.shape) == 2
        assert "tensor.ci" in str(call)

    def test_tensor_ci_descending_kwarg_printed(self):
        call = tensor.ci(10, [1, 16], dtype=DataType.INT32, descending=True)
        assert "descending=True" in str(call)

    def test_tensor_ci_rejects_float_dtype(self):
        with pytest.raises(ValueError, match=r"INT16.*INT32.*UINT16.*UINT32"):
            tensor.ci(0, [1, 32], dtype=DataType.FP32)

    @pytest.mark.parametrize("dtype", [DataType.INT16, DataType.UINT16, DataType.UINT32])
    def test_tensor_ci_accepts_non_int32_dtypes(self, dtype):
        call = tensor.ci(0, [1, 16], dtype=dtype)
        t = call.type
        assert isinstance(t, ir.TensorType)
        assert t.dtype == dtype

    def test_tensor_ci_rejects_cols_equal_one(self):
        with pytest.raises(ValueError, match="innermost dimension"):
            tensor.ci(0, [32, 1], dtype=DataType.INT32)

    def test_tensor_ci_rejects_multi_row_shape(self):
        """pto.tci only populates the first row, so leading dims must be 1."""
        with pytest.raises(ValueError, match=r"leading dimensions must be 1"):
            tensor.ci(0, [4, 32], dtype=DataType.INT32)

    def test_tensor_arange_alias_is_ci(self):
        assert pl.tensor.arange is pl.tensor.ci

    def test_top_level_arange_is_tensor_ci(self):
        assert pl.arange is pl.tensor.ci


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
