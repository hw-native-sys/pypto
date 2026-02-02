# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor operations for PyPTO Language DSL.

This module provides type-safe wrappers around pypto.ir.op.tensor operations
that accept and return Tensor types instead of raw Expr/Call objects.
"""

from typing import List, Literal, Optional, Union

from pypto.ir.op import tensor_ops as _ir_ops
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr

from ..scalar import Scalar
from ..tensor import Tensor


def create(shape: List[int], dtype: DataType) -> Tensor:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes
        dtype: Data type of tensor elements

    Returns:
        Tensor wrapping the create operation
    """
    call_expr = _ir_ops.create(shape, dtype)
    return Tensor(expr=call_expr)


def view(tensor: Tensor, shape: List[Union[int, Expr]], offset: List[Union[int, Expr]]) -> Tensor:
    """Create a view/slice of a tensor with new shape and offset.

    Args:
        tensor: Input tensor
        shape: New shape dimensions
        offset: Offset dimensions for the view

    Returns:
        Tensor wrapping the view operation
    """
    tensor_expr = tensor.unwrap()
    call_expr = _ir_ops.view(tensor_expr, shape, offset)
    return Tensor(expr=call_expr)


def matmul(
    lhs: Tensor,
    rhs: Tensor,
    out_dtype: Optional[Union[int, DataType]] = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> Tensor:
    """Matrix multiplication with optional transpose.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        out_dtype: Output data type (optional, inferred if not provided)
        a_trans: Whether to transpose lhs
        b_trans: Whether to transpose rhs
        c_matrix_nz: C matrix non-zero flag

    Returns:
        Tensor wrapping the matmul operation
    """
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap()
    call_expr = _ir_ops.matmul(lhs_expr, rhs_expr, out_dtype, a_trans, b_trans, c_matrix_nz)
    return Tensor(expr=call_expr)


def mul(lhs: Tensor, rhs: Union[int, float, Tensor, Scalar]) -> Tensor:
    """Element-wise multiplication of tensor and tensor or scalar.

    Automatically selects between tensor.mul (tensor x tensor) and
    tensor.mul_scalar (tensor x scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

    Returns:
        Tensor wrapping the mul operation
    """
    lhs_expr = lhs.unwrap()
    if isinstance(rhs, Tensor):
        rhs_expr = rhs.unwrap()
    elif isinstance(rhs, Scalar):
        rhs_expr = rhs.unwrap()
    else:
        rhs_expr = rhs
    call_expr = _ir_ops.mul(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def mul_scalar(lhs: Tensor, rhs: Union[int, float, Expr]) -> Tensor:
    """Element-wise multiplication of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the mul_scalar operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.mul_scalar(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def add(lhs: Tensor, rhs: Union[int, float, Tensor, Scalar]) -> Tensor:
    """Element-wise addition of tensor and tensor or scalar.

    Automatically selects between tensor.add (tensor + tensor) and
    tensor.add_scalar (tensor + scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

    Returns:
        Tensor wrapping the add operation
    """
    lhs_expr = lhs.unwrap()
    if isinstance(rhs, Tensor):
        rhs_expr = rhs.unwrap()
    elif isinstance(rhs, Scalar):
        rhs_expr = rhs.unwrap()
    else:
        rhs_expr = rhs
    call_expr = _ir_ops.add(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def add_scalar(lhs: Tensor, rhs: Union[int, float, Expr]) -> Tensor:
    """Element-wise addition of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the add_scalar operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.add_scalar(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def sub(lhs: Tensor, rhs: Union[int, float, Tensor]) -> Tensor:
    """Element-wise subtraction of tensor and tensor or scalar.

    Automatically selects between tensor.sub (tensor - tensor) and
    tensor.sub_scalar (tensor - scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Tensor)

    Returns:
        Tensor wrapping the sub operation
    """
    lhs_expr = lhs.unwrap()
    if isinstance(rhs, Tensor):
        rhs_expr = rhs.unwrap()
    else:
        rhs_expr = rhs
    call_expr = _ir_ops.sub(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def sub_scalar(lhs: Tensor, rhs: Union[int, float, Expr]) -> Tensor:
    """Element-wise subtraction of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the sub_scalar operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.sub_scalar(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def div(lhs: Tensor, rhs: Union[int, float, Tensor]) -> Tensor:
    """Element-wise division of tensor and tensor or scalar.

    Automatically selects between tensor.div (tensor / tensor) and
    tensor.div_scalar (tensor / scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Tensor)

    Returns:
        Tensor wrapping the div operation
    """
    lhs_expr = lhs.unwrap()
    if isinstance(rhs, Tensor):
        rhs_expr = rhs.unwrap()
    else:
        rhs_expr = rhs
    call_expr = _ir_ops.div(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def div_scalar(lhs: Tensor, rhs: Union[int, float, Expr]) -> Tensor:
    """Element-wise division of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the div_scalar operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.div_scalar(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def maximum(lhs: Tensor, rhs: Tensor) -> Tensor:
    """Element-wise maximum of two tensors.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor

    Returns:
        Tensor wrapping the maximum operation
    """
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap()
    call_expr = _ir_ops.maximum(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def row_max(input: Tensor, axis: int = -1, keep_dim: Union[int, bool] = 1) -> Tensor:
    """Row-wise maximum reduction along specified axis.

    Args:
        input: Input tensor
        axis: Reduction axis (default: -1, last axis)
        keep_dim: Keep reduced dimension as 1

    Returns:
        Tensor wrapping the row_max operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.row_max(input_expr, axis, keep_dim)
    return Tensor(expr=call_expr)


def row_sum(input: Tensor, axis: int = -1, keep_dim: Union[int, bool] = 1) -> Tensor:
    """Row-wise sum reduction along specified axis.

    Args:
        input: Input tensor
        axis: Reduction axis (default: -1, last axis)
        keep_dim: Keep reduced dimension as 1

    Returns:
        Tensor wrapping the row_sum operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.row_sum(input_expr, axis, keep_dim)
    return Tensor(expr=call_expr)


def exp(input: Tensor) -> Tensor:
    """Element-wise exponential operation.

    Args:
        input: Input tensor

    Returns:
        Tensor wrapping the exp operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.exp(input_expr)
    return Tensor(expr=call_expr)


def cast(
    input: Tensor,
    target_type: Union[int, DataType],
    mode: Literal["round", "floor", "ceil"] = "round",
) -> Tensor:
    """Type casting operation.

    Args:
        input: Input tensor
        target_type: Target data type
        mode: Rounding mode

    Returns:
        Tensor wrapping the cast operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.cast(input_expr, target_type, mode)
    return Tensor(expr=call_expr)


def assemble(target: Tensor, source: Tensor, offset: List[Union[int, Expr]]) -> Tensor:
    """Write/update tensor values at specified offset.

    Args:
        target: Target tensor to update
        source: Source tensor to write
        offset: Offset dimensions for where to write

    Returns:
        Tensor wrapping the assemble operation
    """
    target_expr = target.unwrap()
    source_expr = source.unwrap()
    call_expr = _ir_ops.assemble(target_expr, source_expr, offset)
    return Tensor(expr=call_expr)
