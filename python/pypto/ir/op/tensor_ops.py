# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor operations for PyPTO IR."""

from typing import List, Literal, Union

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, ScalarType, Span

from ..utils import _normalize_expr


def create(shape: List[int], dtype: DataType) -> Call:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes
        dtype: Data type of tensor elements

    Returns:
        Call expression creating a new tensor
    """
    span = Span.unknown()
    args = []

    # Add shape dimensions
    for dim in shape:
        args.append(ConstInt(dim, DataType.INT32, span))

    # Add dtype as last argument
    args.append(ConstInt(dtype.code(), DataType.INT32, span))

    return _ir_core.create_op_call("tensor.create", args, span)


def view(tensor: Expr, shape: List[Union[int, Expr]], offset: List[Union[int, Expr]]) -> Call:
    """Create a view/slice of a tensor with new shape and offset.

    Args:
        tensor: Input tensor expression
        shape: New shape dimensions
        offset: Offset dimensions for the view

    Returns:
        Call expression creating a tensor view
    """
    span = Span.unknown()
    args = [tensor]

    # Add the number of shape dimensions as a ConstInt
    # This allows the C++ side to correctly split shape and offset arguments
    args.append(ConstInt(len(shape), DataType.INT32, span))

    # Add shape dimensions
    for dim in shape:
        args.append(_normalize_expr(dim, int_dtype=DataType.INT32))

    # Add offset dimensions
    for off in offset:
        args.append(_normalize_expr(off, int_dtype=DataType.INT32))

    return _ir_core.create_op_call("tensor.view", args, span)


def matmul(  # noqa: PLR0913
    lhs: Expr,
    rhs: Expr,
    out_dtype: Union[int, DataType],
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> Call:
    """Matrix multiplication with optional transpose.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        out_dtype: Output data type
        a_trans: Whether to transpose lhs
        b_trans: Whether to transpose rhs
        c_matrix_nz: C matrix non-zero flag

    Returns:
        Call expression for matrix multiplication
    """
    span = Span.unknown()

    # Convert out_dtype to int if it's a DataType enum
    if isinstance(out_dtype, DataType):
        out_dtype = out_dtype.code()

    args = [
        lhs,
        rhs,
        ConstInt(out_dtype, DataType.INT32, span),
        ConstInt(1 if a_trans else 0, DataType.INT32, span),
        ConstInt(1 if b_trans else 0, DataType.INT32, span),
        ConstInt(1 if c_matrix_nz else 0, DataType.INT32, span),
    ]

    return _ir_core.create_op_call("tensor.matmul", args, span)


def mul(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise multiplication of tensor and tensor or scalar.

    Automatically selects between tensor.mul (tensor x tensor) and
    tensor.mul_scalar (tensor x scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)

    Returns:
        Call expression for element-wise multiplication
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )

    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.mul_scalar", [lhs, rhs_expr], span)
    else:
        return _ir_core.create_op_call("tensor.mul", [lhs, rhs_expr], span)


def mul_scalar(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise multiplication of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Call expression for element-wise multiplication with scalar
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.mul_scalar", [lhs, rhs_expr], span)


def add(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise addition of tensor and tensor or scalar.

    Automatically selects between tensor.add (tensor + tensor) and
    tensor.add_scalar (tensor + scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)

    Returns:
        Call expression for element-wise addition
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )

    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.add_scalar", [lhs, rhs_expr], span)
    else:
        return _ir_core.create_op_call("tensor.add", [lhs, rhs_expr], span)


def add_scalar(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise addition of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Call expression for element-wise addition with scalar
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.add_scalar", [lhs, rhs_expr], span)


def sub(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise subtraction of tensor and tensor or scalar.

    Automatically selects between tensor.sub (tensor - tensor) and
    tensor.sub_scalar (tensor - scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)

    Returns:
        Call expression for element-wise subtraction
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )

    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.sub_scalar", [lhs, rhs_expr], span)
    else:
        return _ir_core.create_op_call("tensor.sub", [lhs, rhs_expr], span)


def sub_scalar(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise subtraction of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Call expression for element-wise subtraction with scalar
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.sub_scalar", [lhs, rhs_expr], span)


def div(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise division of tensor and tensor or scalar.

    Automatically selects between tensor.div (tensor / tensor) and
    tensor.div_scalar (tensor / scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)

    Returns:
        Call expression for element-wise division
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )

    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.div_scalar", [lhs, rhs_expr], span)
    else:
        return _ir_core.create_op_call("tensor.div", [lhs, rhs_expr], span)


def div_scalar(lhs: Expr, rhs: Union[int, float, Expr]) -> Call:
    """Element-wise division of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Call expression for element-wise division with scalar
    """
    span = Span.unknown()
    rhs_expr = (
        _normalize_expr(rhs, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.div_scalar", [lhs, rhs_expr], span)


def maximum(lhs: Expr, rhs: Expr) -> Call:
    """Element-wise maximum of two tensors.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor

    Returns:
        Call expression for element-wise maximum
    """
    span = Span.unknown()
    return _ir_core.create_op_call("tensor.maximum", [lhs, rhs], span)


def row_max(input: Expr, axis: int = -1, keep_dim: Union[int, bool] = 1) -> Call:
    """Row-wise maximum reduction along specified axis.

    Args:
        input: Input tensor
        axis: Reduction axis (default: -1, last axis)
        keep_dim: Keep reduced dimension as 1

    Returns:
        Call expression for row-wise maximum reduction
    """
    span = Span.unknown()
    keep_dim_val = 1 if keep_dim else 0
    args = [
        input,
        ConstInt(axis, DataType.INT32, span),
        ConstInt(keep_dim_val, DataType.INT32, span),
    ]
    return _ir_core.create_op_call("tensor.row_max", args, span)


def row_sum(input: Expr, axis: int = -1, keep_dim: Union[int, bool] = 1) -> Call:
    """Row-wise sum reduction along specified axis.

    Args:
        input: Input tensor
        axis: Reduction axis (default: -1, last axis)
        keep_dim: Keep reduced dimension as 1

    Returns:
        Call expression for row-wise sum reduction
    """
    span = Span.unknown()
    keep_dim_val = 1 if keep_dim else 0
    args = [
        input,
        ConstInt(axis, DataType.INT32, span),
        ConstInt(keep_dim_val, DataType.INT32, span),
    ]
    return _ir_core.create_op_call("tensor.row_sum", args, span)


def exp(input: Expr) -> Call:
    """Element-wise exponential operation.

    Args:
        input: Input tensor

    Returns:
        Call expression for element-wise exponential
    """
    span = Span.unknown()
    return _ir_core.create_op_call("tensor.exp", [input], span)


def cast(
    input: Expr,
    target_type: Union[int, DataType],
    mode: Literal["round", "floor", "ceil"] = "round",
) -> Call:
    """Type casting operation.

    Args:
        input: Input tensor
        targetType: Target data type
        mode: Rounding mode (e.g., 'round', 'floor', 'ceil')

    Returns:
        Call expression for type casting
    """

    modes = {
        "round": 0,
        "floor": 1,
        "ceil": 2,
    }
    mode_val = modes.get(mode)
    if mode_val is None:
        raise ValueError(f"Invalid rounding mode '{mode}'. Expected one of {list(modes.keys())}.")

    span = Span.unknown()

    # Convert target_type to int if it's a DataType enum
    if isinstance(target_type, DataType):
        target_type = target_type.code()

    args = [
        input,
        ConstInt(target_type, DataType.INT32, span),
        ConstInt(mode_val, DataType.INT32, span),
    ]

    return _ir_core.create_op_call("tensor.cast", args, span)


def assemble(target: Expr, source: Expr, offset: List[Union[int, Expr]]) -> Call:
    """Write/update tensor values at specified offset.

    Args:
        target: Target tensor to update
        source: Source tensor to write
        offset: Offset dimensions for where to write

    Returns:
        Call expression for tensor assembly
    """
    span = Span.unknown()
    args = [target, source]

    # Add offset dimensions
    offset = [_normalize_expr(off, int_dtype=DataType.INT32) for off in offset]
    return _ir_core.create_op_call("tensor.assemble", args, span)
