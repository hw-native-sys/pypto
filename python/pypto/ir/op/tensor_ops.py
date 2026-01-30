# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor operations for PyPTO IR."""

from typing import Any, Dict, List, Literal, Optional, Union

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, ScalarType, Span

from ..utils import _get_span_or_capture, _normalize_expr


def create(shape: List[int], dtype: DataType, span: Optional[Span] = None) -> Call:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes
        dtype: Data type of tensor elements
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a new tensor
    """
    actual_span = _get_span_or_capture(span)

    # Convert shape to Expr list
    args = [ConstInt(dim, DataType.INT32, actual_span) for dim in shape]

    kwargs: Dict[str, Any] = {"dtype": dtype}

    return _ir_core.create_op_call("tensor.create", args, kwargs, actual_span)


def view(
    tensor: Expr, shape: List[Union[int, Expr]], offset: List[Union[int, Expr]], span: Optional[Span] = None
) -> Call:
    """Create a view/slice of a tensor with new shape and offset.

    Args:
        tensor: Input tensor expression
        shape: New shape dimensions
        offset: Offset dimensions for the view
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tensor view
    """
    actual_span = _get_span_or_capture(span)
    args = [tensor]

    # Add the number of shape dimensions as a ConstInt
    # This allows the C++ side to correctly split shape and offset arguments
    args.append(ConstInt(len(shape), DataType.INT32, actual_span))

    # Add shape dimensions
    for dim in shape:
        args.append(_normalize_expr(dim, actual_span, int_dtype=DataType.INT32))

    # Add offset dimensions
    for off in offset:
        args.append(_normalize_expr(off, actual_span, int_dtype=DataType.INT32))

    return _ir_core.create_op_call("tensor.view", args, {}, actual_span)


def matmul(
    lhs: Expr,
    rhs: Expr,
    out_dtype: Optional[Union[int, DataType]] = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
    span: Optional[Span] = None,
) -> Call:
    """Matrix multiplication with optional transpose.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        out_dtype: Output data type (optional, inferred if not provided)
        a_trans: Whether to transpose lhs
        b_trans: Whether to transpose rhs
        c_matrix_nz: C matrix non-zero flag
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for matrix multiplication
    """
    actual_span = _get_span_or_capture(span)

    # Only Expr arguments
    args = [lhs, rhs]

    kwargs: Dict[str, Any] = {
        "a_trans": a_trans,
        "b_trans": b_trans,
        "c_matrix_nz": c_matrix_nz,
    }
    if out_dtype is not None:
        kwargs["out_dtype"] = out_dtype

    return _ir_core.create_op_call("tensor.matmul", args, kwargs, actual_span)


def mul(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise multiplication of tensor and tensor or scalar.

    Automatically selects between tensor.mul (tensor x tensor) and
    tensor.mul_scalar (tensor x scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )

    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.mul_scalar", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.mul", [lhs, rhs_expr], {}, actual_span)


def mul_scalar(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise multiplication of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise multiplication with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.mul_scalar", [lhs, rhs_expr], {}, actual_span)


def add(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise addition of tensor and tensor or scalar.

    Automatically selects between tensor.add (tensor + tensor) and
    tensor.add_scalar (tensor + scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )

    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.add_scalar", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.add", [lhs, rhs_expr], {}, actual_span)


def add_scalar(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise addition of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise addition with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.add_scalar", [lhs, rhs_expr], {}, actual_span)


def sub(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise subtraction of tensor and tensor or scalar.

    Automatically selects between tensor.sub (tensor - tensor) and
    tensor.sub_scalar (tensor - scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )

    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.sub_scalar", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.sub", [lhs, rhs_expr], {}, actual_span)


def sub_scalar(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise subtraction of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise subtraction with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.sub_scalar", [lhs, rhs_expr], {}, actual_span)


def div(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise division of tensor and tensor or scalar.

    Automatically selects between tensor.div (tensor / tensor) and
    tensor.div_scalar (tensor / scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise division
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )

    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.div_scalar", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.div", [lhs, rhs_expr], {}, actual_span)


def div_scalar(lhs: Expr, rhs: Union[int, float, Expr], span: Optional[Span] = None) -> Call:
    """Element-wise division of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise division with scalar
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.div_scalar", [lhs, rhs_expr], {}, actual_span)


def maximum(lhs: Expr, rhs: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise maximum of two tensors.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise maximum
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.maximum", [lhs, rhs], {}, actual_span)


def row_max(input: Expr, axis: int = -1, keep_dim: Union[int, bool] = 1, span: Optional[Span] = None) -> Call:
    """Row-wise maximum reduction along specified axis.

    Args:
        input: Input tensor
        axis: Reduction axis (default: -1, last axis)
        keep_dim: Keep reduced dimension as 1
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise maximum reduction
    """
    actual_span = _get_span_or_capture(span)

    args = [input]
    kwargs: Dict[str, Any] = {
        "axis": axis,
        "keep_dim": bool(keep_dim),
    }

    return _ir_core.create_op_call("tensor.row_max", args, kwargs, actual_span)


def row_sum(input: Expr, axis: int = -1, keep_dim: Union[int, bool] = 1, span: Optional[Span] = None) -> Call:
    """Row-wise sum reduction along specified axis.

    Args:
        input: Input tensor
        axis: Reduction axis (default: -1, last axis)
        keep_dim: Keep reduced dimension as 1
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise sum reduction
    """
    actual_span = _get_span_or_capture(span)

    args = [input]
    kwargs: Dict[str, Any] = {
        "axis": axis,
        "keep_dim": bool(keep_dim),
    }

    return _ir_core.create_op_call("tensor.row_sum", args, kwargs, actual_span)


def exp(input: Expr, span: Optional[Span] = None) -> Call:
    """Element-wise exponential operation.

    Args:
        input: Input tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise exponential
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.exp", [input], {}, actual_span)


def cast(
    input: Expr,
    target_type: Union[int, DataType],
    mode: Literal["round", "floor", "ceil"] = "round",
    span: Optional[Span] = None,
) -> Call:
    """Type casting operation.

    Args:
        input: Input tensor
        target_type: Target data type
        mode: Rounding mode
        span: Optional source span for debugging (auto-captured if not provided)

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

    actual_span = _get_span_or_capture(span)

    args = [input]
    kwargs: Dict[str, Any] = {
        "target_type": target_type,
        "mode": mode_val,
    }

    return _ir_core.create_op_call("tensor.cast", args, kwargs, actual_span)


def assemble(target: Expr, source: Expr, offset: List[Union[int, Expr]], span: Optional[Span] = None) -> Call:
    """Write/update tensor values at specified offset.

    Args:
        target: Target tensor to update
        source: Source tensor to write
        offset: Offset dimensions for where to write
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tensor assembly
    """
    actual_span = _get_span_or_capture(span)
    args = [target, source]

    # Add offset dimensions
    for off in offset:
        args.append(_normalize_expr(off, actual_span, int_dtype=DataType.INT32))

    return _ir_core.create_op_call("tensor.assemble", args, {}, actual_span)
