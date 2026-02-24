# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor operations for PyPTO IR."""

from collections.abc import Sequence
from typing import Any, Literal

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, ScalarType, Span

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def create(
    shape: Sequence[int | Expr] | _ir_core.MakeTuple, dtype: DataType, span: Span | None = None
) -> Call:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes (int or Expr), or a MakeTuple
        dtype: Data type of tensor elements
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a new tensor
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)

    args = [shape_tuple]
    kwargs: dict[str, Any] = {"dtype": dtype}

    return _ir_core.create_op_call("tensor.create", args, kwargs, actual_span)


def read(tensor: Expr, indices: list[int | Expr] | _ir_core.MakeTuple, span: Span | None = None) -> Call:
    """Read a scalar value from a tensor at given indices.

    Args:
        tensor: Input tensor expression
        indices: List of index expressions (one per tensor dimension), or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression reading a scalar from the tensor
    """
    actual_span = _get_span_or_capture(span)

    indices_tuple = _to_make_tuple(indices, actual_span)

    args = [tensor, indices_tuple]
    return _ir_core.create_op_call("tensor.read", args, {}, actual_span)


def dim(tensor: Expr, axis: int | Expr, span: Span | None = None) -> Call:
    """Extract a shape dimension from a tensor as a scalar value.

    Args:
        tensor: Input tensor expression
        axis: Dimension index (supports negative indexing)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning the dimension size as ScalarType(INT64)
    """
    actual_span = _get_span_or_capture(span)
    axis_expr = _normalize_expr(axis, actual_span, int_dtype=DataType.INDEX)
    args = [tensor, axis_expr]
    return _ir_core.create_op_call("tensor.dim", args, {}, actual_span)


def view(
    tensor: Expr,
    shape: list[int | Expr] | _ir_core.MakeTuple,
    offset: list[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Create a view/slice of a tensor with new shape and offset.

    Args:
        tensor: Input tensor expression
        shape: New shape dimensions, or a MakeTuple
        offset: Offset dimensions for the view, or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tensor view
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)
    offset_tuple = _to_make_tuple(offset, actual_span)

    args = [tensor, shape_tuple, offset_tuple]
    return _ir_core.create_op_call("tensor.view", args, {}, actual_span)


def matmul(
    lhs: Expr,
    rhs: Expr,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
    span: Span | None = None,
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
    args = [lhs, rhs]

    kwargs: dict[str, Any] = {
        "a_trans": a_trans,
        "b_trans": b_trans,
        "c_matrix_nz": c_matrix_nz,
    }
    if out_dtype is not None:
        kwargs["out_dtype"] = out_dtype

    return _ir_core.create_op_call("tensor.matmul", args, kwargs, actual_span)


def mul(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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


def mul_scalar(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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


def add(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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


def add_scalar(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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


def sub(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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


def sub_scalar(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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


def div(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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


def div_scalar(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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


def maximum(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
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


def row_max(input: Expr, span: Span | None = None) -> Call:
    """Row-wise max reduction (reduces along last axis, keeps dim).

    Args:
        input: Input tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise max reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.row_max", [input], {}, actual_span)


def row_sum(input: Expr, span: Span | None = None) -> Call:
    """Row-wise sum reduction (reduces along last axis, keeps dim).

    Args:
        input: Input tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise sum reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.row_sum", [input], {}, actual_span)


def exp(input: Expr, span: Span | None = None) -> Call:
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
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
    span: Span | None = None,
) -> Call:
    """Type casting operation.

    Args:
        input: Input tensor
        target_type: Target data type
        mode: Rounding mode in None(0), RINT(1), ROUND(2), FLOOR(3), CEIL(4), TRUNC(5), ODD(6)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for type casting
    """
    modes = {"none": 0, "rint": 1, "round": 2, "floor": 3, "ceil": 4, "trunc": 5, "odd": 6}
    mode_val = modes.get(mode)
    if mode_val is None:
        raise ValueError(f"Invalid rounding mode '{mode}'. Expected one of {list(modes.keys())}.")

    actual_span = _get_span_or_capture(span)

    args = [input]
    kwargs: dict[str, Any] = {
        "target_type": target_type,
        "mode": mode_val,
    }

    return _ir_core.create_op_call("tensor.cast", args, kwargs, actual_span)


def assemble(
    target: Expr, source: Expr, offset: list[int | Expr] | _ir_core.MakeTuple, span: Span | None = None
) -> Call:
    """Write/update tensor values at specified offset.

    Args:
        target: Target tensor to update
        source: Source tensor to write
        offset: Offset dimensions for where to write, or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tensor assembly
    """
    actual_span = _get_span_or_capture(span)

    offset_tuple = _to_make_tuple(offset, actual_span)

    args = [target, source, offset_tuple]
    return _ir_core.create_op_call("tensor.assemble", args, {}, actual_span)


def reshape(tensor: Expr, shape: list[int | Expr] | _ir_core.MakeTuple, span: Span | None = None) -> Call:
    """Reshape tensor to new shape.

    Args:
        tensor: Input tensor expression
        shape: New shape dimensions, or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tensor reshape
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)

    args = [tensor, shape_tuple]
    return _ir_core.create_op_call("tensor.reshape", args, {}, actual_span)


def transpose(tensor: Expr, axis1: int, axis2: int, span: Span | None = None) -> Call:
    """Transpose tensor by swapping two axes.

    Args:
        tensor: Input tensor expression
        axis1: First axis to swap (supports negative indexing)
        axis2: Second axis to swap (supports negative indexing)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tensor transpose
    """
    actual_span = _get_span_or_capture(span)
    axis1_expr = ConstInt(axis1, DataType.INDEX, actual_span)
    axis2_expr = ConstInt(axis2, DataType.INDEX, actual_span)

    args = [tensor, axis1_expr, axis2_expr]

    return _ir_core.create_op_call("tensor.transpose", args, {}, actual_span)
