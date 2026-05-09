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
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstFloat, ConstInt, Expr, PadValue, ScalarType, Span, TensorLayout

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple, resolve_cast_mode
from ._pad_value import normalize_pad_value


def create(
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    dtype: DataType,
    layout: TensorLayout = TensorLayout.ND,
    manual_dep: bool = False,
    span: Span | None = None,
) -> Call:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes (int or Expr), or a MakeTuple
        dtype: Data type of tensor elements
        layout: Tensor layout (default: ND)
        manual_dep: **Internal-only.** When True, the codegen marks this
            ``tensor.create`` call so the runtime skips OverlapMap dep
            tracking. Set programmatically by the ``InjectGMPipeBuffer``
            pass for compiler-injected scratch buffers; user code should
            never set this — order kernel calls with ``pl.no_dep(...)``
            or ``with pl.manual_scope():`` instead.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a new tensor
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)

    args = [shape_tuple]
    kwargs: dict[str, Any] = {"dtype": dtype, "layout": layout}
    if manual_dep:
        kwargs["manual_dep"] = True

    return _ir_core.create_op_call("tensor.create", args, kwargs, actual_span)


create_tensor = create


def full(
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    dtype: DataType,
    value: int | float,
    span: Span | None = None,
) -> Call:
    """Create a tensor of specified shape filled with a constant value.

    Args:
        shape: Shape of the tensor (list of int/Expr, or MakeTuple)
        dtype: Data type of tensor elements
        value: Filling scalar value (int or float)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression that returns a TensorType
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    if isinstance(value, int):
        value_expr = ConstInt(value, dtype, actual_span)
    else:
        value_expr = ConstFloat(value, dtype, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype}
    return _ir_core.create_op_call("tensor.full", [shape_tuple, value_expr], kwargs, actual_span)


def ci(
    start: int | Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    dtype: DataType = DataType.INT32,
    descending: bool = False,
    span: Span | None = None,
) -> Call:
    """Generate a contiguous integer sequence into a tensor (lowers to tile.ci).

    Note:
        Lowers to ``pto.tci`` which only populates the first row. Leading
        dimensions must be 1 — prefer shapes of the form ``[1, N]``.

    Args:
        start: Starting integer (plain int or scalar Expr). Must match ``dtype``.
        shape: Destination shape (leading dims must be 1, innermost dim != 1).
        dtype: Destination dtype. One of {INT16, INT32}.
        descending: If True, generate a descending sequence.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression that returns a TensorType.
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(start, Expr):
        if isinstance(start, ConstInt) and start.dtype != dtype:
            start_expr = ConstInt(start.value, dtype, actual_span)
        else:
            start_expr = start
    else:
        start_expr = ConstInt(start, dtype, actual_span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype, "descending": descending}
    return _ir_core.create_op_call("tensor.ci", [start_expr, shape_tuple], kwargs, actual_span)


arange = ci


def read(
    tensor: Expr, indices: Expr | list[int | Expr] | _ir_core.MakeTuple, span: Span | None = None
) -> Call:
    """Read a scalar value from a tensor at given indices.

    Args:
        tensor: Input tensor expression
        indices: A single index expression (for 1-D flat access), a list of index
            expressions (one per tensor dimension), or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression reading a scalar from the tensor
    """
    actual_span = _get_span_or_capture(span)

    # Allow a bare Expr as a flat 1-D index for backwards compatibility
    if isinstance(indices, Expr) and not isinstance(indices, _ir_core.MakeTuple):
        indices = [indices]

    indices_tuple = _to_make_tuple(indices, actual_span)

    args = [tensor, indices_tuple]
    return _ir_core.create_op_call("tensor.read", args, {}, actual_span)


def write(
    tensor: Expr,
    indices: Expr | list[int | Expr] | _ir_core.MakeTuple,
    value: Expr,
    span: Span | None = None,
) -> Call:
    """Write a scalar value into a tensor at given indices.

    Args:
        tensor: Destination tensor expression (TensorType)
        indices: A single index expression (for 1-D flat access), a list of index
            expressions (one per tensor dimension), or a MakeTuple
        value: Scalar value to write (ScalarType, must match tensor dtype)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning the tensor (for chaining)
    """
    actual_span = _get_span_or_capture(span)

    # Allow a bare Expr as a flat 1-D index for backwards compatibility
    if isinstance(indices, Expr) and not isinstance(indices, _ir_core.MakeTuple):
        indices = [indices]

    indices_tuple = _to_make_tuple(indices, actual_span)

    args = [tensor, indices_tuple, value]
    return _ir_core.create_op_call("tensor.write", args, {}, actual_span)


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


def slice(
    tensor: Expr,
    shape: list[int | Expr] | _ir_core.MakeTuple,
    offset: list[int | Expr] | _ir_core.MakeTuple,
    valid_shape: list[int | Expr] | _ir_core.MakeTuple | None = None,
    pad_value: PadValue | int | float | None = None,
    span: Span | None = None,
) -> Call:
    """Create a slice of a tensor with new shape and offset.

    Args:
        tensor: Input tensor expression
        shape: New shape dimensions, or a MakeTuple
        offset: Offset dimensions for the slice, or a MakeTuple
        valid_shape: Valid shape dimensions (optional, defaults to empty)
        pad_value: Optional padding mode for out-of-valid-shape elements.
            Accepts ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or
            the literal sugars ``0``, ``math.inf``, ``-math.inf`` (normalized
            via :func:`normalize_pad_value`). ``PadValue.null`` is passed
            through unchanged and means "no padding". When omitted (``None``),
            the kwarg is not forwarded — the deducer defaults to
            ``PadValue.null``.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tensor slice
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)
    offset_tuple = _to_make_tuple(offset, actual_span)

    args = [tensor, shape_tuple, offset_tuple]
    if valid_shape is not None:
        args.append(_to_make_tuple(valid_shape, actual_span))

    kwargs: dict[str, Any] = {}
    if pad_value is not None:
        # PadValue.null is a legal "no padding" signal for slice (unlike
        # fillpad, which requires a real padding mode). Pass it through;
        # normalize the rest via the shared helper so numeric sugar and
        # validation match tensor.fillpad exactly.
        kwargs["pad_value"] = pad_value if pad_value is PadValue.null else normalize_pad_value(pad_value)

    return _ir_core.create_op_call("tensor.slice", args, kwargs, actual_span)


def fillpad(
    tensor: Expr, pad_value: PadValue | int | float = PadValue.zero, span: Span | None = None
) -> Call:
    """Fill invalid tensor view elements with the specified padding value.

    Args:
        tensor: Input tensor expression
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Other values
            raise — the hardware only supports the three padding modes.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a padded tensor
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "tensor.fillpad", [tensor], {"pad_value": normalize_pad_value(pad_value)}, actual_span
    )


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


def matmul_acc(
    acc: Expr,
    lhs: Expr,
    rhs: Expr,
    a_trans: bool = False,
    b_trans: bool = False,
    span: Span | None = None,
) -> Call:
    """Matrix multiplication with accumulation: acc = acc + lhs @ rhs.

    Args:
        acc: Accumulator tensor
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        a_trans: Whether to transpose lhs
        b_trans: Whether to transpose rhs
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for matrix multiplication with accumulation
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"a_trans": a_trans, "b_trans": b_trans}
    return _ir_core.create_op_call("tensor.matmul_acc", [acc, lhs, rhs], kwargs, actual_span)


def mul(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise multiplication of tensor and tensor or scalar.

    Automatically selects between tensor.mul (tensor x tensor) and
    tensor.muls (tensor x scalar) based on the rhs type.

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
        return _ir_core.create_op_call("tensor.muls", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.mul", [lhs, rhs_expr], {}, actual_span)


def muls(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
    return _ir_core.create_op_call("tensor.muls", [lhs, rhs_expr], {}, actual_span)


def add(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise addition of tensor and tensor or scalar.

    Automatically selects between tensor.add (tensor + tensor) and
    tensor.adds (tensor + scalar) based on the rhs type.

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
        return _ir_core.create_op_call("tensor.adds", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.add", [lhs, rhs_expr], {}, actual_span)


def adds(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
    return _ir_core.create_op_call("tensor.adds", [lhs, rhs_expr], {}, actual_span)


def sub(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise subtraction of tensor and tensor or scalar.

    Automatically selects between tensor.sub (tensor - tensor) and
    tensor.subs (tensor - scalar) based on the rhs type.

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
        return _ir_core.create_op_call("tensor.subs", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.sub", [lhs, rhs_expr], {}, actual_span)


def subs(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
    return _ir_core.create_op_call("tensor.subs", [lhs, rhs_expr], {}, actual_span)


def div(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise division of tensor and tensor or scalar.

    Automatically selects between tensor.div (tensor / tensor) and
    tensor.divs (tensor / scalar) based on the rhs type.

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
        return _ir_core.create_op_call("tensor.divs", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.div", [lhs, rhs_expr], {}, actual_span)


def divs(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
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
    return _ir_core.create_op_call("tensor.divs", [lhs, rhs_expr], {}, actual_span)


def maximum(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise maximum of tensor and tensor or scalar.

    Emits a single ``tensor.maximum`` op; the conversion pass dispatches to
    ``tile.maximum`` (tensor-vs-tensor) or ``tile.maximums`` (tensor-vs-scalar)
    based on the rhs operand type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise maximum
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.maximum", [lhs, rhs_expr], {}, actual_span)


def minimum(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    """Element-wise minimum of tensor and tensor or scalar.

    Emits a single ``tensor.minimum`` op; the conversion pass dispatches to
    ``tile.minimum`` (tensor-vs-tensor) or ``tile.minimums`` (tensor-vs-scalar)
    based on the rhs operand type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise minimum
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.minimum", [lhs, rhs_expr], {}, actual_span)


def cmp(lhs: Expr, rhs: int | float | Expr, cmp_type: int = 0, span: Span | None = None) -> Call:
    """Element-wise comparison of tensor and tensor or scalar (returns 0/1 tensor).

    Emits a single ``tensor.cmp`` op; the conversion pass dispatches to
    ``tile.cmp`` (tensor-vs-tensor) or ``tile.cmps`` (tensor-vs-scalar)
    based on the rhs operand type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Expr)
        cmp_type: Comparison type code (0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise comparison (0/1 tensor)
    """
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    return _ir_core.create_op_call("tensor.cmp", [lhs, rhs_expr], {"cmp_type": cmp_type}, actual_span)


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


def row_min(input: Expr, span: Span | None = None) -> Call:
    """Row-wise min reduction (reduces along last axis, keeps dim).

    Args:
        input: Input tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise min reduction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.row_min", [input], {}, actual_span)


def row_expand(target: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise expansion: expand row_vec [M, 1] to target shape [M, N].

    Args:
        target: Target tensor defining output shape (TensorType [M, N])
        row_vec: Row vector to expand (TensorType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise expansion
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.row_expand", [target, row_vec], {}, actual_span)


def row_expand_mul(tensor: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast multiplication.

    Multiplies each row of the tensor by the corresponding row vector value.
    tensor[i, :] * row_vec[i, 0] for all i.

    Args:
        tensor: Input tensor (TensorType [M, N])
        row_vec: Row vector (TensorType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.row_expand_mul", [tensor, row_vec], {}, actual_span)


def row_expand_div(tensor: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast division.

    Divides each row of the tensor by the corresponding row vector value.
    tensor[i, :] / row_vec[i, 0] for all i.

    Args:
        tensor: Input tensor (TensorType [M, N])
        row_vec: Row vector (TensorType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast division
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.row_expand_div", [tensor, row_vec], {}, actual_span)


def row_expand_add(tensor: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast addition.

    Adds a row vector to each row of the tensor.
    tensor[i, :] + row_vec[i, 0] for all i.

    Args:
        tensor: Input tensor (TensorType [M, N])
        row_vec: Row vector (TensorType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast addition
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.row_expand_add", [tensor, row_vec], {}, actual_span)


def row_expand_sub(tensor: Expr, row_vec: Expr, span: Span | None = None) -> Call:
    """Row-wise broadcast subtraction.

    Subtracts a row vector from each row of the tensor.
    tensor[i, :] - row_vec[i, 0] for all i.

    Args:
        tensor: Input tensor (TensorType [M, N])
        row_vec: Row vector (TensorType [M, 1])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for row-wise broadcast subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.row_expand_sub", [tensor, row_vec], {}, actual_span)


def col_expand_mul(tensor: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Column-wise broadcast multiplication.

    Multiplies each column of the tensor by the corresponding column vector value.
    tensor[:, j] * col_vec[0, j] for all j.

    Args:
        tensor: Input tensor (TensorType [M, N])
        col_vec: Column vector (TensorType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast multiplication
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.col_expand_mul", [tensor, col_vec], {}, actual_span)


def col_expand(tensor: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Column-wise expansion: expand col_vec [1, N] to target shape [M, N].

    Args:
        tensor: Target tensor defining output shape (TensorType [M, N])
        col_vec: Column vector to expand (TensorType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise expansion
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.col_expand", [tensor, col_vec], {}, actual_span)


def col_expand_sub(tensor: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Column-wise broadcast subtraction.

    Subtracts a column vector from each column of the tensor.
    tensor[:, j] - col_vec[0, j] for all j.

    Args:
        tensor: Input tensor (TensorType [M, N])
        col_vec: Column vector (TensorType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast subtraction
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.col_expand_sub", [tensor, col_vec], {}, actual_span)


def col_expand_div(tensor: Expr, col_vec: Expr, span: Span | None = None) -> Call:
    """Column-wise broadcast division.

    Divides each column of the tensor by the corresponding column vector value.
    tensor[:, j] / col_vec[0, j] for all j.

    Args:
        tensor: Input tensor (TensorType [M, N])
        col_vec: Column vector (TensorType [1, N])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise broadcast division
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.col_expand_div", [tensor, col_vec], {}, actual_span)


def expands(target: Expr, scalar: int | float | Expr, span: Span | None = None) -> Call:
    """Expand scalar to target tensor shape.

    Args:
        target: Target tensor defining output shape (TensorType)
        scalar: Scalar value to expand (int/float/Expr with ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for scalar expansion
    """
    actual_span = _get_span_or_capture(span)
    scalar_expr = (
        _normalize_expr(scalar, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(scalar, Expr)
        else scalar
    )
    return _ir_core.create_op_call("tensor.expands", [target, scalar_expr], {}, actual_span)


def expand_clone(
    src: Expr,
    target: Expr,
    span: Span | None = None,
) -> Call:
    """Expand tensor to new shape.

    Args:
        src: Source tensor expression
        target: Target tensor defining output shape
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tensor expand_clone
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.expand_clone", [src, target], {}, actual_span)


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


def sin(input: Expr, span: Span | None = None) -> Call:
    """Element-wise sine operation (input in radians). FP32-only.

    Args:
        input: Input tensor (must be FP32; cast explicitly otherwise)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise sine
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.sin", [input], {}, actual_span)


def cos(input: Expr, span: Span | None = None) -> Call:
    """Element-wise cosine operation (input in radians). FP32-only.

    Args:
        input: Input tensor (must be FP32; cast explicitly otherwise)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise cosine
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.cos", [input], {}, actual_span)


def neg(input: Expr, span: Span | None = None) -> Call:
    """Element-wise negation operation.

    Args:
        input: Input tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise negation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.neg", [input], {}, actual_span)


def abs(input: Expr, span: Span | None = None) -> Call:
    """Element-wise absolute value operation.

    Args:
        input: Input tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise absolute value
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.abs", [input], {}, actual_span)


def recip(input: Expr, span: Span | None = None) -> Call:
    """Element-wise reciprocal (1/x) operation.

    Args:
        input: Input tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.recip", [input], {}, actual_span)


def sqrt(input: Expr, span: Span | None = None) -> Call:
    """Element-wise square root operation.

    Args:
        input: Input tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise square root
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.sqrt", [input], {}, actual_span)


def rsqrt(input: Expr, high_precision: bool = False, span: Span | None = None) -> Call:
    """Element-wise reciprocal square root operation.

    Args:
        input: Input tensor
        high_precision: If True, lower to the high-precision PTO path that
            uses a scratch buffer (compiler-allocated during conversion).
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for element-wise reciprocal square root
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict = {"high_precision": high_precision} if high_precision else {}
    return _ir_core.create_op_call("tensor.rsqrt", [input], kwargs, actual_span)


def cast(
    input: Expr,
    target_type: int | DataType,
    mode: str | int = "round",
    span: Span | None = None,
) -> Call:
    """Type casting operation.

    Args:
        input: Input tensor
        target_type: Target data type
        mode: Rounding mode — string name ("none", "rint", "round", "floor",
              "ceil", "trunc", "odd") or int (0–6)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for type casting
    """
    mode_val = resolve_cast_mode(mode)

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


def concat(
    src0: Expr,
    src1: Expr,
    span: Span | None = None,
) -> Call:
    """Concatenate two tensors along the column dimension.

    Args:
        src0: First source tensor (TensorType)
        src1: Second source tensor (TensorType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for column-wise concatenation
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.concat", [src0, src1], {}, actual_span)


def reshape(
    tensor: Expr,
    shape: list[int | Expr] | _ir_core.MakeTuple,
    valid_shape: list[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Reshape tensor to new shape.

    Args:
        tensor: Input tensor expression
        shape: New shape dimensions, or a MakeTuple
        valid_shape: Valid shape dimensions (optional, defaults to empty)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tensor reshape
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)

    args = [tensor, shape_tuple]
    if valid_shape is not None:
        args.append(_to_make_tuple(valid_shape, actual_span))
    return _ir_core.create_op_call("tensor.reshape", args, {}, actual_span)


def transpose(
    tensor: Expr,
    axis1: int | ConstInt,
    axis2: int | ConstInt,
    valid_shape: list[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Transpose tensor by swapping two axes.

    Args:
        tensor: Input tensor expression
        axis1: First axis to swap as an int or ConstInt (supports negative indexing)
        axis2: Second axis to swap as an int or ConstInt (supports negative indexing)
        valid_shape: Valid shape dimensions (optional, defaults to empty)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tensor transpose
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(axis1, ConstInt):
        axis1_expr = axis1
    elif isinstance(axis1, int):
        axis1_expr = ConstInt(axis1, DataType.INDEX, actual_span)
    else:
        raise TypeError(f"axis1 must be int or ConstInt, got {type(axis1)}")

    if isinstance(axis2, ConstInt):
        axis2_expr = axis2
    elif isinstance(axis2, int):
        axis2_expr = ConstInt(axis2, DataType.INDEX, actual_span)
    else:
        raise TypeError(f"axis2 must be int or ConstInt, got {type(axis2)}")

    args = [tensor, axis1_expr, axis2_expr]
    if valid_shape is not None:
        args.append(_to_make_tuple(valid_shape, actual_span))
    return _ir_core.create_op_call("tensor.transpose", args, {}, actual_span)


def as_layout(
    tensor: Expr,
    layout: TensorLayout,
    span: Span | None = None,
) -> Call:
    """Flip ``tensor``'s layout tag over the same physical memory (RFC #1300 §3.3).

    .. note::
        Internal IR builder — this op is not exposed via ``pypto.language``.
        Passes (e.g. ``LowerTransposeLoadParamLayout`` in P6) inject
        ``tensor.as_layout`` at orch ↔ InCore call sites to bridge ND ↔ DN views
        over the same physical buffer. The op emits no PTOAS instruction;
        downstream ``make_tensor_view`` consumes the new view directly.

    The trailing-two-dim shape swap that comes with a cross-layout flip is
    mechanical (RFC §4.2: row-major ``[..., a, b]`` ND ≡ ``[..., b, a]``
    DN-packed) and derived from the source — callers don't pass a target
    shape. For shape changes, use ``tensor.reshape``.

    Validity (enforced by ``DeduceTensorAsLayoutType``):

    1. ``layout`` must not be ``NZ`` (NZ is tile-only and fractal).
    2. ``tensor`` must be packed canonical or bare (strided sub-views are
       rejected — the §4.2 equivalence only holds for packed forms).
    3. Cross-layout flips require rank ≥ 2.

    Args:
        tensor: Source TensorType (packed canonical or bare).
        layout: Target ``TensorLayout`` (must not be ``NZ``).
        span: Optional source span (auto-captured when omitted).

    Returns:
        ``Call`` expression carrying a TensorType with the canonical
        ``(shape, stride, layout)`` for the target view.
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {"layout": layout}
    return _ir_core.create_op_call("tensor.as_layout", [tensor], kwargs, actual_span)


def set_validshape(
    tensor: Expr,
    valid_rows: int | Expr,
    valid_cols: int | Expr,
    span: Span | None = None,
) -> Call:
    """Update valid-shape metadata of a tensor without data movement.

    .. note::
        Internal API — this op is intended for compiler-generated code only
        and should not be exposed to end users in future releases.

    Args:
        tensor: Input tensor expression (must be 2D TensorType)
        valid_rows: Number of valid rows (int or Scalar INDEX expression)
        valid_cols: Number of valid columns (int or Scalar INDEX expression)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for tensor.set_validshape
    """
    actual_span = _get_span_or_capture(span)
    vr_expr = (
        valid_rows if isinstance(valid_rows, Expr) else ConstInt(valid_rows, DataType.INDEX, actual_span)
    )
    vc_expr = (
        valid_cols if isinstance(valid_cols, Expr) else ConstInt(valid_cols, DataType.INDEX, actual_span)
    )
    return _ir_core.create_op_call("tensor.set_validshape", [tensor, vr_expr, vc_expr], {}, actual_span)


def scatter_update(
    input: Expr,
    *args: Expr | int,
    dim: int | Expr | None = None,
    index: Expr | None = None,
    src: Expr | None = None,
    span: Span | None = None,
) -> Call:
    """Update input tensor rows at positions specified by 2D index with values from src.

    Supports two variants based on input/src rank:
    - 2D: input [rows, d], src [b*s, d], index [b, s]
    - 4D: input [blockNum, blockSize, 1, d], src [b, s, 1, d], index [b, s]

    For each (i, j): input[index[i*s+j]] row = src[i*s+j] row (linear layout).

    Accepts both call forms:
    - scatter_update(input, dim, index, src)
    - scatter_update(input, index, src, dim=-2)

    Args:
        input: Destination tensor (2D or 4D TensorType)
        dim: Dimension to scatter along (default: -2, currently the only supported value)
        index: 2D index tensor [b, s] of integer dtype
        src: Source tensor (2D [b*s, d] or 4D [b, s, 1, d])
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning the updated input tensor
    """
    if len(args) == 3 and dim is None and index is None and src is None:
        dim, index, src = args
    elif len(args) == 2 and dim is not None and index is None and src is None:
        index, src = args
    elif len(args) == 1 and dim is None and index is not None and src is not None:
        # (input, dim, index=..., src=...) — dim passed positionally
        dim = args[0]
    elif len(args) != 0:
        raise TypeError(
            "scatter_update expects (input, dim, index, src), "
            "(input, index, src, dim=...), or (input, dim, index=..., src=...)"
        )

    if dim is None or index is None or src is None:
        raise TypeError("scatter_update requires input, dim, index, and src")

    actual_span = _get_span_or_capture(span)
    if isinstance(dim, ConstInt):
        dim_val = int(dim.value)
    elif isinstance(dim, int):
        dim_val = dim
    else:
        raise TypeError(f"dim must be int or ConstInt, got {type(dim)}")

    if not isinstance(index, Expr):
        raise TypeError(f"index must be Expr, got {type(index)}")
    if not isinstance(src, Expr):
        raise TypeError(f"src must be Expr, got {type(src)}")
    op_args: list[Expr] = [input, index, src]
    kwargs: dict[str, Any] = {"dim": dim_val}
    return _ir_core.create_op_call("tensor.scatter_update", op_args, kwargs, actual_span)


# ============================================================================
# Sort Operations
# ============================================================================


def sort32(src: Expr, idx: Expr, span: Span | None = None) -> Call:
    """Sort fixed 32-element blocks with explicit index tensor (tensor-level).

    Tensor-level counterpart of ``tile.sort32``. Sorts 32-element blocks in src
    and permutes idx accordingly. Output tensor stores sorted value-index pairs
    with the last dimension doubled.

    Args:
        src: Input value tensor (TensorType, FP16 or FP32)
        idx: Input index tensor (TensorType) with sequential offsets
        span: Optional source span for debugging

    Returns:
        Call expression returning sorted tensor with doubled last dimension
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tensor.sort32", [src, idx], {}, actual_span)


def mrgsort(
    src0: Expr,
    src1: Expr | None = None,
    src2: Expr | None = None,
    src3: Expr | None = None,
    *,
    exhausted: bool = False,
    block_len: int | Expr | None = None,
    span: Span | None = None,
) -> Call:
    """Merge sort — format1 (single-list) or format2 (2-4 way merge), tensor-level.

    Tensor-level counterpart of ``tile.mrgsort``. Format1 sorts a tensor
    containing multiple pre-sorted runs of length ``block_len``. Format2 merges
    2, 3, or 4 pre-sorted input tensors into one sorted output.

    The scratch ``tmp`` and ``executed`` tiles required by the tile-level op are
    synthesized automatically during ConvertTensorToTileOps as local Vec tiles —
    users do not pass them at the tensor level.

    Args:
        src0: For format1: input tensor with pre-sorted runs (FP16 or FP32).
              For format2: first sorted input tensor.
        src1: (format2) Second sorted input tensor.
        src2: (format2, optional) Third sorted input tensor (3-way or 4-way).
        src3: (format2, optional) Fourth sorted input tensor (4-way only).
        exhausted: (format2) If True, marks inputs as exhausted (default: False).
        block_len: (format1, keyword-only) Run length, must be multiple of 64.
        span: Optional source span for debugging.

    Returns:
        Call expression returning merged sorted tensor.
    """
    actual_span = _get_span_or_capture(span)
    if block_len is not None:
        if exhausted or any(arg is not None for arg in (src1, src2, src3)):
            raise ValueError(
                "mrgsort() format1 (block_len=...) and format2 (src1, ...) "
                "are mutually exclusive; do not pass format2 arguments or exhausted=True with block_len"
            )
        if isinstance(block_len, _ir_core.ConstInt):
            block_len_expr = _ir_core.ConstInt(block_len.value, DataType.INT32, actual_span)
        elif isinstance(block_len, Expr):
            block_len_expr = block_len
        else:
            block_len_expr = _ir_core.ConstInt(block_len, DataType.INT32, actual_span)
        return _ir_core.create_op_call("tensor.mrgsort_format1", [src0, block_len_expr], {}, actual_span)
    # format2: 2-4 way merge
    if src1 is None:
        raise ValueError(
            "mrgsort() requires either block_len=<int> for format1, or at least (src0, src1) for format2"
        )
    if src2 is None and src3 is not None:
        raise ValueError("mrgsort() format2 requires src2 when src3 is provided")
    kwargs: dict[str, Any] = {"exhausted": exhausted}
    if src2 is None:
        args = [src0, src1]
    elif src3 is None:
        args = [src0, src1, src2]
    else:
        args = [src0, src1, src2, src3]
    return _ir_core.create_op_call("tensor.mrgsort_format2", args, kwargs, actual_span)


def mrgsort_format1(src0: Expr, block_len: int | Expr, span: Span | None = None) -> Call:
    """Single-list merge sort (format1). Used by the parser for roundtrip fidelity.

    Prefer ``mrgsort(src, block_len=...)`` in user code.
    """
    return mrgsort(src0, block_len=block_len, span=span)


def mrgsort_format2(*args: Expr, exhausted: bool = False, span: Span | None = None) -> Call:
    """2-4 way merge sort (format2). Used by the parser for roundtrip fidelity.

    Positional args: ``(src0, src1[, src2[, src3]])``.

    Prefer ``mrgsort(src0, src1[, src2[, src3]])`` in user code.
    """
    if len(args) < 2 or len(args) > 4:
        raise ValueError(
            f"mrgsort_format2() requires 2-4 positional arguments "
            f"(src0, src1[, src2[, src3]]), got {len(args)}"
        )
    src0 = args[0]
    src1 = args[1]
    src2 = args[2] if len(args) > 2 else None
    src3 = args[3] if len(args) > 3 else None
    return mrgsort(src0, src1, src2, src3, exhausted=exhausted, span=span)


# ============================================================================
# Gather Operation
# ============================================================================


def gather(
    input: Expr,
    dim: int | None = None,
    index: Expr | None = None,
    *,
    mask_pattern: int | None = None,
    output_dtype: int | DataType | None = None,
    span: Span | None = None,
) -> Call:
    """Gather elements of ``input`` (tensor-level) — index form or mask-pattern form.

    Index form (``dim`` + ``index``): output[b, k] = input[b, index[b, k]].
        MVP limitation: only rank-2 inputs with ``dim == -1`` (or ``rank - 1``).
        ``index`` must be an INT32 tensor whose shape matches ``input`` on every
        axis except ``dim``; output shape == ``index.shape``, dtype == ``input.dtype``.

    Mask form (``mask_pattern=<int>``): selects columns of each row by a fixed
        hardware mask. Last-dim shrinks by 2 (P0101/P1010) or 4 (P0001..P1000),
        or stays the same for P1111. Optional ``output_dtype`` keyword reinterprets
        result bits to a same-bit-width dtype (e.g. FP32 → UINT32).

    Args:
        input: Source tensor (TensorType, FP16/FP32/INT16/INT32).
        dim: (index form) Axis along which to gather. Only ``-1`` / ``rank - 1`` accepted in MVP.
        index: (index form) Index tensor (TensorType, INT32) with the same rank as ``input``.
        mask_pattern: (mask form, keyword-only) Mask pattern selector in [1, 7].
            1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111
        output_dtype: (mask form, keyword-only) Optional output dtype with the same
            bit width as ``input.dtype``.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression with the appropriate result type for the chosen form.
    """
    actual_span = _get_span_or_capture(span)
    if mask_pattern is not None:
        if dim is not None or index is not None:
            raise ValueError(
                "gather() mask form (mask_pattern=...) and index form (dim, index) "
                "are mutually exclusive; do not pass dim or index with mask_pattern"
            )
        kwargs: dict[str, Any] = {"mask_pattern": mask_pattern}
        if output_dtype is not None:
            kwargs["output_dtype"] = output_dtype
        return _ir_core.create_op_call("tensor.gather_mask", [input], kwargs, actual_span)
    if dim is None or index is None:
        raise ValueError(
            "gather() requires either (dim, index) for index form, or mask_pattern=<int> for mask form"
        )
    if output_dtype is not None:
        raise ValueError("gather() output_dtype is only valid for the mask form; use mask_pattern=<int>")
    kwargs = {"dim": dim}
    return _ir_core.create_op_call("tensor.gather", [input, index], kwargs, actual_span)


def gather_mask(
    input: Expr,
    mask_pattern: int,
    output_dtype: int | DataType | None = None,
    span: Span | None = None,
) -> Call:
    """Gather elements of ``input`` (tensor-level) by mask pattern. Used by the parser
    for roundtrip fidelity.

    Prefer ``gather(input, mask_pattern=...)`` in user code.
    """
    return gather(input, mask_pattern=mask_pattern, output_dtype=output_dtype, span=span)
