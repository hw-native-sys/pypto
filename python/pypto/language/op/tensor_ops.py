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

from collections.abc import Sequence

__all__ = [
    "create_tensor",
    "create",
    "read",
    "write",
    "dim",
    "slice",
    "fillpad",
    "matmul",
    "mul",
    "muls",
    "add",
    "adds",
    "sub",
    "subs",
    "div",
    "divs",
    "maximum",
    "row_max",
    "row_sum",
    "row_min",
    "row_expand",
    "row_expand_mul",
    "row_expand_div",
    "row_expand_add",
    "row_expand_sub",
    "col_expand_mul",
    "col_expand",
    "col_expand_div",
    "col_expand_sub",
    "expands",
    "exp",
    "neg",
    "recip",
    "sqrt",
    "rsqrt",
    "cast",
    "assemble",
    "concat",
    "reshape",
    "transpose",
]

from pypto.ir.op import tensor_ops as _ir_ops
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, PadValue, TensorLayout

from ..typing import IntLike, Scalar, Tensor


def _unwrap_rhs(rhs: int | float | Tensor | Scalar) -> int | float | Expr:
    """Unwrap rhs operand: extract Expr from Tensor/Scalar wrappers, pass through primitives."""
    if isinstance(rhs, (Tensor, Scalar)):
        return rhs.unwrap()
    return rhs


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


def create(shape: Sequence[IntLike], dtype: DataType, layout: TensorLayout = TensorLayout.ND) -> Tensor:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes (int or Expr)
        dtype: Data type of tensor elements
        layout: Tensor layout (default: ND)

    Returns:
        Tensor wrapping the create operation
    """
    call_expr = _ir_ops.create(_normalize_intlike(shape), dtype, layout)
    return Tensor(expr=call_expr)


create_tensor = create


def read(tensor: Tensor, indices: IntLike | Sequence[IntLike]) -> Scalar:
    """Read a scalar value from a tensor at given indices.

    Args:
        tensor: Input tensor
        indices: A single index expression (for 1-D flat access) or a list of
            index expressions (one per tensor dimension)

    Returns:
        Scalar wrapping the read operation
    """
    tensor_expr = tensor.unwrap()
    # Allow a bare IntLike as a flat 1-D index for backwards compatibility
    indices_seq: Sequence[IntLike] = [indices] if not isinstance(indices, Sequence) else indices
    call_expr = _ir_ops.read(tensor_expr, _normalize_intlike(indices_seq))
    return Scalar(expr=call_expr)


def write(tensor: Tensor, indices: IntLike | Sequence[IntLike], value: Scalar) -> None:
    """Write a scalar value into a tensor at given indices.

    Args:
        tensor: Destination tensor
        indices: A single index expression (for 1-D flat access) or a list of
            index expressions (one per tensor dimension)
        value: Scalar value to write
    """
    # Allow a bare IntLike as a flat 1-D index for backwards compatibility
    indices_seq: Sequence[IntLike] = [indices] if not isinstance(indices, Sequence) else indices
    call_expr = _ir_ops.write(tensor.unwrap(), _normalize_intlike(indices_seq), value.unwrap())
    _ = call_expr  # result is the tensor itself; discarded here


def dim(tensor: Tensor, axis: int) -> Scalar:
    """Extract a shape dimension from a tensor as a scalar value.

    Args:
        tensor: Input tensor
        axis: Dimension index (supports negative indexing)

    Returns:
        Scalar wrapping the dim operation (INT64)
    """
    tensor_expr = tensor.unwrap()
    call_expr = _ir_ops.dim(tensor_expr, axis)
    return Scalar(expr=call_expr)


def slice(
    tensor: Tensor,
    shape: Sequence[IntLike],
    offset: Sequence[IntLike],
    valid_shape: Sequence[IntLike] | None = None,
) -> Tensor:
    """Create a slice of a tensor with new shape and optional valid shape.

    Args:
        tensor: Input tensor
        shape: New shape dimensions
        offset: Offset dimensions for the slice
        valid_shape: Valid shape dimensions. When omitted, the full shape is valid.

    Returns:
        Tensor wrapping the slice operation
    """
    tensor_expr = tensor.unwrap()
    normalized_valid_shape = None if valid_shape is None else _normalize_intlike(valid_shape)
    call_expr = _ir_ops.slice(
        tensor_expr,
        _normalize_intlike(shape),
        _normalize_intlike(offset),
        normalized_valid_shape,
    )
    return Tensor(expr=call_expr)


def fillpad(tensor: Tensor, pad_value: PadValue = PadValue.zero) -> Tensor:
    """Fill invalid tensor view elements with the specified padding value.

    Args:
        tensor: Input tensor
        pad_value: Padding mode (PadValue.zero, PadValue.max, or PadValue.min)

    Returns:
        Tensor wrapping the fillpad operation
    """
    call_expr = _ir_ops.fillpad(tensor.unwrap(), pad_value=pad_value)
    return Tensor(expr=call_expr)


def matmul(
    lhs: Tensor,
    rhs: Tensor,
    out_dtype: int | DataType | None = None,
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


def mul(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    """Element-wise multiplication of tensor and tensor or scalar.

    Automatically selects between tensor.mul (tensor x tensor) and
    tensor.muls (tensor x scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

    Returns:
        Tensor wrapping the mul operation
    """
    lhs_expr = lhs.unwrap()
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_ops.mul(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def muls(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
    """Element-wise multiplication of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the muls operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.muls(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def add(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    """Element-wise addition of tensor and tensor or scalar.

    Automatically selects between tensor.add (tensor + tensor) and
    tensor.adds (tensor + scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

    Returns:
        Tensor wrapping the add operation
    """
    lhs_expr = lhs.unwrap()
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_ops.add(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def adds(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
    """Element-wise addition of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the adds operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.adds(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def sub(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    """Element-wise subtraction of tensor and tensor or scalar.

    Automatically selects between tensor.sub (tensor - tensor) and
    tensor.subs (tensor - scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

    Returns:
        Tensor wrapping the sub operation
    """
    lhs_expr = lhs.unwrap()
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_ops.sub(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def subs(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
    """Element-wise subtraction of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the subs operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.subs(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def div(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    """Element-wise division of tensor and tensor or scalar.

    Automatically selects between tensor.div (tensor / tensor) and
    tensor.divs (tensor / scalar) based on the rhs type.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

    Returns:
        Tensor wrapping the div operation
    """
    lhs_expr = lhs.unwrap()
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_ops.div(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def divs(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    """Element-wise division of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr/Scalar)

    Returns:
        Tensor wrapping the divs operation
    """
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.divs(lhs_expr, rhs_expr)
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


def row_max(input: Tensor) -> Tensor:
    """Row-wise max reduction (reduces along last axis, keeps dim).

    Args:
        input: Input tensor

    Returns:
        Tensor wrapping the row_max operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.row_max(input_expr)
    return Tensor(expr=call_expr)


def row_sum(input: Tensor) -> Tensor:
    """Row-wise sum reduction (reduces along last axis, keeps dim).

    Args:
        input: Input tensor

    Returns:
        Tensor wrapping the row_sum operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.row_sum(input_expr)
    return Tensor(expr=call_expr)


def row_min(input: Tensor) -> Tensor:
    """Row-wise min reduction (reduces along last axis, keeps dim).

    Args:
        input: Input tensor

    Returns:
        Tensor wrapping the row_min operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.row_min(input_expr)
    return Tensor(expr=call_expr)


def row_expand(input: Tensor) -> Tensor:
    """Row-wise broadcast: dst[i, j] = src[i, 0].

    Args:
        input: Input tensor (TensorType [M, N])

    Returns:
        Tensor wrapping the row_expand operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.row_expand(input_expr)
    return Tensor(expr=call_expr)


def row_expand_mul(tensor: Tensor, row_vec: Tensor) -> Tensor:
    """Row-wise broadcast multiplication: tensor[i,:] * row_vec[i,0].

    Args:
        tensor: Input tensor (TensorType [M, N])
        row_vec: Row vector (TensorType [M, 1])

    Returns:
        Tensor wrapping the row_expand_mul operation
    """
    tensor_expr = tensor.unwrap()
    row_vec_expr = row_vec.unwrap()
    call_expr = _ir_ops.row_expand_mul(tensor_expr, row_vec_expr)
    return Tensor(expr=call_expr)


def row_expand_div(tensor: Tensor, row_vec: Tensor) -> Tensor:
    """Row-wise broadcast division: tensor[i,:] / row_vec[i,0].

    Args:
        tensor: Input tensor (TensorType [M, N])
        row_vec: Row vector (TensorType [M, 1])

    Returns:
        Tensor wrapping the row_expand_div operation
    """
    tensor_expr = tensor.unwrap()
    row_vec_expr = row_vec.unwrap()
    call_expr = _ir_ops.row_expand_div(tensor_expr, row_vec_expr)
    return Tensor(expr=call_expr)


def row_expand_add(tensor: Tensor, row_vec: Tensor) -> Tensor:
    """Row-wise broadcast addition: tensor[i,:] + row_vec[i,0].

    Args:
        tensor: Input tensor (TensorType [M, N])
        row_vec: Row vector (TensorType [M, 1])

    Returns:
        Tensor wrapping the row_expand_add operation
    """
    tensor_expr = tensor.unwrap()
    row_vec_expr = row_vec.unwrap()
    call_expr = _ir_ops.row_expand_add(tensor_expr, row_vec_expr)
    return Tensor(expr=call_expr)


def row_expand_sub(tensor: Tensor, row_vec: Tensor) -> Tensor:
    """Row-wise broadcast subtraction: tensor[i,:] - row_vec[i,0].

    Args:
        tensor: Input tensor (TensorType [M, N])
        row_vec: Row vector (TensorType [M, 1])

    Returns:
        Tensor wrapping the row_expand_sub operation
    """
    tensor_expr = tensor.unwrap()
    row_vec_expr = row_vec.unwrap()
    call_expr = _ir_ops.row_expand_sub(tensor_expr, row_vec_expr)
    return Tensor(expr=call_expr)


def col_expand_mul(tensor: Tensor, col_vec: Tensor) -> Tensor:
    """Column-wise broadcast multiplication: tensor[:,j] * col_vec[0,j].

    Args:
        tensor: Input tensor (TensorType [M, N])
        col_vec: Column vector (TensorType [1, N])

    Returns:
        Tensor wrapping the col_expand_mul operation
    """
    tensor_expr = tensor.unwrap()
    col_vec_expr = col_vec.unwrap()
    call_expr = _ir_ops.col_expand_mul(tensor_expr, col_vec_expr)
    return Tensor(expr=call_expr)


def col_expand(tensor: Tensor, col_vec: Tensor) -> Tensor:
    """Column-wise expansion: expand col_vec [1, N] to target shape [M, N].

    Args:
        tensor: Target tensor defining output shape (TensorType [M, N])
        col_vec: Column vector to expand (TensorType [1, N])

    Returns:
        Tensor wrapping the col_expand operation
    """
    tensor_expr = tensor.unwrap()
    col_vec_expr = col_vec.unwrap()
    call_expr = _ir_ops.col_expand(tensor_expr, col_vec_expr)
    return Tensor(expr=call_expr)


def col_expand_sub(tensor: Tensor, col_vec: Tensor) -> Tensor:
    """Column-wise broadcast subtraction: tensor[:,j] - col_vec[0,j].

    Args:
        tensor: Input tensor (TensorType [M, N])
        col_vec: Column vector (TensorType [1, N])

    Returns:
        Tensor wrapping the col_expand_sub operation
    """
    tensor_expr = tensor.unwrap()
    col_vec_expr = col_vec.unwrap()
    call_expr = _ir_ops.col_expand_sub(tensor_expr, col_vec_expr)
    return Tensor(expr=call_expr)


def col_expand_div(tensor: Tensor, col_vec: Tensor) -> Tensor:
    """Column-wise broadcast division: tensor[:,j] / col_vec[0,j].

    Args:
        tensor: Input tensor (TensorType [M, N])
        col_vec: Column vector (TensorType [1, N])

    Returns:
        Tensor wrapping the col_expand_div operation
    """
    tensor_expr = tensor.unwrap()
    col_vec_expr = col_vec.unwrap()
    call_expr = _ir_ops.col_expand_div(tensor_expr, col_vec_expr)
    return Tensor(expr=call_expr)


def expands(target: Tensor, scalar: int | float | Scalar) -> Tensor:
    """Expand scalar to target tensor shape.

    Args:
        target: Target tensor defining output shape
        scalar: Scalar value to expand

    Returns:
        Tensor wrapping the expands operation
    """
    target_expr = target.unwrap()
    scalar_expr = scalar.unwrap() if isinstance(scalar, Scalar) else scalar
    call_expr = _ir_ops.expands(target_expr, scalar_expr)
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


def neg(input: Tensor) -> Tensor:
    """Element-wise negation operation.

    Args:
        input: Input tensor

    Returns:
        Tensor wrapping the neg operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.neg(input_expr)
    return Tensor(expr=call_expr)


def recip(input: Tensor) -> Tensor:
    """Element-wise reciprocal (1/x) operation.

    Args:
        input: Input tensor

    Returns:
        Tensor wrapping the recip operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.recip(input_expr)
    return Tensor(expr=call_expr)


def sqrt(input: Tensor) -> Tensor:
    """Element-wise square root operation.

    Args:
        input: Input tensor

    Returns:
        Tensor wrapping the sqrt operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.sqrt(input_expr)
    return Tensor(expr=call_expr)


def rsqrt(input: Tensor) -> Tensor:
    """Element-wise reciprocal square root operation.

    Args:
        input: Input tensor

    Returns:
        Tensor wrapping the rsqrt operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.rsqrt(input_expr)
    return Tensor(expr=call_expr)


def cast(
    input: Tensor,
    target_type: int | DataType,
    mode: str | int = "round",
) -> Tensor:
    """Type casting operation.

    Args:
        input: Input tensor
        target_type: Target data type
        mode: Rounding mode — string name ("none", "rint", "round", "floor",
              "ceil", "trunc", "odd") or int (0–6)

    Returns:
        Tensor wrapping the cast operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.cast(input_expr, target_type, mode)
    return Tensor(expr=call_expr)


def assemble(target: Tensor, source: Tensor, offset: Sequence[IntLike]) -> Tensor:
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
    call_expr = _ir_ops.assemble(target_expr, source_expr, _normalize_intlike(offset))
    return Tensor(expr=call_expr)


def concat(src0: Tensor, src1: Tensor) -> Tensor:
    """Concatenate two tensors along the column dimension.

    Args:
        src0: First source tensor
        src1: Second source tensor

    Returns:
        Tensor with concatenated columns
    """
    src0_expr = src0.unwrap()
    src1_expr = src1.unwrap()
    call_expr = _ir_ops.concat(src0_expr, src1_expr)
    return Tensor(expr=call_expr)


def reshape(tensor: Tensor, shape: Sequence[IntLike]) -> Tensor:
    """Reshape tensor to new shape.

    Args:
        tensor: Input tensor
        shape: New shape dimensions

    Returns:
        Tensor wrapping the reshape operation
    """
    tensor_expr = tensor.unwrap()
    call_expr = _ir_ops.reshape(tensor_expr, _normalize_intlike(shape))
    return Tensor(expr=call_expr)


def transpose(tensor: Tensor, axis1: int, axis2: int) -> Tensor:
    """Transpose tensor by swapping two axes.

    Args:
        tensor: Input tensor
        axis1: First axis to swap (supports negative indexing)
        axis2: Second axis to swap (supports negative indexing)

    Returns:
        Tensor wrapping the transpose operation
    """
    tensor_expr = tensor.unwrap()
    call_expr = _ir_ops.transpose(tensor_expr, axis1, axis2)
    return Tensor(expr=call_expr)
