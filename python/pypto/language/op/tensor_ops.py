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

import warnings
from collections.abc import Sequence
from typing import overload

__all__ = [
    "create_tensor",
    "create",
    "read",
    "write",
    "dim",
    "slice",
    "fillpad",
    "full",
    "matmul",
    "matmul_acc",
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
    "expand_clone",
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
    "scatter_update",
    "set_validshape",
    "sort32",
    "mrgsort",
    "gather",
    "alloc",
]

from pypto.ir.op import tensor_ops as _ir_ops
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, MemorySpace, PadValue, TensorLayout

from ..typing import IntLike, Scalar, Tensor
from .tile_ops import MemRefType


def _unwrap_rhs(rhs: int | float | Expr | Tensor | Scalar) -> int | float | Expr:
    """Unwrap rhs operands into the IR-layer representation."""
    if isinstance(rhs, (Tensor, Scalar)):
        return rhs.unwrap()
    return rhs


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


def create(
    shape: Sequence[IntLike],
    dtype: DataType,
    layout: TensorLayout = TensorLayout.ND,
    manual_dep: bool = False,
) -> Tensor:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes (int or Expr)
        dtype: Data type of tensor elements
        layout: Tensor layout (default: ND)
        manual_dep: When True, the tensor opts out of automatic dependency
            tracking by codegen (used internally for orchestrator-injected
            workspace buffers like the GM pipe buffer). Most user code should
            leave this as False.

    Returns:
        Tensor wrapping the create operation
    """
    call_expr = _ir_ops.create(_normalize_intlike(shape), dtype, layout, manual_dep=manual_dep)
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
    pad_value: PadValue | int | float | None = None,
) -> Tensor:
    """Create a slice of a tensor with new shape and optional valid shape.

    Args:
        tensor: Input tensor
        shape: New shape dimensions
        offset: Offset dimensions for the slice
        valid_shape: Valid shape dimensions. When omitted, the full shape is valid.
        pad_value: Optional padding mode for out-of-valid-shape elements.
            ``None`` or ``PadValue.null`` means no padding (the default).
            Accepts ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or
            the literal sugars ``0``, ``math.inf``, ``-math.inf`` (same
            spelling as :func:`tensor.fillpad`). Only meaningful when
            ``valid_shape`` is smaller than ``shape``.

    Returns:
        Tensor wrapping the slice operation
    """
    if pad_value is not None and pad_value is not PadValue.null and valid_shape is None:
        warnings.warn(
            f"tensor.slice received pad_value={pad_value!r} but no valid_shape. "
            f"pad_value has no effect unless valid_shape is smaller than shape. "
            f"If you intend to narrow the valid region later via "
            f"tensor.set_validshape, you can ignore this warning; otherwise "
            f"pass valid_shape=... to tensor.slice.",
            stacklevel=2,
        )

    tensor_expr = tensor.unwrap()
    normalized_valid_shape = None if valid_shape is None else _normalize_intlike(valid_shape)
    call_expr = _ir_ops.slice(
        tensor_expr,
        _normalize_intlike(shape),
        _normalize_intlike(offset),
        normalized_valid_shape,
        pad_value=pad_value,
    )
    return Tensor(expr=call_expr)


def fillpad(tensor: Tensor, pad_value: PadValue | int | float = PadValue.zero) -> Tensor:
    """Fill invalid tensor view elements with the specified padding value.

    Args:
        tensor: Input tensor
        pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
            the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
            ``PadValue.zero``. Other values raise — the hardware only supports
            the three padding modes.

    Returns:
        Tensor wrapping the fillpad operation
    """
    call_expr = _ir_ops.fillpad(tensor.unwrap(), pad_value=pad_value)
    return Tensor(expr=call_expr)


def set_validshape(tensor: Tensor, valid_rows: IntLike, valid_cols: IntLike) -> Tensor:
    """Update valid-shape metadata of a tensor without data movement.

    .. note::
        Internal API — this op is intended for compiler-generated code only
        and should not be exposed to end users in future releases.

    Args:
        tensor: Input tensor (must be 2D)
        valid_rows: Number of valid rows (int or Scalar[INDEX])
        valid_cols: Number of valid columns (int or Scalar[INDEX])

    Returns:
        Tensor with updated valid_shape metadata
    """
    tensor_expr = tensor.unwrap()
    vr = valid_rows.unwrap() if isinstance(valid_rows, Scalar) else valid_rows
    vc = valid_cols.unwrap() if isinstance(valid_cols, Scalar) else valid_cols
    call_expr = _ir_ops.set_validshape(tensor_expr, vr, vc)
    return Tensor(expr=call_expr)


def full(shape: Sequence[IntLike], dtype: DataType, value: int | float) -> Tensor:
    """Create a tensor of specified shape filled with a constant value.

    Args:
        shape: Shape of the tensor
        dtype: Data type of tensor elements
        value: Filling scalar value (int or float)

    Returns:
        Tensor wrapping the full operation
    """
    call_expr = _ir_ops.full(_normalize_intlike(shape), dtype, value)
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


def matmul_acc(
    acc: Tensor,
    lhs: Tensor,
    rhs: Tensor,
    a_trans: bool = False,
    b_trans: bool = False,
) -> Tensor:
    """Matrix multiplication with accumulation: acc += lhs @ rhs.

    Args:
        acc: Accumulator tensor
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        a_trans: Whether to transpose lhs
        b_trans: Whether to transpose rhs

    Returns:
        Tensor wrapping the matmul_acc operation
    """
    call_expr = _ir_ops.matmul_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap(), a_trans, b_trans)
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
    call_expr = _ir_ops.mul(lhs_expr, _unwrap_rhs(rhs))
    return Tensor(expr=call_expr)


def muls(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    """Element-wise multiplication of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the muls operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.muls(lhs_expr, _unwrap_rhs(rhs))
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
    call_expr = _ir_ops.add(lhs_expr, _unwrap_rhs(rhs))
    return Tensor(expr=call_expr)


def adds(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    """Element-wise addition of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the adds operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.adds(lhs_expr, _unwrap_rhs(rhs))
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
    call_expr = _ir_ops.sub(lhs_expr, _unwrap_rhs(rhs))
    return Tensor(expr=call_expr)


def subs(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    """Element-wise subtraction of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr with ScalarType)

    Returns:
        Tensor wrapping the subs operation
    """
    lhs_expr = lhs.unwrap()
    call_expr = _ir_ops.subs(lhs_expr, _unwrap_rhs(rhs))
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
    call_expr = _ir_ops.div(lhs_expr, _unwrap_rhs(rhs))
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
    call_expr = _ir_ops.divs(lhs_expr, _unwrap_rhs(rhs))
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


def row_expand(target: Tensor, row_vec: Tensor) -> Tensor:
    """Row-wise expansion: expand row_vec [M, 1] to target shape [M, N].

    Args:
        target: Target tensor defining output shape (TensorType [M, N])
        row_vec: Row vector to expand (TensorType [M, 1])

    Returns:
        Tensor wrapping the row_expand operation
    """
    target_expr = target.unwrap()
    row_vec_expr = row_vec.unwrap()
    call_expr = _ir_ops.row_expand(target_expr, row_vec_expr)
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


def expand_clone(src: Tensor, target: Tensor) -> Tensor:
    """Clone and expand input to target shape.

    Args:
        src: Source tensor
        target: Target tensor defining output shape

    Returns:
        Tensor wrapping the expand_clone operation
    """
    src_expr = src.unwrap()
    target_expr = target.unwrap()
    call_expr = _ir_ops.expand_clone(src_expr, target_expr)
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


def rsqrt(input: Tensor, high_precision: bool = False) -> Tensor:
    """Element-wise reciprocal square root operation.

    Args:
        input: Input tensor
        high_precision: If True, lower to the higher-precision PTO path. The
            compiler allocates a scratch buffer during tensor-to-tile conversion.

    Returns:
        Tensor wrapping the rsqrt operation
    """
    input_expr = input.unwrap()
    call_expr = _ir_ops.rsqrt(input_expr, high_precision=high_precision)
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


def scatter_update(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
) -> Tensor:
    """Update input tensor rows at positions specified by 2D index with values from src.

    Supports two variants based on input/src rank:
    - 2D: input [rows, d], src [b*s, d], index [b, s]
    - 4D: input [blockNum, blockSize, 1, d], src [b, s, 1, d], index [b, s]

    Args:
        input: Destination tensor (2D or 4D)
        dim: Dimension to scatter along (currently only -2 is supported)
        index: 2D index tensor [b, s] of integer dtype
        src: Source tensor (2D [b*s, d] or 4D [b, s, 1, d])

    Returns:
        Tensor wrapping the scatter_update operation
    """
    call_expr = _ir_ops.scatter_update(input.unwrap(), dim, index.unwrap(), src.unwrap())
    return Tensor(expr=call_expr)


def sort32(src: Tensor, idx: Tensor) -> Tensor:
    """Sort fixed 32-element blocks with explicit index tensor (tensor-level).

    Tensor-level counterpart of ``pl.tile.sort32``. Sorts 32-element blocks in
    src, permuting idx alongside. Returns sorted value-index pairs tensor with
    doubled last dimension.

    For FP16 src: initialize idx with [0, 1, 2, ..., 31] per block.
    For FP32 src: initialize idx with [0, 2, 4, ..., 62] per block.

    Args:
        src: Input value tensor (FP16 or FP32)
        idx: Input index tensor with sequential offsets

    Returns:
        Tensor wrapping the sort32 operation (last dim doubled)
    """
    call_expr = _ir_ops.sort32(src.unwrap(), idx.unwrap())
    return Tensor(expr=call_expr)


@overload
def mrgsort(src0: Tensor, *, block_len: int | Scalar) -> Tensor: ...


@overload
def mrgsort(src0: Tensor, src1: Tensor, *, exhausted: bool = ...) -> Tensor: ...


@overload
def mrgsort(src0: Tensor, src1: Tensor, src2: Tensor, *, exhausted: bool = ...) -> Tensor: ...


@overload
def mrgsort(src0: Tensor, src1: Tensor, src2: Tensor, src3: Tensor, *, exhausted: bool = ...) -> Tensor: ...


def mrgsort(
    src0: Tensor,
    src1: Tensor | None = None,
    src2: Tensor | None = None,
    src3: Tensor | None = None,
    *,
    exhausted: bool = False,
    block_len: int | Scalar | None = None,
) -> Tensor:
    """Merge sort — format1 (single-list) or format2 (2-4 way merge), tensor-level.

    Tensor-level counterpart of ``pl.tile.mrgsort``. The scratch ``tmp`` and
    ``executed`` tiles required by the tile-level op are synthesized
    automatically during conversion as local Vec tiles — users do not pass them.

    Format1 usage (keyword block_len):
        out = mrgsort(src, block_len=64)

    Format2 usage:
        out = mrgsort(src0, src1)                # 2-way
        out = mrgsort(src0, src1, src2)          # 3-way
        out = mrgsort(src0, src1, src2, src3)    # 4-way
        out = mrgsort(src0, src1, exhausted=True)

    Args:
        src0: For format1: input tensor with pre-sorted runs (FP16 or FP32).
              For format2: first sorted input tensor.
        src1: (format2) Second sorted input tensor.
        src2: (format2, optional) Third sorted input tensor (3-way or 4-way).
        src3: (format2, optional) Fourth sorted input tensor (4-way only).
        exhausted: (format2) If True, marks inputs as exhausted (default: False).
        block_len: (format1, keyword-only) Run length, must be multiple of 64.

    Returns:
        Tensor with merged sorted elements
    """
    if block_len is not None:
        if exhausted or any(arg is not None for arg in (src1, src2, src3)):
            raise ValueError(
                "mrgsort() format1 (block_len=...) and format2 (src1, ...) "
                "are mutually exclusive; do not pass format2 arguments or exhausted=True with block_len"
            )
        block_len_expr = block_len.unwrap() if isinstance(block_len, Scalar) else block_len
        call_expr = _ir_ops.mrgsort(src0.unwrap(), block_len=block_len_expr)
        return Tensor(expr=call_expr)
    # format2: 2-4 way merge
    if src1 is None:
        raise ValueError(
            "mrgsort() requires either block_len=<int> for format1, or at least (src0, src1) for format2"
        )
    call_expr = _ir_ops.mrgsort(
        src0.unwrap(),
        src1.unwrap(),
        src2.unwrap() if src2 is not None else None,
        src3.unwrap() if src3 is not None else None,
        exhausted=exhausted,
    )
    return Tensor(expr=call_expr)


@overload
def gather(input: Tensor, dim: int, index: Tensor) -> Tensor: ...


@overload
def gather(input: Tensor, *, mask_pattern: int, output_dtype: int | DataType | None = None) -> Tensor: ...


def gather(
    input: Tensor,
    dim: int | None = None,
    index: Tensor | None = None,
    *,
    mask_pattern: int | None = None,
    output_dtype: int | DataType | None = None,
) -> Tensor:
    """Gather elements of ``input`` (tensor-level) — index form or mask-pattern form.

    Index form (``dim`` + ``index``):
        ``output[b, k] = input[b, index[b, k]]``
        MVP: only rank-2 inputs with ``dim == -1`` (or ``rank - 1``).
        ``index`` must be an INT32 tensor whose shape matches ``input`` on every
        axis except ``dim``.

    Mask form (``mask_pattern=<int>``):
        Selects columns of each row by a fixed hardware mask pattern (lowered
        directly to ``tile.gather_mask``). Last-dim shrinks by 2 (P0101/P1010)
        or 4 (P0001..P1000), or stays the same for P1111.

    Args:
        input: Source tensor (FP16/FP32/INT16/INT32).
        dim: (index form) Axis to gather along; only ``-1`` / ``rank - 1`` accepted in MVP.
        index: (index form) Index tensor (INT32) with same rank as input.
        mask_pattern: (mask form, keyword-only) Mask pattern selector (1-7).
            1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111.
        output_dtype: (mask form, keyword-only) Optional output dtype with the same
            bit width as ``input.dtype`` (e.g. FP32 → UINT32 for sort32 index bits).

    Returns:
        Tensor of shape ``index.shape`` (index form) or shape with shrunk last dim
        (mask form), and dtype ``input.dtype`` (or ``output_dtype`` if provided).

    Examples:
        out = gather(input, dim=-1, index=idx)
        out = gather(input, mask_pattern=1)
        out = gather(input, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.UINT32)
    """
    if mask_pattern is not None:
        if dim is not None or index is not None:
            raise ValueError(
                "gather() mask form (mask_pattern=...) and index form (dim, index) "
                "are mutually exclusive; do not pass dim or index with mask_pattern"
            )
        call_expr = _ir_ops.gather(input.unwrap(), mask_pattern=mask_pattern, output_dtype=output_dtype)
        return Tensor(expr=call_expr)
    if output_dtype is not None:
        raise ValueError("output_dtype is only valid for the mask form of gather(); use mask_pattern=<int>")
    if dim is None or index is None:
        raise ValueError(
            "gather() requires either (dim, index) for index form, or mask_pattern=<int> for mask form"
        )
    call_expr = _ir_ops.gather(input.unwrap(), dim, index.unwrap())
    return Tensor(expr=call_expr)


def alloc(
    memory_space: MemorySpace,
    size: int,
) -> MemRefType:
    """Stub for the internal ``tensor.alloc`` IR operation.

    This function is never called in user-written DSL code. It is emitted
    by the C++ python-printer after the InitMemRef pass and must be
    importable so that the printed source is valid Python.
    """
    return MemRefType()
