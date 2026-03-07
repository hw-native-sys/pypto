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
from typing import Literal

__all__ = [
    "create_tensor",
    "read",
    "dim",
    "view",
    "matmul",
    "mul",
    "mul_scalar",
    "add",
    "add_scalar",
    "sub",
    "sub_scalar",
    "div",
    "div_scalar",
    "maximum",
    "row_max",
    "row_sum",
    "exp",
    "cast",
    "assemble",
    "reshape",
    "transpose",
]

from pypto.ir.op import tensor_ops as _ir_ops
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr

from ..typing import IntLike, Scalar, Tensor


def _unwrap_rhs(rhs: int | float | Tensor | Scalar) -> int | float | Expr:
    """Unwrap rhs operand: extract Expr from Tensor/Scalar wrappers, pass through primitives."""
    if isinstance(rhs, (Tensor, Scalar)):
        return rhs.unwrap()
    return rhs


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


def create_tensor(shape: Sequence[IntLike], dtype: DataType) -> Tensor:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes (int or Expr)
        dtype: Data type of tensor elements

    Returns:
        Tensor wrapping the create operation
    """
    call_expr = _ir_ops.create(_normalize_intlike(shape), dtype)
    return Tensor(expr=call_expr)


def read(tensor: Tensor, indices: Sequence[IntLike]) -> Scalar:
    """Read a scalar value from a tensor at given indices.

    Args:
        tensor: Input tensor
        indices: List of index expressions (one per tensor dimension)

    Returns:
        Scalar wrapping the read operation
    """
    tensor_expr = tensor.unwrap()
    call_expr = _ir_ops.read(tensor_expr, _normalize_intlike(indices))
    return Scalar(expr=call_expr)


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


def view(tensor: Tensor, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tensor:
    """Create a view/slice of a tensor with new shape and offset.

    Args:
        tensor: Input tensor
        shape: New shape dimensions
        offset: Offset dimensions for the view

    Returns:
        Tensor wrapping the view operation
    """
    tensor_expr = tensor.unwrap()
    call_expr = _ir_ops.view(tensor_expr, _normalize_intlike(shape), _normalize_intlike(offset))
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
    tensor.mul_scalar (tensor x scalar) based on the rhs type.

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


def mul_scalar(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
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


def add(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
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
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_ops.add(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def add_scalar(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
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


def sub(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    """Element-wise subtraction of tensor and tensor or scalar.

    Automatically selects between tensor.sub (tensor - tensor) and
    tensor.sub_scalar (tensor - scalar) based on the rhs type.

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


def sub_scalar(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
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


def div(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    """Element-wise division of tensor and tensor or scalar.

    Automatically selects between tensor.div (tensor / tensor) and
    tensor.div_scalar (tensor / scalar) based on the rhs type.

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


def div_scalar(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    """Element-wise division of tensor and scalar.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side scalar (int/float/Expr/Scalar)

    Returns:
        Tensor wrapping the div_scalar operation
    """
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.div_scalar(lhs_expr, rhs_expr)
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
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
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


def _simple_tensor_unary(op_name: str, input: Tensor) -> Tensor:
    input_expr = input.unwrap()
    call_expr = getattr(_ir_ops, op_name)(input_expr)
    return Tensor(expr=call_expr)


def _simple_tensor_binary(op_name: str, lhs: Tensor, rhs: Tensor) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap()
    call_expr = getattr(_ir_ops, op_name)(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def neg(input: Tensor) -> Tensor:
    return _simple_tensor_unary("neg", input)


def recip(input: Tensor) -> Tensor:
    return _simple_tensor_unary("recip", input)


def sqrt(input: Tensor) -> Tensor:
    return _simple_tensor_unary("sqrt", input)


def rsqrt(input: Tensor) -> Tensor:
    return _simple_tensor_unary("rsqrt", input)


def log(input: Tensor) -> Tensor:
    return _simple_tensor_unary("log", input)


def abs(input: Tensor) -> Tensor:
    return _simple_tensor_unary("abs", input)


def relu(input: Tensor) -> Tensor:
    return _simple_tensor_unary("relu", input)


def minimum(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("minimum", lhs, rhs)


def row_expand(input: Tensor) -> Tensor:
    return _simple_tensor_unary("row_expand", input)


def row_expand_sub(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("row_expand_sub", lhs, rhs)


def row_expand_div(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("row_expand_div", lhs, rhs)


def row_expand_mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("row_expand_mul", lhs, rhs)


def row_expand_add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("row_expand_add", lhs, rhs)


def col_expand(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("col_expand", lhs, rhs)


def col_expand_mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("col_expand_mul", lhs, rhs)


def col_expand_div(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("col_expand_div", lhs, rhs)


def col_expand_sub(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("col_expand_sub", lhs, rhs)


def _simple_tensor_scalar_binary(op_name: str, lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = getattr(_ir_ops, op_name)(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def _simple_tensor_ternary(
    op_name: str, a: Tensor, b: Tensor | int | float | Expr | Scalar, c: Tensor
) -> Tensor:
    a_expr = a.unwrap()
    b_expr = b.unwrap() if isinstance(b, (Tensor, Scalar)) else b
    c_expr = c.unwrap()
    call_expr = getattr(_ir_ops, op_name)(a_expr, b_expr, c_expr)
    return Tensor(expr=call_expr)


def rem(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("rem", lhs, rhs)


def adds(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("adds", lhs, rhs)


def subs(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("subs", lhs, rhs)


def muls(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("muls", lhs, rhs)


def divs(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("divs", lhs, rhs)


def rems(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("rems", lhs, rhs)


def and_(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("and_", lhs, rhs)


def ands(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("ands", lhs, rhs)


def or_(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("or_", lhs, rhs)


def ors(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("ors", lhs, rhs)


def shl(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("shl", lhs, rhs)


def shls(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("shls", lhs, rhs)


def shr(lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_binary("shr", lhs, rhs)


def shrs(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("shrs", lhs, rhs)


def maxs(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("maxs", lhs, rhs)


def mins(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("mins", lhs, rhs)


def not_(input: Tensor) -> Tensor:
    return _simple_tensor_unary("not_", input)


def xor(lhs: Tensor, rhs: Tensor, tmp: Tensor) -> Tensor:
    return _simple_tensor_ternary("xor", lhs, rhs, tmp)


def xors(lhs: Tensor, rhs: int | float | Expr | Scalar, tmp: Tensor) -> Tensor:
    return _simple_tensor_ternary("xors", lhs, rhs, tmp)


def prelu(lhs: Tensor, rhs: Tensor, tmp: Tensor) -> Tensor:
    return _simple_tensor_ternary("prelu", lhs, rhs, tmp)


def addc(lhs: Tensor, rhs: Tensor, rhs2: Tensor) -> Tensor:
    return _simple_tensor_ternary("addc", lhs, rhs, rhs2)


def subc(lhs: Tensor, rhs: Tensor, rhs2: Tensor) -> Tensor:
    return _simple_tensor_ternary("subc", lhs, rhs, rhs2)


def addsc(lhs: Tensor, rhs: int | float | Expr | Scalar, rhs2: Tensor) -> Tensor:
    return _simple_tensor_ternary("addsc", lhs, rhs, rhs2)


def subsc(lhs: Tensor, rhs: int | float | Expr | Scalar, rhs2: Tensor) -> Tensor:
    return _simple_tensor_ternary("subsc", lhs, rhs, rhs2)


def lrelu(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("lrelu", lhs, rhs)


def sel(mask: Tensor, lhs: Tensor, rhs: Tensor) -> Tensor:
    return _simple_tensor_ternary("sel", mask, lhs, rhs)


def sels(lhs: Tensor, rhs: Tensor, select_mode: int | float | Expr | Scalar) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap()
    mode_expr = select_mode.unwrap() if isinstance(select_mode, Scalar) else select_mode
    call_expr = _ir_ops.sels(lhs_expr, rhs_expr, mode_expr)
    return Tensor(expr=call_expr)


def cmp(lhs: Tensor, rhs: Tensor, cmp_type: int = 0) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap()
    call_expr = _ir_ops.cmp(lhs_expr, rhs_expr, cmp_type)
    return Tensor(expr=call_expr)


def cmps(lhs: Tensor, rhs: int | float | Expr | Scalar, cmp_type: int = 0) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_ops.cmps(lhs_expr, rhs_expr, cmp_type)
    return Tensor(expr=call_expr)


def sum(input: Tensor, axis: int = -1, keep_dim: bool = True) -> Tensor:
    return Tensor(expr=_ir_ops.sum(input.unwrap(), axis, keep_dim))


def max(input: Tensor, axis: int = -1, keep_dim: bool = True) -> Tensor:
    return Tensor(expr=_ir_ops.max(input.unwrap(), axis, keep_dim))


def min(input: Tensor, axis: int = -1, keep_dim: bool = True) -> Tensor:
    return Tensor(expr=_ir_ops.min(input.unwrap(), axis, keep_dim))


def row_min(input: Tensor, axis: int = -1, keep_dim: bool = True) -> Tensor:
    return Tensor(expr=_ir_ops.row_min(input.unwrap(), axis, keep_dim))


def full(shape: Sequence[IntLike], dtype: DataType, value: int | float | Expr | Scalar) -> Tensor:
    value_expr = value.unwrap() if isinstance(value, Scalar) else value
    return Tensor(expr=_ir_ops.full(_normalize_intlike(shape), dtype, value_expr))


def expands(target: Tensor, scalar: int | float | Expr | Scalar) -> Tensor:
    return _simple_tensor_scalar_binary("expands", target, scalar)


def fillpad(input: Tensor) -> Tensor:
    return _simple_tensor_unary("fillpad", input)


def matmul_acc(
    acc: Tensor,
    lhs: Tensor,
    rhs: Tensor,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> Tensor:
    return Tensor(expr=_ir_ops.matmul_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap(), out_dtype, a_trans, b_trans, c_matrix_nz))


def matmul_bias(
    lhs: Tensor,
    rhs: Tensor,
    bias: Tensor,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> Tensor:
    return Tensor(expr=_ir_ops.matmul_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap(), out_dtype, a_trans, b_trans, c_matrix_nz))


def gemv(
    lhs: Tensor,
    rhs: Tensor,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> Tensor:
    return Tensor(expr=_ir_ops.gemv(lhs.unwrap(), rhs.unwrap(), out_dtype, a_trans, b_trans, c_matrix_nz))


def gemv_acc(
    acc: Tensor,
    lhs: Tensor,
    rhs: Tensor,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> Tensor:
    return Tensor(expr=_ir_ops.gemv_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap(), out_dtype, a_trans, b_trans, c_matrix_nz))


def gemv_bias(
    lhs: Tensor,
    rhs: Tensor,
    bias: Tensor,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> Tensor:
    return Tensor(expr=_ir_ops.gemv_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap(), out_dtype, a_trans, b_trans, c_matrix_nz))


__all__.extend(
    [
        "neg",
        "recip",
        "sqrt",
        "rsqrt",
        "log",
        "abs",
        "relu",
        "minimum",
        "row_expand",
        "row_expand_sub",
        "row_expand_div",
        "row_expand_mul",
        "row_expand_add",
        "col_expand",
        "col_expand_mul",
        "col_expand_div",
        "col_expand_sub",
        "rem",
        "adds",
        "subs",
        "muls",
        "divs",
        "rems",
        "and_",
        "ands",
        "or_",
        "ors",
        "shl",
        "shls",
        "shr",
        "shrs",
        "maxs",
        "mins",
        "not_",
        "xor",
        "xors",
        "prelu",
        "addc",
        "subc",
        "addsc",
        "subsc",
        "lrelu",
        "sel",
        "sels",
        "cmp",
        "cmps",
        "sum",
        "max",
        "min",
        "row_min",
        "full",
        "expands",
        "fillpad",
        "matmul_acc",
        "matmul_bias",
        "gemv",
        "gemv_acc",
        "gemv_bias",
    ]
)
