# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type stubs for PyPTO IR helper functions."""

from typing import List, Union

from pypto import DataType
from pypto.pypto_core import ir

# DataType constants
FP16: DataType
FP32: DataType
INT32: DataType
INT64: DataType

def tensor_create(shape: List[int], dtype: DataType) -> ir.Call:
    """Create a new tensor with specified shape and dtype.

    Args:
        shape: List of dimension sizes
        dtype: Data type of tensor elements

    Returns:
        Call expression creating a new tensor
    """

def tensor_view(
    tensor: ir.Expr, shape: List[Union[int, ir.Expr]], offset: List[Union[int, ir.Expr]]
) -> ir.Call:
    """Create a view/slice of a tensor with new shape and offset.

    Args:
        tensor: Input tensor expression
        shape: New shape dimensions
        offset: Offset dimensions for the view

    Returns:
        Call expression creating a tensor view
    """

def tensor_matmul(
    lhs: ir.Expr,
    rhs: ir.Expr,
    outDtype: Union[int, DataType],
    aTrans: bool = False,
    bTrans: bool = False,
    cMatrixNz: bool = False,
) -> ir.Call:
    """Matrix multiplication with optional transpose.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        outDtype: Output data type
        aTrans: Whether to transpose lhs
        bTrans: Whether to transpose rhs
        cMatrixNz: C matrix non-zero flag

    Returns:
        Call expression for matrix multiplication
    """

def tensor_mul(lhs: ir.Expr, rhs: Union[ir.Expr, int, float]) -> ir.Call:
    """Element-wise multiplication of two tensors.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor or scalar

    Returns:
        Call expression for element-wise multiplication
    """

def tensor_add(lhs: ir.Expr, rhs: ir.Expr) -> ir.Call:
    """Element-wise addition of two tensors.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor

    Returns:
        Call expression for element-wise addition
    """

def tensor_sub(lhs: ir.Expr, rhs: ir.Expr) -> ir.Call:
    """Element-wise subtraction of two tensors.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor

    Returns:
        Call expression for element-wise subtraction
    """

def tensor_div(lhs: ir.Expr, rhs: ir.Expr) -> ir.Call:
    """Element-wise division of two tensors.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor

    Returns:
        Call expression for element-wise division
    """

def tensor_maximum(lhs: ir.Expr, rhs: ir.Expr) -> ir.Call:
    """Element-wise maximum of two tensors.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor

    Returns:
        Call expression for element-wise maximum
    """

def tensor_row_max(
    input: ir.Expr, kind: str = "rowmax", axis: int = -1, keepDim: Union[int, bool] = 1
) -> ir.Call:
    """Row-wise maximum reduction along specified axis.

    Args:
        input: Input tensor
        kind: Reduction kind string
        axis: Reduction axis (default: -1, last axis)
        keepDim: Keep reduced dimension as 1

    Returns:
        Call expression for row-wise maximum reduction
    """

def tensor_row_sum(
    input: ir.Expr, kind: str = "rowsum", axis: int = -1, keepDim: Union[int, bool] = 1
) -> ir.Call:
    """Row-wise sum reduction along specified axis.

    Args:
        input: Input tensor
        kind: Reduction kind string
        axis: Reduction axis (default: -1, last axis)
        keepDim: Keep reduced dimension as 1

    Returns:
        Call expression for row-wise sum reduction
    """

def tensor_exp(input: ir.Expr) -> ir.Call:
    """Element-wise exponential operation.

    Args:
        input: Input tensor

    Returns:
        Call expression for element-wise exponential
    """

def tensor_cast(input: ir.Expr, targetType: Union[int, DataType], mode: str = "round") -> ir.Call:
    """Type casting operation.

    Args:
        input: Input tensor
        targetType: Target data type
        mode: Rounding mode (e.g., 'round', 'floor', 'ceil')

    Returns:
        Call expression for type casting
    """

def tensor_assemble(target: ir.Expr, source: ir.Expr, offset: List[Union[int, ir.Expr]]) -> ir.Call:
    """Write/update tensor values at specified offset.

    Args:
        target: Target tensor to update
        source: Source tensor to write
        offset: Offset dimensions for where to write

    Returns:
        Call expression for tensor assembly
    """
