# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Array operations for PyPTO Language DSL.

Type-safe wrappers around the IR-level ``pypto.ir.op.array_ops`` that accept
and return ``Array`` / ``Scalar`` wrappers.

Writes are SSA-functional: ``update_element(arr, i, v)`` returns a new
``Array`` representing the updated array, mirroring ``tensor.assemble``.
"""

from pypto.ir.op import array as _ir_ops
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr

from ..typing import IntLike, Scalar
from ..typing.array import Array

__all__ = ["create", "get_element", "update_element"]


def _unwrap_intlike(value: IntLike) -> int | Expr:
    if isinstance(value, Scalar):
        return value.unwrap()
    return value


def create(extent: int, dtype: DataType) -> Array:
    """Allocate an on-core array (C-stack local).

    Args:
        extent: Number of elements (positive Python int — compile-time constant).
        dtype: Element data type (must be integer or BOOL).

    Returns:
        ``Array`` wrapping the ``array.create`` Call expression.
    """
    call_expr = _ir_ops.create(extent, dtype)
    return Array(expr=call_expr)


def get_element(array: Array, index: IntLike) -> Scalar:
    """Read an element from an array at the given index.

    Args:
        array: Source ``Array``.
        index: Element index (Python int, raw Expr, or Scalar DSL value).

    Returns:
        ``Scalar`` wrapping the ``array.get_element`` Call expression.
    """
    call_expr = _ir_ops.get_element(array.unwrap(), _unwrap_intlike(index))
    return Scalar(expr=call_expr)


def update_element(array: Array, index: IntLike, value: IntLike) -> Array:
    """Functional update: return a new Array with the given element replaced.

    Args:
        array: Source ``Array``.
        index: Element index (Python int, raw Expr, or Scalar DSL value).
        value: Replacement value (Python int, raw Expr, or Scalar DSL value).

    Returns:
        ``Array`` wrapping the ``array.update_element`` Call expression.
    """
    value_expr = _unwrap_intlike(value)
    if isinstance(value_expr, int) and not isinstance(value_expr, Expr):
        # The IR-level op requires an Expr value; wrap a bare int as ConstInt
        # using the array's element dtype.
        from pypto.pypto_core.ir import ArrayType, ConstInt, Span  # noqa: PLC0415

        array_type = array.unwrap().type
        wrap_dtype = array_type.dtype if isinstance(array_type, ArrayType) else DataType.INT64
        value_expr = ConstInt(value_expr, wrap_dtype, Span.unknown())
    call_expr = _ir_ops.update_element(array.unwrap(), _unwrap_intlike(index), value_expr)
    return Array(expr=call_expr)
