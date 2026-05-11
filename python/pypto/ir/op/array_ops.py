# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR-level wrappers for array operations.

Mirrors the pattern in ``pypto/ir/op/tensor_ops.py`` — each function builds a
``Call`` expression for the corresponding registered op. The DSL layer in
``pypto/language/op/array_ops.py`` wraps these into ``Array`` / ``Scalar``
values for the user-facing API.

Writes are SSA-functional: ``update_element(arr, i, v)`` returns a new SSA
value of ``ArrayType`` — semantically equivalent to ``tensor.assemble``.
"""

from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, Span

from ..utils import _get_span_or_capture, _normalize_expr


def create(extent: int | Expr, dtype: DataType, span: Span | None = None) -> Call:
    """Allocate an on-core array (C-stack local).

    Args:
        extent: Number of elements. Accepts a Python int (auto-wrapped into a
            ``ConstInt(INDEX)``) or a ``ConstInt`` expression already produced
            by the parser. Must be a compile-time constant.
        dtype: Element data type (must be integer or BOOL).
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression that returns an ``ArrayType``.
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(extent, Expr):
        extent_expr: Expr = extent
    else:
        extent_expr = ConstInt(int(extent), DataType.INDEX, actual_span)
    kwargs: dict[str, Any] = {"dtype": dtype}
    return _ir_core.create_op_call("array.create", [extent_expr], kwargs, actual_span)


def get_element(array: Expr, index: int | Expr, span: Span | None = None) -> Call:
    """Read an element from an array at the given index.

    Args:
        array: Source array expression.
        index: Element index (Python int or scalar Expr, integer dtype).
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression that returns a ``ScalarType``.
    """
    actual_span = _get_span_or_capture(span)
    index_expr = _normalize_expr(index, actual_span)
    return _ir_core.create_op_call("array.get_element", [array, index_expr], {}, actual_span)


def update_element(array: Expr, index: int | Expr, value: Expr, span: Span | None = None) -> Call:
    """Functional update: return a new ArrayType value with the given element replaced.

    Args:
        array: Source array expression.
        index: Element index (Python int or scalar Expr, integer dtype).
        value: Replacement value (Expr — dtype must match the array's dtype).
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression representing the updated array (same shape and dtype).
    """
    actual_span = _get_span_or_capture(span)
    index_expr = _normalize_expr(index, actual_span)
    return _ir_core.create_op_call("array.update_element", [array, index_expr, value], {}, actual_span)
