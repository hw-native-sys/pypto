# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Canonical scalar SIMT operations on borrowed UB tiles or GM tensors."""

from collections.abc import Sequence

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, SimtAxis, Span

from ..utils import _get_span_or_capture, _to_make_tuple


def thread_idx(axis: SimtAxis, span: Span | None = None) -> Call:
    """Get the INT32 thread coordinate along the required typed axis."""
    return _ir_core.create_op_call("simt.thread_idx", [], {"axis": axis}, _get_span_or_capture(span))


def block_idx(axis: SimtAxis, span: Span | None = None) -> Call:
    """Get the INT32 block coordinate along the required typed axis."""
    return _ir_core.create_op_call("simt.block_idx", [], {"axis": axis}, _get_span_or_capture(span))


def block_dim(axis: SimtAxis, span: Span | None = None) -> Call:
    """Get the INT32 block extent along the required typed axis."""
    return _ir_core.create_op_call("simt.block_dim", [], {"axis": axis}, _get_span_or_capture(span))


def grid_dim(axis: SimtAxis, span: Span | None = None) -> Call:
    """Get the INT32 grid extent along the required typed axis."""
    return _ir_core.create_op_call("simt.grid_dim", [], {"axis": axis}, _get_span_or_capture(span))


def lane_id(span: Span | None = None) -> Call:
    """Get the INT32 lane identifier within the current warp."""
    return _ir_core.create_op_call("simt.lane_id", [], {}, _get_span_or_capture(span))


def load(
    storage: Expr,
    indices: Sequence[int | Expr],
    *,
    l1cache: str | None = None,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Load one element directly from UB Tile or GM Tensor storage.

    Indices are logical element coordinates. GM accesses honor Tensor view
    strides and offsets without implicit UB staging. Cache policies apply only
    to GM and use the PTOAS spellings.
    """
    span = _get_span_or_capture(span)
    kwargs = {k: v for k, v in (("l1cache", l1cache), ("l2cache", l2cache)) if v is not None}
    return _ir_core.create_op_call("simt.load", [storage, _to_make_tuple(indices, span)], kwargs, span)


def store(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    l1cache: str | None = None,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Store one scalar element directly to UB Tile or GM Tensor storage.

    Returns VoidType: place this call in EvalStmt. The enclosing SimtLaunchStmt
    defines updated storage handles; untouched elements retain their old value.
    Cache policies apply only to GM and use the PTOAS spellings.
    """
    span = _get_span_or_capture(span)
    kwargs = {k: v for k, v in (("l1cache", l1cache), ("l2cache", l2cache)) if v is not None}
    return _ir_core.create_op_call(
        "simt.store", [storage, _to_make_tuple(indices, span), value], kwargs, span
    )
