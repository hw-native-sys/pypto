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

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, SimtAxis, Span

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


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


def lanemask_eq(span: Span | None = None) -> Call:
    """Return the physical equal-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return _ir_core.create_op_call("simt.lanemask_eq", [], {}, _get_span_or_capture(span))


def lanemask_le(span: Span | None = None) -> Call:
    """Return the physical lower-or-equal-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return _ir_core.create_op_call("simt.lanemask_le", [], {}, _get_span_or_capture(span))


def lanemask_lt(span: Span | None = None) -> Call:
    """Return the physical lower-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return _ir_core.create_op_call("simt.lanemask_lt", [], {}, _get_span_or_capture(span))


def lanemask_ge(span: Span | None = None) -> Call:
    """Return the physical higher-or-equal-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return _ir_core.create_op_call("simt.lanemask_ge", [], {}, _get_span_or_capture(span))


def lanemask_gt(span: Span | None = None) -> Call:
    """Return the physical higher-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return _ir_core.create_op_call("simt.lanemask_gt", [], {}, _get_span_or_capture(span))


def bitcast(value: Expr, dtype: DataType, span: Span | None = None) -> Call:
    """Reinterpret an unpacked numeric scalar as dtype without changing any bits.

    Source and destination must have equal bit width: FP16/BF16/FP32 or
    8/16/32/64-bit integers. NaN payloads and signed zero bits
    are preserved; this is not a numerical conversion or a storage view.
    """
    return _ir_core.create_op_call("simt.bitcast", [value], {"dtype": dtype}, _get_span_or_capture(span))


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


def _predicate_call(name: str, predicate: bool | Expr, span: Span) -> Call:
    value = _ir_core.ConstBool(predicate, span) if isinstance(predicate, bool) else predicate
    return _ir_core.create_op_call(name, [value], {}, span)


def vote_all(predicate: bool | Expr, span: Span | None = None) -> Call:
    """Return whether every active lane has a true BOOL predicate.

    Uses the current active lanes; this operation does not provide a memory fence.
    """
    return _predicate_call("simt.vote_all", predicate, _get_span_or_capture(span))


def vote_any(predicate: bool | Expr, span: Span | None = None) -> Call:
    """Return whether any active lane has a true BOOL predicate.

    Uses the current active lanes; this operation does not provide a memory fence.
    """
    return _predicate_call("simt.vote_any", predicate, _get_span_or_capture(span))


def vote_uni(predicate: bool | Expr, span: Span | None = None) -> Call:
    """Return whether all active lanes agree on their BOOL predicate.

    Uses the current active lanes; this operation does not provide a memory fence.
    """
    return _predicate_call("simt.vote_uni", predicate, _get_span_or_capture(span))


def vote_ballot(predicate: bool | Expr, span: Span | None = None) -> Call:
    """Return an INT32 bit mask of true predicates in active lanes; bit i denotes lane i.

    Uses the current active lanes; this operation does not provide a memory fence.
    """
    return _predicate_call("simt.vote_ballot", predicate, _get_span_or_capture(span))


def _lane_control(value: int | Expr, span: Span) -> Expr:
    """Normalize Python integer controls without treating bool as a lane index."""
    if isinstance(value, bool):
        raise TypeError("SIMT lane control requires INT32, got bool")
    return _normalize_expr(value, span, int_dtype=DataType.INT32)


def shuffle_idx(
    value: Expr, index: int | Expr, *, width: int = 32, bound: int | None = None, span: Span | None = None
) -> Call:
    """Read an FP16/FP32 or 32/64-bit integer value from a subgroup-relative lane index.

    Width must be 16 or 32. Bound must be in [0, width); omitted means the full source window.
    Out-of-window sources return the caller value; inactive sources are undefined.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.shuffle_idx",
        [value, _lane_control(index, span)],
        {"width": width, **({"bound": bound} if bound is not None else {})},
        span,
    )


def shuffle_up(
    value: Expr, offset: int | Expr, *, width: int = 32, bound: int | None = None, span: Span | None = None
) -> Call:
    """Read an FP16/FP32 or 32/64-bit integer value from a lower lane in the subgroup.

    Width must be 16 or 32. Bound must be in [0, width); omitted means the full source window.
    Out-of-window sources return the caller value; inactive sources are undefined.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.shuffle_up",
        [value, _lane_control(offset, span)],
        {"width": width, **({"bound": bound} if bound is not None else {})},
        span,
    )


def shuffle_down(
    value: Expr, offset: int | Expr, *, width: int = 32, bound: int | None = None, span: Span | None = None
) -> Call:
    """Read an FP16/FP32 or 32/64-bit integer value from a higher lane in the subgroup.

    Width must be 16 or 32. Bound must be in [0, width); omitted means the full source window.
    Out-of-window sources return the caller value; inactive sources are undefined.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.shuffle_down",
        [value, _lane_control(offset, span)],
        {"width": width, **({"bound": bound} if bound is not None else {})},
        span,
    )


def shuffle_bfly(
    value: Expr, mask: int | Expr, *, width: int = 32, bound: int | None = None, span: Span | None = None
) -> Call:
    """Read an FP16/FP32 or 32/64-bit integer value from the XOR-selected lane in the subgroup.

    Width must be 16 or 32. Bound must be in [0, width); omitted means the full source window.
    Out-of-window sources return the caller value; inactive sources are undefined.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.shuffle_bfly",
        [value, _lane_control(mask, span)],
        {"width": width, **({"bound": bound} if bound is not None else {})},
        span,
    )


def redux_add(value: Expr, span: Span | None = None) -> Call:
    """Return the sum of active-lane FP16/FP32 or 32-bit integer values to each active lane.

    The result retains the input dtype. INT32 sums wrap at 32 bits.
    FP32 reductions follow the target arithmetic order and special-value rules.
    This operation does not provide a memory fence.
    """
    return _ir_core.create_op_call("simt.redux_add", [value], {}, _get_span_or_capture(span))


def redux_min(value: Expr, span: Span | None = None, *, unsigned: bool = False) -> Call:
    """Return the minimum of active-lane FP16/FP32 or 32-bit integer values to each active lane.

    The result retains the input dtype. unsigned=True compares INT32 bit carriers as unsigned.
    UINT32 inputs always use unsigned comparison.
    FP32 reductions follow the target arithmetic order and special-value rules.
    This operation does not provide a memory fence.
    """
    return _ir_core.create_op_call(
        "simt.redux_min", [value], {"unsigned": unsigned}, _get_span_or_capture(span)
    )


def redux_max(value: Expr, span: Span | None = None, *, unsigned: bool = False) -> Call:
    """Return the maximum of active-lane FP16/FP32 or 32-bit integer values to each active lane.

    The result retains the input dtype. unsigned=True compares INT32 bit carriers as unsigned.
    UINT32 inputs always use unsigned comparison.
    FP32 reductions follow the target arithmetic order and special-value rules.
    This operation does not provide a memory fence.
    """
    return _ir_core.create_op_call(
        "simt.redux_max", [value], {"unsigned": unsigned}, _get_span_or_capture(span)
    )


def atomic_sub(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically subtract from an initialized element and return its old value.

    Supports 32/64-bit integers and FP16/BF16/FP32; 64-bit atomics require GM.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
    return _ir_core.create_op_call(
        "simt.atomic_sub",
        [storage, _to_make_tuple(indices, span), value],
        kwargs,
        span,
    )


def atomic_and(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically apply bitwise AND to one initialized 32/64-bit integer element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
    return _ir_core.create_op_call(
        "simt.atomic_and",
        [storage, _to_make_tuple(indices, span), value],
        kwargs,
        span,
    )


def atomic_or(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically apply bitwise OR to one initialized 32/64-bit integer element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
    return _ir_core.create_op_call(
        "simt.atomic_or",
        [storage, _to_make_tuple(indices, span), value],
        kwargs,
        span,
    )


def atomic_xor(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically apply bitwise XOR to one initialized 32/64-bit integer element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
    return _ir_core.create_op_call(
        "simt.atomic_xor",
        [storage, _to_make_tuple(indices, span), value],
        kwargs,
        span,
    )


def atomic_add(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically add one initialized 32/64-bit integer or FP16/BF16/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
    return _ir_core.create_op_call(
        "simt.atomic_add",
        [storage, _to_make_tuple(indices, span), value],
        kwargs,
        span,
    )


def atomic_min(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    unsigned: bool = False,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically min one initialized 32/64-bit integer or FP16/BF16/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    unsigned=True compares INT32 bit carriers as unsigned; FP32 special values follow the target rules.
    """
    span = _get_span_or_capture(span)
    kwargs: dict[str, str | bool] = {"unsigned": unsigned}
    if l2cache is not None:
        kwargs["l2cache"] = l2cache
    return _ir_core.create_op_call(
        "simt.atomic_min",
        [storage, _to_make_tuple(indices, span), value],
        kwargs,
        span,
    )


def atomic_max(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    unsigned: bool = False,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically max one initialized 32/64-bit integer or FP16/BF16/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    unsigned=True compares INT32 bit carriers as unsigned; FP32 special values follow the target rules.
    """
    span = _get_span_or_capture(span)
    kwargs: dict[str, str | bool] = {"unsigned": unsigned}
    if l2cache is not None:
        kwargs["l2cache"] = l2cache
    return _ir_core.create_op_call(
        "simt.atomic_max",
        [storage, _to_make_tuple(indices, span), value],
        kwargs,
        span,
    )


def atomic_exch(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically exch one initialized 32/64-bit integer or FP16/BF16/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
    return _ir_core.create_op_call(
        "simt.atomic_exch",
        [storage, _to_make_tuple(indices, span), value],
        kwargs,
        span,
    )


def atomic_cas(
    storage: Expr,
    indices: Sequence[int | Expr],
    compare: Expr,
    value: Expr,
    *,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically cas one initialized 32/64-bit integer or FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    Comparison is bitwise, including FP32 NaN payloads and signed zeros.
    Replace only on a match; the result is the observed old value.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
    return _ir_core.create_op_call(
        "simt.atomic_cas",
        [storage, _to_make_tuple(indices, span), compare, value],
        kwargs,
        span,
    )


def syncthreads(span: Span | None = None) -> Call:
    """Synchronize all threads in the current SIMT entry and order their memory effects.

    Must be reached unconditionally by all threads. Nested control flow is
    conservatively rejected. This is not an outer MTE/Vector pipeline barrier."""
    return _ir_core.create_op_call("simt.syncthreads", [], {}, _get_span_or_capture(span))


def threadfence(span: Span | None = None) -> Call:
    """Issue the target workitem memory fence for the calling thread.

    Does not wait for other threads or provide cross-AIV rendezvous.
    Ordering and visibility follow the native target fence contract."""
    return _ir_core.create_op_call("simt.threadfence", [], {}, _get_span_or_capture(span))


def threadfence_block(span: Span | None = None) -> Call:
    """Issue the target block-scoped memory fence for the calling thread.

    Does not wait for other threads; use syncthreads for a rendezvous.
    This does not replace outer MTE/Vector pipeline synchronization."""
    return _ir_core.create_op_call("simt.threadfence_block", [], {}, _get_span_or_capture(span))


def abs(value: Expr, span: Span | None = None) -> Call:
    """Absolute value; clears the FP32 sign bit.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.abs", [value], {}, _get_span_or_capture(span))


def min(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """FP32 minimum with minNum semantics; one NaN yields the numeric operand.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.min", [lhs, rhs], {}, _get_span_or_capture(span))


def max(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """FP32 maximum with maxNum semantics; one NaN yields the numeric operand.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.max", [lhs, rhs], {}, _get_span_or_capture(span))


def exp(value: Expr, span: Span | None = None) -> Call:
    """Compute the FP32 natural exponential using target math.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.exp", [value], {}, _get_span_or_capture(span))


def log(value: Expr, span: Span | None = None) -> Call:
    """Compute the FP32 natural logarithm using target math.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.log", [value], {}, _get_span_or_capture(span))


def sqrt(value: Expr, span: Span | None = None) -> Call:
    """Compute the FP32 square root using target math.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.sqrt", [value], {}, _get_span_or_capture(span))


def sin(value: Expr, span: Span | None = None) -> Call:
    """Compute the FP32 sine in radians using target math.

    PTOAS uses FP32 software range reduction; large angles lose accuracy.
    Numerical acceptance covers [-4, 4]; no full-range accuracy is promised.
    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.sin", [value], {}, _get_span_or_capture(span))


def cos(value: Expr, span: Span | None = None) -> Call:
    """Compute the FP32 cosine in radians using target math.

    PTOAS uses FP32 software range reduction; large angles lose accuracy.
    Numerical acceptance covers [-4, 4]; no full-range accuracy is promised.
    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.cos", [value], {}, _get_span_or_capture(span))


def ceil(value: Expr, span: Span | None = None) -> Call:
    """Round FP32 toward positive infinity, retaining FP32.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.ceil", [value], {}, _get_span_or_capture(span))


def floor(value: Expr, span: Span | None = None) -> Call:
    """Round FP32 toward negative infinity, retaining FP32.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.floor", [value], {}, _get_span_or_capture(span))


def rint(value: Expr, span: Span | None = None) -> Call:
    """Round FP32 to nearest integer, ties to even, retaining FP32.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.rint", [value], {}, _get_span_or_capture(span))


def round(value: Expr, span: Span | None = None) -> Call:
    """Round FP32 to nearest integer, ties away from zero, retaining FP32.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.round", [value], {}, _get_span_or_capture(span))


def pow(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Compute FP32 power using target math.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.pow", [lhs, rhs], {}, _get_span_or_capture(span))


def fma(lhs: Expr, rhs: Expr, acc: Expr, span: Span | None = None) -> Call:
    """Compute FP32 lhs * rhs + acc with one final rounding.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return _ir_core.create_op_call("simt.fma", [lhs, rhs, acc], {}, _get_span_or_capture(span))


def prmt(lhs: Expr, rhs: Expr, selector: Expr, span: Span | None = None) -> Call:
    """Permute bytes from two INT32 bit carriers with an INT32 selector.

    Operands must have INT32 scalar dtype; no implicit promotion is performed.
    The low three bits of each low selector nibble pick a byte from rhs
    (0-3) or lhs (4-7). Other selector bits are ignored on A5.
    """
    return _ir_core.create_op_call("simt.prmt", [lhs, rhs, selector], {}, _get_span_or_capture(span))


def mulhi(lhs: Expr, rhs: Expr, *, unsigned: bool = False, span: Span | None = None) -> Call:
    """Return the high 32 bits of the signed or unsigned 64-bit product.

    Operands must have INT32 scalar dtype; no implicit promotion is performed.
    unsigned=True interprets both carriers as unsigned; the result stays INT32.
    """
    return _ir_core.create_op_call(
        "simt.mulhi", [lhs, rhs], {"unsigned": unsigned}, _get_span_or_capture(span)
    )


def ftoi(
    value: Expr,
    *,
    dtype: DataType | None = None,
    rounding: str = "z",
    unsigned: bool = False,
    span: Span | None = None,
) -> Call:
    """Convert FP16/BF16/FP32 to a saturated 32/64-bit integer with explicit rounding.

    The default destination is INT32. A 64-bit destination requires FP32 input.
    Rounding: r = nearest even, a = nearest away, f = floor, c = ceil,
    z = toward zero. unsigned interprets signed integer carriers as unsigned; unsigned dtypes
    always use unsigned conversion.
    Saturation is mandatory: out-of-range values clamp to the integer limits.
    NaN conversion follows the target contract; NaN payloads are not preserved.
    """
    return _ir_core.create_op_call(
        "simt.ftoi",
        [value],
        {"rounding": rounding, "unsigned": unsigned, **({"dtype": dtype} if dtype is not None else {})},
        _get_span_or_capture(span),
    )


def itof(
    value: Expr,
    *,
    dtype: DataType | None = None,
    rounding: str = "r",
    unsigned: bool = False,
    span: Span | None = None,
) -> Call:
    """Convert a 32/64-bit integer to FP16/BF16/FP32 with explicit rounding.

    The default destination is FP32. A 64-bit source requires FP32 output.
    Rounding: r = nearest even, a = nearest away, f = floor, c = ceil,
    z = toward zero. unsigned interprets signed integer carriers as unsigned; unsigned dtypes
    always use unsigned conversion.
    """
    return _ir_core.create_op_call(
        "simt.itof",
        [value],
        {"rounding": rounding, "unsigned": unsigned, **({"dtype": dtype} if dtype is not None else {})},
        _get_span_or_capture(span),
    )


def veccore_id(span: Span | None = None) -> Call:
    """Return the target vector core identifier as an INT32 scalar."""
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.veccore_id", [], {}, span)


def trap(span: Span | None = None) -> Call:
    """Request device termination at runtime.

    Termination and host error reporting depend on the runtime; use explicit
    control flow to guard unsafe accesses. This is not a compile-time assertion.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.trap", [], {}, span)


def assert_(cond: bool | Expr, span: Span | None = None) -> Call:
    """Request a device trap when the BOOL runtime predicate is false.

    Termination and host error reporting depend on the runtime; use explicit
    control flow to guard unsafe accesses. This is not a compile-time assertion.
    """
    span = _get_span_or_capture(span)
    return _predicate_call("simt.assert_", cond, span)


def div(lhs: Expr, rhs: Expr, *, precision: str = "intrinsic", span: Span | None = None) -> Call:
    """Divide FP32 operands with explicit target precision.

    precision is intrinsic or precision_0ulp_ftz_true. The latter requests
    the target 0-ULP division mode with flush-to-zero behavior on A5.
    Ordinary / is unchanged; exceptional values follow target semantics.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.div", [lhs, rhs], {"precision": precision}, span)


def rem(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Return the FP32 remainder using a quotient truncated toward zero.

    This follows fmod-style semantics, including the dividend sign; it is
    distinct from Python floor modulo. Subnormals and signed zero are preserved.
    A zero divisor, infinite dividend, or NaN operand produces NaN; a finite
    dividend with an infinite divisor is returned unchanged.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.rem", [lhs, rhs], {}, span)


def mul_i32toi64(lhs: Expr, rhs: Expr, *, unsigned: bool = False, span: Span | None = None) -> Call:
    """Return the widened INT32 product as an INT64 bit carrier.

    unsigned=True zero-extends the two INT32 bit carriers before multiplication;
    otherwise operands are sign-extended. The result retains all 64 product bits.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.mul_i32toi64", [lhs, rhs], {"unsigned": unsigned}, span)


def ftof(
    value: Expr, dtype: DataType, *, rounding: str = "r", saturate: bool = False, span: Span | None = None
) -> Call:
    """Convert between distinct FP16/BF16/FP32 formats with explicit rounding.

    Rounding r/a/f/c/z is supported for every pair; o (round to odd) is
    supported only for FP32 to FP16. saturate requests finite overflow clamping.
    NaN and signed-zero handling follows the target conversion contract.
    """
    return _ir_core.create_op_call(
        "simt.ftof",
        [value],
        {"dtype": dtype, "rounding": rounding, "saturate": saturate},
        _get_span_or_capture(span),
    )
