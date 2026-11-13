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


def _vote_call(name: str, predicate: bool | Expr, span: Span) -> Call:
    value = _ir_core.ConstBool(predicate, span) if isinstance(predicate, bool) else predicate
    return _ir_core.create_op_call(name, [value], {}, span)


def vote_all(predicate: bool | Expr, span: Span | None = None) -> Call:
    """Return whether every active lane has a true BOOL predicate.

    Uses the current active lanes; this operation does not provide a memory fence.
    """
    return _vote_call("simt.vote_all", predicate, _get_span_or_capture(span))


def vote_any(predicate: bool | Expr, span: Span | None = None) -> Call:
    """Return whether any active lane has a true BOOL predicate.

    Uses the current active lanes; this operation does not provide a memory fence.
    """
    return _vote_call("simt.vote_any", predicate, _get_span_or_capture(span))


def vote_uni(predicate: bool | Expr, span: Span | None = None) -> Call:
    """Return whether all active lanes agree on their BOOL predicate.

    Uses the current active lanes; this operation does not provide a memory fence.
    """
    return _vote_call("simt.vote_uni", predicate, _get_span_or_capture(span))


def vote_ballot(predicate: bool | Expr, span: Span | None = None) -> Call:
    """Return an INT32 bit mask of true predicates in active lanes; bit i denotes lane i.

    Uses the current active lanes; this operation does not provide a memory fence.
    """
    return _vote_call("simt.vote_ballot", predicate, _get_span_or_capture(span))


def _lane_control(value: int | Expr, span: Span) -> Expr:
    """Normalize Python integer controls without treating bool as a lane index."""
    if isinstance(value, bool):
        raise TypeError("SIMT lane control requires INT32, got bool")
    return _normalize_expr(value, span, int_dtype=DataType.INT32)


def shuffle_idx(value: Expr, index: int | Expr, *, width: int = 32, span: Span | None = None) -> Call:
    """Read an INT32 or FP32 value from a subgroup-relative lane index.

    Width must be 16 or 32. The source must be an active lane inside the
    subgroup; results for inactive or out-of-range sources are unspecified.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.shuffle_idx",
        [value, _lane_control(index, span)],
        {"width": width},
        span,
    )


def shuffle_up(value: Expr, offset: int | Expr, *, width: int = 32, span: Span | None = None) -> Call:
    """Read an INT32 or FP32 value from a lower lane in the subgroup.

    Width must be 16 or 32. The source must be an active lane inside the
    subgroup; results for inactive or out-of-range sources are unspecified.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.shuffle_up",
        [value, _lane_control(offset, span)],
        {"width": width},
        span,
    )


def shuffle_down(value: Expr, offset: int | Expr, *, width: int = 32, span: Span | None = None) -> Call:
    """Read an INT32 or FP32 value from a higher lane in the subgroup.

    Width must be 16 or 32. The source must be an active lane inside the
    subgroup; results for inactive or out-of-range sources are unspecified.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.shuffle_down",
        [value, _lane_control(offset, span)],
        {"width": width},
        span,
    )


def shuffle_bfly(value: Expr, mask: int | Expr, *, width: int = 32, span: Span | None = None) -> Call:
    """Read an INT32 or FP32 value from the XOR-selected lane in the subgroup.

    Width must be 16 or 32. The source must be an active lane inside the
    subgroup; results for inactive or out-of-range sources are unspecified.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.shuffle_bfly",
        [value, _lane_control(mask, span)],
        {"width": width},
        span,
    )


def redux_add(value: Expr, span: Span | None = None) -> Call:
    """Return the sum of active-lane INT32 or FP32 values to each active lane.

    The result retains the input dtype. INT32 sums wrap at 32 bits.
    FP32 reductions follow the target arithmetic order and special-value rules.
    This operation does not provide a memory fence.
    """
    return _ir_core.create_op_call("simt.redux_add", [value], {}, _get_span_or_capture(span))


def redux_min(value: Expr, span: Span | None = None) -> Call:
    """Return the minimum of active-lane INT32 or FP32 values to each active lane.

    The result retains the input dtype. INT32 comparisons are signed.
    FP32 reductions follow the target arithmetic order and special-value rules.
    This operation does not provide a memory fence.
    """
    return _ir_core.create_op_call("simt.redux_min", [value], {}, _get_span_or_capture(span))


def redux_max(value: Expr, span: Span | None = None) -> Call:
    """Return the maximum of active-lane INT32 or FP32 values to each active lane.

    The result retains the input dtype. INT32 comparisons are signed.
    FP32 reductions follow the target arithmetic order and special-value rules.
    This operation does not provide a memory fence.
    """
    return _ir_core.create_op_call("simt.redux_max", [value], {}, _get_span_or_capture(span))


def atomic_add(
    storage: Expr,
    indices: Sequence[int | Expr],
    value: Expr,
    *,
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically add one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
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
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically min one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    INT32 comparison is signed; FP32 special values follow the target rules.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
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
    l2cache: str | None = None,
    span: Span | None = None,
) -> Call:
    """Atomically max one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    INT32 comparison is signed; FP32 special values follow the target rules.
    """
    span = _get_span_or_capture(span)
    kwargs = {"l2cache": l2cache} if l2cache is not None else {}
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
    """Atomically exch one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
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
    """Atomically cas one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
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


def ftoi(value: Expr, *, rounding: str = "z", unsigned: bool = False, span: Span | None = None) -> Call:
    """Convert FP32 to a saturated INT32 bit carrier with explicit rounding.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    Rounding: r = nearest even, a = nearest away, f = floor, c = ceil,
    z = toward zero. unsigned interprets the integer carrier as UINT32.
    Saturation is mandatory: out-of-range values clamp to the integer limits.
    NaN conversion follows the target contract; NaN payloads are not preserved.
    """
    return _ir_core.create_op_call(
        "simt.ftoi", [value], {"rounding": rounding, "unsigned": unsigned}, _get_span_or_capture(span)
    )


def itof(value: Expr, *, rounding: str = "r", unsigned: bool = False, span: Span | None = None) -> Call:
    """Convert an INT32 bit carrier to FP32 with explicit rounding.

    Operands must have INT32 scalar dtype; no implicit promotion is performed.
    Rounding: r = nearest even, a = nearest away, f = floor, c = ceil,
    z = toward zero. unsigned interprets the integer carrier as UINT32.
    """
    return _ir_core.create_op_call(
        "simt.itof", [value], {"rounding": rounding, "unsigned": unsigned}, _get_span_or_capture(span)
    )
