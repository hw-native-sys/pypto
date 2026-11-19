# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar operations inside an explicit SIMT execution region."""

from collections.abc import Sequence
from typing import Any

from pypto.ir.op import simt as _simt
from pypto.language.typing import Scalar, Tensor, Tile
from pypto.pypto_core import DataType, ir

Axis = ir.SimtAxis


def _unwrap(value: Scalar | Tensor | Tile | ir.Expr) -> ir.Expr:
    return value if isinstance(value, ir.Expr) else value.unwrap()


def thread_idx(axis: ir.SimtAxis) -> Scalar:
    """Thread coordinate along the required typed axis."""
    return Scalar(expr=_simt.thread_idx(axis))


def block_idx(axis: ir.SimtAxis) -> Scalar:
    """Block coordinate along the required typed axis."""
    return Scalar(expr=_simt.block_idx(axis))


def block_dim(axis: ir.SimtAxis) -> Scalar:
    """Thread-block extent along the required typed axis."""
    return Scalar(expr=_simt.block_dim(axis))


def grid_dim(axis: ir.SimtAxis) -> Scalar:
    """Grid extent along the required typed axis."""
    return Scalar(expr=_simt.grid_dim(axis))


def lane_id() -> Scalar:
    """Lane identifier within the current warp."""
    return Scalar(expr=_simt.lane_id())


def lanemask_eq() -> Scalar:
    """Return the physical equal-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return Scalar(expr=_simt.lanemask_eq())


def lanemask_le() -> Scalar:
    """Return the physical lower-or-equal-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return Scalar(expr=_simt.lanemask_le())


def lanemask_lt() -> Scalar:
    """Return the physical lower-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return Scalar(expr=_simt.lanemask_lt())


def lanemask_ge() -> Scalar:
    """Return the physical higher-or-equal-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return Scalar(expr=_simt.lanemask_ge())


def lanemask_gt() -> Scalar:
    """Return the physical higher-lane mask as an INT32 bit carrier.

    Bit i describes lane i relative to this lane; this is not an active-lane mask.
    """
    return Scalar(expr=_simt.lanemask_gt())


def bitcast(value: Scalar | ir.Expr, dtype: DataType) -> Scalar:
    """Reinterpret an unpacked numeric scalar as dtype without changing any bits.

    Source and destination must have equal bit width: FP16/BF16/FP32 or
    8/16/32/64-bit integers. NaN payloads and signed zero bits
    are preserved; this is not a numerical conversion or a storage view.
    """
    return Scalar(expr=_simt.bitcast(_unwrap(value), dtype))


def load(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    *,
    l1cache: str | None = None,
    l2cache: str | None = None,
) -> Scalar:
    """Read a scalar from existing GM or Vec/UB storage."""
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.load(_unwrap(storage), coordinates, l1cache=l1cache, l2cache=l2cache))


def store(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l1cache: str | None = None,
    l2cache: str | None = None,
) -> ir.Call:
    """Write a scalar; untouched elements retain their pre-region contents."""
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return _simt.store(_unwrap(storage), coordinates, _unwrap(value), l1cache=l1cache, l2cache=l2cache)


def launch(
    helper: object,
    *args: object,
    block_dims: tuple[int, int, int],
    result_param_indices: tuple[int, ...] = (),
) -> Any:
    """Explicit canonical helper launch, interpreted by the DSL parser."""
    raise RuntimeError("pl.simt.launch is only valid inside a parsed PyPTO function")


def _predicate(value: bool | Scalar | ir.Expr) -> bool | ir.Expr:
    return value if isinstance(value, bool) else _unwrap(value)


def vote_all(predicate: bool | Scalar | ir.Expr) -> Scalar:
    """Whether every active lane has a true BOOL predicate.

    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.vote_all(_predicate(predicate)))


def vote_any(predicate: bool | Scalar | ir.Expr) -> Scalar:
    """Whether any active lane has a true BOOL predicate.

    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.vote_any(_predicate(predicate)))


def vote_uni(predicate: bool | Scalar | ir.Expr) -> Scalar:
    """Whether all active lanes agree on their BOOL predicate.

    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.vote_uni(_predicate(predicate)))


def vote_ballot(predicate: bool | Scalar | ir.Expr) -> Scalar:
    """INT32 bit mask of true active-lane predicates; bit i denotes lane i.

    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.vote_ballot(_predicate(predicate)))


def redux_add(value: Scalar | ir.Expr) -> Scalar:
    """Sum active-lane FP16/FP32 or 32-bit integer values, returning the input dtype.

    INT32 sums wrap at 32 bits.
    FP32 follows the target reduction order and special-value rules.
    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.redux_add(_unwrap(value)))


def redux_min(value: Scalar | ir.Expr, *, unsigned: bool = False) -> Scalar:
    """Minimum active-lane FP16/FP32 or 32-bit integer value, returning the input dtype.

    unsigned=True compares INT32 bit carriers as unsigned.
    FP32 follows the target reduction order and special-value rules.
    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.redux_min(_unwrap(value), unsigned=unsigned))


def redux_max(value: Scalar | ir.Expr, *, unsigned: bool = False) -> Scalar:
    """Maximum active-lane FP16/FP32 or 32-bit integer value, returning the input dtype.

    unsigned=True compares INT32 bit carriers as unsigned.
    FP32 follows the target reduction order and special-value rules.
    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.redux_max(_unwrap(value), unsigned=unsigned))


def _shuffle_control(value: int | Scalar | ir.Expr) -> int | ir.Expr:
    """Give DSL integer literals the fixed i32 lane ABI without narrowing variables."""
    control = value if isinstance(value, int) else _unwrap(value)
    # The parser uses INDEX placeholders for bare integers. Explicitly typed
    # expressions and runtime variables retain their dtype for IR validation.
    if isinstance(control, ir.ConstInt) and control.dtype == DataType.INDEX:
        return ir.ConstInt(control.value, DataType.INT32, control.span)
    return control


def shuffle_idx(
    value: Scalar | ir.Expr, index: int | Scalar | ir.Expr, *, width: int = 32, bound: int | None = None
) -> Scalar:
    """Read an FP16/FP32 or 32/64-bit integer value from a subgroup-relative lane index.

    Width must be 16 or 32. Bound must be in [0, width); omitted means the full source window.
    Out-of-window sources return the caller value; inactive sources are undefined.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    control = _shuffle_control(index)
    return Scalar(expr=_simt.shuffle_idx(_unwrap(value), control, width=width, bound=bound))


def shuffle_up(
    value: Scalar | ir.Expr, offset: int | Scalar | ir.Expr, *, width: int = 32, bound: int | None = None
) -> Scalar:
    """Read an FP16/FP32 or 32/64-bit integer value from a lower lane in the subgroup.

    Width must be 16 or 32. Bound must be in [0, width); omitted means the full source window.
    Out-of-window sources return the caller value; inactive sources are undefined.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    control = _shuffle_control(offset)
    return Scalar(expr=_simt.shuffle_up(_unwrap(value), control, width=width, bound=bound))


def shuffle_down(
    value: Scalar | ir.Expr, offset: int | Scalar | ir.Expr, *, width: int = 32, bound: int | None = None
) -> Scalar:
    """Read an FP16/FP32 or 32/64-bit integer value from a higher lane in the subgroup.

    Width must be 16 or 32. Bound must be in [0, width); omitted means the full source window.
    Out-of-window sources return the caller value; inactive sources are undefined.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    control = _shuffle_control(offset)
    return Scalar(expr=_simt.shuffle_down(_unwrap(value), control, width=width, bound=bound))


def shuffle_bfly(
    value: Scalar | ir.Expr, mask: int | Scalar | ir.Expr, *, width: int = 32, bound: int | None = None
) -> Scalar:
    """Read an FP16/FP32 or 32/64-bit integer value from the XOR-selected lane in the subgroup.

    Width must be 16 or 32. Bound must be in [0, width); omitted means the full source window.
    Out-of-window sources return the caller value; inactive sources are undefined.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    control = _shuffle_control(mask)
    return Scalar(expr=_simt.shuffle_bfly(_unwrap(value), control, width=width, bound=bound))


def atomic_sub(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically subtract from an initialized element and return its old value.

    Supports 32/64-bit integers and FP16/BF16/FP32; 64-bit atomics require GM.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.atomic_sub(_unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache))


def atomic_and(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically apply bitwise AND to one initialized 32/64-bit integer element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.atomic_and(_unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache))


def atomic_or(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically apply bitwise OR to one initialized 32/64-bit integer element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.atomic_or(_unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache))


def atomic_xor(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically apply bitwise XOR to one initialized 32/64-bit integer element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.atomic_xor(_unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache))


def atomic_add(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically add one initialized 32/64-bit integer or FP16/BF16/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.atomic_add(_unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache))


def atomic_min(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    unsigned: bool = False,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically min one initialized 32/64-bit integer or FP16/BF16/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    unsigned=True compares INT32 bit carriers as unsigned; FP32 special values follow the target rules.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(
        expr=_simt.atomic_min(
            _unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache, unsigned=unsigned
        )
    )


def atomic_max(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    unsigned: bool = False,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically max one initialized 32/64-bit integer or FP16/BF16/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    unsigned=True compares INT32 bit carriers as unsigned; FP32 special values follow the target rules.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(
        expr=_simt.atomic_max(
            _unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache, unsigned=unsigned
        )
    )


def atomic_exch(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically exch one initialized 32/64-bit integer or FP16/BF16/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.atomic_exch(_unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache))


def atomic_cas(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    compare: Scalar | ir.Expr,
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically cas one initialized 32/64-bit integer or FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    All 64-bit atomics require GM storage; UB accepts only the narrower forms.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    Comparison is bitwise, including FP32 NaN payloads and signed zeros.
    Replace only on a match; the result is the observed old value.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(
        expr=_simt.atomic_cas(
            _unwrap(storage), coordinates, _unwrap(compare), _unwrap(value), l2cache=l2cache
        )
    )


def syncthreads() -> ir.Call:
    """Synchronize all threads in the current SIMT entry and order their memory effects.

    Must be reached unconditionally by all threads. Nested control flow is
    conservatively rejected. This is not an outer MTE/Vector pipeline barrier."""
    return _simt.syncthreads()


def threadfence() -> ir.Call:
    """Issue the target workitem memory fence for the calling thread.

    Does not wait for other threads or provide cross-AIV rendezvous.
    Ordering and visibility follow the native target fence contract."""
    return _simt.threadfence()


def threadfence_block() -> ir.Call:
    """Issue the target block-scoped memory fence for the calling thread.

    Does not wait for other threads; use syncthreads for a rendezvous.
    This does not replace outer MTE/Vector pipeline synchronization."""
    return _simt.threadfence_block()


def abs(value: Scalar | ir.Expr) -> Scalar:
    """Absolute value; clears the FP32 sign bit.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.abs(_unwrap(value)))


def min(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr) -> Scalar:
    """FP32 minimum with minNum semantics; one NaN yields the numeric operand.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.min(_unwrap(lhs), _unwrap(rhs)))


def max(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr) -> Scalar:
    """FP32 maximum with maxNum semantics; one NaN yields the numeric operand.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.max(_unwrap(lhs), _unwrap(rhs)))


def exp(value: Scalar | ir.Expr) -> Scalar:
    """Compute the FP32 natural exponential using target math.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.exp(_unwrap(value)))


def log(value: Scalar | ir.Expr) -> Scalar:
    """Compute the FP32 natural logarithm using target math.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.log(_unwrap(value)))


def sqrt(value: Scalar | ir.Expr) -> Scalar:
    """Compute the FP32 square root using target math.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.sqrt(_unwrap(value)))


def sin(value: Scalar | ir.Expr) -> Scalar:
    """Compute the FP32 sine in radians using target math.

    PTOAS uses FP32 software range reduction; large angles lose accuracy.
    Numerical acceptance covers [-4, 4]; no full-range accuracy is promised.
    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.sin(_unwrap(value)))


def cos(value: Scalar | ir.Expr) -> Scalar:
    """Compute the FP32 cosine in radians using target math.

    PTOAS uses FP32 software range reduction; large angles lose accuracy.
    Numerical acceptance covers [-4, 4]; no full-range accuracy is promised.
    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.cos(_unwrap(value)))


def ceil(value: Scalar | ir.Expr) -> Scalar:
    """Round FP32 toward positive infinity, retaining FP32.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.ceil(_unwrap(value)))


def floor(value: Scalar | ir.Expr) -> Scalar:
    """Round FP32 toward negative infinity, retaining FP32.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.floor(_unwrap(value)))


def rint(value: Scalar | ir.Expr) -> Scalar:
    """Round FP32 to nearest integer, ties to even, retaining FP32.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.rint(_unwrap(value)))


def round(value: Scalar | ir.Expr) -> Scalar:
    """Round FP32 to nearest integer, ties away from zero, retaining FP32.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.round(_unwrap(value)))


def pow(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr) -> Scalar:
    """Compute FP32 power using target math.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.pow(_unwrap(lhs), _unwrap(rhs)))


def fma(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr, acc: Scalar | ir.Expr) -> Scalar:
    """Compute FP32 lhs * rhs + acc with one final rounding.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    """
    return Scalar(expr=_simt.fma(_unwrap(lhs), _unwrap(rhs), _unwrap(acc)))


def prmt(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr, selector: Scalar | ir.Expr) -> Scalar:
    """Permute bytes from two INT32 bit carriers with an INT32 selector.

    Operands must have INT32 scalar dtype; no implicit promotion is performed.
    The low three bits of each low selector nibble pick a byte from rhs
    (0-3) or lhs (4-7). Other selector bits are ignored on A5.
    """
    return Scalar(expr=_simt.prmt(_unwrap(lhs), _unwrap(rhs), _unwrap(selector)))


def mulhi(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr, *, unsigned: bool = False) -> Scalar:
    """Return the high 32 bits of the signed or unsigned 64-bit product.

    Operands must have INT32 scalar dtype; no implicit promotion is performed.
    unsigned=True interprets both carriers as unsigned; the result stays INT32.
    """
    return Scalar(expr=_simt.mulhi(_unwrap(lhs), _unwrap(rhs), unsigned=unsigned))


def ftoi(
    value: Scalar | ir.Expr, *, dtype: DataType | None = None, rounding: str = "z", unsigned: bool = False
) -> Scalar:
    """Convert FP16/BF16/FP32 to a saturated 32/64-bit integer with explicit rounding.

    The default destination is INT32. A 64-bit destination requires FP32 input.
    Rounding: r = nearest even, a = nearest away, f = floor, c = ceil,
    z = toward zero. unsigned interprets signed integer carriers as unsigned; unsigned dtypes
    always use unsigned conversion.
    Saturation is mandatory: out-of-range values clamp to the integer limits.
    NaN conversion follows the target contract; NaN payloads are not preserved.
    """
    return Scalar(expr=_simt.ftoi(_unwrap(value), dtype=dtype, rounding=rounding, unsigned=unsigned))


def itof(
    value: Scalar | ir.Expr, *, dtype: DataType | None = None, rounding: str = "r", unsigned: bool = False
) -> Scalar:
    """Convert a 32/64-bit integer to FP16/BF16/FP32 with explicit rounding.

    The default destination is FP32. A 64-bit source requires FP32 output.
    Rounding: r = nearest even, a = nearest away, f = floor, c = ceil,
    z = toward zero. unsigned interprets signed integer carriers as unsigned; unsigned dtypes
    always use unsigned conversion.
    """
    return Scalar(expr=_simt.itof(_unwrap(value), dtype=dtype, rounding=rounding, unsigned=unsigned))


def veccore_id() -> Scalar:
    """Return the target vector core identifier as an INT32 scalar."""
    return Scalar(expr=_simt.veccore_id())


def trap() -> ir.Call:
    """Request device termination at runtime.

    Termination and host error reporting depend on the runtime; use explicit
    control flow to guard unsafe accesses. This is not a compile-time assertion.
    """
    return _simt.trap()


def assert_(cond: bool | Scalar | ir.Expr) -> ir.Call:
    """Request a device trap when the BOOL runtime predicate is false.

    Termination and host error reporting depend on the runtime; use explicit
    control flow to guard unsafe accesses. This is not a compile-time assertion.
    """
    return _simt.assert_(cond if isinstance(cond, bool) else _unwrap(cond))


def div(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr, *, precision: str = "intrinsic") -> Scalar:
    """Divide FP32 operands with explicit target precision.

    precision is intrinsic or precision_0ulp_ftz_true. The latter requests
    the target 0-ULP division mode with flush-to-zero behavior on A5.
    Ordinary / is unchanged; exceptional values follow target semantics.
    """
    return Scalar(expr=_simt.div(_unwrap(lhs), _unwrap(rhs), precision=precision))


def rem(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr) -> Scalar:
    """Return the FP32 remainder using a quotient truncated toward zero.

    This follows fmod-style semantics, including the dividend sign; it is
    distinct from Python floor modulo. Exceptional values follow the target.
    """
    return Scalar(expr=_simt.rem(_unwrap(lhs), _unwrap(rhs)))


def mul_i32toi64(lhs: Scalar | ir.Expr, rhs: Scalar | ir.Expr, *, unsigned: bool = False) -> Scalar:
    """Return the widened INT32 product as an INT64 bit carrier.

    unsigned=True zero-extends the two INT32 bit carriers before multiplication;
    otherwise operands are sign-extended. The result retains all 64 product bits.
    """
    return Scalar(expr=_simt.mul_i32toi64(_unwrap(lhs), _unwrap(rhs), unsigned=unsigned))


def ftof(value: Scalar | ir.Expr, dtype: DataType, *, rounding: str = "r", saturate: bool = False) -> Scalar:
    """Convert between distinct FP16/BF16/FP32 formats with explicit rounding.

    Rounding r/a/f/c/z is supported for every pair; o (round to odd) is
    supported only for FP32 to FP16. saturate requests finite overflow clamping.
    NaN and signed-zero handling follows the target conversion contract.
    """
    return Scalar(expr=_simt.ftof(_unwrap(value), dtype, rounding=rounding, saturate=saturate))
