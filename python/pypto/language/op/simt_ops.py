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
    """Sum active-lane INT32 or FP32 values, returning the input dtype.

    INT32 sums wrap at 32 bits.
    FP32 follows the target reduction order and special-value rules.
    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.redux_add(_unwrap(value)))


def redux_min(value: Scalar | ir.Expr) -> Scalar:
    """Minimum active-lane INT32 (signed) or FP32 value, returning the input dtype.

    FP32 follows the target reduction order and special-value rules.
    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.redux_min(_unwrap(value)))


def redux_max(value: Scalar | ir.Expr) -> Scalar:
    """Maximum active-lane INT32 (signed) or FP32 value, returning the input dtype.

    FP32 follows the target reduction order and special-value rules.
    Uses the current active lanes and does not provide a memory fence.
    """
    return Scalar(expr=_simt.redux_max(_unwrap(value)))


def _shuffle_control(value: int | Scalar | ir.Expr) -> int | ir.Expr:
    """Give DSL integer literals the fixed i32 lane ABI without narrowing variables."""
    control = value if isinstance(value, int) else _unwrap(value)
    # The parser uses INDEX placeholders for bare integers. Explicitly typed
    # expressions and runtime variables retain their dtype for IR validation.
    if isinstance(control, ir.ConstInt) and control.dtype == DataType.INDEX:
        return ir.ConstInt(control.value, DataType.INT32, control.span)
    return control


def shuffle_idx(value: Scalar | ir.Expr, index: int | Scalar | ir.Expr, *, width: int = 32) -> Scalar:
    """Read an INT32 or FP32 value from a subgroup-relative lane index.

    Width must be 16 or 32. The source must be an active lane inside the
    subgroup; results for inactive or out-of-range sources are unspecified.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    control = _shuffle_control(index)
    return Scalar(expr=_simt.shuffle_idx(_unwrap(value), control, width=width))


def shuffle_up(value: Scalar | ir.Expr, offset: int | Scalar | ir.Expr, *, width: int = 32) -> Scalar:
    """Read an INT32 or FP32 value from a lower lane in the subgroup.

    Width must be 16 or 32. The source must be an active lane inside the
    subgroup; results for inactive or out-of-range sources are unspecified.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    control = _shuffle_control(offset)
    return Scalar(expr=_simt.shuffle_up(_unwrap(value), control, width=width))


def shuffle_down(value: Scalar | ir.Expr, offset: int | Scalar | ir.Expr, *, width: int = 32) -> Scalar:
    """Read an INT32 or FP32 value from a higher lane in the subgroup.

    Width must be 16 or 32. The source must be an active lane inside the
    subgroup; results for inactive or out-of-range sources are unspecified.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    control = _shuffle_control(offset)
    return Scalar(expr=_simt.shuffle_down(_unwrap(value), control, width=width))


def shuffle_bfly(value: Scalar | ir.Expr, mask: int | Scalar | ir.Expr, *, width: int = 32) -> Scalar:
    """Read an INT32 or FP32 value from the XOR-selected lane in the subgroup.

    Width must be 16 or 32. The source must be an active lane inside the
    subgroup; results for inactive or out-of-range sources are unspecified.
    Lane controls are INT32. This operation does not provide a memory fence.
    """
    control = _shuffle_control(mask)
    return Scalar(expr=_simt.shuffle_bfly(_unwrap(value), control, width=width))


def atomic_add(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically add one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
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
    l2cache: str | None = None,
) -> Scalar:
    """Atomically min one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    INT32 comparison is signed; FP32 special values follow the target rules.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.atomic_min(_unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache))


def atomic_max(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically max one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
    The update is retained when the result is unused. No additional ordering or
    rendezvous is implied; l2cache is available only for GM storage.
    INT32 comparison is signed; FP32 special values follow the target rules.
    """
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.atomic_max(_unwrap(storage), coordinates, _unwrap(value), l2cache=l2cache))


def atomic_exch(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l2cache: str | None = None,
) -> Scalar:
    """Atomically exch one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
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
    """Atomically cas one initialized INT32/FP32 element and return its old value.

    Operands must match the storage dtype. Indices use the storage logical layout.
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


def ftoi(value: Scalar | ir.Expr, *, rounding: str = "z", unsigned: bool = False) -> Scalar:
    """Convert FP32 to a saturated INT32 bit carrier with explicit rounding.

    Operands must have FP32 scalar dtype; no implicit promotion is performed.
    Rounding: r = nearest even, a = nearest away, f = floor, c = ceil,
    z = toward zero. unsigned interprets the integer carrier as UINT32.
    Saturation is mandatory: out-of-range values clamp to the integer limits.
    NaN conversion follows the target contract; NaN payloads are not preserved.
    """
    return Scalar(expr=_simt.ftoi(_unwrap(value), rounding=rounding, unsigned=unsigned))


def itof(value: Scalar | ir.Expr, *, rounding: str = "r", unsigned: bool = False) -> Scalar:
    """Convert an INT32 bit carrier to FP32 with explicit rounding.

    Operands must have INT32 scalar dtype; no implicit promotion is performed.
    Rounding: r = nearest even, a = nearest away, f = floor, c = ceil,
    z = toward zero. unsigned interprets the integer carrier as UINT32.
    """
    return Scalar(expr=_simt.itof(_unwrap(value), rounding=rounding, unsigned=unsigned))
