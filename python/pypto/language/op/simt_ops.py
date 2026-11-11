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
