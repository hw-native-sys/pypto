# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System operations for PyPTO Language DSL.

Sync/barrier ops are straight pass-through (no Tensor/Tile args).
tpush ops wrap the IR-level functions, unwrapping Tile to Expr.
tpop ops accept optional shape/dtype kwargs to create typed results.
"""

from pypto.ir.op import system_ops as _ir_ops
from pypto.ir.op import tile_ops as _ir_tile_ops
from pypto.ir.op.system_ops import (
    AUTO,
    aic_initialize_pipe,
    aiv_initialize_pipe,
    bar_all,
    bar_m,
    bar_v,
    sync_dst,
    sync_src,
)
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Call, ConstInt, Expr, Span

from ..typing import Scalar, Tensor, Tile

__all__ = [
    "AUTO",
    "sync_src",
    "sync_dst",
    "bar_v",
    "bar_m",
    "bar_all",
    "tpush_to_aiv",
    "tpush_to_aic",
    "tpop_from_aic",
    "tpop_from_aiv",
    "aic_initialize_pipe",
    "aiv_initialize_pipe",
    "reserve_buffer",
    "import_peer_buffer",
    "tfree_to_aic",
    "tfree_to_aiv",
    "comm_notify",
    "comm_wait",
    "comm_test",
]


def tpush_to_aiv(tile: Tile, *, split: int, id: int | None = None, span: Span | None = None) -> Call:
    """Push tile data from AIC to AIV via cross-core pipe."""
    return _ir_ops.tpush_to_aiv(tile.unwrap(), split=split, id=id, span=span)


def tpush_to_aic(tile: Tile, *, split: int, id: int | None = None, span: Span | None = None) -> Call:
    """Push tile data from AIV to AIC via cross-core pipe."""
    return _ir_ops.tpush_to_aic(tile.unwrap(), split=split, id=id, span=span)


def tfree_to_aic(tile: Tile, span: Span | None = None, *, id: int | None = None) -> Call:
    """Release ring buffer slot back to AIC producer."""
    return _ir_ops.tfree_to_aic(tile.unwrap(), id=id, span=span)


def tfree_to_aiv(tile: Tile, span: Span | None = None, *, id: int | None = None) -> Call:
    """Release ring buffer slot back to AIV producer."""
    return _ir_ops.tfree_to_aiv(tile.unwrap(), id=id, span=span)


def tpop_from_aic(
    *,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    split: int = 0,
    id: int | None = None,
    span: Span | None = None,
) -> Tile:
    """Pop tile data from AIC cross-core pipe into AIV.

    Args:
        shape: Shape of the tile to receive
        dtype: Data type of the tile to receive
        split: Split mode (0=none, 1=up-down, 2=left-right)
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    call = _ir_ops.tpop_from_aic(shape=shape, dtype=dtype, split=split, id=id, span=span)
    return Tile(expr=call)


def tpop_from_aiv(
    *,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    split: int = 0,
    id: int | None = None,
    span: Span | None = None,
) -> Tile:
    """Pop tile data from AIV cross-core pipe into AIC.

    Args:
        shape: Shape of the tile to receive
        dtype: Data type of the tile to receive
        split: Split mode (0=none, 1=up-down, 2=left-right)
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    call = _ir_ops.tpop_from_aiv(shape=shape, dtype=dtype, split=split, id=id, span=span)
    return Tile(expr=call)


def reserve_buffer(*, name: str, size: int, base: int = AUTO, span: Span | None = None) -> Scalar:
    """Reserve a named buffer for cross-core communication.

    Args:
        name: Buffer name for cross-core reference.
        size: Buffer size in bytes.
        base: Base address in local SRAM. Use AUTO (-1) to let the compiler
              pick a non-conflicting address, or an explicit integer for
              manual kernels.
        span: Optional source span.

    Returns:
        ``pl.Scalar[pl.INT32]`` wrapping the ``system.reserve_buffer`` IR call (PTO ``... -> i32``).
    """
    call = _ir_ops.reserve_buffer(name=name, size=size, base=base, span=span)
    return Scalar(DataType.INT32, call)


def import_peer_buffer(*, name: str, peer_func: str, span: Span | None = None) -> Scalar:
    """Import a buffer from a peer function in the same group.

    Args:
        name: Buffer name to import (must match peer's reserve_buffer name).
        peer_func: Name of the peer function that owns the buffer.
        span: Optional source span.

    Returns:
        ``pl.Scalar[pl.INT32]`` wrapping the ``system.import_peer_buffer`` IR call (PTO ``... -> i32``).
    """
    call = _ir_ops.import_peer_buffer(name=name, peer_func=peer_func, span=span)
    return Scalar(DataType.INT32, call)


def comm_notify(
    signal: Tensor,
    value: int | Scalar | Expr,
    *,
    op: str,
    span: Span | None = None,
) -> Call:
    """Send a flag notification to a remote rank's signal slot.

    Lowers to ``pto::comm::TNOTIFY`` via PTOAS ``pto.comm.tnotify``. The
    signal is a 1-element INT32 ``pl.Tensor`` (GM) that views the destination
    rank's signal location in its HCCL window — typically obtained via
    :func:`import_peer_buffer`.

    Note:
        The cross-rank done-barrier pattern used in production (see the
        ``dispatch.cpp`` / ``combine.cpp`` examples) relies on payload writes
        becoming visible to the peer **before** the done signal is sent.
        PyPTO inserts the required pipe synchronization automatically, but
        callers should ensure no late writes to the payload region remain
        in-flight after ``comm_notify``.

    Args:
        signal: Destination signal tensor (1-element INT32) in remote rank's window.
        value: INT32 scalar value to write or atomic-add (Python int, Scalar, or Expr).
        op: Notify operation, ``"atomic_add"`` or ``"set"``.
        span: Optional source span.

    Returns:
        The IR ``Call`` for ``tile.comm_notify`` (used for its side effect; no return value).
    """
    if isinstance(value, Scalar):
        value_expr: Expr = value.unwrap()
    elif isinstance(value, Expr):
        value_expr = value
    else:
        value_expr = ConstInt(int(value), DataType.INT32, Span.unknown())
    return _ir_tile_ops.comm_notify(signal.unwrap(), value_expr, op=op, span=span)


def comm_wait(
    signal: Tensor,
    cmp_value: int | Scalar | Expr,
    *,
    cmp: str,
    span: Span | None = None,
) -> Call:
    """Block until a local INT32 signal slot satisfies a comparison.

    Lowers to ``pto::comm::TWAIT`` via PTOAS ``pto.comm.twait``. The signal
    is a 1-element INT32 ``pl.Tensor`` (GM) in the local rank's window — the
    slot peers ``pl.tile.comm_notify`` into.

    Args:
        signal: Local signal tensor (1-element INT32) to poll.
        cmp_value: INT32 scalar comparison value (Python int, Scalar, or Expr).
        cmp: Comparison predicate, one of ``"eq"`` | ``"ne"`` | ``"gt"`` |
            ``"ge"`` | ``"lt"`` | ``"le"``.
        span: Optional source span.

    Returns:
        The IR ``Call`` for ``tile.comm_wait`` (used for its side effect; no return value).
    """
    if isinstance(cmp_value, Scalar):
        cmp_expr: Expr = cmp_value.unwrap()
    elif isinstance(cmp_value, Expr):
        cmp_expr = cmp_value
    else:
        cmp_expr = ConstInt(int(cmp_value), DataType.INT32, Span.unknown())
    return _ir_tile_ops.comm_wait(signal.unwrap(), cmp_expr, cmp=cmp, span=span)


def comm_test(
    signal: Tensor,
    cmp_value: int | Scalar | Expr,
    *,
    cmp: str,
    span: Span | None = None,
) -> Scalar:
    """Non-blocking poll of a local INT32 signal slot, returning a BOOL Scalar.

    Lowers to ``pto::comm::TTEST`` via PTOAS ``pto.comm.ttest``. Same operand
    shape as :func:`comm_wait`, but does not block — the result is
    ``pl.Scalar[pl.BOOL]`` and equals ``signal <cmp> cmp_value``.

    Args:
        signal: Local signal tensor (1-element INT32) to poll.
        cmp_value: INT32 scalar comparison value (Python int, Scalar, or Expr).
        cmp: Comparison predicate, one of ``"eq"`` | ``"ne"`` | ``"gt"`` |
            ``"ge"`` | ``"lt"`` | ``"le"``.
        span: Optional source span.

    Returns:
        ``pl.Scalar[pl.BOOL]`` wrapping the ``tile.comm_test`` IR call (PTO ``... -> i1``).
    """
    if isinstance(cmp_value, Scalar):
        cmp_expr: Expr = cmp_value.unwrap()
    elif isinstance(cmp_value, Expr):
        cmp_expr = cmp_value
    else:
        cmp_expr = ConstInt(int(cmp_value), DataType.INT32, Span.unknown())
    call = _ir_tile_ops.comm_test(signal.unwrap(), cmp_expr, cmp=cmp, span=span)
    return Scalar(DataType.BOOL, call)
