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
from pypto.pypto_core.ir import Call, Span

from ..typing import Scalar, Tile

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
