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
    tfree_to_aic,
    tfree_to_aiv,
)
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Call, Span

from ..typing import Tile


class ReservedBuffer:
    """Return value from pl.reserve_buffer(), providing access to buffer metadata.

    Attributes:
        base: Base address in local SRAM (int literal or AUTO sentinel).
        size: Buffer size in bytes.
        name: Buffer name for cross-core reference.
    """

    def __init__(self, expr: Call, name: str, size: int, base: int) -> None:
        self._expr = expr
        self.name = name
        self.size = size
        self.base = base


class ImportedBuffer:
    """Return value from pl.import_peer_buffer(), providing access to peer buffer metadata.

    Attributes:
        base: Peer buffer base address (resolved by allocator if peer uses AUTO).
        name: Buffer name matching the peer's reserve_buffer name.
        peer_func: Name of the peer function that owns the buffer.
    """

    def __init__(self, expr: Call, name: str, peer_func: str) -> None:
        self._expr = expr
        self.name = name
        self.peer_func = peer_func
        self.base: int = AUTO  # resolved by allocator pass


__all__ = [
    "AUTO",
    "ImportedBuffer",
    "ReservedBuffer",
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


def tpush_to_aiv(tile: Tile, *, split: int, span: Span | None = None) -> Call:
    """Push tile data from AIC to AIV via cross-core pipe."""
    return _ir_ops.tpush_to_aiv(tile.unwrap(), split=split, span=span)


def tpush_to_aic(tile: Tile, *, split: int, span: Span | None = None) -> Call:
    """Push tile data from AIV to AIC via cross-core pipe."""
    return _ir_ops.tpush_to_aic(tile.unwrap(), split=split, span=span)


def tpop_from_aic(
    *,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    aiv_idx: int,
    span: Span | None = None,
) -> Tile:
    """Pop tile data from AIC cross-core pipe into AIV.

    Args:
        shape: Shape of the tile to receive
        dtype: Data type of the tile to receive
        aiv_idx: Target AIV core index
        span: Optional source span
    """
    call = _ir_ops.tpop_from_aic(shape=shape, dtype=dtype, aiv_idx=aiv_idx, span=span)
    return Tile(expr=call)


def tpop_from_aiv(
    *,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    aiv_idx: int,
    span: Span | None = None,
) -> Tile:
    """Pop tile data from AIV cross-core pipe into AIC.

    Args:
        shape: Shape of the tile to receive
        dtype: Data type of the tile to receive
        aiv_idx: Source AIV core index
        span: Optional source span
    """
    call = _ir_ops.tpop_from_aiv(shape=shape, dtype=dtype, aiv_idx=aiv_idx, span=span)
    return Tile(expr=call)


def reserve_buffer(*, name: str, size: int, base: int = AUTO, span: Span | None = None) -> ReservedBuffer:
    """Reserve a named buffer for cross-core communication.

    Args:
        name: Buffer name for cross-core reference.
        size: Buffer size in bytes.
        base: Base address in local SRAM. Use AUTO (-1) to let the compiler
              pick a non-conflicting address, or an explicit integer for
              manual kernels.
        span: Optional source span.

    Returns:
        ReservedBuffer with .base, .size, .name attributes.
    """
    call = _ir_ops.reserve_buffer(name=name, size=size, base=base, span=span)
    return ReservedBuffer(expr=call, name=name, size=size, base=base)


def import_peer_buffer(*, name: str, peer_func: str, span: Span | None = None) -> ImportedBuffer:
    """Import a buffer from a peer function in the same group.

    Args:
        name: Buffer name to import (must match peer's reserve_buffer name).
        peer_func: Name of the peer function that owns the buffer.
        span: Optional source span.

    Returns:
        ImportedBuffer with .base, .name, .peer_func attributes.
        The .base value is resolved by the allocator pass.
    """
    call = _ir_ops.import_peer_buffer(name=name, peer_func=peer_func, span=span)
    return ImportedBuffer(expr=call, name=name, peer_func=peer_func)
