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
Cross-core ops that take tile args get pl-level wrappers that
accept Tile and unwrap to Expr before calling the IR-level functions.
"""

from pypto.ir.op import system_ops as _ir_ops
from pypto.ir.op.system_ops import (
    aic_initialize_pipe,
    aiv_initialize_pipe,
    bar_all,
    bar_m,
    bar_v,
    import_peer_buffer,
    reserve_buffer,
    sync_dst,
    sync_src,
)
from pypto.pypto_core.ir import Call, Span

from ..typing import Tile

__all__ = [
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
]


def tpush_to_aiv(tile: Tile, *, aiv_idx: int, span: Span | None = None) -> Call:
    """Push tile data from AIC to AIV via cross-core pipe."""
    return _ir_ops.tpush_to_aiv(tile.unwrap(), aiv_idx=aiv_idx, span=span)


def tpush_to_aic(tile: Tile, *, aiv_idx: int, span: Span | None = None) -> Call:
    """Push tile data from AIV to AIC via cross-core pipe."""
    return _ir_ops.tpush_to_aic(tile.unwrap(), aiv_idx=aiv_idx, span=span)


def tpop_from_aic(tile: Tile, *, aiv_idx: int, span: Span | None = None) -> Tile:
    """Pop tile data from AIC cross-core pipe into AIV."""
    return Tile(expr=_ir_ops.tpop_from_aic(tile.unwrap(), aiv_idx=aiv_idx, span=span))


def tpop_from_aiv(tile: Tile, *, aiv_idx: int, span: Span | None = None) -> Tile:
    """Pop tile data from AIV cross-core pipe into AIC."""
    return Tile(expr=_ir_ops.tpop_from_aiv(tile.unwrap(), aiv_idx=aiv_idx, span=span))
