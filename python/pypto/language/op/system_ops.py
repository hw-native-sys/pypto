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

from collections.abc import Sequence

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
from pypto.ir.utils import _get_span_or_capture
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Call, ConstInt, MemorySpace, Span

from ..typing import Array, Scalar, Tensor, Tile

# pto::SYNCALL soft barrier reserves 8 int32 slots per participating core.
_SYNCALL_SOFT_SLOT_INT32 = 8

__all__ = [
    "AUTO",
    "sync_src",
    "sync_dst",
    "bar_v",
    "bar_m",
    "bar_all",
    "syncall",
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
    "task_invalid",
    "task_dummy",
]


def syncall(
    *,
    core_type: str = "mix",
    mode: str = "hard",
    gm_workspace: Tensor | None = None,
    used_cores: int = 0,
    scratch: Tile | None = None,
    span: Span | None = None,
) -> Call:
    """Cross-core all-participant barrier (``pto::SYNCALL``).

    Two modes:

    - ``mode="hard"`` (default): FFTS barrier with no operands. Requires the
      kernel to fill **all** physical cores of ``core_type`` (a partial launch
      deadlocks). See :func:`pypto.ir.op.system_ops.syncall`.
    - ``mode="soft"``: GM-polling barrier that works at partial occupancy.
      Each participant bumps a per-core counter in a shared GM workspace and
      polls until all ``used_cores`` participants arrive.

    Soft-mode arguments:

    Args:
        core_type: Participant set, one of "aiv_only", "aic_only", or "mix".
        mode: "hard" or "soft".
        gm_workspace: Soft mode only. A shared, zero-initialized GM ``INT32``
            tensor with at least ``used_cores * 8`` elements, visible to every
            participating block (pass it as a kernel parameter so all SPMD
            blocks share one buffer). The compiler synthesizes the local
            UB/L1 staging tile automatically.
        used_cores: Soft mode only. Number of participating cores (a positive
            compile-time int).
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for system.syncall.
    """
    if mode == "hard":
        return _ir_ops.syncall(core_type=core_type, span=span)
    if mode != "soft":
        raise ValueError(f"syncall mode must be 'hard' or 'soft', got {mode!r}")
    # Soft form currently supports only aiv_only. aic_only needs a flat L1 (cbuf)
    # staging buffer (a fractal-laid-out Mat tile mis-places the counter slots),
    # and mix needs both UB + L1 scratch across a mixed kernel — both follow-ups.
    if core_type != "aiv_only":
        raise ValueError(f"soft syncall currently supports only core_type='aiv_only', got {core_type!r}")
    if gm_workspace is None:
        raise ValueError("soft syncall requires gm_workspace (a shared, zero-initialized GM INT32 tensor)")
    if not isinstance(used_cores, int) or used_cores <= 0:
        raise ValueError(f"soft syncall requires a positive compile-time used_cores, got {used_cores!r}")

    actual_span = _get_span_or_capture(span, frame_offset=1)
    if scratch is None:
        # User path: synthesize the local UB staging tile. (On reparse the printer
        # threads the existing scratch back via scratch=, so we reuse it instead.)
        from . import tile_ops as _tile  # noqa: PLC0415  # deferred: tile_ops imports system_ops (cycle)

        slots = used_cores * _SYNCALL_SOFT_SLOT_INT32
        scratch = _tile.create([1, slots], DataType.INT32, target_memory=MemorySpace.Vec)
    args = [gm_workspace.unwrap(), scratch.unwrap(), ConstInt(used_cores, DataType.INT32, actual_span)]
    return _ir_ops.syncall_soft(core_type, args, span=actual_span)


def tpush_to_aiv(tile: Tile, *, split: int, id: int | None = None, span: Span | None = None) -> Call:
    """Push tile data from AIC to AIV via cross-core pipe."""
    return _ir_ops.tpush_to_aiv(tile.unwrap(), split=split, id=id, span=span)


def tpush_to_aic(tile: Tile, *, split: int, id: int | None = None, span: Span | None = None) -> Call:
    """Push tile data from AIV to AIC via cross-core pipe."""
    return _ir_ops.tpush_to_aic(tile.unwrap(), split=split, id=id, span=span)


def tfree_to_aic(
    tile: Tile, span: Span | None = None, *, split: int | None = None, id: int | None = None
) -> Call:
    """Release ring buffer slot back to AIC producer."""
    return _ir_ops.tfree_to_aic(tile.unwrap(), split=split, id=id, span=span)


def tfree_to_aiv(
    tile: Tile, span: Span | None = None, *, split: int | None = None, id: int | None = None
) -> Call:
    """Release ring buffer slot back to AIV producer."""
    return _ir_ops.tfree_to_aiv(tile.unwrap(), split=split, id=id, span=span)


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


def task_invalid(*, span: Span | None = None) -> Scalar:
    """Sentinel ``pl.Scalar[pl.TASK_ID]`` for the "no producer" TaskId.

    DSL surface of the IR-level ``system.task_invalid`` op. The printer emits
    ``pl.system.task_invalid()`` for auto-scope TaskId placeholders so dumps
    are valid Python; user code in ``pl.manual_scope`` normally writes ``None``
    and the parser lowers it to this op.

    Args:
        span: Optional source span.

    Returns:
        ``pl.Scalar[pl.TASK_ID]`` wrapping the ``system.task_invalid`` IR call.
    """
    call = _ir_ops.task_invalid(span=span)
    return Scalar(DataType.TASK_ID, call)


def task_dummy(*, deps: Sequence[Scalar | Array | None]) -> Scalar:
    """Dependency-only dummy TaskId barrier.

    This is a parser construct: ``pl.system.task_dummy(deps=[...])`` is
    intercepted syntactically and lowered to ``system.task_dummy`` with manual
    dep edges. The body exists so the public DSL name resolves for static
    checkers and imports.
    """
    raise RuntimeError(
        "pl.system.task_dummy is a DSL parser construct and cannot be called directly; "
        "use it as `barrier = pl.system.task_dummy(deps=[task_id])` inside a @pl.function body."
    )
