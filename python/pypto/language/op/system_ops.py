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


_SYNCALL_SOFT_CORE_TYPES = ("aiv_only", "aic_only", "mix")


def syncall(
    *,
    core_type: str = "mix",
    mode: str = "hard",
    gm_workspace: Tensor | None = None,
    used_cores: int = 0,
    scratch: Tile | None = None,
    scratch_l1: Tile | None = None,
    span: Span | None = None,
) -> Call:
    """Cross-core all-participant barrier (``pto::SYNCALL``).

    Two modes:

    - ``mode="hard"`` (default): FFTS barrier with no operands. Requires the
      kernel to fill **all** physical cores of ``core_type`` (a partial launch
      deadlocks). See :func:`pypto.ir.op.system_ops.syncall`.
    - ``mode="soft"``: GM-polling barrier that works at partial occupancy.
      Each participant bumps a per-core counter in a shared GM workspace and
      polls until all ``used_cores`` participants arrive. Supported for every
      ``core_type`` ("aiv_only", "aic_only", "mix").

    Soft-mode arguments:

    Args:
        core_type: Participant set, one of "aiv_only", "aic_only", or "mix".
            For "mix", ``used_cores`` is the *total* AIC + AIV participant count.
        mode: "hard" or "soft".
        gm_workspace: Soft mode only. A shared, zero-initialized GM ``INT32``
            tensor with at least ``used_cores * 8`` elements, visible to every
            participating block (pass it as a kernel parameter so all SPMD
            blocks share one buffer). The compiler synthesizes the local
            UB/L1 staging tile(s) automatically.
        used_cores: Soft mode only. Number of participating cores (a positive
            compile-time int).
        scratch: Compiler-internal. The local staging tile threaded back by the
            printer on reparse (UB/Vec tile for "aiv_only" and "mix"; flat
            L1/Mat tile for "aic_only"). Leave ``None`` in user code.
        scratch_l1: Compiler-internal. The flat L1/Mat staging tile for the
            "mix" form, threaded back by the printer on reparse. Leave ``None``
            in user code.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for system.syncall.
    """
    if mode == "hard":
        # Reject soft-only kwargs so a typo like syncall(gm_workspace=ws) does not
        # silently fall back to the full-occupancy hard barrier (the deadlock path
        # the soft form exists to avoid).
        if gm_workspace is not None or scratch is not None or scratch_l1 is not None or used_cores:
            raise ValueError(
                "syncall(mode='hard') takes no gm_workspace/scratch/scratch_l1/used_cores; "
                "pass mode='soft' to use the GM-polling barrier"
            )
        return _ir_ops.syncall(core_type=core_type, span=span)
    if mode != "soft":
        raise ValueError(f"syncall mode must be 'hard' or 'soft', got {mode!r}")
    if core_type not in _SYNCALL_SOFT_CORE_TYPES:
        raise ValueError(
            f"soft syncall core_type must be one of {_SYNCALL_SOFT_CORE_TYPES}, got {core_type!r}"
        )
    if gm_workspace is None:
        raise ValueError("soft syncall requires gm_workspace (a shared, zero-initialized GM INT32 tensor)")
    if not isinstance(used_cores, int) or used_cores <= 0:
        raise ValueError(f"soft syncall requires a positive compile-time used_cores, got {used_cores!r}")

    actual_span = _get_span_or_capture(span, frame_offset=1)
    # Deferred import: tile_ops imports system_ops (cycle).
    from . import tile_ops as _tile  # noqa: PLC0415

    def _ub_scratch(existing: Tile | None) -> Tile:
        # UB (Vec) staging tile. The AIV barrier bulk-reads every participant's
        # slot into it, so it needs used_cores * 8 int32 (flat by default).
        if existing is not None:
            return existing
        return _tile.create(
            [1, used_cores * _SYNCALL_SOFT_SLOT_INT32], DataType.INT32, target_memory=MemorySpace.Vec
        )

    def _l1_scratch(existing: Tile | None) -> Tile:
        # Flat L1 (Mat/cbuf) staging tile. The cube path only stages its own
        # single counter line via create_cbuf_matrix, so 8 int32 suffice; it must
        # be flat (slayout=none_box) or the counter slot is mis-placed.
        if existing is not None:
            return existing
        return _tile.create(
            [1, _SYNCALL_SOFT_SLOT_INT32], DataType.INT32, target_memory=MemorySpace.Mat, flat_layout=True
        )

    used_const = ConstInt(used_cores, DataType.INT32, actual_span)
    if core_type == "aiv_only":
        scratch = _ub_scratch(scratch)
        args = [gm_workspace.unwrap(), scratch.unwrap(), used_const]
    elif core_type == "aic_only":
        scratch = _l1_scratch(scratch)
        args = [gm_workspace.unwrap(), scratch.unwrap(), used_const]
    else:  # mix: both a UB and a flat L1 staging tile
        scratch = _ub_scratch(scratch)
        scratch_l1 = _l1_scratch(scratch_l1)
        args = [gm_workspace.unwrap(), scratch.unwrap(), scratch_l1.unwrap(), used_const]
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
