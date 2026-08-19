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
from typing import overload

from pypto.ir.op import system_ops as _ir_ops
from pypto.ir.op.system_ops import (
    AUTO,
    aic_initialize_pipe,
    aiv_initialize_pipe,
    bar_all,
    bar_m,
    bar_v,
    fence,
    sync_dst,
    sync_src,
)
from pypto.ir.utils import _get_span_or_capture
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Call, ConstInt, Expr, PipeType, Span

from ..typing import Array, IntLike, Scalar, Tensor, Tile

__all__ = [
    "AUTO",
    "sync_src",
    "sync_dst",
    "sync_set",
    "sync_wait",
    "set_ffts",
    "bar_v",
    "bar_m",
    "bar_all",
    "fence",
    "cacheinvalid",
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
    "available_cluster_count",
    "available_aiv_count",
]


def sync_set(
    event_id: IntLike,
    *,
    pipe: PipeType,
    ffts_mode: int | None = None,
    core_type: str | None = None,
    span: Span | None = None,
) -> Call:
    """Set a Cube/Vector cross-core event using a static or dynamic event id.

    Args:
        event_id: Event to signal. An int in the user-available range 0-13, or a
            dynamic ``pl.Scalar[pl.INDEX]``. IDs 14 and 15 are reserved.
        pipe: Pipe the event is raised on. The matching [`sync_wait`][pypto.language.system.sync_wait] must
            name the same pipe -- pairing event ids and pipes is the author's responsibility.
        ffts_mode: Optional FFTS mode, 0, 1 or 2. Accepted by ``sync_set`` only.
        core_type: ``"aic"`` or ``"aiv"``, to keep the event on the intended lane
            when a mixed InCore kernel is expanded. Omit in an explicitly typed kernel.
        span: Optional source span
    """
    event_expr = event_id.unwrap() if isinstance(event_id, Scalar) else event_id
    return _ir_ops.sync_set(event_expr, pipe=pipe, ffts_mode=ffts_mode, core_type=core_type, span=span)


def sync_wait(
    event_id: IntLike,
    *,
    pipe: PipeType,
    core_type: str | None = None,
    span: Span | None = None,
) -> Call:
    """Wait for a Cube/Vector cross-core event using a static or dynamic event id.

    Args:
        event_id: Event to wait on -- the one a matching [`sync_set`][pypto.language.system.sync_set] raises.
        pipe: Pipe the event is awaited on; must match the ``sync_set``.
        core_type: ``"aic"`` or ``"aiv"``, to keep the wait on the intended lane
            when a mixed InCore kernel is expanded. Omit in an explicitly typed kernel.
        span: Optional source span
    """
    event_expr = event_id.unwrap() if isinstance(event_id, Scalar) else event_id
    return _ir_ops.sync_wait(event_expr, pipe=pipe, core_type=core_type, span=span)


def set_ffts(workspace: Tensor, *, span: Span | None = None) -> Call:
    """Declare the A3 FFTS setup operand for explicit cross-core synchronization."""
    if not isinstance(workspace, Tensor):
        raise TypeError(f"set_ffts workspace must be a Tensor, got {type(workspace).__name__}")
    return _ir_ops.set_ffts(workspace.unwrap(), span=span)


_SYNCALL_SOFT_CORE_TYPES = ("aiv_only", "aic_only", "mix")
_SYNCALL_MAX_USED_CORES = (1 << 31) - 1


def syncall(
    *,
    core_type: str = "mix",
    mode: str = "hard",
    gm_workspace: Tensor | None = None,
    used_cores: IntLike | None = None,
    span: Span | None = None,
) -> Call:
    """Cross-core all-participant barrier (``pto::SYNCALL``).

    Two modes:

    - ``mode="hard"`` (default): FFTS barrier with no operands. Requires the
      enclosing ``pl.spmd`` launch to fill **all** physical cores of
      ``core_type`` (a partial launch deadlocks on device — error 507018). The
      compiler rejects a partial-occupancy hard launch at compile time
      (``HardSyncallOccupancy`` verifier, issue #1935). See
      ``pypto.ir.op.system_ops.syncall``.
    - ``mode="soft"``: GM-polling barrier that works at partial occupancy.
      Each participant updates a shared counter in an exclusive 64-byte GM
      cache line and polls until all participants arrive. Supported for every
      ``core_type`` ("aiv_only", "aic_only", "mix").

    Both modes synchronize arrival only. They do not wait for preceding data
    instructions or publish/invalidate business-data cache lines. For a
    cross-core GM handoff that may span multiple cache lines, conservatively
    call whole-GM ``pl.system.cacheinvalid()`` and ``pl.system.fence()`` before
    the barrier, then call ``pl.system.cacheinvalid()`` again on the consumer
    before it reads. The tensor-region overload covers only the cache line
    containing the view's base address.

    Soft-mode arguments:

    Args:
        core_type: Participant set, one of "aiv_only", "aic_only", or "mix".
            For "mix", ``used_cores`` is the *total* AIC + AIV participant count.
        mode: "hard" or "soft".
        gm_workspace: Soft mode only. A shared, zero-initialized GM ``INT32``
            tensor with at least 16 elements (64 bytes), visible to every
            participating block. Pass it as a kernel parameter so all SPMD
            blocks share one buffer. The buffer must occupy an exclusive cache
            line and be zero-initialized before its first use.
        used_cores: Soft mode only. Required participant count as a Python int
            in the INT32 range or an ``INT32`` scalar. Pass 0 explicitly to ask
            PTO-ISA to derive the count from the device launch configuration.
            That opt-in is unsafe when the runtime's logical grid differs from
            the device launch registers, including the currently pinned Simpler
            runtime.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for system.syncall.
    """
    if mode == "hard":
        # Reject soft-only kwargs so a typo like syncall(gm_workspace=ws) does not
        # silently fall back to the full-occupancy hard barrier (the deadlock path
        # the soft form exists to avoid).
        if gm_workspace is not None or used_cores is not None:
            raise ValueError(
                "syncall(mode='hard') takes no gm_workspace/used_cores; "
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
    if not isinstance(gm_workspace, Tensor):
        raise TypeError(f"soft syncall gm_workspace must be a Tensor, got {type(gm_workspace).__name__}")
    if used_cores is None:
        raise ValueError(
            "soft syncall requires explicit used_cores; pass 0 only when the device launch "
            "configuration matches the runtime's logical grid"
        )
    actual_span = _get_span_or_capture(span, frame_offset=1)
    used_expr: Expr | None
    if isinstance(used_cores, int):
        if not 0 <= used_cores <= _SYNCALL_MAX_USED_CORES:
            raise ValueError(
                "soft syncall used_cores must be in the INT32 range "
                f"[0, {_SYNCALL_MAX_USED_CORES}], got {used_cores!r}"
            )
        used_expr = ConstInt(used_cores, DataType.INT32, actual_span) if used_cores else None
    elif isinstance(used_cores, Scalar):
        used_expr = used_cores.unwrap()
    elif isinstance(used_cores, Expr):
        used_expr = used_cores
    else:
        raise TypeError(
            "soft syncall used_cores must be a non-negative Python int or an INT32 scalar, "
            f"got {type(used_cores).__name__}"
        )
    return _ir_ops.syncall_soft(core_type, gm_workspace.unwrap(), used_expr, span=actual_span)


@overload
def cacheinvalid(*, span: Span | None = None) -> Call: ...
@overload
def cacheinvalid(
    tensor: Tensor,
    shapes: Sequence[int | Scalar],
    offsets: Sequence[int | Scalar],
    *,
    span: Span | None = None,
) -> Call: ...
def cacheinvalid(
    tensor: Tensor | None = None,
    shapes: Sequence[int | Scalar] | None = None,
    offsets: Sequence[int | Scalar] | None = None,
    *,
    span: Span | None = None,
) -> Call:
    """Invalidate one addressed cache line, or the whole GM address space.

    Two forms selected by arity:

    - No arguments: invalidate the entire GM address space; lowers to
      ``pto.cmo.cacheinvalid all #pto.address_space<gm>``.
    - ``(tensor, shapes, offsets)``: locate a tensor sub-region and invalidate
      only the cache line containing that view's base address; lowers to
      ``pto.partition_view`` +
      ``pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>``
      for every region size, a single element included. ``shapes`` does not
      make the operation walk every cache line in the region.

    Args:
        tensor: Target tensor whose view base addresses the cache line; omit for whole-GM.
        shapes: Per-dimension region sizes; length must equal the tensor rank.
        offsets: Per-dimension start offsets; length must equal the tensor rank.
        span: Optional source span for debugging (auto-captured if not provided).
    """
    if tensor is None:
        if shapes is not None or offsets is not None:
            raise ValueError(
                "system.cacheinvalid whole-GM form takes no shapes/offsets; "
                "pass (tensor, shapes, offsets) for the region form"
            )
        return _ir_ops.cacheinvalid(span=span)
    if shapes is None or offsets is None:
        raise ValueError(
            "system.cacheinvalid region form requires both shapes and offsets "
            "(or pass no arguments for the whole-GM form)"
        )
    shp = [s.unwrap() if isinstance(s, Scalar) else s for s in shapes]
    off = [o.unwrap() if isinstance(o, Scalar) else o for o in offsets]
    return _ir_ops.cacheinvalid(tensor.unwrap(), shp, off, span=span)


def tpush_to_aiv(
    tile: Tile,
    *,
    split: int,
    lane_stride: int | None = None,
    id: int | None = None,
    span: Span | None = None,
) -> Call:
    """Push tile data from AIC to AIV via cross-core pipe.

    The Vector side receives it with [`tpop_from_aic`][pypto.language.system.tpop_from_aic] and releases the
    slot with [`tfree_to_aic`][pypto.language.system.tfree_to_aic]; ``split`` and ``id`` must match across all
    three.

    Args:
        tile: Tile to send. Its Cube-side buffer stays live until the consumer frees the slot.
        split: pto-isa split code (0=none, 1/2=up-down/left-right, 3/4=the same
            axes over an odd extent). Selects the axis along which the two AIV
            lanes divide the tile, and how their extents relate; 0 sends it whole.
        lane_stride: Partition stride carried when a ragged boundary was balanced
            across the two AIV lanes; omit for the default box partition.
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    return _ir_ops.tpush_to_aiv(tile.unwrap(), split=split, lane_stride=lane_stride, id=id, span=span)


def tpush_to_aic(tile: Tile, *, split: int, id: int | None = None, span: Span | None = None) -> Call:
    """Push tile data from AIV to AIC via cross-core pipe.

    The Cube side receives it with [`tpop_from_aiv`][pypto.language.system.tpop_from_aiv] and releases the
    slot with [`tfree_to_aiv`][pypto.language.system.tfree_to_aiv]; ``split`` and ``id`` must match across all
    three.

    Args:
        tile: Tile to send. Its Vector-side buffer stays live until the consumer frees the slot.
        split: Split mode (0=none, 1=up-down, 2=left-right). Selects the axis along
            which the two AIV lanes divide the tile; 0 sends it whole.
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    return _ir_ops.tpush_to_aic(tile.unwrap(), split=split, id=id, span=span)


def tfree_to_aic(
    tile: Tile, span: Span | None = None, *, split: int | None = None, id: int | None = None
) -> Call:
    """Release ring buffer slot back to AIC producer.

    Call this once the tile from [`tpop_from_aic`][pypto.language.system.tpop_from_aic] has been consumed.
    Until it runs, the slot stays occupied and the producer blocks once the ring fills.

    Args:
        tile: The tile returned by the matching [`tpop_from_aic`][pypto.language.system.tpop_from_aic].
        span: Optional source span
        split: Leave ``None``. The ``StampTfreeSplit`` pass always takes this from
            the originating ``tpop``, so a value passed here cannot override it.
        id: Optional frontend pipe id, inherited from the originating ``tpop`` when
            omitted. Supplying one that disagrees with the ``tpop`` is rejected.
    """
    return _ir_ops.tfree_to_aic(tile.unwrap(), split=split, id=id, span=span)


def tfree_to_aiv(
    tile: Tile, span: Span | None = None, *, split: int | None = None, id: int | None = None
) -> Call:
    """Release ring buffer slot back to AIV producer.

    Call this once the tile from [`tpop_from_aiv`][pypto.language.system.tpop_from_aiv] has been consumed.
    Until it runs, the slot stays occupied and the producer blocks once the ring fills.

    Args:
        tile: The tile returned by the matching [`tpop_from_aiv`][pypto.language.system.tpop_from_aiv].
        span: Optional source span
        split: Leave ``None``. The ``StampTfreeSplit`` pass always takes this from
            the originating ``tpop``, so a value passed here cannot override it.
        id: Optional frontend pipe id, inherited from the originating ``tpop`` when
            omitted. Supplying one that disagrees with the ``tpop`` is rejected.
    """
    return _ir_ops.tfree_to_aiv(tile.unwrap(), split=split, id=id, span=span)


def tpop_from_aic(
    *,
    shape: list[int] | None = None,
    dtype: DataType | None = None,
    split: int = 0,
    lane_stride: int | None = None,
    id: int | None = None,
    span: Span | None = None,
) -> Tile:
    """Pop tile data from AIC cross-core pipe into AIV.

    Args:
        shape: Shape of the tile to receive
        dtype: Data type of the tile to receive
        split: pto-isa split code (0=none, 1/2=up-down/left-right, 3/4=the same
            axes over an odd extent)
        lane_stride: Partition stride carried when a ragged boundary was
            balanced across the two AIV lanes; omit for the box partition
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    call = _ir_ops.tpop_from_aic(
        shape=shape, dtype=dtype, split=split, lane_stride=lane_stride, id=id, span=span
    )
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

    Args:
        deps: TaskIds the barrier waits on -- each a ``pl.Scalar[pl.TASK_ID]``
            returned by ``pl.submit``, or a ``pl.array`` of them for fan-in.
            ``None`` entries are skipped, so a conditionally-produced TaskId needs
            no branch at the call site.

    Returns:
        A ``pl.Scalar[pl.TASK_ID]`` that becomes ready once every dep has. Depend on
        it instead of listing all of `deps` again at each consumer.
    """
    raise RuntimeError(
        "pl.system.task_dummy is a DSL parser construct and cannot be called directly; "
        "use it as `barrier = pl.system.task_dummy(deps=[task_id])` inside a @pl.function body."
    )


def available_cluster_count(*, span: Span | None = None) -> Scalar:
    """This run's MIX cluster (= AIC) count, as reported by the runtime.

    Use it as the ``core_num`` of a mixed (AIC+AIV) or cube-only
    ``pl.spmd`` launch instead of a literal: the count belongs to the device
    the run lands on, so a baked-in literal under- or over-fills the launch on
    any device with a different usable cluster count. A hard
    ``pl.system.syncall`` needs full occupancy to complete, so this is the only
    launch width that keeps it deadlock-free across devices.

    Codegen lowers it to the orchestration helper ``rt_available_cluster_count()``.

    Args:
        span: Optional source span.

    Returns:
        ``pl.Scalar[pl.INT32]`` wrapping the ``system.available_cluster_count`` IR call.

    Example:
        >>> with pl.spmd(pl.system.available_cluster_count(), sync_start=True):
        ...     out = self.mixed_kernel(a, b, out)

    Pass the call directly as ``core_num`` as shown. Binding it to a name first
    (``n = pl.system.available_cluster_count()``) also compiles, but the launch
    width then reaches the outlined Spmd wrapper as a reference to a variable in
    the caller, which the IR printer cannot re-parse.
    """
    call = _ir_ops.available_cluster_count(span=span)
    return Scalar(DataType.INT32, call)


def available_aiv_count(*, span: Span | None = None) -> Scalar:
    """This run's standalone AIV core count, as reported by the runtime.

    The AIV counterpart of [`available_cluster_count`][pypto.language.system.available_cluster_count] — the
    ``core_num`` of a vector-only ``pl.spmd`` launch. A mixed launch sizes itself on
    [`available_cluster_count`][pypto.language.system.available_cluster_count] (one block per cluster), not on
    this.

    Codegen lowers it to the orchestration helper ``rt_available_aiv_count()``.

    Args:
        span: Optional source span.

    Returns:
        ``pl.Scalar[pl.INT32]`` wrapping the ``system.available_aiv_count`` IR call.

    Example:
        >>> with pl.spmd(pl.system.available_aiv_count(), sync_start=True):
        ...     out = self.aiv_kernel(a, b, out)
    """
    call = _ir_ops.available_aiv_count(span=span)
    return Scalar(DataType.INT32, call)
