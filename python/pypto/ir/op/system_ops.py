# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System operations for PyPTO IR.

System operations handle hardware synchronization and cross-core communication:
- sync_src / sync_dst: Set/Wait flag-based synchronization between pipes
- set_ffts / sync_set / sync_wait: Explicit Cube/Vector cross-core event synchronization
- bar_v / bar_m / bar_all: Barrier synchronization for vector, matrix, or all units
- tpush_to_aiv / tpush_to_aic: Push tile data across cores
- tpop_from_aic / tpop_from_aiv: Pop tile data from cross-core pipe
- aic_initialize_pipe / aiv_initialize_pipe: Initialize cross-core pipes
- reserve_buffer / import_peer_buffer: Cross-core buffer management (i32 SSA results)
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, PipeType, ScalarType, Span, TensorType

from ..utils import _get_span_or_capture, _to_make_tuple
from .tile_ops import (  # noqa: F401
    tpop_from_aic,
    tpop_from_aiv,
    tpush_to_aic,
    tpush_to_aiv,
)


@runtime_checkable
class _UnwrapsToExpr(Protocol):
    """Language wrappers (e.g. ``pl.Scalar[dtype]``) that expose ``unwrap() -> Expr``."""

    def unwrap(self) -> Expr: ...


def _create_sync_op(
    op_name: str,
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int,
    span: Span | None,
) -> Call:
    """Create a flag-based synchronization operation.

    Args:
        op_name: Operation name (e.g., "system.sync_src")
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that waits on the flag
        event_id: Event identifier
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    kwargs = {"set_pipe": set_pipe, "wait_pipe": wait_pipe, "event_id": event_id}
    return _ir_core.create_op_call(op_name, [], kwargs, actual_span)


def _create_barrier_op(op_name: str, *, span: Span | None) -> Call:
    """Create a barrier synchronization operation.

    Args:
        op_name: Operation name (e.g., "system.bar_v")
        span: Optional source span for debugging
    """
    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _ir_core.create_op_call(op_name, [], {}, actual_span)


def sync_src(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int,
    span: Span | None = None,
) -> Call:
    """Send a synchronization signal (Set Flag).

    Args:
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that will wait on the flag
        event_id: Event identifier
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.sync_src
    """
    return _create_sync_op(
        "system.sync_src", set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=event_id, span=span
    )


def sync_dst(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: int,
    span: Span | None = None,
) -> Call:
    """Wait for a synchronization signal (Wait Flag).

    Args:
        set_pipe: Pipe that sets the flag
        wait_pipe: Pipe that waits on the flag
        event_id: Event identifier
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.sync_dst
    """
    return _create_sync_op(
        "system.sync_dst", set_pipe=set_pipe, wait_pipe=wait_pipe, event_id=event_id, span=span
    )


_MIN_FFTS_WORKSPACE_ELEMENTS = 256


def set_ffts(workspace: Expr, *, span: Span | None = None) -> Call:
    """Declare the A3 FFTS setup operand for explicit cross-core synchronization."""
    workspace_type = workspace.type
    if not isinstance(workspace_type, TensorType):
        raise TypeError(f"system.set_ffts workspace must be a Tensor, got {workspace_type}")
    if workspace_type.dtype != DataType.INT64:
        raise TypeError(f"system.set_ffts workspace must have INT64 dtype, got {workspace_type.dtype}")
    if len(workspace_type.shape) != 1:
        raise ValueError(f"system.set_ffts workspace must be 1-D, got rank {len(workspace_type.shape)}")
    workspace_size = workspace_type.shape[0]
    if not isinstance(workspace_size, ConstInt) or workspace_size.value < _MIN_FFTS_WORKSPACE_ELEMENTS:
        raise ValueError(
            "system.set_ffts workspace must have a static length of at least "
            f"{_MIN_FFTS_WORKSPACE_ELEMENTS} INT64 elements"
        )

    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _ir_core.create_op_call("system.set_ffts", [workspace], {}, actual_span)


def _create_cross_core_sync_op(
    op_name: str,
    event_id: int | Expr,
    *,
    pipe: PipeType,
    ffts_mode: int | None,
    core_type: str | None,
    span: Span | None,
) -> Call:
    """Create a PTO cross-core sync set/wait operation."""
    args: list[Expr] = []
    kwargs: dict[str, Any] = {"pipe": pipe}
    if isinstance(event_id, int) and not isinstance(event_id, bool):
        kwargs["event_id"] = event_id
    elif isinstance(event_id, Expr):
        args.append(event_id)
    else:
        raise TypeError(f"{op_name} event_id must be int or Expr, got {type(event_id).__name__}")

    if ffts_mode is not None:
        kwargs["ffts_mode"] = ffts_mode
    if core_type is not None:
        if core_type not in ("aic", "aiv"):
            raise ValueError(f"{op_name} core_type must be 'aic' or 'aiv', got {core_type!r}")
        kwargs["core_type"] = core_type

    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _ir_core.create_op_call(op_name, args, kwargs, actual_span)


def sync_set(
    event_id: int | Expr,
    *,
    pipe: PipeType,
    ffts_mode: int | None = None,
    core_type: str | None = None,
    span: Span | None = None,
) -> Call:
    """Set an explicit Cube/Vector cross-core synchronization event.

    ``core_type`` targets the operation to one lane when expanding a mixed
    InCore kernel. It may be omitted in an explicitly typed AIC/AIV function.
    """
    return _create_cross_core_sync_op(
        "system.sync_set", event_id, pipe=pipe, ffts_mode=ffts_mode, core_type=core_type, span=span
    )


def sync_wait(
    event_id: int | Expr,
    *,
    pipe: PipeType,
    core_type: str | None = None,
    span: Span | None = None,
) -> Call:
    """Wait for an explicit Cube/Vector cross-core synchronization event.

    ``core_type`` targets the operation to one lane when expanding a mixed
    InCore kernel. It may be omitted in an explicitly typed AIC/AIV function.
    """
    return _create_cross_core_sync_op(
        "system.sync_wait", event_id, pipe=pipe, ffts_mode=None, core_type=core_type, span=span
    )


def bar_v(*, span: Span | None = None) -> Call:
    """Vector unit barrier."""
    return _create_barrier_op("system.bar_v", span=span)


def bar_m(*, span: Span | None = None) -> Call:
    """Matrix unit barrier."""
    return _create_barrier_op("system.bar_m", span=span)


def bar_all(*, span: Span | None = None) -> Call:
    """Global barrier synchronization."""
    return _create_barrier_op("system.bar_all", span=span)


def fence(*, span: Span | None = None) -> Call:
    """Memory barrier over global memory.

    Lowers to ``pto.fence.barrier_all #pto.fence_scope<gm>``.
    """
    return _create_barrier_op("system.fence", span=span)


def cacheinvalid(
    tensor: Expr,
    shapes: Sequence[int | Expr],
    offsets: Sequence[int | Expr],
    *,
    span: Span | None = None,
) -> Call:
    """Invalidate the cache lines backing a tensor sub-region.

    The op carries an N-D ``shapes`` and ``offsets`` (both matching the tensor
    rank). Codegen picks the lowering by the region size:

    - ``shapes`` all 1 (scalar write): flatten ``offsets`` and lower to
      ``pto.addptr`` + ``pto.cmo.cacheinvalid %write_ptr single_cache_line``.
    - otherwise (tile store): lower to ``pto.partition_view`` +
      ``pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>``.

    Args:
        tensor: Target tensor whose sub-region is invalidated
        shapes: Per-dimension region sizes; length must equal the tensor rank
            (all 1 selects the scalar-write / ptr form)
        offsets: Per-dimension start offsets; length must equal the tensor rank
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.cacheinvalid
    """
    actual_span = _get_span_or_capture(span)

    tensor_type = tensor.type
    if not isinstance(tensor_type, TensorType):
        raise TypeError(f"system.cacheinvalid tensor must have TensorType, got {tensor_type}")
    rank = len(tensor_type.shape)

    shapes = list(shapes)
    if len(shapes) != rank:
        raise ValueError(f"system.cacheinvalid shapes must match tensor rank {rank}, got {len(shapes)}")
    offsets = list(offsets)
    if len(offsets) != rank:
        raise ValueError(f"system.cacheinvalid offsets must match tensor rank {rank}, got {len(offsets)}")

    shapes_tuple = _to_make_tuple(shapes, actual_span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    for name, elems in (("shapes", shapes_tuple.elements), ("offsets", offsets_tuple.elements)):
        for elem in elems:
            elem_type = elem.type
            if isinstance(elem_type, ScalarType) and elem_type.dtype.is_float():
                raise TypeError(f"system.cacheinvalid {name} must be integers, got dtype {elem_type.dtype}")
    return _ir_core.create_op_call(
        "system.cacheinvalid", [tensor, shapes_tuple, offsets_tuple], {}, actual_span
    )


_SYNCALL_CORE_TYPES = ("aiv_only", "aic_only", "mix")


def syncall(*, core_type: str = "mix", span: Span | None = None) -> Call:
    """Cross-core all-participant barrier (``pto::SYNCALL``, hard/FFTS form).

    Every core in the participant set selected by ``core_type`` must execute
    past this point before any participant may proceed. Lowers to
    ``pto.syncall() mode = #pto.sync_all_mode<hard>``.

    .. warning::
        The hard/FFTS form waits for **all** physical cores of the participant
        set to arrive. The kernel must therefore be launched at full occupancy
        (one block per physical core of that type). A partial-occupancy launch
        leaves some cores unreached, so the barrier never completes and the
        AICore times out (error 507018). The compiler enforces this at compile
        time (``HardSyncallOccupancy`` verifier, issue #1935): a hard-mode
        ``syncall`` whose enclosing ``pl.spmd`` does not fill all physical cores
        of ``core_type`` is rejected. Use a full-core SPMD dispatch, or the soft
        form (``mode="soft"``) for partial occupancy.

    Args:
        core_type: Participant set, one of "aiv_only", "aic_only", or "mix".
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.syncall
    """
    if core_type not in _SYNCALL_CORE_TYPES:
        raise ValueError(f"syncall core_type must be one of {_SYNCALL_CORE_TYPES}, got {core_type!r}")
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.syncall", [], {"core_type": core_type}, actual_span)


def syncall_soft(core_type: str, args: list[Expr], *, span: Span | None = None) -> Call:
    """Soft (GM-polling) form of ``system.syncall``.

    Unlike the hard/FFTS form, the soft form polls a shared GM workspace and so
    works at partial occupancy. ``args`` is the positional operand list, already
    assembled by the DSL layer:

    - aiv_only: ``[gm_workspace, ub_scratch, used_cores]``
    - aic_only: ``[gm_workspace, l1_scratch, used_cores]``
    - mix: ``[gm_workspace, ub_scratch, l1_scratch, used_cores]``

    Args:
        core_type: Participant set, one of "aiv_only", "aic_only", or "mix".
        args: Positional operand Exprs (see above).
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for the soft-mode system.syncall.
    """
    if core_type not in _SYNCALL_CORE_TYPES:
        raise ValueError(f"soft syncall core_type must be one of {_SYNCALL_CORE_TYPES}, got {core_type!r}")
    # aiv_only/aic_only carry one scratch tile (3 operands); mix carries both a UB
    # and a flat L1 scratch (4 operands). Gate the arity here so direct IR callers
    # cannot build a malformed barrier.
    expected = 4 if core_type == "mix" else 3
    if len(args) != expected:
        raise ValueError(
            f"soft syncall core_type={core_type!r} requires {expected} operands, got {len(args)}"
        )
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.syncall", args, {"core_type": core_type, "mode": "soft"}, actual_span
    )


# Sentinel value: compiler auto-assigns the buffer base address
AUTO: int = -1

PipeBufOperand = Expr | int | float | _UnwrapsToExpr


def _consumer_buf_operand(buf: PipeBufOperand, span: Span) -> Expr:
    """Build positional operand for pipe init: Expr passthrough; int (incl. 0 / ``AUTO``) -> ConstInt."""
    if isinstance(buf, Expr):
        return buf
    if isinstance(buf, _UnwrapsToExpr):
        return buf.unwrap()
    if isinstance(buf, float):
        return ConstInt(int(buf), DataType.INT32, span)
    return ConstInt(buf, DataType.INT32, span)


def _build_pipe_init_args(
    c2v_consumer_buf: PipeBufOperand,
    v2c_consumer_buf: PipeBufOperand,
    span: Span,
) -> list[Expr]:
    """Positional args (c2v_consumer_buf, v2c_consumer_buf) for aic/aiv_initialize_pipe."""
    return [
        _consumer_buf_operand(c2v_consumer_buf, span),
        _consumer_buf_operand(v2c_consumer_buf, span),
    ]


def _build_pipe_init_kwargs(
    dir_mask: int,
    slot_size: int,
    slot_num: int | None,
    local_slot_num: int | None,
    id: int | None,
) -> dict[str, int]:
    """Build the attribute kwargs shared by aic/aiv_initialize_pipe.

    Value constraints (slot_num > 0, local_slot_num > 0, local_slot_num <=
    slot_num) are enforced downstream by the IR verifier and PTOAS, matching how
    dir_mask / slot_size are handled, so they are not re-checked here.
    """
    kwargs: dict[str, int] = {"dir_mask": dir_mask, "slot_size": slot_size}
    if slot_num is not None:
        kwargs["slot_num"] = slot_num
    if local_slot_num is not None:
        kwargs["local_slot_num"] = local_slot_num
    if id is not None:
        kwargs["id"] = id
    return kwargs


def aic_initialize_pipe(
    c2v_consumer_buf: PipeBufOperand = 0,
    v2c_consumer_buf: PipeBufOperand = 0,
    *,
    dir_mask: int,
    slot_size: int,
    slot_num: int | None = None,
    local_slot_num: int | None = None,
    id: int | None = None,
    span: Span | None = None,
) -> Call:
    """Initialize cross-core pipe on AIC side.

    Args:
        c2v_consumer_buf: C2V consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
        v2c_consumer_buf: V2C consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
        dir_mask: Direction mask for pipe
        slot_size: Size of each pipe slot
        slot_num: Optional ring-buffer slot count. Omit to let PTOAS pick its
            default (8 unidirectional, 4 per direction bidirectional).
        local_slot_num: Optional local slot count (a2/a3 only, must be
            ``<= slot_num``). On a3 the reserved/imported buffer is sized
            ``slot_size * local_slot_num``; on a5 it is ``slot_size * slot_num``.
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = _build_pipe_init_kwargs(dir_mask, slot_size, slot_num, local_slot_num, id)
    args = _build_pipe_init_args(c2v_consumer_buf, v2c_consumer_buf, actual_span)
    return _ir_core.create_op_call("system.aic_initialize_pipe", args, kwargs, actual_span)


def aiv_initialize_pipe(
    c2v_consumer_buf: PipeBufOperand = 0,
    v2c_consumer_buf: PipeBufOperand = 0,
    *,
    dir_mask: int,
    slot_size: int,
    slot_num: int | None = None,
    local_slot_num: int | None = None,
    id: int | None = None,
    span: Span | None = None,
) -> Call:
    """Initialize cross-core pipe on AIV side.

    Args:
        c2v_consumer_buf: C2V consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
        v2c_consumer_buf: V2C consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
        dir_mask: Direction mask for pipe
        slot_size: Size of each pipe slot
        slot_num: Optional ring-buffer slot count. Omit to let PTOAS pick its
            default (8 unidirectional, 4 per direction bidirectional).
        local_slot_num: Optional local slot count (a2/a3 only, must be
            ``<= slot_num``). On a3 the reserved/imported buffer is sized
            ``slot_size * local_slot_num``; on a5 it is ``slot_size * slot_num``.
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = _build_pipe_init_kwargs(dir_mask, slot_size, slot_num, local_slot_num, id)
    args = _build_pipe_init_args(c2v_consumer_buf, v2c_consumer_buf, actual_span)
    return _ir_core.create_op_call("system.aiv_initialize_pipe", args, kwargs, actual_span)


def reserve_buffer(*, name: str, size: int, base: int = AUTO, span: Span | None = None) -> Call:
    """Reserve a named buffer for cross-core communication.

    Result type is ``ScalarType(INT32)`` (PTO ``pto.reserve_buffer ... -> i32``).

    Args:
        name: Buffer name
        size: Buffer size in bytes
        base: Base address in local SRAM. Use AUTO (-1) to let the compiler
              pick a non-conflicting address, or an explicit integer for
              manual kernels.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.reserve_buffer", [], {"name": name, "size": size, "base": base}, actual_span
    )


def import_peer_buffer(*, name: str, peer_func: str, span: Span | None = None) -> Call:
    """Import a buffer from a peer function in the same group.

    Result type is ``ScalarType(INT32)`` (PTO ``pto.import_reserved_buffer ... -> i32``).

    Args:
        name: Buffer name to import
        peer_func: Name of the peer function that owns the buffer
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.import_peer_buffer", [], {"name": name, "peer_func": peer_func}, actual_span
    )


# ============================================================================
# Slot release operations (split consumer protocol)
# ============================================================================


def tfree_to_aic(
    tile: Expr, span: Span | None = None, *, split: int | None = None, id: int | None = None
) -> Call:
    """Release ring buffer slot back to AIC producer.

    Called by AIV consumer after finishing with data from tpop_from_aic.

    Args:
        tile: Tile expression obtained from tpop_from_aic to release
        split: Split mode, copied from the originating tpop by StampTfreeSplit.
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = {}
    if split is not None:
        kwargs["split"] = split
    if id is not None:
        kwargs["id"] = id
    return _ir_core.create_op_call("system.tfree_to_aic", [tile], kwargs, actual_span)


def tfree_to_aiv(
    tile: Expr, span: Span | None = None, *, split: int | None = None, id: int | None = None
) -> Call:
    """Release ring buffer slot back to AIV producer.

    Called by AIC consumer after finishing with data from tpop_from_aiv.

    Args:
        tile: Tile expression obtained from tpop_from_aiv to release
        split: Split mode, copied from the originating tpop by StampTfreeSplit.
        id: Optional frontend pipe id. Omit to use PTOAS default id 0.
        span: Optional source span
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    kwargs = {}
    if split is not None:
        kwargs["split"] = split
    if id is not None:
        kwargs["id"] = id
    return _ir_core.create_op_call("system.tfree_to_aiv", [tile], kwargs, actual_span)


# ============================================================================
# Manual-scope TaskId primitives
# ============================================================================


def task_invalid(*, span: Span | None = None) -> Call:
    """Construct an invalid ``PTO2TaskId`` sentinel.

    Returns a ``Call`` of result type ``Scalar[TASK_ID]`` that codegen lowers
    to ``PTO2TaskId::invalid()`` — the "no producer" sentinel that downstream
    ``set_dependencies`` calls skip via an ``is_valid()`` guard. Surfaced in
    the DSL as the Python literal ``None`` in TaskId-typed positions.

    Args:
        span: Optional source span (auto-captured if not provided).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.task_invalid", [], {}, actual_span)
