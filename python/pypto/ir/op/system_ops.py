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
from enum import Enum, auto
from typing import Any, Protocol, TypeVar, overload, runtime_checkable

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


class KernelType(Enum):
    """Which generated kernel an op belongs to.

    A mixed InCore function is expanded into an AIC kernel and an AIV kernel.
    This says which of the two a cross-core sync op lands in; ``MIX`` means both
    take part, which only a barrier can ask for.

    Two neighbouring enums mean different things:

    - ``FunctionType.AIC`` / ``.AIV`` classify *a function*, and this classifies
      *an op inside one*. A function already declared ``FunctionType.AIV`` needs
      no ``KernelType`` on its ops -- there is only one kernel to land in.
    - ``ir.CoreType`` labels one physical core in the SoC inventory, which is
      what ``Backend::GetCoreCount`` counts. That is hardware; this is not, and
      ``MIX`` has no ``CoreType`` counterpart at all.

    Members carry no wire value: each op spells the same kernel differently in
    its IR attr (``system.syncall`` writes ``"aic_only"`` where
    ``system.sync_set`` writes ``"aic"``), so the lowering tables are explicit.
    """

    AIC = auto()
    AIV = auto()
    MIX = auto()


class SyncAllMode(Enum):
    """Barrier implementation selected by ``system.syncall``.

    - ``HARD``: FFTS barrier with no operands; requires full-core occupancy.
    - ``SOFT``: GM-polling barrier; works at partial occupancy.
    """

    HARD = "hard"
    SOFT = "soft"


# Kernel -> IR attr spelling, per op. ``system.syncall`` names a participant set
# and can rendezvous both kernels; the event ops pin one op to one kernel, so
# they have no MIX spelling (omitting the kwarg is what leaves an event in both).
_SYNCALL_CORE_TYPE: dict[KernelType, str] = {
    KernelType.AIC: "aic_only",
    KernelType.AIV: "aiv_only",
    KernelType.MIX: "mix",
}
_SYNC_EVENT_CORE_TYPE: dict[KernelType, str] = {
    KernelType.AIC: "aic",
    KernelType.AIV: "aiv",
}
# Each table's keys are also the op's domain: what it accepts as ``core_type``.
_SYNCALL_KERNELS: tuple[KernelType, ...] = tuple(_SYNCALL_CORE_TYPE)
_SYNC_EVENT_KERNELS: tuple[KernelType, ...] = tuple(_SYNC_EVENT_CORE_TYPE)

_SYNC_EVENT_MIX_HINT = ". Omit core_type to leave the event in both kernels"

_EnumT = TypeVar("_EnumT", bound=Enum)


def _check_enum(
    value: Any,
    enum_cls: type[_EnumT],
    param: str,
    op_name: str,
    *,
    allowed: tuple[_EnumT, ...] | None = None,
    hint: str = "",
) -> _EnumT:
    """Validate that an op keyword is an enum member this op accepts.

    ``allowed`` is the per-op domain: an op that takes only part of the enum
    passes just those members, and one outside them is rejected. Omit it to
    accept every member.

    Args:
        value: Value passed by the caller
        enum_cls: Enum the keyword is typed as
        param: Keyword name, for the messages
        op_name: Operation name, for the messages
        allowed: Members this op accepts; defaults to every member
        hint: Appended to the rejection message, to point at the alternative

    Returns:
        ``value``, once confirmed to be a member this op accepts

    Raises:
        ValueError: If ``value`` is a member outside this op's domain
        TypeError: If ``value`` is not an ``enum_cls`` member at all
    """
    accepted = allowed if allowed is not None else tuple(enum_cls)
    valid = ", ".join(f"{enum_cls.__name__}.{member.name}" for member in accepted)
    if not isinstance(value, enum_cls):
        raise TypeError(
            f"{op_name} {param} must be a {enum_cls.__name__} member, got {value!r}. Valid values: {valid}"
        )
    if value not in accepted:
        raise ValueError(
            f"{op_name} {param} must be one of {valid}, got {enum_cls.__name__}.{value.name}{hint}"
        )
    return value


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
    core_type: KernelType | None,
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
        _check_enum(
            core_type,
            KernelType,
            "core_type",
            op_name,
            allowed=_SYNC_EVENT_KERNELS,
            hint=_SYNC_EVENT_MIX_HINT,
        )
        kwargs["core_type"] = _SYNC_EVENT_CORE_TYPE[core_type]

    actual_span = _get_span_or_capture(span, frame_offset=2)
    return _ir_core.create_op_call(op_name, args, kwargs, actual_span)


def sync_set(
    event_id: int | Expr,
    *,
    pipe: PipeType,
    ffts_mode: int | None = None,
    core_type: KernelType | None = None,
    span: Span | None = None,
) -> Call:
    """Set an explicit Cube/Vector cross-core synchronization event.

    ``core_type`` (``KernelType.AIC`` / ``KernelType.AIV``) targets the operation
    to one kernel when expanding a mixed InCore function. Omit it to leave the
    event in both, which is what an explicitly typed AIC/AIV function wants.
    """
    return _create_cross_core_sync_op(
        "system.sync_set", event_id, pipe=pipe, ffts_mode=ffts_mode, core_type=core_type, span=span
    )


def sync_wait(
    event_id: int | Expr,
    *,
    pipe: PipeType,
    core_type: KernelType | None = None,
    span: Span | None = None,
) -> Call:
    """Wait for an explicit Cube/Vector cross-core synchronization event.

    ``core_type`` (``KernelType.AIC`` / ``KernelType.AIV``) targets the operation
    to one kernel when expanding a mixed InCore function. Omit it to leave the
    wait in both, which is what an explicitly typed AIC/AIV function wants.
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


@overload
def cacheinvalid(*, span: Span | None = None) -> Call: ...
@overload
def cacheinvalid(
    tensor: Expr,
    shapes: Sequence[int | Expr],
    offsets: Sequence[int | Expr],
    *,
    span: Span | None = None,
) -> Call: ...
def cacheinvalid(
    tensor: Expr | None = None,
    shapes: Sequence[int | Expr] | None = None,
    offsets: Sequence[int | Expr] | None = None,
    *,
    span: Span | None = None,
) -> Call:
    """Invalidate one addressed cache line, or the whole GM address space.

    Two forms selected by arity:

    - No arguments: invalidate the entire GM address space; lowers to
      ``pto.cmo.cacheinvalid all #pto.address_space<gm>``.
    - ``(tensor, shapes, offsets)``: locate a tensor sub-region and invalidate
      only the cache line containing that view's base address. Both ``shapes``
      and ``offsets`` are N-D and match the tensor rank. Every region size — a
      single element included — lowers to ``pto.partition_view`` +
      ``pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>``.
      ``shapes`` does not make the operation walk every cache line in the region.

    Args:
        tensor: Target tensor whose view base addresses the cache line; omit for whole-GM
        shapes: Per-dimension region sizes; length must equal the tensor rank
        offsets: Per-dimension start offsets; length must equal the tensor rank
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.cacheinvalid
    """
    actual_span = _get_span_or_capture(span)

    if tensor is None:
        if shapes is not None or offsets is not None:
            raise ValueError(
                "system.cacheinvalid whole-GM form takes no shapes/offsets; "
                "pass (tensor, shapes, offsets) for the region form"
            )
        return _ir_core.create_op_call("system.cacheinvalid", [], {}, actual_span)
    if shapes is None or offsets is None:
        raise ValueError(
            "system.cacheinvalid region form requires both shapes and offsets "
            "(or pass no arguments for the whole-GM form)"
        )

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


def syncall(*, core_type: KernelType = KernelType.MIX, span: Span | None = None) -> Call:
    """Cross-core all-participant barrier (``pto::SYNCALL``, hard/FFTS form).

    Every core in the participant set selected by ``core_type`` must execute
    past this point before any participant may proceed. Lowers to
    ``pto.syncall() mode = #pto.sync_all_mode<hard>``.

    This is an arrival barrier only: it neither waits for preceding data
    instructions nor publishes or invalidates business-data cache lines.
    Cross-core GM handoff requires explicit cache maintenance and a GM fence.

    .. warning::
        The hard/FFTS form waits for **all** physical cores of the participant
        set to arrive. The kernel must therefore be launched at full occupancy
        (one block per physical core of that type). A partial-occupancy launch
        leaves some cores unreached, so the barrier never completes and the
        AICore times out (error 507018). The compiler enforces this at compile
        time (``HardSyncallOccupancy`` verifier, issue #1935): a hard-mode
        ``syncall`` whose enclosing ``pl.spmd`` does not fill all physical cores
        of ``core_type`` is rejected. Use a full-core SPMD dispatch, or the soft
        form (``mode=SyncAllMode.SOFT``) for partial occupancy.

    Args:
        core_type: Participant set, a :class:`KernelType` member —
            ``MIX`` rendezvouses both kernels.
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for system.syncall
    """
    _check_enum(core_type, KernelType, "core_type", "syncall", allowed=_SYNCALL_KERNELS)
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call(
        "system.syncall", [], {"core_type": _SYNCALL_CORE_TYPE[core_type]}, actual_span
    )


def syncall_soft(
    core_type: KernelType,
    gm_workspace: Expr,
    used_cores: Expr | None = None,
    *,
    span: Span | None = None,
) -> Call:
    """Soft (GM-polling) form of ``system.syncall``.

    Unlike the hard/FFTS form, the soft form polls a shared GM workspace and so
    works at partial occupancy. All participant sets use the same operand ABI:
    ``[gm_workspace]`` when the launch determines the participant count, or
    ``[gm_workspace, used_cores]`` when it is explicit.

    This is an arrival barrier only: it neither waits for preceding data
    instructions nor publishes or invalidates business-data cache lines.
    Cross-core GM handoff requires explicit cache maintenance and a GM fence.

    Args:
        core_type: Participant set, a :class:`KernelType` member —
            ``MIX`` rendezvouses both kernels.
        gm_workspace: Shared, zero-initialized GM INT32 workspace with at least
            16 elements (64 bytes).
        used_cores: Optional INT32 participant count. Omit to derive it from the
            device launch configuration. Runtimes with a synthetic logical grid
            should pass it explicitly.
        span: Optional source span for debugging (auto-captured if not provided).

    Returns:
        Call expression for the soft-mode system.syncall.
    """
    _check_enum(core_type, KernelType, "core_type", "soft syncall", allowed=_SYNCALL_KERNELS)
    if used_cores is not None:
        used_type = used_cores.type
        if not isinstance(used_type, ScalarType) or used_type.dtype != DataType.INT32:
            raise TypeError(f"soft syncall used_cores must be an INT32 scalar, got {used_type}")
        if isinstance(used_cores, ConstInt):
            if not 0 <= used_cores.value <= (1 << 31) - 1:
                raise ValueError(
                    "soft syncall used_cores must be in the INT32 range "
                    f"[0, {(1 << 31) - 1}], got {used_cores.value}"
                )
            if used_cores.value == 0:
                used_cores = None
    actual_span = _get_span_or_capture(span, frame_offset=1)
    args = [gm_workspace]
    if used_cores is not None:
        args.append(used_cores)
    return _ir_core.create_op_call(
        "system.syncall",
        args,
        {"core_type": _SYNCALL_CORE_TYPE[core_type], "mode": SyncAllMode.SOFT.value},
        actual_span,
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
    """Construct an invalid ``TaskId`` sentinel.

    Returns a ``Call`` of result type ``Scalar[TASK_ID]`` that codegen lowers
    to ``TaskId::invalid()`` — the "no producer" sentinel that downstream
    ``set_dependencies`` calls skip via an ``is_valid()`` guard. Surfaced in
    the DSL as the Python literal ``None`` in TaskId-typed positions.

    Args:
        span: Optional source span (auto-captured if not provided).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.task_invalid", [], {}, actual_span)


# ============================================================================
# SPMD launch-shape queries
# ============================================================================


def available_cluster_count(*, span: Span | None = None) -> Call:
    """Query this run's MIX cluster (= AIC) count.

    Returns a ``Call`` of result type ``Scalar[INT32]`` that orchestration
    codegen lowers to ``rt_available_cluster_count()``. The count belongs to
    the device the run lands on, so it is the only launch width that keeps a
    mixed (AIC+AIV) or cube-only SPMD launch at full occupancy — which a hard
    ``system.syncall`` requires.

    Args:
        span: Optional source span (auto-captured if not provided).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.available_cluster_count", [], {}, actual_span)


def available_aiv_count(*, span: Span | None = None) -> Call:
    """Query this run's standalone AIV core count.

    The AIV counterpart of :func:`available_cluster_count`; orchestration
    codegen lowers it to ``rt_available_aiv_count()``. Sizes a vector-only
    SPMD launch — a mixed launch uses :func:`available_cluster_count`.

    Args:
        span: Optional source span (auto-captured if not provided).
    """
    actual_span = _get_span_or_capture(span, frame_offset=1)
    return _ir_core.create_op_call("system.available_aiv_count", [], {}, actual_span)
