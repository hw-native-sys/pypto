# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ergonomic ``pld.*`` collective wrappers — auto-managed barrier signals.

Each wrapper allocates a **fresh**, correctly shaped INT32 signal window (via
``pld.tensor.alloc_window_buffer(..., name=...)`` + ``pld.tensor.window``) and
delegates to the corresponding ``pld.tensor.*`` HOST builtin, removing the
per-call signal-buffer boilerplate. Two exceptions: ``all_reduce(mode="mesh")``
uses a compiler-synthesized signal (no explicit window), and ``barrier`` takes
a caller-provided, comm-domain-covered signal.

HOST-orchestration only: the underlying collective ops require window-bound
:class:`pld.DistributedTensor` operands and host-only allocation primitives
(``alloc_window_buffer`` / ``window`` / ``world_size``).

Signals self-clear under the credit-barrier protocol (pypto #2175, merged), so
they are safe to reuse across calls. The wrappers still allocate a **fresh**
buffer per call, which remains correct and is the simplest safe default.
"""

import itertools
from collections.abc import Sequence
from typing import Literal, TypeGuard, overload

from pypto.language.typing import IntLike, Tensor
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import ReduceOp

from ..typing.distributed_tensor import DistributedTensor
from . import tensor_ops as _tensor
from ._utils import _unwrap
from .system_ops import world_size

__all__ = [
    "all_gather",
    "all_reduce",
    "all_to_all",
    "all_to_all_v",
    "barrier",
    "broadcast",
    "reduce_scatter",
]

_SIGNAL_COUNTER = itertools.count()


def _is_static_positive_int(value: object) -> TypeGuard[int]:
    """True when ``value`` is a plain positive int (bool excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fresh_signal(op_name: str, shape: Sequence[IntLike]) -> DistributedTensor:
    """Allocate a fresh, correctly shaped INT32 signal window for a collective call.

    Args:
        op_name: Short op name used to derive a unique buffer identifier
            (``__auto_<op_name>_<n>``). The parser injects ``alloc_window_buffer``
            names only from program-body assignments; an explicit generated name
            is required here because the call lives inside helper Python.
        shape: Per-rank signal shape (e.g. ``[world_size(), 1]`` or a static
            ring shape).

    Returns:
        A window-bound :class:`pld.DistributedTensor` INT32 signal.
    """
    name = f"__auto_{op_name}_{next(_SIGNAL_COUNTER)}"
    buf = _tensor.alloc_window_buffer(shape, dtype=DataType.INT32, name=name)
    return _tensor.window(buf, shape, dtype=DataType.INT32)


@overload
def all_reduce(
    target: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: Literal["mesh"] = "mesh",
    nranks: None = None,
) -> DistributedTensor: ...


@overload
def all_reduce(
    target: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: Literal["ring"],
    nranks: int,
) -> DistributedTensor: ...


def all_reduce(
    target: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
    mode: str = "mesh",
    nranks: int | None = None,
) -> DistributedTensor:
    """In-place cross-rank allreduce, auto-managing the barrier signal.

    HOST-orchestration only; ``target`` is a window-bound
    :class:`pld.DistributedTensor` holding each rank's partial result.

    * ``mode="mesh"`` (default): no signal is allocated — the compiler
      synthesizes a private INT32 signal (``[world_size, 1]``).
    * ``mode="ring"``: a fresh signal of shape ``[2*(nranks-1)+1, nranks]`` is
      allocated; ``nranks`` (the static world size) is required because the
      ring signal shape is a compile-time constant.

    Args:
        target: Window-bound :class:`pld.DistributedTensor`, reduced in place.
        op: :class:`pld.ReduceOp` (``Sum`` / ``Max`` / ``Min`` / ``Prod``).
        mode: ``"mesh"`` (default) or ``"ring"``. ``"ring"`` currently supports
            ``ReduceOp.Sum`` + FP32 only.
        nranks: Static world size, required for ``mode="ring"`` and rejected
            otherwise (the ring signal shape is a compile-time constant).

    Returns:
        The rebound ``target`` (window-as-result).
    """
    if mode not in ("mesh", "ring"):
        raise ValueError(f'pld.all_reduce mode must be "mesh" or "ring", got {mode!r}')
    if mode == "ring":
        if not _is_static_positive_int(nranks):
            raise ValueError(
                "pld.all_reduce(mode='ring') requires a positive static int `nranks` "
                "(the ring signal shape [2*(NR-1)+1, NR] is a compile-time constant)"
            )
        if op != ReduceOp.Sum:
            raise ValueError(
                "pld.all_reduce(mode='ring') supports only ReduceOp.Sum "
                "(the HOST ring schedule implements Sum only)"
            )
        # Guarded read: a non-tensor arg (e.g. an int) falls through _unwrap, so
        # skip the dtype check and let the delegation raise its TypeError.
        target_type = getattr(_unwrap(target), "type", None)
        if isinstance(target_type, _ir.ShapedType) and target_type.dtype != DataType.FP32:
            raise ValueError(
                "pld.all_reduce(mode='ring') supports only FP32 targets (the HOST ring schedule is FP32-only)"
            )
        signal = _fresh_signal("allreduce_ring", [2 * (nranks - 1) + 1, nranks])
        return _tensor.allreduce(target, signal, op=op, mode="ring")
    if nranks is not None:
        raise ValueError(
            "pld.all_reduce nranks is only used with mode='ring'; "
            "omit it for mesh allreduce (or pass mode='ring')"
        )
    return _tensor.allreduce(target, op=op)


def all_gather(local_data: DistributedTensor, target: DistributedTensor) -> DistributedTensor:
    """All-gather ``local_data`` into ``target`` (push-based), auto-managing the signal.

    Args:
        local_data: Window-bound :class:`pld.DistributedTensor` ``[1, SIZE]``
            holding this rank's chunk. Must differ from ``target``.
        target: Window-bound :class:`pld.DistributedTensor` ``[NR, SIZE]``
            result window.

    Returns:
        The ``target`` :class:`pld.DistributedTensor` (window-as-result).
    """
    signal = _fresh_signal("allgather", [world_size(), 1])
    return _tensor.allgather(local_data, target, signal)


def reduce_scatter(
    target: DistributedTensor,
    *,
    op: ReduceOp = ReduceOp.Sum,
) -> DistributedTensor:
    """Reduce-scatter ``target`` in place, auto-managing the signal.

    Args:
        target: Window-bound :class:`pld.DistributedTensor` ``[NR, SIZE]``
            (each rank stages all NR chunks, one per row).
        op: :class:`pld.ReduceOp`. HOST ``reduce_scatter`` supports ``Sum``
            only; the wrapper rejects other ops up front.

    Returns:
        The rebound ``target`` :class:`pld.DistributedTensor`.
    """
    # The HOST builtin.tensor.reduce_scatter requires a rank-1 [world_size] signal.
    if op != ReduceOp.Sum:
        raise ValueError(
            "pld.reduce_scatter supports only ReduceOp.Sum (the HOST builtin implements Sum only)"
        )
    signal = _fresh_signal("reduce_scatter", [world_size()])
    return _tensor.reduce_scatter(target, signal, op=op)


def broadcast(target: DistributedTensor, *, root: int) -> DistributedTensor:
    """Broadcast ``root``'s data to all ranks, auto-managing the signal.

    Args:
        target: Window-bound :class:`pld.DistributedTensor`; root must stage
            its data before the call.
        root: Root rank index (int).

    Returns:
        The rebound ``target`` :class:`pld.DistributedTensor`.
    """
    # The HOST builtin.tensor.broadcast requires a rank-1 [world_size] signal.
    signal = _fresh_signal("broadcast", [world_size()])
    return _tensor.broadcast(target, signal, root=root)


def barrier(signal: DistributedTensor) -> DistributedTensor:
    """Cross-rank barrier using an explicit, comm-domain-covered signal.

    Unlike the other collectives, ``pld.barrier`` has **no data buffer** from
    which ``MaterializeCommDomainScopes`` can inherit comm-domain coverage for
    an auto-allocated signal, and a barrier-only signal must be consumed by a
    device-tagged chip dispatch on ``main``. The caller therefore passes a
    signal that already carries coverage (e.g. an INT32 window also passed to a
    publish/dispatch call). A zero-arg auto-signal ``pld.barrier()`` becomes
    possible once the coverage fallback lands (pypto #2243, plan 65).

    Args:
        signal: Window-bound INT32 :class:`pld.DistributedTensor` with
            comm-domain coverage.

    Returns:
        The rebound signal :class:`pld.DistributedTensor`.
    """
    return _tensor.barrier(signal)


def all_to_all(input: Tensor | DistributedTensor, target: DistributedTensor) -> DistributedTensor:
    """Symmetric all-to-all (push-based), auto-managing the signal.

    Args:
        input: ``[NR, SIZE]`` :class:`pl.Tensor` or :class:`pld.DistributedTensor`
            with per-destination chunks, distinct from ``target``.
        target: Window-bound :class:`pld.DistributedTensor` ``[NR, SIZE]``
            result window.

    Returns:
        The ``target`` :class:`pld.DistributedTensor` (window-as-result).
    """
    signal = _fresh_signal("all_to_all", [world_size(), 1])
    return _tensor.all_to_all(input, target, signal)


def all_to_all_v(
    input: Tensor | DistributedTensor,
    target: DistributedTensor,
    send_counts: Tensor | DistributedTensor,
    recv_counts: DistributedTensor,
    *,
    nranks: int,
) -> DistributedTensor:
    """Variable-size all-to-all (push-based), auto-managing the signal.

    Args:
        input: Flat ``[NR*MAX_RECV, SIZE]`` send buffer.
        target: Flat window-bound :class:`pld.DistributedTensor`
            ``[NR*MAX_RECV, SIZE]`` staging/result window.
        send_counts: Local INT32 ``[NR]`` per-peer send row counts.
        recv_counts: Published INT32 :class:`pld.DistributedTensor` ``[NR, 1]``.
        nranks: Static world size (int), required — the signal's first
            dimension and ``MAX_RECV = target.shape[0] // NR`` must be
            compile-time constants, so a dynamic ``world_size()`` is rejected
            by the ``pld.tensor.all_to_all_v`` type deducer.

    Returns:
        The ``target`` :class:`pld.DistributedTensor` (window-as-result).
    """
    if not _is_static_positive_int(nranks):
        raise ValueError(
            "pld.all_to_all_v requires a positive static int `nranks` "
            "(the signal shape [nranks, 1] and MAX_RECV = target.shape[0] // NR "
            "need a compile-time NR)"
        )
    signal = _fresh_signal("all_to_all_v", [nranks, 1])
    return _tensor.all_to_all_v(input, target, signal, send_counts, recv_counts)
