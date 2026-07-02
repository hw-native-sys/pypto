# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""User-facing handle to worker-resident device memory.

A :class:`DeviceTensor` is an opaque ``(data_ptr, shape, dtype)`` triple bound
to a specific :class:`~pypto.runtime.Worker`'s address space.  Pass it to
:class:`~pypto.ir.compiled_program.CompiledProgram` in place of a
``torch.Tensor`` to skip the host→device copy on entry and the device→host
copy on exit — the runtime treats the underlying buffer as already resident
on the worker (``Tensor.child_memory == 1``).

Lifetime is **caller-managed**: every :meth:`~pypto.runtime.Worker.malloc`
(or :meth:`~pypto.runtime.Worker.alloc_tensor`) must be paired with a
matching :meth:`~pypto.runtime.Worker.free` (or
:meth:`~pypto.runtime.Worker.free_tensor`) before the Worker is closed.

Because no D2H copy happens for a ``DeviceTensor``, callers that want to
read the data back must do so explicitly via
:meth:`~pypto.runtime.Worker.copy_from`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceTensor:
    """Handle to a buffer already resident on a Worker.

    Attributes:
        data_ptr: Device pointer in the owning Worker's address space.
        shape: Logical tensor shape (all dimensions positive).
        dtype: Element ``torch.dtype``.
    """

    data_ptr: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __init__(self, data_ptr: int, shape: Sequence[int], dtype: torch.dtype) -> None:
        # bool is an int subclass — exclude it explicitly so True/False can't pose as a pointer or dim.
        if isinstance(data_ptr, bool) or not isinstance(data_ptr, int) or data_ptr <= 0:
            raise ValueError(f"DeviceTensor.data_ptr must be a positive int, got {data_ptr!r}")
        raw_shape = tuple(shape)
        for d in raw_shape:
            if isinstance(d, bool) or not isinstance(d, int):
                raise TypeError(f"DeviceTensor.shape must contain ints, got {raw_shape!r}")
        if not raw_shape:
            raise ValueError("DeviceTensor.shape must be non-empty")
        if any(d <= 0 for d in raw_shape):
            raise ValueError(f"DeviceTensor.shape must be all positive, got {raw_shape}")
        shape_t = raw_shape
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"DeviceTensor.dtype must be torch.dtype, got {type(dtype).__name__}")
        object.__setattr__(self, "data_ptr", data_ptr)
        object.__setattr__(self, "shape", shape_t)
        object.__setattr__(self, "dtype", dtype)

    @property
    def nbytes(self) -> int:
        """Total bytes referenced by this handle."""
        elem = torch.tensor([], dtype=self.dtype).element_size()
        n = 1
        for d in self.shape:
            n *= d
        return n * elem

    def __repr__(self) -> str:
        return f"DeviceTensor(data_ptr=0x{self.data_ptr:x}, shape={self.shape}, dtype={self.dtype})"


@dataclass(frozen=True)
class StackedDeviceTensor:
    """A leading-dim-stacked ``[B, *tail]`` tensor whose ``B`` shards are each
    resident on a (possibly different) worker.

    Pass it to a distributed program in place of a host ``torch.Tensor`` for a
    parameter that the orchestrator slices along its leading dimension and
    dispatches per rank (``for r in range(world_size): child(x[r], device=...)``).
    Indexing ``obj[i, ...]`` returns shard ``i``'s :class:`DeviceTensor`, so the
    generated ``host_orch`` wraps it as ``child_memory=True`` and the runtime
    skips the per-dispatch H2D upload — the shards are uploaded once (e.g. via
    :meth:`~pypto.runtime.distributed_runner.DistributedWorker.alloc_stacked_tensor`)
    and reused across dispatches.

    Correctness contract: ``worker_ids[i]`` is the worker that holds shard ``i``
    and MUST equal the worker the compiled program submits ``x[i]``'s task to
    (its ``device=`` expression). The canonical ``device=r`` program uses the
    identity ``worker_ids == range(B)``; a permuted or subset placement (e.g.
    ``device=perm[r]`` / ``device=2*r``) requires the matching ``worker_ids``.

    Attributes:
        shards: One :class:`DeviceTensor` per leading-dim index; ``shards[i]`` is
            resident on worker ``worker_ids[i]`` with shape ``full_shape[1:]``.
        full_shape: The logical stacked shape ``(B, *tail)``.
        worker_ids: ``worker_ids[i]`` is the worker holding ``shards[i]``.
    """

    shards: tuple[DeviceTensor, ...]
    full_shape: tuple[int, ...]
    worker_ids: tuple[int, ...]

    def __init__(
        self,
        shards: Sequence[DeviceTensor],
        full_shape: Sequence[int],
        worker_ids: Sequence[int],
    ) -> None:
        shards_t = tuple(shards)
        full_t = tuple(full_shape)
        workers_t = tuple(worker_ids)
        if len(full_t) < 2:
            raise ValueError(f"StackedDeviceTensor.full_shape must have rank >= 2, got {full_t}")
        b = full_t[0]
        if b < 1:
            raise ValueError(
                f"StackedDeviceTensor needs at least one shard in the leading dim, got full_shape {full_t}"
            )
        tail = full_t[1:]
        if len(shards_t) != b:
            raise ValueError(
                f"StackedDeviceTensor expects {b} shards (leading dim of {full_t}); got {len(shards_t)}"
            )
        if len(workers_t) != b:
            raise ValueError(
                f"StackedDeviceTensor expects {b} worker_ids (one per shard); got {len(workers_t)}"
            )
        if len(set(workers_t)) != len(workers_t):
            raise ValueError(f"StackedDeviceTensor.worker_ids must be distinct, got {workers_t}")
        for i, shard in enumerate(shards_t):
            if not isinstance(shard, DeviceTensor):
                raise TypeError(f"shard {i} must be a DeviceTensor, got {type(shard).__name__}")
            if shard.shape != tail:
                raise ValueError(
                    f"shard {i} has shape {shard.shape}; expected per-shard shape {tail} "
                    f"(leading dim dropped from {full_t})"
                )
            if shard.dtype != shards_t[0].dtype:
                raise ValueError(
                    f"shard {i} dtype {shard.dtype} differs from shard 0 dtype {shards_t[0].dtype}"
                )
        object.__setattr__(self, "shards", shards_t)
        object.__setattr__(self, "full_shape", full_t)
        object.__setattr__(self, "worker_ids", workers_t)

    @property
    def dtype(self) -> torch.dtype:
        """Element ``torch.dtype`` shared by all shards."""
        return self.shards[0].dtype

    def __getitem__(self, idx: int | slice | tuple) -> DeviceTensor:
        """Return shard ``i`` for a leading-index ``i`` or ``(i, <full slices>)``.

        The generated ``host_orch`` emits either ``x[r]`` or ``x[r, 0:N, 0:M]``,
        and callers may use the ``x[r, ...]`` (Ellipsis) whole-shard form; all
        resolve to shard ``r``. Any non-whole-shard trailing slice is rejected
        loudly — a stacked tensor only supports whole-shard slicing on the
        leading dimension.
        """
        if isinstance(idx, tuple):
            if not idx:
                raise IndexError("StackedDeviceTensor requires a leading-dim index")
            rank, rest = idx[0], idx[1:]
        else:
            rank, rest = idx, ()
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"StackedDeviceTensor leading index must be int, got {type(rank).__name__}")
        if not 0 <= rank < len(self.shards):
            raise IndexError(f"shard index {rank} out of range [0, {len(self.shards)})")
        tail = self.full_shape[1:]
        # Expand a single Ellipsis into full slices so each trailing index maps
        # to a concrete shard axis; ``x[i, ...]`` is the documented whole-shard
        # form and must behave like ``x[i]``.
        if any(s is Ellipsis for s in rest):
            if sum(s is Ellipsis for s in rest) > 1:
                raise IndexError("StackedDeviceTensor index accepts at most one Ellipsis")
            e = rest.index(Ellipsis)
            n_fill = len(tail) - (len(rest) - 1)
            if n_fill < 0:
                raise IndexError(f"too many indices for StackedDeviceTensor of shape {self.full_shape}")
            rest = rest[:e] + tuple(slice(None) for _ in range(n_fill)) + rest[e + 1 :]
        if len(rest) > len(tail):
            raise IndexError(f"too many indices for StackedDeviceTensor of shape {self.full_shape}")
        for axis, s in enumerate(rest):
            full = slice(0, tail[axis])
            if s != full and s != slice(None):
                raise ValueError(
                    f"StackedDeviceTensor only supports whole-shard slicing on the leading "
                    f"dim; got partial slice {s} on axis {axis + 1} (shard shape {tail})"
                )
        return self.shards[rank]

    def __repr__(self) -> str:
        return (
            f"StackedDeviceTensor(full_shape={self.full_shape}, "
            f"worker_ids={self.worker_ids}, dtype={self.dtype})"
        )


def default_init_prep(init: torch.Tensor) -> torch.Tensor:
    """Default host-buffer prep for an upload: a defensive contiguous CPU copy."""
    return init.contiguous().cpu()


def alloc_device_tensor(
    *,
    malloc: Callable[[int], int],
    copy_to: Callable[[int, int, int], None],
    free: Callable[[int], None],
    shape: Sequence[int],
    dtype: torch.dtype,
    init: torch.Tensor | None = None,
    init_prep: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> DeviceTensor:
    """Allocate a device buffer and (optionally) upload host data.

    Shared by :meth:`pypto.runtime.ChipWorker.alloc_tensor` (L2) and
    :meth:`pypto.runtime.distributed_runner.DistributedWorker.alloc_tensor`
    (L3). The ``malloc`` / ``copy_to`` / ``free`` callables are injected with
    any ``worker_id`` already bound, so this helper stays free of worker scope.

    When *init* is provided its dtype and shape must match exactly. The host
    buffer actually uploaded is ``init_prep(init)``: the default makes a
    defensive contiguous CPU copy (L2), while L3 overrides it to *reject* a copy
    and require ``init`` already be shared memory (the upload runs in a forked
    child that can only see host memory inherited at fork). If any step after
    ``malloc`` raises, the allocation is rolled back via ``free`` before the
    exception propagates so callers never observe a leaked pointer.

    Args:
        malloc: ``malloc(nbytes) -> device_ptr``.
        copy_to: ``copy_to(dst_dev_ptr, src_host_ptr, nbytes) -> None`` (H2D).
        free: ``free(device_ptr) -> None`` (rollback on failure).
        shape: Logical tensor shape (all dimensions positive).
        dtype: Element ``torch.dtype``.
        init: Optional host tensor to upload into the buffer.
        init_prep: Maps ``init`` to the host tensor actually uploaded. Defaults
            to a defensive ``init.contiguous().cpu()`` copy.

    Returns:
        A :class:`DeviceTensor` referencing the allocated buffer.
    """
    # Validate the shape up front (before malloc) and without coercion, mirroring
    # DeviceTensor's constructor contract: bool is an int subclass, so reject it
    # explicitly; only positive int dimensions are allowed. This avoids allocating
    # for a wrong logical shape (e.g. an empty shape would make n_elems == 1) and
    # gives the same error the resulting DeviceTensor would raise — just earlier.
    shape_t = tuple(shape)
    if not shape_t:
        raise ValueError("shape must be non-empty")
    for d in shape_t:
        if isinstance(d, bool) or not isinstance(d, int):
            raise TypeError(f"shape must contain ints, got {shape_t!r}")
    if any(d <= 0 for d in shape_t):
        raise ValueError(f"shape must contain only positive dimensions, got {shape_t}")
    n_elems = 1
    for d in shape_t:
        n_elems *= d
    elem = torch.tensor([], dtype=dtype).element_size()
    nbytes = n_elems * elem
    ptr = malloc(nbytes)
    try:
        if init is not None:
            if init.dtype != dtype or tuple(init.shape) != shape_t:
                raise ValueError(
                    f"init must have shape={shape_t} dtype={dtype}, "
                    f"got shape={tuple(init.shape)} dtype={init.dtype}"
                )
            host = (init_prep or default_init_prep)(init)
            copy_to(ptr, host.data_ptr(), nbytes)
        return DeviceTensor(ptr, shape_t, dtype)
    except Exception:
        free(ptr)
        raise
