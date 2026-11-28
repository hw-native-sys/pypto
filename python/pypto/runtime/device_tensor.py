# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""User-facing handle to worker-resident device memory.

A :class:`DeviceTensor` exposes an opaque ``(data_ptr, shape, dtype)`` triple
bound to a specific :class:`~pypto.runtime.Worker`'s address space. Tensors
allocated by a Worker also retain simpler's owning ``Buffer`` handle; the
address-free distributed wire ABI derives its ``Tensor`` descriptor from that
handle instead of transporting the process-local pointer. Pass it to
:class:`~pypto.ir.compiled_program.CompiledProgram` in place of a
``torch.Tensor`` to skip the host→device copy on entry and the device→host
copy on exit. Public L2 and distributed dispatch both derive an address-free
wire ``Tensor`` from the retained owner ``Buffer``. The direct ``ChipTensor``
adapter remains available only as a low-level compatibility helper.

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
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch


def _validate_strides(shape: tuple[int, ...], strides: Sequence[int] | None) -> tuple[int, ...]:
    if strides is None:
        result: list[int] = []
        pitch = 1
        for extent in reversed(shape):
            result.append(pitch)
            pitch *= extent
        return tuple(reversed(result))
    result_t = tuple(strides)
    if len(result_t) != len(shape) or any(type(s) is not int or s <= 0 for s in result_t):
        raise ValueError(f"strides must contain {len(shape)} positive integers, got {result_t}")
    return result_t


def _view_elements(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    return 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides, strict=True))


def _indices(shape: tuple[int, ...], index: Any) -> tuple[int | slice, ...]:
    indices: tuple[Any, ...] = cast(tuple[Any, ...], index) if isinstance(index, tuple) else (index,)
    if sum(value is Ellipsis for value in indices) > 1:
        raise IndexError("tensor index accepts at most one Ellipsis")
    expanded: list[int | slice] = []
    for value in indices:
        if value is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - len(indices) + 1))
        elif type(value) is int or isinstance(value, slice):
            expanded.append(cast(int | slice, value))
        else:
            raise TypeError(f"tensor index must be int or slice, got {value!r}")
    if len(expanded) > len(shape):
        raise IndexError(f"too many indices for tensor shape {shape}")
    expanded.extend([slice(None)] * (len(shape) - len(expanded)))
    normalized: list[int | slice] = []
    for extent, value in zip(shape, expanded, strict=True):
        if isinstance(value, int):
            value = value + extent if value < 0 else value
            if not 0 <= value < extent:
                raise IndexError(f"index {value} is out of range for dimension {extent}")
        else:
            start, stop, step = value.indices(extent)
            if step <= 0 or stop <= start:
                raise ValueError("device views require nonempty positive-step slices")
            value = slice(start, stop, step)
        normalized.append(value)
    return tuple(normalized)


def _slice_resident(tensor: Any, index: Any) -> Any:
    indices = _indices(tensor.shape, index)
    shape: list[int] = []
    strides: list[int] = []
    offset = tensor.byte_offset
    itemsize = torch.empty((), dtype=tensor.dtype).element_size()
    for value, stride in zip(indices, tensor.strides, strict=True):
        if isinstance(value, int):
            offset += value * stride * itemsize
        else:
            start, stop, step = value.start, value.stop, value.step
            assert start is not None and stop is not None and step is not None
            offset += start * stride * itemsize
            shape.append(len(range(start, stop, step)))
            strides.append(stride * step)
    if not shape:
        raise ValueError("device indexing must retain at least one tensor dimension")
    if tuple(shape) == tensor.shape and offset == tensor.byte_offset and tuple(strides) == tensor.strides:
        return tensor
    if isinstance(tensor, RemoteDeviceTensor):
        return replace(
            tensor,
            shape=tuple(shape),
            strides=tuple(strides),
            byte_offset=offset,
            _source=tensor._source if tensor._source is not None else tensor,
        )
    return DeviceTensor(
        tensor.data_ptr + offset - tensor.byte_offset,
        shape,
        tensor.dtype,
        buffer=tensor.buffer,
        byte_offset=offset,
        strides=strides,
    )


@dataclass(frozen=True)
class DeviceTensor:
    """Handle to a buffer already resident on a Worker.

    Attributes:
        data_ptr: Device pointer of the first element this handle addresses.
            For a borrowed view this is ``buffer.base + byte_offset``, not the
            allocation base.
        shape: Logical tensor shape (all dimensions positive).
        dtype: Element ``torch.dtype``.
        byte_offset: Distance in bytes from the owning allocation's base to
            ``data_ptr``. Zero for a whole allocation, nonzero for a view
            created by :meth:`Worker.view_tensor`; must be aligned to ``dtype``.
        buffer: Optional simpler ``Buffer`` retained for address-free wire
            dispatch. It is populated by :meth:`Worker.alloc_tensor` and
            inherited by :meth:`Worker.view_tensor`; a manually constructed
            raw-pointer tensor can be used only with low-level direct-chip
            compatibility helpers. All public ``Worker.run`` paths require
            :meth:`Worker.alloc_tensor` and its retained owner ``Buffer``.
    """

    data_ptr: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    buffer: Any | None = field(default=None, repr=False, compare=False, hash=False)
    byte_offset: int = 0
    strides: tuple[int, ...] | None = None

    def __init__(
        self,
        data_ptr: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        buffer: Any | None = None,
        byte_offset: int = 0,
        strides: Sequence[int] | None = None,
    ) -> None:
        # bool is an int subclass — an exact-type check excludes it so True/False
        # can't pose as a pointer or dim.
        if type(data_ptr) is not int or data_ptr <= 0:
            raise ValueError(f"DeviceTensor.data_ptr must be a positive int, got {data_ptr!r}")
        raw_shape = tuple(shape)
        for d in raw_shape:
            if type(d) is not int:
                raise TypeError(f"DeviceTensor.shape must contain ints, got {raw_shape!r}")
        if not raw_shape:
            raise ValueError("DeviceTensor.shape must be non-empty")
        if any(d <= 0 for d in raw_shape):
            raise ValueError(f"DeviceTensor.shape must be all positive, got {raw_shape}")
        shape_t = raw_shape
        strides_t = _validate_strides(shape_t, strides)
        if type(dtype) is not torch.dtype:
            raise TypeError(f"DeviceTensor.dtype must be torch.dtype, got {type(dtype).__name__}")
        if type(byte_offset) is not int or byte_offset < 0:
            raise ValueError(f"DeviceTensor.byte_offset must be a nonnegative int, got {byte_offset!r}")
        element_size = torch.empty((), dtype=dtype).element_size()
        if byte_offset % element_size:
            raise ValueError("DeviceTensor.byte_offset must be aligned to its dtype")
        if buffer is not None:
            try:
                buffer_base = int(buffer.base)
            except (AttributeError, TypeError, ValueError) as e:
                raise TypeError("DeviceTensor.buffer must be a simpler Buffer with an integer base") from e
            if buffer_base + byte_offset != data_ptr:
                raise ValueError(
                    f"DeviceTensor.buffer.base (0x{buffer_base:x}) plus byte_offset "
                    f"does not match data_ptr (0x{data_ptr:x})"
                )
            if not callable(getattr(buffer, "tensor", None)):
                raise TypeError("DeviceTensor.buffer must provide buffer.tensor(shapes, dtype)")
            if hasattr(buffer, "nbytes"):
                if byte_offset + _view_elements(shape_t, strides_t) * element_size > buffer.nbytes:
                    raise ValueError("DeviceTensor view exceeds its owning buffer")
        object.__setattr__(self, "data_ptr", data_ptr)
        object.__setattr__(self, "shape", shape_t)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "buffer", buffer)
        object.__setattr__(self, "byte_offset", byte_offset)
        object.__setattr__(self, "strides", strides_t)

    def __getitem__(self, index: Any) -> DeviceTensor:
        """Create a bounded address-free view, preserving the owning Buffer."""
        return _slice_resident(self, index)

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


@dataclass(frozen=True, eq=False)
class RemoteDeviceTensor:
    """A view of a remote card's retained Device-source domain buffer.

    There is deliberately no data_ptr: the controller cannot dereference the
    owner's address. The creating DistributedWorker retains the domain and
    validates this view before dispatch. Releasing the domain invalidates it.
    General remote malloc is not used, since its backing is HOST staging.
    """

    domain: Any = field(repr=False)
    domain_rank: int
    buffer_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    buffer: Any = field(repr=False)
    byte_offset: int = 0
    strides: tuple[int, ...] | None = None
    _source: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from simpler.buffer import (  # noqa: PLC0415 -- optional runtime or import-cycle boundary
            AddressSpace,  # noqa: PLC0415 -- avoid importing the optional runtime during compilation
        )

        shape = tuple(self.shape)
        if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("RemoteDeviceTensor.shape must contain positive ints")
        if type(self.dtype) is not torch.dtype:
            raise TypeError("RemoteDeviceTensor.dtype must be a torch.dtype")
        if type(self.domain_rank) is not int or self.domain_rank < 0:
            raise ValueError("domain_rank must be a non-negative int")
        if type(self.byte_offset) is not int or self.byte_offset < 0:
            raise ValueError("byte_offset must be a non-negative int")
        if self.buffer.address_space != AddressSpace.DEVICE:
            raise ValueError("RemoteDeviceTensor requires Device-source storage")
        if self.domain.member(self.domain_rank).buffers[self.buffer_name] is not self.buffer:
            raise ValueError("buffer must be the named domain member's original Buffer")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", _validate_strides(shape, self.strides))
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        if (
            self.byte_offset % itemsize
            or self.byte_offset + _view_elements(shape, _validate_strides(shape, self.strides)) * itemsize
            > self.buffer.nbytes
        ):
            raise ValueError("RemoteDeviceTensor view is misaligned or exceeds its buffer")

    def __getitem__(self, index: Any) -> RemoteDeviceTensor:
        """Create a bounded borrowed view without exposing an owner address."""
        return _slice_resident(self, index)

    @property
    def nbytes(self) -> int:
        """Logical byte length, excluding domain alignment and other slices."""
        count = 1
        for dim in self.shape:
            count *= dim
        return count * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class StackedDeviceTensor:
    """A tensor split on ``shard_axis`` whose shards are each
    resident on a (possibly different) worker.

    Pass it to a distributed program in place of a host ``torch.Tensor`` for a
    parameter that the orchestrator slices along its declared shard axis and
    dispatches per rank (``for r in range(world_size): child(x[r], device=...)``).
    Indexing the declared axis selects a local or remote resident shard, so the
    generated ``host_orch`` derives a wire Tensor from its retained Buffer and
    skips the per-dispatch H2D upload — the shards are uploaded once (e.g. via
    :meth:`~pypto.runtime.distributed_runner.DistributedWorker.alloc_stacked_tensor`)
    and reused across dispatches.

    Correctness contract: ``worker_ids[i]`` is the worker that holds shard ``i``
    and MUST equal the worker the compiled program submits ``x[i]``'s task to
    (its ``device=`` expression). The canonical ``device=r`` program uses the
    identity ``worker_ids == range(B)``; a permuted or subset placement (e.g.
    ``device=perm[r]`` / ``device=2*r``) requires the matching ``worker_ids``.

    Attributes:
        shards: One local or remote resident tensor per shard-axis index; ``shards[i]`` is
            resident on worker ``worker_ids[i]`` with shape ``full_shape[1:]``.
        full_shape: The logical tensor shape before splitting on ``shard_axis``.
        shape: Alias for ``full_shape``, matching the :class:`DeviceTensor`
            interface.
        worker_ids: ``worker_ids[i]`` is the worker holding ``shards[i]``.
    """

    shards: tuple[DeviceTensor | RemoteDeviceTensor, ...]
    full_shape: tuple[int, ...]
    worker_ids: tuple[int, ...]
    shard_axis: int

    def __init__(
        self,
        shards: Sequence[DeviceTensor | RemoteDeviceTensor],
        full_shape: Sequence[int],
        worker_ids: Sequence[int],
        *,
        shard_axis: int = 0,
    ) -> None:
        shards_t = tuple(shards)
        full_t = tuple(full_shape)
        workers_t = tuple(worker_ids)
        if len(full_t) < 2:
            raise ValueError(f"StackedDeviceTensor.full_shape must have rank >= 2, got {full_t}")
        if type(shard_axis) is not int or not 0 <= shard_axis < len(full_t):
            raise ValueError(f"shard_axis must be in [0, {len(full_t)}), got {shard_axis!r}")
        b = full_t[shard_axis]
        if b < 1:
            raise ValueError(
                f"StackedDeviceTensor needs at least one shard on the shard axis, got full_shape {full_t}"
            )
        if any(type(dim) is not int or dim <= 0 for dim in full_t):
            raise ValueError(f"full_shape must contain positive integers, got {full_t}")
        if any(type(worker) is not int or worker < 0 for worker in workers_t):
            raise ValueError(f"worker_ids must contain nonnegative integers, got {workers_t}")
        tail = full_t[:shard_axis] + full_t[shard_axis + 1 :]
        if len(shards_t) != b:
            raise ValueError(
                f"StackedDeviceTensor expects {b} shards (axis {shard_axis} of {full_t}); got {len(shards_t)}"
            )
        if len(workers_t) != b:
            raise ValueError(
                f"StackedDeviceTensor expects {b} worker_ids (one per shard); got {len(workers_t)}"
            )
        if len(set(workers_t)) != len(workers_t):
            raise ValueError(f"StackedDeviceTensor.worker_ids must be distinct, got {workers_t}")
        for i, shard in enumerate(shards_t):
            if type(shard) not in (DeviceTensor, RemoteDeviceTensor):
                raise TypeError(
                    f"shard {i} must be a DeviceTensor or RemoteDeviceTensor, got {type(shard).__name__}"
                )
            if shard.shape != tail:
                raise ValueError(
                    f"shard {i} has shape {shard.shape}; expected per-shard shape {tail} "
                    f"(shard axis dropped from {full_t})"
                )
            if shard.dtype != shards_t[0].dtype:
                raise ValueError(
                    f"shard {i} dtype {shard.dtype} differs from shard 0 dtype {shards_t[0].dtype}"
                )
        object.__setattr__(self, "shards", shards_t)
        object.__setattr__(self, "full_shape", full_t)
        object.__setattr__(self, "worker_ids", workers_t)
        object.__setattr__(self, "shard_axis", shard_axis)

    @property
    def dtype(self) -> torch.dtype:
        """Element ``torch.dtype`` shared by all shards."""
        return self.shards[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical stacked shape, matching the ``DeviceTensor`` interface."""
        return self.full_shape

    def __getitem__(self, index: Any) -> DeviceTensor | RemoteDeviceTensor | StackedDeviceTensor:
        """Resolve global indexing to rank-owned views on the declared shard axis."""
        indices = _indices(self.full_shape, index)
        rank = indices[self.shard_axis]
        local = indices[: self.shard_axis] + indices[self.shard_axis + 1 :]
        if isinstance(rank, int):
            return _slice_resident(self.shards[rank], local)
        assert rank.start is not None and rank.stop is not None and rank.step is not None
        selected = tuple(range(rank.start, rank.stop, rank.step))
        shards = tuple(_slice_resident(self.shards[i], local) for i in selected)
        shape = tuple(len(range(v.start, v.stop, v.step)) for v in indices if isinstance(v, slice))
        axis = sum(isinstance(v, slice) for v in indices[: self.shard_axis])
        return StackedDeviceTensor(
            shards, shape, tuple(self.worker_ids[i] for i in selected), shard_axis=axis
        )

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
    buffer_for_ptr: Callable[[int], Any | None] | None = None,
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
    defensive contiguous CPU copy. L3 accepts an ordinary CPU-contiguous tensor
    and stages it through a runtime-owned shared Buffer. If any step after
    ``malloc`` raises, the allocation is rolled back via ``free`` before the
    exception propagates so callers never observe a leaked pointer.

    Args:
        malloc: ``malloc(nbytes) -> device_ptr``.
        copy_to: ``copy_to(dst_dev_ptr, src_host_ptr, nbytes) -> None`` (H2D).
        free: ``free(device_ptr) -> None`` (rollback on failure).
        buffer_for_ptr: Resolve the pointer to its owning simpler ``Buffer``.
            Real runtime Workers provide this so the resulting tensor can be
            packed into address-free wire ``TaskArgs``. A custom Worker that
            omits it can still expose raw-pointer memory operations, but its
            resulting ``DeviceTensor`` cannot be dispatched.
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
        if type(d) is not int:
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
        buffer = buffer_for_ptr(ptr) if buffer_for_ptr is not None else None
        return DeviceTensor(ptr, shape_t, dtype, buffer=buffer)
    except Exception:
        free(ptr)
        raise
