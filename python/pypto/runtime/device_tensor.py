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
on the worker (``ContinuousTensor.child_memory == 1``).

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
