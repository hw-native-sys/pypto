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

from collections.abc import Sequence
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
