# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared device-memory + DeviceTensor surface for reuse handles.

The L2 reuse handle :class:`~pypto.runtime.Worker` and the L3 reuse handle
:class:`~pypto.runtime.distributed_runner.DistributedRuntime` expose the same
``malloc`` / ``free`` / ``copy_to`` / ``copy_from`` device-memory primitives plus
the ``alloc_tensor`` / ``free_tensor`` :class:`~pypto.runtime.DeviceTensor`
conveniences. This module factors that surface into a single ABC so the
conveniences live in exactly one place.

The four primitives are abstract: each subclass routes them to its own backend
(``Worker`` to ``self._impl.*``; ``DistributedRuntime`` through its orchestrator
facade) and applies its own readiness guard. Two hooks cover the genuine
differences between the levels:

- :meth:`DeviceMemoryHandle._require_ready` — the per-op readiness guard.
- :meth:`DeviceMemoryHandle._prepare_init` — the host-init upload policy. L2 makes
  a defensive CPU copy; L3 forbids the copy and requires shared memory, because
  its upload runs inside a forked child that only sees host memory it inherited
  at fork.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from .device_tensor import DeviceTensor, alloc_device_tensor, default_init_prep


class DeviceMemoryHandle(ABC):
    """Shared device-memory + :class:`DeviceTensor` surface for reuse handles.

    Subclasses implement the four primitives (:meth:`malloc`, :meth:`free`,
    :meth:`copy_to`, :meth:`copy_from`) with their own backend routing and
    readiness guard; the :class:`DeviceTensor` conveniences (:meth:`alloc_tensor`,
    :meth:`free_tensor`) are provided here once.
    """

    #: Noun used in the ``worker_id != 0`` error message; subclasses override.
    _WORKER_KIND: str = "worker"

    # ------------------------------------------------------------------
    # Primitives — implemented per subclass (routing + readiness guard).
    # ------------------------------------------------------------------

    @abstractmethod
    def malloc(self, nbytes: int, *, worker_id: int = 0) -> int:
        """Allocate ``nbytes`` of device memory; return an opaque pointer."""

    @abstractmethod
    def free(self, ptr: int, *, worker_id: int = 0) -> None:
        """Release a pointer previously returned by :meth:`malloc`."""

    @abstractmethod
    def copy_to(self, dst_dev_ptr: int, src_host_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        """H2D copy: ``nbytes`` bytes from host *src_host_ptr* to device *dst_dev_ptr*."""

    @abstractmethod
    def copy_from(self, dst_host_ptr: int, src_dev_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        """D2H copy: ``nbytes`` bytes from device *src_dev_ptr* back to host *dst_host_ptr*."""

    # ------------------------------------------------------------------
    # Hooks — overridable behaviour that genuinely differs per level.
    # ------------------------------------------------------------------

    def _require_ready(self, op: str) -> None:
        """Raise if this handle is not ready for device-memory ops.

        Default is a no-op; subclasses raise (e.g. before ``init()`` or after
        ``close()``).
        """

    def _prepare_init(self, init: torch.Tensor) -> torch.Tensor:
        """Return the host tensor to upload into a freshly allocated buffer.

        Default makes a defensive contiguous CPU copy. Subclasses that upload
        from a forked child override this to require shared memory instead.
        """
        return default_init_prep(init)

    # ------------------------------------------------------------------
    # DeviceTensor conveniences — shared.
    # ------------------------------------------------------------------

    def alloc_tensor(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        init: torch.Tensor | None = None,
        worker_id: int = 0,
    ) -> DeviceTensor:
        """Allocate a device buffer and (optionally) upload host data.

        Convenience wrapper around :meth:`malloc` + :meth:`copy_to`. When *init*
        is provided its dtype and shape must match exactly; the host buffer
        uploaded is :meth:`_prepare_init` applied to *init*. If any step after
        :meth:`malloc` raises, the allocation is rolled back via :meth:`free`
        before the exception propagates so callers never observe a leaked
        pointer.

        Returns:
            A :class:`DeviceTensor` referencing the allocated buffer. Free it via
            :meth:`free_tensor` before this handle is closed.
        """
        self._require_ready("alloc_tensor")
        # DeviceTensor only carries (data_ptr, shape, dtype) — no worker_id.
        # Until the handle encodes worker scope, restrict the convenience helpers
        # to the default worker so free_tensor cannot silently free a different
        # worker's pointer.
        if worker_id != 0:
            raise ValueError(
                f"{type(self).__name__}.alloc_tensor currently only supports worker_id=0. "
                f"Use malloc/copy_to directly if you need a different {self._WORKER_KIND}."
            )
        return alloc_device_tensor(
            malloc=lambda nbytes: self.malloc(nbytes, worker_id=worker_id),
            copy_to=lambda dst, src, nbytes: self.copy_to(dst, src, nbytes, worker_id=worker_id),
            free=lambda ptr: self.free(ptr, worker_id=worker_id),
            shape=shape,
            dtype=dtype,
            init=init,
            init_prep=self._prepare_init,
        )

    def free_tensor(self, t: DeviceTensor, *, worker_id: int = 0) -> None:
        """Release a buffer previously returned by :meth:`alloc_tensor`."""
        if worker_id != 0:
            raise ValueError(
                f"{type(self).__name__}.free_tensor currently only supports worker_id=0. "
                f"Use free directly if you need a different {self._WORKER_KIND}."
            )
        self.free(t.data_ptr, worker_id=worker_id)
