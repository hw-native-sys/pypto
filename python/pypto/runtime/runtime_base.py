# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared :class:`Worker` ABC for all PyPTO runtime handles.

PyPTO's L2 handle :class:`~pypto.runtime.ChipWorker` and L3 handle
:class:`~pypto.runtime.distributed_runner.DistributedWorker` both inherit from
:class:`Worker`. The ABC factors out three orthogonal surfaces shared by every
level:

- **Device-memory primitives** — ``malloc`` / ``free`` / ``copy_to`` /
  ``copy_from`` are abstract; each subclass routes them to its own backend
  (``ChipWorker`` to ``self._impl.*``; ``DistributedWorker`` via the
  orchestrator facade).
- **DeviceTensor lifecycle** — :meth:`alloc_tensor` and :meth:`free_tensor` are
  concrete here. Tensors allocated through :meth:`alloc_tensor` are tracked in
  ``self._owned_tensors`` and released by :meth:`_close_owned_tensors`, which
  subclasses call from their own ``close()`` so a missed ``free_tensor`` never
  leaks past the Worker.
- **Dispatch surface** — :meth:`run` and :meth:`register` are abstract; library
  code that doesn't care about the level should type-hint against
  :class:`Worker` and call these.

Two hooks cover the genuine differences between the levels:

- :meth:`Worker._require_ready` — the per-op readiness guard.
- :meth:`Worker._prepare_init` — the host-init upload policy. L2 makes a
  defensive CPU copy; L3 forbids the copy and requires shared memory, because
  its upload runs inside a forked child that only sees host memory it inherited
  at fork.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from .device_tensor import DeviceTensor, alloc_device_tensor, default_init_prep

if TYPE_CHECKING:
    from .worker import RegistrationHandle


_log = logging.getLogger(__name__)


class Worker(ABC):
    """Abstract runtime handle: memory + DeviceTensor + dispatch surface.

    Concrete subclasses:

    - :class:`~pypto.runtime.ChipWorker` — L2, single-chip execution.
    - :class:`~pypto.runtime.distributed_runner.DistributedWorker` — L3+,
      multi-chip / forked execution.

    Library code that doesn't care about the execution level should type-hint
    against this base. Most users should construct one of the concrete
    subclasses directly.

    Subclass contract:

    - Implement :meth:`malloc`, :meth:`free`, :meth:`copy_to`,
      :meth:`copy_from` with backend-appropriate routing and a readiness guard.
    - Implement :meth:`run` and :meth:`register` for the level's dispatch
      semantics. ``register`` should return a :class:`RegistrationHandle`.
    - Override :meth:`_require_ready` to match the level's lifecycle (e.g.,
      ``ChipWorker`` requires :meth:`init`; ``DistributedWorker`` requires
      not-yet-closed).
    - Override :meth:`_prepare_init` if the host-init upload policy differs
      from L2's defensive CPU copy.
    - Call :meth:`_close_owned_tensors` from the subclass's ``close()`` before
      tearing down its backend so any tensor the caller forgot to release is
      reclaimed.
    """

    def __init__(self) -> None:
        # Subclasses MUST call ``super().__init__()`` so this map exists before
        # any ``alloc_tensor`` call. Tracks DeviceTensors allocated via
        # ``alloc_tensor`` so ``_close_owned_tensors`` can release any the
        # caller forgot. Keyed by ``(worker_id, data_ptr)`` so buffers allocated
        # on a non-default worker are freed against the correct worker. The
        # DeviceTensor value is the allocation identity: a stale handle must
        # not free a newer allocation that reused the same device address.
        self._owned_tensors: dict[tuple[int, int], DeviceTensor] = {}
        # Reverse allocation-identity index used by the generated L3 tensor
        # packing path. A wide distributed dispatch may validate thousands of
        # resident shards, so looking up the owning ``(worker_id, data_ptr)``
        # key must not scan the complete allocation table for every argument.
        self._owned_tensor_keys: dict[int, tuple[int, int]] = {}
        # Raw simpler Workers are bound back to this owner for DeviceTensor
        # wire conversion.  The concrete runtime sets this whenever it creates
        # (or recreates) its backend; tensor_arg.py rejects stale backends whose
        # binding no longer matches this current object.
        self._tensor_arg_worker: Any | None = None

    # ------------------------------------------------------------------
    # Memory primitives — implemented per subclass.
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
    # Dispatch — implemented per subclass.
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, compiled: Any, *args: Any, config: Any = None) -> Any:
        """Dispatch *compiled* on this Worker explicitly.

        Subclasses define the accepted *compiled* type:

        - :class:`~pypto.runtime.ChipWorker` accepts a
          :class:`~pypto.ir.CompiledProgram`.
        - :class:`~pypto.runtime.distributed_runner.DistributedWorker` accepts
          a :class:`~pypto.ir.DistributedCompiledProgram`.

        Return value follows the same rules as the implicit
        ``compiled(...)`` callable surface.
        """

    @abstractmethod
    def register(self, compiled: Any) -> RegistrationHandle:
        """Pre-register *compiled* on this Worker. Returns a callable handle.

        Calling the handle dispatches without re-checking the registry cache.
        Multiple ``register`` calls for the same *compiled* return aliases of
        the same underlying cid. ``RegistrationHandle.unregister()`` only
        marks that handle closed; the underlying cid is released once, in
        :meth:`close`.
        """

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

    def _buffer_for_ptr(self, ptr: int, *, worker_id: int = 0) -> Any | None:
        """Return the owning simpler ``Buffer`` for a public device pointer.

        PyPTO preserves its raw-pointer memory API, while current simpler uses
        owner ``Buffer`` objects internally and address-free ``Tensor`` values
        on the dispatch wire. Concrete runtime Workers override this hook for
        pointers returned by :meth:`malloc`. A custom Worker that returns only
        raw pointers can keep the default, but its ``DeviceTensor`` values
        cannot be dispatched through public ``Worker.run`` paths.
        """
        return None

    def _require_owned_resident_tensor(
        self,
        tensor: DeviceTensor,
        label: str,
        *,
        worker_id: int | None = None,
    ) -> None:
        """Require a live owner ``Buffer`` allocated by this Worker.

        ``DeviceTensor`` keeps its Buffer handle after ``free_tensor()`` so
        that the immutable public handle remains inspectable.  The retained
        handle alone is therefore not proof of liveness.  The authoritative
        check combines the exact ``DeviceTensor`` allocation identity with
        object identity in ``_device_buffers``: this rejects foreign Workers,
        freed handles, and pointer-reuse (ABA) cases before a stale wire
        descriptor can enter simpler ``Worker.run``. The reverse identity
        index keeps this check O(1), including when *worker_id* is omitted by
        generated orchestration code.
        """
        self._require_ready("dispatch DeviceTensor")
        worker_name = type(self).__name__
        if tensor.buffer is None:
            raise TypeError(
                f"{label}: a raw-pointer DeviceTensor cannot be dispatched by {worker_name}; "
                f"use this same {worker_name}.alloc_tensor() to create it."
            )

        buffers = getattr(self, "_device_buffers", None)
        if not isinstance(buffers, dict):
            raise TypeError(
                f"{label}: {worker_name} does not retain owner Buffers required for DeviceTensor dispatch."
            )
        key = self._owned_tensor_keys.get(id(tensor))
        owned = (
            key is not None
            and key[1] == tensor.data_ptr
            and (worker_id is None or key[0] == worker_id)
            and self._owned_tensors.get(key) is tensor
            and buffers.get(key) is tensor.buffer
        )
        if not owned:
            placement = "" if worker_id is None else f" on worker_id={worker_id}"
            raise ValueError(
                f"{label}: DeviceTensor is not a live allocation owned by this {worker_name}"
                f"{placement}; allocate it with this same worker and do not dispatch it after free_tensor()."
            )

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

        The returned :class:`DeviceTensor` is tracked by this Worker keyed by
        ``(worker_id, data_ptr)`` and retained as the allocation identity: if
        the caller does not :meth:`free_tensor` it before ``close()``, the
        subclass's ``close()`` (via
        :meth:`_close_owned_tensors`) reclaims it against the same worker.
        Because a :class:`DeviceTensor` does not itself carry its worker scope,
        a caller that allocates on a non-default worker MUST pass the same
        ``worker_id`` to :meth:`free_tensor`.

        Returns:
            A :class:`DeviceTensor` referencing the allocated buffer.
        """
        self._require_ready("alloc_tensor")
        t = alloc_device_tensor(
            malloc=lambda nbytes: self.malloc(nbytes, worker_id=worker_id),
            copy_to=lambda dst, src, nbytes: self.copy_to(dst, src, nbytes, worker_id=worker_id),
            free=lambda ptr: self.free(ptr, worker_id=worker_id),
            buffer_for_ptr=lambda ptr: self._buffer_for_ptr(ptr, worker_id=worker_id),
            shape=shape,
            dtype=dtype,
            init=init,
            init_prep=self._prepare_init,
        )
        key = (worker_id, t.data_ptr)
        self._owned_tensors[key] = t
        self._owned_tensor_keys[id(t)] = key
        return t

    def free_tensor(self, t: DeviceTensor, *, worker_id: int = 0) -> None:
        """Release a buffer previously returned by :meth:`alloc_tensor`.

        Frees the underlying pointer against *worker_id* (which MUST match the
        ``worker_id`` used to allocate *t*), then untracks *t* only after the
        backend reports success. A failed backend admission/free keeps the
        ownership key so the caller can retry.
        Idempotent for a genuine double-free: if ``t.data_ptr`` is no longer
        tracked under *any* worker (e.g. ``_close_owned_tensors`` ran first, or
        the caller already freed it), this is a no-op — the underlying ``free``
        is NOT called a second time. This protects against a double-free at the
        C++ layer, and against a post-close ``RuntimeError`` when explicit
        cleanup races with auto-free.

        Raises:
            ValueError: if ``t.data_ptr`` is still tracked but under a different
                ``worker_id`` — silently no-oping there would leak the buffer
                until ``close()`` — or if the address is now owned by a newer
                allocation, so a stale handle cannot free the replacement.
        """
        key = (worker_id, t.data_ptr)
        if key not in self._owned_tensors:
            # A missing exact key is only idempotent if this ptr is no longer
            # owned at all. If it is still tracked under another worker_id, the
            # caller passed the wrong worker — surface it rather than leak.
            owners = sorted(w for w, ptr in self._owned_tensors if ptr == t.data_ptr)
            if owners:
                raise ValueError(
                    f"free_tensor(..., worker_id={worker_id}) does not match the owning worker for "
                    f"ptr=0x{t.data_ptr:x} (allocated on worker(s) {owners}); pass the same worker_id."
                )
            return
        if self._owned_tensors[key] is not t:
            raise ValueError(
                f"free_tensor(..., worker_id={worker_id}) received a stale DeviceTensor for "
                f"ptr=0x{t.data_ptr:x}; that address now belongs to a newer allocation."
            )
        self.free(t.data_ptr, worker_id=worker_id)
        # Preserve ownership across a backend admission/free failure so callers
        # can retry. A successful first free removes the key and keeps the
        # genuine second-free path idempotent.
        del self._owned_tensors[key]
        del self._owned_tensor_keys[id(t)]

    def _close_owned_tensors(self) -> None:
        """Release any DeviceTensors the caller forgot to :meth:`free_tensor`.

        Subclasses MUST call this from their ``close()`` **before** tearing
        down the backend (the underlying ``free`` needs the backend live).
        Errors raised by ``free`` are logged and swallowed so ``close()`` is
        guaranteed to complete.
        """
        # Snapshot then clear so subsequent free_tensor calls (or a retry on
        # close()) don't double-iterate.
        leaked = self._owned_tensors
        self._owned_tensors = {}
        self._owned_tensor_keys.clear()
        for worker_id, ptr in leaked:
            try:
                self.free(ptr, worker_id=worker_id)
            except Exception:
                _log.warning(
                    "%s._close_owned_tensors: free(worker_id=%d, ptr=0x%x) failed; leaking",
                    type(self).__name__,
                    worker_id,
                    ptr,
                    exc_info=True,
                )
