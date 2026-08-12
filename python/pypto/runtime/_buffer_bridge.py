# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared plumbing for naming device memory to Simpler's Buffer-based APIs.

pypto's public surface hands out integer device pointers (``DeviceTensor.data_ptr``),
while Simpler's device-memory and task-arg APIs take ``Buffer`` objects. Both worker
levels therefore need the same three things, and they need to agree:

* a registry from ``(worker_id, ptr)`` back to the allocating ``Buffer`` — the pointer
  alone cannot be re-wrapped, because a Buffer carries the identity a consumer keys on;
* a POSIX-shm staging Buffer for host memory that a copy cannot name directly;
* one conversion from :class:`DeviceTensor` to a wire ``Tensor``.

L2 and L3 differ only in *how* they allocate (``malloc`` on a leaf worker vs
``alloc_child_tensor`` on a chip worker), so only that part stays in each subclass.
Keeping the rest here is what stops the two levels from drifting apart.
"""

from typing import Any


class BufferBridge:
    """Mixin: Buffer registry, shm staging, and DeviceTensor → wire ``Tensor``."""

    def _init_buffer_bridge(self) -> None:
        self._device_buffers: dict[tuple[int, int], Any] = {}
        self._staging_buffers: dict[tuple[int, int], Any] = {}

    # ------------------------------------------------------------------
    # Device-buffer registry
    # ------------------------------------------------------------------

    def _register_device_buffer(self, worker_id: int, ptr: int, handle: Any) -> None:
        self._device_buffers[(int(worker_id), int(ptr))] = handle

    def _device_buffer_for(self, ptr: int, worker_id: int, api: str) -> Any:
        handle = self._device_buffers.get((int(worker_id), int(ptr)))
        if handle is None:
            raise ValueError(f"{api}: 0x{int(ptr):x} was not allocated on worker {int(worker_id)} by this Worker")
        return handle

    def _pop_device_buffer(self, ptr: int, worker_id: int, api: str) -> Any:
        handle = self._device_buffers.pop((int(worker_id), int(ptr)), None)
        if handle is None:
            raise ValueError(f"{api}: 0x{int(ptr):x} was not allocated on worker {int(worker_id)} by this Worker")
        return handle

    # ------------------------------------------------------------------
    # Host staging
    # ------------------------------------------------------------------

    def _staging_buffer(self, worker_id: int, nbytes: int) -> Any:
        """A worker-owned POSIX-shm Buffer used to move host bytes a copy cannot name.

        Host memory allocated after the chip children forked cannot be named with
        ``wrap_fork_inherited`` at all — the child has no mapping for it. ``create_buffer``
        gives shared memory whose descriptor travels with the Tensor and which the
        consumer materializes on first receipt, so one host memcpy buys reachability.

        Cached per ``(worker_id, nbytes)``: sizes repeat every step, and keying on the
        worker too keeps two chips' concurrent copies off one another's staging area.
        Reuse is safe because the copy completes before the call returns.
        """
        import os  # noqa: PLC0415

        if os.environ.get("PYPTO_STAGING_FRESH") == "1":
            # Diagnostic escape hatch: one Buffer per copy, never reused. Reuse is only
            # safe if the copy has fully drained by the time the call returns; if a
            # pipelined run still references the staging area, reuse is a data race and
            # this flag makes that visible instead of leaving it to inference.
            return self._simpler_worker().create_buffer(int(nbytes))
        key = (int(worker_id), int(nbytes))
        buffer = self._staging_buffers.get(key)
        if buffer is None:
            buffer = self._simpler_worker().create_buffer(int(nbytes))
            self._staging_buffers[key] = buffer
        return buffer

    @staticmethod
    def _staging_view(buffer: Any, nbytes: int) -> Any:
        import ctypes  # noqa: PLC0415

        return (ctypes.c_char * int(nbytes)).from_buffer(buffer.shm.buf)

    def _stage_host_to_buffer(self, worker_id: int, host_ptr: int, nbytes: int) -> Any:
        import ctypes  # noqa: PLC0415

        staging = self._staging_buffer(worker_id, nbytes)
        ctypes.memmove(self._staging_view(staging, nbytes), int(host_ptr), int(nbytes))
        return staging

    def _stage_buffer_to_host(self, staging: Any, host_ptr: int, nbytes: int) -> None:
        import ctypes  # noqa: PLC0415

        ctypes.memmove(int(host_ptr), self._staging_view(staging, nbytes), int(nbytes))

    def _release_staging_buffers(self) -> None:
        """Close the shm staging Buffers while the worker is still alive.

        Best-effort: one that refuses to close must not stop the rest of teardown, and
        there is nothing the caller could do about it anyway.
        """
        for buffer in self._staging_buffers.values():
            try:
                buffer.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
        self._staging_buffers.clear()

    # ------------------------------------------------------------------
    # Task args
    # ------------------------------------------------------------------

    def device_tensor_arg(self, dt: Any) -> Any:
        """Name a worker-resident :class:`DeviceTensor` as a Simpler wire ``Tensor``.

        Replaces the retired ``Tensor.make(data=ptr, child_memory=True)``: ``make`` now
        lives on ``ChipTensor``, and the wire ``Tensor`` is a view over a Buffer rather
        than a raw address. The view comes from the Buffer that allocated the pointer,
        which is also what makes "already on the device, skip H2D" structural instead of
        a flag the caller has to remember to pass.
        """
        from .task_interface import torch_dtype_to_datatype  # noqa: PLC0415

        if getattr(dt, "worker_id", None) is None:
            raise ValueError(
                "DeviceTensor reached a task-arg conversion without a worker_id, so the chip that "
                "owns its pointer is unknown. Allocate it through alloc_tensor/alloc_stacked_tensor, "
                "which record the owner."
            )
        buffer = self._device_buffer_for(int(dt.data_ptr), int(dt.worker_id), "device_tensor_arg")
        try:
            dtype = torch_dtype_to_datatype(dt.dtype)
        except KeyError as exc:
            raise ValueError(f"Unsupported DeviceTensor dtype: {dt.dtype}") from exc
        return buffer.tensor(tuple(int(d) for d in dt.shape), dtype)

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _simpler_worker(self) -> Any:
        """The underlying Simpler ``Worker``. Subclasses hold it under different names."""
        raise NotImplementedError
