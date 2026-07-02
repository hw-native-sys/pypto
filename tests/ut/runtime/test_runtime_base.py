# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the shared :class:`Worker` ABC (formerly ``DeviceMemoryHandle``).

These exercise the convenience surface (``alloc_tensor`` / ``free_tensor``) and
the two subclass hooks (``_require_ready`` / ``_prepare_init``) with an in-test
dict-backed subclass — no ``simpler`` package or device required.
"""

import pytest
import torch
from pypto.runtime import ChipWorker, DeviceTensor, Worker


class FakeHandle(Worker):
    """Dict-backed Worker that records every primitive call.

    Used to exercise the ABC's concrete methods (alloc_tensor / free_tensor /
    _close_owned_tensors) without instantiating a real ``ChipWorker``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ready = True
        self._next_ptr = 0x1000
        self.live: dict[int, int] = {}  # ptr -> nbytes
        self.uploads: list[tuple[int, int, int]] = []
        self.freed: list[int] = []
        self.free_calls: list[tuple[int, int]] = []  # (ptr, worker_id)
        self.fail_copy = False

    def malloc(self, nbytes: int, *, worker_id: int = 0) -> int:
        ptr = self._next_ptr
        self._next_ptr += 0x1000
        self.live[ptr] = nbytes
        return ptr

    def free(self, ptr: int, *, worker_id: int = 0) -> None:
        self.freed.append(ptr)
        self.free_calls.append((ptr, worker_id))
        self.live.pop(ptr, None)

    def copy_to(self, dst_dev_ptr: int, src_host_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        if self.fail_copy:
            raise RuntimeError("simulated copy failure")
        self.uploads.append((dst_dev_ptr, src_host_ptr, nbytes))

    def copy_from(self, dst_host_ptr: int, src_dev_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        pass

    # Abstract dispatch methods — stubbed; not exercised by these tests.
    def run(self, compiled, *args, config=None):
        raise NotImplementedError("FakeHandle does not implement dispatch")

    def register(self, compiled):
        raise NotImplementedError("FakeHandle does not implement dispatch")

    def _require_ready(self, op: str) -> None:
        if not self.ready:
            raise RuntimeError(f"FakeHandle.{op}() not ready")


def test_alloc_tensor_without_init_allocates_only():
    h = FakeHandle()
    t = h.alloc_tensor((4,), torch.float32)
    assert isinstance(t, DeviceTensor)
    assert t.shape == (4,) and t.dtype == torch.float32
    assert h.live == {t.data_ptr: 16}  # 4 * float32(4 bytes)
    assert h.uploads == []  # no init -> no H2D copy


def test_alloc_tensor_with_init_uploads():
    h = FakeHandle()
    init = torch.ones(4, dtype=torch.float32)
    t = h.alloc_tensor((4,), torch.float32, init=init)
    assert len(h.uploads) == 1
    dst, _src, nbytes = h.uploads[0]
    assert dst == t.data_ptr and nbytes == 16


def test_alloc_tensor_rollback_on_copy_failure():
    h = FakeHandle()
    h.fail_copy = True
    with pytest.raises(RuntimeError, match="simulated copy failure"):
        h.alloc_tensor((4,), torch.float32, init=torch.ones(4, dtype=torch.float32))
    # The buffer malloc'd before the failed copy must be freed (no leak).
    assert h.freed == [0x1000]
    assert h.live == {}


def test_alloc_tensor_tracks_in_owned_set():
    h = FakeHandle()
    t = h.alloc_tensor((4,), torch.float32)
    # Tracking is keyed by (worker_id, data_ptr); default worker_id is 0.
    assert (0, t.data_ptr) in h._owned_tensors


def test_free_tensor_untracks_and_frees():
    h = FakeHandle()
    t = h.alloc_tensor((4,), torch.float32)
    h.free_tensor(t)
    assert (0, t.data_ptr) not in h._owned_tensors
    assert h.freed == [t.data_ptr]


def test_close_owned_tensors_frees_leaked():
    h = FakeHandle()
    a = h.alloc_tensor((4,), torch.float32)
    b = h.alloc_tensor((8,), torch.float32)
    h.free_tensor(a)  # released early
    # Only b is leaked; _close_owned_tensors should free it.
    h._close_owned_tensors()
    assert b.data_ptr in h.freed
    assert h._owned_tensors == set()


def test_close_owned_tensors_swallows_free_errors():
    h = FakeHandle()
    h.alloc_tensor((4,), torch.float32)

    def failing_free(_ptr, *, worker_id=0):
        raise RuntimeError("backend torn down")

    h.free = failing_free  # type: ignore[method-assign]
    # Must NOT raise; the failure is logged-and-swallowed so close() can complete.
    h._close_owned_tensors()
    assert h._owned_tensors == set()


def test_default_prepare_init_makes_contiguous_cpu_copy():
    h = FakeHandle()
    src = torch.ones(4, 4, dtype=torch.float32).t()  # non-contiguous view
    prepared = h._prepare_init(src)
    assert prepared.is_contiguous() and prepared.device.type == "cpu"


def test_prepare_init_override_is_honored():
    calls: list[torch.Tensor] = []

    class TaggingHandle(FakeHandle):
        def _prepare_init(self, init: torch.Tensor) -> torch.Tensor:
            calls.append(init)
            return init

    h = TaggingHandle()
    init = torch.ones(4, dtype=torch.float32)
    h.alloc_tensor((4,), torch.float32, init=init)
    assert calls == [init]  # override invoked; no defensive copy made


def test_free_tensor_forwards_data_ptr():
    h = FakeHandle()
    t = h.alloc_tensor((4,), torch.float32)
    h.free_tensor(t)
    assert h.freed == [t.data_ptr]


def test_nonzero_worker_id_allocates_tracks_and_frees_against_that_worker():
    """alloc_tensor / free_tensor support a non-default worker_id, tracking and
    freeing the buffer against the worker it was allocated on."""
    h = FakeHandle()
    t = h.alloc_tensor((4,), torch.float32, worker_id=3)
    # Tracked under (worker_id, ptr) — NOT the default-worker key.
    assert (3, t.data_ptr) in h._owned_tensors
    assert (0, t.data_ptr) not in h._owned_tensors

    h.free_tensor(t, worker_id=3)
    assert (3, t.data_ptr) not in h._owned_tensors
    # free was forwarded to worker 3, not worker 0.
    assert (t.data_ptr, 3) in h.free_calls


def test_free_tensor_wrong_worker_id_raises_instead_of_leaking():
    """Freeing a still-owned ptr with the wrong worker_id surfaces the contract
    bug rather than silently no-oping (which would leak until close())."""
    h = FakeHandle()
    t = h.alloc_tensor((4,), torch.float32, worker_id=3)
    with pytest.raises(ValueError, match="does not match the owning worker"):
        h.free_tensor(t, worker_id=0)  # wrong worker id
    # Still tracked (not leaked-and-forgotten) and never freed.
    assert (3, t.data_ptr) in h._owned_tensors
    assert h.freed == []
    # The correct worker_id still frees it.
    h.free_tensor(t, worker_id=3)
    assert h.freed == [t.data_ptr]


def test_free_tensor_genuine_double_free_is_still_noop():
    """A second free of a fully-released ptr is an idempotent no-op (no raise)."""
    h = FakeHandle()
    t = h.alloc_tensor((4,), torch.float32, worker_id=2)
    h.free_tensor(t, worker_id=2)
    h.free_tensor(t, worker_id=2)  # ptr no longer owned under any worker -> no-op
    assert h.freed == [t.data_ptr]  # freed exactly once


def test_close_frees_multi_worker_tensors_against_their_workers():
    """_close_owned_tensors releases each leaked buffer against its own worker."""
    h = FakeHandle()
    a = h.alloc_tensor((4,), torch.float32, worker_id=0)
    b = h.alloc_tensor((4,), torch.float32, worker_id=2)
    h._close_owned_tensors()
    assert h._owned_tensors == set()
    assert (a.data_ptr, 0) in h.free_calls
    assert (b.data_ptr, 2) in h.free_calls


def test_require_ready_consulted_by_alloc_tensor():
    h = FakeHandle()
    h.ready = False
    with pytest.raises(RuntimeError, match="alloc_tensor.. not ready"):
        h.alloc_tensor((4,), torch.float32)


def test_chip_worker_subclasses_worker_abc():
    assert issubclass(ChipWorker, Worker)


def test_distributed_worker_subclasses_worker_abc():
    drm = pytest.importorskip("pypto.runtime.distributed_runner")
    assert issubclass(drm.DistributedWorker, Worker)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
