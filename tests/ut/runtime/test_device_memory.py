# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the shared ``DeviceMemoryHandle`` ABC.

These exercise the convenience surface (``alloc_tensor`` / ``free_tensor``) and
the two subclass hooks (``_require_ready`` / ``_prepare_init``) with an in-test
dict-backed subclass — no ``simpler`` package or device required.
"""

import pytest
import torch
from pypto.runtime import DeviceMemoryHandle, DeviceTensor, Worker


class FakeHandle(DeviceMemoryHandle):
    """Dict-backed handle that records every primitive call."""

    _WORKER_KIND = "fake worker"

    def __init__(self) -> None:
        self.ready = True
        self._next_ptr = 0x1000
        self.live: dict[int, int] = {}  # ptr -> nbytes
        self.uploads: list[tuple[int, int, int]] = []
        self.freed: list[int] = []
        self.fail_copy = False

    def malloc(self, nbytes: int, *, worker_id: int = 0) -> int:
        ptr = self._next_ptr
        self._next_ptr += 0x1000
        self.live[ptr] = nbytes
        return ptr

    def free(self, ptr: int, *, worker_id: int = 0) -> None:
        self.freed.append(ptr)
        self.live.pop(ptr, None)

    def copy_to(self, dst_dev_ptr: int, src_host_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        if self.fail_copy:
            raise RuntimeError("simulated copy failure")
        self.uploads.append((dst_dev_ptr, src_host_ptr, nbytes))

    def copy_from(self, dst_host_ptr: int, src_dev_ptr: int, nbytes: int, *, worker_id: int = 0) -> None:
        pass

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


def test_worker_id_nonzero_rejected_with_kind_noun():
    h = FakeHandle()
    with pytest.raises(ValueError, match="fake worker"):
        h.alloc_tensor((4,), torch.float32, worker_id=1)
    t = DeviceTensor(0x1000, (4,), torch.float32)
    with pytest.raises(ValueError, match="fake worker"):
        h.free_tensor(t, worker_id=1)


def test_require_ready_consulted_by_alloc_tensor():
    h = FakeHandle()
    h.ready = False
    with pytest.raises(RuntimeError, match="alloc_tensor.. not ready"):
        h.alloc_tensor((4,), torch.float32)


def test_worker_subclasses_handle():
    assert issubclass(Worker, DeviceMemoryHandle)


def test_distributed_runtime_subclasses_handle():
    drm = pytest.importorskip("pypto.runtime.distributed_runner")
    assert issubclass(drm.DistributedRuntime, DeviceMemoryHandle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
