# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``Worker`` device-memory primitives and ``alloc_tensor``.

Patches ``_SimplerWorker`` so tests run without a device.  Each test asserts
that the call is forwarded to the underlying chip worker with the expected
arguments (positional + ``worker_id`` trailing arg).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import DeviceTensor, RunConfig, Worker


@pytest.fixture
def fake_simpler_worker():
    """Patch ``simpler.worker.Worker`` so Worker construction does not touch a device."""
    with patch("pypto.runtime.worker._SimplerWorker") as cls:
        instance = MagicMock()
        cls.return_value = instance
        yield instance


@pytest.fixture
def worker(fake_simpler_worker):
    w = Worker(config=RunConfig(platform="a2a3sim"))
    yield w
    if w.initialized:
        w.close()


class TestMallocFree:
    def test_malloc_forwards_with_default_worker_id(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = 0x4000
        ptr = worker.malloc(1024)
        assert ptr == 0x4000
        fake_simpler_worker.malloc.assert_called_once_with(1024, 0)

    def test_malloc_forwards_explicit_worker_id(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = 0x5000
        worker.malloc(2048, worker_id=3)
        fake_simpler_worker.malloc.assert_called_once_with(2048, 3)

    def test_malloc_zero_raises(self, worker):
        with pytest.raises(ValueError, match="positive int"):
            worker.malloc(0)

    def test_malloc_negative_raises(self, worker):
        with pytest.raises(ValueError, match="positive int"):
            worker.malloc(-1)

    def test_malloc_after_close_raises(self, fake_simpler_worker, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized Worker"):
            worker.malloc(1024)
        fake_simpler_worker.malloc.assert_not_called()

    def test_free_forwards(self, fake_simpler_worker, worker):
        worker.free(0x4000)
        fake_simpler_worker.free.assert_called_once_with(0x4000, 0)

    def test_free_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized Worker"):
            worker.free(0x4000)


class TestCopy:
    def test_copy_to_forwards(self, fake_simpler_worker, worker):
        worker.copy_to(0x100, 0x200, 64)
        fake_simpler_worker.copy_to.assert_called_once_with(0x100, 0x200, 64, 0)

    def test_copy_from_forwards(self, fake_simpler_worker, worker):
        worker.copy_from(0x100, 0x200, 64, worker_id=2)
        fake_simpler_worker.copy_from.assert_called_once_with(0x100, 0x200, 64, 2)

    def test_copy_to_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized Worker"):
            worker.copy_to(0x100, 0x200, 64)

    def test_copy_from_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized Worker"):
            worker.copy_from(0x100, 0x200, 64)


class TestAllocTensor:
    def test_alloc_no_init(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = 0x9000
        t = worker.alloc_tensor((4, 8), torch.float32)
        assert isinstance(t, DeviceTensor)
        assert t.data_ptr == 0x9000
        assert t.shape == (4, 8)
        assert t.dtype is torch.float32
        assert t.nbytes == 4 * 8 * 4
        fake_simpler_worker.malloc.assert_called_once_with(4 * 8 * 4, 0)
        fake_simpler_worker.copy_to.assert_not_called()

    def test_alloc_with_init_uploads(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = 0x9000
        host = torch.full((4, 8), 1.5, dtype=torch.float32)
        t = worker.alloc_tensor((4, 8), torch.float32, init=host)
        assert t.data_ptr == 0x9000
        # nbytes is the third positional arg
        fake_simpler_worker.copy_to.assert_called_once()
        call = fake_simpler_worker.copy_to.call_args
        assert call.args[0] == 0x9000
        assert call.args[2] == 4 * 8 * 4
        assert call.args[3] == 0  # default worker_id

    def test_alloc_init_shape_mismatch_frees_and_raises(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = 0x9000
        bad = torch.zeros((4, 4), dtype=torch.float32)
        with pytest.raises(ValueError, match="must have shape"):
            worker.alloc_tensor((4, 8), torch.float32, init=bad)
        fake_simpler_worker.free.assert_called_once_with(0x9000, 0)
        fake_simpler_worker.copy_to.assert_not_called()

    def test_alloc_init_dtype_mismatch_frees_and_raises(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = 0x9000
        bad = torch.zeros((4, 8), dtype=torch.float16)
        with pytest.raises(ValueError, match="must have shape"):
            worker.alloc_tensor((4, 8), torch.float32, init=bad)
        fake_simpler_worker.free.assert_called_once_with(0x9000, 0)

    def test_free_tensor_uses_data_ptr(self, fake_simpler_worker, worker):
        t = DeviceTensor(0x9000, (4, 8), torch.float32)
        worker.free_tensor(t)
        fake_simpler_worker.free.assert_called_once_with(0x9000, 0)

    def test_alloc_makes_non_contiguous_init_contiguous(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = 0x9000
        # transpose makes it non-contiguous; .contiguous() inside alloc_tensor must fix it.
        host = torch.zeros((8, 4), dtype=torch.float32).t()
        assert tuple(host.shape) == (4, 8)
        worker.alloc_tensor((4, 8), torch.float32, init=host)
        fake_simpler_worker.copy_to.assert_called_once()

    def test_alloc_negative_dim_rejected_before_malloc(self, fake_simpler_worker, worker):
        # Negative dims would silently produce a positive nbytes via tuple(int(d) for ...).
        # The pre-malloc validation must reject them before any allocation happens.
        with pytest.raises(ValueError, match="non-negative"):
            worker.alloc_tensor((-1, 4), torch.float32)
        fake_simpler_worker.malloc.assert_not_called()

    def test_alloc_copy_to_failure_frees_pointer(self, fake_simpler_worker, worker):
        # Simulate a runtime copy_to failure (hardware error, etc.).  The pointer
        # malloc'd up-front must be freed before the exception propagates so the
        # device buffer is not leaked.
        fake_simpler_worker.malloc.return_value = 0x9000
        fake_simpler_worker.copy_to.side_effect = RuntimeError("copy failed")
        host = torch.zeros((4, 8), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="copy failed"):
            worker.alloc_tensor((4, 8), torch.float32, init=host)
        fake_simpler_worker.free.assert_called_once_with(0x9000, 0)

    def test_alloc_rejects_non_zero_worker_id(self, fake_simpler_worker, worker):
        # DeviceTensor doesn't encode worker_id yet, so allowing alloc on a
        # different worker would produce a handle that free_tensor (default
        # worker_id=0) would mis-route.  Fail loud at the API boundary.
        with pytest.raises(ValueError, match="worker_id=0"):
            worker.alloc_tensor((4, 8), torch.float32, worker_id=3)
        fake_simpler_worker.malloc.assert_not_called()

    def test_free_tensor_rejects_non_zero_worker_id(self, fake_simpler_worker, worker):
        t = DeviceTensor(0x9000, (4, 8), torch.float32)
        with pytest.raises(ValueError, match="worker_id=0"):
            worker.free_tensor(t, worker_id=3)
        fake_simpler_worker.free.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
