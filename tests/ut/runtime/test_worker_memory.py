# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``ChipWorker`` device-memory primitives and ``alloc_tensor``.

Patches ``_SimplerWorker`` so tests run without a device.  Each test asserts
that PyPTO's stable pointer API is translated to simpler's owner ``Buffer`` API.
"""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import ChipWorker, DeviceTensor, RunConfig


class FakeBuffer:
    def __init__(self, base: int, nbytes: int = 4096) -> None:
        self.base = base
        self.nbytes = nbytes

    def tensor(self, shapes, dtype):
        return shapes, dtype


@pytest.fixture(autouse=True)
def fake_tensor_arg_modules():
    """Keep DeviceTensor wire-conversion checks independent of Simpler."""
    from pypto.runtime import device_tensor as device_tensor_module  # noqa: PLC0415

    task_interface = SimpleNamespace(
        Tensor=type("Tensor", (), {}),
        torch_dtype_to_datatype=lambda dtype: dtype,
    )
    torch_interop = SimpleNamespace(make_tensor_arg=MagicMock(name="make_tensor_arg"))
    with patch(
        "pypto.runtime.tensor_arg._modules",
        return_value=(task_interface, device_tensor_module, torch_interop),
    ):
        yield


@pytest.fixture
def fake_simpler_worker():
    """Patch ``simpler.worker.Worker`` so ChipWorker construction does not touch a device."""
    with (
        patch("pypto.runtime.worker._SimplerWorker") as cls,
        # init() builds a prewarm CallConfig; patch the cache so no simpler import happens.
        patch("pypto.runtime.worker._SimplerCallConfig", MagicMock()),
    ):
        instance = MagicMock()
        cls.return_value = instance
        yield instance


@pytest.fixture
def worker(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    yield w
    if w.initialized:
        w.close()


class TestMallocFree:
    def test_malloc_forwards_with_default_worker_id(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x4000)
        fake_simpler_worker.malloc.return_value = buffer
        ptr = worker.malloc(1024)
        assert ptr == 0x4000
        fake_simpler_worker.malloc.assert_called_once_with(1024)
        assert worker._buffer_for_ptr(ptr) is buffer

    def test_malloc_rejects_nonzero_worker_id(self, fake_simpler_worker, worker):
        with pytest.raises(ValueError, match="only supports worker_id=0"):
            worker.malloc(2048, worker_id=3)
        fake_simpler_worker.malloc.assert_not_called()

    def test_malloc_zero_raises(self, worker):
        with pytest.raises(ValueError, match="positive int"):
            worker.malloc(0)

    def test_malloc_negative_raises(self, worker):
        with pytest.raises(ValueError, match="positive int"):
            worker.malloc(-1)

    def test_invalid_backend_buffer_is_released(self, fake_simpler_worker, worker):
        invalid = object()
        fake_simpler_worker.malloc.return_value = invalid
        with pytest.raises(TypeError, match="must return a Buffer"):
            worker.malloc(1024)
        fake_simpler_worker.free.assert_called_once_with(invalid)

    def test_malloc_after_close_raises(self, fake_simpler_worker, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized ChipWorker"):
            worker.malloc(1024)
        fake_simpler_worker.malloc.assert_not_called()

    def test_free_forwards(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x4000)
        fake_simpler_worker.malloc.return_value = buffer
        ptr = worker.malloc(1024)
        worker.free(ptr)
        fake_simpler_worker.free.assert_called_once_with(buffer)

    def test_free_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized ChipWorker"):
            worker.free(0x4000)

    def test_failed_backend_free_can_be_retried(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x4000)
        fake_simpler_worker.malloc.return_value = buffer
        ptr = worker.malloc(1024)
        fake_simpler_worker.free.side_effect = RuntimeError("native free failed")
        with pytest.raises(RuntimeError, match="native free failed"):
            worker.free(ptr)
        fake_simpler_worker.free.side_effect = None
        worker.free(ptr)
        assert fake_simpler_worker.free.call_count == 2


class TestCopy:
    def test_interior_device_pointer_is_rejected_explicitly(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = FakeBuffer(0x100)
        ptr = worker.malloc(64)
        host = (ctypes.c_ubyte * 32)()
        with pytest.raises(ValueError, match="interior pointer"):
            worker.copy_to(ptr + 32, ctypes.addressof(host), 32)

    def test_copy_to_forwards(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x100)
        fake_simpler_worker.malloc.return_value = buffer
        ptr = worker.malloc(64)
        host = (ctypes.c_ubyte * 64)()
        worker.copy_to(ptr, ctypes.addressof(host), 64)
        args = fake_simpler_worker.copy_to.call_args.args
        assert args[0] is buffer and len(args[1]) == 64

    def test_copy_from_forwards(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x200)
        fake_simpler_worker.malloc.return_value = buffer
        ptr = worker.malloc(64)
        host = (ctypes.c_ubyte * 64)()
        worker.copy_from(ctypes.addressof(host), ptr, 64)
        args = fake_simpler_worker.copy_from.call_args.args
        assert len(args[0]) == 64 and args[1] is buffer

    def test_copy_to_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized ChipWorker"):
            worker.copy_to(0x100, 0x200, 64)

    def test_copy_from_after_close_raises(self, worker):
        worker.close()
        with pytest.raises(RuntimeError, match="initialized ChipWorker"):
            worker.copy_from(0x100, 0x200, 64)


class TestAllocTensor:
    def test_alloc_no_init(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x9000)
        fake_simpler_worker.malloc.return_value = buffer
        t = worker.alloc_tensor((4, 8), torch.float32)
        assert isinstance(t, DeviceTensor)
        assert t.data_ptr == 0x9000
        assert t.shape == (4, 8)
        assert t.dtype is torch.float32
        assert t.nbytes == 4 * 8 * 4
        assert t.buffer is buffer
        fake_simpler_worker.malloc.assert_called_once_with(4 * 8 * 4)
        fake_simpler_worker.copy_to.assert_not_called()

    def test_live_tensor_wire_conversion_uses_owning_worker(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x9100)
        fake_simpler_worker.malloc.return_value = buffer
        tensor = worker.alloc_tensor((4, 8), torch.float32)

        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        wire = make_tensor_arg(fake_simpler_worker, tensor)
        assert wire[0] == (4, 8)

    def test_foreign_tensor_is_rejected_before_wire_conversion(self, fake_simpler_worker, worker):
        foreign_buffer = FakeBuffer(0x9200)
        foreign = DeviceTensor.from_buffer(foreign_buffer, (4,), torch.float32)

        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        with pytest.raises(ValueError, match="not a live allocation owned by this ChipWorker"):
            make_tensor_arg(fake_simpler_worker, foreign)

    def test_freed_tensor_is_rejected_before_wire_conversion(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x9300)
        fake_simpler_worker.malloc.return_value = buffer
        tensor = worker.alloc_tensor((4,), torch.float32)
        worker.free_tensor(tensor)

        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        with pytest.raises(ValueError, match="not a live allocation owned by this ChipWorker"):
            make_tensor_arg(fake_simpler_worker, tensor)

    def test_pointer_reuse_does_not_revive_old_tensor(self, fake_simpler_worker, worker):
        old_buffer = FakeBuffer(0x9400)
        new_buffer = FakeBuffer(0x9400)
        fake_simpler_worker.malloc.side_effect = [old_buffer, new_buffer]
        old_tensor = worker.alloc_tensor((4,), torch.float32)
        worker.free_tensor(old_tensor)
        new_tensor = worker.alloc_tensor((4,), torch.float32)

        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        with pytest.raises(ValueError, match="not a live allocation owned by this ChipWorker"):
            make_tensor_arg(fake_simpler_worker, old_tensor)
        assert make_tensor_arg(fake_simpler_worker, new_tensor)[0] == (4,)

        fake_simpler_worker.free.reset_mock()
        with pytest.raises(ValueError, match="stale DeviceTensor"):
            worker.free_tensor(old_tensor)
        fake_simpler_worker.free.assert_not_called()
        assert worker._buffer_for_ptr(new_tensor.data_ptr) is new_buffer

        worker.free_tensor(new_tensor)
        fake_simpler_worker.free.assert_called_once_with(new_buffer)

    def test_close_reinit_rejects_old_raw_backend(self):
        old_impl = MagicMock(name="old_impl")
        new_impl = MagicMock(name="new_impl")
        old_impl.malloc.return_value = FakeBuffer(0x9500)
        with (
            patch("pypto.runtime.worker._SimplerWorker", side_effect=[old_impl, new_impl]),
            patch("pypto.runtime.worker._SimplerCallConfig", MagicMock()),
        ):
            worker = ChipWorker(config=RunConfig(platform="a2a3sim"))
            tensor = worker.alloc_tensor((4,), torch.float32)
            worker.close()
            worker.init()

            from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

            with pytest.raises(ValueError, match="stale simpler Worker backend"):
                make_tensor_arg(old_impl, tensor)
            assert new_impl.__dict__["_pypto_tensor_owner_ref"]() is worker
            worker.close()

    def test_alloc_with_init_uploads(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x9000)
        fake_simpler_worker.malloc.return_value = buffer
        host = torch.full((4, 8), 1.5, dtype=torch.float32)
        t = worker.alloc_tensor((4, 8), torch.float32, init=host)
        assert t.data_ptr == 0x9000
        fake_simpler_worker.copy_to.assert_called_once()
        call = fake_simpler_worker.copy_to.call_args
        assert call.args[0] is buffer
        assert len(call.args[1]) == 4 * 8 * 4

    def test_alloc_init_shape_mismatch_frees_and_raises(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x9000)
        fake_simpler_worker.malloc.return_value = buffer
        bad = torch.zeros((4, 4), dtype=torch.float32)
        with pytest.raises(ValueError, match="must have shape"):
            worker.alloc_tensor((4, 8), torch.float32, init=bad)
        fake_simpler_worker.free.assert_called_once_with(buffer)
        fake_simpler_worker.copy_to.assert_not_called()

    def test_alloc_init_dtype_mismatch_frees_and_raises(self, fake_simpler_worker, worker):
        buffer = FakeBuffer(0x9000)
        fake_simpler_worker.malloc.return_value = buffer
        bad = torch.zeros((4, 8), dtype=torch.float16)
        with pytest.raises(ValueError, match="must have shape"):
            worker.alloc_tensor((4, 8), torch.float32, init=bad)
        fake_simpler_worker.free.assert_called_once_with(buffer)

    def test_free_tensor_uses_data_ptr(self, fake_simpler_worker, worker):
        # ``free_tensor`` is the dual of ``alloc_tensor``; only tensors the
        # Worker actually allocated are tracked (and therefore freed). Going
        # through alloc_tensor puts the ptr in ``_owned_tensors`` so the
        # subsequent free_tensor forwards through to the underlying ``free``.
        buffer = FakeBuffer(0x9000)
        fake_simpler_worker.malloc.return_value = buffer
        t = worker.alloc_tensor((4, 8), torch.float32)
        fake_simpler_worker.free.reset_mock()
        worker.free_tensor(t)
        fake_simpler_worker.free.assert_called_once_with(buffer)

    def test_alloc_makes_non_contiguous_init_contiguous(self, fake_simpler_worker, worker):
        fake_simpler_worker.malloc.return_value = FakeBuffer(0x9000)
        # transpose makes it non-contiguous; .contiguous() inside alloc_tensor must fix it.
        host = torch.zeros((8, 4), dtype=torch.float32).t()
        assert tuple(host.shape) == (4, 8)
        worker.alloc_tensor((4, 8), torch.float32, init=host)
        fake_simpler_worker.copy_to.assert_called_once()

    def test_alloc_non_positive_dim_rejected_before_malloc(self, fake_simpler_worker, worker):
        # The shape contract (mirroring DeviceTensor) requires positive int dims.
        # Negative and zero dims must be rejected before any allocation happens —
        # a zero dim would otherwise compute nbytes as 0 (empty shape -> nbytes 1).
        with pytest.raises(ValueError, match="positive"):
            worker.alloc_tensor((-1, 4), torch.float32)
        with pytest.raises(ValueError, match="positive"):
            worker.alloc_tensor((0, 4), torch.float32)
        fake_simpler_worker.malloc.assert_not_called()

    def test_alloc_empty_shape_rejected_before_malloc(self, fake_simpler_worker, worker):
        # An empty shape would make n_elems collapse to 1 and malloc a bogus buffer.
        with pytest.raises(ValueError, match="non-empty"):
            worker.alloc_tensor((), torch.float32)
        fake_simpler_worker.malloc.assert_not_called()

    def test_alloc_copy_to_failure_frees_pointer(self, fake_simpler_worker, worker):
        # Simulate a runtime copy_to failure (hardware error, etc.).  The pointer
        # malloc'd up-front must be freed before the exception propagates so the
        # device buffer is not leaked.
        fake_simpler_worker.malloc.return_value = FakeBuffer(0x9000)
        fake_simpler_worker.copy_to.side_effect = RuntimeError("copy failed")
        host = torch.zeros((4, 8), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="copy failed"):
            worker.alloc_tensor((4, 8), torch.float32, init=host)
        assert fake_simpler_worker.free.call_args.args[0].base == 0x9000

    def test_alloc_rejects_non_zero_worker_id(self, fake_simpler_worker, worker):
        with pytest.raises(ValueError, match="only supports worker_id=0"):
            worker.alloc_tensor((4, 8), torch.float32, worker_id=3)
        fake_simpler_worker.malloc.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
