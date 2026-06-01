# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``Worker`` ABC's DeviceTensor lifecycle tracking.

Exercises ``_owned_tensors`` bookkeeping and ``_close_owned_tensors`` end-to-end
via a real :class:`ChipWorker` (with ``simpler.Worker`` mocked out). These
tests verify that the ABC's auto-track / auto-free path actually fires from
the concrete subclass's ``close()``.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import ChipWorker, RunConfig


@pytest.fixture
def fake_simpler_worker():
    """Patch ``simpler.worker.Worker`` so ChipWorker construction does no I/O."""
    with patch("pypto.runtime.worker._SimplerWorker") as cls:
        instance = MagicMock()
        # Deterministic malloc: incrementing pointer.
        ptr_state = {"next": 0x1000}

        def fake_malloc(nbytes, _wid):
            ptr_state["next"] += 0x1000
            return ptr_state["next"]

        instance.malloc.side_effect = fake_malloc
        cls.return_value = instance
        yield instance


def test_alloc_tensor_enters_owned_set(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    t = w.alloc_tensor((4,), torch.float32)
    assert t.data_ptr in w._owned_tensors
    w.close()


def test_free_tensor_exits_owned_set(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    t = w.alloc_tensor((4,), torch.float32)
    w.free_tensor(t)
    assert t.data_ptr not in w._owned_tensors
    fake_simpler_worker.free.assert_called_with(t.data_ptr, 0)
    w.close()


def test_close_auto_frees_leaked_tensors(fake_simpler_worker):
    """A DeviceTensor allocated but never freed should be released by close()."""
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    t = w.alloc_tensor((4,), torch.float32)
    # Caller "forgets" to free_tensor.
    fake_simpler_worker.free.reset_mock()
    w.close()
    # close() must have invoked free for the leaked ptr.
    freed_ptrs = [call.args[0] for call in fake_simpler_worker.free.call_args_list]
    assert t.data_ptr in freed_ptrs


def test_close_idempotent_after_auto_free(fake_simpler_worker):
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    w.alloc_tensor((4,), torch.float32)
    w.close()
    # Second close is a no-op; _owned_tensors already drained.
    fake_simpler_worker.free.reset_mock()
    w.close()
    assert fake_simpler_worker.free.call_count == 0


def test_free_tensor_twice_is_idempotent(fake_simpler_worker):
    """A second free_tensor on the same DeviceTensor must NOT call free again.

    Guards against double-free at the C++ layer when the caller's explicit
    cleanup races with the auto-free path.
    """
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    t = w.alloc_tensor((4,), torch.float32)
    fake_simpler_worker.free.reset_mock()
    w.free_tensor(t)
    w.free_tensor(t)  # second call is a no-op
    assert fake_simpler_worker.free.call_count == 1
    w.close()


def test_explicit_free_tensor_then_close_does_not_double_free(fake_simpler_worker):
    """If caller already freed, close()'s auto-free must NOT free it again."""
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    t = w.alloc_tensor((4,), torch.float32)
    w.free_tensor(t)  # explicit
    fake_simpler_worker.free.reset_mock()
    w.close()
    # No additional free for the already-released ptr.
    for call in fake_simpler_worker.free.call_args_list:
        assert call.args[0] != t.data_ptr


def test_close_auto_free_swallows_errors(fake_simpler_worker):
    """A failure in the auto-free loop must not block close()."""
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    w.alloc_tensor((4,), torch.float32)
    # Make the next free raise. close() should still complete.
    fake_simpler_worker.free.side_effect = RuntimeError("backend torn down")
    w.close()  # must not raise


def test_raw_malloc_not_tracked(fake_simpler_worker):
    """Raw ``malloc()`` is user-managed; it must NOT enter _owned_tensors."""
    w = ChipWorker(config=RunConfig(platform="a2a3sim"))
    ptr = w.malloc(64)
    assert ptr not in w._owned_tensors
    w.free(ptr)
    w.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
