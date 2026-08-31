# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify the pypto-owned ``make_tensor_arg`` used by generated distributed
orchestration code.

It must:
- derive an address-free wire ``Tensor`` from a worker-resident
  :class:`DeviceTensor`'s retained ``Buffer``, and build it only once per handle;
- leave liveness validation to the runtime on an L3 dispatch, whose
  ``submit_next_level`` runs an identity-keyed provenance guard, while still
  validating on an L2 dispatch, whose submit path has no equivalent;
- pass an already-built ``Tensor`` through unchanged;
- delegate a host ``torch.Tensor`` to simpler's worker-aware wire helper.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import DeviceTensor

# ``task_interface`` eagerly imports the optional ``simpler`` runtime package;
# skip the module when simpler is unavailable (same pattern as
# test_execute_compiled_device_tensor.py).
try:
    import simpler  # noqa: F401  # pyright: ignore[reportMissingImports]
except ImportError:
    _has_simpler = False
else:
    _has_simpler = True

pytestmark = pytest.mark.skipif(not _has_simpler, reason="make_tensor_arg requires the simpler package")


def _worker(level: int, name: str = "worker") -> MagicMock:
    """A stand-in simpler Worker at *level*.

    ``level`` is what decides whether this module validates DeviceTensor
    liveness itself (L2) or defers to the runtime's submit-time guard (L3+).
    """
    worker = MagicMock(name=name)
    worker.level = level
    return worker


class _RecordingBuffer:
    """Retained owner Buffer that records every wire-descriptor build."""

    def __init__(self, base: int = 0xABCD) -> None:
        self.base = base
        self.owner_worker_id = 0
        self.tensor_calls: list[tuple[tuple[int, ...], object]] = []

    def tensor(self, *, shapes, dtype):
        self.tensor_calls.append((tuple(shapes), dtype))
        return MagicMock(name=f"wire_tensor_{len(self.tensor_calls)}")


class _RejectingBuffer:
    """Retained owner Buffer whose descriptor build must never be reached."""

    def __init__(self, base: int = 0xABCD) -> None:
        self.base = base
        self.owner_worker_id = 0

    def tensor(self, *, shapes, dtype):
        raise AssertionError("a rejected DeviceTensor must not reach buffer.tensor()")


def test_device_tensor_derives_wire_tensor_from_retained_buffer():
    buffer = _RecordingBuffer()
    dt = DeviceTensor(buffer.base, (8, 16), torch.float16, buffer=buffer)
    worker = _worker(3)

    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    with patch(
        "pypto.runtime.task_interface.torch_dtype_to_datatype",
        side_effect=lambda d: f"<dtype:{d}>",
    ):
        make_tensor_arg(worker, dt)

    assert len(buffer.tensor_calls) == 1
    shapes, dtype = buffer.tensor_calls[0]
    assert shapes == (8, 16)
    assert dtype == "<dtype:torch.float16>"


def test_wire_tensor_is_built_once_per_device_tensor():
    """A stable handle re-dispatched every step must not rebuild its descriptor.

    The wire ``Tensor`` is address-free and a pure function of the handle's
    (Buffer, shape, dtype), all immutable — so rebuilding it per rank per
    dispatch was pure host-side latency.
    """
    buffer = _RecordingBuffer()
    dt = DeviceTensor(buffer.base, (8, 16), torch.float16, buffer=buffer)
    worker = _worker(3)

    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    with patch(
        "pypto.runtime.task_interface.torch_dtype_to_datatype",
        side_effect=lambda d: f"<dtype:{d}>",
    ):
        first = make_tensor_arg(worker, dt)
        second = make_tensor_arg(worker, dt)
        third = make_tensor_arg(_worker(3, "another_rank"), dt)

    assert first is second is third
    assert len(buffer.tensor_calls) == 1


def test_memo_is_per_handle_so_pointer_reuse_cannot_share_a_descriptor():
    """Two handles at one address must not share a cached descriptor.

    ``DeviceTensor`` excludes ``buffer`` from ``__eq__`` / ``__hash__``, so a
    freed handle and the fresh allocation that reuses its address compare equal.
    Memoizing per handle keeps them apart; a dict keyed on the handle would not.
    """
    freed_buffer = _RecordingBuffer(0x1000)
    fresh_buffer = _RecordingBuffer(0x1000)
    freed = DeviceTensor(0x1000, (8, 16), torch.float16, buffer=freed_buffer)
    fresh = DeviceTensor(0x1000, (8, 16), torch.float16, buffer=fresh_buffer)
    assert freed == fresh and hash(freed) == hash(fresh)

    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    with patch(
        "pypto.runtime.task_interface.torch_dtype_to_datatype",
        side_effect=lambda d: f"<dtype:{d}>",
    ):
        freed_wire = make_tensor_arg(_worker(3), freed)
        fresh_wire = make_tensor_arg(_worker(3), fresh)

    assert freed_wire is not fresh_wire
    assert len(freed_buffer.tensor_calls) == 1
    assert len(fresh_buffer.tensor_calls) == 1


def test_l3_dispatch_defers_liveness_validation_to_the_runtime():
    """Regression guard for the DP8 host-dispatch staircase (O(N^2) packing).

    ``orch.submit_next_level`` validates every device argument against the
    owner's identity-keyed allocation table, under a lock held through the
    native submit. Re-deriving a weaker answer here cost a walk of the whole
    buffer table per argument per rank.
    """
    buffer = _RecordingBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)
    worker = _worker(3)
    owner = MagicMock(name="pypto_owner")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    bind_tensor_arg_owner(worker, owner)

    with patch(
        "pypto.runtime.task_interface.torch_dtype_to_datatype",
        side_effect=lambda d: f"<dtype:{d}>",
    ):
        make_tensor_arg(worker, dt)

    owner._require_owned_resident_tensor.assert_not_called()


def test_l2_dispatch_still_validates_against_the_owning_worker():
    """L2's ``_submit_l2_locked`` has no provenance guard, so this is the only one."""
    buffer = _RecordingBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)
    worker = _worker(2)
    owner = MagicMock(name="pypto_owner")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    bind_tensor_arg_owner(worker, owner)

    with patch(
        "pypto.runtime.task_interface.torch_dtype_to_datatype",
        side_effect=lambda d: f"<dtype:{d}>",
    ):
        make_tensor_arg(worker, dt)

    owner._require_owned_resident_tensor.assert_called_once_with(dt, "Tensor argument")


def test_l2_validation_runs_again_on_every_reuse_of_a_cached_descriptor():
    """The memo must not become a way around the L2 liveness check.

    A cached descriptor is exactly as stale as the handle it was built from, so
    validation has to precede the cache lookup rather than sit behind it.
    """
    buffer = _RecordingBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)
    worker = _worker(2)
    owner = MagicMock(name="pypto_owner")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    bind_tensor_arg_owner(worker, owner)

    with patch(
        "pypto.runtime.task_interface.torch_dtype_to_datatype",
        side_effect=lambda d: f"<dtype:{d}>",
    ):
        make_tensor_arg(worker, dt)
        owner._require_owned_resident_tensor.side_effect = ValueError("not a live allocation")
        with pytest.raises(ValueError, match="not a live allocation"):
            make_tensor_arg(worker, dt)

    assert owner._require_owned_resident_tensor.call_count == 2
    assert len(buffer.tensor_calls) == 1


def test_unrecognizable_worker_level_falls_back_to_validating():
    """A Worker whose level cannot be read must fail closed into checking."""
    buffer = _RejectingBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)

    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    # No ``level`` an int check accepts, and no owner binding: the fallback path
    # rejects rather than silently converting.
    with pytest.raises(TypeError, match="one-shot or raw simpler Worker"):
        make_tensor_arg(MagicMock(name="unbound_worker"), dt)


def test_l2_retained_buffer_is_rejected_without_pypto_owner_binding():
    buffer = _RejectingBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)

    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    with pytest.raises(TypeError, match="one-shot or raw simpler Worker"):
        make_tensor_arg(_worker(2, "unbound_worker"), dt)


def test_l2_owner_liveness_failure_prevents_wire_tensor_creation():
    buffer = _RejectingBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)
    worker = _worker(2)
    owner = MagicMock(name="pypto_owner")
    owner._require_owned_resident_tensor.side_effect = ValueError("not a live allocation")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    bind_tensor_arg_owner(worker, owner)

    with pytest.raises(ValueError, match="not a live allocation"):
        make_tensor_arg(worker, dt)


def test_l2_stale_raw_backend_is_rejected_before_owner_validation():
    buffer = _RejectingBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)
    old_worker = _worker(2, "old_worker")
    owner = MagicMock(name="pypto_owner")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    bind_tensor_arg_owner(old_worker, owner)
    owner._tensor_arg_worker = MagicMock(name="replacement_worker")

    with pytest.raises(ValueError, match="stale simpler Worker backend"):
        make_tensor_arg(old_worker, dt)
    owner._require_owned_resident_tensor.assert_not_called()


def test_raw_pointer_device_tensor_is_rejected_for_wire_dispatch():
    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    with pytest.raises(TypeError, match="raw-pointer DeviceTensor"):
        make_tensor_arg(_worker(3), DeviceTensor(0x1000, (4,), torch.float32))


def test_wire_tensor_passes_through():
    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    class FakeWireTensor:
        pass

    wire = FakeWireTensor()
    with patch("pypto.runtime.task_interface.Tensor", FakeWireTensor):
        assert make_tensor_arg(_worker(3), wire) is wire


def test_host_tensor_delegates_to_simpler():
    host = torch.zeros(4, 4, dtype=torch.float32)
    sentinel = MagicMock(name="Tensor(host)")
    worker = _worker(3)

    with patch("simpler_setup.torch_interop.make_tensor_arg", return_value=sentinel) as impl:
        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        result = make_tensor_arg(worker, host)

    impl.assert_called_once_with(worker, host)
    assert result is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
