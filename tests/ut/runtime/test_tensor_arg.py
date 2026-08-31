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
  :class:`DeviceTensor`'s retained ``Buffer``;
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


def test_device_tensor_derives_wire_tensor_from_retained_buffer():
    captured: dict = {"tensor_calls": []}

    class FakeBuffer:
        base = 0xABCD

        def tensor(self, *, shapes, dtype):
            captured["tensor_calls"].append({"shapes": tuple(shapes), "dtype": dtype})
            return MagicMock(name="wire_tensor")

    buffer = FakeBuffer()
    dt = DeviceTensor(buffer.base, (8, 16), torch.float16, buffer=buffer)
    worker = MagicMock(name="worker")
    owner = MagicMock(name="pypto_owner")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    bind_tensor_arg_owner(worker, owner)

    with patch(
        "pypto.runtime.task_interface.torch_dtype_to_datatype",
        side_effect=lambda d: f"<dtype:{d}>",
    ):
        first = make_tensor_arg(worker, dt)
        second = make_tensor_arg(worker, dt)

    assert first is second
    assert owner._require_owned_resident_tensor.call_count == 2
    owner._require_owned_resident_tensor.assert_called_with(dt, "Tensor argument")
    assert len(captured["tensor_calls"]) == 1
    call = captured["tensor_calls"][0]
    assert call["shapes"] == (8, 16)
    assert call["dtype"] == "<dtype:torch.float16>"


def test_retained_buffer_is_rejected_without_pypto_owner_binding():
    class FakeBuffer:
        base = 0xABCD

        def tensor(self, *, shapes, dtype):
            raise AssertionError("an unowned Buffer must be rejected before tensor()")

    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    buffer = FakeBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)
    with pytest.raises(TypeError, match="one-shot or raw simpler Worker"):
        make_tensor_arg(MagicMock(name="unbound_worker"), dt)


def test_owner_liveness_failure_prevents_wire_tensor_creation():
    class FakeBuffer:
        base = 0xABCD

        def tensor(self, *, shapes, dtype):
            raise AssertionError("a stale Buffer must be rejected before tensor()")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    worker = MagicMock(name="worker")
    owner = MagicMock(name="pypto_owner")
    owner._require_owned_resident_tensor.side_effect = ValueError("not a live allocation")
    bind_tensor_arg_owner(worker, owner)
    buffer = FakeBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)

    with pytest.raises(ValueError, match="not a live allocation"):
        make_tensor_arg(worker, dt)


def test_cached_wire_tensor_still_requires_live_owner():
    """Descriptor reuse must not bypass freed or foreign allocation checks."""

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    worker = MagicMock(name="worker")
    owner = MagicMock(name="pypto_owner")
    bind_tensor_arg_owner(worker, owner)
    buffer = MagicMock(name="buffer")
    buffer.base = 0xABCD
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)

    make_tensor_arg(worker, dt)
    owner._require_owned_resident_tensor.side_effect = ValueError("allocation was freed")

    with pytest.raises(ValueError, match="allocation was freed"):
        make_tensor_arg(worker, dt)

    buffer.tensor.assert_called_once()


def test_owner_liveness_lookup_does_not_scan_device_buffers():
    """Wide resident dispatch uses an O(1) allocation identity lookup."""

    class NoIterationDict(dict):
        def items(self):
            raise AssertionError("resident tensor validation must not scan all device buffers")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    owner = MagicMock(name="pypto_owner")
    owner._tensor_arg_worker = None
    worker = MagicMock(name="worker")
    bind_tensor_arg_owner(worker, owner)

    buffer = MagicMock(name="buffer")
    buffer.base = 0xABCD
    buffer.tensor.return_value = MagicMock(name="wire_tensor")
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)
    key = (7, dt.data_ptr)
    owner._require_owned_resident_tensor = None

    from pypto.runtime.runtime_base import Worker  # noqa: PLC0415

    owner._require_owned_resident_tensor = Worker._require_owned_resident_tensor.__get__(owner)
    owner._require_ready = MagicMock(name="require_ready")
    owner._owned_tensor_keys = {id(dt): key}
    owner._owned_tensors = {key: dt}
    owner._device_buffers = NoIterationDict({key: buffer})

    make_tensor_arg(worker, dt)

    owner._require_ready.assert_called_once_with("dispatch DeviceTensor")
    buffer.tensor.assert_called_once()


def test_stale_raw_backend_is_rejected_before_owner_validation():
    class FakeBuffer:
        base = 0xABCD

        def tensor(self, *, shapes, dtype):
            raise AssertionError("a stale backend must be rejected before tensor()")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    old_worker = MagicMock(name="old_worker")
    owner = MagicMock(name="pypto_owner")
    bind_tensor_arg_owner(old_worker, owner)
    owner._tensor_arg_worker = MagicMock(name="replacement_worker")
    buffer = FakeBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)

    with pytest.raises(ValueError, match="stale simpler Worker backend"):
        make_tensor_arg(old_worker, dt)
    owner._require_owned_resident_tensor.assert_not_called()


def test_raw_pointer_device_tensor_is_rejected_for_wire_dispatch():
    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    with pytest.raises(TypeError, match="raw-pointer DeviceTensor"):
        make_tensor_arg(MagicMock(name="worker"), DeviceTensor(0x1000, (4,), torch.float32))


def test_wire_tensor_passes_through():
    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    class FakeWireTensor:
        pass

    wire = FakeWireTensor()
    with patch("pypto.runtime.task_interface.Tensor", FakeWireTensor):
        assert make_tensor_arg(MagicMock(name="worker"), wire) is wire


def test_host_tensor_delegates_to_simpler():
    host = torch.zeros(4, 4, dtype=torch.float32)
    sentinel = MagicMock(name="Tensor(host)")
    worker = MagicMock(name="worker")

    with patch("simpler_setup.torch_interop.make_tensor_arg", return_value=sentinel) as impl:
        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        result = make_tensor_arg(worker, host)

    impl.assert_called_once_with(worker, host)
    assert result is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
