# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Queue adapter dispatch without requiring the optional native NPU module."""

import ctypes
import struct
from types import SimpleNamespace

import pytest
from pypto._kernel_abi import SIMPLER_KERNEL_REVISION, KernelABI, KernelParameter
from pypto.runtime.kernel.callable import KernelRegistration
from pypto.torch import interop, launch

from tests.ut.torch.test_interop import _tensor


@pytest.fixture
def dispatch(monkeypatch):
    calls = []
    stream = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), stream_id=12)
    npu = SimpleNamespace(
        get_npu_format=lambda tensor: 2,
        npu=SimpleNamespace(current_device=lambda: 0, current_stream=lambda device: stream),
    )
    monkeypatch.setattr(interop, "_load_torch_npu", lambda: npu)
    worker = SimpleNamespace(native_launch_target=object())
    native = SimpleNamespace(prepare=lambda *args: calls.append(args) or object(), check_call=lambda *args: 0)
    monkeypatch.setattr(launch, "_load_native", lambda: native)
    owner = SimpleNamespace(
        config=SimpleNamespace(device_id=0),
        require_registration=lambda registration: None,
        submit=lambda registration, prepare: prepare(worker),
    )
    abi = KernelABI(
        "a2a3",
        "tensormap_and_ringbuffer",
        (
            KernelParameter("x", "fp32", "In", (2, 3)),
            KernelParameter("scale", "fp32", "In", None),
            KernelParameter("out", "fp32", "Out", (2, 3)),
            KernelParameter("count", "int32", "In", None),
        ),
        (2,),
    )
    registration = KernelRegistration(b"id", 1, 1, 7, owner, object(), SimpleNamespace(kernel_abi=abi))
    return registration, calls, stream


def test_enqueue_snapshots_scalar_bits_and_preserves_output_alias(dispatch):
    registration, calls, stream = dispatch
    x, out = _tensor(), _tensor()
    scalar = ctypes.c_float(1.25)
    assert launch.enqueue(registration, (x, scalar, out, -7)) is out
    scalar.value = 2.5
    stream.stream_id = 33
    launch.enqueue(registration, (x, scalar, out, 9))
    first, second = calls
    assert first[1] == 7 and first[2][0] is x and first[2][1] is out
    assert first[3] == [0, 0]
    assert first[4] == [int.from_bytes(struct.pack("<f", 1.25), "little"), 2**32 - 7]
    assert second[4] == [int.from_bytes(struct.pack("<f", 2.5), "little"), 9]
    assert first[5:] == (12, 0) and second[5:] == (33, 0)


def test_invalid_arguments_never_reach_native(dispatch):
    registration, calls, _ = dispatch
    with pytest.raises(TypeError, match="expects"):
        launch.enqueue(registration, (_tensor(), 2.0))
    assert not calls


def test_wrong_worker_device_is_rejected(dispatch):
    registration, calls, _ = dispatch
    registration.owner.config.device_id = 1
    with pytest.raises(ValueError, match="Worker device"):
        launch.enqueue(registration, (_tensor(), 2.0, _tensor(), 1))
    assert not calls


def test_stale_registration_precedes_frame_or_native_queries(dispatch):
    registration, calls, _ = dispatch

    def stale(registration):
        raise RuntimeError("stale handle")

    registration.owner.require_registration = stale
    with pytest.raises(RuntimeError, match="stale handle"):
        launch.enqueue(registration, ())
    assert not calls


def test_missing_or_incompatible_optional_adapter(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(launch.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="PYPTO_BUILD_TORCH_NPU"):
        launch._load_native()
    monkeypatch.setattr(
        launch.importlib, "import_module", lambda name: SimpleNamespace(simpler_revision="stale")
    )
    with pytest.raises(RuntimeError, match="different Simpler revision"):
        launch._load_native()
    native = SimpleNamespace(simpler_revision=SIMPLER_KERNEL_REVISION)
    monkeypatch.setattr(launch.importlib, "import_module", lambda name: native)
    assert launch._load_native() is native


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
