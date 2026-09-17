# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit kernel execution setup, validated with a fake native Worker."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from pypto.runtime import _execution_mode, _kernel_artifact
from pypto.runtime.kernel.abi import KernelConfig
from pypto.runtime.kernel.context import KernelState
from pypto.torch import init, interop, launch

from tests.ut.runtime import test_kernel_context

setup = test_kernel_context.setup


@pytest.fixture
def framework(setup, monkeypatch):
    fake = SimpleNamespace(
        __version__="2.6.0.post2",
        device=0,
        capture=0,
        streams=[],
    )
    fake.npu = SimpleNamespace(
        current_device=lambda: fake.device,
        current_stream=lambda device: fake.streams.append(device) or SimpleNamespace(stream_id=12),
    )
    monkeypatch.setattr(interop, "_load_torch_npu", lambda: fake)
    monkeypatch.setattr(
        launch, "_load_native", lambda: SimpleNamespace(check_call=lambda stream_id, device: fake.capture)
    )
    fake.native_revision_error = None

    def require_kernel_native(abi):
        if fake.native_revision_error is not None:
            raise fake.native_revision_error

    monkeypatch.setattr(_kernel_artifact, "require_kernel_native", require_kernel_native)
    return fake


def test_init_binds_current_device_and_initializes_once(setup, framework):
    state, _, calls, _ = setup
    framework.device = 3
    init(aicpu_thread_num=4)
    assert state.require_config() == KernelConfig("a2a3", "tensormap_and_ringbuffer", 3, 4)
    assert len(calls.workers) == len(calls.inits) == 1 and framework.streams == [3]
    init(device=3, aicpu_thread_num=4)
    assert len(calls.workers) == 1
    with pytest.raises(ValueError, match="configuration conflict"):
        init(device=3)
    assert len(calls.workers) == len(calls.inits) == 1
    state.close()


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("platform", ValueError, "supports"),
        ("runtime", ValueError, "supports"),
        ("device", ValueError, r"torch\.npu\.set_device\(1\)"),
        ("threads", ValueError, "AICPU"),
        ("framework", RuntimeError, "verified for torch_npu 2.6.0.post2"),
        ("native", ValueError, "requires Simpler"),
        ("capture", RuntimeError, "outside graph capture"),
    ],
)
def test_invalid_init_leaves_process_unclaimed_and_retryable(setup, framework, case, error, match):
    state, _, calls, _ = setup
    kwargs = {
        "platform": {"platform": "a5"},
        "runtime": {"runtime": "host_build_graph"},
        "device": {"device": 1},
        "threads": {"aicpu_thread_num": 1},
    }.get(case, {})
    if case == "framework":
        framework.__version__ = "2.7.0"
    elif case == "capture":
        framework.capture = 42
    elif case == "native":
        framework.native_revision_error = ValueError(
            "Kernel ABI requires Simpler abc; native binding is 'def'"
        )
    with pytest.raises(error, match=match):
        init(**kwargs)
    assert not calls.workers and _execution_mode._gate.mode is None
    assert state.state is KernelState.UNINITIALIZED
    framework.__version__, framework.capture, framework.native_revision_error = "2.6.0.post2", 0, None
    init()
    assert state.require_config().device_id == 0
    state.close()


def test_concurrent_init_shares_one_worker(setup, framework):
    state, config, calls, _ = setup
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: init(), range(16)))
    assert len(calls.workers) == len(calls.inits) == 1
    assert state.require_config() == config
    state.close()


def test_unsupported_target_is_rejected_before_loading_the_framework(setup, monkeypatch):
    def forbidden():
        pytest.fail("an unsupported target must not load torch_npu")

    monkeypatch.setattr(interop, "_load_torch_npu", forbidden)
    with pytest.raises(ValueError, match="supports"):
        init(platform="a5")
    text_device: Any = "3"
    with pytest.raises(TypeError, match="int NPU index"):
        init(device=text_device)


def test_init_respects_process_mode_and_shutdown(setup, framework):
    state, _, calls, _ = setup
    _execution_mode.claim_program_mode()
    with pytest.raises(RuntimeError, match="already claimed program"):
        init()
    assert not calls.workers
    state.close()
    with pytest.raises(RuntimeError, match="closing or closed"):
        init()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
