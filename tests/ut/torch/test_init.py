# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public initialization preflight and optional-dependency isolation."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pypto.runtime import _execution_mode
from pypto.runtime.kernel import session
from pypto.runtime.kernel.context import KernelState
from pypto.torch import init, interop, launch


@pytest.fixture
def framework(kernel_env, monkeypatch):
    fake = SimpleNamespace(__version__="2.6.0.post2", device=0, capture=False)
    fake._C = SimpleNamespace(_npu_shutdown_synchronize=lambda: None, _npu_shutdown=lambda: None)
    fake.npu = SimpleNamespace(
        current_device=lambda: fake.device,
        current_stream=lambda device: SimpleNamespace(stream_id=12),
    )
    monkeypatch.setattr(interop, "_load_torch_npu", lambda: fake)

    class CallConfig:
        def __init__(self):
            self.runtime_env = SimpleNamespace()

        def validate(self):
            if fake.config_error is not None:
                raise fake.config_error

    fake.config_error = None
    monkeypatch.setattr(session, "require_runtime", lambda: SimpleNamespace(CallConfig=CallConfig))
    monkeypatch.setattr(
        launch,
        "_load_native",
        lambda: SimpleNamespace(
            check_call=lambda stream, device: fake.capture,
            borrow_context=lambda device: 123,
        ),
    )
    return fake


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_public_init_is_idempotent_and_binds_current_device(kernel_env, framework, runtime):
    framework.device = 3
    init(runtime=runtime, ring_heap=4096)
    init(device=3, runtime=runtime, ring_heap=(4096,) * 4)
    assert len(kernel_env.calls.inits) == 1
    assert kernel_env.state.require_config().device_id == 3
    with pytest.raises(ValueError, match="configuration conflict"):
        init(runtime=runtime, aicpu_thread_num=4)
    kernel_env.state.close()


@pytest.mark.parametrize(
    "options,error",
    [
        ({"device": "0"}, TypeError),
        ({"device": True}, TypeError),
        ({"device": -1}, ValueError),
        ({"device": 1}, ValueError),
        ({"platform": "a2a3sim"}, ValueError),
        ({"runtime": "missing"}, ValueError),
        ({"aicpu_thread_num": 1}, ValueError),
        ({"enable_dep_gen": 1}, TypeError),
        ({"enable_chip_swimlane": True}, ValueError),
        ({"ring_heap": (1, 2)}, ValueError),
        ({"workspace_budget_bytes": True}, ValueError),
        ({"workspace_budget_bytes": 1023}, ValueError),
    ],
)
def test_invalid_config_does_not_claim_mode(kernel_env, framework, options, error):
    with pytest.raises(error):
        init(**options)
    assert not kernel_env.calls.workers and _execution_mode._gate.mode is None
    init()
    kernel_env.state.close()


@pytest.mark.parametrize("failure", ["capture", "framework", "runtime", "context", "shutdown", "call_config"])
def test_preflight_failure_is_retryable(kernel_env, framework, monkeypatch, failure):
    with monkeypatch.context() as patch:
        if failure == "capture":
            patch.setattr(framework, "capture", True)
        elif failure == "framework":
            patch.setattr(framework, "__version__", "0.0.0")
        elif failure == "call_config":
            patch.setattr(framework, "config_error", ValueError("invalid native ring sizing"))
        elif failure == "shutdown":
            patch.delattr(framework._C, "_npu_shutdown")
        elif failure == "runtime":

            def fail_runtime():
                raise NotImplementedError("missing borrowed init")

            patch.setattr(session, "require_runtime", fail_runtime)
        else:

            def fail_context(device):
                raise RuntimeError("torn-down context")

            patch.setattr(
                launch,
                "_load_native",
                lambda: SimpleNamespace(
                    check_call=lambda *args: False,
                    borrow_context=fail_context,
                ),
            )
        with pytest.raises((RuntimeError, NotImplementedError, ValueError)):
            init()
    assert kernel_env.state.state is KernelState.UNINITIALIZED
    assert _execution_mode._gate.mode is None
    init()
    kernel_env.state.close()


def test_concurrent_public_initialization(kernel_env, framework):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: init(), range(16)))
    assert len(kernel_env.calls.workers) == len(kernel_env.calls.inits) == 1
    kernel_env.state.close()


def test_program_claim_and_closed_state_reject_init(kernel_env, framework):
    _execution_mode.claim_program_mode()
    with pytest.raises(RuntimeError, match="already claimed program"):
        init()
    assert not kernel_env.calls.workers
    kernel_env.state.close()
    with pytest.raises(RuntimeError, match="closing or closed"):
        init()


def test_new_modules_import_without_optional_runtime(run_without_optional_runtime):
    result = run_without_optional_runtime("""
import pypto.torch
from pypto.runtime.kernel import context, pending, session
from pypto.torch import execution, launch, shutdown
assert pypto.torch.__all__ == ['init']
assert session.KernelConfig('a2a3', 'host_build_graph', 0).device_id == 0
assert context.get_process_kernel_state().state is context.KernelState.UNINITIALIZED
""")
    assert result.returncode == 0, result.stderr


def test_native_loader_reports_missing_and_incomplete_adapter(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(launch.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="PYPTO_BUILD_TORCH_NPU"):
        launch._load_native()
    monkeypatch.setattr(launch.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(NotImplementedError, match="check_call"):
        launch._load_native()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
