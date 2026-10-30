# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Worker adapter contract; fake objects expose no private slots or kernel API."""

import sys
import threading
from types import SimpleNamespace

import pytest
from pypto.runtime.kernel import session
from pypto.runtime.kernel.context import _ProcessKernelState
from pypto.runtime.kernel.session import KernelConfig
from pypto.torch import shutdown


@pytest.fixture
def public_runtime(monkeypatch):
    events = []
    handles = []

    class CallConfig:
        def __init__(self):
            self.runtime_env = SimpleNamespace()

        def validate(self):
            events.append(("validate", threading.get_ident()))

    class Worker:
        device_location = object()

        def __init__(self, **kwargs):
            self.options = kwargs
            self.config = None
            events.append(("construct", threading.get_ident(), kwargs))

        def init(self, prewarm_config=None, *, borrowed_context_generation=None):
            self.config = prewarm_config
            events.append(("init", threading.get_ident(), borrowed_context_generation))

        def register(self, target):
            handle = object()
            handles.append(handle)
            events.append(("register", threading.get_ident(), target))
            return handle

        def unregister(self, handle):
            raise AssertionError("no eager slot eviction")

        def submit(self, target, args=None, config=None, *, caller_stream=None):
            raise AssertionError("lifecycle does not submit")

        def flush_diagnostics(self):
            raise AssertionError("DFX windows are a later step")

        def release_buffer(self, buffer):
            buffer.close()

        def close(self):
            events.append(("close", threading.get_ident()))

    class RunHandle:
        done = property(lambda self: True)

        def wait(self):
            pass

    class Buffer:
        @classmethod
        def wrap(cls):
            pass

        def close(self):
            pass

    class TaskArgs:
        def add_tensor(self):
            pass

        def add_scalar(self):
            pass

    api = SimpleNamespace(
        Worker=Worker,
        CallConfig=CallConfig,
        Buffer=Buffer,
        TaskArgs=TaskArgs,
        Tensor=object,
        RunHandle=RunHandle,
        events=events,
        handles=handles,
    )
    for name in ("simpler", "simpler.task_interface", "simpler.worker"):
        monkeypatch.setitem(sys.modules, name, api)
    monkeypatch.setattr(session, "check_runtime_pin", lambda: events.append(("pin",)))
    native = SimpleNamespace(
        borrow_context=lambda device: 987,
        bind_context=lambda context: events.append(("bind", threading.get_ident(), context)),
    )
    monkeypatch.setattr(shutdown, "install_shutdown", lambda state: native)
    return api, native


def test_public_worker_receives_config_generation_and_opaque_handle(public_runtime, tmp_path):
    api, _ = public_runtime
    cfg = KernelConfig(
        "a2a3",
        "host_build_graph",
        3,
        4,
        True,
        True,
        tmp_path,
        8,
        (1024, 2048, 4096, 8192),
        16,
        4096,
    )
    state = _ProcessKernelState()
    adapter = session._BorrowedWorkerSession(cfg, state)
    caller = threading.get_ident()
    adapter.init(cfg)
    target = object()
    handle = adapter.prepare(target)
    assert handle is api.handles[0]
    assert adapter.worker.options == dict(
        level=2,
        platform="a2a3",
        runtime="host_build_graph",
        device_id=3,
        enable_sdma=False,
        collect_across_runs=True,
        workspace_budget_bytes=4096,
    )
    call_config = adapter.worker.config
    assert call_config.aicpu_thread_num == 4 and call_config.enable_chip_swimlane == 4
    assert call_config.enable_dep_gen and call_config.output_prefix == str(tmp_path)
    assert call_config.runtime_env.ring_heap == [1024, 2048, 4096, 8192]
    assert call_config.runtime_env.ring_task_window == [8] * 4
    assert call_config.runtime_env.ring_dep_pool == [16] * 4
    assert adapter.device_location is api.Worker.device_location
    adapter.close()
    operations = [e for e in api.events if e[0] in ("init", "register", "close")]
    assert [e[0] for e in operations] == ["init", "register", "close"]
    assert len({e[1] for e in operations}) == 1 and operations[0][1] != caller
    assert operations[0][2] == state.generation and state.generation > 0
    assert [e[2] for e in api.events if e[0] == "bind"] == [987] * 3


def test_context_mismatch_and_register_error_keep_original_semantics(public_runtime, monkeypatch):
    api, native = public_runtime
    cfg = KernelConfig("a2a3", "tensormap_and_ringbuffer", 0)
    adapter = session._BorrowedWorkerSession(cfg, _ProcessKernelState())
    adapter.init(cfg)
    monkeypatch.setattr(native, "borrow_context", lambda device: 988)
    with pytest.raises(RuntimeError, match="different borrowed context"):
        adapter.check_context()
    error = ValueError("capacity exhausted")

    def register(self, target):
        raise error

    monkeypatch.setattr(api.Worker, "register", register)
    with pytest.raises(ValueError) as caught:
        adapter.prepare(object())
    assert caught.value is error
    adapter.close()


@pytest.mark.parametrize(
    "missing", ["init_keyword", "submit_keyword", "register", "done", "buffer", "scalar", "release", "dfx"]
)
def test_capability_guard_rejects_missing_public_contract(public_runtime, monkeypatch, missing):
    api, _ = public_runtime
    if missing == "init_keyword":
        monkeypatch.setattr(api.Worker, "init", lambda self: None)
    elif missing == "submit_keyword":
        monkeypatch.setattr(api.Worker, "submit", lambda self: None)
    elif missing == "register":
        monkeypatch.delattr(api.Worker, "register")
    elif missing == "done":
        monkeypatch.delattr(api.RunHandle, "done")
    elif missing == "buffer":
        monkeypatch.delattr(api.Buffer, "wrap")
    elif missing == "scalar":
        monkeypatch.delattr(api.TaskArgs, "add_scalar")
    elif missing == "release":
        monkeypatch.delattr(api.Worker, "release_buffer")
    else:
        monkeypatch.delattr(api.Worker, "flush_diagnostics")
    with pytest.raises(NotImplementedError):
        session.require_runtime()


def test_pin_error_is_not_reclassified_and_old_supported_bit_is_ignored(public_runtime, monkeypatch):
    api, _ = public_runtime
    api.kernel_mode_supported = 0
    assert session.require_runtime().Worker is api.Worker
    error = ValueError("native revision mismatch")

    def reject():
        raise error

    monkeypatch.setattr(session, "check_runtime_pin", reject)
    with pytest.raises(ValueError) as caught:
        session.require_runtime()
    assert caught.value is error


@pytest.mark.parametrize("operation", ["init", "close", "prepare"])
def test_owner_thread_reentrancy_fails_instead_of_waiting(kernel_env, public_runtime, monkeypatch, operation):
    from pypto.runtime.kernel import context  # noqa: PLC0415

    env = kernel_env
    api, _ = public_runtime
    monkeypatch.setattr(context, "_BorrowedWorkerSession", session._BorrowedWorkerSession)
    if operation == "init":
        monkeypatch.setattr(api.Worker, "init", lambda self, **kwargs: env.state.ensure_worker(env.config))
    elif operation == "close":
        monkeypatch.setattr(api.Worker, "init", lambda self, **kwargs: env.state.close())
    else:
        env.state.ensure_worker(env.config)
        monkeypatch.setattr(
            api.Worker, "register", lambda self, target: env.state.ensure_callable(env.artifact(), env.config)
        )
    # The signature capability check is a separate test; preserve the real adapter here.
    monkeypatch.setattr(session, "require_runtime", lambda: api)
    try:
        with pytest.raises(RuntimeError, match="[Rr]eentrant|owner thread"):
            if operation == "prepare":
                env.state.ensure_callable(env.artifact(), env.config)
            else:
                env.state.ensure_worker(env.config)
    finally:
        env.state.close()


def test_retirement_binds_context_on_owner_before_releasing_sources(public_runtime):
    api, _ = public_runtime
    config = KernelConfig("a2a3", "tensormap_and_ringbuffer", 0)
    adapter = session._BorrowedWorkerSession(config, _ProcessKernelState())
    adapter.init(config)

    def wait():
        api.events.append(("wait", threading.get_ident()))

    adapter.wait(SimpleNamespace(wait=wait))
    adapter.close()
    actions = [e for e in api.events if e[0] in ("init", "wait", "close")]
    assert len({e[1] for e in actions}) == 1
    wait_index = next(i for i, e in enumerate(api.events) if e[0] == "wait")
    assert api.events[wait_index - 1][0] == "bind"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
