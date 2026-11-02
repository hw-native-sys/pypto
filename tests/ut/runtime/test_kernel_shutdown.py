# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shutdown ordering and owner-thread affinity with mocked framework boundaries."""

import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pypto.runtime.kernel.context import KernelState
from pypto.runtime.kernel.owner import _OwnerThread
from pypto.torch import shutdown


@pytest.fixture
def framework(kernel_env, monkeypatch):
    worker_cls = kernel_env.worker_cls
    events = []
    native = SimpleNamespace(
        framework_alive=lambda: True,
        retain_until_exit=lambda owner: events.append(("retain", owner)),
    )
    framework = SimpleNamespace(
        __version__="2.6.0.post2",
        _C=SimpleNamespace(
            _npu_shutdown_synchronize=lambda: events.append("framework-sync") or True,
            _npu_shutdown=lambda success: events.append(("framework-finalize", success)),
        ),
    )
    monkeypatch.setattr(shutdown, "_load_torch_npu", lambda: framework)
    monkeypatch.setattr(shutdown, "_load_native", lambda: native)
    original = worker_cls.close

    def close(worker):
        events.append("worker-close")
        original(worker)

    monkeypatch.setattr(worker_cls, "close", close)
    return framework, native, events


@pytest.mark.parametrize("version", ["2.6.0.post2", "2.6.0.post2+cxx11-abi", "2.6.0.post2+cxx11.abi"])
def test_framework_hook_drains_then_closes_once_before_teardown(kernel_env, framework, version):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    fw, native, events = framework
    fw.__version__ = version
    shutdown.install_shutdown(state)
    hook = fw._pypto_kernel_shutdown
    shutdown.install_shutdown(state)
    assert fw._pypto_kernel_shutdown is hook
    state.ensure_worker(config)
    state.ensure_callable(kernel_env.artifact(), config)
    state.ensure_callable(kernel_env.artifact(b"other"), config)
    state._submissions.append(SimpleNamespace(wait=lambda: events.append("ticket-wait")))
    assert fw._C._npu_shutdown_synchronize() is True
    fw._C._npu_shutdown(True)
    fw._C._npu_shutdown_synchronize()
    assert events == [
        "ticket-wait",
        "worker-close",
        "framework-sync",
        ("framework-finalize", True),
        "framework-sync",
    ]
    assert len(calls.workers) == len(calls.closes) == 1
    assert state.state is KernelState.CLOSED and not state._registrations
    with pytest.raises(RuntimeError, match="closing or closed"):
        state.ensure_worker(config)


def test_unused_shutdown_never_constructs_a_worker(kernel_env, framework):
    state, _, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    fw, _, events = framework
    shutdown.install_shutdown(state)
    fw._C._npu_shutdown(True)
    assert events == [("framework-finalize", True)]
    assert not calls.workers and not calls.closes


def test_partial_initialization_is_closed_before_framework(kernel_env, framework, monkeypatch):
    state, config, calls, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )
    fw, _, events = framework
    shutdown.install_shutdown(state)

    def fail(worker, config):
        raise RuntimeError("partial init")

    monkeypatch.setattr(worker_cls, "init", fail)
    with pytest.raises(RuntimeError, match="partial init"):
        state.ensure_worker(config)
    fw._C._npu_shutdown_synchronize()
    assert state.state is KernelState.CLOSED
    assert calls.closes == calls.workers
    assert events == ["worker-close", "framework-sync"]


def test_warning_as_error_does_not_interrupt_framework_shutdown(kernel_env, framework, capsys):
    state, config, _, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    fw, native, events = framework
    state.ensure_worker(config)
    state.ensure_callable(kernel_env.artifact(), config)
    shutdown.install_shutdown(state)
    native.framework_alive = lambda: False
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fw._C._npu_shutdown_synchronize()
        fw._C._npu_shutdown(False)
    assert events == [("retain", state), "framework-sync", ("framework-finalize", False)]
    assert "retained until process exit" in capsys.readouterr().err


def test_shutdown_waits_for_initialization_and_rejects_new_work(kernel_env, monkeypatch):
    state, config, calls, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )
    entered, release = threading.Event(), threading.Event()
    original = worker_cls.init

    def init(worker, config):
        entered.set()
        assert release.wait(5)
        original(worker, config)

    monkeypatch.setattr(worker_cls, "init", init)
    with ThreadPoolExecutor(max_workers=2) as pool:
        initialized = pool.submit(state.ensure_worker, config)
        assert entered.wait(5)
        closed = pool.submit(state.close)
        try:
            with state._condition:
                # Close publishes the admission fence before joining initialization.
                assert state._condition.wait_for(lambda: state._stop_requested, timeout=5)
            with pytest.raises(RuntimeError, match="closing or closed"):
                state.ensure_worker(config)
            assert not closed.done() and not calls.closes
        finally:
            release.set()
        initialized.result(timeout=5)
        closed.result(timeout=5)
    assert state.state is KernelState.CLOSED and len(calls.closes) == 1


def test_concurrent_close_is_idempotent(kernel_env):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: state.close(), range(8)))
    assert len(calls.closes) == 1 and state.state is KernelState.CLOSED


def test_initialization_does_not_override_abandoned_shutdown(kernel_env, framework, monkeypatch):
    state, config, _, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )
    fw, native, events = framework
    shutdown.install_shutdown(state)
    entered, release = threading.Event(), threading.Event()

    def init(worker, config):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(worker_cls, "init", init)
    with ThreadPoolExecutor(max_workers=1) as pool:
        initialized = pool.submit(state.ensure_worker, config)
        try:
            assert entered.wait(5)
            native.framework_alive = lambda: False
            with pytest.warns(RuntimeWarning, match="retained until process exit"):
                fw._C._npu_shutdown_synchronize()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="failed, expected ready"):
            initialized.result(timeout=5)
    assert state.state is KernelState.FAILED
    assert events == [("retain", state), "framework-sync"]


def test_admission_failure_does_not_reopen_a_closing_manager(kernel_env):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
    entered, release = threading.Event(), threading.Event()

    def enqueue():
        entered.set()
        assert release.wait(5)
        raise RuntimeError("admission failed")

    def wait():
        assert state.state is KernelState.CLOSING

    ticket = SimpleNamespace(enqueue=enqueue, done=lambda: False, wait=wait)
    with ThreadPoolExecutor(max_workers=3) as pool:
        submitting = pool.submit(state.submit, registration, lambda worker: ticket)
        assert entered.wait(5)
        closing = pool.submit(state.close)
        try:
            with state._condition:
                assert state._condition.wait_for(lambda: state.state is KernelState.CLOSING, timeout=5)
            again = pool.submit(state.close)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="admission failed"):
            submitting.result(timeout=5)
        closing.result(timeout=5)
        again.result(timeout=5)
    assert state.state is KernelState.CLOSED and len(calls.closes) == 1


def test_owner_thread_survives_initial_caller_and_preserves_affinity():
    with ThreadPoolExecutor(max_workers=1) as caller:
        owner = caller.submit(_OwnerThread).result(timeout=5)
        initialized = caller.submit(owner.call, threading.get_ident).result(timeout=5)
    try:
        prepared = owner.call(threading.get_ident)
        finalized = owner.call(threading.get_ident)
        assert initialized == prepared == finalized == owner.thread.ident
        assert initialized != threading.get_ident()
        with pytest.raises(RuntimeError, match="Reentrant"):
            owner.call(lambda: owner.call(lambda: None))
        # A failed operation leaves the owner alive for cleanup/retry.
        assert owner.call(threading.get_ident) == initialized
    finally:
        owner.stop()
    assert not owner.thread.is_alive()


@pytest.mark.parametrize("version", ["2.5.1", "2.7.1", "2.6.0.post1+cxx11-abi"])
def test_unverified_framework_fails_before_install(kernel_env, framework, version):
    state, _, _, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    fw, _, _ = framework
    fw.__version__ = version
    original = fw._C._npu_shutdown
    with pytest.raises(RuntimeError, match="teardown contract"):
        shutdown.install_shutdown(state)
    assert fw._C._npu_shutdown is original and not hasattr(fw, "_pypto_kernel_shutdown")


@pytest.mark.parametrize("finalize_first", [False, True])
@pytest.mark.parametrize("parent_attempted", [False, True])
def test_forked_hook_retains_state_without_cleanup_or_inherited_lock(
    kernel_env, framework, monkeypatch, finalize_first, parent_attempted
):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    fw, native, events = framework
    shutdown.install_shutdown(state)
    state.ensure_worker(config)
    hook = fw._pypto_kernel_shutdown
    hook._attempted = parent_attempted
    # A lock inherited from a vanished parent thread cannot be entered. An
    # unusable sentinel makes any attempt fail instead of hanging the test.
    monkeypatch.setattr(hook, "_lock", None)
    with monkeypatch.context() as child:
        child.setattr(shutdown.os, "getpid", lambda: hook.pid + 1)
        if finalize_first:
            fw._C._npu_shutdown(True)
        else:
            fw._C._npu_shutdown_synchronize()
        fw._C._npu_shutdown_synchronize()
        fw._C._npu_shutdown(True)
    first_teardown = ("framework-finalize", True) if finalize_first else "framework-sync"
    assert events == [("retain", state), first_teardown, "framework-sync", ("framework-finalize", True)]
    assert state.state is KernelState.READY and not calls.closes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
