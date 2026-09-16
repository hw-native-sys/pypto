# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel process ownership, concurrency, failure and shutdown contracts."""

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pypto._kernel_abi import KernelABI
from pypto.runtime import _execution_mode
from pypto.runtime.kernel import context
from pypto.runtime.kernel.abi import KernelConfig
from pypto.runtime.kernel.context import KernelState


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(_execution_mode, "_gate", _execution_mode._ModeGate())
    calls = SimpleNamespace(workers=[], inits=[], prepares=[], closes=[])

    class FakeWorker:
        def __init__(self, config, state):
            calls.workers.append(self)

        def init(self, config):
            calls.inits.append(config)

        def prepare(self, callable_):
            calls.prepares.append(callable_)
            return len(calls.prepares)

        def close(self):
            calls.closes.append(self)

    monkeypatch.setattr(context, "_NativeWorker", FakeWorker)
    monkeypatch.setattr(context, "callable_identity", lambda callable_, abi: abi.binary_tag() + callable_)
    state = context._ProcessKernelState()
    monkeypatch.setattr(context, "_process", SimpleNamespace(state=state))
    return state, KernelConfig("a2a3", "tensormap_and_ringbuffer", 0), calls, FakeWorker


def artifact(key=b"kernel"):
    return SimpleNamespace(kernel_abi=KernelABI("a2a3", "tensormap_and_ringbuffer", ()), load=lambda: key)


def test_concurrent_operators_share_one_worker_and_registration(setup):
    state, config, calls, _ = setup
    assert context.get_process_kernel_state() is state
    with ThreadPoolExecutor(max_workers=8) as pool:
        registrations = list(pool.map(lambda _: state.ensure_callable(artifact(), config), range(32)))
    assert len(calls.workers) == len(calls.inits) == len(calls.prepares) == 1
    assert all(reg is registrations[0] for reg in registrations)
    second = state.ensure_callable(artifact(b"different code"), config)
    assert second is not registrations[0]
    assert len(calls.prepares) == 2 and len(calls.workers) == 1
    registrations[0].require_live()


@pytest.mark.parametrize(
    "change", [{"device_id": 1}, {"runtime": "host_build_graph"}, {"aicpu_thread_num": 2}]
)
def test_existing_configuration_is_immutable(setup, change):
    state, config, calls, _ = setup
    state.ensure_worker(config)
    with pytest.raises(ValueError, match="configuration conflict"):
        state.ensure_worker(replace(config, **change))
    assert len(calls.workers) == len(calls.inits) == 1
    state.close()


def test_artifact_target_mismatch_does_not_initialize(setup):
    state, config, calls, _ = setup
    with pytest.raises(ValueError, match="platform/runtime"):
        state.ensure_callable(artifact(), replace(config, platform="a5"))
    assert not calls.workers


def test_init_failure_is_terminal_and_retains_cleanup_owner(setup, monkeypatch):
    state, config, calls, worker_cls = setup

    def fail(self, config):
        raise RuntimeError("partial init")

    monkeypatch.setattr(worker_cls, "init", fail)
    with pytest.raises(RuntimeError, match="partial init"):
        state.ensure_worker(config)
    with pytest.raises(RuntimeError, match="failed") as error:
        state.ensure_worker(config)
    assert str(error.value.__cause__) == "partial init"
    assert len(calls.workers) == 1
    state.close()
    assert calls.closes == calls.workers
    assert state.state is KernelState.CLOSED


def test_failed_prepare_is_not_published_and_can_retry(setup, monkeypatch):
    state, config, calls, worker_cls = setup
    prepare = worker_cls.prepare

    def fail(self, callable_):
        raise RuntimeError("prepare rejected")

    monkeypatch.setattr(worker_cls, "prepare", fail)
    with pytest.raises(RuntimeError, match="prepare rejected"):
        state.ensure_callable(artifact(), config)
    assert not state._registrations and not state._preparing
    assert state.state is KernelState.READY
    monkeypatch.setattr(worker_cls, "prepare", prepare)
    state.ensure_callable(artifact(), config).require_live()
    assert len(calls.prepares) == 1 and len(calls.workers) == 1
    state.close()


def test_concurrent_failed_prepare_shares_failure(setup, monkeypatch):
    state, config, calls, worker_cls = setup
    state.ensure_worker(config)
    entered, joined, release = threading.Event(), threading.Event(), threading.Event()
    real_future = context.Future

    class ObservedFuture(real_future):
        def result(self, timeout=None):
            joined.set()
            return super().result(timeout)

    def fail(self, callable_):
        calls.prepares.append(callable_)
        entered.set()
        assert release.wait(5)
        raise RuntimeError("shared rejection")

    monkeypatch.setattr(context, "Future", ObservedFuture)
    monkeypatch.setattr(worker_cls, "prepare", fail)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(state.ensure_callable, artifact(), config)
        assert entered.wait(5)
        second = pool.submit(state.ensure_callable, artifact(), config)
        try:
            assert joined.wait(5)
        finally:
            release.set()
        for future in (first, second):
            with pytest.raises(RuntimeError, match="shared rejection"):
                future.result(timeout=5)
    assert len(calls.prepares) == 1 and not state._registrations
    state.close()


def test_close_from_another_thread_invalidates_handles(setup):
    state, config, calls, _ = setup
    registration = state.ensure_callable(artifact(), config)
    registration.require_live()
    with pytest.raises(RuntimeError, match="generation"):
        state.require_registration(replace(registration, generation=registration.generation + 1))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(state.close).result(timeout=5)
    state.close()
    assert len(calls.closes) == 1
    with pytest.raises(RuntimeError, match="closed"):
        registration.require_live()
    with pytest.raises(RuntimeError, match="closed"):
        state.ensure_worker(config)


def test_close_failure_retains_owner_and_registrations_for_retry(setup, monkeypatch):
    state, config, calls, worker_cls = setup
    registration = state.ensure_callable(artifact(), config)
    close = worker_cls.close

    def fail(self):
        raise RuntimeError("cleanup pending")

    monkeypatch.setattr(worker_cls, "close", fail)
    with pytest.raises(RuntimeError, match="cleanup pending"):
        state.close()
    assert state._registrations[registration.identity] is registration
    assert state._worker is calls.workers[0]
    with pytest.raises(RuntimeError, match="failed"):
        registration.require_live()
    monkeypatch.setattr(worker_cls, "close", close)
    state.close()
    assert state.state is KernelState.CLOSED and not state._registrations


def test_uninitialized_close_does_no_native_work(setup):
    state, _, calls, _ = setup
    state.close()
    assert state.state is KernelState.CLOSED
    assert not calls.workers and not calls.closes


def test_program_claim_blocks_kernel_before_construction(setup):
    state, config, calls, _ = setup
    _execution_mode.claim_program_mode()
    with pytest.raises(RuntimeError, match="already claimed program"):
        state.ensure_worker(config)
    assert not calls.workers and state.state is KernelState.UNINITIALIZED


def test_close_waits_for_in_flight_prepare(setup, monkeypatch):
    state, config, calls, worker_cls = setup
    state.ensure_worker(config)
    entered, release = threading.Event(), threading.Event()

    def prepare(self, callable_):
        entered.set()
        assert release.wait(5)
        return 1

    monkeypatch.setattr(worker_cls, "prepare", prepare)
    with ThreadPoolExecutor(max_workers=2) as pool:
        preparing = pool.submit(state.ensure_callable, artifact(), config)
        assert entered.wait(5)

        def observe_close():
            # Condition.wait() releases the lock while close drains prepare.
            with state._condition:
                assert state._condition.wait_for(lambda: state.state is KernelState.CLOSING, timeout=5)
                assert not calls.closes
            release.set()

        observer = pool.submit(observe_close)
        state.close()
        observer.result(timeout=5)
        registration = preparing.result(timeout=5)
    with pytest.raises(RuntimeError, match="closed"):
        registration.require_live()
    assert len(calls.closes) == 1


def test_fork_rejects_state_before_touching_inherited_mutex(setup, monkeypatch):
    state, config, _, _ = setup
    registration = state.ensure_callable(artifact(), config)
    monkeypatch.setattr(context.os, "getpid", lambda: state.pid + 1)
    with pytest.raises(RuntimeError, match="PID"):
        context.get_process_kernel_state()
    with pytest.raises(RuntimeError, match="PID"):
        registration.require_live()
    with pytest.raises(RuntimeError, match="PID"):
        state.close()


def test_registration_retains_worker_after_operator_is_deleted(setup):
    state, config, _, _ = setup
    registration = state.ensure_callable(artifact(), config)
    owner = weakref.ref(state)
    del state
    assert registration.owner is owner()
    registration.require_live()
    registration.owner.close()


def test_reentrant_prepare_and_close_are_rejected(setup, monkeypatch):
    state, config, _, worker_cls = setup

    def prepare(self, callable_):
        with pytest.raises(RuntimeError, match="Reentrant prepare"):
            state.ensure_callable(artifact(), config)
        with pytest.raises(RuntimeError, match="within prepare"):
            state.close()
        return 1

    monkeypatch.setattr(worker_cls, "prepare", prepare)
    state.ensure_callable(artifact(), config).require_live()
    state.close()


@pytest.mark.parametrize("first", ["program", "kernel"])
def test_process_mode_claim_is_mutually_exclusive(setup, first):
    claims = {"program": _execution_mode.claim_program_mode, "kernel": _execution_mode.claim_kernel_mode}
    claims[first]()
    claims[first]()
    other = "kernel" if first == "program" else "program"
    with pytest.raises(RuntimeError, match=f"already claimed {first}"):
        claims[other]()


class _Ticket:
    def __init__(self, calls, error=None):
        self.calls = calls
        self.error = error
        self.finished = False

    def enqueue(self):
        self.calls.append("enqueue")
        if self.error:
            raise self.error

    def done(self):
        return self.finished

    def quiesce(self):
        self.wait()

    def wait(self):
        self.calls.append("wait")
        if self.error:
            raise self.error
        self.finished = True


def test_launch_ownership_reaping_and_close_order(setup, monkeypatch):
    state, config, calls, _ = setup
    registration = state.ensure_callable(artifact(), config)
    order = []
    ticket = _Ticket(order)
    owner = weakref.ref(ticket)
    state.submit(registration, lambda worker, ticket=ticket: ticket)
    del ticket
    assert owner() is not None
    second = _Ticket(order)
    state.submit(registration, lambda worker: second)
    assert order == ["enqueue", "enqueue"]
    monkeypatch.setattr(calls.workers[0], "close", lambda: order.append("close"))
    state.close()
    assert order == ["enqueue", "enqueue", "wait", "wait", "close"]
    assert owner() is None


def test_completed_launch_is_reaped_on_next_call(setup):
    state, config, _, _ = setup
    registration = state.ensure_callable(artifact(), config)
    ticket = _Ticket([])
    state.submit(registration, lambda worker, ticket=ticket: ticket)
    ticket.finished = True
    owner = weakref.ref(ticket)
    del ticket
    state.submit(registration, lambda worker: _Ticket([]))
    assert owner() is None
    state.close()


def test_partial_enqueue_failure_retains_owners_and_refuses_new_work(setup):
    state, config, calls, _ = setup
    registration = state.ensure_callable(artifact(), config)
    ticket = _Ticket([], RuntimeError("partial enqueue"))
    owner = weakref.ref(ticket)
    with pytest.raises(RuntimeError, match="partial enqueue"):
        state.submit(registration, lambda worker, ticket=ticket: ticket)
    del ticket
    assert owner() is not None
    with pytest.raises(RuntimeError, match="failed"):
        state.submit(registration, lambda worker: pytest.fail("must not submit"))
    with pytest.raises(RuntimeError, match="partial enqueue"):
        state.close()
    assert not calls.closes and owner() is not None


def test_close_waits_for_in_progress_admission(setup, monkeypatch):
    state, config, calls, _ = setup
    registration = state.ensure_callable(artifact(), config)
    entered, release = threading.Event(), threading.Event()
    order = []
    ticket = _Ticket(order)

    def enqueue():
        entered.set()
        assert release.wait(5)
        order.append("enqueue")

    monkeypatch.setattr(ticket, "enqueue", enqueue)
    with ThreadPoolExecutor(max_workers=2) as pool:
        submit = pool.submit(state.submit, registration, lambda worker, ticket=ticket: ticket)
        assert entered.wait(5)
        unblock = pool.submit(lambda: (release.set()))
        state.close()
        submit.result()
        unblock.result()
    assert order == ["enqueue", "wait"]
    assert len(calls.closes) == 1


def test_close_reports_submission_error_after_proven_quiescence(setup, monkeypatch):
    state, config, calls, _ = setup
    registration = state.ensure_callable(artifact(), config)
    ticket = _Ticket([], RuntimeError("native rejection"))
    with pytest.raises(RuntimeError, match="native rejection"):
        state.submit(registration, lambda worker: ticket)
    quiescence = []
    monkeypatch.setattr(ticket, "quiesce", lambda: quiescence.append("all streams idle"))
    with pytest.raises(RuntimeError, match="native rejection"):
        state.close()
    assert quiescence == ["all streams idle"]
    assert len(calls.closes) == 1 and not state._submissions
    assert state.state is KernelState.CLOSED
    state.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
