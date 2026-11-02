# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Process ownership and lifecycle regression tests with opaque public handles."""

import multiprocessing
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from pypto.runtime import _execution_mode
from pypto.runtime.kernel import context
from pypto.runtime.kernel.context import KernelState
from pypto.runtime.kernel.session import KernelConfig


def test_kernel_lookup_waits_for_concurrent_initialization(kernel_env, monkeypatch):
    state, config, calls, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )
    entered, release = threading.Event(), threading.Event()

    def init(self, config):
        entered.set()
        assert release.wait(5)
        calls.inits.append(config)

    monkeypatch.setattr(worker_cls, "init", init)
    with ThreadPoolExecutor(max_workers=2) as pool:
        initializing = pool.submit(state.ensure_worker, config)
        assert entered.wait(5)
        lookup = pool.submit(state.require_config)
        assert not lookup.done()
        release.set()
        assert lookup.result(timeout=5) == config
        initializing.result(timeout=5)
    assert len(calls.workers) == len(calls.inits) == 1
    state.close()


@pytest.mark.parametrize(
    "change", [{"device_id": 1}, {"runtime": "host_build_graph"}, {"aicpu_thread_num": 2}]
)
def test_existing_configuration_is_immutable(kernel_env, change):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    with pytest.raises(ValueError, match="configuration conflict"):
        state.ensure_worker(replace(config, **change))
    assert len(calls.workers) == len(calls.inits) == 1
    state.close()


def test_init_failure_is_terminal_and_retains_cleanup_owner(kernel_env, monkeypatch):
    state, config, calls, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )

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


def test_failed_prepare_is_not_published_and_can_retry(kernel_env, monkeypatch):
    state, config, calls, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )
    prepare = worker_cls.prepare

    def fail(self, callable_):
        raise RuntimeError("prepare rejected")

    monkeypatch.setattr(worker_cls, "prepare", fail)
    state.ensure_worker(config)
    with pytest.raises(RuntimeError, match="prepare rejected"):
        state.ensure_callable(kernel_env.artifact(), config)
    assert not state._registrations and not state._preparing
    assert state.state is KernelState.READY
    monkeypatch.setattr(worker_cls, "prepare", prepare)
    state.ensure_callable(kernel_env.artifact(), config).require_live()
    assert len(calls.prepares) == 1 and len(calls.workers) == 1
    state.close()


def test_concurrent_failed_prepare_shares_failure(kernel_env, monkeypatch):
    state, config, calls, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )
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
        first = pool.submit(state.ensure_callable, kernel_env.artifact(), config)
        assert entered.wait(5)
        second = pool.submit(state.ensure_callable, kernel_env.artifact(), config)
        try:
            assert joined.wait(5)
        finally:
            release.set()
        for future in (first, second):
            with pytest.raises(RuntimeError, match="shared rejection"):
                future.result(timeout=5)
    assert len(calls.prepares) == 1 and not state._registrations
    state.close()


def test_close_from_another_thread_invalidates_handles(kernel_env):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
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


def test_close_failure_retains_owner_and_registrations_for_retry(kernel_env, monkeypatch):
    state, config, calls, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
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


def test_uninitialized_close_does_no_native_work(kernel_env):
    state, _, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.close()
    assert state.state is KernelState.CLOSED
    assert not calls.workers and not calls.closes


def test_program_claim_blocks_kernel_before_construction(kernel_env):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    _execution_mode.claim_program_mode()
    with pytest.raises(RuntimeError, match="already claimed program"):
        state.ensure_worker(config)
    assert not calls.workers and state.state is KernelState.UNINITIALIZED


def test_close_waits_for_in_flight_prepare(kernel_env, monkeypatch):
    state, config, calls, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )
    state.ensure_worker(config)
    entered, release = threading.Event(), threading.Event()

    def prepare(self, callable_):
        entered.set()
        assert release.wait(5)
        return 1

    monkeypatch.setattr(worker_cls, "prepare", prepare)
    with ThreadPoolExecutor(max_workers=2) as pool:
        preparing = pool.submit(state.ensure_callable, kernel_env.artifact(), config)
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


def test_fork_rejects_state_before_touching_inherited_mutex(kernel_env, monkeypatch):
    state, config, _, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
    monkeypatch.setattr(context.os, "getpid", lambda: state.pid + 1)
    with pytest.raises(RuntimeError, match="PID"):
        context.get_process_kernel_state()
    with pytest.raises(RuntimeError, match="PID"):
        registration.require_live()
    with pytest.raises(RuntimeError, match="PID"):
        state.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_fork_resets_unused_manager_before_concurrent_child_initialization(kernel_env, monkeypatch):
    parent_state = kernel_env.state
    created = []
    manager_type = context._ProcessKernelState

    def create_manager():
        manager = manager_type()
        created.append(manager)
        return manager

    monkeypatch.setattr(context, "_ProcessKernelState", create_manager)

    def run_child():
        # Check before any lookup: the at-fork handler must publish exactly one
        # fresh manager, rather than letting first-use threads compete to do so.
        child_state = context._process.state
        assert child_state is not parent_state and child_state.pid == os.getpid()
        assert created == [child_state]
        start = threading.Barrier(2)

        def initialize():
            start.wait(timeout=5)
            manager = context.get_process_kernel_state()
            return manager, manager.ensure_worker(kernel_env.config)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(initialize) for _ in range(2)]
            results = [future.result(timeout=5) for future in futures]
        assert all(manager is child_state for manager, _ in results)
        assert results[0][1] is results[1][1]
        assert created == [child_state]
        assert len(kernel_env.calls.workers) == len(kernel_env.calls.inits) == 1
        child_state.close()

    # Both locks are held across fork; child initialization must use new locks.
    with parent_state._condition, parent_state._submission_lock:
        child = multiprocessing.get_context("fork").Process(target=run_child)
        child.start()
        try:
            child.join(timeout=15)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
    assert context.get_process_kernel_state() is parent_state
    assert parent_state.state is KernelState.UNINITIALIZED
    assert not created and not kernel_env.calls.workers


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
@pytest.mark.parametrize("state", [state for state in KernelState if state is not KernelState.UNINITIALIZED])
def test_fork_preserves_used_manager_for_pid_rejection(kernel_env, state):
    parent_state = kernel_env.state
    parent_state.state = state

    def run_child():
        assert context._process.state is parent_state
        with pytest.raises(RuntimeError, match="PID"):
            context.get_process_kernel_state()

    with parent_state._condition, parent_state._submission_lock:
        child = multiprocessing.get_context("fork").Process(target=run_child)
        child.start()
        try:
            child.join(timeout=10)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)


def test_registration_retains_worker_after_operator_is_deleted(kernel_env):
    state, config, _, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
    owner = weakref.ref(state)
    del state
    assert registration.owner is owner()
    registration.require_live()
    registration.owner.close()


def test_reentrant_prepare_and_close_are_rejected(kernel_env, monkeypatch):
    state, config, _, worker_cls = (
        kernel_env.state,
        kernel_env.config,
        kernel_env.calls,
        kernel_env.worker_cls,
    )

    def prepare(self, callable_):
        with pytest.raises(RuntimeError, match="Reentrant prepare"):
            state.ensure_callable(kernel_env.artifact(), config)
        with pytest.raises(RuntimeError, match="within prepare"):
            state.close()
        return 1

    monkeypatch.setattr(worker_cls, "prepare", prepare)
    state.ensure_worker(config)
    state.ensure_callable(kernel_env.artifact(), config).require_live()
    state.close()


@pytest.mark.parametrize("first", ["program", "kernel"])
def test_process_mode_claim_is_mutually_exclusive(kernel_env, first):
    claims = {"program": _execution_mode.claim_program_mode, "kernel": _execution_mode.claim_kernel_mode}
    claims[first]()
    claims[first]()
    other = "kernel" if first == "program" else "program"
    with pytest.raises(RuntimeError, match=f"already claimed {first}"):
        claims[other]()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
@pytest.mark.parametrize("parent_mode", [None, "program", "kernel"])
@pytest.mark.parametrize("child_modes", [("program", "kernel"), ("program", "program"), ("kernel", "kernel")])
def test_fork_mode_claims_share_one_child_gate(kernel_env, parent_mode, child_modes):
    claims = {"program": _execution_mode.claim_program_mode, "kernel": _execution_mode.claim_kernel_mode}
    parent_gate = _execution_mode._gate
    if parent_mode is not None:
        claims[parent_mode]()
    original_mode = parent_gate.mode
    parent_lock = parent_gate.lock

    def run_child():
        gate = _execution_mode._gate
        if parent_mode == "kernel":
            assert gate is parent_gate and gate.pid != os.getpid()
        else:
            # No claim has run yet: the at-fork callback must already have
            # reset the gate and replaced its inherited, locked mutex.
            assert gate is parent_gate and gate.pid == os.getpid()
            assert gate.mode is None and gate.lock is not parent_lock
        start = threading.Barrier(2)

        def claim(mode):
            start.wait(timeout=5)
            try:
                claims[mode]()
            except RuntimeError as exc:
                return mode, str(exc)
            return mode, None

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(claim, mode) for mode in child_modes]
            results = [future.result(timeout=5) for future in futures]
        assert _execution_mode._gate is gate
        accepted = [mode for mode, error in results if error is None]
        rejected = [error for _, error in results if error is not None]
        if parent_mode == "kernel":
            assert not accepted and len(rejected) == 2
            assert all("use spawn" in error for error in rejected)
            assert gate.mode is original_mode
        else:
            expected = 2 if child_modes[0] == child_modes[1] else 1
            assert len(accepted) == expected
            winner = accepted[0]
            assert gate.mode is not None and gate.mode.value == winner
            assert all(f"already claimed {winner}" in error for error in rejected)
            # The winning claim remains sticky; repeated calls cannot reset it.
            claims[winner]()
            other = "kernel" if winner == "program" else "program"
            with pytest.raises(RuntimeError, match=f"already claimed {winner}"):
                claims[other]()

    with parent_gate.lock:
        child = multiprocessing.get_context("fork").Process(target=run_child)
        child.start()
        try:
            child.join(timeout=15)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
    assert _execution_mode._gate is parent_gate and parent_gate.mode is original_mode


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


def test_completed_launch_is_reaped_on_next_call(kernel_env):
    state, config, _, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
    ticket = _Ticket([])
    state.submit(registration, lambda worker, ticket=ticket: ticket)
    ticket.finished = True
    owner = weakref.ref(ticket)
    del ticket
    state.submit(registration, lambda worker: _Ticket([]))
    assert owner() is None
    state.close()


def test_partial_enqueue_failure_retains_owners_and_refuses_new_work(kernel_env):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
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


def test_close_waits_for_in_progress_admission(kernel_env, monkeypatch):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
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


def test_prepared_specializations_retain_registration_until_close(kernel_env):
    state, config, calls, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    key = (object(), "specialization")
    value = kernel_env.artifact()
    registration = state.ensure_callable(value, config)
    assert state.find_specialization(key, config) is None
    state.publish_specialization(key, registration)
    assert state.find_specialization(key, config) is registration
    assert registration.artifact is value
    assert state.find_specialization((object(), "specialization"), config) is None
    with pytest.raises(ValueError, match="configuration conflict"):
        state.find_specialization(key, replace(config, device_id=1))
    state.close()
    assert not state._specializations
    with pytest.raises(RuntimeError, match="closed"):
        state.publish_specialization(key, registration)
    with pytest.raises(RuntimeError, match="closed"):
        state.find_specialization(key, config)
    assert len(calls.prepares) == 1


def test_specialization_cannot_publish_or_resolve_foreign_registration(kernel_env):
    state, config, _, _ = (kernel_env.state, kernel_env.config, kernel_env.calls, kernel_env.worker_cls)
    state.ensure_worker(config)
    registration = state.ensure_callable(kernel_env.artifact(), config)
    stale = replace(registration, generation=registration.generation + 1)
    with pytest.raises(RuntimeError, match="live Worker generation"):
        state.publish_specialization("key", stale)
    assert not state._specializations
    # Defense in depth: a corrupted/stale index entry must not bypass ownership.
    state._specializations["key"] = stale
    with pytest.raises(RuntimeError, match="live Worker generation"):
        state.find_specialization("key", config)
    state.close()


def test_kernel_config_normalizes_ring_sizing():
    # Scalar sizing broadcasts to all four rings, matching RunOptions.
    config = KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, ring_heap=64 * 1024 * 1024)
    assert config.ring_heap == (64 * 1024 * 1024,) * 4
    assert config.ring_task_window is None and config.ring_dep_pool is None
    # Snapshot mutable per-ring lists so repeated initialization compares stable values.
    config = KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, ring_task_window=[16, 32, 128, 256])
    assert config.ring_task_window == (16, 32, 128, 256)
    assert hash(config) == hash(
        KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, ring_task_window=(16, 32, 128, 256))
    )
    # Explicit sizing must remain distinct from runtime defaults.
    assert KernelConfig("a2a3", "tensormap_and_ringbuffer", 0) != config


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("ring_heap", [1, 2, 3], "exactly 4 per-ring values"),
        ("ring_task_window", (4, 8, 16, 32, 64), "exactly 4 per-ring values"),
        ("ring_dep_pool", 0, "positive ints"),
        ("ring_heap", [1024, 1024, -1, 1024], "positive ints"),
    ],
)
def test_kernel_config_rejects_bad_ring_sizing(field, value, match):
    with pytest.raises(ValueError, match=match):
        KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, **{field: value})


def test_concurrent_registration_reuses_shared_artifact_key(kernel_env):
    env = kernel_env

    def prepare(_):
        env.state.ensure_worker(env.config)
        return env.state.ensure_callable(env.artifact(), env.config)

    with ThreadPoolExecutor(max_workers=8) as pool:
        registrations = list(pool.map(prepare, range(32)))
    assert len(env.calls.workers) == len(env.calls.inits) == len(env.calls.prepares) == 1
    assert all(reg is registrations[0] for reg in registrations)
    assert env.state.require_callable(env.artifact(), env.config) is registrations[0]
    env.state.close()


def test_reentrant_init_is_rejected_without_deadlock(kernel_env, monkeypatch):
    env = kernel_env
    monkeypatch.setattr(env.worker_cls, "init", lambda self, config: env.state.ensure_worker(config))
    with pytest.raises(RuntimeError, match="Reentrant"):
        env.state.ensure_worker(env.config)
    assert env.state.state is KernelState.FAILED
    env.state.close()


def test_registration_rejects_program_capability_and_runtime_mismatch(kernel_env):
    from pypto._artifact_contract import ExecutionCapabilities  # noqa: PLC0415

    env = kernel_env
    env.state.ensure_worker(env.config)
    value = env.artifact()
    value.metadata = replace(value.metadata, execution_capabilities=ExecutionCapabilities())
    with pytest.raises(ValueError):
        env.state.ensure_callable(value, env.config)
    value = env.artifact()
    value.load()
    value._runtime_name = "host_build_graph"
    with pytest.raises(ValueError, match="runtime"):
        env.state.ensure_callable(value, env.config)
    assert not env.calls.prepares
    env.state.close()


def test_register_busy_does_not_wait_or_drain(kernel_env, monkeypatch):
    env = kernel_env
    env.state.ensure_worker(env.config)
    error = RuntimeError("registry busy")

    def register(self, callable_):
        raise error

    monkeypatch.setattr(env.worker_cls, "prepare", register)
    monkeypatch.setattr(env.state, "drain", lambda: pytest.fail("unexpected drain"))
    with pytest.raises(RuntimeError) as caught:
        env.state.ensure_callable(env.artifact(), env.config)
    assert caught.value is error and env.state.state is KernelState.READY
    env.state.close()


@pytest.mark.parametrize("runtime_name", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_real_compilation_and_metadata_reload_reach_registration(
    kernel_env, monkeypatch, tmp_path, runtime_name, device_runner
):
    import pypto.language as pl  # noqa: PLC0415
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415
    from pypto.ir.compiled_artifact import CompiledArtifact  # noqa: PLC0415
    from pypto.pypto_core import passes  # noqa: PLC0415

    @pl.program
    class Add:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [16, 16])
                pl.store(pl.add(tile, tile), [0, 0], b)
            return b

    runtime = (
        passes.RuntimeKind.HOST_BUILD_GRAPH
        if runtime_name == "host_build_graph"
        else passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER
    )
    with passes.PassContext([], runtime=runtime):
        artifact = _compile_artifact(
            Add,
            output_dir=str(tmp_path / "compiled"),
            contract=ArtifactExecutionMode.KERNEL,
            skip_ptoas=True,
            platform="a2a3",
        )
    assert artifact.metadata is not None
    assert artifact.metadata.return_aliases == (1,)
    assert list(artifact.output_dir.rglob("*.pto"))
    restored = CompiledArtifact.from_dir(artifact.output_dir)
    assert restored.metadata == artifact.metadata
    # Only assembly/native construction is mocked, after real compilation and restoration.
    target = object()
    runner = device_runner

    loads = []

    def assemble(directory, platform):
        loads.append((directory, platform))
        return target, runtime_name, {}

    monkeypatch.setattr(runner, "_compile_and_assemble", assemble)
    env = kernel_env
    config = replace(env.config, runtime=runtime_name)
    env.state.ensure_worker(config)
    registration = env.state.ensure_callable(restored, config)
    assert registration.callable is target and registration.artifact is restored
    assert env.state.ensure_callable(restored, config) is registration
    assert loads == [(artifact.output_dir, "a2a3")]
    assert env.calls.prepares == [target]
    env.state.close()


def test_reusing_private_directory_does_not_return_old_registration(kernel_env):
    env = kernel_env
    env.state.ensure_worker(env.config)
    first, second = env.artifact(), env.artifact()
    for artifact in (first, second):
        artifact.load()
        artifact._artifact_runtime = None
    second._chip_callable = object()
    assert first.output_dir == second.output_dir
    before = env.state.ensure_callable(first, env.config)
    after = env.state.ensure_callable(second, env.config)
    assert before is not after and after.callable is second.chip_callable
    assert env.state.ensure_callable(second, env.config) is after
    assert len(env.calls.prepares) == 2
    env.state.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
