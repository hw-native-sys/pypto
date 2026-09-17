# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One lazy kernel Worker per process, with shared initialization and registration."""

import itertools
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future
from enum import Enum
from types import SimpleNamespace
from typing import Any

from pypto.runtime._execution_mode import claim_kernel_mode

from .abi import KernelConfig, _NativeWorker
from .callable import KernelRegistration


class KernelState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


_generations = itertools.count(1)
_NOT_INITIALIZED = (
    "Kernel execution is not initialized; call pypto.torch.init(...) once in this process "
    "before the first kernel call, outside graph capture"
)


class _ProcessKernelState:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.generation = next(_generations)
        self.state = KernelState.UNINITIALIZED
        self.config: KernelConfig | None = None
        self._condition = threading.Condition()
        self._initializing_thread: threading.Thread | None = None
        self._closing_thread: threading.Thread | None = None
        self._stop_requested = False
        self._worker: Any = None
        self._failure: BaseException | None = None
        self._registrations: dict[bytes, KernelRegistration] = {}
        self._preparing: dict[bytes, Future[KernelRegistration]] = {}
        self._prepare_threads: dict[bytes, threading.Thread] = {}
        self._submissions: list[Any] = []
        self._submission_lock = threading.Lock()
        self._graph_lifecycle: Any = None

    def _check_pid(self) -> None:
        if self.pid != os.getpid():
            raise RuntimeError("Kernel state belongs to another PID; use a fresh spawned process")

    def _require_ready(self) -> None:
        if self.state is not KernelState.READY or self._stop_requested:
            status = (
                "stopping" if self._stop_requested and self.state is KernelState.READY else self.state.value
            )
            raise RuntimeError(f"Kernel Worker is {status}, expected ready") from self._failure

    def ensure_worker(self, config: KernelConfig) -> Any:
        """Initialize for ``pypto.torch.init``; incompatible requests never create another Worker."""
        self._check_pid()
        with self._condition:
            if self._stop_requested:
                raise RuntimeError("Kernel Worker is closing or closed; new initialization is disabled")
            if self.config is not None and self.config != config:
                raise ValueError(
                    f"Kernel Worker configuration conflict: bound {self.config}, requested {config}"
                )
            while self.state is KernelState.INITIALIZING:
                if self._initializing_thread is threading.current_thread():
                    raise RuntimeError("Reentrant kernel Worker initialization is not supported")
                self._condition.wait()
            if self.state is KernelState.READY:
                self._require_ready()
                return self._worker
            if self.state is not KernelState.UNINITIALIZED:
                self._require_ready()
            claim_kernel_mode()
            self.config = config
            self._initializing_thread = threading.current_thread()
            self.state = KernelState.INITIALIZING
        try:
            self._worker = _NativeWorker(config, self)
            self._worker.init(config)
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                self.state = KernelState.FAILED
                self._condition.notify_all()
            raise
        with self._condition:
            if self.state is KernelState.INITIALIZING:
                self.state = KernelState.READY
            self._condition.notify_all()
            if self.state is not KernelState.READY:
                self._require_ready()
            return self._worker

    def require_config(self) -> KernelConfig:
        """Return the configuration bound by ``pypto.torch.init``; kernel calls never initialize."""
        self._check_pid()
        with self._condition:
            while (
                self.state is KernelState.INITIALIZING
                and self._initializing_thread is not threading.current_thread()
            ):
                self._condition.wait()
            if self.state is KernelState.UNINITIALIZED and not self._stop_requested:
                raise RuntimeError(_NOT_INITIALIZED)
            self._require_ready()
            if self.config is None:
                raise RuntimeError("Internal error: a ready kernel Worker has no bound configuration")
            return self.config

    def bound_worker(self, config: KernelConfig) -> Any:
        """Return the initialized Worker for a matching configuration without initializing one."""
        bound = self.require_config()
        if bound != config:
            raise ValueError(f"Kernel Worker configuration conflict: bound {bound}, requested {config}")
        with self._condition:
            self._require_ready()
            return self._worker

    def ensure_callable(self, artifact: Any, config: KernelConfig) -> KernelRegistration:
        """Prepare an entire artifact once; concurrent callers share success or failure."""
        self._check_pid()
        abi = artifact.kernel_abi
        if (abi.platform, abi.runtime) != (config.platform, config.runtime):
            raise ValueError("Kernel artifact platform/runtime does not match the Worker configuration")
        worker = self.bound_worker(config)
        identity = artifact.identity()
        with self._condition:
            self._require_ready()
            if identity in self._registrations:
                return self._registrations[identity]
            future = self._preparing.get(identity)
            leader = future is None
            if future is None:
                future = self._preparing[identity] = Future()
                self._prepare_threads[identity] = threading.current_thread()
            elif self._prepare_threads[identity] is threading.current_thread():
                raise RuntimeError("Reentrant prepare of the same kernel callable is not supported")
        if not leader:
            return future.result()
        try:
            callable_ = artifact.load()
            handle = worker.prepare(callable_)
            registration = KernelRegistration(
                identity, self.pid, self.generation, handle, self, callable_, artifact
            )
            with self._condition:
                self._registrations[identity] = registration
            future.set_result(registration)
            return registration
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._condition:
                del self._preparing[identity]
                del self._prepare_threads[identity]
                self._condition.notify_all()

    def require_callable(self, artifact: Any, config: KernelConfig) -> KernelRegistration:
        """Look up completed warmup without initializing, loading or registering."""
        self._check_pid()
        identity = artifact.loaded_identity()
        with self._condition:
            if self.state is KernelState.UNINITIALIZED and not self._stop_requested:
                raise RuntimeError(_NOT_INITIALIZED)
            self._require_ready()
            if self.config != config:
                raise ValueError(
                    f"Kernel Worker configuration conflict: bound {self.config}, requested {config}"
                )
            if identity is not None:
                registration = self._registrations.get(identity)
                if registration is not None:
                    return registration
            raise RuntimeError("Kernel capture requires warmup outside capture for this specialization")

    def require_registration(self, registration: KernelRegistration) -> None:
        self._check_pid()
        with self._condition:
            self._require_ready()
            if (
                registration.owner is not self
                or registration.pid != self.pid
                or registration.generation != self.generation
                or self._registrations.get(registration.identity) is not registration
            ):
                raise RuntimeError("Kernel registration does not belong to this live Worker generation")

    def _fail_admission(self, error: BaseException) -> None:
        """Retain the first observed submission error without interrupting close."""
        with self._condition:
            if self._failure is None:
                self._failure = error
            if self.state is not KernelState.CLOSING:
                self.state = KernelState.FAILED

    def submit(self, registration: KernelRegistration, prepare: Callable[[Any], Any]) -> None:
        """Serialize admission and retain native tickets before any enqueue can fail."""
        self._check_pid()
        with self._submission_lock:
            self.require_registration(registration)
            try:
                self._submissions = [ticket for ticket in self._submissions if not ticket.done()]
            except BaseException as exc:
                self._fail_admission(exc)
                raise RuntimeError("Kernel Worker failed after an earlier asynchronous launch error") from exc
            ticket = prepare(self._worker)
            self._submissions.append(ticket)
            try:
                ticket.enqueue()
            except BaseException as exc:
                self._fail_admission(exc)
                raise

    def drain(self) -> None:
        """Wait for admitted eager work; failed tickets remain owned for diagnosis."""
        self._check_pid()
        with self._submission_lock:
            try:
                for ticket in self._submissions:
                    ticket.wait()
            except BaseException as exc:
                self._fail_admission(exc)
                raise
            self._submissions.clear()

    def close(self) -> None:
        """Stop admission, join in-flight operations and finalize on the native owner."""
        self._check_pid()
        with self._condition:
            current = threading.current_thread()
            if (self.state is KernelState.INITIALIZING and self._initializing_thread is current) or (
                self.state is KernelState.CLOSING and self._closing_thread is current
            ):
                raise RuntimeError(f"Cannot close kernel Worker while {self.state.value} on this thread")
            if current in self._prepare_threads.values():
                raise RuntimeError("Cannot close kernel Worker from within prepare")
            self._stop_requested = True
            self._condition.notify_all()
            while self.state in (KernelState.INITIALIZING, KernelState.CLOSING):
                self._condition.wait()
            if self.state is KernelState.CLOSED:
                return
            self._closing_thread = current
            self.state = KernelState.CLOSING
            self._condition.notify_all()
            while self._preparing:
                self._condition.wait()
        submission_error: BaseException | None = None
        try:
            with self._submission_lock:
                if self._graph_lifecycle is not None:
                    self._graph_lifecycle.close()
                for ticket in self._submissions:
                    try:
                        ticket.wait()
                    except BaseException as exc:
                        submission_error = submission_error or exc
                        # A missing caller join requires full device quiescence.
                        # Failure here keeps every owner and prevents finalize.
                        ticket.quiesce()
                if self._worker is not None:
                    self._worker.close()
                self._submissions.clear()
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                self.state = KernelState.FAILED
                self._closing_thread = None
                self._condition.notify_all()
            raise
        with self._condition:
            self._registrations.clear()
            self._worker = None
            self._failure = None
            self.state = KernelState.CLOSED
            self._closing_thread = None
            self._condition.notify_all()
        if submission_error is not None:
            raise submission_error

    def retain_shutdown_failure(self, error: BaseException) -> None:
        """Stop admission without touching device resources after unsafe teardown."""
        self._check_pid()
        with self._condition:
            self._stop_requested = True
            if self.state is not KernelState.CLOSED:
                self._failure = error
                self.state = KernelState.FAILED
            self._condition.notify_all()


_process = SimpleNamespace(state=_ProcessKernelState())


def get_process_kernel_state() -> _ProcessKernelState:
    """Return the process-owned manager shared by all internal kernel consumers."""
    if _process.state.pid != os.getpid():
        if _process.state.state is not KernelState.UNINITIALIZED:
            _process.state._check_pid()
        # No native state existed at fork. Replace inherited synchronization too.
        _process.state = _ProcessKernelState()
    return _process.state
