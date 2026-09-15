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
from concurrent.futures import Future
from enum import Enum
from types import SimpleNamespace
from typing import Any

from pypto.runtime._execution_mode import claim_kernel_mode

from .abi import KernelConfig, _NativeWorker
from .callable import KernelRegistration, callable_identity


class KernelState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


_generations = itertools.count(1)


class _ProcessKernelState:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.generation = next(_generations)
        self.state = KernelState.UNINITIALIZED
        self.config: KernelConfig | None = None
        self._condition = threading.Condition()
        self._owner_thread: threading.Thread | None = None
        self._worker: Any = None
        self._failure: BaseException | None = None
        self._registrations: dict[bytes, KernelRegistration] = {}
        self._preparing: dict[bytes, Future[KernelRegistration]] = {}
        self._prepare_threads: dict[bytes, threading.Thread] = {}

    def _check_pid(self) -> None:
        if self.pid != os.getpid():
            raise RuntimeError("Kernel state belongs to another PID; use a fresh spawned process")

    def _require_ready(self) -> None:
        if self.state is not KernelState.READY:
            raise RuntimeError(f"Kernel Worker is {self.state.value}, expected ready") from self._failure

    def ensure_worker(self, config: KernelConfig) -> Any:
        """Share the first init result; incompatible requests never create another Worker."""
        self._check_pid()
        with self._condition:
            if self.config is not None and self.config != config:
                raise ValueError(
                    f"Kernel Worker configuration conflict: bound {self.config}, requested {config}"
                )
            while self.state is KernelState.INITIALIZING:
                if self._owner_thread is threading.current_thread():
                    raise RuntimeError("Reentrant kernel Worker initialization is not supported")
                self._condition.wait()
            if self.state is KernelState.READY:
                return self._worker
            if self.state is not KernelState.UNINITIALIZED:
                self._require_ready()
            claim_kernel_mode()
            self.config = config
            self._owner_thread = threading.current_thread()
            self.state = KernelState.INITIALIZING
        try:
            self._worker = _NativeWorker(config)
            self._worker.init(config)
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                self.state = KernelState.FAILED
                self._condition.notify_all()
            raise
        with self._condition:
            self.state = KernelState.READY
            self._condition.notify_all()
            return self._worker

    def ensure_callable(self, artifact: Any, config: KernelConfig) -> KernelRegistration:
        """Prepare an entire artifact once; concurrent callers share success or failure."""
        self._check_pid()
        abi = artifact.kernel_abi
        if (abi.platform, abi.runtime) != (config.platform, config.runtime):
            raise ValueError("Kernel artifact platform/runtime does not match the Worker configuration")
        worker = self.ensure_worker(config)
        callable_ = artifact.load()
        identity = callable_identity(callable_, abi)
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

    def close(self) -> None:
        """Internal terminal close; callers must already have drained launch/graph work."""
        self._check_pid()
        with self._condition:
            if self.state is KernelState.CLOSED:
                return
            if self._owner_thread is not None and self._owner_thread is not threading.current_thread():
                raise RuntimeError("Kernel Worker close must run on its init-owner thread")
            if self.state in (KernelState.INITIALIZING, KernelState.CLOSING):
                raise RuntimeError(f"Cannot close kernel Worker while {self.state.value}")
            if threading.current_thread() in self._prepare_threads.values():
                raise RuntimeError("Cannot close kernel Worker from within prepare")
            self.state = KernelState.CLOSING
            self._condition.notify_all()
            while self._preparing:
                self._condition.wait()
        try:
            if self._worker is not None:
                self._worker.close()
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                self.state = KernelState.FAILED
            raise
        with self._condition:
            self._registrations.clear()
            self._worker = None
            self.state = KernelState.CLOSED


_process = SimpleNamespace(state=_ProcessKernelState())


def get_process_kernel_state() -> _ProcessKernelState:
    """Return the process-owned manager shared by all internal kernel consumers."""
    if _process.state.pid != os.getpid():
        if _process.state.state is not KernelState.UNINITIALIZED:
            _process.state._check_pid()
        # No native state existed at fork. Replace inherited synchronization too.
        _process.state = _ProcessKernelState()
    return _process.state
