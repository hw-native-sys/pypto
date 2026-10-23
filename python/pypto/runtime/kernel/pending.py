# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ownership of a queued callback, its published RunHandle and external sources."""

import threading
from collections.abc import Callable
from typing import Any

from .callable import KernelRegistration


class PendingTicket:
    """Retain resources before enqueue and retire them only after RunHandle.wait.

    The queue adapter calls publish/fail from its callback. Those methods
    never acquire process admission or lifecycle locks, so close can await the
    callback without preventing it from publishing. ``release`` closes borrowed
    sources after run retirement and must support retry after a partial failure.
    """

    def __init__(
        self,
        registration: KernelRegistration,
        *,
        enqueue: Callable[["PendingTicket"], None],
        synchronize: Callable[[], None],
        release: Callable[[], None],
        owners: tuple[Any, ...],
    ) -> None:
        self.registration = registration
        self._enqueue: Callable[[PendingTicket], None] | None = enqueue
        self._synchronize: Callable[[], None] | None = synchronize
        self._release: Callable[[], None] | None = release
        self._owners = owners
        self._callback: Any = None
        self._published = threading.Event()
        self._publish_lock = threading.Lock()
        self._retire_lock = threading.Lock()
        self._run: Any = None
        self._error: BaseException | None = None
        self._enqueued = False
        self._retired = False

    def enqueue(self) -> None:
        if self._enqueued:
            raise RuntimeError("Kernel ticket can only be enqueued once")
        self._enqueued = True
        assert self._enqueue is not None
        self._enqueue(self)

    def retain_callback(self, callback: Any) -> None:
        """Own the Python callback before the native queue borrows its pointer."""
        self._callback = callback

    def publish(self, run: Any) -> None:
        with self._publish_lock:
            if self._published.is_set():
                raise RuntimeError("Kernel callback already published its result")
            self._run = run
            self._published.set()

    def fail(self, error: BaseException) -> None:
        with self._publish_lock:
            if self._published.is_set():
                raise RuntimeError("Kernel callback already published its result")
            self._error = error
            self._published.set()

    def _check_error(self) -> None:
        if self._error is not None:
            raise self._error

    def done(self) -> bool:
        if self._retired:
            return True
        if not self._published.is_set():
            return False
        self._check_error()
        if not self._run.done:
            return False
        self._retire()
        return True

    def _retire(self) -> None:
        with self._retire_lock:
            if self._retired:
                return
            self._run.wait()  # done alone does not retire native run resources.
            assert self._release is not None
            self._release()
            self._owners = ()
            self._callback = None
            # Closures also own frames/arguments. Drop them under the GIL only
            # after successful retirement; failures keep all owners retryable.
            self._enqueue = None
            self._synchronize = None
            self._release = None
            self._retired = True

    def wait(self) -> None:
        if not self._published.is_set():
            # Explicit shutdown/drain may flush the framework host queue.
            # Ordinary admission only calls done and never synchronizes.
            assert self._synchronize is not None
            self._synchronize()
            if not self._published.is_set():
                raise RuntimeError("Kernel callback did not publish a RunHandle; owners retained")
        self._check_error()
        self._retire()
