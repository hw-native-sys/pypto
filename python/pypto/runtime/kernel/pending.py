# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python owner retention around Simpler's GIL-free prepared submissions."""

import threading
from collections.abc import Callable
from typing import Any

from .callable import KernelRegistration


class PendingTicket:
    """Retain external owners until Simpler proves safe retirement.

    Register this ticket before preparation. An enqueue exception can occur
    after native consumption starts; cancellation alone is not a cleanup proof.
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
        self.submission: Any = None
        self._enqueue: Callable[[PendingTicket], None] | None = enqueue
        self._synchronize: Callable[[], None] | None = synchronize
        self._release: Callable[[], None] | None = release
        self._owners = owners
        self._retire_lock = threading.Lock()
        self._enqueued = False
        self.retired = False

    def enqueue(self) -> None:
        if self._enqueued:
            raise RuntimeError("Kernel ticket can only be enqueued once")
        self._enqueued = True
        assert self._enqueue is not None
        try:
            self._enqueue(self)
        except BaseException as error:
            try:
                self._cancel()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _cancel(self) -> None:
        try:
            if self.submission is not None:
                # False means already consumed/cancelled. Cancellation may
                # also raise after establishing safe retirement.
                self.submission.cancel()
        finally:
            self._release_if_safe()

    def _release_if_safe(self) -> None:
        if self.retired or (self.submission is not None and not self.submission.safe_to_release):
            return
        assert self._release is not None
        self._release()
        self._owners = ()
        self._enqueue = None
        self._synchronize = None
        self._release = None
        self.submission = None
        self.retired = True

    def _retire(self, timeout: float | None = None) -> None:
        with self._retire_lock:
            if self.retired:
                return
            try:
                if self.submission is not None:
                    self.submission.wait(timeout=timeout)
            except BaseException as error:
                try:
                    self._release_if_safe()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                raise
            self._release_if_safe()

    def done(self) -> bool:
        if self.retired:
            return True
        if self.submission is None or not self.submission.done:
            return False
        try:
            # Physical completion need not make observation/cleanup nonblocking.
            self._retire(timeout=0)
        except TimeoutError:
            if self.retired:
                raise
            return False
        return self.retired

    def wait(self) -> None:
        if self.retired:
            return
        if self.submission is not None and not self.submission.done:
            assert self._synchronize is not None
            try:
                self._synchronize()
            except BaseException as error:
                # A failed native callback makes Torch discard later callbacks.
                # Their unconsumed tokens must be cancelled before waiting.
                try:
                    self._cancel()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                self._retire()  # Already-consumed work still needs a cleanup proof.
                raise
        self._retire()
        if not self.retired:
            raise RuntimeError("Kernel submission cleanup is incomplete; owners retained")
