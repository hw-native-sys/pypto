# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One persistent thread for a kernel Worker's native lifecycle."""

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any


class _OwnerThread:
    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], Future[Any]] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._stopped = False
        # Normal cleanup joins this thread before framework teardown. It cannot
        # be non-daemon: Python joins those threads before torch_npu's exit hook.
        self.thread = threading.Thread(target=self._run, name="pypto-kernel-owner", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            function, future = job
            try:
                future.set_result(function())
            except BaseException as exc:
                future.set_exception(exc)
            # A failed job must not keep its closure/traceback alive on success.
            del function, future, job

    def call(self, function: Callable[[], Any]) -> Any:
        if threading.current_thread() is self.thread:
            raise RuntimeError("Reentrant kernel lifecycle request on the owner thread")
        future: Future[Any] = Future()
        with self._lock:
            if self._stopped:
                raise RuntimeError("Kernel lifecycle thread has stopped")
            self._jobs.put((function, future))
        return future.result()

    def stop(self) -> None:
        with self._lock:
            if not self._stopped:
                self._stopped = True
                self._jobs.put(None)
        self.thread.join()
