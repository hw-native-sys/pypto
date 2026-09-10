# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Let xdist workers overlap their compiles by serializing only the device phase.

A distributed test spends most of its time not using a card. Phase timers over
one full ``tests/st/distributed`` run (a2a3, 2 cards) put 57% of the wall clock
in card-free compilation -- ``ir.compile`` plus the chip-kernel build inside
``_compile_and_assemble`` -- performed while the job holds both NPUs and nothing
else can have them.

Running the suite under ``pytest -n N --dist loadfile`` fixes that without any
test having to change: while one worker is on the cards, the others compile. All
it needs is something to stop two workers driving the same cards at once, which
is what this module installs.

Two properties make it work, and both are easy to lose:

* **The lock covers the device phase only.** ``_assemble_chip_callables`` is
  called from inside ``_execute_distributed``, so a lock placed around that
  function alone holds the cards through the chip-kernel build and leaves almost
  nothing to overlap -- measured, a 9% gain for the whole exercise. Running the
  same build *before* taking the lock writes each binary to its sidecar beside
  the generated source, so the call inside the guarded region reads it back
  (1.65s -> 0.05s) and the cards are held only for work that needs them.
* **It spans a worker's whole lifetime**, from construction to ``close()``, not
  each dispatch. A ``DistributedWorker`` keeps forked chip children with open
  device contexts between dispatches, so another process may not construct one
  meanwhile.

The lock is named after the cards it protects, so two CI jobs holding different
cards on one host never wait on each other.

Measured on an a2a3 pair, `tests/st/distributed`, identical results each time:

    serial                                  669.36s
    -n 3, compile hoisted out of the lock   376.58s

The first bullet's number comes from an earlier tree that still had a per-file
prepared-worker fixture, so it is not comparable with the two above; on that tree
serial was 591.54s and locking `_execute_distributed` whole gave 536.41s at -n 2,
against 373.71s once the compile was hoisted out. The ratio is what carries over.
"""

import fcntl
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_LOCK_HELD_ATTR = "_st_card_lock_held"


class CardLock:
    """A reentrant cross-process lock over one set of NPUs.

    Reentrant because the two guarded entry points nest: the prepared-worker
    fixture constructs a ``DistributedWorker`` (which takes the lock) and the
    tests it serves may reach ``_execute_distributed`` (which takes it again).
    ``flock`` is per open file description, so re-locking the same descriptor
    would succeed silently and the first ``release`` would drop the card for
    everyone -- the depth counter is what prevents that.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mu = threading.Lock()
        self._depth = 0
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> None:
        with self._mu:
            if self._depth == 0:
                # 0o600: the lock file is named predictably so concurrent jobs can
                # find it, so keep it unwritable by anyone but this user.
                fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(fd)
                    raise
                self._fd = fd
            self._depth += 1

    def release(self) -> None:
        with self._mu:
            if self._depth == 0:
                return
            self._depth -= 1
            if self._depth == 0 and self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                self._fd = None

    def release_all(self) -> None:
        """Drop the lock however deep it is. Session teardown backstop only.

        A worker that dies between constructing a ``DistributedWorker`` and
        closing it would otherwise hold the cards until its process exits, which
        under xdist is the whole session.
        """
        with self._mu:
            self._depth = 0
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                self._fd = None


def lock_path_for(device_ids: Sequence[int]) -> Path:
    """Name the lock after the cards it protects, so unrelated jobs never meet."""
    ids = "-".join(str(device_id) for device_id in sorted(device_ids)) or "none"
    return Path(tempfile.gettempdir()) / f"pypto-st-cards-{ids}.lock"


# One installation per process, held in a dict so the helpers below mutate it in
# place rather than rebinding module globals.
_state: dict[str, Any] = {}


def _programs(compiled: Any) -> list[Any]:
    """Mirror ``DistributedWorker.__init__``'s own one-or-many argument handling."""
    return list(compiled) if isinstance(compiled, Sequence) else [compiled]


def _prewarm(compiled: Any) -> None:
    """Build each program's chip callables *before* taking the lock.

    The returned callables are discarded; the side effect is the point.
    ``_compile_single_kernel`` and ``_compile_single_orchestration`` each write
    their binary to a sidecar beside the generated source, and both consult that
    sidecar first, so the call inside the guarded region reads bytes instead of
    compiling them. Both calls see the same ``output_dir``, which is what makes
    the sidecar reachable.

    A failure here is swallowed, because the guarded call is about to make it
    again and raise it with the context the caller expects.
    """
    from pypto.runtime import distributed_runner  # noqa: PLC0415

    for program in _programs(compiled):
        try:
            distributed_runner._assemble_chip_callables(program)
        except Exception:  # noqa: BLE001 - resurfaces immediately under the lock
            return


def install(device_ids: Sequence[int]) -> CardLock:
    """Guard every entry point that opens a card, and return the lock."""
    from pypto.runtime import distributed_runner  # noqa: PLC0415

    installed = _state.get("lock")
    if installed is not None:
        return installed
    card_lock = CardLock(lock_path_for(device_ids))

    original_execute = distributed_runner._execute_distributed
    original_init = distributed_runner.DistributedWorker.__init__
    original_close = distributed_runner.DistributedWorker.close
    _state.update(lock=card_lock, execute=original_execute, init=original_init, close=original_close)

    def guarded_execute(compiled: Any, *args: Any, **kwargs: Any) -> Any:
        _prewarm(compiled)
        card_lock.acquire()
        try:
            return original_execute(compiled, *args, **kwargs)
        finally:
            card_lock.release()

    def guarded_init(self: Any, compiled: Any, *args: Any, **kwargs: Any) -> None:
        _prewarm(compiled)
        card_lock.acquire()
        # Set before the call: a failing __init__ never returns a worker whose
        # close() could release, so the except clause below is the only chance.
        setattr(self, _LOCK_HELD_ATTR, True)
        try:
            original_init(self, compiled, *args, **kwargs)
        except BaseException:
            setattr(self, _LOCK_HELD_ATTR, False)
            card_lock.release()
            raise

    def guarded_close(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_close(self, *args, **kwargs)
        finally:
            if getattr(self, _LOCK_HELD_ATTR, False):
                setattr(self, _LOCK_HELD_ATTR, False)
                card_lock.release()

    distributed_runner._execute_distributed = guarded_execute
    distributed_runner.DistributedWorker.__init__ = guarded_init
    distributed_runner.DistributedWorker.close = guarded_close
    return card_lock


def uninstall() -> None:
    """Restore the runtime and drop the lock. Called at session teardown."""
    card_lock = _state.get("lock")
    if card_lock is None:
        return
    from pypto.runtime import distributed_runner  # noqa: PLC0415

    distributed_runner._execute_distributed = _state["execute"]
    distributed_runner.DistributedWorker.__init__ = _state["init"]
    distributed_runner.DistributedWorker.close = _state["close"]
    _state.clear()
    card_lock.release_all()
