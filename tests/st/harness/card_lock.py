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

Running the suite under ``pytest -n N`` fixes that without any test having to
change: while one worker is on the cards, the others compile. All it needs is
something to stop two workers driving the same cards at once, which is what this
module installs.

**The guard sits on ``simpler.worker.Worker``, not on the PyPTO wrappers.** That
class is where a card is actually opened, and everything reaches it:
``_execute_distributed``, ``DistributedWorker``, ``ChipWorker``, and a test that
assembles ``Worker(level=3, ...)`` by hand -- ``test_l3_manual`` does exactly
that. Guarding the wrappers instead covers the callers one thought to enumerate,
and a test that opens a card another way drives it unlocked with no error at
all: just two processes on one card set.

Locking at ``init`` / ``close`` also draws the boundary in the right place for
free. Both worker paths assemble every chip callable *before* constructing the
runtime worker -- ``_assemble_chip_callables``, then ``_construct_worker``, then
``init`` -- so the chip-kernel build sits outside the guarded region by
construction, with nothing to arrange. Held any wider, around
``_execute_distributed`` say, the cards stay busy through that build and there is
almost nothing left to overlap: a 9% return for the whole exercise, measured.

The region spans ``init`` to ``close`` rather than each dispatch, because a
worker keeps forked chip children with open device contexts between dispatches,
so another process may not construct one meanwhile.

The lock is named after the cards it protects, so two CI jobs holding different
cards on one host never wait on each other.

Measured on an a2a3 pair, `tests/st/distributed`, identical results each time:

    serial                                  669.36s
    -n 3, device phase serialized           376.58s
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


def install(device_ids: Sequence[int]) -> CardLock:
    """Guard every path that opens a card, and return the lock.

    ``simpler.worker.Worker`` is that path: every PyPTO wrapper builds one
    (`_execute_distributed`, `DistributedWorker`, `ChipWorker`), and so does a
    test that assembles a worker by hand -- ``test_l3_manual`` does. Guarding the
    class rather than its callers is what makes a future test safe without
    having to remember to be: there is no way to open a card that skips it.
    """
    from simpler.worker import Worker  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    installed = _state.get("lock")
    if installed is not None:
        return installed
    card_lock = CardLock(lock_path_for(device_ids))

    original_init = Worker.init
    original_close = Worker.close
    # The pid that owns this lock. `Worker.init` forks the chip children *inside*
    # the guarded region, and a child goes on to init its own inner worker
    # (`worker.py`, the `if pid == 0:` arm of the hierarchical start), so the
    # guard runs there too -- against an inherited `CardLock` whose depth and fd
    # are copies of ours. `flock` belongs to the open file description, which
    # fork shares, so a child reaching depth zero would unlock the *parent's*
    # hold; and a child that tried to acquire honestly would deadlock, since the
    # parent holds the lock while forking it. A child is already inside our
    # critical section, so its guard passes through.
    owner_pid = os.getpid()
    _state.update(
        lock=card_lock,
        worker=Worker,
        init=original_init,
        close=original_close,
        owner_pid=owner_pid,
    )

    def guarded_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        if os.getpid() != owner_pid:
            return original_init(self, *args, **kwargs)
        card_lock.acquire()
        # Set before the call: an ``init`` that raises never returns a worker
        # whose ``close`` could release, so the except clause is the only chance.
        setattr(self, _LOCK_HELD_ATTR, True)
        try:
            return original_init(self, *args, **kwargs)
        except BaseException:
            setattr(self, _LOCK_HELD_ATTR, False)
            card_lock.release()
            raise

    def guarded_close(self: Any, *args: Any, **kwargs: Any) -> Any:
        if os.getpid() != owner_pid:
            return original_close(self, *args, **kwargs)
        try:
            return original_close(self, *args, **kwargs)
        finally:
            # Only a worker this guard let onto a card releases one: ``close`` is
            # also reached after a construction that failed before ``init``, and
            # again on a retry of an already-closed worker.
            if getattr(self, _LOCK_HELD_ATTR, False):
                setattr(self, _LOCK_HELD_ATTR, False)
                card_lock.release()

    Worker.init = guarded_init
    Worker.close = guarded_close
    return card_lock


def uninstall() -> None:
    """Restore the runtime and drop the lock. Called at session teardown."""
    card_lock = _state.get("lock")
    if card_lock is None:
        return
    if os.getpid() != _state["owner_pid"]:
        return  # a forked child never owned it and must not unlock it
    worker = _state["worker"]
    worker.init = _state["init"]
    worker.close = _state["close"]
    _state.clear()
    card_lock.release_all()
