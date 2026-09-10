# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for the ST card lock that lets xdist workers overlap compiles.

The lock is what makes `pytest -n N` on `tests/st/distributed` safe, so its two
load-bearing properties are pinned here rather than only on hardware: it is
reentrant (the prepared-worker fixture nests a dispatch inside a construction),
and it actually excludes a second process.
"""

import importlib.util
import multiprocessing
import sys
from pathlib import Path

import pytest


@pytest.fixture
def card_lock(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "st" / "harness"
    monkeypatch.syspath_prepend(str(directory.parent))
    spec = importlib.util.spec_from_file_location("st_card_lock", directory / "card_lock.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_lock_path_names_the_cards_it_protects(card_lock):
    """Two jobs holding different cards on one host must not wait on each other."""
    assert card_lock.lock_path_for([3, 1]).name == "pypto-st-cards-1-3.lock"
    assert card_lock.lock_path_for([1, 3]) == card_lock.lock_path_for([3, 1])
    assert card_lock.lock_path_for([1, 3]) != card_lock.lock_path_for([5, 7])
    assert card_lock.lock_path_for([]).name == "pypto-st-cards-none.lock"


def test_lock_is_reentrant_and_only_the_last_release_frees_it(card_lock, tmp_path):
    """`flock` re-locks its own descriptor silently; the depth counter is the guard."""
    lock = card_lock.CardLock(tmp_path / "cards.lock")

    lock.acquire()
    lock.acquire()
    lock.release()
    assert _held_by_another_process(lock.path), "inner release dropped the card"

    lock.release()
    assert not _held_by_another_process(lock.path)


def test_release_all_drops_a_lock_left_held(card_lock, tmp_path):
    """Session teardown must not leave a dead worker holding the cards."""
    lock = card_lock.CardLock(tmp_path / "cards.lock")
    lock.acquire()
    lock.acquire()

    lock.release_all()

    assert not _held_by_another_process(lock.path)
    lock.release_all()  # idempotent


def test_release_without_acquire_is_a_no_op(card_lock, tmp_path):
    """A guarded close that never acquired must not free someone else's lock."""
    lock = card_lock.CardLock(tmp_path / "cards.lock")
    lock.release()
    assert not _held_by_another_process(lock.path)


def _try_lock(path, result):
    import fcntl  # noqa: PLC0415
    import os  # noqa: PLC0415

    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        result.value = 1  # someone else holds it
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        result.value = 0
    finally:
        os.close(fd)


def _held_by_another_process(path) -> bool:
    """`flock` is per open file description, so ask a separate process."""
    ctx = multiprocessing.get_context("spawn")
    result = ctx.Value("i", -1)
    proc = ctx.Process(target=_try_lock, args=(str(path), result))
    proc.start()
    proc.join(30)
    assert proc.exitcode == 0, f"probe process failed: exitcode={proc.exitcode}"
    assert result.value in (0, 1)
    return result.value == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
