# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Every way `tests/st/distributed` opens a card must reach the guarded class.

`tests/st/distributed` runs under `pytest -n`, where the card lock in
`harness/card_lock.py` is the only thing keeping two workers off one card set.
The lock is installed on `simpler.worker.Worker`, so a caller is covered exactly
when it opens its card through that class.

Nothing enforces that at runtime: a test that opened a card another way would
simply drive it unlocked, with no error -- the failure is two processes on one
card set, which surfaces as a wrong number or a poisoned lane somewhere else
entirely. These tests are the tripwire instead, and they are deliberately
written against PyPTO's own source rather than the suite's, so a *new* way to
reach a card fails here before it can be used.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2].parent
_RUNTIME = _REPO / "python" / "pypto" / "runtime"

# The runtime functions that construct a worker, and the helper each one uses.
# `_construct_worker` and `ChipWorker._new_impl` both return a
# `simpler.worker.Worker`, which is the class the lock guards.
_WORKER_FACTORIES = {"_construct_worker", "_new_impl", "_get_simpler_worker_cls"}


def _calls_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.fixture
def card_lock(monkeypatch):
    directory = _REPO / "tests" / "st" / "harness"
    spec = importlib.util.spec_from_file_location("st_card_lock_cov", directory / "card_lock.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_the_lock_is_installed_on_the_class_that_opens_a_card(card_lock):
    """Guarding a wrapper would cover only the callers someone enumerated."""
    source = (_REPO / "tests" / "st" / "harness" / "card_lock.py").read_text()
    assert "from simpler.worker import Worker" in source
    assert "Worker.init = guarded_init" in source
    assert "Worker.close = guarded_close" in source


@pytest.mark.parametrize(
    ("module", "function"),
    [
        ("distributed_runner.py", "_execute_distributed"),
        ("distributed_runner.py", "__init__"),
        ("worker.py", "__init__"),
    ],
)
def test_every_pypto_worker_path_builds_a_simpler_worker(module, function):
    """The wrappers reach a card only through a guarded `Worker`, not around it.

    If a path ever opened a device directly -- its own `rtSetDevice`, a second
    runtime class -- it would escape the lock, so pin that they all go through
    one of the known factories.
    """
    del function  # the assertion is file-level; the id names the path it covers
    calls = _calls_in(_RUNTIME / module)
    assert calls & _WORKER_FACTORIES, (
        f"{module} no longer builds its worker through {sorted(_WORKER_FACTORIES)}. "
        "If it opens a card another way, tests/st/distributed can run it unlocked "
        "under `pytest -n`; extend harness/card_lock.py to cover the new path."
    )


def test_the_distributed_suite_opens_cards_only_through_known_entry_points():
    """A new way to reach a card in the suite should fail here, not race in CI."""
    suite = _REPO / "tests" / "st" / "distributed"
    # Each of these ends at `simpler.worker.Worker`: the first three via PyPTO
    # wrappers, `Worker` itself where a test assembles one by hand.
    allowed = {"Worker", "DistributedWorker", "ChipWorker", "prepare"}
    # A bare `compiled(...)` reaches `_execute_distributed`; it is a call on a
    # local name, so it cannot be matched by callee name and needs no listing.
    forbidden = {"execute_compiled", "_execute_on_device", "attach_current_thread", "set_device"}

    offenders = {
        path.relative_to(_REPO).as_posix(): sorted(_calls_in(path) & forbidden)
        for path in sorted(suite.rglob("test_*.py"))
        if _calls_in(path) & forbidden
    }
    assert not offenders, (
        f"these tests open a card through an unguarded path: {offenders}. "
        f"Guarded entry points are {sorted(allowed)}; see tests/st/harness/card_lock.py."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
