# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the system-test platform matrix.

These pin the two halves of "any case, any platform":

- the *guard*: a case pinning ``get_backend_type()`` is skipped instead of
  being compiled for one architecture and executed on another;
- the *matrix*: ``pytest_generate_tests`` expands ``test_runner``-based tests
  over the ``--platform`` allowlist, and each variant keys its own artefact.

The harness package (``harness.core.*``) and the system-test ``conftest.py``
live under ``tests/st``; both are loaded by path so this stays a device-free
unit test.
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pypto.backend import BackendType
from pypto.pypto_core.passes import MemoryPlanner

_ST_DIR = Path(__file__).resolve().parents[2] / "st"
if str(_ST_DIR) not in sys.path:
    sys.path.insert(0, str(_ST_DIR))

harness = importlib.import_module("harness.core.harness")
test_runner = importlib.import_module("harness.core.test_runner")


def _load_st_conftest():
    """Import tests/st/conftest.py under a private name.

    Loaded by path rather than ``import conftest`` so pytest's own conftest
    collection is not disturbed.
    """
    path = _ST_DIR / "conftest.py"
    spec = importlib.util.spec_from_file_location("_st_conftest_matrix_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_published_platform():
    """Keep the module-global item platform from leaking between tests.

    The ``platform_xfail`` tests call the conftest hook that publishes it, and
    ``_resolve_platform`` reads it as a fallback — so without this a later test
    in the same worker sees a platform no one set for it.
    """
    saved = test_runner._current_item_platform["value"]
    yield
    test_runner.set_current_item_platform(saved)


class _Case(harness.PTOTestCase):
    """Minimal concrete case: no platform, no backend pin."""

    def get_name(self) -> str:
        return "matrix_case"

    def define_tensors(self) -> list[Any]:
        return []

    def get_program(self) -> Any:
        return object()

    def compute_expected(self, tensors, params=None) -> None:
        pass


class _LeftoverPinCase(_Case):
    """A case that still carries the removed backend knob."""

    def get_name(self) -> str:
        return "leftover_pin_case"

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


# ---------------------------------------------------------------------------
# The platform is a case's only architecture knob
# ---------------------------------------------------------------------------


def test_a_case_carries_no_backend_of_its_own():
    # Not "the default is 910B" — there is no such accessor to default.
    assert not hasattr(harness.PTOTestCase, "get_backend_type")
    with pytest.raises(TypeError):
        _Case(backend_type=BackendType.Ascend910B)  # pyright: ignore[reportCallIssue]


def test_the_backend_is_derived_from_the_bound_platform():
    case = _Case()
    test_runner._bind_item_platform(case, "a5")
    assert case.get_platform() == "a5"
    assert harness.platform_to_backend(case.get_platform()) is BackendType.Ascend950


def test_a_leftover_backend_override_is_rejected():
    # Silent dead code otherwise: the harness stopped reading this method, so a
    # case still defining it would compile for whatever the platform says while
    # its author believes the override decides.
    with pytest.raises(pytest.UsageError, match="get_backend_type"):
        test_runner._bind_item_platform(_LeftoverPinCase(), "a5")


def test_bind_platform_never_overrides_an_explicit_pin():
    case = _Case(platform="a2a3sim")
    case.bind_platform("a5")
    assert case.get_platform() == "a2a3sim"


def test_bind_platform_rejects_an_unknown_id():
    with pytest.raises(ValueError, match="Unknown platform"):
        _Case().bind_platform("a9")


def test_cache_key_separates_the_platform_variants():
    a2a3, a5 = _Case(), _Case()
    a2a3.bind_platform("a2a3")
    a5.bind_platform("a5sim")
    key_a2a3 = test_runner._cache_key(a2a3, session_memory_planner=MemoryPlanner.PYPTO)
    key_a5 = test_runner._cache_key(a5, session_memory_planner=MemoryPlanner.PYPTO)
    assert key_a2a3 != key_a5
    assert key_a2a3.endswith("@a2a3@pypto") and key_a5.endswith("@a5sim@pypto")


def test_cache_key_refuses_an_artifact_with_no_platform():
    # Two architectures sharing one key would share one compile.
    with pytest.raises(ValueError, match="no platform resolved or bound"):
        test_runner._cache_key(_Case(), session_memory_planner=MemoryPlanner.PYPTO)


# ---------------------------------------------------------------------------
# The matrix: pytest_generate_tests
# ---------------------------------------------------------------------------


def _fn_taking(*argnames: str):
    """Return a function whose signature requests exactly *argnames*."""

    def _test(*_args, **_kwargs):
        pass

    _test.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in argnames]
    )
    return _test


class _FakeMetafunc:
    """Stand-in for ``pytest.Metafunc`` recording a single parametrize call."""

    def __init__(self, platform_option: str, fixturenames: list[str], marker_args=None, signature=None):
        self.fixturenames = fixturenames
        self.function = _fn_taking(*(signature if signature is not None else fixturenames))
        self.config = SimpleNamespace(getoption=lambda name: {"--platform": platform_option}[name])
        marker = SimpleNamespace(args=tuple(marker_args), kwargs={}) if marker_args is not None else None
        self.definition = SimpleNamespace(
            name="test_fake",
            get_closest_marker=lambda name: marker if name == "platforms" else None,
        )
        self.calls: list[tuple] = []

    def parametrize(self, argname, argvalues, ids=None, indirect=False):
        recorded_indirect = list(indirect) if isinstance(indirect, (list, tuple)) else indirect
        self.calls.append((argname, list(argvalues), list(ids or []), recorded_indirect))


_RUNNER_FIXTURES = ["test_runner", "_st_platform", "test_config"]


def _generate(metafunc):
    _load_st_conftest().pytest_generate_tests(metafunc)
    return metafunc.calls


def test_single_platform_keeps_todays_node_ids():
    # One active platform: no parametrize, so `test_foo` does not become
    # `test_foo[a2a3]` for every existing CI selector.
    assert _generate(_FakeMetafunc("a2a3", _RUNNER_FIXTURES)) == []


def test_multi_platform_expands_over_the_allowlist():
    calls = _generate(_FakeMetafunc("a2a3,a5sim", _RUNNER_FIXTURES))
    assert calls == [("_st_platform", ["a2a3", "a5sim"], ["a2a3", "a5sim"], True)]


def test_expansion_intersects_the_platforms_marker():
    calls = _generate(_FakeMetafunc("a2a3,a2a3sim,a5", _RUNNER_FIXTURES, marker_args=["a2a3", "a2a3sim"]))
    assert calls == [("_st_platform", ["a2a3", "a2a3sim"], ["a2a3", "a2a3sim"], True)]


def test_marker_narrowing_to_one_platform_still_binds_it():
    # Without the parametrize the item would carry no platform and fall back to
    # the first CLI id — a2a3 — which its own marker excludes.
    calls = _generate(_FakeMetafunc("a2a3,a5", _RUNNER_FIXTURES, marker_args=["a5"]))
    assert calls == [("_st_platform", ["a5"], ["a5"], True)]


def test_marker_outside_the_allowlist_leaves_the_item_for_deselection():
    # Empty intersection: nothing to parametrize, and pytest_collection_modifyitems
    # drops the item because its effective allowed set is empty.
    assert _generate(_FakeMetafunc("a2a3", _RUNNER_FIXTURES, marker_args=["a5", "a5sim"])) == []


def test_explicitly_parametrized_tests_are_left_alone():
    fixtures = [*_RUNNER_FIXTURES, "platform"]
    assert _generate(_FakeMetafunc("a2a3,a5", fixtures)) == []


def test_tests_without_the_runner_are_untouched():
    assert _generate(_FakeMetafunc("a2a3,a5", ["tmp_path"])) == []


def test_a_runner_reached_through_another_fixture_is_not_expanded():
    # The runner arrives via a module-scoped fixture, which is built once for
    # the whole module and so cannot hold one artefact per platform.
    metafunc = _FakeMetafunc(
        "a2a3,a5",
        ["swimlane", "test_runner", "_st_platform", "test_config"],
        signature=["swimlane"],
    )
    assert _generate(metafunc) == []


def test_empty_platform_option_expands_over_every_platform():
    calls = _generate(_FakeMetafunc("", _RUNNER_FIXTURES))
    assert calls == [("_st_platform", list(harness.ALL_PLATFORM_IDS), list(harness.ALL_PLATFORM_IDS), True)]


def test_a_typo_in_the_platforms_marker_fails_collection():
    # Before, an unknown id narrowed the marker to nothing and the test was
    # silently deselected on every platform.
    with pytest.raises(pytest.UsageError, match="unknown platform id"):
        _generate(_FakeMetafunc("a2a3,a5", _RUNNER_FIXTURES, marker_args=["a2a3", "a5typo"]))


def test_an_empty_platforms_marker_fails_collection():
    with pytest.raises(pytest.UsageError, match="no platform ids"):
        _generate(_FakeMetafunc("a2a3,a5", _RUNNER_FIXTURES, marker_args=[]))


# ---------------------------------------------------------------------------
# Per-item platform resolution
# ---------------------------------------------------------------------------


class _FakeNode:
    """Stand-in for a collected item, for the platform resolver."""

    def __init__(self, params=None, marker_args=None, name="test_fake"):
        self.name = name
        self.callspec = SimpleNamespace(params=dict(params or {})) if params is not None else None
        self._marker = SimpleNamespace(args=tuple(marker_args)) if marker_args is not None else None

    def get_closest_marker(self, name):
        return self._marker if name == "platforms" else None


def _resolve(node, platform_option):
    config = SimpleNamespace(getoption=lambda name: {"--platform": platform_option}[name])
    return _load_st_conftest()._resolve_item_platform(node, config)


def test_the_parametrized_platform_wins():
    assert _resolve(_FakeNode(params={"_st_platform": "a5sim"}), "a2a3,a5sim") == "a5sim"
    assert _resolve(_FakeNode(params={"platform": "a5"}), "a2a3,a5") == "a5"


def test_an_unexpanded_item_falls_back_to_the_first_cli_platform():
    assert _resolve(_FakeNode(), "a5,a2a3") == "a5"


def test_the_fallback_honours_the_platforms_marker():
    # The item is not parametrized (its runner comes from a shared fixture), so
    # the fallback must not hand it a platform its own marker excludes.
    assert _resolve(_FakeNode(marker_args=["a5"]), "a2a3,a5") == "a5"


# ---------------------------------------------------------------------------
# platform_xfail
# ---------------------------------------------------------------------------


class _FakeItem:
    """Stand-in for a collected item carrying a ``platform_xfail`` marker."""

    def __init__(self, platform_option: str, args, kwargs, params=None):
        self.name = "test_fake"
        self.config = SimpleNamespace(getoption=lambda name: {"--platform": platform_option}[name])
        self.callspec = SimpleNamespace(params=dict(params or {}))
        self._marker = SimpleNamespace(args=tuple(args), kwargs=dict(kwargs))
        self.added: list[Any] = []

    def get_closest_marker(self, name):
        return self._marker if name == "platform_xfail" else None

    def add_marker(self, marker):
        self.added.append(marker)


def _setup(item):
    _load_st_conftest().pytest_runtest_setup(item)
    return item.added


def test_platform_xfail_applies_on_the_listed_platform():
    item = _FakeItem("a5", ["a5"], {"reason": "950 board only"}, params={"_st_platform": "a5"})
    (marker,) = _setup(item)
    assert marker.kwargs["reason"] == "950 board only"
    # Strict by default: a deterministic failure that starts passing must be
    # reported, not silently absorbed.
    assert marker.kwargs["strict"] is True


def test_platform_xfail_is_inert_on_other_platforms():
    item = _FakeItem("a2a3", ["a5"], {"reason": "950 board only"}, params={"_st_platform": "a2a3"})
    assert _setup(item) == []


def test_platform_xfail_falls_back_to_the_cli_platform():
    # cross_core-style items carry no platform param; the session's first
    # --platform id is what their module-level generator resolves too.
    item = _FakeItem("a5", ["a5"], {"reason": "950 board only"})
    assert len(_setup(item)) == 1


def test_platform_xfail_honours_an_explicit_non_strict():
    item = _FakeItem("a5", ["a5"], {"reason": "flaky on 950", "strict": False})
    (marker,) = _setup(item)
    assert marker.kwargs["strict"] is False


def test_platform_xfail_requires_a_reason():
    item = _FakeItem("a5", ["a5"], {})
    with pytest.raises(pytest.UsageError, match="needs reason"):
        _setup(item)


def test_platform_xfail_rejects_a_typo():
    item = _FakeItem("a5", ["a5typo"], {"reason": "950 board only"})
    with pytest.raises(pytest.UsageError, match="unknown platform id"):
        _setup(item)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
