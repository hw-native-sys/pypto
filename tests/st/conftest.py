# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
pytest configuration and fixtures for PyPTO integration tests.

This configuration sets up the testing environment using the internal
harness package (migrated from pto-testing-framework).
"""

import inspect
import random
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Add harness to path (internal package in tests/st/)
_ST_DIR = Path(__file__).parent
if str(_ST_DIR) not in sys.path:
    sys.path.insert(0, str(_ST_DIR))

# Add project root to path (for examples package)
_PROJECT_ROOT = _ST_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest  # noqa: E402
from harness.core.environment import (  # noqa: E402
    get_simpler_python_path,
    get_simpler_scripts_path,
)
from harness.core.harness import ALL_PLATFORM_IDS, PTOTestCase  # noqa: E402
from harness.core.test_runner import (  # noqa: E402
    TestRunner,
    _cache_key,
    _precompile_cache,
    prebuild_binaries,
    precompile_test_cases,
)
from pypto import LogLevel, set_log_level  # noqa: E402
from pypto.runtime.runner import RunConfig  # noqa: E402

# Temp directories created for pre-compilation (when --save-kernels is not set).
# Cleaned up in pytest_sessionfinish.
_temp_precompile_dirs: list[Path] = []


@pytest.fixture(scope="session", autouse=True)
def setup_simpler_dependency(request):
    """Add Simpler submodule Python paths to sys.path.

    Skipped when --codegen-only is specified (Simpler not needed).
    """
    if request.config.getoption("--codegen-only"):
        return

    for path in [get_simpler_python_path(), get_simpler_scripts_path()]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--platform",
        action="store",
        default="a2a3sim,a5sim",
        help=(
            "Comma-separated allowlist of target platforms; each test under "
            "tests/st/runtime/ is parametrized over a2a3, a5, a2a3sim, a5sim "
            "and only variants whose id appears here are run "
            "(default: a2a3sim,a5sim)."
        ),
    )
    parser.addoption(
        "--device",
        action="store",
        default=0,
        type=int,
        help="Device ID for hardware tests (default: 0)",
    )
    parser.addoption(
        "--strategy",
        action="store",
        default="Default",
        choices=["Default"],
        help="Optimization strategy for PyPTO pass pipeline (default: Default)",
    )
    parser.addoption(
        "--fuzz-count",
        action="store",
        default=10,
        type=int,
        help="Number of fuzz test iterations (default: 10)",
    )
    parser.addoption(
        "--fuzz-seed",
        action="store",
        default=None,
        type=int,
        help="Random seed for fuzz tests (default: random)",
    )
    parser.addoption(
        "--kernels-dir",
        action="store",
        default=None,
        help="Output directory for generated kernels (default: build/outputs/output_{timestamp}/)",
    )
    parser.addoption(
        "--save-kernels",
        action="store_true",
        default=False,
        help="Save generated kernels to --kernels-dir (default: False)",
    )
    parser.addoption(
        "--dump-passes",
        action="store_true",
        default=False,
        help="Dump intermediate IR after each pass (default: False)",
    )
    parser.addoption(
        "--codegen-only",
        action="store_true",
        default=False,
        help="Only generate code, skip runtime execution (default: False)",
    )
    parser.addoption(
        "--precompile-workers",
        action="store",
        default=None,
        type=int,
        help="Number of parallel threads for pre-compilation phase (default: min(32, cpu_count+4))",
    )
    parser.addoption(
        "--pypto-log-level",
        action="store",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "FATAL", "EVENT", "NONE"],
        help="PyPTO C++ log level threshold (default: ERROR)",
    )
    parser.addoption(
        "--pto-isa-commit",
        action="store",
        default=None,
        help="Pin the pto-isa clone to a specific git commit (hash or tag). Default: use latest remote HEAD.",
    )
    parser.addoption(
        "--runtime-profiling",
        action="store_true",
        default=False,
        help="Enable on-device runtime profiling and generate swimlane.json after execution.",
    )


def _parse_platform_filter(raw: str) -> set[str]:
    """Parse the comma-separated --platform value into a set of platform ids.

    Unknown ids are silently dropped so that bogus user input degrades to the
    intersection with the canonical platform set rather than producing a
    confusing collection error.
    """
    requested = {tok.strip() for tok in str(raw).split(",") if tok.strip()}
    return requested & set(ALL_PLATFORM_IDS)


@pytest.fixture(scope="session")
def test_config(request) -> RunConfig:
    """Session-scoped fixture providing test configuration from CLI options.

    Session scope means the config is created once and shared across all tests,
    which is appropriate since CLI options don't change during a test run.

    ``RunConfig.platform`` carries a single representative platform id; this
    is only used as a fallback for legacy code paths that have not been
    migrated to ``PTOTestCase.get_platform()``. Per-test parametrized variants
    forward their own ``platform`` to the test case constructor and therefore
    override this value via ``tc.get_platform()`` inside ``TestRunner``.
    """
    save_kernels = request.config.getoption("--save-kernels")
    save_kernels_dir = None
    if save_kernels:
        kernels_dir = request.config.getoption("--kernels-dir")
        # If --kernels-dir is specified, use it; otherwise None will use session output directory
        save_kernels_dir = kernels_dir

    platform_filter = _parse_platform_filter(request.config.getoption("--platform"))
    fallback_platform = next(iter(platform_filter), "a2a3sim")

    return RunConfig(
        platform=fallback_platform,
        device_id=request.config.getoption("--device"),
        save_kernels=save_kernels,
        save_kernels_dir=save_kernels_dir,
        dump_passes=request.config.getoption("--dump-passes"),
        codegen_only=request.config.getoption("--codegen-only"),
        pto_isa_commit=request.config.getoption("--pto-isa-commit"),
        runtime_profiling=request.config.getoption("--runtime-profiling"),
    )


@pytest.fixture(scope="session")
def test_runner(test_config) -> TestRunner:
    """Session-scoped fixture providing a test runner instance.

    Session scope is used because the same runner can be reused across all tests.
    """
    return TestRunner(test_config)


@pytest.fixture
def optimization_strategy(request) -> str:
    """Fixture providing the optimization strategy from CLI options."""
    return request.config.getoption("--strategy")


@pytest.fixture
def fuzz_count(request) -> int:
    """Fixture providing fuzz test iteration count."""
    return request.config.getoption("--fuzz-count")


@pytest.fixture
def fuzz_seed(request) -> int:
    """Fixture providing fuzz test seed."""
    seed = request.config.getoption("--fuzz-seed")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    return seed


# Standard test shapes for parameterized tests
STANDARD_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
]


@pytest.fixture(params=STANDARD_SHAPES)
def tensor_shape(request):
    """Parameterized fixture for tensor shapes."""
    return list(request.param)


# Skip markers
def pytest_configure(config):
    """Register custom markers and apply early global settings."""
    config.addinivalue_line(
        "markers",
        "platforms(*ids): restrict the test to the given platform ids "
        "(intersected with the --platform CLI filter)",
    )
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "fuzz: mark test as fuzz test")

    # Set C++ log level as early as possible so it applies to collection too.
    # Forked child processes inherit this setting via os.fork().
    try:
        level_name: str = config.getoption("--pypto-log-level")
        set_log_level(LogLevel[level_name])
    except (ValueError, KeyError):
        pass  # option not yet registered (e.g. during --co --help)


def pytest_collection_modifyitems(config, items):
    """Deselect items that fall outside the active platform allowlist.

    Two layers of filtering are applied:

    1. The ``--platform`` CLI option is parsed into a set of platform ids
       and intersected with the canonical ``ALL_PLATFORM_IDS``.
    2. Each item may carry a ``@pytest.mark.platforms(...)`` whitelist; the
       effective allowed set for that item is ``cli_filter & item_filter``.

    For parametrized variants (named after the platform id, e.g. ``[a5sim]``),
    the variant's own platform must lie inside the effective allowed set.
    Items without a platform parameter pass as long as the effective set is
    non-empty.
    """
    cli_filter = _parse_platform_filter(config.getoption("--platform"))
    if not cli_filter:
        cli_filter = set(ALL_PLATFORM_IDS)
    canonical = set(ALL_PLATFORM_IDS)

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        item_marker = next(item.iter_markers(name="platforms"), None)
        if item_marker is not None:
            item_filter = {p for p in item_marker.args if p in canonical}
        else:
            item_filter = canonical
        allowed = cli_filter & item_filter

        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec else {}
        platform_param = params.get("platform")

        if platform_param is not None:
            if platform_param in allowed:
                selected.append(item)
            else:
                deselected.append(item)
        elif allowed:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _collect_test_case_from_item(item: pytest.Item, seen: dict[str, PTOTestCase]) -> None:
    """Inspect *item* and add any newly discovered PTOTestCase instance to *seen*."""
    if any(m.name == "skip" for m in item.iter_markers()):
        return

    module = item.module

    # Collect PTOTestCase subclasses visible in this module.
    testcase_classes: dict[str, type] = {}
    for attr in dir(module):
        obj = getattr(module, attr, None)
        if (
            obj is not None
            and isinstance(obj, type)
            and issubclass(obj, PTOTestCase)
            and obj is not PTOTestCase
        ):
            testcase_classes[attr] = obj

    if not testcase_classes:
        return

    # callspec params for @pytest.mark.parametrize (empty dict if none).
    callspec = getattr(item, "callspec", None)
    call_params: dict[str, Any] = callspec.params if callspec else {}

    # Scan test function source to find which class name is referenced.
    try:
        source = inspect.getsource(item.function)
    except (OSError, TypeError):
        return

    for cls_name, cls in testcase_classes.items():
        if not re.search(r"\b" + re.escape(cls_name) + r"\s*\(", source):
            continue
        # Filter callspec params to those accepted by __init__.
        try:
            sig = inspect.signature(cls.__init__)
            valid = {k: v for k, v in call_params.items() if k in sig.parameters}
            instance = cls(**valid)
        except Exception:
            continue  # constructor mismatch — skip
        key = _cache_key(instance)
        if key not in seen:
            seen[key] = instance


def pytest_collection_finish(session: pytest.Session) -> None:
    """Phase 1: discover and pre-compile all test cases in parallel after collection.

    After pytest finishes collecting tests, this hook inspects each test item to
    find which PTOTestCase subclass it uses, instantiates those cases, and
    compiles them all concurrently via a thread pool.

    Discovery strategy (best-effort, no test file changes required):
    - Find PTOTestCase subclasses in each collected item's module.
    - Scan the test function source for ``ClassName(`` to identify which class
      is used in that test.
    - For parametrised tests, match ``callspec.params`` to ``__init__`` kwargs.
    - Cases that cannot be discovered fall back to the original
      compile-on-demand path inside ``TestRunner.run()``.
    """
    if not session.items:
        return

    # ── discover PTOTestCase instances ───────────────────────────────────────
    seen: dict[str, PTOTestCase] = {}  # cache_key → instance (deduped)

    for item in session.items:
        _collect_test_case_from_item(item, seen)

    if not seen:
        return

    dump_passes: bool = session.config.getoption("--dump-passes")
    max_workers: int | None = session.config.getoption("--precompile-workers")

    # Without --precompile-workers the pre-compilation/cache phases are skipped
    # entirely; each test compiles on demand inside TestRunner.run().
    if max_workers is None:
        return

    # ── determine cache directory ─────────────────────────────────────────────
    save_kernels: bool = session.config.getoption("--save-kernels")
    kernels_dir: str | None = session.config.getoption("--kernels-dir")
    if save_kernels:
        if kernels_dir:
            cache_dir = Path(kernels_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_dir = _PROJECT_ROOT / "build_output" / f"precompile_{timestamp}"
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="pypto_precompile_"))
        _temp_precompile_dirs.append(cache_dir)

    # ── compile in parallel ───────────────────────────────────────────────────
    test_cases = list(seen.values())
    workers_str = str(max_workers) if max_workers is not None else "auto"
    print(f"\n[PyPTO] Pre-compiling {len(test_cases)} test case(s) in parallel (workers={workers_str})…")
    precompile_test_cases(test_cases, cache_dir, dump_passes=dump_passes, max_workers=max_workers)

    n_ok = sum(1 for _, err in _precompile_cache.values() if err is None)
    n_fail = len(_precompile_cache) - n_ok
    print(f"[PyPTO] Pre-compilation done — {n_ok} ok, {n_fail} failed\n")

    # ── Phase 2: pre-build binary artifacts ──────────────────────────────
    # Compile incore kernels and orchestration .so in parallel.
    # Results are saved to work_dir/cache/.
    if n_ok > 0 and not session.config.getoption("--codegen-only"):
        ok_cases = [
            tc
            for tc in test_cases
            if _cache_key(tc) in _precompile_cache and _precompile_cache[_cache_key(tc)][1] is None
        ]
        # ``--platform`` is a CSV allowlist; the per-test value resolved by
        # ``tc.get_platform()`` overrides this fallback inside ``TestRunner``,
        # so any one entry from the filter is sufficient here.
        platform_filter = _parse_platform_filter(session.config.getoption("--platform"))
        platform: str = next(iter(platform_filter), "a2a3sim")
        pto_isa_commit: str | None = session.config.getoption("--pto-isa-commit")
        print(
            f"[PyPTO] Pre-building binary artifacts for {len(ok_cases)} test case(s)"
            f" in parallel (workers={workers_str})…"
        )
        n_built = prebuild_binaries(
            ok_cases, cache_dir, platform, max_workers=max_workers, pto_isa_commit=pto_isa_commit
        )
        print(f"[PyPTO] Binary pre-build done — {n_built} case(s) compiled\n")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Clean up temporary pre-compilation directories created during the session."""
    for d in _temp_precompile_dirs:
        shutil.rmtree(d, ignore_errors=True)
