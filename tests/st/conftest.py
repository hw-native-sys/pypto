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
import os
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
    ensure_simpler_available,
    get_simpler_python_path,
    get_simpler_scripts_path,
)
from harness.core.harness import PTOTestCase  # noqa: E402
from harness.core.test_runner import (  # noqa: E402
    TestRunner,
    _precompile_cache,
    prebuild_binaries,
    precompile_test_cases,
    pregenerate_golden_inputs,
)
from pypto import LogLevel, set_log_level  # noqa: E402
from pypto.runtime.runner import RunConfig  # noqa: E402

# Temp directories created for pre-compilation (when --save-kernels is not set).
# Cleaned up in pytest_sessionfinish.
_temp_precompile_dirs: list[Path] = []


def _init_simpler_root_if_needed() -> None:
    """Populate SIMPLER_ROOT if not already set.

    pytest_collection_finish runs before session fixtures, so
    setup_simpler_dependency may not have set SIMPLER_ROOT yet when
    prebuild_binaries is called.  This function bridges that gap.
    """
    if os.environ.get("SIMPLER_ROOT"):
        return
    try:
        os.environ["SIMPLER_ROOT"] = str(ensure_simpler_available())
    except Exception:
        pass  # SIMPLER_ROOT unavailable; prebuild_binaries will bail out early


@pytest.fixture(scope="session", autouse=True)
def setup_simpler_dependency(request):
    """Ensure Simpler dependency is available.

    This fixture runs once per session before any tests. It:
    1. Checks if Simpler is available (raises error if not)
    2. Sets SIMPLER_ROOT environment variable for test runner
    3. Adds simpler's Python paths to sys.path

    Skipped when --codegen-only is specified (Simpler not needed).
    """
    if request.config.getoption("--codegen-only"):
        return  # Code generation only, Simpler not needed

    simpler_root = ensure_simpler_available()
    os.environ["SIMPLER_ROOT"] = str(simpler_root)

    # Add simpler to sys.path after ensuring it's available
    for path in [get_simpler_python_path(), get_simpler_scripts_path()]:
        if path is not None and path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--platform",
        action="store",
        default="a2a3",
        choices=["a2a3sim", "a2a3", "a5sim", "a5"],
        help="Target platform for tests (default: a2a3sim)",
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
        "--enable-profiling",
        action="store_true",
        default=False,
        help="Enable runtime profiling and generate swimlane.json after execution.",
    )


@pytest.fixture(scope="session")
def test_config(request) -> RunConfig:
    """Session-scoped fixture providing test configuration from CLI options.

    Session scope means the config is created once and shared across all tests,
    which is appropriate since CLI options don't change during a test run.
    """
    save_kernels = request.config.getoption("--save-kernels")
    save_kernels_dir = None
    if save_kernels:
        kernels_dir = request.config.getoption("--kernels-dir")
        # If --kernels-dir is specified, use it; otherwise None will use session output directory
        save_kernels_dir = kernels_dir

    return RunConfig(
        platform=request.config.getoption("--platform"),
        device_id=request.config.getoption("--device"),
        save_kernels=save_kernels,
        save_kernels_dir=save_kernels_dir,
        dump_passes=request.config.getoption("--dump-passes"),
        codegen_only=request.config.getoption("--codegen-only"),
        pto_isa_commit=request.config.getoption("--pto-isa-commit"),
        enable_profiling=request.config.getoption("--enable-profiling"),
    )


@pytest.fixture(scope="session")
def test_runner(test_config) -> TestRunner:
    """Session-scoped fixture providing a test runner instance.

    Session scope is used because:
    1. The runner caches compiled runtime binaries
    2. Building the runtime takes significant time
    3. The same runner can be reused across all tests
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
    config.addinivalue_line("markers", "hardware: mark test as requiring hardware (--platform=a2a3)")
    config.addinivalue_line("markers", "a5: mark test as requiring Ascend 950 (--platform=a5 or a5sim)")
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
    """Modify test collection based on platform."""
    platform = config.getoption("--platform")

    skip_hardware = pytest.mark.skip(reason="hardware tests require --platform=a2a3")
    skip_a5 = pytest.mark.skip(reason="Ascend 950 tests require --platform=a5 or a5sim")

    for item in items:
        if "hardware" in item.keywords and platform != "a2a3":
            item.add_marker(skip_hardware)
        if "a5" in item.keywords and not platform.startswith("a5"):
            item.add_marker(skip_a5)


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
        name = instance.get_name()
        if name not in seen:
            seen[name] = instance


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
    seen: dict[str, PTOTestCase] = {}  # test_name → instance (deduped)

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

    # ── Phase 0: pre-generate golden inputs ──────────────────────────────────
    # Only makes sense when there are successful pre-compilations (golden.py
    # files must exist before we can load and call generate_inputs).
    if n_ok > 0:
        ok_cases = [
            tc
            for tc in test_cases
            if tc.get_name() in _precompile_cache and _precompile_cache[tc.get_name()][1] is None
        ]
        print(
            f"[PyPTO] Pre-generating golden inputs for {len(ok_cases)} test case(s)"
            f" in parallel (workers={workers_str})…"
        )
        n_gen = pregenerate_golden_inputs(ok_cases, cache_dir, max_workers=max_workers)
        print(f"[PyPTO] Golden inputs pre-generated — {n_gen} case(s) cached\n")

        # ── Phase 2: pre-build binary artifacts ──────────────────────────────
        # Compile incore kernels, orchestration .so, and runtime binaries in parallel.
        # Results are saved to work_dir/cache/ and the global binary_cache/runtimes/
        # directory. _execute_on_device installs a write-through patch so subsequent
        # CodeRunner calls serve from disk without recompiling.
        if not session.config.getoption("--codegen-only"):
            platform: str = session.config.getoption("--platform")
            pto_isa_commit: str | None = session.config.getoption("--pto-isa-commit")
            # Ensure SIMPLER_ROOT is available before prebuild_binaries checks it.
            # This hook runs before session fixtures, so setup_simpler_dependency
            # may not have set it yet.
            _init_simpler_root_if_needed()
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
