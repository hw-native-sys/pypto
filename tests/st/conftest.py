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

import json
import os
import random
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

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
from harness.core.test_runner import TestRunner  # noqa: E402
from pypto.runtime.runner import RunConfig  # noqa: E402

# Environment variable names shared with test_runner.py
_FAILURE_RECORDS_ENV = "PYPTO_FAILURE_RECORDS_DIR"
_CURRENT_TEST_ENV = "PYPTO_CURRENT_TEST_NODEID"

# Module-level storage so pytest_terminal_summary can find the records dir
# even after the session fixture has torn down.
_failure_records_dir: str | None = None
# True only in the process that created the temp dir. Under --forked, each
# child subprocess sees _failure_records_dir set but is NOT the owner.
_is_failure_records_owner: bool = False


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


def pytest_sessionstart(session):
    """Create the shared failure records directory at session start.

    This hook runs at the beginning of each pytest session. In the parent
    process it creates the temp dir and owns it. Under --forked, each child
    subprocess inherits PYPTO_FAILURE_RECORDS_DIR and reuses the same
    directory without taking ownership. Only the owner renders the summary
    and cleans up; child sessions skip both steps.
    """
    global _failure_records_dir, _is_failure_records_owner  # noqa: PLW0603
    existing = os.environ.get(_FAILURE_RECORDS_ENV)
    if existing:
        _failure_records_dir = existing
        return
    records_dir = tempfile.mkdtemp(prefix="pypto_st_failures_")
    _failure_records_dir = records_dir
    _is_failure_records_owner = True
    os.environ[_FAILURE_RECORDS_ENV] = records_dir


@pytest.fixture(autouse=True)
def _set_current_test_nodeid(request):
    """Expose the current pytest node ID to TestRunner via env var.

    TestRunner.run() reads PYPTO_CURRENT_TEST_NODEID to tag failure records
    with the full pytest test path. Under --forked, this fixture runs in the
    forked child process, which is where TestRunner also runs — correct.
    """
    os.environ[_CURRENT_TEST_ENV] = request.node.nodeid
    yield
    os.environ.pop(_CURRENT_TEST_ENV, None)


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
        choices=["Default", "CCE"],
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
    """Register custom markers."""
    config.addinivalue_line("markers", "hardware: mark test as requiring hardware (--platform=a2a3)")
    config.addinivalue_line("markers", "a5: mark test as requiring Ascend 950 (--platform=a5 or a5sim)")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "fuzz: mark test as fuzz test")


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


# ---------------------------------------------------------------------------
# Harness failure report
# ---------------------------------------------------------------------------


def _load_failure_records(records_dir: Path) -> list[dict]:
    """Read all JSON failure records from *records_dir*."""
    records = []
    for path in sorted(records_dir.glob("failure_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    return records


def _render_failure_table(terminalreporter, records: list[dict]) -> None:
    """Render the harness failure summary table to the terminal.

    Produces one row per (test case, function) pair. Compilation errors expand
    into one row per failed kernel function; all other errors produce a single
    row with function="-".
    """
    max_error_width = 60

    # Flatten records into rows: (case_name, file_path, error_type, function, summary)
    rows: list[tuple[str, str, str, str, str]] = []
    for rec in records:
        nodeid = rec.get("nodeid", "")
        file_path = nodeid.split("::")[0] if "::" in nodeid else nodeid
        case_name = rec.get("case_name", "?")
        error_type = rec.get("error_type", "Error")
        for fe in rec.get("func_errors", [{"function": "-", "summary": ""}]):
            rows.append((case_name, file_path, error_type, fe.get("function", "-"), fe.get("summary", "")))

    headers = ("Case Name", "File", "Error Type", "Function", "Error")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    col_widths[-1] = min(col_widths[-1], max_error_width)
    col_widths = [w + 2 for w in col_widths]

    sep = "-+-".join("-" * w for w in col_widths)
    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))

    tw = terminalreporter
    tw.write_sep("=", "ST Failure Summary", red=True, bold=True)
    tw.write_line("")
    tw.write_line("  " + header_line)
    tw.write_line("  " + sep)

    for case_name, file_path, error_type, function, summary in rows:
        # Wrap long error summaries
        wrapped = textwrap.wrap(summary, width=max_error_width) or [summary]
        cells = [
            f"{case_name:<{col_widths[0]}}",
            f"{file_path:<{col_widths[1]}}",
            f"{error_type:<{col_widths[2]}}",
            f"{function:<{col_widths[3]}}",
            f"{wrapped[0]:<{col_widths[4]}}",
        ]
        tw.write_line("  " + " | ".join(cells), red=True)
        for continuation in wrapped[1:]:
            blank_cells = [" " * col_widths[i] for i in range(4)]
            tw.write_line("  " + " | ".join(blank_cells) + " | " + continuation, red=True)

    tw.write_line("  " + sep)
    tw.write_line("")
    unique_cases = len({r.get("case_name") for r in records})
    tw.write_line(f"  {unique_cases} test case(s) failed ({len(rows)} failure(s) total)", red=True)
    tw.write_sep("=", "", red=True)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Render the harness failure summary after all tests complete.

    Only the owner process (the one that created the temp dir) renders the
    table and cleans up. Under --forked, child subprocesses are not owners
    and skip this step entirely to avoid deleting the shared directory before
    the parent has had a chance to read it.
    """
    if not _is_failure_records_owner:
        return
    records_dir_str = _failure_records_dir or os.environ.get(_FAILURE_RECORDS_ENV)
    if not records_dir_str:
        return
    records_dir = Path(records_dir_str)
    if not records_dir.is_dir():
        return
    try:
        records = _load_failure_records(records_dir)
        if records:
            _render_failure_table(terminalreporter, records)
    finally:
        shutil.rmtree(records_dir, ignore_errors=True)
