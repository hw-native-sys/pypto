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

import os
import random
import sys
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
from harness.core.environment import ensure_simpler_available  # noqa: E402
from harness.core.harness import TestConfig  # noqa: E402
from harness.core.test_runner import TestRunner  # noqa: E402


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

    from harness.core.environment import (  # noqa: PLC0415
        get_simpler_python_path,
        get_simpler_scripts_path,
    )

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
        choices=["a2a3sim", "a2a3"],
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
        choices=["Default", "PTOAS"],
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
def test_config(request) -> TestConfig:
    """Session-scoped fixture providing test configuration from CLI options.

    Session scope means the config is created once and shared across all tests,
    which is appropriate since CLI options don't change during a test run.
    """
    # Determine save_kernels_dir
    save_kernels = request.config.getoption("--save-kernels")
    save_kernels_dir = None
    if save_kernels:
        kernels_dir = request.config.getoption("--kernels-dir")
        # If --kernels-dir is specified, use it; otherwise None will use session output directory
        save_kernels_dir = kernels_dir

    return TestConfig(
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
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "fuzz: mark test as fuzz test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on platform."""
    platform = config.getoption("--platform")

    skip_hardware = pytest.mark.skip(reason="hardware tests require --platform=a2a3")

    for item in items:
        if "hardware" in item.keywords and platform != "a2a3":
            item.add_marker(skip_hardware)
