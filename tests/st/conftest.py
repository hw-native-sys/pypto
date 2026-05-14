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
import queue
import random
import re
import shutil
import sys
import tempfile
import warnings
from collections import Counter
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
    shutdown_pipeline,
    start_pipeline,
)
from pypto import LogLevel, set_log_level  # noqa: E402
from pypto.runtime.runner import RunConfig  # noqa: E402

# Temp directories created for pre-compilation (when --save-kernels is not set).
# Cleaned up in pytest_sessionfinish.
_temp_precompile_dirs: list[Path] = []

# Per-device test counter populated by ``_report_device`` and dumped at
# session end via ``pytest_terminal_summary``.
_device_counter: Counter[int] = Counter()


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
        default="a2a3",
        help=(
            "Comma-separated allowlist of target platforms; each test under "
            "tests/st/runtime/ is parametrized over a2a3, a5, a2a3sim, a5sim "
            "and only variants whose id appears here are run "
            "(default: a2a3, matching legacy CI behaviour). Legacy "
            "non-parametrized tests inherit this value as their platform."
        ),
    )
    parser.addoption(
        "--device",
        action="store",
        default="0",
        type=str,
        help=(
            "Device id(s) for hardware tests. Accepts a single id ('0'), an "
            "inclusive range ('0-7'), or a comma-separated list ('0,1,12'). "
            "Ranges and lists may be mixed ('0-3,8,12-15'). All ids are "
            "placed into a session-wide pool that bounds execute-task "
            "parallelism; per-test device selection happens inside the "
            "pipeline execute task (default: 0)."
        ),
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
        "--runtime-log-level",
        action="store",
        default=None,
        help="PyPTO runtime log level (debug, v0..v9, info, warn, error, null). "
        "Default: leave the runtime logger at its V5/INFO default.",
    )
    parser.addoption(
        "--pto-isa-commit",
        action="store",
        default=None,
        help="Pin the pto-isa clone to a specific git commit (hash or tag). Default: use latest remote HEAD.",
    )
    # ── DFX (Design For X) toggles ────────────────────────────────────────
    # Each maps 1:1 to the same-named field on ``RunConfig`` and to the
    # corresponding ``CallConfig`` member on the runtime side. Names match
    # ``runtime/conftest.py`` so the two surfaces stay aligned.
    parser.addoption(
        "--enable-l2-swimlane",
        action="store_true",
        default=False,
        help="Capture per-task L2 perf records into <work_dir>/dfx_outputs/l2_perf_records.json "
        "and render merged_swimlane_*.json after execution.",
    )
    parser.addoption(
        "--runtime-profiling",
        action="store_true",
        default=False,
        help=(
            "DEPRECATED alias for --enable-l2-swimlane. Kept so existing "
            "scripts keep working; will be removed in a future release."
        ),
    )
    parser.addoption(
        "--dump-tensor",
        action="store_true",
        default=False,
        help="Dump per-task tensor I/O into <work_dir>/dfx_outputs/tensor_dump/.",
    )
    parser.addoption(
        "--enable-dep-gen",
        action="store_true",
        default=False,
        help="Capture PTO2 dependency edges into <work_dir>/dfx_outputs/deps.json "
        "and render deps_graph.html.",
    )
    parser.addoption(
        "--enable-pmu",
        nargs="?",
        const=2,
        default=0,
        type=int,
        metavar="EVENT_TYPE",
        help="Enable AICore PMU CSV collection. Bare flag = PIPE_UTILIZATION(2). "
        "Pass an event type (e.g. 4 = MEMORY) to override.",
    )


def _parse_device_option(raw: str | int) -> list[int]:
    """Parse the ``--device`` option into a list of device ids.

    Accepts a single integer (``"0"`` or ``0``), an inclusive range
    (``"0-7"``), a comma-separated list (``"0,1,12"``), or any combination
    (``"0-3,8,12-15"``). Device ids may be non-contiguous.
    """
    text = str(raw).strip()
    if not text:
        raise pytest.UsageError("--device must not be empty")

    devices: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                start_str, end_str = token.split("-", 1)
                start, end = int(start_str), int(end_str)
                if end < start:
                    raise pytest.UsageError(f"--device range must be non-decreasing, got {token!r}")
                devices.extend(range(start, end + 1))
            else:
                devices.append(int(token))
        except ValueError:
            raise pytest.UsageError(f"Invalid device ID or range in --device: {token!r}") from None

    if not devices:
        raise pytest.UsageError(f"--device yielded no device ids: {raw!r}")
    # Preserve order while deduplicating (user ordering dictates worker mapping).
    return list(dict.fromkeys(devices))


def _resolve_device_id(raw: str | int) -> int:
    """Return a representative device id for the session ``RunConfig``.

    Per-test device selection happens inside the pipeline execute task
    (see ``_fused_execute_task`` in ``harness.core.test_runner``), which
    pulls from a session-wide pool seeded with every id from ``--device``.
    This value is consulted only by the legacy inline-compile fallback in
    :meth:`TestRunner._run_inline` when a test case was not discovered at
    collection time, so the first id is sufficient.
    """
    return _parse_device_option(raw)[0]


def _parse_platform_filter(raw: str) -> tuple[str, ...]:
    """Parse the comma-separated ``--platform`` value into an ordered tuple.

    The returned tuple preserves the order in which the user wrote the ids on
    the command line (and de-duplicates them) so that downstream consumers
    that need a single representative platform – e.g. the precompile fallback
    – pick a deterministic value instead of an arbitrary set element.

    An empty string (the user passed nothing) yields an empty tuple, which
    callers expand to "every known platform". A non-empty input that contains
    *only* unknown ids raises ``pytest.UsageError`` so a typo such as
    ``--platform=a2a3typo`` fails loudly instead of silently expanding to the
    full platform set.
    """
    tokens = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    canonical = set(ALL_PLATFORM_IDS)
    valid = tuple(dict.fromkeys(tok for tok in tokens if tok in canonical))
    if tokens and not valid:
        raise pytest.UsageError(
            f"--platform must include at least one of: {', '.join(ALL_PLATFORM_IDS)}; got {raw!r}"
        )
    return valid


@pytest.fixture(autouse=True)
def _report_device(request) -> None:
    """Report which device executed each test at the end of the test body.

    ``TestRunner.run`` writes the resolved device id into a single-slot
    stash (``_last_device``) right before returning to the test body.  We
    read it after ``yield`` so the line shows the device the test actually
    ran on.  Tests that don't go through ``TestRunner.run`` see ``None``
    and are skipped from the per-device counter.
    """
    yield
    from harness.core.test_runner import _last_device  # noqa: PLC0415

    device_id = _last_device["value"]
    _last_device["value"] = None
    if device_id is None:
        return
    sys.stdout.write(f"\n[DEVICE] {request.node.nodeid} -> device {device_id}\n")
    sys.stdout.flush()
    _device_counter[device_id] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    """Emit a per-device test count summary at the end of the session."""
    if not _device_counter:
        return
    total = sum(_device_counter.values())
    terminalreporter.write_sep("=", f"per-device test count ({total} total)")
    for dev in sorted(_device_counter):
        terminalreporter.write_line(f"  device {dev:>3}: {_device_counter[dev]} tests")


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
    fallback_platform = platform_filter[0] if platform_filter else "a2a3"

    return RunConfig(
        platform=fallback_platform,
        device_id=_resolve_device_id(request.config.getoption("--device")),
        save_kernels=save_kernels,
        save_kernels_dir=save_kernels_dir,
        dump_passes=request.config.getoption("--dump-passes"),
        codegen_only=request.config.getoption("--codegen-only"),
        pto_isa_commit=request.config.getoption("--pto-isa-commit"),
        # ``--runtime-profiling`` is the deprecated alias for
        # ``--enable-l2-swimlane``; OR them so existing scripts keep working
        # (``RunConfig.__post_init__`` emits the DeprecationWarning on the
        # field path; the bool here doesn't need its own warning).
        enable_l2_swimlane=(
            request.config.getoption("--enable-l2-swimlane")
            or request.config.getoption("--runtime-profiling")
        ),
        enable_dump_tensor=request.config.getoption("--dump-tensor"),
        enable_pmu=request.config.getoption("--enable-pmu"),
        enable_dep_gen=request.config.getoption("--enable-dep-gen"),
    )


@pytest.fixture(scope="session")
def device_ids(request) -> list[int]:
    """Session-scoped fixture returning the full ``--device`` list.

    Distributed tests need access to all allocated device ids (not just the
    first one stored in ``RunConfig.device_id``) so they can pick a slice
    that matches the CI runner's dynamic allocation rather than hardcoding.
    """
    return _parse_device_option(request.config.getoption("--device"))


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

    # Set PyPTO runtime log level (orthogonal to PyPTO C++ logger above).
    try:
        runtime_level = config.getoption("--runtime-log-level")
    except KeyError:
        pass  # option not yet registered (e.g. during --co --help)
    else:
        if runtime_level is not None:
            from pypto.runtime import configure_log  # noqa: PLC0415

            configure_log(runtime_level)  # ValueError propagates: invalid CLI value must fail fast


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
    cli_platforms = _parse_platform_filter(config.getoption("--platform"))
    cli_filter = set(cli_platforms or ALL_PLATFORM_IDS)
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

    max_workers: int | None = session.config.getoption("--precompile-workers")
    # Without --precompile-workers the pipeline is skipped entirely; each
    # test compiles + executes inline inside TestRunner._run_inline().
    if max_workers is None:
        return

    dump_passes: bool = session.config.getoption("--dump-passes")
    codegen_only: bool = session.config.getoption("--codegen-only")
    pto_isa_commit: str | None = session.config.getoption("--pto-isa-commit")
    # ``--runtime-profiling`` is the deprecated alias for ``--enable-l2-swimlane``;
    # OR them so existing scripts keep working. Warn once at the session boundary
    # so the deprecation surfaces even when no per-test ``RunConfig`` is built.
    deprecated_runtime_profiling: bool = session.config.getoption("--runtime-profiling")
    if deprecated_runtime_profiling:
        warnings.warn(
            "--runtime-profiling is deprecated; use --enable-l2-swimlane instead. "
            "The two are aliased today and --runtime-profiling will be removed in "
            "a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    enable_l2_swimlane: bool = (
        session.config.getoption("--enable-l2-swimlane") or deprecated_runtime_profiling
    )
    enable_dump_tensor: bool = session.config.getoption("--dump-tensor")
    enable_pmu: int = session.config.getoption("--enable-pmu")
    enable_dep_gen: bool = session.config.getoption("--enable-dep-gen")

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

    # ``--platform`` is a CSV allowlist; the per-test value resolved by
    # ``tc.get_platform()`` overrides this fallback inside the pipeline.
    platform_filter = _parse_platform_filter(session.config.getoption("--platform"))
    session_platform: str = platform_filter[0] if platform_filter else "a2a3"

    # Build the device pool from --device.  N parallel executes max.
    devices = _parse_device_option(session.config.getoption("--device"))
    device_pool: queue.Queue[int] = queue.Queue()
    for d in devices:
        device_pool.put(d)

    test_cases = list(seen.values())
    print(
        f"\n[PyPTO] Pipeline: {len(test_cases)} test case(s); "
        f"compile_workers={max_workers}, devices={devices}"
    )
    start_pipeline(
        test_cases=test_cases,
        cache_dir=cache_dir,
        session_platform=session_platform,
        dump_passes=dump_passes,
        codegen_only=codegen_only,
        pto_isa_commit=pto_isa_commit,
        compile_workers=max_workers,
        device_pool=device_pool,
        enable_l2_swimlane=enable_l2_swimlane,
        enable_dump_tensor=enable_dump_tensor,
        enable_pmu=enable_pmu,
        enable_dep_gen=enable_dep_gen,
    )
    print("[PyPTO] Pipeline scheduled — pytest item loop starting\n")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Tear down the pipeline and clean up temporary precompile directories."""
    shutdown_pipeline()
    for d in _temp_precompile_dirs:
        shutil.rmtree(d, ignore_errors=True)
