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

import ast
import inspect
import queue
import shutil
import sys
import tempfile
import textwrap
import warnings
from collections import Counter
from contextlib import nullcontext
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
    configure_inline_task_submit,
    execution_summary_lines,
    set_current_item_platform,
    shutdown_pipeline,
    start_pipeline,
)
from pypto import LogLevel  # noqa: E402
from pypto.pypto_core import _clear_thread_log_level, _set_thread_log_level  # noqa: E402
from pypto.pypto_core.passes import MemoryPlanner  # noqa: E402
from pypto.runtime.runner import (  # noqa: E402
    _SWIMLANE_CLI_HELP,
    _SWIMLANE_FULL_LEVEL,
    _SWIMLANE_MAX_LEVEL,
    RunConfig,
)

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


def _resolve_swimlane_option(config: pytest.Config) -> int:
    """Return the requested chip-swimlane collection level.

    Precedence: an explicit ``--chip-swimlane-level N`` wins, then the bare
    ``--enable-chip-swimlane`` (full level), then the deprecated bare
    ``--enable-l2-swimlane`` (same level, with a warning). Absent means off.
    """
    level: int | None = config.getoption("--chip-swimlane-level")
    if level is not None:
        return level
    if config.getoption("--enable-chip-swimlane"):
        return _SWIMLANE_FULL_LEVEL
    if config.getoption("enable_l2_swimlane_deprecated"):
        warnings.warn(
            "--enable-l2-swimlane is deprecated; use --enable-chip-swimlane (or --chip-swimlane-level N).",
            DeprecationWarning,
            stacklevel=2,
        )
        return _SWIMLANE_FULL_LEVEL
    return 0


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--platform",
        action="store",
        default="a2a3",
        help=(
            "Comma-separated allowlist of target platforms out of a2a3, a5, "
            "a2a3sim, a5sim (default: a2a3, matching legacy CI behaviour). "
            "Every test using the test_runner fixture is expanded over this "
            "list, intersected with its @pytest.mark.platforms marker; tests "
            "that parametrize `platform` themselves keep their own list and "
            "only the variants named here run. A single active platform is not "
            "parametrized, so node ids stay unsuffixed."
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
        "--memory-planner",
        action="store",
        default="default",
        choices=["default", "pypto", "dsa-rp", "ptoas"],
        help=(
            "Session-wide memory planner for test cases that do not select one explicitly: "
            "default (defer to PyPTO), pypto, dsa-rp, or ptoas. An explicit planner on a "
            "PTOTestCase takes precedence (default: default)."
        ),
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
        "--execute-via-task-submit",
        action="store_true",
        default=False,
        help="Borrow an NPU per case from the host-level 'task-submit --device auto' queue for "
        "the execute step (compile + golden stay card-free). Orthogonal to --precompile-workers; "
        "machines without task-submit must NOT pass this (default: False).",
    )
    parser.addoption(
        "--task-max-time",
        action="store",
        default=600,
        type=int,
        help="Per-case execution cap passed to 'task-submit --max-time' (seconds, default: 600).",
    )
    parser.addoption(
        "--task-queue-timeout",
        action="store",
        default=1800,
        type=int,
        help="Card-queue wait cap passed to 'task-submit --timeout' (seconds). Must be >= the "
        "longest task; deep queues need a large value (default: 1800).",
    )
    parser.addoption(
        "--task-submit-device",
        action="store",
        default="auto",
        help="Value for 'task-submit --device' in task-submit mode: 'auto' (borrow any free "
        'card) or a specific id/range to pin to (e.g. "$DEVICE_RANGE") for validating the '
        "flow before trusting auto-allocation (default: auto).",
    )
    parser.addoption(
        "--execute-batch-size",
        action="store",
        default=64,
        type=int,
        help="Artifacts per task-submit task in task-submit mode. Each batch runs in ONE hot "
        "process (one torch/NPU init for the whole batch), amortizing cold-start cost. Smaller "
        "= more parallel tasks (more cards) but more init; larger = fewer inits but less "
        "parallelism (default: 64).",
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
        help="PyPTO runtime log level (debug, info, timing, warn, error, null). "
        "Default: leave the runtime logger at its TIMING default.",
    )
    parser.addoption(
        "--analyze-auto-scopes-for-deps",
        action="store_true",
        default=False,
        help=(
            "Enable compile-time AUTO-scope task dependency derivation for both inline and precompiled runs."
        ),
    )
    # ── DFX (Design For X) toggles ────────────────────────────────────────
    # Each maps to the same-named field on ``RunConfig`` and to the corresponding
    # runtime ``CallConfig`` member.
    #
    # The bare enable flag and the level-valued form are deliberately two
    # options. An ``nargs="?"`` option greedily eats the next non-dash token, so
    # a single option would turn the conventional
    # ``pytest --enable-chip-swimlane tests/st/runtime/`` into
    # "invalid int value: 'tests/st/runtime/'" — the flag used to be
    # ``store_true`` and that ordering has always worked. Keeping the bare flag
    # valueless makes it order-independent again.
    parser.addoption(
        "--enable-chip-swimlane",
        action="store_const",
        const=_SWIMLANE_FULL_LEVEL,
        default=0,
        help=f"Enable chip swimlane capture at the full level ({_SWIMLANE_FULL_LEVEL}), matching the "
        "runtime harness's bare --enable-chip-swimlane. Use --chip-swimlane-level N for a lower "
        "level. Records are written into <work_dir>/dfx_outputs/chip_swimlane_records.json. "
        "On onboard platforms, also render merged_swimlane_*.json and run the kernel twice: a dep_gen "
        "pass to capture deps.json (the converter's task graph) then a clean swimlane pass, since "
        "dep_gen collection perturbs the timing. Simulator platforms emit only the records (the merged "
        "swimlane is skipped).",
    )
    parser.addoption(
        "--chip-swimlane-level",
        default=None,
        type=int,
        choices=range(_SWIMLANE_MAX_LEVEL + 1),
        metavar="PERF_LEVEL",
        help=_SWIMLANE_CLI_HELP + " Takes precedence over --enable-chip-swimlane.",
    )
    # Deprecated spelling, kept because CI and existing scripts still pass it.
    # Valueless like the original ``store_true`` form it replaces.
    parser.addoption(
        "--enable-l2-swimlane",
        dest="enable_l2_swimlane_deprecated",
        action="store_true",
        default=False,
        help="Deprecated alias for --enable-chip-swimlane.",
    )
    parser.addoption(
        "--dump-args",
        nargs="?",
        type=int,
        const=1,
        default=0,
        help="Per-task argument dump level into <work_dir>/dfx_outputs/args_dump/. "
        "Bare flag = 1 (partial: only pl.dump_tag / dumps= marked tensors); "
        "'--dump-args 2' = full (every task); absent = 0 (off).",
    )
    parser.addoption(
        "--enable-dep-gen",
        action="store_true",
        default=False,
        help="Capture simpler dependency edges into <work_dir>/dfx_outputs/deps.json "
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
    parser.addoption(
        "--enable-scope-stats",
        action="store_true",
        default=False,
        help="Capture per-scope ring-fill peaks into <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl.",
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


def _parse_memory_planner(raw: str) -> MemoryPlanner | None:
    """Translate the system-test CLI spelling into the public planner enum."""
    return {
        "default": None,
        "pypto": MemoryPlanner.PYPTO,
        "dsa-rp": MemoryPlanner.DSA_RP,
        "ptoas": MemoryPlanner.PTOAS,
    }[raw]


@pytest.fixture(autouse=True)
def _report_device(request) -> None:
    """Report which device executed each test at the end of the test body.

    ``TestRunner.run`` writes the resolved device id into a single-slot
    stash (``_last_device``) right before returning to the test body.  We
    read it after ``yield`` so the line shows the device the test actually
    ran on.  Tests that don't go through ``TestRunner.run`` see ``None``
    and are skipped from the per-device counter.

    The write runs in the fixture's teardown phase, which pytest captures
    per-test and discards for passing tests.  We suspend capture so the line
    reaches the real terminal regardless of test outcome (otherwise it only
    surfaced on failures or under ``-s``).
    """
    yield
    from harness.core.test_runner import _last_device  # noqa: PLC0415

    device_id = _last_device["value"]
    _last_device["value"] = None
    if device_id is None:
        return
    line = f"[DEVICE] {request.node.nodeid} -> device {device_id}"
    capmanager = request.config.pluginmanager.getplugin("capturemanager")
    # Suspend capture (when a capturemanager is present) so the line reaches the
    # real terminal regardless of test outcome; otherwise write it directly.
    disabled = capmanager.global_and_fixture_disabled() if capmanager is not None else nullcontext()
    with disabled:
        sys.stdout.write(f"\n{line}\n")
        sys.stdout.flush()
    _device_counter[device_id] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    """Emit a per-device test count + task-submit batch summary at session end.

    In task-submit mode the per-artifact device markers are suppressed for a
    clean log, so this is the one place the borrowed-card batched execution is
    made visible: how many batches ran and how the runs spread across cards.
    """
    batch_lines = execution_summary_lines()
    if not _device_counter and not batch_lines:
        return
    if _device_counter:
        total = sum(_device_counter.values())
        terminalreporter.write_sep("=", f"per-device test count ({total} total)")
        for dev in sorted(_device_counter):
            terminalreporter.write_line(f"  device {dev:>3}: {_device_counter[dev]} tests")
    for line in batch_lines:
        terminalreporter.write_line(f"  task-submit: {line}")


@pytest.fixture(autouse=True)
def _redirect_prog_build_dir(request, tmp_path, monkeypatch):
    """Redirect default ir.compile() output into pytest's per-test tmp dir.

    Direct ``ir.compile()`` calls and the inline-compile fallback in
    ``TestRunner`` otherwise write to ``build_output/<name>_<timestamp>``
    relative to the working directory, leaving stale dirs behind. The
    precompile pipeline already passes an explicit ``output_dir`` and so is
    unaffected by ``PYPTO_PROG_BUILD_DIR``.

    When ``--save-kernels`` is set the user wants artifacts preserved under
    ``build_output/``, so redirection is skipped — and any ``PYPTO_PROG_BUILD_DIR``
    inherited from the outer environment is cleared so the default base
    genuinely stays ``build_output``.
    """
    if request.config.getoption("--save-kernels"):
        monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
        return
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "build_output"))


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
        enable_chip_swimlane=_resolve_swimlane_option(request.config),
        enable_dump_args=request.config.getoption("--dump-args"),
        enable_pmu=request.config.getoption("--enable-pmu"),
        enable_dep_gen=request.config.getoption("--enable-dep-gen"),
        enable_scope_stats=request.config.getoption("--enable-scope-stats"),
        analyze_auto_scopes_for_deps=request.config.getoption("--analyze-auto-scopes-for-deps"),
        memory_planner=_parse_memory_planner(request.config.getoption("--memory-planner")),
    )


@pytest.fixture(scope="session")
def device_ids(request) -> list[int]:
    """Session-scoped fixture returning the full ``--device`` list.

    Distributed tests need access to all allocated device ids (not just the
    first one stored in ``RunConfig.device_id``) so they can pick a slice
    that matches the CI runner's dynamic allocation rather than hardcoding.
    """
    return _parse_device_option(request.config.getoption("--device"))


def _resolve_item_platform(node: pytest.Item | pytest.FixtureRequest, config: pytest.Config) -> str:
    """Return the platform *node* runs on.

    Resolution order:
        1. An explicit ``platform`` parametrize on the test itself.
        2. The value injected by :func:`pytest_generate_tests` (the platform
           matrix), for tests that never mention ``platform``.
        3. The first ``--platform`` id its ``@pytest.mark.platforms`` marker
           allows, matching the legacy single-platform behaviour.

    Args:
        node: The item (or a request pointing at one) whose platform is wanted.
        config: The session config, read for the ``--platform`` fallback.

    Returns:
        One of :data:`ALL_PLATFORM_IDS`.
    """
    callspec = getattr(node, "callspec", None)
    params = callspec.params if callspec else {}
    resolved = params.get("platform") or params.get("_st_platform")
    if resolved:
        return resolved
    cli = list(_parse_platform_filter(config.getoption("--platform")) or ALL_PLATFORM_IDS)
    marker = node.get_closest_marker("platforms") if hasattr(node, "get_closest_marker") else None
    if marker is not None:
        item_filter = _marker_platforms(marker, "platforms", getattr(node, "name", "<item>"))
        # An item the marker excludes from every active platform is deselected by
        # pytest_collection_modifyitems; keep the raw first id so this resolver
        # never has to invent one.
        cli = [p for p in cli if p in item_filter] or cli
    return cli[0]


def _marker_platforms(marker: pytest.Mark, marker_name: str, test_name: str) -> set[str]:
    """Return the platform ids a ``platforms`` / ``platform_xfail`` marker names.

    A typo used to narrow the marker's id set silently: the unknown id matched
    no platform, so the test was deselected everywhere and looked like it had
    simply never been written. Fail collection instead.

    Args:
        marker: The marker to read.
        marker_name: Its name, for the error message.
        test_name: The test carrying it, for the error message.

    Returns:
        The validated platform ids.

    Raises:
        pytest.UsageError: The marker names no platform, or an unknown one.
    """
    unknown = [arg for arg in marker.args if arg not in ALL_PLATFORM_IDS]
    if not marker.args or unknown:
        detail = f"unknown platform id(s) {unknown}" if unknown else "no platform ids"
        raise pytest.UsageError(
            f"@pytest.mark.{marker_name} on {test_name} has {detail}; "
            f"expected one or more of: {', '.join(ALL_PLATFORM_IDS)}"
        )
    return set(marker.args)


def _direct_argnames(func: Any) -> set[str]:
    """Return the fixture names *func* requests in its own signature.

    ``Metafunc.fixturenames`` is the whole closure, so it cannot tell a test
    that takes ``test_runner`` from one that merely depends on a fixture which
    does. Only the former can be expanded per platform.

    Args:
        func: The test function.

    Returns:
        The parameter names of the function signature.
    """
    try:
        return set(inspect.signature(func).parameters)
    except (TypeError, ValueError):  # builtins / unsupported callables
        return set()


@pytest.fixture(autouse=True)
def _st_platform(request) -> str:
    """Return the platform this item runs on (see :func:`_resolve_item_platform`).

    Autouse so that :func:`pytest_generate_tests` always has this fixture in the
    closure to parametrize; the platform matrix expands *this* name rather than
    the test signature, which is what keeps test bodies unchanged.
    """
    return _resolve_item_platform(request.node, request.config)


@pytest.fixture(scope="session")
def test_runner(test_config) -> TestRunner:
    """Session-scoped fixture providing a test runner instance.

    Deliberately still session-scoped: module- and session-scoped fixtures in
    the suite request it, and a function-scoped runner makes those a
    ``ScopeMismatch``. The per-item platform reaches the runner through
    ``set_current_item_platform`` (published by :func:`pytest_runtest_setup`)
    rather than through a differently-scoped fixture.
    """
    return TestRunner(test_config)


@pytest.fixture
def optimization_strategy(request) -> str:
    """Fixture providing the optimization strategy from CLI options."""
    return request.config.getoption("--strategy")


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
    """Register custom markers and apply early runtime settings."""
    config.addinivalue_line(
        "markers",
        "platforms(*ids, reason=...): restrict the test to the given platform ids "
        "(intersected with the --platform CLI filter). State why in reason= — it "
        "is the only record of whether the case is limited by the platform or "
        "waiting on a fix.",
    )
    config.addinivalue_line(
        "markers",
        "platform_xfail(*ids, reason=..., strict=True): expect failure on the "
        "given platforms. The case still runs, so an XPASS reports the fix "
        "instead of freezing the verdict; prefer this over excluding the case "
        "from a guard job with -k.",
    )
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line(
        "markers",
        "device_batch: auto-applied to tests that execute via test_runner.run "
        "(PTOTestCase) — compile/golden card-free, device run batched through "
        "task-submit. Tests WITHOUT it call the device directly (@pl.jit / "
        "config=test_config) and must run in-process. CI selects them with "
        "`-m device_batch` (batched step) vs `-m 'not device_batch'` (in-process "
        "step); the split is by fixture usage, so new tests self-classify with no "
        "ci.yml change.",
    )

    # Set the PyPTO runtime log level independently of the per-ST-item C++ logger.
    try:
        runtime_level = config.getoption("--runtime-log-level")
    except KeyError:
        pass  # option not yet registered (e.g. during --co --help)
    else:
        if runtime_level is not None:
            from pypto.runtime import configure_log  # noqa: PLC0415

            configure_log(runtime_level)  # ValueError propagates: invalid CLI value must fail fast


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Apply the requested C++ log level only while an ST item is running.

    ``tests/st/conftest.py`` can be loaded in a mixed ST/UT session.  Changing
    the process-global C++ logger in ``pytest_configure`` therefore suppressed
    diagnostics expected by later unit tests.  Surround the complete pytest
    protocol (fixture setup, test call, and teardown) so forked runtime workers
    still inherit the requested level, then clear the thread-local override.
    """
    del nextitem
    if not item.path.is_relative_to(_ST_DIR):
        yield
        return

    try:
        level_name: str = item.config.getoption("--pypto-log-level")
        _set_thread_log_level(LogLevel[level_name])
        yield
    finally:
        _clear_thread_log_level()


def pytest_itemcollected(item):
    """Auto-classify each test into the batched (A) vs in-process (B) execute path.

    The discriminator is fixture usage, resolved at collection time — so a new
    test needs no ci.yml edit to be routed correctly, and a file mixing both
    styles is split per-test:

    - Uses the ``test_runner`` fixture (``test_runner.run(PTOTestCase)``) → mark
      ``device_batch``: the harness pre-compiles it card-free and runs its device
      step through the batched task-submit pipeline.
    - Otherwise (direct ``@pl.jit`` kernel call / ``config=test_config``, or no
      device at all) → left unmarked: CI runs it in-process via
      ``-m 'not device_batch'`` (one task-submit card session), which is the only
      correct path for a synchronous direct-device call.
    """
    if isinstance(item, pytest.Function) and "test_runner" in getattr(item, "fixturenames", ()):
        item.add_marker("device_batch")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Expand ``test_runner``-based tests over the active platform allowlist.

    A test that declares ``platform`` itself keeps its own parametrize.
    Everything else is expanded over the ``--platform`` allowlist intersected
    with its ``@pytest.mark.platforms``, so a case joins a new platform's guard
    job without touching its test body.

    The test must take ``test_runner`` itself. A test that reaches the runner
    through a module- or session-scoped fixture cannot vary per item — that
    fixture is built once and shared — so expanding it would hand every variant
    the first platform's artefact. Those items run on the single platform
    :func:`_resolve_item_platform` picks for them.

    Expansion only happens when the CLI names more than one platform: a plain
    ``--platform=a2a3`` run keeps today's node ids, and only a multi-platform
    invocation grows the ``[a2a3]`` / ``[a5]`` suffixes. Within such a run a
    marker that narrows the test to one platform is still parametrized, so the
    item carries that platform instead of falling back to the first CLI id.
    """
    direct = _direct_argnames(metafunc.function)
    if "test_runner" not in direct:
        return
    if "platform" in metafunc.fixturenames:
        return
    cli = list(_parse_platform_filter(metafunc.config.getoption("--platform")) or ALL_PLATFORM_IDS)
    marker = metafunc.definition.get_closest_marker("platforms")
    item_filter = (
        _marker_platforms(marker, "platforms", metafunc.definition.name) if marker is not None else None
    )
    # A marker that excludes every active platform leaves the variants to
    # pytest_collection_modifyitems, which deselects them; parametrizing an
    # empty list would instead leave one item behind marked "empty parameter set".
    allowed = [p for p in cli if item_filter is None or p in item_filter] or cli
    if len(cli) > 1:
        metafunc.parametrize("_st_platform", allowed, ids=list(allowed), indirect=True)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Publish this item's platform, then apply ``@pytest.mark.platform_xfail``.

    ``tryfirst`` so the platform is published before pytest's own
    ``pytest_runtest_setup`` builds this item's fixtures: a module-scoped
    fixture that calls ``test_runner.run()`` during that setup must compile for
    the platform of the item that triggered it, not for the session default.

    The xfail half is a platform-conditional expectation, which a plain
    ``xfail`` cannot spell.

    A platform-conditional expectation cannot be spelled with a plain
    ``xfail``, and raising ``pytest.xfail()`` inside the body aborts before the
    test runs, so a later fix can never surface as an XPASS. Attaching the
    marker here keeps the case running on every platform and reports the fix
    the moment it lands.

    Args:
        item: The item about to run.

    Raises:
        pytest.UsageError: The marker names no/unknown platforms, or no reason.
    """
    set_current_item_platform(_resolve_item_platform(item, item.config))

    marker = item.get_closest_marker("platform_xfail")
    if marker is None:
        return
    platforms = _marker_platforms(marker, "platform_xfail", item.name)
    reason = marker.kwargs.get("reason")
    if not reason:
        raise pytest.UsageError(
            f"@pytest.mark.platform_xfail on {item.name} needs reason=... — an "
            "unexplained expected failure cannot be told apart from a stale one."
        )
    if _resolve_item_platform(item, item.config) in platforms:
        item.add_marker(pytest.mark.xfail(reason=reason, strict=marker.kwargs.get("strict", True)))


def pytest_runtest_teardown(item: pytest.Item) -> None:  # noqa: ARG001
    """Drop the published platform so it cannot leak into the next item."""
    set_current_item_platform(None)


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
            item_filter = _marker_platforms(item_marker, "platforms", item.name)
        else:
            item_filter = canonical
        allowed = cli_filter & item_filter

        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec else {}
        platform_param = params.get("platform") or params.get("_st_platform")

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


class _Unresolvable(Exception):
    """A constructor-argument AST node that cannot be resolved statically."""


def _eval_arg_node(
    node: ast.AST, params: dict[str, Any], localns: dict[str, Any], globalns: dict[str, Any]
) -> Any:
    """Resolve a constructor-argument AST node to a Python value.

    Handles the shapes test bodies actually use to build a ``PTOTestCase``:
    literals, parametrize names, local variables assigned earlier in the body,
    module globals, attribute/enum access (``DataType.FP16``), tuples/lists,
    negated numbers, and *calls* (``RunConfig(...)``, ``_cfg()``) — the call is
    re-evaluated exactly as the body would.  Name lookup order is
    params → locals → globals, mirroring how the body would evaluate it.
    Anything else (arithmetic on non-constants, ``**kwargs``) raises
    :class:`_Unresolvable` so the case falls back to the inline path rather than
    being mis-reconstructed.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in params:
            return params[node.id]
        if node.id in localns:
            return localns[node.id]
        if node.id in globalns:
            return globalns[node.id]
        raise _Unresolvable(node.id)
    if isinstance(node, ast.Attribute):
        return getattr(_eval_arg_node(node.value, params, localns, globalns), node.attr)
    if isinstance(node, (ast.Tuple, ast.List)):
        elts = [_eval_arg_node(e, params, localns, globalns) for e in node.elts]
        return tuple(elts) if isinstance(node, ast.Tuple) else elts
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_arg_node(node.operand, params, localns, globalns)
    if isinstance(node, ast.Call):
        fn = _eval_arg_node(node.func, params, localns, globalns)
        a = [_eval_arg_node(x, params, localns, globalns) for x in node.args]
        kw = {k.arg: _eval_arg_node(k.value, params, localns, globalns) for k in node.keywords if k.arg}
        if any(k.arg is None for k in node.keywords):
            raise _Unresolvable("**kwargs")
        return fn(*a, **kw)
    raise _Unresolvable(ast.dump(node))


def _collect_test_case_from_item(
    item: pytest.Item,
    seen: dict[str, PTOTestCase],
    session_memory_planner: MemoryPlanner | None,
    session_platform: str,
) -> None:
    """Inspect *item* and add any discovered PTOTestCase instances to *seen*.

    Parses the test body and resolves every ``SomeCase(...)`` constructor call
    (callee + all positional/keyword args) against this item's parametrize
    params, locals assigned earlier in the body, and the test module globals,
    then instantiates it exactly as the body would.  This reconstructs the case
    regardless of parametrize→__init__ name renames (``valid`` →
    ``valid_shape``), hard-coded literal args (``dtype=DataType.FP16``),
    positional args, the class-as-parameter pattern (``op_cls(...)``), or a
    locally-built config (``cfg = RunConfig(...); run(Case(config=cfg))``) — so
    the case is pre-compiled and batched instead of falling to the per-case
    inline path.  Cases whose args genuinely can't be resolved (built in a loop,
    arithmetic on params) are left for the inline path.
    """
    if any(m.name == "skip" for m in item.iter_markers()):
        return

    module = item.module
    if module is None:
        return
    globalns = vars(module)

    callspec = getattr(item, "callspec", None)
    params: dict[str, Any] = callspec.params if callspec else {}

    try:
        source = textwrap.dedent(inspect.getsource(item.function))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return

    # Build a local namespace from simple ``name = <expr>`` assignments, in
    # source order, so a constructor arg referencing a local (``config=cfg``)
    # resolves. Unresolvable assignments (e.g. ``result = test_runner.run(...)``)
    # are skipped — the fixture call can't and shouldn't be evaluated here.
    localns: dict[str, Any] = {}
    assigns = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Assign)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    for stmt in assigns:
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            try:
                localns[stmt.targets[0].id] = _eval_arg_node(stmt.value, params, localns, globalns)
            except Exception:  # noqa: BLE001 — best-effort; unresolved locals just stay unknown
                continue

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # The callee must resolve to a concrete PTOTestCase subclass. Any
        # evaluation failure (unresolvable name, an attribute/call on a fixture
        # or other non-constructor callee, a side-effecting helper that raises)
        # just means "not a discoverable case here" — skip, don't crash
        # collection.
        try:
            func = _eval_arg_node(node.func, params, localns, globalns)
        except Exception:  # noqa: BLE001 — best-effort discovery; never abort collection
            continue
        if not (isinstance(func, type) and issubclass(func, PTOTestCase) and func is not PTOTestCase):
            continue
        # Resolve every arg exactly as the body passes them (positional + kw).
        try:
            if any(kw.arg is None for kw in node.keywords):
                raise _Unresolvable("**kwargs")
            args = [_eval_arg_node(a, params, localns, globalns) for a in node.args]
            kwargs = {kw.arg: _eval_arg_node(kw.value, params, localns, globalns) for kw in node.keywords}
            instance = func(*args, **kwargs)
        except Exception:
            # _Unresolvable arg, or a constructor mismatch — leave for inline.
            continue
        # Bind the item's platform so each matrix variant compiles its own
        # artefact; without this the variants share one cache key and only the
        # first platform is ever built. An item the matrix did not expand runs
        # on the session platform, which is what the pipeline resolves for it.
        platform = params.get("platform") or params.get("_st_platform") or session_platform
        instance.bind_platform(platform)
        seen.setdefault(_cache_key(instance, platform, session_memory_planner), instance)


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
    session_memory_planner = _parse_memory_planner(session.config.getoption("--memory-planner"))
    platform_filter = _parse_platform_filter(session.config.getoption("--platform"))
    session_platform: str = platform_filter[0] if platform_filter else "a2a3"
    seen: dict[str, PTOTestCase] = {}  # effective cache_key → instance (deduped)

    for item in session.items:
        _collect_test_case_from_item(item, seen, session_memory_planner, session_platform)

    # Read the task-submit / pipeline options *before* the empty-discovery guard:
    # a suite that only creates PTOTestCases dynamically leaves ``seen`` empty yet
    # still runs each case through ``TestRunner._run_inline()``, which must be
    # routed through task-submit too. Bailing on ``not seen`` before this would
    # silently ignore ``--execute-via-task-submit`` for that inline path.
    execute_via_task_submit: bool = session.config.getoption("--execute-via-task-submit")
    task_max_time: int = session.config.getoption("--task-max-time")
    task_queue_timeout: int = session.config.getoption("--task-queue-timeout")
    task_submit_device: str = (session.config.getoption("--task-submit-device") or "").strip()
    execute_batch_size: int = session.config.getoption("--execute-batch-size")

    # Guard against a silently-wrong card. ``--task-submit-device="$DEVICE_RANGE"``
    # with an *unset* DEVICE_RANGE (e.g. the var didn't propagate to a de-dockered
    # runner) collapses to an empty string, which would make ``task-submit
    # --device ""`` fall back to auto-allocation — borrowing a host-free card the
    # runner may not own (→ halMemCtl EACCES). Fail loudly instead.
    if execute_via_task_submit and not task_submit_device:
        raise pytest.UsageError(
            "--task-submit-device is empty (an unset $DEVICE_RANGE on a de-dockered runner?). "
            "Pass 'auto' to borrow any free card, or a specific id/range (e.g. \"$DEVICE_RANGE\") "
            "to pin. Refusing to fall back to task-submit's default card silently."
        )

    max_workers: int | None = session.config.getoption("--precompile-workers")
    # Without --precompile-workers the pipeline is skipped entirely; each
    # test compiles + executes inline inside TestRunner._run_inline().  When
    # task-submit is still requested, route that inline execute through it too
    # (the no-pipeline + task-submit matrix cell). This applies whether or not any
    # case was *statically* discovered — dynamically-created cases still run
    # inline — so it must precede the ``not seen`` return below.
    if max_workers is None:
        if execute_via_task_submit:
            configure_inline_task_submit(
                task_max_time=task_max_time,
                task_queue_timeout=task_queue_timeout,
                task_submit_device=task_submit_device,
            )
        return

    # The pre-compile pipeline only has work when cases were statically
    # discovered; undiscovered suites fall back to the inline path. Those
    # dynamically-created cases still run through TestRunner._run_inline(), which
    # must borrow a card via task-submit too — so wire the inline path before
    # returning, exactly as the ``max_workers is None`` branch does above.
    # Otherwise a --precompile-workers + --execute-via-task-submit run with only
    # dynamic cases would execute them in-process on a card-free host.
    if not seen:
        if execute_via_task_submit:
            configure_inline_task_submit(
                task_max_time=task_max_time,
                task_queue_timeout=task_queue_timeout,
                task_submit_device=task_submit_device,
            )
        return

    dump_passes: bool = session.config.getoption("--dump-passes")
    codegen_only: bool = session.config.getoption("--codegen-only")
    enable_chip_swimlane: int = _resolve_swimlane_option(session.config)
    enable_dump_args: int = session.config.getoption("--dump-args")
    enable_pmu: int = session.config.getoption("--enable-pmu")
    enable_dep_gen: bool = session.config.getoption("--enable-dep-gen")
    enable_scope_stats: bool = session.config.getoption("--enable-scope-stats")
    analyze_auto_scopes_for_deps: bool = session.config.getoption("--analyze-auto-scopes-for-deps")

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
    elif execute_via_task_submit:
        # task-submit runs the execute child as a host-level process that must
        # read this (parent-written) cache_dir.  Put it under the repo's
        # build_output — a real host path inside the checkout (reclaimed by the
        # next run's ``actions/checkout`` clean) — rather than /tmp, to avoid
        # private-tmp / cross-user visibility surprises.  Still a temp dir:
        # removed at session end (pytest_sessionfinish), so CI disk does not grow.
        base = _PROJECT_ROOT / "build_output"
        base.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(tempfile.mkdtemp(prefix="pypto_precompile_", dir=str(base)))
        _temp_precompile_dirs.append(cache_dir)
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="pypto_precompile_"))
        _temp_precompile_dirs.append(cache_dir)

    # Build the device pool from --device.  N parallel executes max.
    devices = _parse_device_option(session.config.getoption("--device"))
    device_pool: queue.Queue[int] = queue.Queue()
    for d in devices:
        device_pool.put(d)

    execute_mode = "task-submit" if execute_via_task_submit else "device-pool"
    test_cases = list(seen.values())
    # In task-submit mode the device runs borrow `task_submit_device` via
    # task-submit; the local `--device` pool is unused (only in-process sim would
    # touch it). Show only the device that actually executes, to avoid implying
    # two different cards are in play.
    device_info = (
        f"task_submit_device={task_submit_device}" if execute_via_task_submit else f"devices={devices}"
    )
    print(
        f"\n[PyPTO] Pipeline: {len(test_cases)} test case(s); "
        f"compile_workers={max_workers}, execute_mode={execute_mode}, {device_info}"
    )
    pypto_log_level = LogLevel[session.config.getoption("--pypto-log-level")]
    _set_thread_log_level(pypto_log_level)
    try:
        start_pipeline(
            test_cases=test_cases,
            cache_dir=cache_dir,
            session_platform=session_platform,
            dump_passes=dump_passes,
            codegen_only=codegen_only,
            pypto_log_level=pypto_log_level,
            compile_workers=max_workers,
            device_pool=device_pool,
            enable_chip_swimlane=enable_chip_swimlane,
            enable_dump_args=enable_dump_args,
            enable_pmu=enable_pmu,
            enable_dep_gen=enable_dep_gen,
            enable_scope_stats=enable_scope_stats,
            analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
            execute_mode=execute_mode,
            task_max_time=task_max_time,
            task_queue_timeout=task_queue_timeout,
            task_submit_device=task_submit_device,
            execute_batch_size=execute_batch_size,
            memory_planner=session_memory_planner,
        )
    finally:
        _clear_thread_log_level()
    print("[PyPTO] Pipeline scheduled — pytest item loop starting\n")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Tear down the pipeline and clean up temporary precompile directories."""
    shutdown_pipeline()
    for d in _temp_precompile_dirs:
        shutil.rmtree(d, ignore_errors=True)
