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
import random
import shutil
import sys
import tempfile
import textwrap
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
        help="PyPTO runtime log level (debug, v0..v9, info, warn, error, null). "
        "Default: leave the runtime logger at its V5/INFO default.",
    )
    parser.addoption(
        "--pto-isa-commit",
        action="store",
        default=None,
        help="Pin the pto-isa clone to a specific git commit (hash or tag). Default: use latest remote HEAD.",
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
    # Each maps 1:1 to the same-named field on ``RunConfig`` and to the
    # corresponding ``CallConfig`` member on the runtime side. Names match
    # ``runtime/conftest.py`` so the two surfaces stay aligned.
    parser.addoption(
        "--enable-l2-swimlane",
        action="store_true",
        default=False,
        help="Capture per-task L2 perf records into <work_dir>/dfx_outputs/l2_swimlane_records.json. "
        "On onboard platforms, also render merged_swimlane_*.json and run the kernel twice: a dep_gen "
        "pass to capture deps.json (the converter's task graph) then a clean swimlane pass, since "
        "dep_gen collection perturbs the timing. Simulator platforms emit only the records (the merged "
        "swimlane is skipped).",
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
        pto_isa_commit=request.config.getoption("--pto-isa-commit"),
        enable_l2_swimlane=request.config.getoption("--enable-l2-swimlane"),
        enable_dump_args=request.config.getoption("--dump-args"),
        enable_pmu=request.config.getoption("--enable-pmu"),
        enable_dep_gen=request.config.getoption("--enable-dep-gen"),
        enable_scope_stats=request.config.getoption("--enable-scope-stats"),
        analyze_auto_scopes_for_deps=request.config.getoption("--analyze-auto-scopes-for-deps"),
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


def _collect_test_case_from_item(item: pytest.Item, seen: dict[str, PTOTestCase]) -> None:
    """Inspect *item* and add any discovered PTOTestCase instances to *seen*.

    Parses the test body and resolves every ``SomeCase(...)`` constructor call
    (callee + all positional/keyword args) against this item's parametrize
    params, locals assigned earlier in the body, and the test module globals,
    then instantiates it exactly as the body would.  This reconstructs the case
    regardless of parametrize→__init__ name renames (``valid`` →
    ``valid_shapes``), hard-coded literal args (``dtype=DataType.FP16``),
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
        seen.setdefault(_cache_key(instance), instance)


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
    pto_isa_commit: str | None = session.config.getoption("--pto-isa-commit")
    enable_l2_swimlane: bool = session.config.getoption("--enable-l2-swimlane")
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

    # ``--platform`` is a CSV allowlist; the per-test value resolved by
    # ``tc.get_platform()`` overrides this fallback inside the pipeline.
    platform_filter = _parse_platform_filter(session.config.getoption("--platform"))
    session_platform: str = platform_filter[0] if platform_filter else "a2a3"

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
    )
    print("[PyPTO] Pipeline scheduled — pytest item loop starting\n")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Tear down the pipeline and clean up temporary precompile directories."""
    shutdown_pipeline()
    for d in _temp_precompile_dirs:
        shutil.rmtree(d, ignore_errors=True)
