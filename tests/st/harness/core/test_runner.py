# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test runner for executing PTO test cases.

Orchestrates the full test execution pipeline:
1. Get program from test case (@pl.program or IRBuilder)
2. Generate kernel and orchestration code via PyPTO ir.compile()
3. Generate golden.py
4. Execute via simpler's CodeRunner
5. Validate results
"""

import json
import logging
import math
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.runtime import compile_program
from pypto.runtime.golden_writer import (
    _data_dir_has_files,
    _extract_compute_golden,
    _materialize_tensors,
    _save_data_files,
    generate_golden_source,
)
from pypto.runtime.runner import (
    RunConfig,
    RunResult,
    _DfxOpts,
    _execute_on_device,
    validate_persisted_outputs,
)
from pypto.runtime.tensor_spec import TensorSpec as RuntimeTensorSpec

from harness.core.harness import PTOTestCase

# tests/st/harness/core/test_runner.py -> tests/st/ -> project root
_ST_DIR = Path(__file__).parent.parent.parent
_PROJECT_ROOT = _ST_DIR.parent.parent
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline state (compile pool → device pool)
# ---------------------------------------------------------------------------
#
# Replaces the old "compile-everything then run-everything" two-phase model.
# A compile pool of ``--precompile-workers`` threads fuses IR compile + golden
# generation + .so build into one task per test case.  As each compile future
# completes, an execute pool (sized to the number of devices in --device)
# picks the case up and dispatches it onto the next free device from the
# DevicePool.  Pytest's per-item loop then calls ``TestRunner.run`` which
# blocks on the matching execute future and returns the cached RunResult.

# cache_key → Future[CompileArtifact] populated in start_pipeline().
# TestRunner.run blocks on the compile future, rewrites golden.py with the
# real RunConfig, then submits the execute task itself.  This keeps execute
# work synchronised with pytest's per-item lifecycle so (a) RunConfig
# tolerances reach golden.py and (b) C++ stdout from execute lands in the
# right test's capture window.
_compile_futures: dict[str, Future] = {}

# cache_key → (batch Future, work_dir) for the task-submit BATCH device run.
# In task-submit mode the device runs are NOT submitted one task-submit task per
# artifact (that pays the torch/pypto import + NPU init per artifact — a huge
# cold-start cost).  Instead a background submitter groups every compiled
# artifact into batches of ``--execute-batch-size`` and submits ONE task-submit
# task per batch; that task runs the whole batch in a single hot process,
# reusing one ChipWorker device session (execute_artifact --batch-manifest).
# The batch Future resolves to ``{str(work_dir): (ok, error, device)}``;
# TestRunner.run awaits it, picks out this case, and validates the persisted
# outputs with the test's real tolerance.  ``_batches_ready`` is set once the
# submitter has assigned every case to a batch.
_case_to_batch: dict[str, "tuple[Future, Path]"] = {}
_batches_ready = threading.Event()

# Per-batch device-run stats for the end-of-session summary: one entry per
# submitted batch, ``(batch_name, n_ok, n_total, device)``. Appended from the
# execute-pool thread in _run_batch_via_task_submit (list.append is GIL-atomic).
# Surfaced by execution_summary_lines() in pytest_terminal_summary so the user
# can see batching happened and how the runs spread across the borrowed cards
# (otherwise invisible — the per-artifact markers are suppressed for a clean log).
_batch_stats: "list[tuple[str, int, int, int | None]]" = []

# cache_key → device id actually used by the execute task.  Read by
# TestRunner.run after exec_fut.result() and forwarded to the _report_device
# fixture via _last_device.
_executed_device: dict[str, int] = {}

# Single-slot stash of the device id the most-recently-resolved test ran on.
# pytest's item loop is single-threaded, so one slot is enough: TestRunner.run
# writes, _report_device fixture reads.
_last_device: dict[str, int | None] = {"value": None}

# Session-scoped pipeline resources, set up by start_pipeline() and torn down
# by shutdown_pipeline() from pytest_sessionfinish.
_device_pool: "queue.Queue[int] | None" = None
_execute_pool: ThreadPoolExecutor | None = None
_compile_pools: list[ThreadPoolExecutor] = []
_pipeline_ctx: dict = {}

# Upper bound on the task-submit execute pool size.  In task-submit mode every
# case's device run is submitted at once (task-submit schedules them), so the
# pool tracks the case count; this caps the thread / task-submit-client count on
# very large suites (the excess simply queues in the pool, then task-submit).
_MAX_TASK_SUBMIT_INFLIGHT = 512

# set_backend_type is called once per backend-type group before the thread pool
# starts.  Only get_program() needs serialisation because the @pl.program
# decorator is not thread-safe; compile_program() writes to isolated dirs and
# runs concurrently.
_get_program_lock = threading.Lock()

# Map BackendType to the architecture prefix used by the platform string.
# "a2a3" covers Ascend 910B; "a5" covers Ascend 950.
_BACKEND_TO_ARCH: dict[BackendType, str] = {
    BackendType.Ascend910B: "a2a3",
    BackendType.Ascend950: "a5",
}


def _cache_key(tc: PTOTestCase, resolved_platform: str | None = None) -> str:
    """Return a unique cache key combining test name and target platform.

    The cache key is anchored to the *resolved* platform so that the
    pre-compilation cache, the binary cache and the executor all agree on
    which toolchain a given artifact was produced for. Resolution order:

    1. ``resolved_platform`` (the value returned by :func:`_resolve_platform`
       for the current session). Callers should pass it whenever they have it
       so a legacy test case run with ``--platform=a2a3sim`` is keyed to
       ``a2a3sim`` rather than the backend-derived ``a2a3``.
    2. ``tc.get_platform()`` for parametrized cases that pinned a platform on
       the test case itself.
    3. The backend architecture (``a2a3``/``a5``) as a final fallback for
       cases that neither set a platform nor receive a resolved one.
    """
    if not resolved_platform:
        try:
            resolved_platform = tc.get_platform()
        except AttributeError:
            resolved_platform = None
    if not resolved_platform:
        resolved_platform = _BACKEND_TO_ARCH.get(tc.get_backend_type(), "unknown")
    return f"{tc.get_name()}@{resolved_platform}"


def _resolve_platform(config_platform: str, test_case: PTOTestCase | None = None) -> str:
    """Return the platform string used to compile/execute *test_case*.

    The test-case-level platform (set via the ``platform`` constructor arg or
    overridden in :py:meth:`PTOTestCase.get_platform`) takes precedence over
    the session-wide ``--platform`` value.  When *test_case* is ``None`` the
    function preserves the historical behaviour of returning ``config_platform``
    so legacy code paths still work.
    """
    if test_case is not None:
        try:
            tc_platform = test_case.get_platform()
        except AttributeError:
            tc_platform = None
        if tc_platform:
            return tc_platform
    return config_platform


def _default_work_dir(test_name: str) -> Path:
    """Return the default output path for a saved test: build_output/{testName}_{timestamp}."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _PROJECT_ROOT / "build_output" / f"{test_name}_{timestamp}"


def _inline_work_dir(config: RunConfig, test_name: str, resolved_platform: str) -> "tuple[Path, bool]":
    """Pick the ``_run_inline`` build directory. Returns ``(work_dir, is_temp)``.

    - ``--save-kernels``: a persistent, named directory (never cleaned up here).
    - otherwise a temp dir. When the case will borrow a card via task-submit, the
      device step runs in a separate host process (via runuser) that must READ
      work_dir, and a private-/tmp ``mkdtemp`` isn't guaranteed cross-user
      visible — so mirror the pipeline path and place it under the repo-local
      ``build_output``. A plain /tmp dir is fine for the in-process device run.
    """
    if config.save_kernels:
        if config.save_kernels_dir:
            work_dir = Path(config.save_kernels_dir) / test_name
        else:
            work_dir = _default_work_dir(test_name)
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir, False
    base_dir: str | None = None
    if _pipeline_ctx.get("inline_via_task_submit") and not resolved_platform.endswith("sim"):
        base = _PROJECT_ROOT / "build_output"
        base.mkdir(parents=True, exist_ok=True)
        base_dir = str(base)
    return Path(tempfile.mkdtemp(prefix=f"pypto_test_{test_name}_", dir=base_dir)), True


def _write_golden_for_test_case(test_case: PTOTestCase, output_path: Path) -> None:
    """Generate and write golden.py for *test_case*.

    Converts harness TensorSpec (DataType) to runtime TensorSpec (torch.dtype),
    extracts compute_golden from the compute_expected method, and writes golden.py.

    Args:
        test_case: The PTOTestCase to generate golden for.
        output_path: Destination path for the generated golden.py.
    """
    runtime_specs = [
        RuntimeTensorSpec(
            name=spec.name,
            shape=spec.shape,
            dtype=spec.dtype.torch_dtype,
            init_value=spec.init_value,
            is_output=spec.is_output,
        )
        for spec in test_case.tensor_specs
    ]

    try:
        compute_golden_src = _extract_compute_golden(test_case.compute_expected)
    except RuntimeError:
        output_specs = [s for s in test_case.tensor_specs if s.is_output]
        lines = [
            "def compute_golden(tensors, params):",
            '    """Compute expected outputs - PLACEHOLDER."""',
            "    # TODO: Could not extract compute_expected source.",
            "    # Please implement the expected computation here.",
        ]
        for spec in output_specs:
            lines.append(f'    # tensors["{spec.name}"][:] = ...')
        lines.append("")
        lines.append('    raise NotImplementedError("compute_expected source extraction failed")')
        compute_golden_src = "\n".join(lines)

    data_dir = output_path.parent / "data"
    if not _data_dir_has_files(data_dir, runtime_specs):
        data = _materialize_tensors(runtime_specs)
        in_data = {s.name: data[s.name] for s in runtime_specs if not s.is_output or s.init_value is not None}
        _save_data_files(in_data, data_dir / "in")

        # Compute golden outputs and save to data/out/
        test_case.compute_expected(data)
        out_data = {s.name: data[s.name] for s in runtime_specs if s.is_output}
        _save_data_files(out_data, data_dir / "out")

    write_golden_src = generate_golden_source(
        runtime_specs,
        None,
        test_case.config.rtol,
        test_case.config.atol,
        compute_golden_src=compute_golden_src,
        scalar_specs=test_case.scalar_specs or None,
        use_data_files=True,
    )
    output_path.write_text(write_golden_src, encoding="utf-8")


# ---------------------------------------------------------------------------
# Pre-compilation helpers
# ---------------------------------------------------------------------------


def _compile_for_cache(
    test_case: "PTOTestCase",
    work_dir: Path,
    dump_passes: bool,
    analyze_auto_scopes_for_deps: bool,
) -> None:
    """Compile one test case into *work_dir* (called from thread pool).

    The backend type MUST already be set by the caller before entering the pool.
    Only ``get_program`` is serialised (via ``_get_program_lock``) because the
    ``@pl.program`` decorator is not thread-safe; ``compile_program`` writes to
    an isolated directory and runs concurrently.
    """
    backend_type = test_case.get_backend_type()
    with _get_program_lock:
        program = test_case.get_program()
    if program is None:
        raise ValueError(
            f"Test case {test_case.get_name()} must implement get_program() "
            "to return a @pl.program class or ir.Program"
        )
    compile_program(
        program,
        work_dir,
        strategy=test_case.get_strategy(),
        backend_type=backend_type,
        dump_passes=dump_passes,
        analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
        memory_planner=test_case.get_memory_planner(),
    )
    if not list((work_dir / "kernels").rglob("*.cpp")):
        raise ValueError(f"No kernels generated for {test_case.get_name()}")
    if not list((work_dir / "orchestration").glob("*.cpp")):
        raise ValueError(
            f"No orchestration generated for {test_case.get_name()}. "
            "Ensure your @pl.program includes an orchestration function "
            "(decorated with @pl.function(type=pl.FunctionType.Orchestration))."
        )
    _write_golden_for_test_case(test_case, work_dir / "golden.py")


@dataclass
class CompileArtifact:
    """Outcome of a fused compile task (IR → C++ → golden.py → .so).

    Stored as the result of a compile-pool future and consumed by the matching
    execute-pool task via ``_fused_execute_task``.
    """

    work_dir: Path
    resolved_platform: str
    error: str | None = None
    runtime_name: str | None = None
    chip_callable: Any | None = None


def _fused_compile_task(
    tc: "PTOTestCase",
    cache_dir: Path,
    session_platform: str,
    dump_passes: bool,
    analyze_auto_scopes_for_deps: bool,
    pto_isa_commit: str | None,
) -> CompileArtifact:
    """Compile IR → kernels/orch C++ → golden.py → .so for one test case.

    Runs on a compile-pool worker thread.  ``get_program`` is serialised via
    ``_get_program_lock`` inside ``_compile_for_cache``; everything else runs
    concurrently with other compile-pool workers.  The backend type must
    already be set on the main thread before this task is submitted.
    """
    resolved = _resolve_platform(session_platform, tc)
    work_dir = cache_dir / _cache_key(tc, resolved)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        _compile_for_cache(tc, work_dir, dump_passes, analyze_auto_scopes_for_deps)
        # Codegen-only runs skip assembly: the .so is never loaded by the
        # execute task (see _fused_execute_task) and assembling here would
        # both waste work and race on PTO_ISA_ROOT (start_pipeline skips
        # the pre-resolve under codegen_only).
        if _pipeline_ctx.get("codegen_only"):
            return CompileArtifact(
                work_dir=work_dir,
                resolved_platform=resolved,
                error=None,
            )
        from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

        chip_callable, runtime_name, _ = compile_and_assemble(
            work_dir, resolved, pto_isa_commit=pto_isa_commit
        )
        return CompileArtifact(
            work_dir=work_dir,
            resolved_platform=resolved,
            error=None,
            runtime_name=runtime_name,
            chip_callable=chip_callable,
        )
    except Exception as exc:
        return CompileArtifact(
            work_dir=work_dir,
            resolved_platform=resolved,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# ---------------------------------------------------------------------------
# task-submit execute path (opt-in via --execute-via-task-submit)
# ---------------------------------------------------------------------------
#
# In this mode the execute half borrows an NPU per case from the host-level
# ``task-submit --device auto`` queue instead of a local DevicePool, so the CI
# job itself holds no card during compile + golden.  The compiled .o/.so already
# live in ``work_dir`` (a host-visible path); the child re-binds them with no
# recompile.  Everything below is reached ONLY when execute_mode == "task-submit"
# — the default device-pool path never imports or calls task-submit, so machines
# without the binary are unaffected.

# Sentinel emitted by pypto.runtime.execute_artifact; lets us tell a genuine
# case failure apart from an infra kill (queue timeout / watchdog / missing
# binary).  Keep in sync with execute_artifact._RESULT_PREFIX.
_RESULT_MARKER_PREFIX = "PYPTO_EXEC_RESULT"


def _dfx_to_cli(dfx: "_DfxOpts") -> list[str]:
    """Inverse of ``execute_artifact`` DFX arg parsing.

    Emits only the flags whose value differs from the off-default, so a plain
    run yields an empty list.  Keep in sync with
    ``pypto.runtime.execute_artifact._build_parser``.
    """
    argv: list[str] = []
    if dfx.enable_l2_swimlane:
        argv.append("--enable-l2-swimlane")
    if dfx.enable_dump_tensor:
        argv += ["--dump-tensor", str(dfx.enable_dump_tensor)]
    if dfx.enable_pmu:
        argv += ["--enable-pmu", str(dfx.enable_pmu)]
    if dfx.enable_dep_gen:
        argv.append("--enable-dep-gen")
    if dfx.enable_scope_stats:
        argv.append("--enable-scope-stats")
    return argv


def _marker_value(line: str, key: str) -> str | None:
    """Return the value of a ``key=value`` token in a PYPTO_EXEC_RESULT line.

    e.g. ``_marker_value("...=PASS work_dir=/x device=3", "device=") == "3"``.
    Values never contain spaces (paths in CI, integer ids), so a token split is
    sufficient.
    """
    for tok in line.split():
        if tok.startswith(key):
            return tok[len(key) :]
    return None


def _parse_executed_device(stdout: str) -> int | None:
    """Pull the real device id out of execute_artifact's PASS marker."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(f"{_RESULT_MARKER_PREFIX}=PASS"):
            for tok in line.split():
                if tok.startswith("device="):
                    try:
                        return int(tok.split("=", 1)[1])
                    except ValueError:
                        return None
    return None


def _classify_task_submit_failure(returncode: int, stdout: str, stderr: str) -> str:
    """Build a ``RunResult.error`` distinguishing a real failure from infra.

    ``task-submit --run`` propagates the inner exit code verbatim, so the code +
    the presence/absence of the PASS/FAIL marker pins down what went wrong.
    """
    has_fail = f"{_RESULT_MARKER_PREFIX}=FAIL" in stdout
    has_pass = f"{_RESULT_MARKER_PREFIX}=PASS" in stdout
    has_infra = f"{_RESULT_MARKER_PREFIX}=INFRA" in stdout
    streams = f"---- stdout ----\n{stdout}\n---- stderr ----\n{stderr}"
    if has_infra:
        # Pre-run reconstruction/setup failure (stale/missing cached .o/.so, bad
        # --work-dir, corrupt artifact) — an infra problem, not a device test
        # failure. Kept distinct from FAIL so a cache/setup miss isn't misread
        # as a real numerical regression.
        return f"Artifact reconstruction/setup failed before the device run (infra, not a test).\n{streams}"
    if returncode == 0 and not has_pass:
        return f"task-submit returned 0 without a PASS marker — suspected scheduler anomaly.\n{streams}"
    if returncode == 1 and has_fail:
        return f"Test failed on device.\n{streams}"
    if returncode == 1 and not stdout.strip() and not stderr.strip():
        # No child output at all: task-submit never got a card within --timeout.
        return (
            "Borrowed-card queue wait timed out (task-submit --timeout); the task may still "
            f"be running. Raise --task-queue-timeout.\n{streams}"
        )
    if returncode == 1:
        # The child ran and printed something but emitted no PASS/FAIL/INFRA
        # marker — e.g. a bootstrap/import crash before execute_artifact's own
        # try/except. Don't mislabel it as a queue timeout.
        return f"execute_artifact exited 1 without a result marker (child bootstrap failure?).\n{streams}"
    if returncode in (137, 143):
        return (
            "Execution killed by task-submit watchdog (--max-time exceeded); investigate a "
            f"hang or raise --task-max-time.\n{streams}"
        )
    return f"task-submit rc={returncode}.\n{streams}"


def _shell_quote_run(project_root: Path, inner: "list[str]") -> str:
    """Build the ``task-submit --run`` shell payload, quoting every token.

    ``subprocess.run`` invokes ``task-submit`` with an argv list, but the
    ``--run`` value is a *shell* command line, so spaces / metacharacters in the
    work_dir, platform, or commit tokens would otherwise alter it. Quote the
    project root and each ``inner`` token with ``shlex.quote`` — except the
    literal ``$TASK_DEVICE`` placeholder, which must stay unquoted so the shell
    ``task-submit`` runs the command in expands it.

    The payload re-derives the Ascend/CANN environment on the device host via
    ``source activate.sh`` (which itself sources ``set_env.sh``) instead of
    relying on ``task-submit`` snapshotting the caller's exported environment.
    The snapshot path (``env -0`` -> ``pending/<task_id>.env`` -> daemon read)
    can truncate under the 32-worker precompile submit concurrency, dropping
    e.g. ``ASCEND_HOME_PATH`` for a single batch and failing every artifact in
    it with an ``(infra)`` ArtifactSetupError. Sourcing ``activate.sh`` here
    makes the child self-sufficient — the same pattern the mechanism-B
    ``--run`` payloads already use — so a partial snapshot can no longer strand
    a batch. ``activate.sh`` lives at *project_root*, so the ``cd`` precedes it.

    Sourcing is guarded on the file's existence (``[ ! -f activate.sh ] ||
    source activate.sh``): in CI ``activate.sh`` is always present so the env is
    re-derived, but a developer invoking ``--execute-via-task-submit`` from a
    shell without the CI bootstrap falls back to their already-configured
    environment (the pre-existing behavior) instead of failing on a missing
    file. A present-but-broken ``activate.sh`` still surfaces its error.
    """
    quoted = " ".join("$TASK_DEVICE" if tok == "$TASK_DEVICE" else shlex.quote(tok) for tok in inner)
    return (
        f"cd {shlex.quote(str(project_root))} "
        f"&& {{ [ ! -f activate.sh ] || source activate.sh; }} && {quoted}"
    )


def _build_execute_artifact_cmd(
    mode_args: "list[str]", dfx: "_DfxOpts", pto_isa_commit: str | None
) -> list[str]:
    """Assemble the ``python -m pypto.runtime.execute_artifact`` child argv.

    *mode_args* selects single (``--work-dir`` / ``--platform``) vs batch
    (``--batch-manifest``); everything else — the interpreter, ``--device-id
    $TASK_DEVICE``, ``--no-validate``, the DFX flags and the optional pto-isa pin —
    is shared by both task-submit paths.

    Uses the exact interpreter running the harness (the per-job venv python), not
    bare ``"python"``: task-submit runs the child via ``runuser`` and bare
    ``"python"`` would resolve off PATH to the conda base python, which need not
    have pypto.
    """
    inner = [
        sys.executable,
        "-m",
        "pypto.runtime.execute_artifact",
        *mode_args,
        "--device-id",
        "$TASK_DEVICE",  # expanded by the shell task-submit runs the command in
        # Device run only — persist actual outputs, no allclose. The harness
        # validates with the per-test tolerance after this returns, so the run is
        # tolerance-independent and can be submitted eagerly.
        "--no-validate",
        *_dfx_to_cli(dfx),
    ]
    if pto_isa_commit:
        inner += ["--pto-isa-commit", pto_isa_commit]
    return inner


def _exec_task_submit(
    inner: "list[str]", device: str, max_time: int, queue_timeout: int, log_path: Path
) -> "tuple[subprocess.CompletedProcess | None, str | None]":
    """Run one ``task-submit --run <inner>`` and persist the child output.

    Returns ``(proc, None)`` on a successful exec, or ``(None, error)`` when
    ``task-submit`` itself could not be launched (``OSError`` — missing binary /
    not executable). Never raises. On success the full stdout+stderr is written to
    *log_path* for post-mortem (a write failure is swallowed — best-effort only).

    No ``--env`` / ``--ptoas`` flags are passed: the ``--run`` payload sources
    ``activate.sh`` to re-derive the Ascend/CANN environment on the device host
    (see ``_shell_quote_run``), so the minimal CI ``task-submit`` (which lacks
    those options) works — matching the mechanism-B ``--run`` payloads and the
    ``daily_ci`` a5 job. ``task-submit`` still preserves the caller's exported
    environment via ``runuser`` (PTO_ISA_ROOT / PTOAS_ROOT / PYTHONPATH /
    PTO2_RING_* reach the child that way), but the CANN vars no longer depend on
    that snapshot surviving the high-concurrency submit path.
    """
    argv = [
        "task-submit",
        "--device",
        device,
        "--timeout",
        str(queue_timeout),
        "--max-time",
        str(max_time),
        "--run",
        _shell_quote_run(_PROJECT_ROOT, inner),
    ]
    try:
        proc = subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603
    except OSError as exc:
        return None, f"Failed to exec task-submit ({exc}) — do not pass --execute-via-task-submit here."
    # Persist the full child output for post-mortem. Do NOT reprint it: the device
    # chatter / markers would drown pytest's own per-test output. Failures surface
    # their detail via the returned error instead.
    combined = f"---- stdout ----\n{proc.stdout}\n---- stderr ----\n{proc.stderr}\n"
    try:
        log_path.write_text(combined, encoding="utf-8")
    except OSError:
        pass
    return proc, None


def _run_artifact_via_task_submit(
    work_dir: Path,
    platform: str,
    dfx: "_DfxOpts",
    pto_isa_commit: str | None,
    max_time: int,
    queue_timeout: int,
    device: str = "auto",
) -> tuple[bool, str | None, int | None]:
    """Execute one compiled artifact through ``task-submit``.

    Returns ``(passed, error, device_id)``.  The child re-binds the cached
    .o/.so in *work_dir* (no recompile, no card) and runs on the NPU task-submit
    lends it via ``$TASK_DEVICE``.  Blocks until the child exits.

    *device* is the ``task-submit --device`` value: ``"auto"`` (borrow any free
    card) or a specific id / range (pin to it — used to validate the flow before
    trusting auto-allocation).
    """
    inner = _build_execute_artifact_cmd(
        ["--work-dir", str(work_dir), "--platform", platform], dfx, pto_isa_commit
    )
    proc, exec_err = _exec_task_submit(inner, device, max_time, queue_timeout, work_dir / "execute.log")
    if proc is None:
        return (False, exec_err, None)
    if proc.returncode == 0 and f"{_RESULT_MARKER_PREFIX}=PASS" in proc.stdout:
        return (True, None, _parse_executed_device(proc.stdout))
    return (False, _classify_task_submit_failure(proc.returncode, proc.stdout, proc.stderr), None)


def _run_batch_via_task_submit(
    entries: "list[tuple[Path, str]]",
    manifest_path: Path,
    device: str,
    dfx: "_DfxOpts",
    pto_isa_commit: str | None,
    max_time: int,
    queue_timeout: int,
) -> "dict[str, tuple[bool, str | None, int | None]]":
    """Run a batch of compiled artifacts in ONE task-submit task.

    *entries* is ``[(work_dir, platform), ...]``.  Writes a manifest and submits
    a single ``execute_artifact --batch-manifest`` task — one hot process, one
    ChipWorker device session for the whole batch — then parses the per-artifact
    markers into ``{str(work_dir): (ok, error, device)}``.  Artifacts with no
    marker (the batch process crashed before reaching them, or an infra kill)
    are reported as failures.  Blocks until the batch task exits.
    """
    manifest_path.write_text(
        json.dumps([{"work_dir": str(wd), "platform": plat} for wd, plat in entries]),
        encoding="utf-8",
    )
    inner = _build_execute_artifact_cmd(["--batch-manifest", str(manifest_path)], dfx, pto_isa_commit)
    proc, exec_err = _exec_task_submit(
        inner, device, max_time, queue_timeout, manifest_path.with_suffix(".log")
    )
    if proc is None:
        return {str(wd): (False, exec_err, None) for wd, _ in entries}
    # Walk the markers, attributing the lines printed before each one to that
    # artifact — so a FAIL carries its own traceback inline (the batch log is
    # ephemeral). "PYPTO_EXEC_RESULT=PASS work_dir=<wd> device=<N>".
    results: dict[str, tuple[bool, str | None, int | None]] = {}
    buf: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER_PREFIX) and "work_dir=" in line:
            wd = _marker_value(line, "work_dir=")
            if wd is not None:
                if line.startswith(f"{_RESULT_MARKER_PREFIX}=PASS"):
                    dev = _marker_value(line, "device=")
                    results[wd] = (True, None, int(dev) if dev and dev.isdigit() else None)
                elif line.startswith(f"{_RESULT_MARKER_PREFIX}=INFRA"):
                    # Reconstruction/setup failure for this artifact (stale cache,
                    # missing .o/.so) — infra, not a device test failure. Attributed
                    # to the entry so one bad artifact doesn't sink the whole batch.
                    results[wd] = (
                        False,
                        "artifact reconstruction/setup failed (infra):\n" + "\n".join(buf).strip(),
                        None,
                    )
                else:
                    results[wd] = (False, "device run failed:\n" + "\n".join(buf).strip(), None)
            buf = []
        else:
            buf.append(line)
    # Artifacts without a marker: the batch process died early, or an infra kill
    # (queue timeout / watchdog) that produced no per-artifact markers at all.
    # ``buf`` holds whatever the process printed after the last marker (the crash).
    # rc != 0 is a batch-level failure. Classify it (queue timeout / watchdog /
    # bootstrap crash / batch-level INFRA) only when it looks batch-wide — no
    # per-artifact markers parsed, an INFRA marker present, or no child output at
    # all. Otherwise (rc == 1 with some FAIL/PASS markers) each artifact already
    # carries its own verdict and the unmarked tail is a genuine died-before-this
    # case, so keep the per-artifact "no result marker" message.
    infra_err = None
    if proc.returncode != 0 and (
        not results
        or f"{_RESULT_MARKER_PREFIX}=INFRA" in proc.stdout
        or (not proc.stdout.strip() and not proc.stderr.strip())
    ):
        infra_err = _classify_task_submit_failure(proc.returncode, proc.stdout, proc.stderr)
    tail = "\n".join(buf[-40:]).strip()
    for wd, _ in entries:
        results.setdefault(
            str(wd),
            (
                False,
                infra_err or f"no result marker — batch process likely died before this case:\n{tail}",
                None,
            ),
        )
    # Record one stat line per batch for the end-of-session summary (the live
    # per-artifact markers are suppressed to keep pytest's log readable). The
    # actual card is whatever task-submit handed this batch — read it back from a
    # successful marker rather than the requested --device string ("auto"/range).
    n_ok = sum(1 for ok, _, _ in results.values() if ok)
    actual_device = next((d for ok, _, d in results.values() if ok and d is not None), None)
    _batch_stats.append((manifest_path.stem, n_ok, len(entries), actual_device))
    return results


def _validate_after_device_run(
    tc: "PTOTestCase",
    work_dir: Path,
    execution_time: float,
) -> RunResult:
    """Validate persisted device outputs with *tc*'s real tolerance (split path).

    The task-submit device run is validation-free (``--no-validate``); it leaves
    the actual outputs under ``work_dir/data/actual``.  This compares them
    against the golden using the test's ``RunConfig`` rtol/atol — the tolerance
    is applied here, in pytest's per-item lifecycle, not in the eager run.
    """
    try:
        validate_persisted_outputs(work_dir, tc.config.rtol, tc.config.atol)
    except Exception as exc:  # noqa: BLE001 — surfaced as a test failure
        return RunResult(
            passed=False,
            test_name=tc.get_name(),
            error=(
                f"golden validation failed (rtol={tc.config.rtol}, atol={tc.config.atol}):\n"
                f"{exc}\n{traceback.format_exc()}"
            ),
            execution_time=execution_time,
        )
    return RunResult(passed=True, test_name=tc.get_name(), execution_time=execution_time)


def _fused_execute_task(
    tc: "PTOTestCase",
    cache_key: str,
    artifact: "CompileArtifact",
) -> RunResult:
    """Acquire a device slot, execute on device.

    Submitted by ``TestRunner.run`` after the compile future resolves and
    after golden.py has been rewritten with the real ``RunConfig``.  Runs
    on an execute-pool worker thread; multiple exec tasks can be in flight
    when several pytest items resolve their compile futures concurrently
    (e.g. under xdist), but the device pool bounds parallelism to the
    number of devices in ``--device``.
    """
    start = time.time()
    name = tc.get_name()
    if artifact.error is not None:
        return RunResult(
            passed=False,
            test_name=name,
            error=f"Pre-compilation failed: {artifact.error}",
            execution_time=time.time() - start,
        )
    if _pipeline_ctx.get("codegen_only"):
        return RunResult(
            passed=True,
            test_name=name,
            execution_time=time.time() - start,
        )

    # Local device-pool path (the legacy lazy mode + sim under task-submit).
    # task-submit's onboard device runs go through the batch submitter, not here.
    assert _device_pool is not None, "device pool not initialised"
    device_id = _device_pool.get()
    try:
        _executed_device[cache_key] = device_id
        _execute_on_device(
            artifact.work_dir,
            artifact.work_dir / "golden.py",
            artifact.chip_callable,
            artifact.runtime_name,
            artifact.resolved_platform,
            device_id,
            dfx=_pipeline_ctx.get("dfx", _DfxOpts()),
        )
        return RunResult(
            passed=True,
            test_name=name,
            execution_time=time.time() - start,
        )
    except Exception as exc:
        return RunResult(
            passed=False,
            test_name=name,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            execution_time=time.time() - start,
        )
    finally:
        _device_pool.put(device_id)


def _schedule_exec_after_golden(
    tc: "PTOTestCase",
    cache_key: str,
    artifact: "CompileArtifact",
) -> Future:
    """Rewrite ``golden.py`` for *tc* and submit the execute task.

    Called from ``TestRunner.run`` once the compile future has resolved.
    The compile-time golden was written from a default-constructed
    ``PTOTestCase`` (no ``RunConfig``) and therefore uses default 1e-5
    tolerances; rewriting here picks up the real ``RunConfig`` passed by
    the test body.
    """
    if artifact.error is None and not _pipeline_ctx.get("codegen_only"):
        _write_golden_for_test_case(tc, artifact.work_dir / "golden.py")
    assert _execute_pool is not None, "execute pool not initialised"
    return _execute_pool.submit(_fused_execute_task, tc, cache_key, artifact)


def _await_all_batches() -> None:
    """Block until every submitted batch task has finished (drained).

    Used by the undiscovered-case inline path so its per-case task-submit child
    never cold-inits the card while a batch process is still doing the same.
    """
    _batches_ready.wait()
    for batch_fut in {fut for fut, _ in _case_to_batch.values()}:
        try:
            batch_fut.result()
        except Exception:  # noqa: BLE001 — batch errors surface on their own cases' run()
            pass


def _batch_submitter(batch_size: int, cache_dir: Path) -> None:
    """Stream compiled artifacts into batches and submit one task per batch.

    Runs on a background thread (so it doesn't block pytest's collection from
    returning).  Instead of waiting for the WHOLE compile pool to drain before
    submitting anything, it consumes compile futures in *completion* order and,
    as soon as *batch_size* have compiled, submits that batch to the execute
    pool via :func:`_run_batch_via_task_submit` — so the device run of the early
    batches OVERLAPS the still-running compiles of the later ones (the card
    starts working during the compile phase instead of sitting idle until the
    end).  Each case is mapped to its batch Future in ``_case_to_batch`` so
    ``TestRunner.run`` can await it; ``_batches_ready`` is set once every batch
    has been submitted.

    Compile failures / sim / codegen-only are skipped here and handled by
    ``run``'s lazy per-item path.

    The whole body runs under ``try/finally`` so ``_batches_ready`` is ALWAYS
    set: if this daemon thread raised before setting it, every task-submit case
    would block forever on ``_batches_ready.wait()`` in ``run``. On failure the
    cases simply find no ``_case_to_batch`` entry and report "not assigned to a
    batch" instead of hanging the session.
    """
    try:
        assert _execute_pool is not None, "execute pool not initialised"
        device = _pipeline_ctx.get("task_submit_device", "auto")
        dfx = _pipeline_ctx.get("dfx", _DfxOpts())
        pto_isa_commit = _pipeline_ctx.get("pto_isa_commit")
        max_time = _pipeline_ctx.get("task_max_time", 600)
        queue_timeout = _pipeline_ctx.get("task_queue_timeout", 1800)
        # A 0 / negative batch size would never fill a batch (``len(pending) >= 0``
        # is always true, submitting empty chunks while pending never drains),
        # hanging every task-submit test on ``_batches_ready``. Clamp to a minimum.
        batch_size = max(1, batch_size)

        def _submit(chunk: "list[tuple[str, Path, str]]", idx: int) -> None:
            entries = [(wd, plat) for _, wd, plat in chunk]
            manifest_path = cache_dir / f"batch_{idx}.json"
            # ``--task-max-time`` is a PER-CASE budget, but a batch runs all its
            # entries in one task-submit task, so scale the watchdog by the batch
            # length — otherwise a valid batch of many slow-but-passing artifacts is
            # killed at the single-case limit and the harness reports the survivors
            # as infra failures.
            batch_max_time = max_time * len(entries)
            fut = _execute_pool.submit(
                _run_batch_via_task_submit,
                entries,
                manifest_path,
                device,
                dfx,
                pto_isa_commit,
                batch_max_time,
                queue_timeout,
            )
            for key, wd, _ in chunk:
                _case_to_batch[key] = (fut, wd)

        key_by_fut = {cfut: key for key, cfut in _compile_futures.items()}
        batch_idx = 0
        # A batch child opens ONE ChipWorker from its first entry, so a batch must
        # never mix the (platform, runtime) that the worker is keyed on. ChipWorker
        # reuse matches on (level, platform, device_id, runtime); level/device_id
        # are fixed per batch child, so a differing platform OR runtime inside the
        # batch would miss ChipWorker.current() and open a SECOND Worker.init() on
        # the same card while the batch worker still holds it — the 2-inits-on-a-
        # busy-card halMemCtl EACCES hazard. Bucket by (resolved_platform,
        # runtime_name) to align with that reuse key and chunk within each bucket.
        # (platform, runtime) → list of (key, wd, platform)
        pending: dict[tuple[str, str | None], list[tuple[str, Path, str]]] = {}

        # Consume in completion order: fill a batch as cases finish compiling and
        # submit it immediately, so execution and compilation overlap.
        for cfut in as_completed(key_by_fut):
            try:
                artifact = cfut.result()
            except Exception:  # noqa: BLE001 — run() re-raises the compile crash with context
                continue
            if artifact.error is not None or _pipeline_ctx.get("codegen_only"):
                continue
            if artifact.resolved_platform.endswith("sim"):
                continue
            bucket_key = (artifact.resolved_platform, artifact.runtime_name)
            bucket = pending.setdefault(bucket_key, [])
            bucket.append((key_by_fut[cfut], artifact.work_dir, artifact.resolved_platform))
            if len(bucket) >= batch_size:
                _submit(bucket[:batch_size], batch_idx)
                batch_idx += 1
                pending[bucket_key] = bucket[batch_size:]

        for leftover in pending.values():  # final short batch per (platform, runtime)
            if leftover:
                _submit(leftover, batch_idx)
                batch_idx += 1
    finally:
        _batches_ready.set()


def start_pipeline(  # noqa: PLR0913
    *,
    test_cases: "list[PTOTestCase]",
    cache_dir: Path,
    session_platform: str,
    dump_passes: bool,
    codegen_only: bool,
    pto_isa_commit: str | None,
    compile_workers: int,
    device_pool: "queue.Queue[int]",
    analyze_auto_scopes_for_deps: bool = False,
    enable_l2_swimlane: bool = False,
    enable_dump_tensor: int = 0,
    enable_pmu: int = 0,
    enable_dep_gen: bool = False,
    enable_scope_stats: bool = False,
    execute_mode: str = "device-pool",
    task_max_time: int = 600,
    task_queue_timeout: int = 1800,
    task_submit_device: str = "auto",
    execute_batch_size: int = 64,
) -> None:
    """Spin up the compile pipeline and populate :data:`_compile_futures`.

    Called from ``pytest_collection_finish``.  Test cases are grouped by
    backend type (``set_backend_type`` is a global one-time setter); within
    each group a compile pool of ``compile_workers`` threads feeds the shared
    session-wide execute pool sized to the number of devices in ``device_pool``.

    Only the *non-final* groups block on a barrier before the next
    ``set_backend_type`` call; the last group returns immediately so pytest's
    per-item loop can start consuming execute futures while compile+execute
    are still running in the background.  This preserves the
    ``set_backend_type`` single-shot invariant without stalling pytest's
    progress reporting during collection.
    """
    global _device_pool, _execute_pool, _pipeline_ctx  # noqa: PLW0603

    _batch_stats.clear()  # fresh per session; read by pytest_terminal_summary

    # Resolve PTO_ISA_ROOT once on the main thread before any compile workers
    # start.  Otherwise concurrent workers race on `git clone` into the same
    # path — the first wins, the rest fail with "destination already exists"
    # and propagate "PTO_ISA_ROOT could not be resolved" as a pre-compilation
    # error.  Once the env var is set, workers short-circuit via the env-var
    # branch in ensure_pto_isa_root().
    if not codegen_only:
        from pypto.runtime.device_runner import ensure_pto_isa_root  # noqa: PLC0415

        ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")

    _device_pool = device_pool
    _pipeline_ctx = {
        "cache_dir": cache_dir,
        "session_platform": session_platform,
        "dump_passes": dump_passes,
        "codegen_only": codegen_only,
        "pto_isa_commit": pto_isa_commit,
        "analyze_auto_scopes_for_deps": analyze_auto_scopes_for_deps,
        "dfx": _DfxOpts(
            enable_l2_swimlane=enable_l2_swimlane,
            enable_dump_tensor=enable_dump_tensor,
            enable_pmu=enable_pmu,
            enable_dep_gen=enable_dep_gen,
            enable_scope_stats=enable_scope_stats,
        ),
        "execute_mode": execute_mode,
        "task_max_time": task_max_time,
        "task_queue_timeout": task_queue_timeout,
        "task_submit_device": task_submit_device,
        # In task-submit mode route the undiscovered-case tail through task-submit
        # too, so EVERY device run goes through task-submit's per-card lock. An
        # in-process device run would bypass that lock and collide with the
        # batch processes on the same card (halMemCtl EACCES). One mechanism, one
        # lock, no contention. Sim still runs in-process (it needs no card).
        "inline_via_task_submit": execute_mode == "task-submit",
    }
    # task-submit mode: every compiled artifact goes into a batch of
    # execute_batch_size, and each batch is one task-submit task (one hot
    # process).  Size the pool to the batch count so all batches are submitted at
    # once and task-submit owns the cross-card scheduling.  Device-pool mode keeps
    # tracking the local card count.
    if execute_mode == "task-submit":
        n_batches = max(1, math.ceil(len(test_cases) / max(1, execute_batch_size)))
        n_exec = min(n_batches, _MAX_TASK_SUBMIT_INFLIGHT)
    else:
        n_exec = max(1, device_pool.qsize())
    _execute_pool = ThreadPoolExecutor(max_workers=n_exec, thread_name_prefix="pypto-exec")

    groups: dict[BackendType, list[PTOTestCase]] = {}
    for tc in test_cases:
        groups.setdefault(tc.get_backend_type(), []).append(tc)

    group_items = list(groups.items())
    for i, (backend_type, group) in enumerate(group_items):
        is_last = i == len(group_items) - 1
        set_backend_type(backend_type)
        compile_pool = ThreadPoolExecutor(max_workers=compile_workers, thread_name_prefix="pypto-compile")
        _compile_pools.append(compile_pool)
        group_futs: list[Future] = []
        for tc in group:
            key = _cache_key(tc, _resolve_platform(session_platform, tc))
            cfut = compile_pool.submit(
                _fused_compile_task,
                tc,
                cache_dir,
                session_platform,
                dump_passes,
                analyze_auto_scopes_for_deps,
                pto_isa_commit,
            )
            _compile_futures[key] = cfut
            group_futs.append(cfut)
        if is_last:
            # Don't block: let pytest's per-item loop start running while
            # compiles continue in the background.  The compile pool stays
            # alive; shutdown_pipeline() tears it down at session end.
            continue
        # Non-final group: drain before the next set_backend_type so the
        # global backend state transitions cleanly.
        wait(group_futs)
        compile_pool.shutdown(wait=True)
        _compile_pools.remove(compile_pool)
        reset_for_testing()

    # task-submit mode: a background thread waits for all compiles, groups the
    # artifacts into batches and submits one task-submit task per batch.  Runs in
    # the background so pytest's per-item loop starts immediately; TestRunner.run
    # waits on _batches_ready before reading _case_to_batch.
    if execute_mode == "task-submit":
        threading.Thread(
            target=_batch_submitter,
            args=(execute_batch_size, cache_dir),
            name="pypto-batch-submitter",
            daemon=True,
        ).start()


def configure_inline_task_submit(
    *,
    task_max_time: int = 600,
    task_queue_timeout: int = 1800,
    task_submit_device: str = "auto",
) -> None:
    """Route the inline (no-pipeline) execute path through task-submit.

    Used when ``--execute-via-task-submit`` is passed *without*
    ``--precompile-workers``: no compile pipeline runs, but ``_run_inline``
    consults ``_pipeline_ctx`` to decide how to execute.  ``pto_isa_commit`` and
    DFX still come from the test's ``RunConfig`` on that path, so only the
    execute-mode toggle and task-submit knobs are stashed here.
    """
    _pipeline_ctx["execute_mode"] = "task-submit"
    _pipeline_ctx["task_max_time"] = task_max_time
    _pipeline_ctx["task_queue_timeout"] = task_queue_timeout
    _pipeline_ctx["task_submit_device"] = task_submit_device
    # No compile pipeline here, so there is no batch path: every case is inline
    # and must borrow a card per case via task-submit (the harness host may have
    # no local device). With a pipeline, undiscovered cases stay in-process —
    # see _run_inline.
    _pipeline_ctx["inline_via_task_submit"] = True
    # No _batch_submitter runs on this path, so nothing would ever set
    # _batches_ready. _run_inline still calls _await_all_batches() before its
    # per-case task-submit child (to serialize card init); mark the (empty) batch
    # set ready here so that wait returns immediately instead of hanging forever.
    _case_to_batch.clear()
    _batches_ready.set()


def shutdown_pipeline() -> None:
    """Tear down compile/execute pools; called from ``pytest_sessionfinish``."""
    global _execute_pool, _compile_pools  # noqa: PLW0603
    for pool in _compile_pools:
        pool.shutdown(wait=False, cancel_futures=True)
    _compile_pools = []
    if _execute_pool is not None:
        _execute_pool.shutdown(wait=False, cancel_futures=True)
    _execute_pool = None
    _case_to_batch.clear()
    _batches_ready.clear()
    # NOTE: _batch_stats is intentionally NOT cleared here — pytest_terminal_summary
    # runs *after* sessionfinish (which calls this) and reads it. It is reset at
    # the start of the next pipeline (start_pipeline) instead.


def execution_summary_lines() -> list[str]:
    """End-of-session summary of the task-submit device runs (for the terminal).

    The per-artifact device markers are suppressed to keep pytest's log clean, so
    without this the batched, card-borrowing execution is invisible. Returns one
    line showing how many batches ran, the pass count, and how the runs spread
    across the borrowed cards — or ``[]`` when no batch ran (non-task-submit mode).
    """
    if not _batch_stats:
        return []
    total = sum(n for _, _, n, _ in _batch_stats)
    ok = sum(n for _, n, _, _ in _batch_stats)
    by_device: dict[int | None, int] = {}
    for _, _, n, dev in _batch_stats:
        by_device[dev] = by_device.get(dev, 0) + n
    dist = ", ".join(
        f"device {d}: {c}" if d is not None else f"device ?: {c}"
        for d, c in sorted(by_device.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    )
    return [f"{len(_batch_stats)} batch(es), {ok}/{total} device runs ok | cards used: {dist}"]


class TestRunner:
    """Executes PTO test cases via simpler's CodeRunner.

    This runner integrates with simpler's CodeRunner to execute tests:
    1. Generate kernel and orchestration C++ from PyPTO program via ir.compile()
    2. Generate golden.py for reference computation
    3. Use CodeRunner to compile, execute, and validate

    Example:
        runner = TestRunner(RunConfig(platform="a2a3sim"))
        result = runner.run(my_test_case)
        assert result.passed
    """

    __test__ = False  # Not a pytest test class

    def __init__(self, config: RunConfig | None = None):
        """Initialize test runner.

        Args:
            config: Test configuration. If None, uses default config.
        """
        self.config = config or RunConfig()

    def run(self, test_case: PTOTestCase) -> RunResult:
        """Run a test case and return results.

        When the test case was discovered at collection time, this method
        waits for its compile future, rewrites ``golden.py`` with the
        test's real ``RunConfig`` (compile-time golden used default 1e-5
        tolerances because test classes are instantiated without args at
        collection), then submits and awaits the execute task.  Otherwise
        the legacy inline path runs on the calling thread.

        Args:
            test_case: The test case to run.

        Returns:
            RunResult with pass/fail status and details.
        """
        resolved_platform = _resolve_platform(self.config.platform, test_case)
        cache_k = _cache_key(test_case, resolved_platform)
        cfut = _compile_futures.get(cache_k)
        if cfut is not None:
            try:
                artifact = cfut.result()
            except Exception as exc:
                _last_device["value"] = None
                return RunResult(
                    passed=False,
                    test_name=test_case.get_name(),
                    error=f"compile task crashed: {exc}\n{traceback.format_exc()}",
                    execution_time=0.0,
                )
            # Pre-compilation failure / codegen-only must be handled BEFORE the
            # task-submit batch path: the batch submitter skips errored and
            # codegen-only artifacts (they never enter ``_case_to_batch``), so
            # otherwise a compile failure would surface as the misleading "compiled
            # but was not assigned to any batch" instead of its real traceback.
            if artifact.error is not None:
                _last_device["value"] = None
                return RunResult(
                    passed=False,
                    test_name=test_case.get_name(),
                    error=f"Pre-compilation failed: {artifact.error}",
                    execution_time=0.0,
                )
            if _pipeline_ctx.get("codegen_only"):
                _last_device["value"] = None
                return RunResult(
                    passed=True,
                    test_name=test_case.get_name(),
                    execution_time=0.0,
                )
            if _pipeline_ctx.get("execute_mode") == "task-submit" and not artifact.resolved_platform.endswith(
                "sim"
            ):
                # task-submit: this case's device run was batched and submitted by
                # the background _batch_submitter.  Wait for the batches to be
                # assigned, await this case's batch, pick out its per-artifact
                # result, then validate the persisted outputs with THIS test's
                # real tolerance.  The device run itself was tolerance-independent
                # and ran (with its whole batch) in one hot process.
                _batches_ready.wait()
                entry = _case_to_batch.get(cache_k)
                if entry is not None:
                    batch_fut, work_dir = entry
                    try:
                        batch_results = batch_fut.result()
                    except Exception as exc:
                        _last_device["value"] = None
                        return RunResult(
                            passed=False,
                            test_name=test_case.get_name(),
                            error=f"batch execute crashed: {exc}\n{traceback.format_exc()}",
                            execution_time=0.0,
                        )
                    ok, error, device = batch_results.get(
                        str(work_dir), (False, "no batch result for case", None)
                    )
                    _last_device["value"] = device
                    if not ok:
                        return RunResult(
                            passed=False,
                            test_name=test_case.get_name(),
                            error=error,
                            execution_time=0.0,
                        )
                    return _validate_after_device_run(test_case, work_dir, 0.0)
                # Compiled OK but not assigned to a batch — shouldn't happen.
                _last_device["value"] = None
                return RunResult(
                    passed=False,
                    test_name=test_case.get_name(),
                    error="task-submit mode: case compiled but was not assigned to any batch",
                    execution_time=0.0,
                )
            # device-pool / sim / codegen-only: legacy lazy per-item path
            # (run() rewrites golden + submits + validates in-process).
            exec_fut = _schedule_exec_after_golden(test_case, cache_k, artifact)
            try:
                result = exec_fut.result()
            except Exception as exc:
                _last_device["value"] = None
                return RunResult(
                    passed=False,
                    test_name=test_case.get_name(),
                    error=f"execute task crashed: {exc}\n{traceback.format_exc()}",
                    execution_time=0.0,
                )
            _last_device["value"] = _executed_device.get(cache_k)
            return result
        _last_device["value"] = self.config.device_id
        return self._run_inline(test_case, resolved_platform)

    def _run_inline(self, test_case: PTOTestCase, resolved_platform: str) -> RunResult:
        """Compile + execute on the calling thread.

        Used when ``--precompile-workers`` was not passed (pipeline disabled)
        or for test cases that were not discoverable at collection time
        (e.g. constructed dynamically inside a test body).  Single device only:
        ``self.config.device_id`` (the first id in ``--device``).
        """
        start_time = time.time()
        test_name = test_case.get_name()

        work_dir, use_temp = _inline_work_dir(self.config, test_name, resolved_platform)

        try:
            backend_type = test_case.get_backend_type()
            set_backend_type(backend_type)

            program = test_case.get_program()
            if program is None:
                raise ValueError(
                    f"Test case {test_name} must implement get_program() "
                    "to return a @pl.program class or ir.Program"
                )

            strategy = test_case.get_strategy()
            compile_program(
                program,
                work_dir,
                strategy=strategy,
                backend_type=backend_type,
                dump_passes=self.config.dump_passes,
                analyze_auto_scopes_for_deps=self.config.analyze_auto_scopes_for_deps,
                memory_planner=test_case.get_memory_planner(),
            )

            # External kernels are referenced in the manifest at their original
            # path (not copied into the artifact), so accept them even when no
            # kernel .cpp is generated under kernels/.
            config_path = work_dir / "kernel_config.py"
            kernels_in_manifest = config_path.exists() and '"func_id"' in config_path.read_text()
            if not list((work_dir / "kernels").rglob("*.cpp")) and not kernels_in_manifest:
                raise ValueError(f"No kernels generated for {test_name}")
            if not list((work_dir / "orchestration").glob("*.cpp")):
                raise ValueError(
                    f"No orchestration generated for {test_name}. "
                    "Ensure your @pl.program includes an orchestration function "
                    "(decorated with @pl.function(type=pl.FunctionType.Orchestration))."
                )

            golden_path = work_dir / "golden.py"
            _write_golden_for_test_case(test_case, golden_path)

            if self.config.codegen_only:
                return RunResult(
                    passed=True,
                    test_name=test_name,
                    execution_time=time.time() - start_time,
                )

            from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

            chip_callable, runtime_name, _ = compile_and_assemble(
                work_dir, resolved_platform, pto_isa_commit=self.config.pto_isa_commit
            )
            # Undiscovered cases (the minority the collection scan can't see)
            # borrow a card via task-submit too, so EVERY device run goes through
            # task-submit's per-card lock. Running these in-process would bypass
            # that lock and collide with the batch processes on the same card
            # (halMemCtl EACCES). One mechanism, one lock, no contention. Sim
            # never needs a card, so it always stays in-process.
            if _pipeline_ctx.get("inline_via_task_submit") and not resolved_platform.endswith("sim"):
                # Run this undiscovered case's device step only AFTER every batch
                # has drained, so its task-submit child doesn't cold-init the card
                # concurrently with the batch processes (2+ device inits at once
                # on a busy card → halMemCtl EACCES). Keeps init one-at-a-time.
                _await_all_batches()
                passed, error, device = _run_artifact_via_task_submit(
                    work_dir,
                    resolved_platform,
                    _DfxOpts.from_run_config(self.config),
                    self.config.pto_isa_commit,
                    _pipeline_ctx.get("task_max_time", 600),
                    _pipeline_ctx.get("task_queue_timeout", 1800),
                    _pipeline_ctx.get("task_submit_device", "auto"),
                )
                # Report the card task-submit actually lent (run() optimistically
                # stashed config.device_id before delegating here), so the
                # [DEVICE] line and per-device summary reflect the real card.
                if device is not None:
                    _last_device["value"] = device
                if not passed:
                    return RunResult(
                        passed=False,
                        test_name=test_name,
                        error=error,
                        execution_time=time.time() - start_time,
                    )
                # Device run (--no-validate) persisted outputs; validate here.
                return _validate_after_device_run(test_case, work_dir, time.time() - start_time)
            # RunTiming was dropped (simpler #1177): _execute_on_device returns
            # None now; timing is read from the runtime's [STRACE] log markers.
            _execute_on_device(
                work_dir,
                golden_path,
                chip_callable,
                runtime_name,
                resolved_platform,
                self.config.device_id,
                dfx=_DfxOpts.from_run_config(self.config),
            )

            return RunResult(
                passed=True,
                test_name=test_name,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return RunResult(
                passed=False,
                test_name=test_name,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
            )
        finally:
            if use_temp and work_dir.exists():
                shutil.rmtree(work_dir)


class TestSuite:
    """Collection of test cases that can be run together."""

    __test__ = False  # Not a pytest test class

    def __init__(self, name: str, config: RunConfig | None = None):
        """Initialize test suite.

        Args:
            name: Suite name.
            config: Configuration for all tests in suite.
        """
        self.name = name
        self.config = config or RunConfig()
        self._test_cases: list = []

    def add_test(self, test_case: PTOTestCase) -> "TestSuite":
        """Add a test case to the suite."""
        self._test_cases.append(test_case)
        return self

    def run_all(self, runner: TestRunner | None = None) -> dict[str, RunResult]:
        """Run all test cases in the suite."""
        if runner is None:
            runner = TestRunner(self.config)

        results = {}
        for test_case in self._test_cases:
            result = runner.run(test_case)
            results[test_case.get_name()] = result
            print(result)

        return results

    def summary(self, results: dict[str, RunResult]) -> str:
        """Generate summary of test results."""
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        failed = total - passed

        lines = [
            f"\n{'=' * 50}",
            f"Test Suite: {self.name}",
            f"{'=' * 50}",
            f"Passed: {passed}/{total}",
            f"Failed: {failed}/{total}",
        ]

        if failed > 0:
            lines.append("\nFailed tests:")
            for name, result in results.items():
                if not result.passed:
                    lines.append(f"  - {name}: {result.error}")

        return "\n".join(lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
