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

import logging
import queue
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, wait
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


def _compile_for_cache(test_case: "PTOTestCase", work_dir: Path, dump_passes: bool) -> None:
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
        _compile_for_cache(tc, work_dir, dump_passes)
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
    enable_l2_swimlane: bool = False,
    enable_dump_tensor: bool = False,
    enable_pmu: int = 0,
    enable_dep_gen: bool = False,
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
        "dfx": _DfxOpts(
            enable_l2_swimlane=enable_l2_swimlane,
            enable_dump_tensor=enable_dump_tensor,
            enable_pmu=enable_pmu,
            enable_dep_gen=enable_dep_gen,
        ),
    }
    n_devices = device_pool.qsize()
    _execute_pool = ThreadPoolExecutor(max_workers=max(1, n_devices), thread_name_prefix="pypto-exec")

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
                pto_isa_commit,
            )
            # TestRunner.run() blocks on this compile future, rewrites
            # golden.py with the real RunConfig, then submits the exec
            # task itself.  Keeping exec submission on the main thread
            # aligns C++ stdout with pytest's per-test capture window and
            # ensures the real tolerances reach golden.py.
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


def shutdown_pipeline() -> None:
    """Tear down compile/execute pools; called from ``pytest_sessionfinish``."""
    global _execute_pool, _compile_pools  # noqa: PLW0603
    for pool in _compile_pools:
        pool.shutdown(wait=False, cancel_futures=True)
    _compile_pools = []
    if _execute_pool is not None:
        _execute_pool.shutdown(wait=False, cancel_futures=True)
    _execute_pool = None


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

        if self.config.save_kernels:
            if self.config.save_kernels_dir:
                work_dir = Path(self.config.save_kernels_dir) / test_name
            else:
                work_dir = _default_work_dir(test_name)
            work_dir.mkdir(parents=True, exist_ok=True)
            use_temp = False
        else:
            work_dir = Path(tempfile.mkdtemp(prefix=f"pypto_test_{test_name}_"))
            use_temp = True

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
            )

            if not list((work_dir / "kernels").rglob("*.cpp")):
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
