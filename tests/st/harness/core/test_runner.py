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

import concurrent.futures
import importlib.util
import logging
import os
import shutil
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import pytest
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.runtime import compile_program
from pypto.runtime.device_runner import (
    compile_and_assemble,
    compile_single_kernel,
    compile_single_orchestration,
    ensure_pto_isa_root,
)
from pypto.runtime.golden_writer import (
    _data_dir_has_files,
    _extract_compute_golden,
    _materialize_tensors,
    _save_data_files,
    generate_golden_source,
)
from pypto.runtime.kernel_compiler import KernelCompiler
from pypto.runtime.runner import (
    RunConfig,
    RunResult,
    _execute_on_device,
)
from pypto.runtime.tensor_spec import TensorSpec as RuntimeTensorSpec

from harness.core.harness import PTOTestCase

# tests/st/harness/core/test_runner.py -> tests/st/ -> project root
_ST_DIR = Path(__file__).parent.parent.parent
_PROJECT_ROOT = _ST_DIR.parent.parent
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-compilation cache (Phase 1 / Phase 2 split)
# ---------------------------------------------------------------------------

# Maps cache_key → (work_dir, error_str | None).
# The cache key combines test name and backend architecture (e.g.
# "matmul_64x64x64@a2a3") so the same PTOTestCase can be compiled for
# multiple backends without collisions.
# Populated by precompile_test_cases() in the parent process during
# pytest_collection_finish, before any test forks.  Forked children inherit
# the populated dict via os.fork() copy-on-write and find their pre-compiled
# artefacts on the filesystem under work_dir.
_precompile_cache: dict[str, tuple[Path, str | None]] = {}

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


def _cache_key(tc: PTOTestCase) -> str:
    """Return a unique cache key combining test name and backend architecture.

    Using a composite key allows the same ``PTOTestCase`` (same ``get_name()``)
    to be compiled for multiple backends (e.g. Ascend910B *and* Ascend950)
    without cache-key collisions.
    """
    arch = _BACKEND_TO_ARCH.get(tc.get_backend_type(), "unknown")
    return f"{tc.get_name()}@{arch}"


def _resolve_platform(config_platform: str, backend_type: BackendType) -> str:
    """Return the platform string required to compile for *backend_type*.

    Preserves the sim/hardware distinction from *config_platform* (i.e. the
    ``sim`` suffix) while replacing the architecture prefix to match the
    backend.  For example, if the global config says ``"a2a3sim"`` but the
    test case requests ``Ascend950``, this returns ``"a5sim"``.
    """
    is_sim = config_platform.endswith("sim")
    arch = _BACKEND_TO_ARCH.get(backend_type, config_platform.rstrip("sim").rstrip("_"))
    return f"{arch}sim" if is_sim else arch


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


def precompile_test_cases(
    test_cases: "list[PTOTestCase]",
    cache_dir: Path,
    *,
    dump_passes: bool = False,
    max_workers: int | None = None,
) -> None:
    """Compile all *test_cases* in parallel and populate :data:`_precompile_cache`.

    This is **Phase 1** of the two-phase execution model.  Call this once in
    the parent process (e.g. from a ``pytest_collection_finish`` hook) before
    any test forks.  Forked child processes inherit the populated cache and the
    pre-compiled artefacts on the filesystem.

    Because ``set_backend_type`` is a one-time global setter (calling it again
    with a *different* value raises ``ValueError``), test cases are grouped by
    backend type.  Each group is compiled sequentially with the backend set once;
    ``get_program()`` is serialised within the group (via ``_get_program_lock``)
    while ``compile_program()`` runs in parallel.  The backend is reset between
    groups via ``reset_for_testing()``.

    Args:
        test_cases: Instances to compile (should be deduplicated by
            ``_cache_key`` before calling).
        cache_dir: Root output directory; each test case is compiled into
            ``cache_dir / <cache_key>``.
        dump_passes: If ``True``, dump intermediate IR after each pass.
        max_workers: Thread-pool size per backend group.  Defaults to
            ``os.cpu_count()``.
    """
    # Group by backend type so set_backend_type is called once per group.
    groups: dict[BackendType, list[PTOTestCase]] = {}
    for tc in test_cases:
        groups.setdefault(tc.get_backend_type(), []).append(tc)

    def _compile_one(tc: "PTOTestCase") -> tuple[str, Path, str | None]:
        key = _cache_key(tc)
        work_dir = cache_dir / key
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            _compile_for_cache(tc, work_dir, dump_passes)
            return key, work_dir, None
        except Exception as exc:
            return key, work_dir, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    for backend_type, group in groups.items():
        # Set the backend type once for the whole group (idempotent if already
        # set to the same value; reset_for_testing() between groups handles reuse).
        set_backend_type(backend_type)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_compile_one, tc): tc for tc in group}
                for fut in concurrent.futures.as_completed(futures):
                    key, work_dir, error = fut.result()
                    _precompile_cache[key] = (work_dir, error)
        finally:
            # Reset so the next group can set a different backend type.
            reset_for_testing()


def prebuild_binaries(
    test_cases: "list[PTOTestCase]",
    cache_dir: Path,
    platform: str,
    *,
    max_workers: int | None = None,
    pto_isa_commit: str | None = None,
) -> int:
    """Phase 2: pre-compile binary artifacts for all test cases in parallel.

    Must be called AFTER :func:`precompile_test_cases` so kernel/orchestration
    C++ sources exist under ``work_dir``. Uses PyPTO's local KernelCompiler
    to compile incore kernels and orchestration ``.so`` files, with binary
    caching integrated into :mod:`pypto.runtime.device_runner`.

    Args:
        test_cases: Test case instances (deduplicated by cache key).
        cache_dir: Root output directory used during precompilation.
        platform: Session platform string (e.g. ``"a2a3"``).
        max_workers: Thread-pool size. Defaults to ``min(32, cpu_count + 4)``.
        pto_isa_commit: If set, pin the pto-isa clone to this specific git
            commit (hash or tag).  ``None`` means use latest remote HEAD.

    Returns:
        Number of test cases whose kernels and orchestration were successfully
        pre-built.
    """
    simpler_root = os.environ.get("SIMPLER_ROOT", "")
    if not simpler_root:
        return 0

    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    if pto_isa_root is None:
        return 0

    def _load_kc(work_dir: Path):
        config_path = work_dir / "kernel_config.py"
        if not config_path.exists():
            return None
        spec = importlib.util.spec_from_file_location(f"_prebin_{work_dir.name}", str(config_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # ── Collect configs for all valid test cases ──────────────────────────────
    case_configs: list[tuple] = []
    for tc in test_cases:
        key = _cache_key(tc)
        if key not in _precompile_cache or _precompile_cache[key][1] is not None:
            continue
        work_dir = _precompile_cache[key][0]
        mod = _load_kc(work_dir)
        if mod is None:
            continue
        tc_platform = _resolve_platform(platform, tc.get_backend_type())
        runtime_name = getattr(mod, "RUNTIME_CONFIG", {}).get("runtime", "host_build_graph")
        compiler = KernelCompiler(platform=tc_platform)
        case_configs.append(
            (tc, compiler, tc_platform, runtime_name, mod.KERNELS, mod.ORCHESTRATION["source"])
        )

    if not case_configs:
        return 0

    # ── Submit ALL tasks to a single flat pool ────────────────────────────────
    def _compile_incore_task(compiler, tc_platform, kernel, runtime_name):
        source = Path(kernel["source"])
        prebuild_cache = source.parent.parent.parent / "cache"
        compile_single_kernel(
            kernel,
            compiler,
            tc_platform,
            pto_isa_root,
            runtime_name,
            cache_dir=prebuild_cache,
        )

    def _compile_orch_task(compiler, runtime_name, orch_source):
        source = Path(orch_source)
        prebuild_cache = source.parent.parent / "cache"
        compile_single_orchestration(
            orch_source,
            compiler,
            runtime_name,
            cache_dir=prebuild_cache,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        all_futs: list = []

        # Per-test-case: each kernel and orchestration as independent tasks
        tc_kernel_futs: list[tuple[list, concurrent.futures.Future]] = []
        for tc, compiler, tc_platform, runtime_name, kernels, orch_source in case_configs:
            kfuts = [
                pool.submit(_compile_incore_task, compiler, tc_platform, k, runtime_name) for k in kernels
            ]
            ofut = pool.submit(_compile_orch_task, compiler, runtime_name, orch_source)
            tc_kernel_futs.append((kfuts, ofut))
            all_futs.extend(kfuts)
            all_futs.append(ofut)

        for fut in concurrent.futures.as_completed(all_futs):
            try:
                fut.result()
            except Exception as e:
                _log.debug("Pre-build task failed (will fall back to live compilation): %s", e)

    n_ok = sum(
        1
        for kfuts, ofut in tc_kernel_futs
        if all(f.exception() is None for f in kfuts) and ofut.exception() is None
    )
    return n_ok


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

        Args:
            test_case: The test case to run.

        Returns:
            RunResult with pass/fail status and details.
        """
        start_time = time.time()
        test_name = test_case.get_name()
        cache_k = _cache_key(test_case)

        # --- Phase 2: pre-compiled artifacts available — skip compilation ---
        if cache_k in _precompile_cache:
            cached_dir, cached_error = _precompile_cache[cache_k]
            if cached_error is not None:
                return RunResult(
                    passed=False,
                    test_name=test_name,
                    error=f"Pre-compilation failed: {cached_error}",
                    execution_time=time.time() - start_time,
                )
            if self.config.codegen_only:
                return RunResult(
                    passed=True,
                    test_name=test_name,
                    execution_time=time.time() - start_time,
                )
            try:
                backend_type = test_case.get_backend_type()
                platform = _resolve_platform(self.config.platform, backend_type)
                # Re-write golden.py with the actual test case's tolerances.
                # The pre-compiled golden.py may have been written with default
                # tolerances (1e-5) because pytest_collection_finish instantiates
                # test cases without their RunConfig constructor args.
                _write_golden_for_test_case(test_case, cached_dir / "golden.py")
                chip_callable, runtime_name = compile_and_assemble(
                    cached_dir, platform, pto_isa_commit=self.config.pto_isa_commit
                )
                _execute_on_device(
                    cached_dir,
                    cached_dir / "golden.py",
                    chip_callable,
                    runtime_name,
                    platform,
                    self.config.device_id,
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

        # --- Original path: no cache, compile + execute in one step ---

        # Determine work directory based on save_kernels configuration
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
            # Set PyPTO backend type for code generation
            backend_type = test_case.get_backend_type()
            set_backend_type(backend_type)

            # 1. Get program
            program = test_case.get_program()
            if program is None:
                raise ValueError(
                    f"Test case {test_name} must implement get_program() "
                    "to return a @pl.program class or ir.Program"
                )

            # 2. Compile: generates kernels/, orchestration/ and patches headers
            strategy = test_case.get_strategy()
            compile_program(
                program,
                work_dir,
                strategy=strategy,
                backend_type=backend_type,
                dump_passes=self.config.dump_passes,
            )

            # 3. Validate that kernels and orchestration were generated
            if not list((work_dir / "kernels").rglob("*.cpp")):
                raise ValueError(f"No kernels generated for {test_name}")
            if not list((work_dir / "orchestration").glob("*.cpp")):
                raise ValueError(
                    f"No orchestration generated for {test_name}. "
                    "Ensure your @pl.program includes an orchestration function "
                    "(decorated with @pl.function(type=pl.FunctionType.Orchestration))."
                )

            # 4. Generate golden.py in work_dir
            golden_path = work_dir / "golden.py"
            _write_golden_for_test_case(test_case, golden_path)

            # 5. Execute via CodeRunner (skip if codegen_only)
            if self.config.codegen_only:
                return RunResult(
                    passed=True,
                    test_name=test_name,
                    execution_time=time.time() - start_time,
                )

            platform = _resolve_platform(self.config.platform, backend_type)
            chip_callable, runtime_name = compile_and_assemble(
                work_dir, platform, pto_isa_commit=self.config.pto_isa_commit
            )
            _execute_on_device(
                work_dir,
                golden_path,
                chip_callable,
                runtime_name,
                platform,
                self.config.device_id,
                runtime_profiling=self.config.runtime_profiling,
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
