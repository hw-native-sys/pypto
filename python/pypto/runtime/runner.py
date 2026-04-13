# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO runtime runner.

Provides :func:`run`, the main entry point for compiling a ``@pl.program`` and
executing it on an Ascend NPU (or simulator), with correctness validation against
a user-supplied golden function.

Typical usage::

    import torch
    from pypto.runtime import run, RunConfig, TensorSpec

    def golden(tensors, params):
        tensors["out"][:] = tensors["a"] + tensors["b"]

    result = run(
        program=MyProgram,
        tensor_specs=[
            TensorSpec("a",   [128, 128], torch.float32, init_value=2.0),
            TensorSpec("b",   [128, 128], torch.float32, init_value=3.0),
            TensorSpec("out", [128, 128], torch.float32, is_output=True),
        ],
        golden=golden,
        config=RunConfig(platform="a2a3sim"),
    )
    print(result)  # PASS / FAIL: ...
"""

import ctypes
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from pypto import ir
from pypto.backend import BackendType, set_backend_type
from pypto.compile_profiling import CompileProfiler, get_active_profiler
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.pypto_core.passes import WarningCheckSet, WarningLevel

from .golden_writer import write_golden
from .tensor_spec import TensorSpec

_OUTPUTS_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Cache file helpers
# ---------------------------------------------------------------------------


def _cache_dir(golden_path: Path) -> Path:
    """Return the ``cache/`` subdirectory co-located with ``golden.py``."""
    return golden_path.parent / "cache"


def _inputs_cache_file(golden_path: Path, case_name: str) -> Path:
    """Return the path for the pre-generated inputs ``.pt`` file.

    All cache artefacts live under ``work_dir/cache/`` alongside the other
    test-case outputs::

        work_dir/
          cache/
            Default_inputs.pt
            Default_golden.pt
            Case1_inputs.pt
            Case1_golden.pt
          golden.py
          kernels/
          orchestration/
    """
    safe = case_name.replace("/", "_").replace(" ", "_")
    return _cache_dir(golden_path) / f"{safe}_inputs.pt"


def _golden_cache_file(golden_path: Path, case_name: str) -> Path:
    """Return the path for the pre-computed golden outputs ``.pt`` file."""
    safe = case_name.replace("/", "_").replace(" ", "_")
    return _cache_dir(golden_path) / f"{safe}_golden.pt"


def _save_inputs(result: list, path: Path) -> None:
    """Serialise ``generate_inputs()`` result to *path* via ``torch.save``.

    Each item in *result* is wrapped in a small dict so that ctypes scalars
    can be reconstructed faithfully on load::

        {"kind": "tensor", "name": "a",    "data": <torch.Tensor>}
        {"kind": "ctypes", "name": "size", "ctype": "c_int64", "value": 1024}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = []
    for name, val in result:
        if isinstance(val, torch.Tensor):
            serialisable.append({"kind": "tensor", "name": name, "data": val})
        elif isinstance(val, ctypes._SimpleCData):
            serialisable.append(
                {
                    "kind": "ctypes",
                    "name": name,
                    "ctype": type(val).__name__,
                    "value": val.value,
                }
            )
        else:
            raise TypeError(f"Cannot serialise arg {name!r}: unsupported type {type(val)}")
    torch.save(serialisable, path)


def _load_inputs(path: Path) -> list | None:
    """Load and reconstruct a ``generate_inputs()`` result from *path*.

    Returns ``None`` if the file does not exist or cannot be read.
    """
    if not path.exists():
        return None
    try:
        items = torch.load(path, weights_only=False)
        result = []
        for item in items:
            name = item["name"]
            if item["kind"] == "tensor":
                result.append((name, item["data"]))
            elif item["kind"] == "ctypes":
                ctype_cls = getattr(ctypes, item["ctype"])
                result.append((name, ctype_cls(item["value"])))
        return result
    except Exception:
        return None


def _save_golden(golden: dict, path: Path) -> None:
    """Serialise pre-computed golden output tensors to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(golden, path)


def _load_golden(path: Path) -> dict | None:
    """Load pre-computed golden output tensors from *path*.

    Returns ``None`` if the file does not exist or cannot be read.
    """
    if not path.exists():
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception:
        return None


@dataclass
class RunConfig:
    """Configuration for a :func:`run` invocation or harness test execution.

    Attributes:
        platform: Target execution platform — ``"a2a3sim"`` / ``"a2a3"``
            (Ascend 910B) or ``"a5sim"`` / ``"a5"`` (Ascend 950).
        device_id: Hardware device index (ignored for simulator).
        rtol: Relative tolerance for result comparison.
        atol: Absolute tolerance for result comparison.
        strategy: PyPTO optimisation strategy applied during compilation.
        backend_type: Code-generation backend (:attr:`BackendType.Ascend910B` by default).
        dump_passes: If ``True``, dump intermediate IR after each pass.
        save_kernels: If ``True``, retain generated artefacts after execution.
            When ``False`` (default), a temporary directory is used and cleaned up.
        save_kernels_dir: Directory to save generated artefacts when *save_kernels*
            is ``True``.  If ``None``, a timestamped directory is created under
            ``build_output/<program_name>_<timestamp>``.
        codegen_only: If ``True``, stop after code generation without executing
            on device.  Useful for validating compilation output.
        pto_isa_commit: If set, pin the pto-isa clone to this specific git
            commit (hash or tag).  ``None`` means use the latest remote HEAD.
        runtime_profiling: If ``True``, enable runtime profiling and
            generate ``swimlane.json`` after execution.
        compile_profiling: If ``True``, enable compile profiling that records
            per-stage wall-clock timings (parse, passes, codegen).
            Results are written to ``report/pipeline_profile.{txt,json}`` in
            the output directory.
        warning_level: Override warning level for compilation. ``None`` uses the
            default (``PrePipeline``, or ``PYPTO_WARNING_LEVEL`` env var).
        disabled_warnings: Set of warning checks to disable during compilation.
            ``None`` uses the default (``UnusedControlFlowResult`` disabled).
        golden_data_dir: Target directory for ``.pt`` data files.  When set,
            the generated ``golden.py`` always loads tensors from this path.
            If the directory already contains all required ``.pt`` files they
            are reused; otherwise the directory is created and data is generated
            there.  Use a path from a previous run
            (e.g. ``build_output/<name>_<ts>/data``) to reuse existing golden
            data, or specify a new path to persist data to a fixed location.
    """

    __test__ = False  # Not a pytest test class

    platform: str = "a2a3sim"
    device_id: int = 0
    rtol: float = 1e-5
    atol: float = 1e-5
    strategy: OptimizationStrategy = field(default_factory=lambda: OptimizationStrategy.Default)
    backend_type: BackendType = field(default_factory=lambda: BackendType.Ascend910B)
    dump_passes: bool = False
    save_kernels: bool = False
    save_kernels_dir: str | None = None
    codegen_only: bool = False
    pto_isa_commit: str | None = None
    runtime_profiling: bool = False
    compile_profiling: bool = False
    warning_level: WarningLevel | None = None
    disabled_warnings: WarningCheckSet | None = None
    golden_data_dir: str | None = None

    def __post_init__(self) -> None:
        if self.platform not in ("a2a3sim", "a2a3", "a5sim", "a5"):
            raise ValueError(
                f"Invalid platform {self.platform!r}. Expected 'a2a3sim', 'a2a3', 'a5sim', or 'a5'."
            )
        # Auto-correct platform to match backend_type so compilation and execution
        # always target the same architecture.
        expected_arch = "a5" if self.backend_type == BackendType.Ascend950 else "a2a3"
        if not self.platform.startswith(expected_arch):
            sim_suffix = "sim" if self.platform.endswith("sim") else ""
            self.platform = f"{expected_arch}{sim_suffix}"
        # Runtime profiling requires kernel artefacts to be retained so
        # swimlane files can reference kernel_config.py.
        if self.runtime_profiling and not self.save_kernels:
            self.save_kernels = True


@dataclass
class RunResult:
    """Result of a program run or harness test execution.

    Attributes:
        passed: ``True`` if the program executed and results matched the golden
            reference within the configured tolerances.
        test_name: Optional test case name.  Set by the harness when running
            a named test case; ``None`` for direct :func:`run` calls.
        error: Human-readable error message when ``passed`` is ``False``.
        execution_time: Wall-clock time in seconds for the full run (compile +
            execute + validate).
    """

    __test__ = False  # Not a pytest test class

    passed: bool
    test_name: str | None = None
    error: str | None = None
    execution_time: float | None = None
    profile: dict[str, Any] | None = None

    def __str__(self) -> str:
        time_str = f" ({self.execution_time:.2f}s)" if self.execution_time else ""
        if self.passed:
            prefix = f"PASS: {self.test_name}" if self.test_name else "PASS"
            return prefix + time_str
        if self.test_name:
            msg = f"FAIL: {self.test_name}"
            if self.error:
                msg += f" - {self.error}"
        else:
            msg = "FAIL"
            if self.error:
                msg += f": {self.error}"
        return msg + time_str


def compile_program(
    program: Any,
    work_dir: Path,
    *,
    strategy: OptimizationStrategy,
    backend_type: BackendType,
    dump_passes: bool = False,
    warning_level: WarningLevel | None = None,
    disabled_warnings: WarningCheckSet | None = None,
    profiling: bool = False,
) -> None:
    """Compile *program* to *work_dir* and patch orchestration headers.

    Runs :func:`ir.compile` then inserts ``runtime.h`` / ``<iostream>`` includes
    into the generated orchestration C++ files (required by Simpler's CodeRunner).

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program`` object.
        work_dir: Output directory for generated artefacts.
        strategy: PyPTO optimisation strategy applied during compilation.
        backend_type: Code-generation backend.
        dump_passes: If ``True``, dump intermediate IR after each pass.
        warning_level: Override warning level for compilation.
        disabled_warnings: Set of warning checks to disable.
        profiling: If ``True``, enable compile profiling.
    """
    ir.compile(
        program,
        output_dir=str(work_dir),
        strategy=strategy,
        dump_passes=dump_passes,
        backend_type=backend_type,
        warning_level=warning_level,
        disabled_warnings=disabled_warnings,
        profiling=profiling,
    )
    _patch_orchestration_headers(work_dir)


def run(
    program: Any,
    tensor_specs: list[TensorSpec],
    golden: Callable,
    config: RunConfig | None = None,
) -> RunResult:
    """Compile *program* and run it on device, validating against *golden*.

    The full pipeline executed by this function:

    1. Call :func:`ir.compile` to generate CCE C++ kernel and orchestration files.
    2. Patch the orchestration file with the required ``runtime.h`` header.
    3. If *codegen_only*, return immediately.
    4. Compile kernels + orchestration to binaries, assemble ``ChipCallable``.
    5. Build tensor arguments, compute golden, execute on device, validate.

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program`` object.
        tensor_specs: Ordered list of tensor specifications.  The order must match
            the parameter order of the program's orchestration function.
        golden: A function with signature ``golden(tensors, params)`` that
            computes the expected outputs in-place (writes to
            ``tensors[output_name]``).  The function name does not matter.
        config: Run configuration.  Uses default :class:`RunConfig` if ``None``.

    Returns:
        :class:`RunResult` with ``passed=True`` on success, or ``passed=False``
        with an ``error`` message on failure.

    Example:
        >>> result = run(MyProgram, specs, my_golden, RunConfig(platform="a2a3sim"))
        >>> assert result.passed, str(result)
    """
    if config is None:
        config = RunConfig()

    start_time = time.time()
    if config.save_kernels_dir:
        work_dir = Path(config.save_kernels_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = Path("build_output") / f"{program.name}_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Compile profiling ---------------------------------------------------
    prof = get_active_profiler()
    owns_profiler = False
    if prof is None and config.compile_profiling:
        prof = CompileProfiler()
        prof.__enter__()
        owns_profiler = True

    def _stage(name: str) -> AbstractContextManager[Any]:
        if prof is not None:
            return prof.stage(name)
        return nullcontext()

    try:
        # 1. Set backend for code generation
        set_backend_type(config.backend_type)

        # 2. Compile: generates kernels/, orchestration/, kernel_config.py
        #    and patches orchestration headers
        with _stage("compile"):
            compile_program(
                program,
                work_dir,
                strategy=config.strategy,
                backend_type=config.backend_type,
                dump_passes=config.dump_passes,
                warning_level=config.warning_level,
                disabled_warnings=config.disabled_warnings,
                profiling=config.compile_profiling,
            )

        # 3. Write golden.py (still used for test harness compatibility)
        golden_path = work_dir / "golden.py"
        with _stage("golden_write"):
            write_golden(
                tensor_specs,
                golden,
                golden_path,
                rtol=config.rtol,
                atol=config.atol,
                data_dir=config.golden_data_dir,
            )

        # 4. If codegen_only, stop here — no binary compilation or device execution
        if config.codegen_only:
            profile_data = prof.to_dict() if prof is not None else None
            return RunResult(passed=True, execution_time=time.time() - start_time, profile=profile_data)

        # 5. Compile C++ → binaries + assemble ChipCallable
        from .device_runner import (  # noqa: PLC0415
            build_orch_args,
            compile_and_assemble,
            execute_on_device,
            validate_golden,
        )

        with _stage("binary_compilation"):
            chip_callable, binaries = compile_and_assemble(
                work_dir, config.platform, pto_isa_commit=config.pto_isa_commit
            )

        # 6. Build tensor arguments
        orch_args, all_tensors, inputs, outputs = build_orch_args(tensor_specs)

        # 7. Compute golden
        golden_outputs = {k: v.clone() for k, v in outputs.items()}
        golden_with_inputs = {**inputs, **golden_outputs}
        golden(golden_with_inputs, {})

        # 8. Execute on device
        if config.runtime_profiling:
            pre_run_logs, device_log_dir, pre_run_perf_files = _snapshot_profiling_state(
                config.platform, config.device_id
            )

        with _stage("device_execution"):
            execute_on_device(
                chip_callable,
                orch_args,
                binaries,
                config.device_id,
                enable_profiling=config.runtime_profiling,
            )

        if config.runtime_profiling:
            _collect_swimlane_data(
                work_dir,
                config.platform,
                config.device_id,
                pre_run_logs,
                device_log_dir,
                pre_run_perf_files,
            )

        # 9. Validate
        validate_golden(outputs, golden_outputs, rtol=config.rtol, atol=config.atol)

        profile_data = prof.to_dict() if prof is not None else None
        return RunResult(passed=True, execution_time=time.time() - start_time, profile=profile_data)

    except Exception:
        profile_data = prof.to_dict() if prof is not None else None
        return RunResult(
            passed=False,
            error=traceback.format_exc(),
            execution_time=time.time() - start_time,
            profile=profile_data,
        )
    finally:
        if prof is not None:
            report_dir = work_dir / "report"
            prof.write_report(str(report_dir))
        if owns_profiler and prof is not None:
            prof.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _execute_on_device(
    work_dir: Path,
    golden_path: Path,
    platform: str,
    device_id: int,
    pto_isa_commit: str | None = None,
    runtime_profiling: bool = False,
) -> None:
    """Compile, load, execute, and validate via PyPTO's device runner.

    This function is used by the test harness (test_runner.py) to execute
    pre-compiled test cases. For direct use, prefer :func:`run`.

    Args:
        work_dir: Root output directory containing ``kernels/`` and ``orchestration/``.
        golden_path: Path to the generated ``golden.py`` file.
        platform: Target execution platform.
        device_id: Hardware device index.
        pto_isa_commit: If set, pin the pto-isa clone to this specific commit.
        runtime_profiling: If ``True``, enable runtime profiling.
    """
    from .device_runner import (  # noqa: PLC0415
        build_orch_args_from_inputs,
        compile_and_assemble,
        execute_on_device,
        validate_golden,
    )

    chip_callable, binaries = compile_and_assemble(work_dir, platform, pto_isa_commit=pto_isa_commit)

    # Load golden.py to get generate_inputs and compute_golden
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("_golden", str(golden_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load golden.py from {golden_path}")
    golden_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden_module)

    # Load golden input cache if available
    params: dict = {"name": "Default"}
    result = _load_inputs(_inputs_cache_file(golden_path, "Default"))
    if result is None:
        result = golden_module.generate_inputs(params)

    output_names = set(getattr(golden_module, "__outputs__", []))
    orch_args, all_tensors, inputs, outputs = build_orch_args_from_inputs(result, output_names)

    # Compute golden
    golden_cache = _load_golden(_golden_cache_file(golden_path, "Default"))
    if golden_cache is not None:
        golden_out = golden_cache
    else:
        golden_out = {k: v.clone() for k, v in outputs.items()}
        golden_with_inputs = {**inputs, **golden_out}
        golden_module.compute_golden(golden_with_inputs, params)

    # Execute
    if runtime_profiling:
        pre_run_logs, device_log_dir, pre_run_perf_files = _snapshot_profiling_state(platform, device_id)

    execute_on_device(
        chip_callable,
        orch_args,
        binaries,
        device_id,
        enable_profiling=runtime_profiling,
    )

    if runtime_profiling:
        _collect_swimlane_data(
            work_dir,
            platform,
            device_id,
            pre_run_logs,
            device_log_dir,
            pre_run_perf_files,
        )

    # Validate
    rtol = getattr(golden_module, "RTOL", 1e-5)
    atol = getattr(golden_module, "ATOL", 1e-5)
    validate_golden(outputs, golden_out, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Swimlane profiling helpers
# ---------------------------------------------------------------------------


def _snapshot_profiling_state(platform: str, device_id: int) -> tuple[set[Path], Path | None, set[Path]]:
    """Snapshot device logs and perf files before a profiled execution.

    Returns:
        ``(pre_run_logs, device_log_dir, pre_run_perf_files)``
    """
    pre_run_logs: set[Path] = set()
    device_log_dir: Path | None = None
    if not platform.endswith("sim"):
        device_log_dir = _get_device_log_dir(device_id)
        if device_log_dir.exists():
            pre_run_logs = set(device_log_dir.glob("*.log"))

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pre_run_perf_files = set(_OUTPUTS_DIR.glob("perf_swimlane_*.json"))

    return pre_run_logs, device_log_dir, pre_run_perf_files


def _collect_swimlane_data(
    work_dir: Path,
    platform: str,
    device_id: int,
    pre_run_logs: set[Path],
    device_log_dir: Path | None,
    pre_run_perf_files: set[Path],
) -> None:
    """Collect swimlane profiling data after a profiled device execution.

    Moves ``perf_swimlane_*.json`` into ``work_dir/swimlane_data/`` and runs
    Simpler's ``swimlane_converter.py`` (if available) to produce merged JSON.
    """
    simpler_root = os.environ.get("SIMPLER_ROOT")
    swimlane_dir = work_dir / "swimlane_data"
    swimlane_dir.mkdir(parents=True, exist_ok=True)

    new_perf_files = set(_OUTPUTS_DIR.glob("perf_swimlane_*.json")) - pre_run_perf_files
    perf_file: Path | None = None
    if new_perf_files:
        perf_file = max(new_perf_files, key=lambda p: p.stat().st_mtime)
        dest = swimlane_dir / perf_file.name
        perf_file.rename(dest)
        perf_file = dest
        try:
            _OUTPUTS_DIR.rmdir()
        except OSError:
            pass

    if not platform.endswith("sim"):
        _generate_swimlane(
            work_dir,
            device_id,
            device_log_dir,
            pre_run_logs,
            simpler_root,
            swimlane_dir,
            perf_file,
        )


def _get_device_log_dir(device_id: int) -> Path:
    """Return the CANN device log directory for *device_id*."""
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        root = Path(ascend_work_path).expanduser() / "log" / "debug"
        if root.exists():
            return root / f"device-{device_id}"
    return Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"


def _wait_for_new_device_log(
    log_dir: Path, pre_run_logs: set[Path], timeout: float = 15, interval: float = 0.5
) -> Path | None:
    """Wait for a new ``*.log`` file in *log_dir* that wasn't present before the run.

    CANN dlog writes device logs asynchronously, so the file may appear
    a few seconds after execution completes.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_dir.exists():
            new_logs = set(log_dir.glob("*.log")) - pre_run_logs
            if new_logs:
                return max(new_logs, key=lambda p: p.stat().st_mtime)
        time.sleep(interval)
    return None


def _generate_swimlane(
    work_dir: Path,
    device_id: int,
    device_log_dir: Path | None,
    pre_run_logs: set[Path],
    simpler_root: str | None,
    swimlane_dir: Path,
    perf_file: Path | None,
) -> None:
    """Run Simpler's swimlane_converter.py to generate ``merged_swimlane_*.json``.

    Output is written to *swimlane_dir* alongside the input ``perf_swimlane_*.json``.

    Args:
        work_dir: Directory containing ``kernel_config.py``.
        device_id: Hardware device index (fallback when no device log found).
        device_log_dir: CANN device log directory snapshotted before the run.
        pre_run_logs: Set of log files that existed before the run.
        simpler_root: Path to the Simpler repository root.
        swimlane_dir: Directory where swimlane JSON files are written.
        perf_file: Path to the ``perf_swimlane_*.json`` file produced by
            CodeRunner and already moved into *swimlane_dir*.  When ``None``,
            swimlane conversion is skipped.
    """
    if not simpler_root:
        return

    swimlane_script = Path(simpler_root) / "tools" / "swimlane_converter.py"
    if not swimlane_script.exists():
        return

    if perf_file is None:
        print("No perf_swimlane_*.json found, skipping swimlane conversion")
        return

    kernel_config_path = work_dir / "kernel_config.py"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = swimlane_dir / f"merged_swimlane_{timestamp}.json"

    cmd = [
        sys.executable,
        str(swimlane_script),
        str(perf_file),
        "-o",
        str(output_path),
        "-k",
        str(kernel_config_path),
    ]

    if device_log_dir is not None:
        device_log_file = _wait_for_new_device_log(device_log_dir, pre_run_logs)
        if device_log_file:
            cmd += ["--device-log", str(device_log_file)]
        else:
            cmd += ["-d", str(device_id)]
    else:
        cmd += ["-d", str(device_id)]

    try:
        subprocess.run(cmd, check=True)
        print(f"Swimlane JSON written to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"swimlane_converter.py failed (exit {e.returncode}), no swimlane generated")


def _patch_orchestration_headers(work_dir: Path) -> None:
    """Add ``runtime.h`` and ``<iostream>`` includes to orchestration C++ files.

    Simpler's CodeRunner requires these headers in the orchestration translation
    unit.  They are added here rather than in the code generator so that the
    compiler back-end remains unaware of runtime-specific requirements.

    Args:
        work_dir: Root output directory produced by :func:`ir.compile`.
    """
    orch_dir = work_dir / "orchestration"
    if not orch_dir.exists():
        return
    for cpp_file in orch_dir.glob("*.cpp"):
        _add_headers_to_file(cpp_file)


def _add_headers_to_file(cpp_file: Path) -> None:
    """Insert missing ``runtime.h`` / ``<iostream>`` headers into *cpp_file*.

    Args:
        cpp_file: Path to a C++ source file that may be missing the headers.
    """
    content = cpp_file.read_text(encoding="utf-8")

    has_runtime_h = '#include "runtime.h"' in content
    has_iostream = "#include <iostream>" in content

    if has_runtime_h and has_iostream:
        return  # Nothing to do

    headers: list[str] = []
    if not has_runtime_h:
        headers.append('#include "runtime.h"')
    if not has_iostream:
        headers.append("#include <iostream>")

    # Find the first non-comment, non-blank line as the insertion point.
    lines = content.splitlines(keepends=True)
    insert_pos = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("//", "/*", "*")):
            insert_pos = i
            break

    header_block = "\n".join(headers) + "\n"
    if insert_pos > 0:
        header_block += "\n"

    lines.insert(insert_pos, header_block)
    cpp_file.write_text("".join(lines), encoding="utf-8")
