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

Provides :func:`run`, the main entry point for compiling a ``@pl.program``
and executing it on an Ascend NPU (or simulator).

Typical usage::

    import torch
    from pypto.runtime import run, RunConfig

    a = torch.full((128, 128), 2.0)
    b = torch.full((128, 128), 3.0)
    c = torch.zeros(128, 128)
    compiled = run(MyProgram, a, b, c, config=RunConfig(platform="a2a3sim"))
"""

import importlib.util
import os
import subprocess
import sys
import time
from ctypes import _SimpleCData
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core.passes import DiagnosticCheckSet, DiagnosticPhase

from .device_tensor import DeviceTensor

_OUTPUTS_DIR = Path("outputs")


def _load_golden_from_data_dir(out_dir: Path, output_names: set[str]) -> dict[str, torch.Tensor] | None:
    """Load pre-computed golden outputs from ``data/out/{name}.pt`` files.

    Returns ``None`` if the directory does not exist or any required file is
    missing, allowing the caller to fall back to live computation.
    """
    if not out_dir.is_dir():
        return None
    result = {}
    for name in output_names:
        pt_file = out_dir / f"{name}.pt"
        if not pt_file.exists():
            return None
        result[name] = torch.load(pt_file, weights_only=True)
    return result


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
        diagnostic_phase: Override the diagnostic phase gate for compilation.
            ``None`` uses the default (``PrePipeline``, or ``PYPTO_WARNING_LEVEL``
            env var). Setting to ``None`` silences warnings AND performance hints;
            finer-grained control uses ``disabled_diagnostics``.
        disabled_diagnostics: Set of diagnostic checks to disable during
            compilation (covers warnings and perf hints). ``None`` uses the
            default (``UnusedControlFlowResult`` disabled, perf hints enabled).
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
    diagnostic_phase: DiagnosticPhase | None = None
    disabled_diagnostics: DiagnosticCheckSet | None = None
    golden_data_dir: str | None = None

    def __post_init__(self) -> None:
        if self.platform not in ("a2a3sim", "a2a3", "a5sim", "a5"):
            raise ValueError(
                f"Invalid platform {self.platform!r}. Expected 'a2a3sim', 'a2a3', 'a5sim', or 'a5'."
            )
        # A caller-provided platform is the public source of truth for runtime
        # toolchain selection. Keep backend_type synchronized with it so codegen
        # and execution target the same architecture, rather than silently
        # rewriting the requested platform back to the default backend.
        if self.platform.startswith("a5"):
            self.backend_type = BackendType.Ascend950
        else:
            self.backend_type = BackendType.Ascend910B

        backend = _backend_core.get_backend_instance(self.backend_type)
        expected_arch = backend.get_handler().get_pto_target_arch()
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
    diagnostic_phase: DiagnosticPhase | None = None,
    disabled_diagnostics: DiagnosticCheckSet | None = None,
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
        diagnostic_phase: Override the diagnostic phase gate for compilation.
        disabled_diagnostics: Set of diagnostic checks to disable.
        profiling: If ``True``, enable compile profiling.
    """
    from pypto import ir  # noqa: PLC0415

    ir.compile(
        program,
        output_dir=str(work_dir),
        strategy=strategy,
        dump_passes=dump_passes,
        backend_type=backend_type,
        diagnostic_phase=diagnostic_phase,
        disabled_diagnostics=disabled_diagnostics,
        profiling=profiling,
    )
    _patch_orchestration_headers(work_dir)


def run(
    program: Any,
    *tensors: torch.Tensor,
    config: RunConfig | None = None,
) -> Any:
    """Compile *program* and execute it with *tensors* on device.

    This is the user-facing entry point for the compile-and-run workflow.
    No golden function, no TensorSpec — just define, compile, call.

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program``.
        *tensors: Positional ``torch.Tensor`` arguments matching the
            orchestration function's parameter order.  Pass no tensors
            for compile-only.
        config: Run configuration (platform, device, profiling, etc.).
            Uses default :class:`RunConfig` if ``None``.

    Returns:
        A :class:`~pypto.ir.compiled_program.CompiledProgram` that can
        be called again with new tensors.

    Example:
        >>> from pypto.runtime import run, RunConfig
        >>> a = torch.full((128, 128), 2.0)
        >>> b = torch.full((128, 128), 3.0)
        >>> c = torch.zeros(128, 128)
        >>> compiled = run(MyProgram, a, b, c, config=RunConfig(platform="a2a3sim"))
        >>> # Re-run with different inputs:
        >>> compiled(a2, b2, c2)
    """
    if config is None:
        config = RunConfig()

    from pypto import ir  # noqa: PLC0415

    compiled = ir.compile(
        program,
        output_dir=config.save_kernels_dir,
        strategy=config.strategy,
        backend_type=config.backend_type,
        dump_passes=config.dump_passes,
        diagnostic_phase=config.diagnostic_phase,
        disabled_diagnostics=config.disabled_diagnostics,
        platform=config.platform,
        profiling=config.compile_profiling,
    )

    if tensors and not config.codegen_only:
        compiled(*tensors, config=config)

    return compiled


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _execute_on_device(
    work_dir: Path,
    golden_path: Path,
    chip_callable: Any,
    runtime_name: str,
    platform: str,
    device_id: int,
    runtime_profiling: bool = False,
) -> None:
    """Load inputs, execute on device, and validate against golden.

    Shared execution logic used by both :func:`run` and the test harness
    (``test_runner.py``).  The caller is responsible for compiling binaries
    via ``compile_and_assemble`` and passing the result here.

    Tolerances (``RTOL``, ``ATOL``) are read from the generated ``golden.py``.

    Args:
        work_dir: Root output directory containing ``data/``, ``golden.py``, etc.
        golden_path: Path to the generated ``golden.py`` file.
        chip_callable: Pre-compiled ``ChipCallable`` from ``compile_and_assemble``.
        runtime_name: Runtime name from ``compile_and_assemble``.
        platform: Target execution platform.
        device_id: Hardware device index.
        runtime_profiling: If ``True``, enable runtime profiling.
    """
    from .device_runner import (  # noqa: PLC0415
        build_orch_args_from_inputs,
        execute_on_device,
        validate_golden,
    )

    # Load golden.py to get generate_inputs and compute_golden
    spec = importlib.util.spec_from_file_location("_golden", str(golden_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load golden.py from {golden_path}")
    golden_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden_module)

    # Generate inputs (loads from data/in/ when use_data_files golden.py)
    params: dict[str, str] = {"name": "Default"}
    result = golden_module.generate_inputs(params)

    output_names = set(getattr(golden_module, "__outputs__", []))
    orch_args, all_tensors, inputs, outputs = build_orch_args_from_inputs(result, output_names)

    # Load pre-computed golden from data/out/ if available
    out_dir = golden_path.parent / "data" / "out"
    golden_out = _load_golden_from_data_dir(out_dir, output_names)
    if golden_out is None:
        golden_out = {k: v.clone() for k, v in outputs.items()}
        golden_with_inputs = {**inputs, **golden_out}
        golden_module.compute_golden(golden_with_inputs, params)

    # Execute
    if runtime_profiling:
        pre_run_logs, device_log_dir, pre_run_perf_files = _snapshot_profiling_state(platform, device_id)

    execute_on_device(
        chip_callable,
        orch_args,
        platform,
        runtime_name,
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
    validate_golden(
        outputs,
        golden_out,
        rtol=getattr(golden_module, "RTOL", 1e-5),
        atol=getattr(golden_module, "ATOL", 1e-5),
    )


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
    pre_run_perf_files = set(_OUTPUTS_DIR.glob("l2_perf_records_*.json"))

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

    Moves ``l2_perf_records_*.json`` into ``work_dir/swimlane_data/`` and runs
    Simpler's swimlane converter via ``python -m simpler_setup.tools.swimlane_converter``
    (if available) to produce merged JSON.
    """
    swimlane_dir = work_dir / "swimlane_data"
    swimlane_dir.mkdir(parents=True, exist_ok=True)

    new_perf_files = set(_OUTPUTS_DIR.glob("l2_perf_records_*.json")) - pre_run_perf_files
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
    swimlane_dir: Path,
    perf_file: Path | None,
) -> None:
    """Run ``python -m simpler_setup.tools.swimlane_converter`` to generate ``merged_swimlane_*.json``.

    Output is written to *swimlane_dir* alongside the input ``l2_perf_records_*.json``.

    Args:
        work_dir: Directory containing ``kernel_config.py``.
        device_id: Hardware device index (fallback when no device log found).
        device_log_dir: CANN device log directory snapshotted before the run.
        pre_run_logs: Set of log files that existed before the run.
        swimlane_dir: Directory where swimlane JSON files are written.
        perf_file: Path to the ``l2_perf_records_*.json`` file produced by
            CodeRunner and already moved into *swimlane_dir*.  When ``None``,
            swimlane conversion is skipped.
    """
    converter_module = "simpler_setup.tools.swimlane_converter"
    try:
        spec = importlib.util.find_spec(converter_module)
    except ImportError:
        spec = None
    if spec is None:
        print(f"Module {converter_module} not found, skipping swimlane conversion")
        return

    if perf_file is None:
        print("No l2_perf_records_*.json found, skipping swimlane conversion")
        return

    kernel_config_path = work_dir / "kernel_config.py"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = swimlane_dir / f"merged_swimlane_{timestamp}.json"

    cmd = [
        sys.executable,
        "-m",
        converter_module,
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
        print(
            f"Swimlane converter module {converter_module!r} failed (exit {e.returncode}), "
            "no swimlane generated"
        )


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


# ---------------------------------------------------------------------------
# Compiled program execution (callable API)
# ---------------------------------------------------------------------------


def execute_compiled(
    work_dir: str | Path,
    args: list[torch.Tensor | DeviceTensor | _SimpleCData],
    *,
    platform: str,
    device_id: int,
    pto_isa_commit: str | None = None,
    runtime_profiling: bool = False,
    level: int = 2,
) -> None:
    """Execute a pre-compiled program with user-provided tensors and scalars.

    Reuses :func:`device_runner.compile_and_assemble` for binary compilation
    (with caching and parallel kernel compilation) and
    :func:`device_runner.execute_on_device` for device dispatch.  Host
    ``torch.Tensor`` outputs in *args* are modified in-place with device
    results; :class:`DeviceTensor` arguments are passed through to the
    runtime as ``child_memory=True`` (no H2D upload, no D2H readback).

    Args:
        work_dir: Root output directory from :func:`ir.compile`, containing
            ``kernels/``, ``orchestration/``, and ``kernel_config.py``.
        args: Ordered list of ``torch.Tensor`` (host),
            :class:`pypto.runtime.DeviceTensor` (worker-resident), or
            ``ctypes._SimpleCData`` scalar arguments matching the
            orchestration function's parameter order.
        platform: Target execution platform.
        device_id: Hardware device index.
        pto_isa_commit: Optional git commit to pin pto-isa clone.
        runtime_profiling: If ``True``, enable runtime profiling and
            generate swimlane JSON after execution.
        level: Hierarchy level. Forwarded to :func:`execute_on_device`,
            which currently only supports ``2``.
    """
    work_dir = Path(work_dir)

    # Ensure orchestration headers are patched (idempotent)
    _patch_orchestration_headers(work_dir)

    from .device_runner import (  # noqa: PLC0415
        ChipStorageTaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
        compile_and_assemble,
        execute_on_device,
        make_tensor_arg,  # pyright: ignore[reportAttributeAccessIssue]
        scalar_to_uint64,  # pyright: ignore[reportAttributeAccessIssue]
    )
    from .task_interface import (  # noqa: PLC0415
        ContinuousTensor,  # pyright: ignore[reportAttributeAccessIssue]
        torch_dtype_to_datatype,  # pyright: ignore[reportAttributeAccessIssue]
    )

    chip_callable, runtime_name = compile_and_assemble(work_dir, platform, pto_isa_commit)

    # Build orch args from user-provided tensors and scalars.
    orch_args = ChipStorageTaskArgs()
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if not arg.is_contiguous():
                raise ValueError(
                    f"Tensor at position {i} is not contiguous. "
                    f"Call .contiguous() before passing to execute_compiled()."
                )
            if arg.device.type != "cpu":
                raise ValueError(
                    f"Tensor at position {i} is on {arg.device}, expected CPU. "
                    f"Call .cpu() before passing to execute_compiled()."
                )
            orch_args.add_tensor(make_tensor_arg(arg))
        elif isinstance(arg, DeviceTensor):
            try:
                dt_enum = torch_dtype_to_datatype(arg.dtype)
            except KeyError as e:
                raise ValueError(f"Unsupported DeviceTensor dtype at position {i}: {arg.dtype}") from e
            orch_args.add_tensor(
                ContinuousTensor.make(
                    data=arg.data_ptr,
                    shapes=arg.shape,
                    dtype=dt_enum,
                    child_memory=True,
                )
            )
        elif isinstance(arg, _SimpleCData):
            orch_args.add_scalar(scalar_to_uint64(arg))
        else:
            raise TypeError(
                f"Argument at position {i} must be torch.Tensor, DeviceTensor or "
                f"ctypes scalar, got {type(arg).__name__}"
            )

    # Snapshot profiling state before execution
    if runtime_profiling:
        pre_run_logs, device_log_dir, pre_run_perf_files = _snapshot_profiling_state(platform, device_id)

    execute_on_device(
        chip_callable,
        orch_args,
        platform,
        runtime_name,
        device_id,
        level=level,
        enable_profiling=runtime_profiling,
    )

    # Collect swimlane data after execution
    if runtime_profiling:
        _collect_swimlane_data(
            work_dir, platform, device_id, pre_run_logs, device_log_dir, pre_run_perf_files
        )
