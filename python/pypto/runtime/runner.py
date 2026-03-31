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
import functools
import importlib
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from pypto import ir
from pypto.backend import BackendType, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy

from .golden_writer import write_golden
from .tensor_spec import TensorSpec

# ---------------------------------------------------------------------------
# Golden inputs pre-generation cache
# ---------------------------------------------------------------------------
# .pt files written by pregenerate_golden_inputs() (see test_runner.py) are
# the persistent cache.  These flags prevent re-patching CodeRunner in
# the same process.
_code_runner_patched: list[bool] = [False]
_binary_cache_patched: list[bool] = [False]
_OUTPUTS_DIR = Path("outputs")


@functools.lru_cache(maxsize=1)
def _get_simpler_stamp() -> str:
    """Return Simpler's current git commit (short hash) as a cache-key stamp.

    The stamp is used to namespace the global runtime binary cache so that
    stale binaries from an older Simpler version are never reused after an
    update.  Falls back to ``"unknown"`` when git is unavailable or
    ``SIMPLER_ROOT`` is not set.

    The value is computed once and cached in-process.
    """
    simpler_root = os.environ.get("SIMPLER_ROOT", "")
    if not simpler_root:
        return "unknown"
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=simpler_root,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


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


def _save_binary(data: bytes, path: Path) -> None:
    """Save compiled binary bytes to *path* atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _load_binary(path: Path) -> bytes | None:
    """Load compiled binary bytes from *path*. Returns ``None`` on miss."""
    if not path.exists():
        return None
    try:
        return path.read_bytes()
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
        enable_profiling: If ``True``, enable runtime profiling and generate
            ``swimlane.json`` after execution.
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
    enable_profiling: bool = False

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
        # Profiling requires kernel artefacts to be retained so swimlane files
        # can reference kernel_config.py.  Enable save_kernels automatically.
        if self.enable_profiling and not self.save_kernels:
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
    """
    ir.compile(
        program,
        output_dir=str(work_dir),
        strategy=strategy,
        dump_passes=dump_passes,
        backend_type=backend_type,
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
    3. Write a ``golden.py`` file from *tensor_specs* and *golden*.
    4. Invoke Simpler's ``CodeRunner`` to compile, load, execute, and validate.

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

    try:
        # 1. Set backend for code generation
        set_backend_type(config.backend_type)

        # 2. Compile: generates kernels/, orchestration/, kernel_config.py
        #    and patches orchestration headers
        compile_program(
            program,
            work_dir,
            strategy=config.strategy,
            backend_type=config.backend_type,
            dump_passes=config.dump_passes,
        )

        # 3. Write golden.py
        golden_path = work_dir / "golden.py"
        write_golden(tensor_specs, golden, golden_path, rtol=config.rtol, atol=config.atol)

        # 4. Execute via Simpler's CodeRunner
        _execute_on_device(
            work_dir,
            golden_path,
            config.platform,
            config.device_id,
            config.pto_isa_commit,
            config.enable_profiling,
        )

        return RunResult(passed=True, execution_time=time.time() - start_time)

    except Exception:
        return RunResult(
            passed=False,
            error=traceback.format_exc(),
            execution_time=time.time() - start_time,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _install_golden_inputs_patch(CodeRunner) -> None:
    """Monkey-patch CodeRunner.__init__ to serve generate_inputs and compute_golden from disk cache.

    Idempotent — safe to call multiple times.  For each new CodeRunner instance:

    - ``generate_inputs``: loads ``cache/{case}_inputs.pt`` when available,
      falls through to the original on a cache miss.
    - ``compute_golden``: copies cached output tensors from
      ``cache/{case}_golden.pt`` into the tensors dict when available,
      falls through to the original on a cache miss.

    Each ``torch.load`` produces fresh tensors, so no cloning is needed.
    """
    if _code_runner_patched[0]:
        return

    orig_init = CodeRunner.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        golden_path = self.golden_path  # Path, already resolved

        # --- patch generate_inputs -------------------------------------------
        orig_gen = self._golden_module.generate_inputs

        def _cached_gen(params):
            case_name = params.get("name", "Default")
            result = _load_inputs(_inputs_cache_file(golden_path, case_name))
            return result if result is not None else orig_gen(params)

        self._golden_module.generate_inputs = _cached_gen

        # --- patch compute_golden --------------------------------------------
        orig_compute = self._golden_module.compute_golden

        def _cached_compute(tensors, params):
            case_name = params.get("name", "Default")
            cached = _load_golden(_golden_cache_file(golden_path, case_name))
            if cached is not None:
                for name, val in cached.items():
                    if name in tensors:
                        tensors[name].copy_(val)
                return
            orig_compute(tensors, params)

        self._golden_module.compute_golden = _cached_compute

    CodeRunner.__init__ = _patched_init
    _code_runner_patched[0] = True


# Persistent runtime binary cache — shared across test cases and sessions.
# Root directory for persistent runtime binary cache.  Actual files live under
# a Simpler-version subdirectory (see _get_simpler_stamp()) so that stale
# binaries are automatically bypassed after a Simpler update.
_BINARY_RUNTIME_CACHE = (
    Path(__file__).parent.parent.parent.parent / "build_output" / "binary_cache" / "runtimes"
)


def _install_binary_cache_patch(KernelCompiler, RuntimeBuilder) -> None:
    """Monkey-patch KernelCompiler and RuntimeBuilder to serve compiled binaries from disk.

    Patches three methods with write-through caches:

    - ``KernelCompiler.compile_incore``: caches at
      ``work_dir/cache/incore_{core_type}_{stem}.bin``
      (derived from the kernel source path structure
      ``work_dir/kernels/{core_type}/{name}.cpp``).
    - ``KernelCompiler.compile_orchestration``: caches at
      ``work_dir/cache/orch_{stem}.bin``
      (derived from ``work_dir/orchestration/{name}.cpp``).
    - ``RuntimeBuilder.get_binaries``: caches at
      ``build_output/binary_cache/runtimes/{name}_{platform}_{host|aicpu|aicore}.bin``
      (global, shared across all test cases).

    Idempotent — safe to call multiple times. Cache miss triggers compilation
    and saves the result; subsequent calls serve from disk.
    """
    if _binary_cache_patched[0]:
        return

    RuntimeBinaries = getattr(sys.modules[RuntimeBuilder.__module__], "RuntimeBinaries")

    # --- KernelCompiler.compile_incore ---
    orig_incore = KernelCompiler.compile_incore

    def _patched_incore(
        self, source_path, core_type="aiv", pto_isa_root=None, extra_include_dirs=None, build_dir=None
    ):
        source = Path(source_path)
        # Only cache for the expected structure: work_dir/kernels/{core_type}/{name}.cpp
        if source.parent.parent.name == "kernels":
            cache_file = source.parent.parent.parent / "cache" / f"incore_{core_type}_{source.stem}.bin"
            cached = _load_binary(cache_file)
            if cached is not None:
                return cached
            result = orig_incore(self, source_path, core_type, pto_isa_root, extra_include_dirs, build_dir)
            _save_binary(result, cache_file)
            return result
        return orig_incore(self, source_path, core_type, pto_isa_root, extra_include_dirs, build_dir)

    KernelCompiler.compile_incore = _patched_incore

    # --- KernelCompiler.compile_orchestration ---
    orig_orch = KernelCompiler.compile_orchestration

    def _patched_orch(self, runtime_name, source_path, extra_include_dirs=None, build_dir=None):
        source = Path(source_path)
        # Only cache for the expected structure: work_dir/orchestration/{name}.cpp
        if source.parent.name == "orchestration":
            cache_file = source.parent.parent / "cache" / f"orch_{source.stem}.bin"
            cached = _load_binary(cache_file)
            if cached is not None:
                return cached
            result = orig_orch(self, runtime_name, source_path, extra_include_dirs, build_dir)
            _save_binary(result, cache_file)
            return result
        return orig_orch(self, runtime_name, source_path, extra_include_dirs, build_dir)

    KernelCompiler.compile_orchestration = _patched_orch

    # --- RuntimeBuilder.get_binaries ---
    orig_get_binaries = RuntimeBuilder.get_binaries

    def _patched_get_binaries(self, name, build=False):
        cache_dir = _BINARY_RUNTIME_CACHE / _get_simpler_stamp()
        host_file = cache_dir / f"{name}_{self.platform}_host.bin"
        aicpu_file = cache_dir / f"{name}_{self.platform}_aicpu.bin"
        aicore_file = cache_dir / f"{name}_{self.platform}_aicore.bin"
        if host_file.exists() and aicpu_file.exists() and aicore_file.exists():
            return RuntimeBinaries(host_path=host_file, aicpu_path=aicpu_file, aicore_path=aicore_file)
        result = orig_get_binaries(self, name, build=build)
        _save_binary(result.host_path.read_bytes(), host_file)
        _save_binary(result.aicpu_path.read_bytes(), aicpu_file)
        _save_binary(result.aicore_path.read_bytes(), aicore_file)
        return result

    RuntimeBuilder.get_binaries = _patched_get_binaries
    _binary_cache_patched[0] = True


def _execute_on_device(
    work_dir: Path,
    golden_path: Path,
    platform: str,
    device_id: int,
    pto_isa_commit: str | None = None,
    enable_profiling: bool = False,
) -> None:
    """Invoke Simpler's CodeRunner to compile, load, execute, and validate.

    Automatically adds SIMPLER_ROOT sub-paths to ``sys.path`` when the
    ``SIMPLER_ROOT`` environment variable is set (mirrors conftest.py behaviour).

    Args:
        work_dir: Root output directory produced by :func:`compile_program`,
            containing ``kernels/`` and ``orchestration/``.
        golden_path: Path to the generated ``golden.py`` file.
        platform: Target execution platform (``"a2a3sim"``, ``"a2a3"``,
            ``"a5sim"``, or ``"a5"``).
        device_id: Hardware device index.
        pto_isa_commit: If set, pin the pto-isa clone to this specific git
            commit (hash or tag).
        enable_profiling: If ``True``, enable runtime profiling and generate
            ``swimlane.json`` after execution.
    """
    simpler_root = os.environ.get("SIMPLER_ROOT")
    if simpler_root:
        for sub in ("examples/scripts", "python"):
            p = str(Path(simpler_root) / sub)
            if p not in sys.path:
                sys.path.insert(0, p)

    CodeRunner = importlib.import_module("code_runner").CodeRunner
    KernelCompiler = importlib.import_module("kernel_compiler").KernelCompiler
    RuntimeBuilder = importlib.import_module("runtime_builder").RuntimeBuilder

    _install_golden_inputs_patch(CodeRunner)
    _install_binary_cache_patch(KernelCompiler, RuntimeBuilder)

    # Snapshot existing device logs before run so we can identify the new one
    # (CANN writes device logs asynchronously after execution).
    # Device logs are only available on real hardware, not on simulators.
    pre_run_logs: set[Path] = set()
    device_log_dir: Path | None = None
    if enable_profiling and not platform.endswith("sim"):
        device_log_dir = _get_device_log_dir(device_id)
        if device_log_dir.exists():
            pre_run_logs = set(device_log_dir.glob("*.log"))

    # Snapshot existing perf_swimlane files so we can identify the new one
    # produced by CodeRunner (written to _OUTPUTS_DIR).
    pre_run_perf_files: set[Path] = set()
    if enable_profiling:
        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        pre_run_perf_files = set(_OUTPUTS_DIR.glob("perf_swimlane_*.json"))

    CodeRunner(
        kernels_dir=str(work_dir),
        golden_path=str(golden_path),
        platform=platform,
        device_id=device_id,
        clone_protocol="https",
        pto_isa_commit=pto_isa_commit,
        enable_profiling=enable_profiling,
    ).run()

    if enable_profiling:
        swimlane_dir = work_dir / "swimlane_data"
        swimlane_dir.mkdir(parents=True, exist_ok=True)

        # Move the newly created perf_swimlane_*.json into swimlane_data/.
        new_perf_files = set(_OUTPUTS_DIR.glob("perf_swimlane_*.json")) - pre_run_perf_files
        perf_file: Path | None = None
        if new_perf_files:
            perf_file = max(new_perf_files, key=lambda p: p.stat().st_mtime)
            dest = swimlane_dir / perf_file.name
            perf_file.rename(dest)
            perf_file = dest
            # Remove outputs/ if it is now empty (it is only a staging area).
            try:
                _OUTPUTS_DIR.rmdir()
            except OSError:
                pass

        if not platform.endswith("sim"):
            _generate_swimlane(
                work_dir, device_id, device_log_dir, pre_run_logs, simpler_root, swimlane_dir, perf_file
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
