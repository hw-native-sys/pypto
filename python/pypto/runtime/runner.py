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
import json
import shlex
import subprocess
import sys
import uuid
from collections.abc import Callable
from ctypes import _SimpleCData
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core.passes import DiagnosticCheckSet, DiagnosticPhase, MemoryPlanner

from .device_tensor import DeviceTensor

if TYPE_CHECKING:
    # Imported under TYPE_CHECKING only: ``distributed_compiled_program`` already
    # imports from ``pypto.runtime`` (``device_tensor``), so importing it eagerly
    # here would risk a partially-initialised ``pypto.runtime`` package at import
    # time. The field is plumbed through to ``ir.compile()`` lazily anyway.
    from pypto.ir.distributed_compiled_program import DistributedConfig


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


# Number of scope-depth rings the runtime sizes independently. Mirrors
# RUNTIME_ENV_RING_COUNT in the runtime's task_interface/call_config.h. A
# per-ring RunConfig override (list/tuple) must supply exactly this many entries.
_RING_DEPTH = 4


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
        enable_l2_swimlane: Capture per-task L2 perf records into
            ``<work_dir>/dfx_outputs/l2_swimlane_records.json``. On onboard
            platforms, ``swimlane_converter`` then produces
            ``merged_swimlane_*.json`` alongside it. Because the converter joins
            the timing against a task graph that only ``deps.json`` carries,
            enabling this on an onboard platform runs the kernel **twice**: a
            first dep_gen pass to capture ``deps.json`` (run in a subprocess so
            its device/SVM state is fully reclaimed before the timing pass — a
            failed capture is logged, not fatal), then a clean in-process
            swimlane pass (dep_gen off, since dep_gen collection perturbs the
            timing). Simulator platforms (``*sim``) stay single-pass
            and only emit ``l2_swimlane_records.json`` — the merged swimlane file
            is intentionally skipped because the simulator does not yet ship the
            task metadata the converter needs. Mirrors runtime's
            ``--enable-l2-swimlane`` flag.
        enable_dump_tensor: Per-task tensor dump **level** written into
            ``<work_dir>/dfx_outputs/args_dump/``. Inspect with
            ``python -m simpler_setup.tools.dump_viewer``. Mirrors
            ``--dump-tensor``:

            * ``0`` / ``False`` — off (no dump).
            * ``1`` / ``True`` — **partial**: only the tensors marked via the
              DSL marker ``pl.dump_tag(t)`` (or ``pl.submit(..., dumps=[...])``).
            * ``2`` — **full**: every task's tensor inputs and outputs.

            Full dump on a large workload can saturate the host-side dump
            collector (~42 MB/s drain rate) and get the AICPU killed by a STARS
            op-execute timeout — prefer partial (level ``1``) plus
            ``pl.dump_tag(t)`` to limit dump to specific tensors
            (simpler#844 selective tensor dump).
        enable_pmu: AICore PMU event type. ``0`` disables collection;
            ``>0`` enables and selects the event (``2`` = PIPE_UTILIZATION,
            ``4`` = MEMORY — see ``runtime/docs/dfx/pmu-profiling.md``).
            Output: ``<work_dir>/dfx_outputs/pmu.csv``. Mirrors
            ``--enable-pmu N``.
        enable_dep_gen: Capture PTO2 dependency edges into
            ``<work_dir>/dfx_outputs/deps.json``. Render to HTML on demand
            via ``python -m simpler_setup.tools.deps_to_graph``. Mirrors
            ``--enable-dep-gen``.
        enable_scope_stats: Capture per-scope heap / task_window / tensormap
            ring-fill peaks into
            ``<work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl``. Render to
            HTML on demand via ``runtime/tools/scope_stats_plot.py``. Mirrors
            ``--enable-scope-stats``.
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
        block_dim: Optional per-invocation override of the logical SPMD
            block count. ``None`` (default) defers to the value baked
            into ``kernel_config.py``'s ``RUNTIME_CONFIG`` at compile
            time (which itself may be unset, in which case the simpler
            runtime default applies). Set this when running the same
            compiled artifact on devices with different usable core
            counts.
        aicpu_thread_num: Optional per-invocation override of the AICPU
            thread count. Same precedence rules as ``block_dim``.
        ring_task_window: Optional per-invocation override of the runtime
            ring's task-slot window (number of in-flight tasks). Forwarded to
            ``CallConfig.runtime_env.ring_task_window``. A scalar (broadcast to
            all scope-depth rings) or a list/tuple of exactly 4 ints sizing
            rings 0..3 independently; each entry must be a power of two ``>= 4`` (a
            ``0`` list-entry leaves that ring at its default). ``None`` (default)
            leaves the field unset so the runtime falls back to its
            ``PTO2_RING_TASK_WINDOW`` env var or compile-time default.
        ring_heap: Optional per-invocation override of the per-ring output-heap
            size in **bytes**. Forwarded to ``CallConfig.runtime_env.ring_heap``.
            A scalar or a list/tuple of 4 ints (per ring 0..3); each entry must
            be a power of two ``>= 1024`` (a ``0`` list-entry leaves that ring at its
            default). ``None`` defers to the runtime's ``PTO2_RING_HEAP`` env var
            or compile-time default.
        ring_dep_pool: Optional per-invocation override of the per-ring
            dependency-edge pool capacity. Forwarded to
            ``CallConfig.runtime_env.ring_dep_pool``. A scalar or a list/tuple
            of 4 ints (per ring 0..3); each entry must be in ``[4, INT32_MAX]`` (a
            ``0`` list-entry leaves that ring at its default). ``None`` defers
            to the runtime's
            ``PTO2_RING_DEP_POOL`` env var or compile-time default.
        distributed_config: Optional L3 distributed-execution config, consumed
            only on the ``@pl.jit`` path. When set, it is forwarded to
            ``ir.compile()`` (via :func:`~pypto.jit.decorator._run_config_compile_kwargs`)
            so a HOST-level ``@pl.jit.host`` kernel compiles to a
            :class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`
            and dispatches per-rank. ``None`` (default) compiles a regular
            single-chip :class:`~pypto.ir.compiled_program.CompiledProgram`. The
            ``@pl.program`` :func:`run` entry point does not read this field; it
            forwards no compile-side overrides, so distributed ``@pl.program``
            execution is driven by ``ir.compile(..., distributed_config=...)``
            directly rather than through ``RunConfig``.
        analyze_auto_scopes_for_deps: If ``True``, enable compiler-derived task
            dependency analysis for AUTO runtime scopes during compilation.
            Defaults to ``False`` so existing runs keep using TensorMap fallback
            unless this behavior is explicitly requested.
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
    enable_l2_swimlane: bool = False
    enable_dump_tensor: int = 0  # 0=off, 1=partial (dump_tag-marked), 2=full
    enable_pmu: int = 0
    enable_dep_gen: bool = False
    enable_scope_stats: bool = False
    compile_profiling: bool = False
    diagnostic_phase: DiagnosticPhase | None = None
    disabled_diagnostics: DiagnosticCheckSet | None = None
    golden_data_dir: str | None = None
    block_dim: int | None = None
    aicpu_thread_num: int | None = None
    # Each accepts a scalar (broadcast to all scope-depth rings) or a list/tuple
    # of exactly ``_RING_DEPTH`` ints sizing rings 0..3 independently; a 0 entry
    # leaves that ring at its env/compile-time default. A tuple is normalized to
    # a list during validation.
    ring_task_window: int | list[int] | tuple[int, ...] | None = None
    ring_heap: int | list[int] | tuple[int, ...] | None = None
    ring_dep_pool: int | list[int] | tuple[int, ...] | None = None
    distributed_config: "DistributedConfig | None" = None
    analyze_auto_scopes_for_deps: bool = False

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

        # Any DFX flag requires kernel artefacts to be retained so the
        # ``<work_dir>/dfx_outputs/`` directory survives the run.
        if self.any_dfx_enabled() and not self.save_kernels:
            self.save_kernels = True

        # Validate ring-sizing overrides early so callers get a clear error here
        # rather than a deep failure inside the runtime's CallConfig::validate().
        self._validate_ring_overrides()

    def _validate_ring_overrides(self) -> None:
        """Validate the per-task ring-sizing overrides (scalar or per-ring list).

        Mirrors the constraints enforced by the runtime's
        ``RuntimeEnv::validate()``. ``None`` means "unset" and is always allowed
        (the runtime falls back to env var / compile-time default).

        A scalar is broadcast to every ring. A list sizes each scope-depth ring
        independently and must have exactly ``_RING_DEPTH`` entries; a ``0``
        list-entry leaves that ring at its env/compile-time default — the same
        fall-through the runtime allows. Scalars do not accept ``0`` (use
        ``None`` to leave the whole field unset).
        """

        def _is_int(v: object) -> bool:
            # bool is an int subtype; reject it so True/False can't masquerade
            # as a ring size. Guards the pow2 bitwise ops below from TypeError
            # on floats and keeps the failure a clear ValueError.
            return isinstance(v, int) and not isinstance(v, bool)

        def _is_pow2(v: int) -> bool:
            return v > 0 and (v & (v - 1)) == 0

        # (field, human-readable constraint, scalar predicate). A scalar is
        # validated directly; a list/tuple must have exactly ``_RING_DEPTH``
        # entries and every entry obeys the predicate (a ``0`` entry is the
        # runtime's "leave this ring at its default" sentinel).
        specs = (
            ("ring_task_window", "be a power of 2 >= 4", lambda v: _is_int(v) and _is_pow2(v) and v >= 4),
            (
                "ring_heap",
                "be a power of 2 >= 1024 (bytes per ring)",
                lambda v: _is_int(v) and _is_pow2(v) and v >= 1024,
            ),
            ("ring_dep_pool", "be in [4, INT32_MAX]", lambda v: _is_int(v) and 4 <= v <= 2**31 - 1),
        )
        for name, phrase, ok in specs:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                value = list(value)  # normalize tuple -> list for downstream use
                if len(value) != _RING_DEPTH:
                    raise ValueError(
                        f"{name} must have exactly {_RING_DEPTH} entries "
                        f"(one per scope-depth ring), got {len(value)}"
                    )
                for v in value:
                    # Reject non-ints (incl. bool) before the 0 sentinel check so
                    # ``False`` can't masquerade as "leave at default".
                    if not _is_int(v) or (v != 0 and not ok(v)):
                        raise ValueError(f"{name} entries must {phrase} (or 0 to keep default), got {v!r}")
                setattr(self, name, value)
            elif not ok(value):
                raise ValueError(f"{name} must {phrase}, got {value!r}")

    def any_dfx_enabled(self) -> bool:
        """Return ``True`` when at least one DFX flag is enabled.

        DFX (Design For X) covers the five runtime diagnostic sub-features
        carried on :class:`~simpler.task_interface.CallConfig`:
        L2 swimlane, tensor dump, PMU, dep_gen and scope_stats. They are
        independent toggles that share an output directory.
        """
        return (
            self.enable_l2_swimlane
            or self.enable_dump_tensor > 0
            or self.enable_pmu > 0
            or self.enable_dep_gen
            or self.enable_scope_stats
        )


@dataclass
class RunResult:
    """Result of a program run or harness test execution.

    Attributes:
        passed: ``True`` if the program executed and results matched the golden
            reference within the configured tolerances.
        test_name: Optional test case name.  Set by the harness when running
            a named test case; ``None`` for direct :func:`run` calls.
        error: Human-readable error message when ``passed`` is ``False``.
        execution_time: Python wall-clock time in seconds for the full run
            (compile + execute + validate). This mixes host-side compile/golden
            overhead with the actual dispatch, so it cannot isolate device time
            — read per-run device/host timing from the runtime's ``[STRACE]``
            log markers (simpler PR #1177) instead.
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
    analyze_auto_scopes_for_deps: bool = False,
    memory_planner: MemoryPlanner | None = None,
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
        analyze_auto_scopes_for_deps: If ``True``, enable compiler-derived task
            dependency analysis for AUTO runtime scopes.
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
        analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
        memory_planner=memory_planner,
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
        analyze_auto_scopes_for_deps=config.analyze_auto_scopes_for_deps,
    )

    if tensors and not config.codegen_only:
        compiled(*tensors, config=config)

    return compiled


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DfxOpts:
    """Bundle of runtime DFX toggles passed through the execute pipeline.

    Each field maps to the same-named ``CallConfig`` member on the runtime
    side. ``any()`` answers whether the runtime needs an ``output_prefix``.
    """

    enable_l2_swimlane: bool = False
    enable_dump_tensor: int = 0  # 0=off, 1=partial, 2=full
    enable_pmu: int = 0
    enable_dep_gen: bool = False
    enable_scope_stats: bool = False

    def any(self) -> bool:
        return (
            self.enable_l2_swimlane
            or self.enable_dump_tensor > 0
            or self.enable_pmu > 0
            or self.enable_dep_gen
            or self.enable_scope_stats
        )

    @classmethod
    def from_run_config(cls, cfg: "RunConfig") -> "_DfxOpts":
        return cls(
            enable_l2_swimlane=cfg.enable_l2_swimlane,
            enable_dump_tensor=cfg.enable_dump_tensor,
            enable_pmu=cfg.enable_pmu,
            enable_dep_gen=cfg.enable_dep_gen,
            enable_scope_stats=cfg.enable_scope_stats,
        )


def _execute_dfx_passes(
    run_pass: Callable[["_DfxOpts"], None],
    capture_deps: Callable[[], None],
    dfx: "_DfxOpts",
    platform: str,
) -> None:
    """Drive device execution, splitting into two passes when swimlane is on.

    The runtime swimlane converter joins per-task timing against a task graph
    that only ``deps.json`` (a dep_gen capture) carries — the device hot path no
    longer records per-task fanout. Because dep_gen collection perturbs timing,
    the graph and the clean timing must come from separate runs (the converter's
    documented "capture the graph once, time many times" workflow). So when
    swimlane is requested on an onboard platform we run:

      * Graph pass — dep_gen only, producing ``deps.json``. Run in a **separate
        subprocess** (*capture_deps*): the runtime's per-run finalize does not
        reliably reclaim the SVM host-register mappings the DFX collectors
        allocate, so a second DFX run in the same process hits the registration
        cap (``halHostRegister`` rc 8). A child process fully reclaims that
        state on exit. Best-effort — a failed capture is logged, not fatal.
      * Timing pass — swimlane (plus any other timing DFX), dep_gen off,
        producing the clean ``l2_swimlane_records.json``. Runs in-process.

    Both passes write into the same ``output_prefix`` (the subprocess is pointed
    at the same ``dfx_outputs/``), so the converter finds ``deps.json`` and the
    records side by side.

    Simulator platforms (``*sim``) stay single-pass: swimlane conversion is
    skipped there anyway (the simulator does not ship the task metadata the
    converter needs), so a second run buys nothing.

    Args:
        run_pass: Executes one in-process device run with the given DFX flags.
            Call-site closure over the static kwargs.
        capture_deps: Captures ``deps.json`` in a subprocess (dep_gen only).
            Call-site closure; invoked once before the timing pass.
        dfx: The DFX toggles the caller requested.
        platform: Target execution platform (used only to detect ``*sim``).
    """
    if not dfx.enable_l2_swimlane or platform.endswith("sim"):
        run_pass(dfx)
        return

    # The two passes look like a double run, so announce what each is for.
    print(
        "[swimlane] L2 swimlane enabled -> running the kernel twice "
        "(dep_gen perturbs timing, so the graph and the timing are captured separately):"
    )

    # Graph pass: capture deps.json in a subprocess so its SVM registrations are
    # fully reclaimed before the in-process timing pass registers its own.
    print(
        "[swimlane] run 1/2: capturing the task dependency graph (deps.json) in a subprocess; "
        "its timing is discarded."
    )
    capture_deps()

    # Timing pass: clean per-task timing for the lanes (dep_gen forced off so it
    # does not perturb the measurement). This is the timing we surface.
    print("[swimlane] run 2/2: measuring clean per-task timing (this run's numbers are the ones reported).")
    return run_pass(replace(dfx, enable_dep_gen=False))


def _load_golden_module(golden_path: "Path", module_name: str = "_golden") -> Any:
    """Import a generated ``golden.py`` from *golden_path* as a fresh module.

    Shared by :func:`_execute_on_device` and the dep_gen subprocess so the load
    semantics (and the error message) stay in one place.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(golden_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load golden.py from {golden_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_args_spec(
    args: "list[torch.Tensor | DeviceTensor | _SimpleCData]",
    save_dir: Path,
    run_id: str = "",
) -> list[dict]:
    """Describe orch arguments for the dep_gen subprocess (see below).

    The captured task graph can be routed by tensor *values*, not just scalars
    (e.g. paged-attention ``block_tables`` / ``seq_lens``), so we preserve real
    data wherever it can cross the process boundary:

    * Host ``torch.Tensor`` — saved verbatim to *save_dir* and reloaded in the
      child, so data-as-control inputs route the same graph.
    * :class:`DeviceTensor` — device-resident, unreachable from a fresh child
      process, so recorded as shape + dtype and rebuilt as a zero tensor. If a
      device-resident tensor routes the graph the capture is approximate.
    * ctypes scalar — value preserved exactly.

    *run_id* (when given) is woven into the saved tensor filenames so concurrent
    captures sharing one *save_dir* do not overwrite each other's args. Mirrors
    the type dispatch in :func:`_coerced_to_orch_args`.
    """
    prefix = f"_dep_gen_arg_{run_id}_" if run_id else "_dep_gen_arg_"
    spec: list[dict] = []
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            path = save_dir / f"{prefix}{i}.pt"
            torch.save(arg.detach().contiguous().cpu(), path)
            spec.append({"kind": "tensor_file", "path": str(path)})
        elif isinstance(arg, DeviceTensor):
            dtype_name = str(arg.dtype).replace("torch.", "")
            spec.append({"kind": "tensor_zeros", "shape": list(arg.shape), "dtype": dtype_name})
        elif isinstance(arg, _SimpleCData):
            spec.append({"kind": "scalar", "ctype": type(arg).__name__, "value": arg.value})
        else:
            raise TypeError(
                f"Cannot describe argument {i} of type {type(arg).__name__} for dep_gen capture; "
                f"expected torch.Tensor, DeviceTensor, or ctypes scalar."
            )
    return spec


# Upper bound for the best-effort dep_gen graph-capture subprocess: it compiles
# (cached) and runs the kernel once, so generous, but bounded so a stalled run
# never hangs the swimlane timing pass.
_DEP_GEN_CAPTURE_TIMEOUT_S = 900


def _capture_deps_subprocess(spec: dict, dfx_dir: Path, run_id: str = "") -> None:
    """Capture ``deps.json`` for swimlane in a child process (best-effort).

    A child process is used so the SVM host-register mappings the dep_gen
    collector allocates are fully reclaimed on exit, before the in-process
    swimlane pass registers its own (see :func:`_execute_dfx_passes`). The spec
    tells :mod:`pypto.runtime._dep_gen_capture` how to rebuild the orch args.
    *run_id* (when given) uniquifies the spec filename so concurrent captures
    sharing one *dfx_dir* do not collide.

    Failure (non-zero exit or timeout) is logged, not raised: the swimlane pass
    still runs, just without a captured graph (lanes degrade to anonymous
    ``task(rXtY)`` with no arrows).
    """
    dfx_dir.mkdir(parents=True, exist_ok=True)
    spec_path = dfx_dir / (f"_dep_gen_spec_{run_id}.json" if run_id else "_dep_gen_spec.json")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    cmd = [sys.executable, "-m", "pypto.runtime._dep_gen_capture", str(spec_path)]
    try:
        # Bounded so a stalled dep_gen run can never hang the timing pass.
        subprocess.run(cmd, check=True, timeout=_DEP_GEN_CAPTURE_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        detail = (
            f"timed out after {_DEP_GEN_CAPTURE_TIMEOUT_S}s"
            if isinstance(e, subprocess.TimeoutExpired)
            else f"exit {e.returncode}"
        )
        # Keep the spec + staged arg tensors so a failed capture can be re-run
        # and debugged by hand.
        print(
            f"dep_gen graph capture subprocess failed ({detail}); the swimlane will "
            f"render without dependency arrows / resolved kernel names (expected "
            f"{dfx_dir / 'deps.json'}). Inputs kept at {spec_path} for re-run."
        )
        return
    # Success: drop the transient staged inputs (argspec mode saves the full
    # host tensors, which can run to gigabytes).
    for entry in spec.get("args", []):
        if entry.get("kind") == "tensor_file":
            Path(entry["path"]).unlink(missing_ok=True)
    spec_path.unlink(missing_ok=True)


def _coerced_to_orch_args(
    coerced: list[torch.Tensor | DeviceTensor | _SimpleCData],
) -> Any:
    """Pack a coerced positional arg list into a simpler ``ChipStorageTaskArgs``.

    Two-pass dispatch (tensors then scalars) to respect simpler's add-order
    constraint: ``ChipStorageTaskArgs`` requires all ``add_tensor`` calls to
    precede any ``add_scalar`` call. Codegen addresses tensors/scalars from
    independent pools (``orch_args.tensor(i)`` / ``orch_args.scalar(i)``), so
    cross-pool order is irrelevant for the binary ABI — only within-pool
    order matters, and we preserve it.

    Used by both :func:`execute_compiled` and the extraction path on
    :class:`pypto.ir.CompiledProgram` (``build_orch_args``).
    """
    from .device_runner import (  # noqa: PLC0415
        ChipStorageTaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
        make_tensor_arg,  # pyright: ignore[reportAttributeAccessIssue]
        scalar_to_uint64,  # pyright: ignore[reportAttributeAccessIssue]
    )
    from .task_interface import (  # noqa: PLC0415
        device_tensor_to_tensor,  # pyright: ignore[reportAttributeAccessIssue]
    )

    orch_args = ChipStorageTaskArgs()
    for i, arg in enumerate(coerced):
        if isinstance(arg, torch.Tensor):
            if not arg.is_contiguous():
                raise ValueError(
                    f"Tensor at position {i} is not contiguous. "
                    f"Call .contiguous() before packing into orch_args."
                )
            if arg.device.type != "cpu":
                raise ValueError(
                    f"Tensor at position {i} is on {arg.device}, expected CPU. "
                    f"Call .cpu() before packing into orch_args."
                )
            orch_args.add_tensor(make_tensor_arg(arg))
        elif isinstance(arg, DeviceTensor):
            try:
                orch_args.add_tensor(device_tensor_to_tensor(arg))
            except ValueError as e:
                raise ValueError(f"At position {i}: {e}") from e
        elif isinstance(arg, _SimpleCData):
            continue  # handled below
        else:
            raise TypeError(
                f"Argument at position {i} must be torch.Tensor, DeviceTensor or "
                f"ctypes scalar, got {type(arg).__name__}"
            )
    for arg in coerced:
        if isinstance(arg, _SimpleCData):
            orch_args.add_scalar(scalar_to_uint64(arg))
    return orch_args


def _apply_ring_overrides(call_config: Any, run_config: "RunConfig") -> None:
    """Overlay a :class:`RunConfig`'s per-task ring sizing onto a ``CallConfig``.

    Each ``runtime_env`` field is left at its ``0`` default when the matching
    ``RunConfig`` override is ``None``, so the runtime applies its own
    ``PTO2_RING_*`` env var / compile-time fallback. Shared by the L2
    (:func:`_build_call_config`) and L3
    (:func:`pypto.runtime.distributed_runner._make_call_config`) dispatch paths
    so both transcribe ring sizing identically.

    Args:
        call_config: A simpler ``CallConfig`` (mutated in place).
        run_config: The :class:`RunConfig` whose ``ring_*`` overrides are copied.
    """
    if run_config.ring_task_window is not None:
        call_config.runtime_env.ring_task_window = run_config.ring_task_window
    if run_config.ring_heap is not None:
        call_config.runtime_env.ring_heap = run_config.ring_heap
    if run_config.ring_dep_pool is not None:
        call_config.runtime_env.ring_dep_pool = run_config.ring_dep_pool


def _build_call_config(
    run_config: "RunConfig",
    *,
    runtime_config: dict[str, Any],
    block_dim_override: int | None = None,
    aicpu_thread_num_override: int | None = None,
    dfx_dir: Path | None = None,
) -> Any:
    """Translate a pypto :class:`RunConfig` into a simpler ``CallConfig``.

    Precedence for ``block_dim`` and ``aicpu_thread_num``:
    explicit *override* > ``run_config`` field > ``runtime_config`` baked
    into ``kernel_config.py``. When all three are unset the field is left
    untouched on ``CallConfig`` so the simpler runtime's own default applies.

    DFX flags are copied straight from *run_config*; *dfx_dir* — when given —
    becomes ``output_prefix``. Callers that enable DFX flags are responsible
    for creating *dfx_dir* before the run (simpler's ``validate()`` rejects
    DFX-enabled calls without a valid prefix).
    """
    from .task_interface import (  # noqa: PLC0415
        CallConfig,  # pyright: ignore[reportAttributeAccessIssue]
    )

    cfg = CallConfig()

    bd = block_dim_override if block_dim_override is not None else run_config.block_dim
    bd = bd if bd is not None else runtime_config.get("block_dim")
    if bd is not None:
        cfg.block_dim = bd

    at = aicpu_thread_num_override if aicpu_thread_num_override is not None else run_config.aicpu_thread_num
    at = at if at is not None else runtime_config.get("aicpu_thread_num")
    if at is not None:
        cfg.aicpu_thread_num = at

    cfg.enable_l2_swimlane = run_config.enable_l2_swimlane
    cfg.enable_dump_tensor = run_config.enable_dump_tensor
    cfg.enable_pmu = run_config.enable_pmu
    cfg.enable_dep_gen = run_config.enable_dep_gen
    cfg.enable_scope_stats = run_config.enable_scope_stats

    # Per-task ring sizing: leave the runtime_env field at its 0 default when
    # unset so the runtime applies its own PTO2_RING_* / compile-time fallback.
    _apply_ring_overrides(cfg, run_config)

    if dfx_dir is not None:
        cfg.output_prefix = str(dfx_dir)
    return cfg


def _execute_on_device(
    work_dir: Path,
    golden_path: Path,
    chip_callable: Any,
    runtime_name: str,
    platform: str,
    device_id: int,
    dfx: _DfxOpts = _DfxOpts(),
    validate: bool = True,
    actual_out_dir: "Path | None" = None,
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
        dfx: Runtime DFX toggles. When any flag is enabled the artefacts
            land under ``<work_dir>/dfx_outputs/`` and the matching
            post-run converter is invoked.
    """
    from .device_runner import (  # noqa: PLC0415
        build_orch_args_from_inputs,
        execute_on_device,
        validate_golden,
    )

    # Load golden.py to get generate_inputs and compute_golden
    golden_module = _load_golden_module(golden_path)

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
    dfx_dir: Path | None = None
    if dfx.any():
        dfx_dir = work_dir / "dfx_outputs"
        dfx_dir.mkdir(parents=True, exist_ok=True)

    def _run_pass(pass_dfx: "_DfxOpts") -> None:
        execute_on_device(
            chip_callable,
            orch_args,
            platform,
            runtime_name,
            device_id,
            output_prefix=str(dfx_dir) if dfx_dir is not None else None,
            enable_l2_swimlane=pass_dfx.enable_l2_swimlane,
            enable_dump_tensor=pass_dfx.enable_dump_tensor,
            enable_pmu=pass_dfx.enable_pmu,
            enable_dep_gen=pass_dfx.enable_dep_gen,
            enable_scope_stats=pass_dfx.enable_scope_stats,
        )

    def _capture_deps() -> None:
        # Harness path: the child regenerates inputs deterministically from
        # golden.py, so the captured graph is faithful (no zero-tensor proxy).
        assert dfx_dir is not None  # swimlane-on implies dfx.any() -> dfx_dir set
        run_id = uuid.uuid4().hex
        _capture_deps_subprocess(
            {
                "mode": "golden",
                "golden_path": str(golden_path),
                "work_dir": str(work_dir),
                "platform": platform,
                "device_id": device_id,
                "dfx_dir": str(dfx_dir),
                "pto_isa_commit": None,
                "level": 2,
            },
            dfx_dir,
            run_id,
        )

    # When swimlane is on (onboard), capture deps.json in a subprocess first,
    # then run the clean-timing swimlane pass in-process. Collection uses the
    # original ``dfx`` so the converter joins the sibling ``deps.json`` and the
    # deps-render hint fires only when the user explicitly asked for dep_gen.
    _execute_dfx_passes(_run_pass, _capture_deps, dfx, platform)

    if dfx_dir is not None:
        _collect_dfx_artifacts(dfx_dir, platform, dfx)

    # Persist actual device outputs (tolerance-independent) for callers that
    # validate separately with the test's real tolerance — the "split
    # execute/validate" path used by the task-submit harness, where the device
    # run is eager/parallel and ``TestRunner.run`` does the allclose later.
    if actual_out_dir is not None:
        from .golden_writer import _save_data_files  # noqa: PLC0415

        _save_data_files(outputs, actual_out_dir)

    # Validate in-process unless the caller defers it.
    if validate:
        validate_golden(
            outputs,
            golden_out,
            rtol=getattr(golden_module, "RTOL", 1e-5),
            atol=getattr(golden_module, "ATOL", 1e-5),
        )


def validate_persisted_outputs(work_dir: Path, rtol: float, atol: float) -> None:
    """Validate persisted device outputs against the golden with a given tolerance.

    The counterpart to ``_execute_on_device(..., validate=False,
    actual_out_dir=...)``: the device run (tolerance-independent) persisted the
    actual outputs under ``data/actual/``; this compares them against the
    pre-computed golden under ``data/out/`` using *rtol*/*atol* — letting the
    harness apply each test's real tolerance after an eager, validation-free
    device run. Raises ``AssertionError`` on mismatch.
    """
    from .device_runner import validate_golden  # noqa: PLC0415

    golden_module = _load_golden_module(work_dir / "golden.py")
    output_names = set(getattr(golden_module, "__outputs__", []))
    actual = _load_golden_from_data_dir(work_dir / "data" / "actual", output_names)
    expected = _load_golden_from_data_dir(work_dir / "data" / "out", output_names)
    if actual is None or expected is None:
        raise AssertionError(
            f"validate_persisted_outputs: missing actual/expected outputs under {work_dir}/data "
            f"(actual={'ok' if actual else 'missing'}, expected={'ok' if expected else 'missing'})"
        )
    validate_golden(actual, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# DFX artefact collection
# ---------------------------------------------------------------------------


def _collect_dfx_artifacts(
    dfx_dir: Path,
    platform: str,
    dfx: "_DfxOpts",
) -> None:
    """Dispatch post-run DFX converters per enabled flag.

    The runtime writes each artefact directly into *dfx_dir* (the
    ``CallConfig.output_prefix`` passed at submit). Each branch below is
    independent and skips silently when its artefact is missing — a
    partial DFX run (e.g. only ``enable_dump_tensor``) must not crash on
    the swimlane converter looking for ``l2_swimlane_records.json``.
    """
    # Synthesise the func_id→name map the profiling tools need for readable
    # labels. simpler's SceneTest harness writes this itself; pypto does not
    # use SceneTest, so we derive it from ``kernel_config.py`` and drop it next
    # to the records. ``deps_to_graph`` auto-discovers ``name_map_*.json`` in
    # the same directory, and ``swimlane_converter`` is pointed at it below via
    # ``--func-names``. Written whenever swimlane or dep_gen is enabled (the two
    # consumers); harmless no-op when no kernel names are available.
    name_map_path: Path | None = None
    if dfx.enable_l2_swimlane or dfx.enable_dep_gen:
        name_map_path = _write_name_map(dfx_dir.parent, dfx_dir)

    if dfx.enable_l2_swimlane and (dfx_dir / "l2_swimlane_records.json").exists():
        # Swimlane conversion is onboard-only — the simulator produces
        # ``l2_swimlane_records.json`` but does not yet ship the matching
        # task metadata the converter expects.
        if not platform.endswith("sim"):
            _generate_swimlane(
                dfx_dir.parent,
                dfx_dir,
                dfx_dir / "l2_swimlane_records.json",
                func_names=name_map_path,
            )
        else:
            print(
                "Skipping swimlane conversion on simulator: "
                "merged_swimlane_*.json is only generated for onboard runs."
            )

    if dfx.enable_dep_gen and (dfx_dir / "deps.json").exists():
        # ``deps_to_graph`` is an offline post-processing tool; leave the
        # artefact in place and point the user at the rendering command.
        # Doing it inline on hot path risks hanging the run on large graphs
        # (Graphviz ``dot`` is O(N²~N³) and has SIGKILL'd taskqueue jobs).
        # ``shlex.quote`` keeps the printed command copy-pasteable even when
        # the path contains spaces or other shell metacharacters.
        deps_path = shlex.quote(str(dfx_dir / "deps.json"))
        print(
            f"deps.json written to {deps_path} — render with:\n"
            f"  python -m simpler_setup.tools.deps_to_graph {deps_path}\n"
            f"  # for large graphs, pass --engine (default 'dot' works for <500 nodes):\n"
            f"  python -m simpler_setup.tools.deps_to_graph {deps_path} --engine sfdp\n"
            f"  # --engine choices: dot | sfdp | fdp | neato | circo | twopi"
        )

    if dfx.enable_dump_tensor > 0 and (dfx_dir / "args_dump" / "args_dump.json").exists():
        # ``dump_viewer`` is interactive; leave the artefact in place and
        # point the user at the inspection command.
        print(
            f"args_dump written to {dfx_dir / 'args_dump'} — inspect with: "
            f"python -m simpler_setup.tools.dump_viewer "
            f"{dfx_dir / 'args_dump'}"
        )

    if dfx.enable_pmu > 0 and (dfx_dir / "pmu.csv").exists():
        print(f"PMU CSV written to: {dfx_dir / 'pmu.csv'}")

    # scope_stats writes a ``scope_stats/`` subdir (sibling of the flat
    # artefacts above), not a top-level file — the collector groups the
    # JSONL alongside any future per-scope companions. ``scope_stats_plot``
    # is an offline renderer; leave the JSONL in place and point the user
    # at the HTML-report command rather than running Graphviz-style layout
    # on the hot path.
    scope_stats_jsonl = dfx_dir / "scope_stats" / "scope_stats.jsonl"
    if dfx.enable_scope_stats and scope_stats_jsonl.exists():
        jsonl_path = shlex.quote(str(scope_stats_jsonl))
        print(
            f"scope_stats written to {jsonl_path} — render an HTML report with:\n"
            f"  python runtime/tools/scope_stats_plot.py {jsonl_path}"
        )


def _write_name_map(work_dir: Path, dfx_dir: Path) -> Path | None:
    """Synthesise a ``name_map_*.json`` in *dfx_dir* from ``kernel_config.py``.

    The profiling tools render human-readable kernel names (``QK(rXtY)``
    instead of the anonymous ``task(rXtY)``) only when a name map sits next to
    the records: ``swimlane_converter`` consumes it via ``--func-names`` and
    ``deps_to_graph`` auto-discovers any sibling ``name_map_*.json``. simpler's
    SceneTest harness writes this file itself, but pypto does not use SceneTest,
    so we build the same ``callable_id_to_name`` mapping from the
    ``func_id``/``name`` fields already emitted into ``kernel_config.py``.

    Args:
        work_dir: Directory containing ``kernel_config.py``.
        dfx_dir: ``dfx_outputs`` directory where the name map is written
            (alongside ``l2_swimlane_records.json`` / ``deps.json``).

    Returns:
        The written path, or ``None`` when ``kernel_config.py`` is absent or
        carries no named kernels (the tools then fall back to default labels).
    """
    kernel_config_path = work_dir / "kernel_config.py"
    if not kernel_config_path.exists():
        return None
    try:
        from simpler_setup.tools.swimlane_converter import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            load_kernel_config,
        )

        func_id_to_name = load_kernel_config(str(kernel_config_path))
    except Exception as e:  # noqa: BLE001 - best-effort diagnostics, never fatal
        print(f"Skipping name_map generation ({type(e).__name__}: {e})")
        return None
    if not func_id_to_name:
        return None

    # level 2 = orchestration entry + incore kernels: the only level pypto's
    # user-API compiles to (device_runner rejects level != 2). orchestrator_name
    # stays None — the C++ orch entry has no SceneTest-style display name.
    name_map = {
        "level": 2,
        "orchestrator_name": None,
        "callable_id_to_name": func_id_to_name,
    }
    out_path = dfx_dir / f"name_map_{work_dir.name}.json"
    out_path.write_text(json.dumps(name_map, indent=2), encoding="utf-8")
    return out_path


def _generate_swimlane(
    work_dir: Path,
    swimlane_dir: Path,
    perf_file: Path | None,
    func_names: Path | None = None,
) -> None:
    """Run ``python -m simpler_setup.tools.swimlane_converter`` to generate ``merged_swimlane_*.json``.

    Output is written to *swimlane_dir* alongside the input ``l2_swimlane_records_*.json``.

    Args:
        work_dir: Directory containing ``kernel_config.py``.
        swimlane_dir: Directory where swimlane JSON files are written.
        perf_file: Path to the ``l2_swimlane_records_*.json`` file produced by
            CodeRunner and already moved into *swimlane_dir*.  When ``None``,
            swimlane conversion is skipped.
        func_names: Optional ``name_map_*.json`` (see :func:`_write_name_map`)
            passed to the converter via ``--func-names``. Takes precedence over
            the ``-k kernel_config.py`` fallback for label resolution.
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
        print("No l2_swimlane_records_*.json found, skipping swimlane conversion")
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
    # ``--func-names`` (the synthesised name_map) takes precedence over ``-k``
    # for label resolution; ``-k`` stays as the fallback when no map was written.
    if func_names is not None:
        cmd += ["--func-names", str(func_names)]

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


def execute_compiled(  # noqa: PLR0913
    work_dir: str | Path,
    args: list[torch.Tensor | DeviceTensor | _SimpleCData],
    *,
    platform: str,
    device_id: int,
    pto_isa_commit: str | None = None,
    dfx: _DfxOpts = _DfxOpts(),
    level: int = 2,
    block_dim: int | None = None,
    aicpu_thread_num: int | None = None,
    analyze_auto_scopes_for_deps: bool = False,
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
        dfx: Runtime DFX toggles. When any flag is enabled the artefacts
            land under ``<work_dir>/dfx_outputs/`` and the matching
            post-run converter is invoked.
        level: Hierarchy level. Forwarded to :func:`execute_on_device`,
            which currently only supports ``2``.
        block_dim: Optional override of the logical SPMD block count.
            When ``None`` (default), the value baked into
            ``kernel_config.py``'s ``RUNTIME_CONFIG`` is used; if that
            is also unset, simpler's runtime default applies (simpler
            validates against device capacity and raises a clear error
            on over-capacity requests). A caller-supplied value takes
            precedence over ``RUNTIME_CONFIG``.
        aicpu_thread_num: Optional override of the AICPU thread count;
            same precedence rules as ``block_dim``.
        analyze_auto_scopes_for_deps: Compile-side compatibility option.
            Accepted here so callers that reuse one config dictionary for
            compile and execute can pass it through safely. It has no effect
            after the program has already been compiled.

    Device results are written back into the host tensors in *args* in
    place; per-run timing is no longer returned — read it from the runtime's
    ``[STRACE]`` log markers (simpler PR #1177) or the L2 swimlane records.
    """
    del analyze_auto_scopes_for_deps

    work_dir = Path(work_dir)

    # Ensure orchestration headers are patched (idempotent)
    _patch_orchestration_headers(work_dir)

    from .device_runner import (  # noqa: PLC0415
        compile_and_assemble,
        execute_on_device,
    )

    chip_callable, runtime_name, runtime_config = compile_and_assemble(work_dir, platform, pto_isa_commit)

    # Caller-supplied values take precedence over the RUNTIME_CONFIG baked
    # into kernel_config.py. When neither is provided, the simpler runtime's
    # own default applies (and is validated against device capacity).
    effective_block_dim = block_dim if block_dim is not None else runtime_config.get("block_dim")
    effective_aicpu_thread_num = (
        aicpu_thread_num if aicpu_thread_num is not None else runtime_config.get("aicpu_thread_num")
    )

    orch_args = _coerced_to_orch_args(args)

    # Snapshot DFX state before execution
    dfx_dir: Path | None = None
    if dfx.any():
        dfx_dir = work_dir / "dfx_outputs"
        dfx_dir.mkdir(parents=True, exist_ok=True)

    def _run_pass(pass_dfx: "_DfxOpts") -> None:
        execute_on_device(
            chip_callable,
            orch_args,
            platform,
            runtime_name,
            device_id,
            level=level,
            block_dim=effective_block_dim,
            aicpu_thread_num=effective_aicpu_thread_num,
            output_prefix=str(dfx_dir) if dfx_dir is not None else None,
            enable_l2_swimlane=pass_dfx.enable_l2_swimlane,
            enable_dump_tensor=pass_dfx.enable_dump_tensor,
            enable_pmu=pass_dfx.enable_pmu,
            enable_dep_gen=pass_dfx.enable_dep_gen,
            enable_scope_stats=pass_dfx.enable_scope_stats,
        )

    def _capture_deps() -> None:
        # Compiled-program path: live args may be device-resident and cannot
        # cross the process boundary, so the child rebuilds zero tensors of the
        # recorded shapes plus the exact scalars (graph is structural).
        assert dfx_dir is not None  # swimlane-on implies dfx.any() -> dfx_dir set
        run_id = uuid.uuid4().hex
        _capture_deps_subprocess(
            {
                "mode": "argspec",
                "args": _build_args_spec(args, dfx_dir, run_id),
                "work_dir": str(work_dir),
                "platform": platform,
                "device_id": device_id,
                "dfx_dir": str(dfx_dir),
                "pto_isa_commit": pto_isa_commit,
                "level": level,
                "block_dim": effective_block_dim,
                "aicpu_thread_num": effective_aicpu_thread_num,
            },
            dfx_dir,
            run_id,
        )

    # When swimlane is on (onboard), capture deps.json in a subprocess first,
    # then run the clean-timing swimlane pass in-process (see _execute_dfx_passes).
    _execute_dfx_passes(_run_pass, _capture_deps, dfx, platform)

    # Collect DFX artefacts after execution (no-op when dfx_dir is None).
    # Original ``dfx`` drives collection so swimlane conversion auto-joins
    # ``deps.json`` and the deps-render hint fires only on explicit dep_gen.
    if dfx_dir is not None:
        _collect_dfx_artifacts(dfx_dir, platform, dfx)
