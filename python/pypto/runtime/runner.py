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

Provides :class:`RunConfig` — the settings that drive a run — and the
dispatch implementation that puts an already-compiled artifact directory
onto an Ascend NPU (or simulator). The supported way in is
:meth:`pypto.ir.CompiledProgram.from_dir`; the module-level
:func:`execute_compiled` is a deprecated wrapper over the same code.

Compilation is :func:`pypto.ir.compile`'s job; nothing here compiles for you.
:meth:`RunConfig.compile_kwargs` carries the compile-side settings across::

    import torch
    from pypto import ir
    from pypto.runtime import RunConfig

    config = RunConfig(platform="a2a3sim")
    compiled = ir.compile(MyProgram, **config.compile_kwargs())

    a = torch.full((128, 128), 2.0)
    b = torch.full((128, 128), 3.0)
    c = torch.zeros(128, 128)
    compiled(a, b, c, config=config)
"""

import functools
import importlib.util
import inspect
import json
import shlex
import subprocess
import sys
import uuid
import warnings
from collections.abc import Callable
from ctypes import _SimpleCData
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassDumpLevel
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

# Artifact name from Simpler's Worker/Chip/Core naming migration (formerly
# ``l2_swimlane_records.json``).
_CHIP_SWIMLANE_RECORDS_NAME = "chip_swimlane_records.json"

# Chip-swimlane collection is *levelled*, not a toggle: each level is a real
# guard in the runtime collectors, so a lower level never records the data a
# higher one does (no post-processing recovers it).  Mirrors
# ``ChipSwimlaneLevel`` and the runtime harness's ``--enable-chip-swimlane``.
#
#   0 DISABLED      off
#   1 AICORE_TIMING AICore per-task start/end + task record buffer
#   2 AICPU_TIMING  + AICPU-stamped dispatch/finish (the [dispatch, start] gap)
#   3 SCHED_PHASES  + scheduler main-loop phases (``sched_overhead_analysis``)
#   4 ORCH_PHASES   + orchestrator phases
_SWIMLANE_MAX_LEVEL = 4
# What a bare ``True`` (or a bare CLI flag) requests.  Level 4 matches both the
# runtime harness's bare ``--enable-chip-swimlane`` and the ``CallConfig``
# nanobind setter, which already maps a Python ``True`` to 4.
_SWIMLANE_FULL_LEVEL = 4
# Shared by every ``--enable-l2-swimlane`` / ``--swimlane`` CLI so the ladder is
# described once. Callers that need extra prose concatenate onto this.
_SWIMLANE_CLI_HELP = (
    f"Chip swimlane collection level. Bare flag = {_SWIMLANE_FULL_LEVEL} (full); "
    "1=AICore timing, 2=+AICPU dispatch/finish, 3=+scheduler phases, "
    "4=+orchestrator phases; absent = 0 (off)."
)


class ExecutionMode(Enum):
    """Whether a run reaches real silicon or the simulator.

    One of the two axes a ``platform`` string packs. It decides how kernels are
    assembled (``.so`` for the simulator, ``.o`` plus a text-section extract for
    silicon) and whether the two-pass swimlane capture applies; it never reaches
    codegen, which sees only the architecture.
    """

    ONBOARD = auto()
    SIM = auto()


_ARCHES: tuple[BackendType, ...] = (BackendType.Ascend910B, BackendType.Ascend950)


def _arch_name(arch: BackendType) -> str:
    """Return the wire name of an architecture (``"a2a3"`` / ``"a5"``).

    Asks the backend handler rather than mapping it here: the C++ side already
    owns this string — it is what codegen stamps as ``pto.target_arch`` — and a
    second copy in Python is a second thing to keep in step.
    """
    return _backend_core.get_backend_instance(arch).get_handler().get_pto_target_arch()


def _platform_string(arch: BackendType, execution_mode: ExecutionMode) -> str:
    """Join the two axes into the wire spelling the runtime and CLI take."""
    return f"{_arch_name(arch)}{'sim' if execution_mode is ExecutionMode.SIM else ''}"


def _parse_platform(platform: str) -> tuple[BackendType, ExecutionMode]:
    """Split a wire platform string back into its two axes.

    The single place that sniffs the string. Everything else asks the axes.
    """
    for arch in _ARCHES:
        name = _arch_name(arch)
        if platform == name:
            return arch, ExecutionMode.ONBOARD
        if platform == f"{name}sim":
            return arch, ExecutionMode.SIM
    expected = ", ".join(f"{_arch_name(a)!r}, {_arch_name(a) + 'sim'!r}" for a in _ARCHES)
    raise ValueError(f"Invalid platform {platform!r}. Expected {expected}.")


def _backend_type_for_platform(platform: str) -> BackendType:
    """Return the codegen backend a runtime platform string selects."""
    return _parse_platform(platform)[0]


_BACKEND_TYPE_DEPRECATION = (
    "RunConfig(backend_type=...) is deprecated and has never taken effect: the backend is "
    "derived from platform, which wins. Drop the argument, or pass the platform that implies "
    "the backend you want. Reading RunConfig.backend_type still reports what platform selected."
)


_SWIMLANE_ALIAS_DEPRECATION = (
    "RunConfig.enable_l2_swimlane is deprecated; use enable_chip_swimlane instead. "
    "Same values and semantics — Simpler renamed the L2 layer to 'chip', and the "
    "runtime artifact is chip_swimlane_records.json. The alias will be removed in "
    "a future release."
)


def _normalize_swimlane_level(value: int | bool, source: str) -> int:
    """Normalize a chip-swimlane request to an explicit collection level.

    ``bool`` is accepted for source compatibility: ``True`` requests full
    collection (level :data:`_SWIMLANE_FULL_LEVEL`), matching the runtime's
    bare ``--enable-chip-swimlane`` and its ``CallConfig`` setter; ``False`` is
    off.  Normalizing here (rather than deferring to the ``CallConfig`` setter)
    keeps the level readable from Python and lets the harness round-trip it
    through the ``--enable-l2-swimlane`` CLI without flattening it.

    Args:
        value: Requested level (``0``-``4``) or ``bool``.
        source: Name of the option being normalized, used in error messages.

    Returns:
        The collection level as an ``int`` in ``[0, _SWIMLANE_MAX_LEVEL]``.

    Raises:
        TypeError: If *value* is neither ``bool`` nor ``int``.
        ValueError: If *value* is an out-of-range level.
    """
    if isinstance(value, bool):
        return _SWIMLANE_FULL_LEVEL if value else 0
    if not isinstance(value, int):
        raise TypeError(
            f"{source} must be an int collection level (0-{_SWIMLANE_MAX_LEVEL}) or a bool, "
            f"got {type(value).__name__}"
        )
    if not 0 <= value <= _SWIMLANE_MAX_LEVEL:
        raise ValueError(
            f"{source} must be a collection level in [0, {_SWIMLANE_MAX_LEVEL}] "
            f"(0=off, 1=AICore timing, 2=+dispatch/finish, 3=+sched phases, "
            f"4=+orch phases), got {value}"
        )
    return value


@dataclass(kw_only=True)
class RunConfig:
    """Configuration for compiling and dispatching a program.

    Carries both halves: :meth:`compile_kwargs` extracts the compile-side
    fields for :func:`pypto.ir.compile`, and the rest is read at dispatch by
    :meth:`~pypto.ir.CompiledProgram.__call__` / :meth:`Worker.run`.

    When passed to :meth:`pypto.jit.decorator.JITFunction.lower`, only
    ``platform``, ``strategy``, diagnostics, dependency analysis, and
    ``memory_planner`` affect pass execution. Runtime and artifact fields such
    as ``device_id``, ``dump_passes``, ``dump_ptoas_passes``,
    ``save_kernels_dir``, and ``compile_profiling`` are ignored; ``lower()``
    does not execute or write compilation artifacts.

    Attributes:
        arch: Target architecture, as the codegen backend that names it —
            ``BackendType.Ascend910B`` (a2a3) or ``BackendType.Ascend950`` (a5).
        execution_mode: :class:`ExecutionMode.SIM` or ``ONBOARD``.
        platform: **Not a field** — the wire spelling of the two axes above
            (``"a2a3sim"`` / ``"a2a3"`` / ``"a5sim"`` / ``"a5"``), derived on
            read. Accepted as a constructor keyword, where it sets both axes.
        device_id: Hardware device index (ignored for simulator).
        backend_type: **Not a field** — a read-only property derived from
            ``platform``. Accepted as a deprecated constructor keyword, which
            warns and is discarded when it contradicts the platform.
        rtol: Relative tolerance for result comparison.
        atol: Absolute tolerance for result comparison.
        strategy: PyPTO optimisation strategy applied during compilation.
        dump_passes: Per-pass IR dump control. A :class:`~pypto.ir.PassDumpLevel`
            (``NONE`` / ``CONCISE`` / ``EXPLICIT``) or a ``bool``
            (``True`` -> ``CONCISE``, ``False`` -> ``NONE``). ``EXPLICIT`` resolves
            implicit tile layouts and distributed window buffers in the dump.
        dump_ptoas_passes: If ``True``, dump full-module intermediate IR after
            every ptoas pass under
            ``<output_dir>/ptoas_passes/<codegen-unit>/``. Has no effect when
            ptoas is unavailable and compilation stops at raw ``.pto`` output.
        save_kernels: If ``True``, retain generated artefacts after execution.
            When ``False`` (default), a temporary directory is used and cleaned up.
        save_kernels_dir: Directory to save generated artefacts when *save_kernels*
            is ``True``.  If ``None``, a timestamped directory is created under
            ``build_output/<program_name>_<timestamp>``.
        codegen_only: If ``True``, stop after code generation without executing
            on device.  Useful for validating compilation output.
        enable_chip_swimlane: Chip swimlane collection **level** — per-task
            timing records written into
            ``<work_dir>/dfx_outputs/chip_swimlane_records.json``. Mirrors the
            runtime harness's ``--enable-chip-swimlane PERF_LEVEL``; each level
            is a real guard in the runtime collectors, so a lower level never
            stamps the data a higher one does and no post-processing recovers
            it:

            * ``0`` / ``False`` — off.
            * ``1`` — AICore per-task start / end plus the task record buffer.
            * ``2`` — ``1`` plus AICPU-stamped dispatch / finish, which is what
              makes the ``[dispatch, start]`` pickup gap readable.
            * ``3`` — ``2`` plus scheduler main-loop phase records, required by
              ``python -m simpler_setup.tools.sched_overhead_analysis`` and by
              the PyPTO Toolkit plugin's Scheduler View.
            * ``4`` / ``True`` — ``3`` plus orchestrator phase records (the
              Toolkit plugin's AICPU Orchestrator view). ``True`` requests this
              full level, matching the runtime harness's bare
              ``--enable-chip-swimlane``.

            The former spelling ``enable_l2_swimlane`` still works (constructor
            keyword and attribute) but emits a ``DeprecationWarning``.

            On onboard platforms, ``swimlane_converter`` then produces
            ``merged_swimlane_*.json`` alongside the records. Because the converter joins
            the timing against a task graph that only ``deps.json`` carries,
            enabling this on an onboard platform runs the workload **twice**: a
            first dep_gen pass captures ``deps.json``, then a clean swimlane pass
            runs with dep_gen off because collection perturbs timing. L2 runs
            the graph pass in a subprocess so its device/SVM state is fully
            reclaimed before the timing pass (a failed capture is logged, not
            fatal). L3 one-shot uses separate Worker lifecycles; a prepared L3
            worker uses two ``Worker.run()`` fences and keeps resident handles
            alive. Both L3 passes execute the program without restoring mutable
            arguments between them. Simulator platforms (``*sim``) stay single-pass
            and only emit ``chip_swimlane_records.json`` — the merged swimlane file
            is intentionally skipped because the simulator does not yet ship the
            task metadata the converter needs. Mirrors runtime's
            ``CallConfig.enable_chip_swimlane`` field.
        enable_dump_args: Per-task argument dump **level** written into
            ``<work_dir>/dfx_outputs/args_dump/``. Inspect with
            ``python -m simpler_setup.tools.dump_viewer``. Mirrors
            ``--dump-args``:

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
        enable_dep_gen: Capture simpler dependency edges into
            ``<work_dir>/dfx_outputs/deps.json``. Render to HTML on demand via
            ``python -m simpler_setup.tools.deps_viewer <deps.json> --format
            html`` (the CLI defaults to text output). Mirrors
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
        aicpu_thread_num: Optional per-invocation override of the AICPU
            thread count. ``None`` (default) defers to the value baked
            into ``kernel_config.py``'s ``RUNTIME_CONFIG`` at compile
            time (which itself may be unset, in which case the simpler
            runtime default applies).
        ring_task_window: Optional per-invocation override of the runtime
            ring's task-slot window (number of in-flight tasks). Forwarded to
            ``CallConfig.runtime_env.ring_task_window``. A scalar (broadcast to
            all scope-depth rings) or a list/tuple of exactly 4 ints sizing
            rings 0..3 independently; each entry must be a power of two ``>= 4`` (a
            ``0`` list-entry leaves that ring at its default). ``None`` (default)
            leaves the field unset so the runtime falls back to its
            compile-time default.
        ring_heap: Optional per-invocation override of the per-ring output-heap
            size in **bytes**. Forwarded to ``CallConfig.runtime_env.ring_heap``.
            A scalar or a list/tuple of 4 ints (per ring 0..3); each entry must
            be a power of two ``>= 1024`` (a ``0`` list-entry leaves that ring at its
            default). ``None`` defers to the runtime's compile-time default.
        ring_dep_pool: Optional per-invocation override of the per-ring
            dependency-edge pool capacity. Forwarded to
            ``CallConfig.runtime_env.ring_dep_pool``. A scalar or a list/tuple
            of 4 ints (per ring 0..3); each entry must be in ``[4, INT32_MAX]`` (a
            ``0`` list-entry leaves that ring at its default). ``None`` defers
            to the runtime's compile-time default.
        distributed_config: Optional L3 distributed-execution config, consumed
            only on the ``@pl.jit`` path. When set, it is forwarded to
            ``ir.compile()`` (via :meth:`RunConfig.compile_kwargs`)
            so a HOST-level ``@pl.jit.host`` kernel compiles to a
            :class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`
            and dispatches per-rank. ``None`` (default) compiles a regular
            single-chip :class:`~pypto.ir.compiled_program.CompiledProgram`.
            :meth:`compile_kwargs` forwards it too, so a ``@pl.program``
            compiled via ``ir.compile(prog, **config.compile_kwargs())`` picks
            up the same distributed target.
        analyze_auto_scopes_for_deps: If ``True``, enable compiler-derived task
            dependency analysis for AUTO runtime scopes during compilation.
            Defaults to ``False`` so existing runs keep using TensorMap fallback
            unless this behavior is explicitly requested.
        memory_planner: Who plans on-chip buffer memory —
            :attr:`~pypto.pypto_core.passes.MemoryPlanner.PYPTO` (PyPTO runs
            ``MemoryReuse`` + ``AllocateMemoryAddr`` and bakes physical
            addresses), ``DSA_RP`` (PyPTO runs its in-tree
            capacity-constrained DSA-RP planner), or ``PTOAS`` (the PyPTO
            allocation passes are skipped and ptoas ``PlanMemory`` owns reuse
            and addressing). ``None`` (default) defers to the active
            ``PassContext``, or to ``PYPTO`` when none is active.
            Forwarded to ``ir.compile()``, which rejects it when a
            ``PassContext`` is already active — set it on that context instead.
    """

    __test__ = False  # Not a pytest test class

    arch: BackendType = field(default_factory=lambda: BackendType.Ascend910B)
    execution_mode: ExecutionMode = ExecutionMode.SIM
    device_id: int = 0
    rtol: float = 1e-5
    atol: float = 1e-5
    strategy: OptimizationStrategy = field(default_factory=lambda: OptimizationStrategy.Default)
    dump_passes: bool | PassDumpLevel = False
    save_kernels: bool = False
    save_kernels_dir: str | None = None
    codegen_only: bool = False
    # 0=off, 1=AICore timing, 2=+dispatch/finish, 3=+sched phases, 4=+orch
    # phases. ``True`` normalizes to 4 (full), ``False`` to 0. The former
    # spelling ``enable_l2_swimlane`` is still accepted; see the deprecation
    # shim installed just below the class.
    enable_chip_swimlane: int | bool = 0
    enable_dump_args: int = 0  # 0=off, 1=partial (dump_tag-marked), 2=full
    enable_pmu: int = 0
    enable_dep_gen: bool = False
    enable_scope_stats: bool = False
    compile_profiling: bool = False
    diagnostic_phase: DiagnosticPhase | None = None
    disabled_diagnostics: DiagnosticCheckSet | None = None
    golden_data_dir: str | None = None
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
    memory_planner: MemoryPlanner | None = None
    dump_ptoas_passes: bool = False

    def __post_init__(self) -> None:
        # The two axes replace what used to be a membership test on the packed
        # platform string. They make a *disagreeing* platform unrepresentable,
        # but not a nonsensical one: ``execution_mode="sim"`` is not
        # ``ExecutionMode.SIM``, and silently reading it as ONBOARD would turn a
        # simulator request into a hardware run. Reject it here, where the value
        # is still attached to the name the caller typed.
        if not isinstance(self.arch, BackendType):
            raise TypeError(
                f"RunConfig.arch must be a BackendType, got {type(self.arch).__name__} "
                f"({self.arch!r}). Pass BackendType.Ascend910B / Ascend950, or use "
                f"platform='a2a3sim' to set both axes from the wire spelling."
            )
        if not isinstance(self.execution_mode, ExecutionMode):
            raise TypeError(
                f"RunConfig.execution_mode must be an ExecutionMode, got "
                f"{type(self.execution_mode).__name__} ({self.execution_mode!r}). Pass "
                f"ExecutionMode.SIM / ONBOARD, or use platform='a2a3sim' to set both axes."
            )

        # Chip swimlane is levelled; normalize ``bool``/int to an explicit
        # level before ``any_dfx_enabled()`` and the CLI round-trip read it.
        self.enable_chip_swimlane = _normalize_swimlane_level(
            self.enable_chip_swimlane, "enable_chip_swimlane"
        )

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
        chip swimlane, argument dump, PMU, dep_gen and scope_stats. They are
        independent toggles that share an output directory.
        """
        return self.dfx_options().any()

    def compile_kwargs(self) -> dict[str, Any]:
        """Return the compile-side fields as ``ir.compile()`` keyword arguments.

        A :class:`RunConfig` carries both compile-time and dispatch-time
        settings. This method extracts the compile-time half so a caller can
        drive the two phases separately without restating the mapping::

            compiled = ir.compile(program, **config.compile_kwargs())
            compiled(*tensors, config=config)

        It is the only mapping onto ``ir.compile``'s parameters: the ``@pl.jit``
        path calls it too, so a knob added here reaches both. (``lower()`` keeps
        its own narrower mapping — it stops before codegen and targets the pass
        pipeline rather than ``ir.compile``.)

        Dispatch-only fields (``device_id``, the DFX toggles, the ring-sizing
        overrides) are not compile inputs and are consumed by
        :meth:`~pypto.ir.CompiledProgram.__call__` instead. ``rtol`` / ``atol``
        and ``golden_data_dir`` reach neither phase: only the system-test
        harness reads them, to compare a dispatch against its golden.

        ``output_dir``, ``distributed_config`` and ``memory_planner`` are
        forwarded only when set, so an unset value defers to ``ir.compile()``'s
        own default. That matters for ``memory_planner`` in particular:
        ``ir.compile()`` rejects an explicit planner while a ``PassContext`` is
        active, and an unset value must defer to that context.

        Returns:
            Keyword arguments accepted by :func:`pypto.ir.compile`.
        """
        return self.compile_options().as_compile_kwargs()

    def compile_options(self) -> "CompileOptions":
        """Return the compile-side half as a :class:`CompileOptions`.

        The field renames are the compiler's own vocabulary:
        ``save_kernels_dir`` is ``ir.compile``'s ``output_dir`` and
        ``compile_profiling`` is its ``profiling``.
        """
        return CompileOptions(
            platform=self.platform,
            strategy=self.strategy,
            dump_passes=self.dump_passes,
            dump_ptoas_passes=self.dump_ptoas_passes,
            profiling=self.compile_profiling,
            diagnostic_phase=self.diagnostic_phase,
            disabled_diagnostics=self.disabled_diagnostics,
            analyze_auto_scopes_for_deps=self.analyze_auto_scopes_for_deps,
            output_dir=self.save_kernels_dir,
            memory_planner=self.memory_planner,
            distributed_config=self.distributed_config,
        )

    def run_options(self) -> "RunOptions":
        """Return the dispatch-side half as a :class:`RunOptions`."""
        return RunOptions(
            platform=self.platform,
            device_id=self.device_id,
            aicpu_thread_num=self.aicpu_thread_num,
            ring_task_window=self.ring_task_window,
            ring_heap=self.ring_heap,
            ring_dep_pool=self.ring_dep_pool,
            dfx=self.dfx_options(),
        )

    def dfx_options(self) -> "DfxOptions":
        """Return the diagnostic toggles as a :class:`DfxOptions`.

        Shorthand for ``run_options().dfx`` — the runtime asks for the
        diagnostics alone far more often than for the whole dispatch half.
        """
        return DfxOptions(
            enable_chip_swimlane=self.enable_chip_swimlane,
            enable_dump_args=self.enable_dump_args,
            enable_pmu=self.enable_pmu,
            enable_dep_gen=self.enable_dep_gen,
            enable_scope_stats=self.enable_scope_stats,
        )

    @property
    def platform(self) -> str:
        """The wire spelling of :attr:`arch` + :attr:`execution_mode`.

        Derived, never stored. It is what the simpler ``Worker``, the artifact
        sidecars and ``--platform`` all take, so it stays a plain ``str`` — but
        it is a serialization of the two axes rather than a field anything can
        set out of step with them.
        """
        return _platform_string(self.arch, self.execution_mode)

    @property
    def backend_type(self) -> BackendType:
        """The codegen backend, which is :attr:`arch` under the compiler's name.

        Kept as a read accessor because the artifact metadata and downstream
        callers read it; :attr:`arch` is the field to set.
        """
        return self.arch

    @property
    def enable_l2_swimlane(self) -> int:
        """Deprecated alias for :attr:`enable_chip_swimlane`.

        Reading is intentionally silent: ``dataclasses.replace()`` and existing
        callers go through here, and warning on every read would make
        ``replace(cfg, ...)`` noisy without pointing at a name the caller chose.
        Assigning, and the constructor keyword, do warn.
        """
        return self.enable_chip_swimlane

    @enable_l2_swimlane.setter
    def enable_l2_swimlane(self, value: int | bool) -> None:
        warnings.warn(_SWIMLANE_ALIAS_DEPRECATION, DeprecationWarning, stacklevel=2)
        self.enable_chip_swimlane = _normalize_swimlane_level(value, "enable_l2_swimlane")


# ---------------------------------------------------------------------------
# Deprecated constructor keywords: ``enable_l2_swimlane`` and ``backend_type``
# ---------------------------------------------------------------------------
# Simpler's Worker/Chip/Core naming migration renamed the L2 layer to "chip"
# (``L2Swimlane*`` -> ``ChipSwimlane*``, ``l2_swimlane_records.json`` ->
# ``chip_swimlane_records.json``). ``RunConfig`` follows that contract, but the
# old spelling stays usable for one release.
#
# The alias is deliberately **not** a dataclass field. ``dataclasses.replace()``
# re-supplies every field from the existing instance, so an alias field (or an
# ``InitVar``) would arrive alongside the canonical one on every ``replace``
# call and there is no way to tell "the caller typed the old name" from "replace
# echoed the old value back". That ambiguity resolves either into spurious
# warnings or — worse — into ``replace(cfg, enable_chip_swimlane=N)`` being
# silently overridden by the stale alias. Keeping the alias off the field list
# leaves ``replace``, ``fields()``, ``asdict()`` and ``repr()`` clean, and routes
# the old name through an ``__init__`` wrapper plus a property instead.
#
# ``backend_type`` is off the field list for the same reason, plus one of its
# own. It is derived from ``platform``, so a caller-supplied value has never
# taken effect. As a field it would come back through ``replace()``:
# ``replace(cfg, platform="a5")`` re-supplies the *old* platform's backend, and
# nothing can tell that echo from a caller who typed a contradicting value —
# so a plain platform switch would warn, and fail outright under
# warnings-as-errors.

_RUN_CONFIG_INIT = RunConfig.__init__


@functools.wraps(_RUN_CONFIG_INIT)
def _run_config_init(self: RunConfig, *args: Any, **kwargs: Any) -> None:
    """``RunConfig.__init__`` that also accepts ``platform=`` and the deprecated keywords."""
    if "platform" in kwargs:
        # ``platform=`` is the wire spelling of the two axes, and setting it
        # sets both. It deliberately wins over an ``arch=`` / ``execution_mode=``
        # in the same call rather than reporting a conflict -- the class is
        # ``kw_only`` so an axis can only arrive as a keyword, and this rewrite
        # therefore reaches every spelling of it: ``replace(cfg,
        # platform=...)`` re-supplies both axes from the existing instance, and
        # nothing can tell that echo from a caller who typed a contradicting
        # value -- the same ambiguity documented for the deprecated keywords
        # below. Rejecting the pair would break every ``replace`` by platform.
        kwargs["arch"], kwargs["execution_mode"] = _parse_platform(kwargs.pop("platform"))

    if "backend_type" in kwargs:
        supplied = kwargs.pop("backend_type")
        if supplied is not None and supplied != kwargs.get("arch", BackendType.Ascend910B):
            warnings.warn(_BACKEND_TYPE_DEPRECATION, DeprecationWarning, stacklevel=2)

    alias = kwargs.pop("enable_l2_swimlane", None)
    if alias is not None:
        warnings.warn(_SWIMLANE_ALIAS_DEPRECATION, DeprecationWarning, stacklevel=2)
        if "enable_chip_swimlane" in kwargs:
            raise ValueError(
                "RunConfig received both enable_chip_swimlane and the deprecated "
                "enable_l2_swimlane; pass only enable_chip_swimlane."
            )
        kwargs["enable_chip_swimlane"] = alias
    _RUN_CONFIG_INIT(self, *args, **kwargs)


RunConfig.__init__ = _run_config_init  # type: ignore[method-assign]

# ``functools.wraps`` copies the dataclass-generated signature, which lists the
# two axes but not the ``platform=`` spelling the wrapper accepts -- so
# ``inspect.signature(RunConfig)``, and every doc tool and IDE reading it, would
# report a keyword the overwhelming majority of call sites use as unsupported.
# Advertise it. The deprecated keywords stay out on purpose: they are accepted
# for compatibility, not offered.
_RUN_CONFIG_SIGNATURE = inspect.signature(_RUN_CONFIG_INIT)
RunConfig.__signature__ = _RUN_CONFIG_SIGNATURE.replace(  # type: ignore[attr-defined]
    parameters=[
        *(p for name, p in _RUN_CONFIG_SIGNATURE.parameters.items() if name != "self"),
        inspect.Parameter(
            "platform",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | None",
        ),
    ]
)


@dataclass
class RunResult:
    """Result of a program run or harness test execution.

    Attributes:
        passed: ``True`` if the program executed and results matched the golden
            reference within the configured tolerances.
        test_name: Optional test case name.  Set by the harness when running
            a named test case; ``None`` outside the harness.
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


# ---------------------------------------------------------------------------
# Option objects
#
# The three concerns :class:`RunConfig` aggregates, each as a value of its own:
# what compilation reads, what a dispatch reads, and which diagnostics a
# dispatch collects. ``RunConfig`` keeps every field and every caller — these
# are the vocabulary underneath it, and what code that only needs one half
# should take. See ``docs/en/dev/08-entry-points.md``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DfxOptions:
    """Bundle of runtime DFX toggles passed through the execute pipeline.

    Each field maps to a ``CallConfig`` member on the runtime side. The public
    ``enable_chip_swimlane`` field carries a collection level (see
    :func:`_normalize_swimlane_level`). ``any()`` answers whether the runtime
    needs an ``output_prefix``.
    """

    # Collection level (0=off .. 4=full); ``__post_init__`` normalizes a bool.
    enable_chip_swimlane: int | bool = 0
    enable_dump_args: int = 0  # 0=off, 1=partial, 2=full
    enable_pmu: int = 0
    enable_dep_gen: bool = False
    enable_scope_stats: bool = False

    def __post_init__(self) -> None:
        # Frozen dataclass: normalize in place so a ``bool`` handed to the
        # constructor (or to ``dataclasses.replace``) becomes the same explicit
        # level ``RunConfig`` produces. ``_dfx_to_cli`` stringifies this field.
        object.__setattr__(
            self,
            "enable_chip_swimlane",
            _normalize_swimlane_level(self.enable_chip_swimlane, "enable_chip_swimlane"),
        )

    def any(self) -> bool:
        return (
            self.enable_chip_swimlane > 0
            or self.enable_dump_args > 0
            or self.enable_pmu > 0
            or self.enable_dep_gen
            or self.enable_scope_stats
        )


@dataclass(frozen=True)
class RunOptions:
    """What a dispatch reads: where it runs, how big its rings are, what it collects.

    Everything here is per-launch. Nothing here reaches compilation — an
    artifact compiled once can be dispatched under any number of these.

    ``platform`` appears in both halves because it is genuinely two decisions
    that must agree: the target codegen builds for, and the device the worker
    opens. A worker rejects an artifact whose platform differs from its own.

    **Not exported from** ``pypto.runtime``, and deliberately so: no dispatch
    entry point accepts one yet. ``CompiledProgram.__call__``,
    ``ChipWorker.run`` and their distributed counterparts all take a
    :class:`RunConfig`, and reach it through ``run_options()``. Until those
    signatures widen, this is the internal shape the dispatch plumbing reads,
    not a configuration a caller can hand in — exporting it would advertise an
    entry point that does not exist.
    """

    platform: str = "a2a3sim"
    device_id: int = 0
    aicpu_thread_num: int | None = None
    # Scalar (broadcast to every scope-depth ring) or a list of ``_RING_DEPTH``
    # ints sizing rings 0..3; a 0 entry leaves that ring at its default.
    ring_task_window: int | list[int] | tuple[int, ...] | None = None
    ring_heap: int | list[int] | tuple[int, ...] | None = None
    ring_dep_pool: int | list[int] | tuple[int, ...] | None = None
    dfx: DfxOptions = DfxOptions()


@dataclass(frozen=True)
class CompileOptions:
    """What compilation reads, in ``ir.compile``'s own vocabulary.

    The typed form of :meth:`RunConfig.compile_kwargs`. A caller that only
    compiles needs this and not a :class:`RunConfig`::

        from pypto import ir
        from pypto.runtime import CompileOptions

        compiled = ir.compile(program, **CompileOptions(platform="a2a3").as_compile_kwargs())

    Field names are ``ir.compile``'s, not ``RunConfig``'s: what ``RunConfig``
    spells ``save_kernels_dir`` and ``compile_profiling`` are ``output_dir`` and
    ``profiling`` here, because this object exists to name the compile side as
    the compiler names it.

    ``platform`` is the only way to name the target. ``ir.compile`` also takes a
    ``backend_type``, but derives it from ``platform`` whenever one is given, so
    carrying both here would offer a pairing that cannot take effect: set them
    to disagree and the platform silently wins. ``ir.compile`` keeps its
    parameter for callers that pass no platform at all; this object always
    passes one.
    """

    platform: str = "a2a3sim"
    strategy: OptimizationStrategy = field(default_factory=lambda: OptimizationStrategy.Default)
    dump_passes: bool | PassDumpLevel = False
    dump_ptoas_passes: bool = False
    profiling: bool = False
    diagnostic_phase: DiagnosticPhase | None = None
    disabled_diagnostics: DiagnosticCheckSet | None = None
    analyze_auto_scopes_for_deps: bool = False
    # Absent rather than ``None`` in ``as_compile_kwargs`` when unset, so
    # ``ir.compile``'s own default applies. That is load-bearing for
    # ``memory_planner``: an explicit one is rejected while a ``PassContext`` is
    # active, so an unset planner has to defer to that context.
    output_dir: str | None = None
    memory_planner: MemoryPlanner | None = None
    distributed_config: "DistributedConfig | None" = None

    def as_compile_kwargs(self) -> dict[str, Any]:
        """Return these options as :func:`pypto.ir.compile` keyword arguments."""
        kwargs: dict[str, Any] = {
            "platform": self.platform,
            "strategy": self.strategy,
            "dump_passes": self.dump_passes,
            "dump_ptoas_passes": self.dump_ptoas_passes,
            "profiling": self.profiling,
            "diagnostic_phase": self.diagnostic_phase,
            "disabled_diagnostics": self.disabled_diagnostics,
            "analyze_auto_scopes_for_deps": self.analyze_auto_scopes_for_deps,
        }
        for name in ("output_dir", "memory_planner", "distributed_config"):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        return kwargs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _execute_dfx_passes(
    run_pass: Callable[["DfxOptions"], None],
    capture_deps: Callable[[], None],
    dfx: "DfxOptions",
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
        producing the clean ``chip_swimlane_records.json``. Runs in-process.

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
    if not dfx.enable_chip_swimlane or platform.endswith("sim"):
        run_pass(dfx)
        return

    # The two passes look like a double run, so announce what each is for.
    print(
        "[swimlane] chip swimlane enabled -> running the kernel twice "
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

    Shared by :func:`_execute_golden_case` and the dep_gen subprocess so the load
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
    worker: Any,
) -> Any:
    """Pack coerced values into address-free simpler ``TaskArgs``.

    Simpler's public ``Worker.run`` accepts ``TaskArgs`` containing wire
    ``Tensor`` descriptors. The owning Worker is required to assign stable
    buffer identities and materializes those descriptors into chip PODs at the
    L2 boundary. Tensors and scalars are added in separate passes because
    codegen addresses them from independent pools.

    Used by both :func:`_execute_compiled` and the extraction path on
    :class:`pypto.ir.CompiledProgram` (``_build_orch_args``).
    """
    from .task_interface import (  # noqa: PLC0415
        TaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
        scalar_to_uint64,  # pyright: ignore[reportAttributeAccessIssue]
    )
    from .tensor_arg import make_tensor_arg  # noqa: PLC0415

    orch_args = TaskArgs()
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
            orch_args.add_tensor(make_tensor_arg(worker, arg))
        elif isinstance(arg, DeviceTensor):
            try:
                orch_args.add_tensor(make_tensor_arg(worker, arg))
            except (TypeError, ValueError) as e:
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


def _apply_ring_overrides(call_config: Any, run_config: "RunConfig | RunOptions") -> None:
    """Overlay per-task ring sizing onto a ``CallConfig``.

    Each ``runtime_env`` field is left at its ``0`` default when the matching
    override is ``None``, so the runtime applies its own compile-time default.
    Shared by the L2 (:func:`_build_call_config`) and L3
    (:func:`pypto.runtime.distributed_runner._make_call_config`) dispatch paths
    so both transcribe ring sizing identically.

    Args:
        call_config: A simpler ``CallConfig`` (mutated in place).
        run_config: A :class:`RunOptions`, or a :class:`RunConfig` to read one
            from. The three ``ring_*`` fields are spelled the same on both.
    """
    options = run_config.run_options() if isinstance(run_config, RunConfig) else run_config
    if options.ring_task_window is not None:
        call_config.runtime_env.ring_task_window = options.ring_task_window
    if options.ring_heap is not None:
        call_config.runtime_env.ring_heap = options.ring_heap
    if options.ring_dep_pool is not None:
        call_config.runtime_env.ring_dep_pool = options.ring_dep_pool


def _build_call_config(
    run_config: "RunConfig",
    *,
    runtime_config: dict[str, Any],
    aicpu_thread_num_override: int | None = None,
    dfx_dir: Path | None = None,
) -> Any:
    """Translate a pypto :class:`RunConfig` into a simpler ``CallConfig``.

    Precedence for ``aicpu_thread_num``: explicit *override* >
    ``run_config`` field > ``runtime_config`` baked into
    ``kernel_config.py``. When all three are unset the field is left
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
    options = run_config.run_options()

    at = aicpu_thread_num_override if aicpu_thread_num_override is not None else options.aicpu_thread_num
    at = at if at is not None else runtime_config.get("aicpu_thread_num")
    if at is not None:
        cfg.aicpu_thread_num = at

    # Already a normalized collection level (0-4), so it lands on the runtime's
    # ``int32_t`` field verbatim rather than through the setter's bool shortcut.
    dfx = options.dfx
    cfg.enable_chip_swimlane = dfx.enable_chip_swimlane
    cfg.enable_dump_args = dfx.enable_dump_args
    cfg.enable_pmu = dfx.enable_pmu
    cfg.enable_dep_gen = dfx.enable_dep_gen
    cfg.enable_scope_stats = dfx.enable_scope_stats

    # Per-task ring sizing: leave the runtime_env field at its 0 default when
    # unset so the runtime applies its own compile-time default.
    _apply_ring_overrides(cfg, options)

    if dfx_dir is not None:
        cfg.output_prefix = str(dfx_dir)
    return cfg


def _execute_golden_case(
    work_dir: Path,
    golden_path: Path,
    chip_callable: Any,
    runtime_name: str,
    platform: str,
    device_id: int,
    dfx: DfxOptions = DfxOptions(),
    validate: bool = True,
    actual_out_dir: "Path | None" = None,
    enable_sdma: bool = False,
) -> None:
    """Load inputs, execute on device, and validate against golden.

    Shared execution logic used by both :func:`_execute_compiled` and the test harness
    (``test_runner.py``).  The caller is responsible for compiling binaries
    via ``_compile_and_assemble`` and passing the result here.

    Tolerances (``RTOL``, ``ATOL``) are read from the generated ``golden.py``.

    Args:
        work_dir: Root output directory containing ``data/``, ``golden.py``, etc.
        golden_path: Path to the generated ``golden.py`` file.
        chip_callable: Pre-compiled ``ChipCallable`` from ``_compile_and_assemble``.
        runtime_name: Runtime name from ``_compile_and_assemble``.
        platform: Target execution platform.
        device_id: Hardware device index.
        enable_sdma: Whether execution requires an SDMA-capable worker.
            Defaults to ``False`` for legacy and hand-built callables.
        dfx: Runtime DFX toggles. When any flag is enabled the artefacts
            land under ``<work_dir>/dfx_outputs/`` and the matching
            post-run converter is invoked.
    """
    from .device_runner import (  # noqa: PLC0415
        _execute_on_device,
        build_orch_args_from_inputs,
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

    def _run_pass(pass_dfx: "DfxOptions") -> None:
        _execute_on_device(
            chip_callable,
            orch_args,
            platform,
            runtime_name,
            device_id,
            enable_sdma=enable_sdma,
            output_prefix=str(dfx_dir) if dfx_dir is not None else None,
            enable_chip_swimlane=pass_dfx.enable_chip_swimlane,
            enable_dump_args=pass_dfx.enable_dump_args,
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

    The counterpart to ``_execute_golden_case(..., validate=False,
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
    dfx: "DfxOptions",
) -> None:
    """Dispatch post-run DFX converters per enabled flag.

    The runtime writes each artefact directly into *dfx_dir* (the
    ``CallConfig.output_prefix`` passed at submit). Each branch below is
    independent and skips silently when its artefact is missing — a
    partial DFX run (e.g. only ``enable_dump_args``) must not crash on
    the swimlane converter looking for ``chip_swimlane_records.json``.
    """
    # Synthesise the func_id→name map the profiling tools need for readable
    # labels. simpler's SceneTest harness writes this itself; pypto does not
    # use SceneTest, so we derive it from ``kernel_config.py`` and drop it next
    # to the records. ``deps_viewer`` auto-discovers ``name_map_*.json`` in
    # the same directory, and ``swimlane_converter`` is pointed at it below via
    # ``--func-names``. Written whenever swimlane or dep_gen is enabled (the two
    # consumers); harmless no-op when no kernel names are available.
    name_map_path: Path | None = None
    if dfx.enable_chip_swimlane or dfx.enable_dep_gen:
        name_map_path = _write_name_map(dfx_dir.parent, dfx_dir)

    chip_swimlane_records = dfx_dir / _CHIP_SWIMLANE_RECORDS_NAME
    if dfx.enable_chip_swimlane and chip_swimlane_records.exists():
        # Swimlane conversion is onboard-only — the simulator produces
        # ``chip_swimlane_records.json`` but does not yet ship the matching
        # task metadata the converter expects.
        if not platform.endswith("sim"):
            _generate_swimlane(
                dfx_dir.parent,
                dfx_dir,
                chip_swimlane_records,
                func_names=name_map_path,
            )
        else:
            print(
                "Skipping swimlane conversion on simulator: "
                "merged_swimlane_*.json is only generated for onboard runs."
            )

    if dfx.enable_dep_gen and (dfx_dir / "deps.json").exists():
        # ``deps_viewer`` is an offline post-processing tool; leave the
        # artefact in place and point the user at the rendering command.
        # Doing it inline on hot path risks hanging the run on large graphs
        # (Graphviz ``dot`` is O(N²~N³) and has SIGKILL'd taskqueue jobs).
        # ``shlex.quote`` keeps the printed command copy-pasteable even when
        # the path contains spaces or other shell metacharacters.
        deps_path = shlex.quote(str(dfx_dir / "deps.json"))
        print(
            f"deps.json written to {deps_path} — render with:\n"
            f"  python -m simpler_setup.tools.deps_viewer {deps_path}\n"
            f"  # for large graphs, render HTML with a scalable layout engine:\n"
            f"  python -m simpler_setup.tools.deps_viewer {deps_path} --format html --engine sfdp\n"
            f"  # --engine choices: dot | sfdp | fdp | neato | circo | twopi"
        )

    if dfx.enable_dump_args > 0 and (dfx_dir / "args_dump" / "args_dump.json").exists():
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
    ``deps_viewer`` auto-discovers any sibling ``name_map_*.json``. simpler's
    SceneTest harness writes this file itself, but pypto does not use SceneTest,
    so we build the same ``callable_id_to_name`` mapping from the
    ``func_id``/``name`` fields already emitted into ``kernel_config.py``.

    Args:
        work_dir: Directory containing ``kernel_config.py``.
        dfx_dir: ``dfx_outputs`` directory where the name map is written
            (alongside ``chip_swimlane_records.json`` / ``deps.json``).

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

    Output is written to *swimlane_dir* alongside the input ``chip_swimlane_records.json``.

    Args:
        work_dir: Directory containing ``kernel_config.py``. Passed to the
            converter as ``-k`` when the file exists, and omitted when it does
            not (the converter rejects a missing path).
        swimlane_dir: Directory where swimlane JSON files are written.
        perf_file: Path to the ``chip_swimlane_records.json`` file produced by
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
        print("No chip_swimlane_records.json found, skipping swimlane conversion")
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
    ]
    # The converter *errors out* on a ``-k`` path that does not exist, so only
    # pass it when there is a config to read: a caller with no single owning
    # program (see ``_collect_l3_swimlane``) still gets a swimlane, just with
    # anonymous task labels.
    if kernel_config_path.exists():
        cmd += ["-k", str(kernel_config_path)]
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


# ---------------------------------------------------------------------------
# Compiled program execution (callable API)
# ---------------------------------------------------------------------------


def _execute_compiled(  # noqa: PLR0913
    work_dir: str | Path,
    args: list[torch.Tensor | DeviceTensor | _SimpleCData],
    *,
    platform: str,
    device_id: int,
    dfx: DfxOptions = DfxOptions(),
    level: int = 2,
    aicpu_thread_num: int | None = None,
    analyze_auto_scopes_for_deps: bool = False,
    config: RunConfig | None = None,
) -> None:
    """Execute a pre-compiled program with user-provided tensors and scalars.

    Reuses :func:`device_runner._compile_and_assemble` for binary compilation
    (with caching and parallel kernel compilation) and
    :func:`device_runner._execute_on_device` for device dispatch.  Host
    ``torch.Tensor`` outputs in *args* are modified in-place with device
    results; :class:`DeviceTensor` arguments retain their owning simpler
    ``Buffer`` and are packed into address-free ``TaskArgs`` (no H2D upload,
    no D2H readback).

    Args:
        work_dir: Root output directory from :func:`ir.compile`, containing
            ``kernels/``, ``orchestration/``, and ``kernel_config.py``.
        args: Ordered list of ``torch.Tensor`` (host),
            :class:`pypto.runtime.DeviceTensor` (worker-resident), or
            ``ctypes._SimpleCData`` scalar arguments matching the
            orchestration function's parameter order.
        platform: Target execution platform.
        device_id: Hardware device index.
        dfx: Runtime DFX toggles. When any flag is enabled the artefacts
            land under ``<work_dir>/dfx_outputs/`` and the matching
            post-run converter is invoked.
        level: Hierarchy level. Forwarded to :func:`_execute_on_device`,
            which currently only supports ``2``.
        aicpu_thread_num: Optional override of the AICPU thread count.
            When ``None`` (default), the value baked into
            ``kernel_config.py``'s ``RUNTIME_CONFIG`` is used; if that is
            also unset, simpler's runtime default applies. A
            caller-supplied value takes precedence over ``RUNTIME_CONFIG``.
        analyze_auto_scopes_for_deps: Compile-side compatibility option.
            Accepted here so callers that reuse one config dictionary for
            compile and execute can pass it through safely. It has no effect
            after the program has already been compiled.
        config: Optional per-dispatch :class:`RunConfig`. Its ring-sizing
            fields are forwarded to ``CallConfig.runtime_env``. The existing
            explicit ``platform``, ``device_id``, ``dfx``, and
            ``aicpu_thread_num`` arguments keep their current behavior and
            precedence.

    Device results are written back into the host tensors in *args* in
    place; per-run timing is no longer returned — read it from the runtime's
    ``[STRACE]`` log markers (simpler PR #1177) or the chip swimlane records.
    """
    del analyze_auto_scopes_for_deps

    work_dir = Path(work_dir)

    # ``ir.compile`` stamps these when it writes the artifact. Re-apply here
    # (idempotent) so a directory produced before that change, or one whose
    # orchestration cpp was hand-edited for a replay, still builds.
    from pypto.ir.compile import _ensure_orchestration_headers  # noqa: PLC0415

    _ensure_orchestration_headers(str(work_dir))

    from .device_runner import (  # noqa: PLC0415
        _compile_and_assemble,
        _execute_on_device,
    )

    chip_callable, runtime_name, runtime_config = _compile_and_assemble(work_dir, platform)
    enable_sdma = bool(runtime_config.get("enable_sdma", False))

    # Caller-supplied values take precedence over the RUNTIME_CONFIG baked
    # into kernel_config.py. When neither is provided, the simpler runtime's
    # own default applies (and is validated against device capacity).
    effective_aicpu_thread_num = (
        aicpu_thread_num if aicpu_thread_num is not None else runtime_config.get("aicpu_thread_num")
    )

    # Snapshot DFX state before execution
    dfx_dir: Path | None = None
    if dfx.any():
        dfx_dir = work_dir / "dfx_outputs"
        dfx_dir.mkdir(parents=True, exist_ok=True)

    def _run_pass(pass_dfx: "DfxOptions") -> None:
        _execute_on_device(
            chip_callable,
            args,
            platform,
            runtime_name,
            device_id,
            level=level,
            aicpu_thread_num=effective_aicpu_thread_num,
            enable_sdma=enable_sdma,
            output_prefix=str(dfx_dir) if dfx_dir is not None else None,
            enable_chip_swimlane=pass_dfx.enable_chip_swimlane,
            enable_dump_args=pass_dfx.enable_dump_args,
            enable_pmu=pass_dfx.enable_pmu,
            enable_dep_gen=pass_dfx.enable_dep_gen,
            enable_scope_stats=pass_dfx.enable_scope_stats,
            config=config,
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
                "level": level,
                "aicpu_thread_num": effective_aicpu_thread_num,
                "ring_overrides": {
                    "ring_task_window": config.ring_task_window if config is not None else None,
                    "ring_heap": config.ring_heap if config is not None else None,
                    "ring_dep_pool": config.ring_dep_pool if config is not None else None,
                },
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


_EXECUTE_COMPILED_DEPRECATION = (
    "pypto.runtime.execute_compiled is deprecated; reconstruct the artifact and "
    "call it: ir.CompiledProgram.from_dir(work_dir)(*args, config=cfg). Fold this "
    "call's explicit platform / device_id / dfx / aicpu_thread_num into cfg first "
    "-- on the artifact path a supplied config is the sole source of all four. "
    "The directory-driven function will be removed in a future release."
)


def execute_compiled(  # noqa: PLR0913
    work_dir: str | Path,
    args: list[torch.Tensor | DeviceTensor | _SimpleCData],
    *,
    platform: str,
    device_id: int,
    dfx: DfxOptions = DfxOptions(),
    level: int = 2,
    aicpu_thread_num: int | None = None,
    analyze_auto_scopes_for_deps: bool = False,
    config: RunConfig | None = None,
) -> None:
    """Deprecated. Dispatch a build directory without recompiling.

    :meth:`pypto.ir.CompiledProgram.from_dir` rebuilds the same handle from the
    same directory and dispatches it through the same code path. **The two
    disagree on precedence, so the migration is not a plain rename.** Here, an
    explicit ``platform`` / ``device_id`` / ``dfx`` / ``aicpu_thread_num`` wins
    and ``config`` supplies only the ring overrides. On the artifact path a
    supplied ``config`` is the sole source of all four — including ``platform``,
    which then shadows ``from_dir(platform=...)``. Since ``RunConfig.platform``
    defaults to ``"a2a3sim"``, dropping the explicit arguments would silently
    move the run to the simulator. Fold them into the config instead::

        # before -- explicit args win; ``cfg`` was read for ring sizing only
        execute_compiled(work_dir, args, platform="a2a3", device_id=0, config=cfg)

        # after -- ``cfg`` carries every execution setting
        cfg = dataclasses.replace(cfg, platform="a2a3", device_id=0)
        ir.CompiledProgram.from_dir(work_dir)(*args, config=cfg)

    ``dfx`` and ``aicpu_thread_num`` have no separate spelling on that path:
    the DFX toggles and ``aicpu_thread_num`` are already ``RunConfig`` fields,
    read fresh on every dispatch. ``from_dir(platform=...)`` still decides the
    platform for a call made *without* a config.

    Emits a :class:`DeprecationWarning` and forwards to the same implementation
    the artifact path uses. Behaviour of this function is unchanged; only the
    name is going away.
    """
    warnings.warn(_EXECUTE_COMPILED_DEPRECATION, DeprecationWarning, stacklevel=2)
    _execute_compiled(
        work_dir,
        args,
        platform=platform,
        device_id=device_id,
        dfx=dfx,
        level=level,
        aicpu_thread_num=aicpu_thread_num,
        analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
        config=config,
    )
