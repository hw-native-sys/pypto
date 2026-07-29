# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""High-level API functions for PyPTO IR compilation."""

import logging
import os
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple

from pypto.backend import BackendType
from pypto.backend.pto_backend import PartialCodegenError, generate
from pypto.compile_profiling import CompileProfiler, get_active_profiler
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes as _passes

from .pass_manager import OptimizationStrategy, PassDumpLevel, PassManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .compiled_program import CompiledProgram
    from .distributed_compiled_program import DistributedCompiledProgram


def _write_files(files: dict[str, str], output_dir: str) -> None:
    """Write a dict of {relative_path: content} to output_dir."""
    for filepath, content in files.items():
        full_path = os.path.join(output_dir, filepath)
        file_dir = os.path.dirname(full_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)


def _backend_type_for_platform(platform: str | None, fallback: BackendType) -> BackendType:
    """Return the codegen backend selected by a runtime platform string."""
    if platform is None:
        return fallback
    if platform in ("a2a3", "a2a3sim"):
        return BackendType.Ascend910B
    if platform in ("a5", "a5sim"):
        return BackendType.Ascend950
    raise ValueError(f"Invalid platform {platform!r}. Expected 'a2a3sim', 'a2a3', 'a5sim', or 'a5'.")


class _PassPipelineResult(NamedTuple):
    transformed_program: _ir_core.Program
    memory_planner: _passes.MemoryPlanner
    backend_type: BackendType


def _select_backend(*, backend_type: BackendType, platform: str | None) -> BackendType:
    """Select and configure the backend before compilation creates artifacts."""
    effective_backend_type = _backend_type_for_platform(platform, backend_type)
    _backend_core.set_backend_type(effective_backend_type)
    return effective_backend_type


def _validate_pass_context_conflicts(
    *,
    operation: str,
    verification_level: _passes.VerificationLevel | None,
    diagnostic_phase: _passes.DiagnosticPhase | None,
    memory_planner: _passes.MemoryPlanner | None,
) -> _passes.PassContext | None:
    """Reject explicit pass settings that conflict with an active context."""
    outer = _passes.PassContext.current()
    if verification_level is not None and outer is not None:
        raise RuntimeError(
            f"{operation}() was called with verification_level while a PassContext is already active. "
            "Set the verification level on the existing PassContext instead."
        )
    if diagnostic_phase is not None and outer is not None:
        raise RuntimeError(
            f"{operation}() was called with diagnostic_phase while a PassContext is already active. "
            "Set the diagnostic phase on the existing PassContext instead."
        )
    if memory_planner is not None and outer is not None:
        raise RuntimeError(
            f"{operation}() was called with memory_planner while a PassContext is already active. "
            "Set the memory planner on the existing PassContext instead."
        )
    return outer


def _run_pass_pipeline(  # noqa: PLR0913
    program: _ir_core.Program,
    *,
    operation: str,
    strategy: OptimizationStrategy = OptimizationStrategy.Default,
    backend_type: BackendType = BackendType.Ascend910B,
    platform: str | None = None,
    verification_level: _passes.VerificationLevel | None = None,
    diagnostic_phase: _passes.DiagnosticPhase | None = None,
    disabled_diagnostics: _passes.DiagnosticCheckSet | None = None,
    memory_planner: _passes.MemoryPlanner | None = None,
    enable_pypto_l0c_double_buffer: bool | None = None,
    analyze_auto_scopes_for_deps: bool = False,
    extra_instruments: tuple[_passes.PassInstrument, ...] = (),
    inherit_outer_report_instruments: bool = True,
    dump_passes: bool | PassDumpLevel = False,
    passes_dump_dir: str | None = None,
) -> _PassPipelineResult:
    """Resolve pass settings and run the configured pass pipeline."""
    effective_backend_type = _select_backend(backend_type=backend_type, platform=platform)
    outer = _validate_pass_context_conflicts(
        operation=operation,
        verification_level=verification_level,
        diagnostic_phase=diagnostic_phase,
        memory_planner=memory_planner,
    )

    default_disabled = _passes.DiagnosticCheckSet()
    default_disabled.insert(_passes.DiagnosticCheck.UnusedControlFlowResult)
    if outer is not None:
        outer_instruments = list(outer.get_instruments())
        if not inherit_outer_report_instruments:
            outer_instruments = [
                instrument
                for instrument in outer_instruments
                if not isinstance(instrument, _passes.ReportInstrument)
            ]
        instruments = outer_instruments + list(extra_instruments)
        vlevel = verification_level if verification_level is not None else outer.get_verification_level()
        dphase = diagnostic_phase if diagnostic_phase is not None else outer.get_diagnostic_phase()
        disabled = (
            disabled_diagnostics if disabled_diagnostics is not None else outer.get_disabled_diagnostics()
        )
        mplan = memory_planner if memory_planner is not None else outer.get_memory_planner()
        dbc_flag = (
            enable_pypto_l0c_double_buffer
            if enable_pypto_l0c_double_buffer is not None
            else outer.get_enable_pypto_l0c_double_buffer()
        )
    else:
        instruments = list(extra_instruments)
        vlevel = (
            verification_level if verification_level is not None else _passes.get_default_verification_level()
        )
        dphase = diagnostic_phase if diagnostic_phase is not None else _passes.get_default_diagnostic_phase()
        disabled = disabled_diagnostics if disabled_diagnostics is not None else default_disabled
        mplan = memory_planner if memory_planner is not None else _passes.MemoryPlanner.PYPTO
        dbc_flag = enable_pypto_l0c_double_buffer if enable_pypto_l0c_double_buffer is not None else False
    ctx = _passes.PassContext(instruments, vlevel, dphase, disabled, mplan, dbc_flag)

    if mplan == _passes.MemoryPlanner.PTOAS:
        logger.warning(
            "memory_planner=PTOAS: skipping PyPTO MemoryReuse + AllocateMemoryAddr; ptoas "
            "PlanMemory (--pto-level=level2) owns lifetime reuse and address assignment. "
            "MaterializeSemanticAliases still runs so semantics-required aliasing (loop-carried "
            "accumulators, in-place ops) is preserved as a shared tile_buf handle. The "
            "Ascend910B load + tpop_from_aic in-place hazard guard and reserve-buffer base "
            "resolution are deferred to ptoas — verify on-device."
        )

    prof = get_active_profiler()
    passes_stage = prof.stage("passes") if prof is not None else nullcontext()
    with ctx:
        pm = PassManager.get_strategy(
            strategy,
            analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
        )
        with passes_stage:
            transformed_program = pm.run_passes(
                program,
                dump_ir=dump_passes,
                output_dir=passes_dump_dir,
            )

    return _PassPipelineResult(transformed_program, mplan, effective_backend_type)


def compile(  # noqa: PLR0913
    program: _ir_core.Program,
    output_dir: str | None = None,
    strategy: OptimizationStrategy = OptimizationStrategy.Default,
    dump_passes: bool | PassDumpLevel = True,
    backend_type: BackendType = BackendType.Ascend910B,
    skip_ptoas: bool = False,
    verification_level: _passes.VerificationLevel | None = None,
    diagnostic_phase: _passes.DiagnosticPhase | None = None,
    disabled_diagnostics: _passes.DiagnosticCheckSet | None = None,
    memory_planner: _passes.MemoryPlanner | None = None,
    enable_pypto_l0c_double_buffer: bool | None = None,
    profiling: bool = False,
    platform: str | None = None,
    distributed_config: Any = None,
    analyze_auto_scopes_for_deps: bool = False,
) -> "CompiledProgram | DistributedCompiledProgram":
    """Compile a Program through passes and codegen.

    This function provides a complete compilation pipeline that:
    1. Runs optimization passes via PassManager
    2. Optionally dumps IR before and after each pass (if dump_passes=True)
    3. Generates code via selected backend
    4. Saves all artifacts to a unified output directory

    Args:
        program: Input Program to compile
        output_dir: Output directory. When None, defaults to
            ``<base>/<program_name>_<timestamp>``, where ``<base>`` is the
            ``PYPTO_PROG_BUILD_DIR`` environment variable if set (and
            non-empty), else ``build_output``.
        strategy: Optimization strategy to use (default: Default)
        dump_passes: Per-pass IR dump control. A ``PassDumpLevel``
            (``NONE`` / ``CONCISE`` / ``EXPLICIT``) or a ``bool``
            (``True`` -> ``CONCISE``, ``False`` -> ``NONE``). ``EXPLICIT`` makes
            each dump self-describing for tile layouts and distributed window
            buffers (issue #2088). Default: ``True`` (``CONCISE``).
        backend_type: Backend type for passes and codegen (default: Ascend910B)
        skip_ptoas: Skip the ptoas compilation step and emit raw MLIR (.pto) files
            instead of compiled C++ kernel wrappers.
        verification_level: Override verification level for this compilation via
            PassContext. None uses the default (Basic, or PYPTO_VERIFY_LEVEL env var).
        diagnostic_phase: Override the diagnostic phase gate for this compilation
            via PassContext. None uses the default (PrePipeline, or
            PYPTO_WARNING_LEVEL env var). Setting to None silences warnings AND
            performance hints; finer-grained control uses ``disabled_diagnostics``.
        disabled_diagnostics: Set of diagnostic checks to disable (covers both
            warnings and performance hints). None uses the default
            (UnusedControlFlowResult disabled, perf hints enabled).
        memory_planner: Who plans on-chip buffer memory. ``None`` uses the
            default (``MemoryPlanner.PYPTO`` — PyPTO's AllocateMemoryAddr bakes
            physical addresses and ptoas runs at ``--pto-level=level3``).
            ``MemoryPlanner.DSA_RP`` keeps memory planning in PyPTO but replaces
            opportunistic coalescing with capacity-constrained DSA and
            automatically recognized reuse penalties.
            ``MemoryPlanner.PTOAS`` skips the opportunistic lifetime reuse
            (MemoryReuse) and address assignment (AllocateMemoryAddr), emits no
            ``pto.alloc_tile addr``, and lets the ptoas PlanMemory pass do both at
            ``--pto-level=level2``. MaterializeSemanticAliases still runs, so
            semantics-required aliasing (loop-carried accumulators, in-place ops)
            is preserved as a shared ``tile_buf`` handle that ptoas keeps as one
            buffer.
        enable_pypto_l0c_double_buffer: Opt in to dbC=2 (L0C double-buffering)
            under the PyPTO-owned ``PYPTO`` and ``DSA_RP`` planners
            (experimental, default off). ``None`` inherits the setting from an
            active outer ``PassContext`` (else ``False``); has no effect under
            ``PTOAS``, which already emits dbC=2 unconditionally.
        profiling: If True, enable compile profiling that records per-stage
            wall-clock timings.  Results are written to ``output_dir/report/``.
        platform: Target execution platform.  One of ``"a2a3sim"``,
            ``"a2a3"``, ``"a5sim"``, or ``"a5"``.  Defaults to the
            simulator for the given *backend_type*.  When set, it also
            selects the matching codegen backend.
        distributed_config: Optional :class:`DistributedConfig` for L3+
            distributed programs.  When ``None`` (default), auto-detected
            from the program: if L3+ functions are found, a default
            ``DistributedConfig()`` is used.
        analyze_auto_scopes_for_deps: If True, let
            ``AutoDeriveTaskDependencies`` analyze AUTO runtime scopes. The
            default is False to preserve the existing TensorMap-fallback
            behavior unless explicitly enabled. User-written manual scopes are
            skipped: they do not get compiler deps or automatic
            NoDep/OutputExisting direction rewrites.

    Returns:
        A :class:`CompiledProgram` that wraps the output directory and can
        be called with torch tensors.  For backward compatibility it also
        behaves like a path string (``str(result)`` returns the output dir).

    Example:
        >>> from pypto import ir
        >>> compiled = ir.compile(program)
        >>> str(compiled)               # backward-compat: returns output dir path
        >>> compiled(a, b, c)           # in-place style
        >>> c = compiled(a, b)          # return style
        >>> compiled(a, b, c, config=RunConfig(device_id=1))  # specify device
    """
    _select_backend(backend_type=backend_type, platform=platform)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ``or`` (not get's default arg) so an empty-but-set env var
        # (``export PYPTO_PROG_BUILD_DIR=``) still falls back to build_output
        # rather than writing artifacts into the current working directory.
        base = os.environ.get("PYPTO_PROG_BUILD_DIR") or "build_output"
        output_dir = os.path.join(base, f"{program.name}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    _validate_pass_context_conflicts(
        operation="compile",
        verification_level=verification_level,
        diagnostic_phase=diagnostic_phase,
        memory_planner=memory_planner,
    )

    # --- Compile profiling ---------------------------------------------------
    prof = get_active_profiler()
    owns_profiler = False
    if prof is None and profiling:
        prof = CompileProfiler()
        prof.__enter__()
        owns_profiler = True

    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_instrument = _passes.ReportInstrument(report_dir)

    def _stage(name: str) -> AbstractContextManager[Any]:
        if prof is not None:
            return prof.stage(name)
        return nullcontext()

    try:
        pipeline = _run_pass_pipeline(
            program,
            operation="compile",
            strategy=strategy,
            backend_type=backend_type,
            platform=platform,
            verification_level=verification_level,
            diagnostic_phase=diagnostic_phase,
            disabled_diagnostics=disabled_diagnostics,
            memory_planner=memory_planner,
            enable_pypto_l0c_double_buffer=enable_pypto_l0c_double_buffer,
            analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
            extra_instruments=(report_instrument,),
            dump_passes=dump_passes,
            passes_dump_dir=os.path.join(output_dir, "passes_dump"),
        )
        transformed_program = pipeline.transformed_program
        mplan = pipeline.memory_planner
        effective_backend_type = pipeline.backend_type

        # Codegen target selection is owned by the per-backend BackendHandler;
        # any value of the ``BackendType`` enum is a valid PTO codegen target.
        try:
            with _stage("codegen"):
                files = generate(
                    transformed_program,
                    output_dir,
                    skip_ptoas=skip_ptoas,
                    memory_planner=mplan,
                )
        except PartialCodegenError as exc:
            _write_files(exc.files, output_dir)
            raise
        _write_files(files, output_dir)
    finally:
        if owns_profiler and prof is not None:
            prof.__exit__(None, None, None)
            prof.write_report(report_dir)

    from .compiled_program import CompiledProgram  # noqa: PLC0415

    # Detect distributed programs: any function with level >= HOST (Linqu level 3).
    # Use the post-pass program so functions promoted to HOST by outlining
    # (e.g. via ``with pl.at(level=pl.Level.HOST, ...)``) are still detected.
    is_distributed = any(
        f.level is not None and _ir_core.level_to_linqu_level(f.level) >= 3
        for f in transformed_program.functions.values()
    )

    if is_distributed:
        from .distributed_compiled_program import (  # noqa: PLC0415
            DistributedCompiledProgram,
            DistributedConfig,
        )

        if distributed_config is None:
            distributed_config = DistributedConfig()
        return DistributedCompiledProgram(
            transformed_program,
            output_dir,
            backend_type=effective_backend_type,
            platform=platform,
            distributed_config=distributed_config,
        )

    return CompiledProgram(
        program,
        output_dir,
        backend_type=effective_backend_type,
        platform=platform,
    )
