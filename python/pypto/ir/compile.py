# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""High-level API functions for PyPTO IR compilation."""

import os
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pypto.backend import BackendType
from pypto.backend.pto_backend import PartialCodegenError, generate
from pypto.compile_profiling import CompileProfiler, get_active_profiler
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes as _passes

from .pass_manager import OptimizationStrategy, PassManager

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


def compile(  # noqa: PLR0913
    program: _ir_core.Program,
    output_dir: str | None = None,
    strategy: OptimizationStrategy = OptimizationStrategy.Default,
    dump_passes: bool = True,
    backend_type: BackendType = BackendType.Ascend910B,
    skip_ptoas: bool = False,
    verification_level: _passes.VerificationLevel | None = None,
    diagnostic_phase: _passes.DiagnosticPhase | None = None,
    disabled_diagnostics: _passes.DiagnosticCheckSet | None = None,
    profiling: bool = False,
    platform: str | None = None,
    distributed_config: Any = None,
) -> "CompiledProgram | DistributedCompiledProgram":
    """Compile a Program through passes and codegen.

    This function provides a complete compilation pipeline that:
    1. Runs optimization passes via PassManager
    2. Optionally dumps IR before and after each pass (if dump_passes=True)
    3. Generates code via selected backend
    4. Saves all artifacts to a unified output directory

    Args:
        program: Input Program to compile
        output_dir: Output directory (default: build_output/<program_name>_<timestamp>)
        strategy: Optimization strategy to use (default: Default)
        dump_passes: Whether to dump IR after each pass (default: True)
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
        profiling: If True, enable compile profiling that records per-stage
            wall-clock timings.  Results are written to ``output_dir/report/``.
        platform: Target execution platform.  One of ``"a2a3sim"``,
            ``"a2a3"``, ``"a5sim"``, or ``"a5"``.  Defaults to the
            simulator for the given *backend_type*.
        distributed_config: Optional :class:`DistributedConfig` for L3+
            distributed programs.  When ``None`` (default), auto-detected
            from the program: if L3+ functions are found, a default
            ``DistributedConfig()`` is used.

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
    _backend_core.set_backend_type(backend_type)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("build_output", f"{program.name}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    outer = _passes.PassContext.current()
    if verification_level is not None and outer is not None:
        raise RuntimeError(
            "compile() was called with verification_level while a PassContext is already active. "
            "Set the verification level on the existing PassContext instead."
        )
    if diagnostic_phase is not None and outer is not None:
        raise RuntimeError(
            "compile() was called with diagnostic_phase while a PassContext is already active. "
            "Set the diagnostic phase on the existing PassContext instead."
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
    report_instrument.enable_report(_passes.ReportType.Memory, "AllocateMemoryAddr")

    instruments: list[_passes.PassInstrument] = [report_instrument]
    # Resolve effective settings: explicit arg > outer context > global default.
    default_disabled = _passes.DiagnosticCheckSet()
    default_disabled.insert(_passes.DiagnosticCheck.UnusedControlFlowResult)
    if outer is not None:
        instruments = list(outer.get_instruments()) + instruments
        vlevel = verification_level if verification_level is not None else outer.get_verification_level()
        dphase = diagnostic_phase if diagnostic_phase is not None else outer.get_diagnostic_phase()
        disabled = (
            disabled_diagnostics if disabled_diagnostics is not None else outer.get_disabled_diagnostics()
        )
    else:
        vlevel = (
            verification_level if verification_level is not None else _passes.get_default_verification_level()
        )
        dphase = diagnostic_phase if diagnostic_phase is not None else _passes.get_default_diagnostic_phase()
        disabled = disabled_diagnostics if disabled_diagnostics is not None else default_disabled
    ctx = _passes.PassContext(instruments, vlevel, dphase, disabled)

    def _stage(name: str) -> AbstractContextManager[Any]:
        if prof is not None:
            return prof.stage(name)
        return nullcontext()

    try:
        with ctx:
            pm = PassManager.get_strategy(strategy)
            passes_dump_dir = os.path.join(output_dir, "passes_dump")
            with _stage("passes"):
                transformed_program = pm.run_passes(program, dump_ir=dump_passes, output_dir=passes_dump_dir)

        # Codegen target selection is owned by the per-backend BackendHandler;
        # any value of the ``BackendType`` enum is a valid PTO codegen target.
        try:
            with _stage("codegen"):
                files = generate(transformed_program, output_dir, skip_ptoas=skip_ptoas)
        except PartialCodegenError as exc:
            _write_files(exc.files, output_dir)
            raise
        _write_files(files, output_dir)

        # Emit the comm-group manifest alongside generated host_orch.py so the
        # runner can re-enter the output directory without holding the live
        # Program. No-op when the program declares no CommGroup.
        from .comm_manifest import emit_comm_manifest  # noqa: PLC0415

        emit_comm_manifest(transformed_program, output_dir)
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
            backend_type=backend_type,
            platform=platform,
            distributed_config=distributed_config,
        )

    return CompiledProgram(
        program,
        output_dir,
        backend_type=backend_type,
        platform=platform,
    )
