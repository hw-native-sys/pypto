# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pass manager for IR transformations."""

import os
import re
from collections.abc import Callable
from enum import Enum

from pypto.compile_profiling import CompileProfiler
from pypto.pypto_core import ir as core_ir
from pypto.pypto_core import passes

from .printer import python_print

# Regex to extract variable name from warning messages like:
#   "Unused variable 'foo' in function 'bar'"
_VAR_NAME_RE = re.compile(r"variable '([^']+)'")


def _format_warnings(
    ir_content: str,
    dump_filename: str,
    warnings: list[passes.Diagnostic],
) -> str:
    """Format warnings with gcc/clang-style source context from the printed IR.

    For each warning, locates the variable's definition line in the printed IR
    and emits a diagnostic pointing at it (file:line:col + source + caret).
    """
    lines = ir_content.splitlines()
    out: list[str] = []

    for d in warnings:
        m = _VAR_NAME_RE.search(d.message)
        if not m:
            # Fallback: no variable name extracted
            out.append(f"{dump_filename}: warning: {d.message} [{d.rule_name}]")
            continue

        var_name = m.group(1)
        # Find the first line where this variable is defined.
        # Patterns: `var:` (annotation), `var =` (assignment), or `var,` / `var ` in
        # multi-assignment like `a, b, c = pl.yield_(...)`.
        found = False
        for lineno_0, line in enumerate(lines):
            idx = line.find(var_name)
            if idx == -1:
                continue
            after = line[idx + len(var_name) :]
            # Must be followed by `:`, ` =`, `,`, or end-of-content (stripped)
            if not (
                after.startswith(":")
                or after.startswith(" =")
                or after.startswith(",")
                or after.lstrip() == ""
            ):
                continue
            # Verify it's not a substring of a longer identifier
            if idx > 0 and (line[idx - 1].isalnum() or line[idx - 1] == "_"):
                continue

            lineno = lineno_0 + 1  # 1-based
            col = idx + 1  # 1-based
            gutter_w = len(str(lineno))
            out.append(f"{dump_filename}:{lineno}:{col}: warning: {d.message} [{d.rule_name}]")
            out.append(f" {lineno:>{gutter_w}} | {line.rstrip()}")
            out.append(f" {' ' * gutter_w} | {' ' * idx}^{'~' * (len(var_name) - 1)}")
            found = True
            break

        if not found:
            out.append(f"{dump_filename}: warning: {d.message} [{d.rule_name}]")

    return "\n".join(out) + "\n" if out else ""


PassSpec = tuple[str, Callable[[], passes.Pass]]


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""

    Default = "Default"  # Full tensor-oriented PTO pipeline
    DebugTileOptimization = "DebugTileOptimization"  # Debug-only PTO tile pipeline


class PassManager:
    """Manager for organizing and executing IR transformation passes.

    PassManager maintains a sequence of Pass instances for different optimization
    strategies and executes them in order on a given Program. It delegates to
    a C++ PassPipeline for execution. Instrumentation (verification, logging)
    is handled by PassContext — see passes.PassContext.

    Usage:
        # Get a pre-configured strategy
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        result = pm.run_passes(program)

        # With property verification via PassContext
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
            result = pm.run_passes(program)
    """

    # Static storage: strategy -> List of (pass_name, pass_factory) tuples
    _strategy_passes: dict[OptimizationStrategy, list[PassSpec]] = {}

    @classmethod
    def _register_passes(cls):
        """Register all strategy Pass configurations."""
        tensor_prefix_passes: list[PassSpec] = [
            # Eliminate FunctionType.Inline functions by splicing their bodies at
            # every call site. Runs FIRST so no downstream pass observes Inline
            # functions or Calls to them.
            ("InlineFunctions", lambda: passes.inline_functions()),
            ("UnrollLoops", lambda: passes.unroll_loops()),
            ("CtrlFlowTransform", lambda: passes.ctrl_flow_transform()),
            ("ConvertToSSA", lambda: passes.convert_to_ssa()),
            # Propagate scalar constants (e.g. `CHUNK_K: Scalar[INDEX] = 512`)
            # into downstream expression and type-annotation uses before tile
            # lowering inspects them. Runs post-SSA to exploit single-definition.
            ("Simplify", lambda: passes.simplify()),
            ("NormalizeStmtStructure", lambda: passes.normalize_stmt_structure()),
            ("FlattenCallExpr", lambda: passes.flatten_call_expr()),
        ]
        tensor_only_passes: list[PassSpec] = [
            ("SplitChunkedLoops", lambda: passes.split_chunked_loops()),
            ("InterchangeChunkLoops", lambda: passes.interchange_chunk_loops()),
            ("OutlineHierarchyScopes", lambda: passes.outline_hierarchy_scopes()),
            ("OutlineIncoreScopes", lambda: passes.outline_incore_scopes()),
            ("OutlineClusterScopes", lambda: passes.outline_cluster_scopes()),
            ("ConvertTensorToTileOps", lambda: passes.convert_tensor_to_tile_ops()),
            ("OptimizeOrchTensors", lambda: passes.optimize_orch_tensors()),
        ]
        tile_pto_passes: list[PassSpec] = [
            ("LowerCompositeOps", lambda: passes.lower_composite_ops()),
            ("FlattenTileNdTo2D", lambda: passes.flatten_tile_nd_to_2d()),
            ("AutoTileMatmulL0", lambda: passes.auto_tile_matmul_l0()),
            ("InferTileMemorySpace", lambda: passes.infer_tile_memory_space()),
            ("LowerTransposeLoadParamLayout", lambda: passes.lower_transpose_load_param_layout()),
            ("ResolveBackendOpLayouts", lambda: passes.resolve_backend_op_layouts()),
            ("ExpandMixedKernel", lambda: passes.expand_mixed_kernel()),
            ("InjectGMPipeBuffer", lambda: passes.inject_gm_pipe_buffer()),
            ("SplitVectorKernel", lambda: passes.split_vector_kernel()),
            ("NormalizeReturnOrder", lambda: passes.normalize_return_order()),
            ("LowerPipelineLoops", lambda: passes.lower_pipeline_loops()),
            ("CanonicalizeIOOrder", lambda: passes.canonicalize_io_order()),
            # MaterializeTensorStrides fills empty stride slots on every
            # TensorView with packed canonical strides (RFC #1300 §2.4).
            # Active in the default pipeline starting at P6 — by this point
            # LowerTransposeLoadParamLayout has produced canonical-form DN
            # parameters, so the materialized strides match the IR shape
            # without going through the legacy `dn_swap` codegen path.
            ("MaterializeTensorStrides", lambda: passes.materialize_tensor_strides()),
            ("InitMemRef", lambda: passes.init_mem_ref()),
            ("MemoryReuse", lambda: passes.memory_reuse()),
            ("LegalizePTOBufferReuse", lambda: passes.legalize_pto_buffer_reuse()),
            ("AllocateMemoryAddr", lambda: passes.allocate_memory_addr()),
            ("FoldNoOpReshape", lambda: passes.fold_no_op_reshape()),
            ("FuseCreateAssembleToSlice", lambda: passes.fuse_create_assemble_to_slice()),
            ("DeriveCallDirections", lambda: passes.derive_call_directions()),
            ("Simplify", lambda: passes.simplify()),
        ]
        cls._strategy_passes = {
            OptimizationStrategy.Default: tensor_prefix_passes + tensor_only_passes + tile_pto_passes,
            OptimizationStrategy.DebugTileOptimization: tensor_prefix_passes + tile_pto_passes,
        }

    @classmethod
    def get_strategy(
        cls,
        strategy: OptimizationStrategy = OptimizationStrategy.Default,
    ) -> "PassManager":
        """Get a PassManager configured for the specified strategy.

        Args:
            strategy: The optimization strategy to use (default: Default)

        Returns:
            A PassManager instance configured with the appropriate passes
        """
        if not cls._strategy_passes:
            cls._register_passes()
        return cls(strategy)

    def __init__(self, strategy: OptimizationStrategy):
        """Initialize PassManager with a specific strategy.

        Args:
            strategy: The optimization strategy to use
        """
        self.strategy = strategy
        self.passes: list[passes.Pass] = []
        self.pass_names: list[str] = []

        # Build pass list
        for pass_name, pass_factory in self._strategy_passes[strategy]:
            self.passes.append(pass_factory())
            self.pass_names.append(pass_name)

        # Build C++ PassPipeline
        self._pipeline = passes.PassPipeline()
        for p in self.passes:
            self._pipeline.add_pass(p)

    def run_passes(
        self,
        input_ir: core_ir.Program,
        dump_ir: bool = False,
        output_dir: str | None = None,
        prefix: str = "pl",
    ) -> core_ir.Program:
        """Execute all passes in sequence on a Program.

        Args:
            input_ir: Input Program to transform
            dump_ir: Whether to dump IR after each pass (default: False)
            output_dir: Directory to dump IR files. Required when dump_ir=True.
            prefix: Module prefix for python_print (default: 'pl')

        Returns:
            Transformed Program after all passes have been applied

        Raises:
            ValueError: If dump_ir=True but output_dir is None
        """
        if not dump_ir:
            prof = CompileProfiler.current()
            if prof is not None:
                return self._run_with_profiling(input_ir, prof)
            return self._pipeline.run(input_ir)

        # Dump mode: validate parameters, use CallbackInstrument for IR dumping
        if output_dir is None:
            raise ValueError("output_dir is required when dump_ir=True")

        if not isinstance(input_ir, core_ir.Program):
            raise ValueError("dump_ir mode only supports Program input")

        os.makedirs(output_dir, exist_ok=True)

        # Save frontend IR
        frontend_path = os.path.join(output_dir, "00_frontend.py")
        with open(frontend_path, "w") as f:
            content = python_print(input_ir, prefix=prefix)
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")

        # Use instrument for IR dumping -- verification handled by C++ pipeline.
        # We index self.pass_names (Python-side names from _register_passes) rather than
        # _pass_obj.get_name() because registered names may differ from C++ names.
        pass_index = 0

        # Resolve diagnostic checks once for post-pass dump.
        ctx = passes.PassContext.current()
        if ctx:
            disabled = ctx.get_disabled_diagnostics()
        else:
            # Match PassContext default: disable UnusedControlFlowResult
            disabled = passes.DiagnosticCheckSet()
            disabled.insert(passes.DiagnosticCheck.UnusedControlFlowResult)
        all_checks = passes.DiagnosticCheckRegistry.get_all_checks()
        effective_checks = all_checks.difference(disabled)

        prof = CompileProfiler.current()
        stage_open = False

        def before_pass_profiling(_pass_obj: passes.Pass, _program: core_ir.Program) -> None:
            nonlocal stage_open
            if prof is not None:
                prof._begin_stage(self.pass_names[pass_index])
                stage_open = True

        def after_pass(_pass_obj: passes.Pass, program: core_ir.Program) -> None:
            nonlocal pass_index, stage_open
            pass_name = self.pass_names[pass_index]
            stem = f"{pass_index + 1:02d}_after_{pass_name}"

            # Dump IR
            dump_path = os.path.join(output_dir, f"{stem}.py")
            with open(dump_path, "w") as f:
                content = python_print(program, prefix=prefix)
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")

            # Dump per-pass warnings alongside the IR
            if not effective_checks.empty():
                diags = passes.DiagnosticCheckRegistry.run_checks(
                    effective_checks, passes.DiagnosticPhase.POST_PASS, program
                )
                warn_diags = [d for d in diags if d.severity == passes.DiagnosticSeverity.Warning]
                if warn_diags:
                    dump_filename = os.path.relpath(os.path.join(output_dir, f"{stem}.py"))
                    formatted = _format_warnings(content, dump_filename, warn_diags)
                    warn_path = os.path.join(output_dir, f"{stem}.log")
                    with open(warn_path, "w") as f:
                        f.write(formatted)

            if prof is not None and stage_open:
                prof._end_stage()
                stage_open = False
            pass_index += 1

        extra_instruments: list[passes.PassInstrument] = []
        dump_instrument = passes.CallbackInstrument(after_pass=after_pass, name="IRDump")
        extra_instruments.append(dump_instrument)

        if prof is not None:
            timing_instrument = passes.CallbackInstrument(
                before_pass=before_pass_profiling, name="PipelineProfilingBeforePass"
            )
            extra_instruments.insert(0, timing_instrument)

        # Compose dump instrument with any outer context's instruments and settings.
        # C++ pipeline handles pre-pipeline warnings (LOG_WARN); post-pass warnings
        # are dumped to files by the Python callback above. Override an outer
        # PostPass/Both setting to PrePipeline only to avoid double-executing
        # post-pass warnings; preserve None (explicit silence) and PostPipeline
        # so callers' diagnostic intent isn't reset.
        outer_instruments = list(ctx.get_instruments()) if ctx else []
        level = ctx.get_verification_level() if ctx else passes.get_default_verification_level()
        outer_phase = ctx.get_diagnostic_phase() if ctx else passes.get_default_diagnostic_phase()
        if outer_phase == passes.DiagnosticPhase.POST_PASS:
            inner_phase = passes.DiagnosticPhase.PRE_PIPELINE
        else:
            inner_phase = outer_phase

        with passes.PassContext([*outer_instruments, *extra_instruments], level, inner_phase, disabled):
            try:
                return self._pipeline.run(input_ir)
            finally:
                if stage_open and prof is not None:
                    prof._end_stage()

    def _run_with_profiling(self, input_ir: core_ir.Program, prof: CompileProfiler) -> core_ir.Program:
        """Run the pipeline with per-pass timing recorded into *prof*."""
        pass_index = 0
        stage_open = False

        def before_pass(_pass_obj: passes.Pass, _program: core_ir.Program) -> None:
            nonlocal pass_index, stage_open
            prof._begin_stage(self.pass_names[pass_index])
            stage_open = True

        def after_pass(_pass_obj: passes.Pass, _program: core_ir.Program) -> None:
            nonlocal pass_index, stage_open
            if stage_open:
                prof._end_stage()
                stage_open = False
            pass_index += 1

        timing_instrument = passes.CallbackInstrument(
            before_pass=before_pass, after_pass=after_pass, name="PipelineProfiling"
        )
        ctx = passes.PassContext.current()
        outer_instruments = list(ctx.get_instruments()) if ctx else []
        level = ctx.get_verification_level() if ctx else passes.get_default_verification_level()
        dphase = ctx.get_diagnostic_phase() if ctx else passes.get_default_diagnostic_phase()
        if ctx:
            disabled = ctx.get_disabled_diagnostics()
        else:
            disabled = passes.DiagnosticCheckSet()
            disabled.insert(passes.DiagnosticCheck.UnusedControlFlowResult)

        with passes.PassContext([*outer_instruments, timing_instrument], level, dphase, disabled):
            try:
                return self._pipeline.run(input_ir)
            finally:
                if stage_open:
                    prof._end_stage()

    def get_pass_names(self) -> list[str]:
        """Get the names of all passes in this manager.

        Returns:
            List of pass names assigned during registration
        """
        return self.pass_names


# Initialize the pass registry when the module is loaded
PassManager._register_passes()
