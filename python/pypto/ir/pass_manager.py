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


PassFactory = Callable[[], passes.Pass]


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

    @staticmethod
    def _get_pass_factories(
        strategy: OptimizationStrategy,
        *,
        analyze_auto_scopes_for_deps: bool,
    ) -> tuple[PassFactory, ...]:
        """Build the immutable pass-factory recipe for an optimization strategy."""
        tensor_prefix_passes: tuple[PassFactory, ...] = (
            # Eliminate FunctionType.Inline functions by splicing their bodies at
            # every call site. Runs FIRST so no downstream pass observes Inline
            # functions or Calls to them.
            passes.inline_functions,
            passes.unroll_loops,
            passes.ctrl_flow_transform,
            passes.convert_to_ssa,
            # Propagate scalar constants (e.g. `CHUNK_K: Scalar[INDEX] = 512`)
            # into downstream expression and type-annotation uses before tile
            # lowering inspects them. Runs post-SSA to exploit single-definition.
            passes.simplify,
            passes.normalize_stmt_structure,
            passes.flatten_call_expr,
        )
        tensor_only_passes: tuple[PassFactory, ...] = (
            passes.outline_hierarchy_scopes,
            passes.outline_incore_scopes,
            passes.outline_cluster_scopes,
            passes.convert_tensor_to_tile_ops,
            passes.optimize_orch_tensors,
        )
        tile_pto_passes: tuple[PassFactory, ...] = (
            passes.lower_composite_ops,
            passes.flatten_tile_nd_to_2d,
            passes.auto_tile_matmul_l0,
            passes.canonicalize_tile_slice,
            passes.infer_tile_memory_space,
            passes.resolve_backend_op_layouts,
            # RFC #1300: convert AUTO pl.split mixed InCore functions into the explicit
            # split_aiv form (aiv_shard/aic_gather + halved vector sub-region) so
            # ExpandMixedKernel folds them into split-stamped tpush/tpop uniformly. This
            # is the live auto-split lowering path; after it runs every split function
            # reaches SplitVectorKernel already split_aiv-marked, so SplitVectorKernel
            # only stamps attrs. Runs immediately before ExpandMixedKernel.
            passes.lower_auto_vector_split,
            passes.expand_mixed_kernel,
            passes.inject_gm_pipe_buffer,
            passes.split_vector_kernel,
            # Copy each cross-core tpop's split/pipe-id onto its matching tfree op so
            # codegen reads them from the op (no codegen-side tpop lookup table). Runs
            # right after SplitVectorKernel finalizes split on tpops and before
            # SkewCrossCorePipeline clones tpop/tfree pairs (so clones carry split).
            passes.stamp_tfree_split,
            passes.normalize_return_order,
            passes.skew_cross_core_pipeline,
            passes.lower_pipeline_loops,
            passes.canonicalize_io_order,
            # MaterializeTensorStrides fills empty stride slots on every
            # TensorView with packed canonical strides (RFC #1300 §2.4).
            passes.materialize_tensor_strides,
            passes.init_mem_ref,
            # MaterializeSemanticAliases forces loop-carried / in-place buffers to
            # share one MemRef (semantics-required aliasing). It always runs; only
            # the opportunistic lifetime coalescing (MemoryReuse) is skippable when
            # ptoas owns reuse (memory_planner=PTOAS).
            passes.materialize_semantic_aliases,
            # MemoryReuse coalesces independent tile buffers by lifetime; on
            # Ascend910B split-AIV it also avoids the load + tpop_from_aic in-place
            # hazard so a separate legalisation pass is no longer needed.
            passes.memory_reuse,
            passes.allocate_memory_addr,
            passes.fold_no_op_reshape,
            passes.fuse_create_assemble_to_slice,
            passes.derive_call_directions,
            lambda: passes.auto_derive_task_dependencies(analyze_auto_scopes=analyze_auto_scopes_for_deps),
            passes.expand_manual_phase_fence,
            # First normalize host allreduce calls that omit signal into the
            # explicit internal allreduce(data, signal, op=...) form. Then
            # trace pld.tensor.alloc_window_buffer -> pld.tensor.window ->
            # dispatch(device=r) / allreduce in each host_orch, materialize
            # WindowBuffer back-references on every DistributedTensorType view,
            # and wrap the host_orch body in nested CommDomainScopeStmts.
            # This sequence runs late, immediately before
            # LowerHostTensorCollectives, because host_orch is never
            # tile-lowered and the alloc/window/dispatch/allreduce chain is
            # still discoverable.
            passes.synthesize_allreduce_signals,
            passes.materialize_comm_domain_scopes,
            passes.lower_host_tensor_collectives,
            passes.materialize_dist_tensor_ctx,
            passes.simplify,
            # Insert explicit AUTO RuntimeScopeStmt nodes (function body + for/if
            # bodies) into Orchestration functions so codegen emits PTO2_SCOPE
            # 1:1 from the IR. Runs dead last, after the final Simplify, so no
            # other transform has to reason about the inserted scope wrappers.
            passes.materialize_runtime_scopes,
            # Classify each Orchestration ForStmt iter_arg as a trivial alias or a
            # materialised rebind carry (and size manual-scope TaskId array
            # carries), stamping the plan onto ForStmt.attrs. Runs after
            # MaterializeRuntimeScopes so the classified IR is exactly the IR
            # orchestration codegen lowers.
            passes.classify_iter_arg_carry,
            # Mark manual-scope-local allocations that codegen must hoist out of
            # their pl.manual_scope into the enclosing scope (#1697), stamping the
            # ``hoistable_alloc`` attr onto each enclosing-scope-valid tensor.create
            # sitting directly in a manual-scope body. Runs after
            # MaterializeRuntimeScopes so the manual-scope boundary is an explicit
            # RuntimeScopeStmt(manual=True) edge, not an emit-time indent heuristic.
            passes.hoist_scope_local_allocs,
        )
        if strategy == OptimizationStrategy.Default:
            return tensor_prefix_passes + tensor_only_passes + tile_pto_passes
        if strategy == OptimizationStrategy.DebugTileOptimization:
            return tensor_prefix_passes + tile_pto_passes
        raise ValueError(f"Unsupported optimization strategy: {strategy!r}")

    @classmethod
    def get_strategy(
        cls,
        strategy: OptimizationStrategy = OptimizationStrategy.Default,
        *,
        analyze_auto_scopes_for_deps: bool = False,
    ) -> "PassManager":
        """Get a PassManager configured for the specified strategy.

        Args:
            strategy: The optimization strategy to use (default: Default)
            analyze_auto_scopes_for_deps: If True, enable compiler-derived task
                dependency analysis for AUTO runtime scopes. The default stays
                False so runtime AUTO tracking remains the only AUTO-scope
                dependency mechanism. User-written manual scopes are skipped:
                they do not get compiler deps or automatic NoDep/OutputExisting
                direction rewrites.

        Returns:
            A PassManager instance configured with the appropriate passes
        """
        return cls(
            strategy,
            analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
        )

    def __init__(
        self,
        strategy: OptimizationStrategy,
        *,
        analyze_auto_scopes_for_deps: bool = False,
    ):
        """Initialize PassManager with a specific strategy.

        Args:
            strategy: The optimization strategy to use
            analyze_auto_scopes_for_deps: If True, enable compiler-derived task
                dependency analysis for AUTO runtime scopes.
        """
        self.strategy = strategy
        self.analyze_auto_scopes_for_deps = analyze_auto_scopes_for_deps

        # When the active PassContext selects ptoas as the memory planner, skip
        # the opportunistic lifetime reuse (MemoryReuse) and address assignment
        # (AllocateMemoryAddr) so ptoas PlanMemory owns them (codegen emits no
        # `pto.alloc_tile addr` and ptoas runs at --pto-level=level2).
        # MaterializeSemanticAliases still runs, so semantics-required aliasing
        # (loop-carried accumulators, in-place ops) is preserved as a shared
        # MemRef that codegen renders as one tile_buf handle — ptoas cannot
        # recover that from independent addr-less allocs. Read here because
        # __init__ runs inside the compile() PassContext (see compile.py).
        ctx = passes.PassContext.current()
        # The construction-time planner fixes the pass LIST (MemoryReuse +
        # AllocateMemoryAddr are dropped only for PTOAS). PTOAS-gated pass *behaviour*
        # (AutoTileMatmulL0's dbC=2) reads the planner again at execution time, so
        # run_passes re-asserts the run-time planner still matches this one — otherwise a
        # PassManager built outside PTOAS but run inside a PTOAS context would keep
        # MemoryReuse yet still select dbC=2, coalescing the two co-live L0C accumulators
        # into one shrunk single-buffer tile (see _check_planner_consistency).
        self._construction_planner = ctx.get_memory_planner() if ctx else passes.MemoryPlanner.PYPTO
        skip_mem_planning = self._construction_planner == passes.MemoryPlanner.PTOAS
        _mem_planning_passes = ("MemoryReuse", "AllocateMemoryAddr")

        # The C++ pipeline is the single source of truth for both pass objects
        # and names. Strategy recipes contain factories only; names always come
        # from the constructed Pass instances.
        self._pipeline = passes.PassPipeline()
        pass_factories = self._get_pass_factories(
            strategy,
            analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
        )
        for pass_factory in pass_factories:
            pass_obj = pass_factory()
            if skip_mem_planning and pass_obj.get_name() in _mem_planning_passes:
                continue
            self._pipeline.add_pass(pass_obj)

    @property
    def passes(self) -> tuple[passes.Pass, ...]:
        """Get the pipeline's passes in execution order as an immutable snapshot."""
        return tuple(self._pipeline.get_passes())

    @property
    def pass_names(self) -> list[str]:
        """Get pass names derived from the pipeline's actual Pass instances."""
        return self._pipeline.get_pass_names()

    def _check_planner_consistency(self) -> None:
        """Fail loud if the run-time memory planner differs from the construction-time one.

        The pass LIST is fixed at construction: ``MemoryReuse`` / ``AllocateMemoryAddr``
        are dropped iff the construction-time ``PassContext`` selected PTOAS. But
        PTOAS-gated pass *behaviour* — ``AutoTileMatmulL0``'s dbC=2 selection — reads
        ``GetMemoryPlanner()`` at *execution* time. If a PassManager built outside PTOAS
        is then run inside a PTOAS context, the pipeline still contains ``MemoryReuse``
        yet the chooser selects dbC=2, so ``MemoryReuse`` coalesces the two co-live L0C
        accumulators into one — a shrunk single-buffer tile, the exact regression dbC=2
        exists to avoid. ``compile()`` builds and runs under the same context, so this
        never fires there; it guards misuse (build under one planner, run under another).
        """
        ctx = passes.PassContext.current()
        run_planner = ctx.get_memory_planner() if ctx else passes.MemoryPlanner.PYPTO
        if run_planner != self._construction_planner:
            raise RuntimeError(
                f"PassManager was constructed under memory_planner={self._construction_planner!r} "
                f"(which fixed whether MemoryReuse/AllocateMemoryAddr are in the pipeline) but is "
                f"being run under memory_planner={run_planner!r}. Build the PassManager inside the "
                f"same PassContext it is run in (compile() does this)."
            )

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
            RuntimeError: If the run-time memory planner differs from the one the
                PassManager was constructed under (see _check_planner_consistency)
        """
        self._check_planner_consistency()
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
        # Snapshot the pipeline-derived names for stable callback indexing during
        # this run. Pass names have no independent Python-side storage.
        pass_names = self.pass_names
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
                prof._begin_stage(pass_names[pass_index])
                stage_open = True

        def after_pass(_pass_obj: passes.Pass, program: core_ir.Program) -> None:
            nonlocal pass_index, stage_open
            pass_name = pass_names[pass_index]
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
        # Propagate the outer memory planner: a nested PassContext otherwise resets it
        # to the PYPTO binding default, which silently disables PTOAS-gated pass
        # behaviour (e.g. AutoTileMatmulL0's dbC=2 tile selection reads
        # GetMemoryPlanner() *during* pass execution) whenever the pipeline dumps IR.
        mplan = ctx.get_memory_planner() if ctx else passes.MemoryPlanner.PYPTO
        outer_phase = ctx.get_diagnostic_phase() if ctx else passes.get_default_diagnostic_phase()
        if outer_phase == passes.DiagnosticPhase.POST_PASS:
            inner_phase = passes.DiagnosticPhase.PRE_PIPELINE
        else:
            inner_phase = outer_phase

        with passes.PassContext(
            [*outer_instruments, *extra_instruments], level, inner_phase, disabled, mplan
        ):
            try:
                return self._pipeline.run(input_ir)
            finally:
                if stage_open and prof is not None:
                    prof._end_stage()

    def _run_with_profiling(self, input_ir: core_ir.Program, prof: CompileProfiler) -> core_ir.Program:
        """Run the pipeline with per-pass timing recorded into *prof*."""
        pass_names = self.pass_names
        pass_index = 0
        stage_open = False

        def before_pass(_pass_obj: passes.Pass, _program: core_ir.Program) -> None:
            nonlocal pass_index, stage_open
            prof._begin_stage(pass_names[pass_index])
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
        # Propagate the outer memory planner (see run_passes) so profiling doesn't
        # silently reset it to PYPTO and disable PTOAS-gated pass behaviour.
        mplan = ctx.get_memory_planner() if ctx else passes.MemoryPlanner.PYPTO
        dphase = ctx.get_diagnostic_phase() if ctx else passes.get_default_diagnostic_phase()
        if ctx:
            disabled = ctx.get_disabled_diagnostics()
        else:
            disabled = passes.DiagnosticCheckSet()
            disabled.insert(passes.DiagnosticCheck.UnusedControlFlowResult)

        with passes.PassContext([*outer_instruments, timing_instrument], level, dphase, disabled, mplan):
            try:
                return self._pipeline.run(input_ir)
            finally:
                if stage_open:
                    prof._end_stage()

    def get_pass_names(self) -> list[str]:
        """Get the names of all passes in this manager.

        Returns:
            Pass names derived from the underlying pipeline
        """
        return self._pipeline.get_pass_names()
