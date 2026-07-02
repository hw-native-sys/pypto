# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Type stubs for PyPTO IR Pass transformations."""

from collections.abc import Callable
from enum import Enum
from types import TracebackType
from typing import overload

from pypto.pypto_core.ir import Function, Program, Span, Stmt

class IRProperty(Enum):
    """Verifiable IR properties."""

    SSAForm = ...
    TypeChecked = ...
    NoNestedCalls = ...
    NormalizedStmtStructure = ...
    NoRedundantBlocks = ...
    SplitIncoreOrch = ...
    HasMemRefs = ...
    IncoreTileOps = ...
    AllocatedMemoryAddr = ...
    MixedKernelExpanded = ...
    ClusterOutlined = ...
    HierarchyOutlined = ...
    TileOps2D = ...
    TileMemoryInferred = ...
    BreakContinueValid = ...
    UseAfterDef = ...
    StructuredCtrlFlow = ...
    VectorKernelSplit = ...
    OutParamNotShadowed = ...
    NoNestedInCore = ...
    InOutUseValid = ...
    PipelineLoopValid = ...
    PipelineResolved = ...
    UnrollResolved = ...
    CallDirectionsResolved = ...
    TileTypeCoherence = ...
    InlineFunctionsEliminated = ...
    OrchestrationReferencesResolved = ...
    TensorViewCanonical = ...
    ArrayNotEscaped = ...
    CommDomainScopesMaterialized = ...
    RuntimeScopesMaterialized = ...
    AssignTypeSymmetry = ...
    ManualDepsOnSubmitOnly = ...
    ReturnParamsExplicit = ...
    AivSplitValid = ...

class IRPropertySet:
    """A set of IR properties backed by a bitset."""

    def __init__(self) -> None: ...
    def insert(self, prop: IRProperty) -> None: ...
    def remove(self, prop: IRProperty) -> None: ...
    def contains(self, prop: IRProperty) -> bool: ...
    def contains_all(self, other: IRPropertySet) -> bool: ...
    def union_with(self, other: IRPropertySet) -> IRPropertySet: ...
    def intersection(self, other: IRPropertySet) -> IRPropertySet: ...
    def difference(self, other: IRPropertySet) -> IRPropertySet: ...
    def empty(self) -> bool: ...
    def to_list(self) -> list[IRProperty]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    # Unhashable — IRPropertySet is mutable via insert/remove. Setting __hash__
    # to None makes hash() raise TypeError, matching Python's convention for
    # mutable value types (e.g. list, dict).
    __hash__ = None  # type: ignore[assignment]

class VerificationMode(Enum):
    """Controls when property verification runs."""

    NONE = ...
    BEFORE = ...
    AFTER = ...
    BEFORE_AND_AFTER = ...

class VerificationLevel(Enum):
    """Controls automatic verification in PassPipeline."""

    NONE = ...
    BASIC = ...
    ROUNDTRIP = ...

class MemoryPlanner(Enum):
    """Selects who plans on-chip buffer memory."""

    PYPTO = ...
    PTOAS = ...

class DiagnosticPhase(Enum):
    """Controls when DiagnosticInstrument runs registered checks (warnings + perf hints)."""

    NONE = ...
    PRE_PIPELINE = ...
    POST_PASS = ...
    POST_PIPELINE = ...

class DiagnosticCheck(Enum):
    """Identifies a specific diagnostic check."""

    UnusedVariable = ...
    UnusedControlFlowResult = ...
    TileInnermostDimGranularity = ...

class DiagnosticCheckSet:
    """A set of diagnostic checks backed by a bitset."""

    def __init__(self) -> None: ...
    def insert(self, check: DiagnosticCheck) -> None: ...
    def remove(self, check: DiagnosticCheck) -> None: ...
    def contains(self, check: DiagnosticCheck) -> bool: ...
    def empty(self) -> bool: ...
    def difference(self, other: DiagnosticCheckSet) -> DiagnosticCheckSet: ...
    def union_with(self, other: DiagnosticCheckSet) -> DiagnosticCheckSet: ...
    def to_list(self) -> list[DiagnosticCheck]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    # Unhashable for the same reason as IRPropertySet — mutable via insert/remove.
    __hash__ = None  # type: ignore[assignment]

class DiagnosticCheckRegistry:
    """Registry of diagnostic checks (warnings + performance hints)."""

    @staticmethod
    def run_checks(
        checks: DiagnosticCheckSet, phase: DiagnosticPhase, program: Program
    ) -> list[Diagnostic]: ...
    @staticmethod
    def get_all_checks() -> DiagnosticCheckSet: ...
    @staticmethod
    def get_warning_checks() -> DiagnosticCheckSet: ...
    @staticmethod
    def get_perf_hint_checks() -> DiagnosticCheckSet: ...

def get_default_diagnostic_phase() -> DiagnosticPhase:
    """Get the default diagnostic phase (from PYPTO_WARNING_LEVEL env var, default: PrePipeline)."""

def get_verified_properties() -> IRPropertySet:
    """Get the set of properties automatically verified during compilation."""

def get_default_verification_level() -> VerificationLevel:
    """Get the default verification level (from PYPTO_VERIFY_LEVEL env var, default: Basic)."""

def verify_properties(
    properties: IRPropertySet,
    program: Program,
    pass_name: str,
) -> None:
    """Verify properties on a program and throw on errors."""

def get_default_verify_properties() -> IRPropertySet:
    """Get default property set for explicit verification."""

def get_structural_properties() -> IRPropertySet:
    """Get structural invariant properties."""

def verify_tensor_view_canonical(
    program: Program,
    require_materialized: bool = False,
) -> list[Diagnostic]:
    """Run the TensorViewCanonical verifier directly (RFC #1300 P2).

    Args:
        program: Program to verify.
        require_materialized: When False (default — weak mode), accept
            ``stride.empty()`` as implicitly packed canonical. When True
            (strict codegen-entry contract), reject empty stride.

    Returns:
        List of diagnostics; empty if the program is canonical.
    """

class Pass:
    """Opaque pass object. Do not instantiate directly - use factory functions."""

    def __call__(self, program: Program) -> Program:
        """Execute the pass on a program."""

    def get_name(self) -> str:
        """Get the name of the pass."""

    def get_required_properties(self) -> IRPropertySet:
        """Get properties required before this pass can run."""

    def get_produced_properties(self) -> IRPropertySet:
        """Get properties produced after this pass runs."""

    def get_invalidated_properties(self) -> IRPropertySet:
        """Get properties invalidated by this pass."""

class PassInstrument:
    """Abstract base class for pass instrumentation."""

    def get_name(self) -> str:
        """Get the name of this instrument."""
        ...

class VerificationInstrument(PassInstrument):
    """Instrument that verifies IR properties before/after passes."""

    def __init__(self, mode: VerificationMode) -> None:
        """Create a verification instrument with the given mode."""
        ...

class CallbackInstrument(PassInstrument):
    """Instrument that invokes callbacks before/after each pass."""

    def __init__(
        self,
        before_pass: Callable[[Pass, Program], None] | None = None,
        after_pass: Callable[[Pass, Program], None] | None = None,
        name: str = "CallbackInstrument",
    ) -> None:
        """Create a callback instrument with optional before/after callbacks."""
        ...

class DiagnosticInstrument(PassInstrument):
    """Instrument that runs registered diagnostic checks (warnings + perf hints)."""

    def __init__(self, checks: DiagnosticCheckSet = ...) -> None:
        """Create a diagnostic instrument running the given check set."""
        ...

class ReportType(Enum):
    """Type of report to generate."""

    Memory = ...
    """Memory usage per MemorySpace."""

class ReportInstrument(PassInstrument):
    """Instrument that generates reports to files after specified passes."""

    def __init__(self, output_dir: str) -> None:
        """Create a report instrument with output directory."""
        ...

    def enable_report(self, type: ReportType, trigger_pass: str) -> None:
        """Enable a report type after a specific pass."""
        ...

    def get_output_dir(self) -> str:
        """Path of the directory that holds report files."""
        ...

class PassContext:
    """Context that holds instruments and pass configuration.

    When active, Pass.__call__ will run the context's instruments
    before/after each pass execution. Also controls automatic
    verification and the diagnostic channel (warnings + performance
    hints) for PassPipeline.
    """

    def __init__(
        self,
        instruments: list[PassInstrument],
        verification_level: VerificationLevel = VerificationLevel.BASIC,
        diagnostic_phase: DiagnosticPhase = DiagnosticPhase.PRE_PIPELINE,
        disabled_diagnostics: DiagnosticCheckSet = ...,  # default: {UnusedControlFlowResult}
        memory_planner: MemoryPlanner = MemoryPlanner.PYPTO,
    ) -> None:
        """Create a PassContext with instruments and pass configuration (incl. memory planner)."""
        ...

    def __enter__(self) -> PassContext: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def get_verification_level(self) -> VerificationLevel:
        """Get the verification level for this context."""
        ...

    def get_diagnostic_phase(self) -> DiagnosticPhase:
        """Get the diagnostic phase gate for this context."""
        ...

    def get_disabled_diagnostics(self) -> DiagnosticCheckSet:
        """Get the diagnostic checks suppressed by this context."""
        ...

    def get_memory_planner(self) -> MemoryPlanner:
        """Get the memory planner selection for this context."""
        ...

    def get_instruments(self) -> list[PassInstrument]:
        """Get the instruments registered on this context."""
        ...

    @staticmethod
    def current() -> PassContext | None:
        """Get the currently active context, or None if no context is active."""
        ...

class PassPipeline:
    """A pipeline of passes executed in sequence."""

    def __init__(self) -> None:
        """Create an empty pipeline."""

    def add_pass(self, pass_obj: Pass) -> None:
        """Add a pass to the pipeline."""

    def run(self, program: Program) -> Program:
        """Execute all passes in sequence."""

    def get_pass_names(self) -> list[str]:
        """Get names of all passes."""

# Factory functions

def init_mem_ref() -> Pass:
    """Create an init memref pass."""

def memory_reuse() -> Pass:
    """Create a memory reuse pass."""

def allocate_memory_addr() -> Pass:
    """Create an allocate memory address pass."""

def fuse_create_assemble_to_slice() -> Pass:
    """Fuse tensor.create + tensor.assemble into tensor.slice in Orchestration functions."""

def fold_no_op_reshape() -> Pass:
    """Fold no-op tile.reshape assignments into Var-to-Var assignments."""

def normalize_return_order() -> Pass:
    """Create a return order normalization pass."""

class VerificationError:
    """Unified verification error information."""

    error_code: int
    message: str
    span: Span

class SSAErrorType(Enum):
    """SSA verification error types."""

    MULTIPLE_ASSIGNMENT = ...
    NAME_SHADOWING = ...
    MISSING_YIELD = ...
    ITER_ARGS_RETURN_VARS_MISMATCH = ...
    YIELD_COUNT_MISMATCH = ...
    SCOPE_VIOLATION = ...

class TypeCheckErrorType(Enum):
    """Type checking error types."""

    TYPE_KIND_MISMATCH = ...
    DTYPE_MISMATCH = ...
    SHAPE_DIMENSION_MISMATCH = ...
    SHAPE_VALUE_MISMATCH = ...
    SIZE_MISMATCH = ...
    IF_CONDITION_MUST_BE_SCALAR = ...
    FOR_RANGE_MUST_BE_SCALAR = ...
    CONDITION_MUST_BE_BOOL = ...

def unroll_loops() -> Pass:
    """Create a loop unrolling pass that expands ForKind.Unroll loops at compile time."""

def skew_cross_core_pipeline() -> Pass:
    """Create the cross-core pipeline skew pass; runs immediately before ``lower_pipeline_loops``.

    For a mixed cube/vector ``pl.pipeline`` loop (``F > 1``) whose body has both a
    cross-core ``tile.tpush_*`` and ``tile.tpop_*``: a single-round-trip producer-role
    loop runs the producer one iteration ahead (produce(start) prologue + a
    ``ForKind.Sequential`` steady loop pairing produce(k)/consume(k-step) +
    consume(last) epilogue); a consumer-role or multi-round-trip loop demotes to a
    plain ``ForKind.Sequential`` loop (order-preserving — cross-core overlap comes
    from the peer's producer skew). The output is Sequential with no
    ``pipeline_stages`` marker, so ``lower_pipeline_loops`` and ``canonicalize_io_order``
    leave it untouched. Non-cross-core pipeline loops are left intact for
    ``lower_pipeline_loops``.
    """

def lower_pipeline_loops() -> Pass:
    """Create a tile-level lowering pass for ``pl.pipeline(N, stage=F)`` loops.

    Triggers when ``F > 1``. Replicates each loop body F times per outer
    iteration. Static bounds emit a bare ``SeqStmts`` tail flattened into the
    outer scope; dynamic bounds emit a cascaded ``IfStmt`` dispatch on ``rem``
    whose branch bodies are bare ``SeqStmts``. The produced outer loop keeps
    ``ForKind.Pipeline`` and downgrades ``pipeline_stages`` to ``1`` as the
    post-lowering marker for ``CanonicalizeIOOrder``. Keeping the (kind, attr)
    pair together preserves the bidirectional invariant ``PipelineLoopValid``
    so the IR survives print/parse round-trip. Re-running the pass sees
    ``factor == 1`` and skips (idempotent).
    """

def canonicalize_io_order() -> Pass:
    """Create an IO-order canonicalization pass, scoped to pipeline bodies.

    Performs a priority-aware stable topological sort over every ``SeqStmts``
    **inside a ``ForKind.Pipeline`` body** with four tiers: scalar-producing
    assigns (e.g. address arithmetic) lift first, then ``tile.load``, then
    remaining tile compute, and finally ``tile.store`` — all subject to the SSA
    dependency graph. Within replicated regions from ``lower_pipeline_loops``,
    sibling clones' input and output tiles become co-live, enabling ping-pong
    buffering once ``MemoryReuse`` runs. On exit, demotes the outer pipeline
    loop's kind to ``Sequential`` — ``ForKind.Pipeline`` must not survive past
    this pass.
    """

def ctrl_flow_transform() -> Pass:
    """Create a control flow structuring pass (eliminate break/continue)."""

def convert_to_ssa() -> Pass:
    """Create an SSA conversion pass."""

def outline_incore_scopes() -> Pass:
    """Create a pass that outlines InCore scopes."""

def outline_cluster_scopes() -> Pass:
    """Create a pass that outlines Cluster scopes to Group and standalone Spmd scopes to Spmd."""

def outline_hierarchy_scopes() -> Pass:
    """Create a pass that outlines Hierarchy scopes into level/role functions."""

def convert_tensor_to_tile_ops() -> Pass:
    """Create a pass that converts tensor ops to tile ops in InCore functions."""

def optimize_orch_tensors() -> Pass:
    """Create a pass that optimizes tensor buffer usage in orchestration and InCore functions."""

def flatten_tile_nd_to_2d() -> Pass:
    """Create a pass that flattens ND tile ops to 2D in InCore functions."""

def auto_tile_matmul_l0() -> Pass:
    """Create a pass that auto-tiles Mat-resident matmul / matmul_acc into a C-stationary K-loop.

    Rewrites each ``tile.matmul`` or ``tile.matmul_acc`` whose Mat operands
    have static 2D shape into a ``range(0, K, k)`` loop:

    * For ``tile.matmul``, the loop body branches on ``ko == 0`` between
      ``tile.matmul`` (fresh accumulator) and ``tile.matmul_acc``
      (accumulating into the iter-arg).
    * For ``tile.matmul_acc``, every iteration is ``tile.matmul_acc`` with
      the iter-arg init set to the caller's accumulator — the chain is
      uniform from the first iteration so no if-else is needed.

    The L0 tile shape ``(m, n, k)`` is chosen by ``utils.choose_l0_tile``
    from the active backend's L0 capacities. The K-loop is marked
    ``ForKind.Pipeline`` with ``pipeline_stages=2`` so the downstream
    ``LowerPipelineLoops`` pass produces a 2-deep ping-pong on the
    auto-inserted Mat→Left/Right moves. Already-L0-sized matmuls are left
    untouched.

    Supported today: ``tile.matmul`` and ``tile.matmul_acc``;
    ``tile.matmul_bias`` is deferred. Only K tiling; M/N tiling and
    ``K % k != 0`` cases emit a perf hint and skip.
    """

def canonicalize_tile_slice() -> Pass:
    """Create a pass that lowers Mat-resident ``tile.slice`` into ``tile.extract``.

    A ``tile.slice`` whose result tile is ``Mem.Mat`` (e.g. a batch-page slice
    emitted by ``FlattenTileNdTo2D`` when unrolling ``tile.batch_matmul``) has
    no standalone hardware lowering. This pass folds each such slice's offset
    into its consumer:

    * consumed by ``tile.extract`` — the extract reads the slice's source
      directly with the slice offset added into its index;
    * consumed by a ``tile.matmul`` family operand — the operand is replaced
      by a ``tile.extract(target_memory=Left|Right)``.

    The now-dead ``tile.slice`` is dropped, unifying Mat→Left/Right movement
    on ``tile.extract`` / ``pto.textract``.
    """

def infer_tile_memory_space() -> Pass:
    """Create a pass that infers memory_space for TileType variables in InCore functions."""

def materialize_tensor_strides() -> Pass:
    """Create the MaterializeTensorStrides pass (RFC #1300 §2.4).

    Walks every TensorType reachable from the program and rewrites any
    ``view.has_value() && view.stride.empty()`` slot to its packed canonical
    stride per ``BuildLogicalStridesFromLayout``. Bare TensorTypes are left
    untouched. Idempotent. Produces ``TensorViewCanonical`` so the registry
    auto-verifies after the pass runs.
    """

def resolve_backend_op_layouts() -> Pass:
    """Create a pass that repairs backend-required layouts for constrained tile ops."""

def expand_mixed_kernel() -> Pass:
    """Create a pass that expands mixed InCore functions into AIC + AIV + Group."""

def lower_auto_vector_split() -> Pass:
    """Lower AUTO ``pl.split`` mixed InCore functions into the explicit ``split_aiv`` form.

    Inserts ``tile.aiv_shard`` at C->V boundaries and ``tile.aic_gather`` at V->C
    boundaries, halves only the vector sub-region (affinity-gated), injects
    ``get_subblock_idx``, and stamps ``split`` + ``split_aiv`` — all BEFORE
    ExpandMixedKernel folds the reshape ops into split-stamped tpush/tpop.

    This is the live auto-split lowering path: it always runs immediately before
    ExpandMixedKernel, so SplitVectorKernel only stamps attrs for the resulting
    ``split_aiv`` functions.
    """

def inject_gm_pipe_buffer() -> Pass:
    """Create a backend-gated pass that injects ``__gm_pipe_buffer`` for cross-core pipes.

    On backends whose cross-core pipe rides through GM (currently Ascend910B), adds a
    ``__gm_pipe_buffer`` Out-tensor parameter to functions that issue ``initialize_pipe``
    ops and propagates it through callers. Orchestration functions materialize the
    buffer locally via ``tensor.create`` instead of receiving it as a parameter. No-op
    on backends that don't require a GM slot buffer.
    """

def split_vector_kernel() -> Pass:
    """Create a pass that splits vector kernels based on SplitMode."""

def simplify() -> Pass:
    """Create a pass that simplifies expressions and statements using algebraic rules and bound analysis."""

def lower_composite_ops() -> Pass:
    """Decompose composite tile/distributed ops into primitive ops.

    Lowering rules are registered through the composite-lowering registry.
    Today the pass handles ``tile.sin`` / ``tile.cos`` and explicit-signal
    InCore ``pld.tensor.allreduce``. Host-level allreduce is skipped here and
    lowered later by :func:`lower_host_tensor_collectives`.

    FP32-only for the trig rules. Non-FP32 inputs are rejected at
    op-construction time.

    Idempotent: registered rules emit ops that are not themselves in the
    dispatch table, so running the pass twice yields the same IR after the
    first run.
    """

def derive_call_directions() -> Pass:
    """Create a pass that derives per-argument :class:`ir.ArgDirection`.

    Walks each ``Function``'s body and writes ``Call.attrs['arg_directions']``
    based on callee :class:`ir.ParamDirection` and argument origin (function
    param vs. locally allocated tensor vs. scalar). Establishes the
    :class:`IRProperty.CallDirectionsResolved` post-condition, which is then
    auto-verified by the pipeline.
    """

def auto_derive_task_dependencies(analyze_auto_scopes: bool = False) -> Pass:
    """Create a pass that derives compiler-owned runtime-scope task dependencies.

    Runs after :func:`derive_call_directions` and writes
    ``Call.attrs['compiler_manual_dep_edges']`` for RAW/WAR/WAW hazards inside
    analyzed AUTO runtime scopes. User-written manual runtime scopes are skipped.
    AUTO scopes are skipped by default; pass ``analyze_auto_scopes=True`` to
    analyze them without changing their runtime scope mode. Unanalyzable hazards
    keep AUTO tracking with partial compiler deps stripped. User-written
    ``deps=[...]`` entries stay in ``Submit::deps_``, preserving the
    ``ManualDepsOnSubmitOnly`` invariant for codegen.
    """

def expand_manual_phase_fence() -> Pass:
    """Create a pass that inserts dummy TaskId barriers for manual phase fences."""

def flatten_call_expr() -> Pass:
    """Create a pass that flattens nested call expressions."""

def inline_functions() -> Pass:
    """Create a pass that eliminates ``FunctionType.Inline`` functions.

    Splices the body of every ``Inline``-typed function into each call site,
    alpha-renaming locals and substituting formal params with actual args. The
    inline functions are then removed from the program. Runs as the first
    pipeline pass so subsequent passes never observe Inline functions.

    Detects cycles in the Inline → Inline call graph and raises
    :class:`pypto.ValueError`. Multi-return inline functions emit
    ``LHS = MakeTuple([rets...])`` at the call site.
    """

def normalize_stmt_structure() -> Pass:
    """Create a pass that normalizes statement structure."""

def synthesize_allreduce_signals() -> Pass:
    """Synthesize private signal windows for host-level allreduce calls.

    Host orchestration calls written as ``pld.tensor.allreduce(target, op=...)``
    are normalized to the internal explicit-signal form by inserting ordinary
    ``pld.tensor.alloc_window_buffer`` and ``pld.tensor.window`` assignments
    before the call. Existing explicit-signal calls are preserved.
    """

def materialize_comm_domain_scopes() -> Pass:
    """Collect comm domains and materialise them as scope statements.

    For each ``@pl.program`` host_orch function, traces
    ``pld.tensor.alloc_window_buffer → pld.tensor.window → dispatch(device=r)``
    chains, constructs a :class:`WindowBuffer` per alloc, back-fills the
    ``DistributedTensorType.window_buffer_`` field on every ``pld.tensor.window``
    result Var, and wraps the host_orch body in nested
    :class:`CommDomainScopeStmt` nodes (one per inferred comm domain,
    outer = first declared, inner = last).

    Runs late in the default pipeline after
    :func:`synthesize_allreduce_signals` and before
    :func:`lower_host_tensor_collectives`, while the host dispatch chain is
    still intact.
    """

def lower_host_tensor_collectives() -> Pass:
    """Lower host-level ``pld.tensor.allreduce`` calls to builtin collective dispatches."""

def stamp_tfree_split() -> Pass:
    """Copy each cross-core tpop's split/pipe-id onto its matching tfree op.

    A ``system.tfree_to_ai{c,v}`` carries no split/id of its own; those live on
    the matching ``tile.tpop_from_ai{c,v}`` call. This pass stamps them onto the
    tfree op so codegen reads them directly. Covers mixed-kernel and explicit
    AIC/AIV tfrees. Runs late, before codegen.
    """

def materialize_runtime_scopes() -> Pass:
    """Materialize implicit orchestration scopes as explicit RuntimeScopeStmt nodes.

    For every ``FunctionType.Orchestration`` function, inserts AUTO
    ``RuntimeScopeStmt`` (``manual=False``) nodes wrapping the function body and
    each ``ForStmt`` / ``IfStmt`` branch body, while skipping insertion inside a
    manual ``RuntimeScopeStmt`` (the runtime forbids AUTO nested in MANUAL).
    Codegen then emits ``PTO2_SCOPE`` only from ``RuntimeScopeStmt`` nodes, 1:1
    with the IR.

    Runs last in the pipeline (after the final :func:`simplify`) so no other
    transform has to reason about the inserted scopes. Only Orchestration
    functions are touched.
    """

class NestedCallErrorType(Enum):
    """Nested call verification error types."""

    CALL_IN_CALL_ARGS = ...
    CALL_IN_IF_CONDITION = ...
    CALL_IN_FOR_RANGE = ...
    CALL_IN_BINARY_EXPR = ...
    CALL_IN_UNARY_EXPR = ...

class UseAfterDefErrorType(Enum):
    """Use-after-def verification error types."""

    USE_BEFORE_DEF = ...
    """Variable used before any definition in scope."""

class DiagnosticSeverity(Enum):
    """Severity level for diagnostics."""

    Error = ...
    Warning = ...
    PerfHint = ...

class Diagnostic:
    """Single diagnostic message from verification."""

    severity: DiagnosticSeverity
    rule_name: str
    error_code: int
    hint_code: str
    message: str
    span: Span

class PropertyVerifierRegistry:
    """Registry of property verifiers for IR verification."""

    @staticmethod
    def verify(properties: IRPropertySet, program: Program) -> list[Diagnostic]: ...
    @staticmethod
    def verify_or_throw(properties: IRPropertySet, program: Program) -> None: ...
    @staticmethod
    def generate_report(diagnostics: list[Diagnostic]) -> str: ...

def run_verifier(properties: IRPropertySet | None = None) -> Pass:
    """Create a verifier pass. Defaults to get_default_verify_properties() if None."""

class stmt_dependency_analysis:
    """Statement dependency analysis and InOut-use discipline check (RFC #1026 Phase 1)."""

    class StmtDependencyGraph:
        """Dataflow dependency graph over a region's top-level statements."""

        stmts: list[Stmt]
        def get_predecessors(self, stmt: Stmt) -> list[Stmt]:
            """Return the predecessor stmts of the given stmt in region order."""

    @staticmethod
    def build_stmt_dependency_graph(region: Stmt, program: Program | None = None) -> StmtDependencyGraph:
        """Build a dataflow dependency graph over a region's top-level stmts.

        When `program` is provided, the InOut-use discipline is checked first
        and any violation raises `pypto.Error` (VerificationError).
        """

    @staticmethod
    def check_inout_use_discipline(region: Stmt, program: Program) -> None:
        """Enforce the InOut-use discipline.

        Raises `pypto.Error` (VerificationError) on any violation so
        compilation halts rather than proceeding with unsound IR.
        """

class l0_tile_chooser:
    """Closed-form chooser for L0 matmul tile shape (m, n, k)."""

    class L0TileConfig:
        """Inputs to choose_l0_tile: problem dims + hardware + schedule knobs."""

        M: int
        N: int
        K: int
        l0a_bytes: int
        l0b_bytes: int
        l0c_bytes: int
        bytes_a: int
        bytes_b: int
        bytes_c: int
        min_m: int
        min_n: int
        min_k: int
        align_m: int
        align_n: int
        align_k: int
        double_buffer_a: bool
        double_buffer_b: bool
        double_buffer_c: bool
        c_read: bool
        allow_padding: bool
        def __init__(self) -> None: ...

    class L0TileResult:
        """Output of choose_l0_tile: the chosen (m, n, k) plus diagnostics."""

        m: int
        n: int
        k: int
        estimated_traffic_bytes: int
        padded_compute_volume: int
        perf_hint: str

    @staticmethod
    def choose_l0_tile(config: L0TileConfig) -> L0TileResult:
        """Pick an approximately-optimal L0 tile shape (m, n, k)."""

__all__ = [
    "IRProperty",
    "IRPropertySet",
    "VerificationMode",
    "VerificationLevel",
    "MemoryPlanner",
    "DiagnosticPhase",
    "DiagnosticCheck",
    "DiagnosticCheckSet",
    "DiagnosticCheckRegistry",
    "DiagnosticInstrument",
    "get_verified_properties",
    "get_default_verification_level",
    "get_default_diagnostic_phase",
    "get_default_verify_properties",
    "get_structural_properties",
    "verify_properties",
    "verify_tensor_view_canonical",
    "Pass",
    "PassInstrument",
    "VerificationInstrument",
    "CallbackInstrument",
    "ReportType",
    "ReportInstrument",
    "PassContext",
    "PassPipeline",
    "init_mem_ref",
    "memory_reuse",
    "allocate_memory_addr",
    "fuse_create_assemble_to_slice",
    "fold_no_op_reshape",
    "stamp_tfree_split",
    "VerificationError",
    "SSAErrorType",
    "TypeCheckErrorType",
    "unroll_loops",
    "ctrl_flow_transform",
    "convert_to_ssa",
    "outline_incore_scopes",
    "outline_cluster_scopes",
    "outline_hierarchy_scopes",
    "convert_tensor_to_tile_ops",
    "optimize_orch_tensors",
    "flatten_tile_nd_to_2d",
    "auto_tile_matmul_l0",
    "canonicalize_tile_slice",
    "infer_tile_memory_space",
    "materialize_tensor_strides",
    "resolve_backend_op_layouts",
    "normalize_return_order",
    "expand_mixed_kernel",
    "lower_auto_vector_split",
    "inject_gm_pipe_buffer",
    "split_vector_kernel",
    "simplify",
    "lower_composite_ops",
    "flatten_call_expr",
    "inline_functions",
    "normalize_stmt_structure",
    "derive_call_directions",
    "auto_derive_task_dependencies",
    "expand_manual_phase_fence",
    "NestedCallErrorType",
    "UseAfterDefErrorType",
    "DiagnosticSeverity",
    "Diagnostic",
    "PropertyVerifierRegistry",
    "run_verifier",
    "PassProperties",
    "create_function_pass",
    "create_program_pass",
    "stmt_dependency_analysis",
    "skew_cross_core_pipeline",
    "lower_pipeline_loops",
    "canonicalize_io_order",
]

class PassProperties:
    """Property declarations for a pass."""

    required: IRPropertySet
    produced: IRPropertySet
    invalidated: IRPropertySet

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        required: IRPropertySet,
        produced: IRPropertySet,
        invalidated: IRPropertySet,
    ) -> None: ...
    def __init__(self, *args, **kwargs) -> None:
        """Create pass properties, optionally with required/produced/invalidated sets."""

def create_function_pass(
    transform: Callable[[Function], Function],
    name: str = "",
    properties: PassProperties = ...,
) -> Pass:
    """Create a pass from a Python function-level transform.

    The transform receives a Function and returns a (possibly new) Function.
    The pass applies this transform to each function in the program.
    """

def create_program_pass(
    transform: Callable[[Program], Program],
    name: str = "",
    properties: PassProperties = ...,
) -> Pass:
    """Create a pass from a Python program-level transform.

    The transform receives a Program and returns a (possibly new) Program.
    """
