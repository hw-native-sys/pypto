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

class WarningLevel(Enum):
    """Controls automatic warning checks in PassPipeline."""

    NONE = ...
    PRE_PIPELINE = ...
    POST_PASS = ...
    BOTH = ...

class WarningCheck(Enum):
    """Identifies a specific warning check."""

    UnusedVariable = ...
    UnusedControlFlowResult = ...

class WarningCheckSet:
    """A set of warning checks backed by a bitset."""

    def __init__(self) -> None: ...
    def insert(self, check: WarningCheck) -> None: ...
    def remove(self, check: WarningCheck) -> None: ...
    def contains(self, check: WarningCheck) -> bool: ...
    def empty(self) -> bool: ...
    def difference(self, other: WarningCheckSet) -> WarningCheckSet: ...
    def to_list(self) -> list[WarningCheck]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...

class WarningVerifierRegistry:
    """Registry of warning verifiers."""

    @staticmethod
    def run_checks(checks: WarningCheckSet, program: Program) -> list[Diagnostic]: ...
    @staticmethod
    def get_all_checks() -> WarningCheckSet: ...

def get_default_warning_level() -> WarningLevel:
    """Get the default warning level (from PYPTO_WARNING_LEVEL env var, default: PrePipeline)."""

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

class WarningInstrument(PassInstrument):
    """Instrument that runs warning checks before/after passes."""

    def __init__(
        self,
        phase: WarningLevel = WarningLevel.PRE_PIPELINE,
        checks: WarningCheckSet = ...,
    ) -> None:
        """Create a warning instrument with optional phase and check set."""
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

class PassContext:
    """Context that holds instruments and pass configuration.

    When active, Pass.__call__ will run the context's instruments
    before/after each pass execution. Also controls automatic
    verification and warning levels for PassPipeline.
    """

    def __init__(
        self,
        instruments: list[PassInstrument],
        verification_level: VerificationLevel = VerificationLevel.BASIC,
        warning_level: WarningLevel = WarningLevel.PRE_PIPELINE,
        disabled_warnings: WarningCheckSet = ...,  # default: {UnusedControlFlowResult}
    ) -> None:
        """Create a PassContext with instruments, verification level, warning level, and disabled warnings."""
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

    def get_warning_level(self) -> WarningLevel:
        """Get the warning level for this context."""
        ...

    def get_disabled_warnings(self) -> WarningCheckSet:
        """Get the disabled warning checks."""
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

def insert_sync() -> Pass:
    """Create an insert sync pass."""

def legalize_pto_buffer_reuse() -> Pass:
    """Create a PTO buffer reuse legalisation pass."""

def allocate_memory_addr() -> Pass:
    """Create an allocate memory address pass."""

def fuse_create_assemble_to_slice() -> Pass:
    """Fuse tensor.create + tensor.assemble into tensor.slice in Orchestration functions."""

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

def split_chunked_loops() -> Pass:
    """Create a pass that splits chunked loops into nested loops."""

def interchange_chunk_loops() -> Pass:
    """Create a pass that interchanges chunk loops and inserts InCore scopes."""

def unroll_loops() -> Pass:
    """Create a loop unrolling pass that expands ForKind.Unroll loops at compile time."""

def ctrl_flow_transform() -> Pass:
    """Create a control flow structuring pass (eliminate break/continue)."""

def convert_to_ssa() -> Pass:
    """Create an SSA conversion pass."""

def outline_incore_scopes() -> Pass:
    """Create a pass that outlines InCore scopes."""

def outline_cluster_scopes() -> Pass:
    """Create a pass that outlines Cluster scopes into Group functions."""

def outline_hierarchy_scopes() -> Pass:
    """Create a pass that outlines Hierarchy scopes into level/role functions."""

def convert_tensor_to_tile_ops() -> Pass:
    """Create a pass that converts tensor ops to tile ops in InCore functions."""

def optimize_orch_tensors() -> Pass:
    """Create a pass that optimizes tensor buffer usage in orchestration and InCore functions."""

def flatten_tile_nd_to_2d() -> Pass:
    """Create a pass that flattens ND tile ops to 2D in InCore functions."""

def infer_tile_memory_space() -> Pass:
    """Create a pass that infers memory_space for TileType variables in InCore functions."""

def resolve_transpose_layout() -> Pass:
    """Create a pass that resolves transpose layout for tile.load with transpose=True."""

def resolve_backend_op_layouts() -> Pass:
    """Create a pass that repairs backend-required layouts for constrained tile ops."""

def expand_mixed_kernel() -> Pass:
    """Create a pass that expands mixed InCore functions into AIC + AIV + Group."""

def split_vector_kernel() -> Pass:
    """Create a pass that splits vector kernels based on SplitMode."""

def simplify() -> Pass:
    """Create a pass that simplifies expressions and statements using algebraic rules and bound analysis."""

def flatten_call_expr() -> Pass:
    """Create a pass that flattens nested call expressions."""

def normalize_stmt_structure() -> Pass:
    """Create a pass that normalizes statement structure."""

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

class Diagnostic:
    """Single diagnostic message from verification."""

    severity: DiagnosticSeverity
    rule_name: str
    error_code: int
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

__all__ = [
    "IRProperty",
    "IRPropertySet",
    "VerificationMode",
    "VerificationLevel",
    "WarningLevel",
    "WarningCheck",
    "WarningCheckSet",
    "WarningVerifierRegistry",
    "WarningInstrument",
    "get_verified_properties",
    "get_default_verification_level",
    "get_default_warning_level",
    "get_default_verify_properties",
    "get_structural_properties",
    "verify_properties",
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
    "legalize_pto_buffer_reuse",
    "insert_sync",
    "allocate_memory_addr",
    "fuse_create_assemble_to_slice",
    "VerificationError",
    "SSAErrorType",
    "TypeCheckErrorType",
    "split_chunked_loops",
    "interchange_chunk_loops",
    "unroll_loops",
    "ctrl_flow_transform",
    "convert_to_ssa",
    "outline_incore_scopes",
    "outline_cluster_scopes",
    "outline_hierarchy_scopes",
    "convert_tensor_to_tile_ops",
    "optimize_orch_tensors",
    "flatten_tile_nd_to_2d",
    "infer_tile_memory_space",
    "resolve_transpose_layout",
    "resolve_backend_op_layouts",
    "normalize_return_order",
    "expand_mixed_kernel",
    "split_vector_kernel",
    "simplify",
    "flatten_call_expr",
    "normalize_stmt_structure",
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
