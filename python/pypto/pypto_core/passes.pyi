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

from pypto.pypto_core.ir import Program, Span

class IRProperty(Enum):
    """Verifiable IR properties."""

    SSAForm = ...
    TypeChecked = ...
    NoNestedCalls = ...
    NormalizedStmtStructure = ...
    FlattenedSingleStmt = ...
    SplitIncoreOrch = ...
    HasMemRefs = ...
    IncoreTileOps = ...
    AllocatedMemoryAddr = ...
    MixedKernelExpanded = ...
    ClusterOutlined = ...
    TileOps2D = ...
    TileMemoryInferred = ...
    BreakContinueValid = ...

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
    verification level for PassPipeline.
    """

    def __init__(
        self,
        instruments: list[PassInstrument],
        verification_level: VerificationLevel = VerificationLevel.BASIC,
    ) -> None:
        """Create a PassContext with instruments and optional verification level."""
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

def basic_memory_reuse() -> Pass:
    """Create a basic memory reuse pass."""

def insert_sync() -> Pass:
    """Create an insert sync pass."""

def allocate_memory_addr() -> Pass:
    """Create an allocate memory address pass."""

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

class TypeCheckErrorType(Enum):
    """Type checking error types."""

    TYPE_KIND_MISMATCH = ...
    DTYPE_MISMATCH = ...
    SHAPE_DIMENSION_MISMATCH = ...
    SHAPE_VALUE_MISMATCH = ...
    SIZE_MISMATCH = ...

def split_chunked_loops() -> Pass:
    """Create a pass that splits chunked loops into nested loops."""

def interchange_chunk_loops() -> Pass:
    """Create a pass that interchanges chunk loops and inserts InCore scopes."""

def unroll_loops() -> Pass:
    """Create a loop unrolling pass that expands ForKind.Unroll loops at compile time."""

def convert_to_ssa() -> Pass:
    """Create an SSA conversion pass."""

def outline_incore_scopes() -> Pass:
    """Create a pass that outlines InCore scopes."""

def outline_cluster_scopes() -> Pass:
    """Create a pass that outlines Cluster scopes into Group functions."""

def convert_tensor_to_tile_ops() -> Pass:
    """Create a pass that converts tensor ops to tile ops in InCore functions."""

def flatten_tile_nd_to_2d() -> Pass:
    """Create a pass that flattens ND tile ops to 2D in InCore functions."""

def infer_tile_memory_space() -> Pass:
    """Create a pass that infers memory_space for TileType variables in InCore functions."""

def resolve_transpose_layout() -> Pass:
    """Create a pass that resolves transpose layout for tile.load with transpose=True."""

def expand_mixed_kernel() -> Pass:
    """Create a pass that expands mixed InCore functions into AIC + AIV + Group."""

def flatten_call_expr() -> Pass:
    """Create a pass that flattens nested call expressions."""

def normalize_stmt_structure() -> Pass:
    """Create a pass that normalizes statement structure."""

def flatten_single_stmt() -> Pass:
    """Create a pass that flattens single-statement blocks."""

class NestedCallErrorType(Enum):
    """Nested call verification error types."""

    CALL_IN_CALL_ARGS = ...
    CALL_IN_IF_CONDITION = ...
    CALL_IN_FOR_RANGE = ...
    CALL_IN_BINARY_EXPR = ...
    CALL_IN_UNARY_EXPR = ...

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

__all__ = [
    "IRProperty",
    "IRPropertySet",
    "VerificationMode",
    "VerificationLevel",
    "get_verified_properties",
    "get_default_verification_level",
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
    "basic_memory_reuse",
    "insert_sync",
    "allocate_memory_addr",
    "VerificationError",
    "SSAErrorType",
    "TypeCheckErrorType",
    "split_chunked_loops",
    "interchange_chunk_loops",
    "unroll_loops",
    "convert_to_ssa",
    "outline_incore_scopes",
    "outline_cluster_scopes",
    "convert_tensor_to_tile_ops",
    "flatten_tile_nd_to_2d",
    "infer_tile_memory_space",
    "expand_mixed_kernel",
    "flatten_call_expr",
    "normalize_stmt_structure",
    "flatten_single_stmt",
    "NestedCallErrorType",
    "DiagnosticSeverity",
    "Diagnostic",
    "PropertyVerifierRegistry",
    "run_verifier",
]
