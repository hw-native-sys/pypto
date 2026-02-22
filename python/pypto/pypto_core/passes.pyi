# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Type stubs for PyPTO IR Pass transformations."""

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
    IncoreBlockOps = ...

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

class PassContext:
    """Context that holds instruments, with with-style nesting.

    When active, Pass.__call__ will run the context's instruments
    before/after each pass execution.
    """

    def __init__(self, instruments: list[PassInstrument]) -> None:
        """Create a PassContext with the given instruments."""
        ...

    def __enter__(self) -> PassContext: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
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

def add_alloc() -> Pass:
    """Create an add alloc pass."""

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

def convert_to_ssa() -> Pass:
    """Create an SSA conversion pass."""

def outline_incore_scopes() -> Pass:
    """Create a pass that outlines InCore scopes."""

def convert_tensor_to_block_ops() -> Pass:
    """Create a pass that converts tensor ops to block ops in InCore functions."""

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

class IRVerifier:
    """IR verification system that manages verification rules."""

    def __init__(self) -> None: ...
    @staticmethod
    def create_default() -> IRVerifier: ...
    def enable_rule(self, name: str) -> None: ...
    def disable_rule(self, name: str) -> None: ...
    def is_rule_enabled(self, name: str) -> bool: ...
    def verify(self, program: Program) -> list[Diagnostic]: ...
    def verify_or_throw(self, program: Program) -> None: ...
    @staticmethod
    def generate_report(diagnostics: list[Diagnostic]) -> str: ...

def run_verifier(disabled_rules: list[str] | None = None) -> Pass:
    """Create a verifier pass with configurable rules."""

__all__ = [
    "IRProperty",
    "IRPropertySet",
    "VerificationMode",
    "Pass",
    "PassInstrument",
    "VerificationInstrument",
    "PassContext",
    "PassPipeline",
    "init_mem_ref",
    "basic_memory_reuse",
    "insert_sync",
    "add_alloc",
    "VerificationError",
    "SSAErrorType",
    "TypeCheckErrorType",
    "convert_to_ssa",
    "outline_incore_scopes",
    "convert_tensor_to_block_ops",
    "flatten_call_expr",
    "normalize_stmt_structure",
    "flatten_single_stmt",
    "NestedCallErrorType",
    "DiagnosticSeverity",
    "Diagnostic",
    "IRVerifier",
    "run_verifier",
]
