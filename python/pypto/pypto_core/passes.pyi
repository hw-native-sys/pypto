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

from pypto.pypto_core.ir import Program, Span

class Pass:
    """Opaque pass object. Do not instantiate directly - use factory functions.

    A Pass represents a transformation that can be applied to a Program.
    Pass objects should be created using factory functions (init_mem_ref, etc.)
    rather than being instantiated directly.
    """

    def __call__(self, program: Program) -> Program:
        """Execute the pass on a program.

        Args:
            program: Input Program to transform

        Returns:
            Transformed Program after the pass has been applied
        """

    def run(self, program: Program) -> Program:
        """Execute the pass on a program (backward compatible).

        Args:
            program: Input Program to transform

        Returns:
            Transformed Program after the pass has been applied
        """

# Factory functions with snake_case names

def init_mem_ref() -> Pass:
    """Create an init memref pass.

    Initializes MemRef for all variables in functions.
    Sets memory space to UB by default, or DDR for block.load/block.store operands.

    Returns:
        Pass object that initializes memrefs
    """

def basic_memory_reuse() -> Pass:
    """Create a basic memory reuse pass.

    Uses dependency analysis to identify memory reuse opportunities.
    Variables with non-overlapping lifetimes in the same memory space can
    share MemRef objects.

    Returns:
        Pass object that performs basic memory reuse optimization
    """

def insert_sync() -> Pass:
    """Create an insert sync pass.

    Analyzes data dependencies and inserts synchronization operations
    (sync_src, sync_dst, bar_v, bar_m) for correct execution across hardware pipes.
    Uses the globally configured backend to obtain pipe information.

    Returns:
        Pass object that inserts synchronization operations

    Raises:
        ValueError: If backend type has not been configured
    """

def add_alloc() -> Pass:
    """Create an add alloc pass.

    This pass traverses all TileType variables in each Function and creates alloc operations
    for each unique MemRef. The alloc operations are added at the beginning of the function.

    The pass performs the following steps:
    1. Identifies all TileType variables in the function
    2. Collects all unique MemRef objects from these TileType variables
    3. Creates an alloc operation for each unique MemRef
    4. Prepends these alloc operations to the function body

    Each alloc operation has no input/output arguments but is bound to a MemRef pointer
    to track memory allocation for that specific buffer.

    Returns:
        Pass object that adds alloc operations
    """

class VerificationError:
    """Unified verification error information."""

    error_code: int
    message: str
    span: Span

class SSAErrorType(Enum):
    """SSA verification error types."""

    MULTIPLE_ASSIGNMENT: int
    NAME_SHADOWING: int
    MISSING_YIELD: int

def verify_ssa() -> Pass:
    """Create an SSA verification pass.

    This pass verifies SSA form of IR by checking:
    1. Each variable is assigned only once (MULTIPLE_ASSIGNMENT)
    2. No variable name shadowing across scopes (NAME_SHADOWING)
    3. ForStmt with iter_args must have YieldStmt as last statement (MISSING_YIELD)
    4. IfStmt with return_vars must have YieldStmt in both then and else branches (MISSING_YIELD)

    The pass collects all errors and generates a verification report instead of
    throwing exceptions, allowing detection of all issues in a single run.

    Returns:
        Pass object that performs SSA verification
    """

class TypeCheckErrorType(Enum):
    """Type checking error types."""

    TYPE_KIND_MISMATCH: int
    DTYPE_MISMATCH: int
    SHAPE_DIMENSION_MISMATCH: int
    SHAPE_VALUE_MISMATCH: int
    SIZE_MISMATCH: int

def type_check() -> Pass:
    """Create a type checking pass.

    This pass checks type consistency in control flow constructs:
    1. ForStmt: iter_args initValue, yield values, and return_vars must have matching types
    2. IfStmt: then and else yield values must have matching types
    3. Shape consistency for TensorType and TileType

    The pass collects all errors and generates a type checking report instead of
    throwing exceptions, allowing detection of all issues in a single run.

    Returns:
        Pass object that performs type checking
    """

def convert_to_ssa() -> Pass:
    """Create an SSA conversion pass.

    This pass converts non-SSA IR to SSA form by:
    1. Renaming variables with version suffixes (x -> x_0, x_1, x_2)
    2. Adding phi nodes (return_vars + YieldStmt) for IfStmt control flow divergence
    3. Converting loop-modified variables to iter_args + return_vars pattern

    The pass handles:
    - Straight-line code: multiple assignments to the same variable
    - If statements: variables modified in one or both branches
    - For loops: variables modified inside the loop body
    - Mixed SSA/non-SSA: preserves existing SSA structure while converting non-SSA parts

    Returns:
        Pass object that converts to SSA form
    """

def flatten_call_expr() -> Pass:
    """Create a pass that flattens nested call expressions into three-address code.

    This pass ensures that call expressions do not appear in nested contexts:
    1. Call arguments cannot be calls
    2. If conditions cannot be calls
    3. For loop ranges (start/stop/step) cannot be calls
    4. Binary/unary expression operands cannot be calls

    Nested calls are extracted into temporary variables (named _t0, _t1, etc.)
    and inserted as AssignStmt before the statement containing the nested call.
    For if/for statements, extracted statements are inserted into the last OpStmts
    before the if/for, or a new OpStmts is created if needed.

    Example transformation:
        c = foo(bar(a))  =>  _t0 = bar(a); c = foo(_t0)

    Returns:
        Pass object that flattens nested call expressions
    """

def normalize_stmt_structure() -> Pass:
    """Create a pass that normalizes statement structure.

    This pass ensures IR is in a normalized form:
    1. Function/IfStmt/ForStmt body must be SeqStmts
    2. Consecutive AssignStmt/EvalStmt in SeqStmts are wrapped in OpStmts

    Example transformations:
        Function body = AssignStmt(x, 1)
        => Function body = SeqStmts([OpStmts([AssignStmt(x, 1)])])

        SeqStmts([AssignStmt(a, 1), AssignStmt(b, 2), IfStmt(...)])
        => SeqStmts([OpStmts([AssignStmt(a, 1), AssignStmt(b, 2)]), IfStmt(...)])

    Returns:
        Pass object that normalizes statement structure
    """

def flatten_single_stmt() -> Pass:
    """Create a pass that recursively flattens single-statement blocks.

    This pass simplifies IR by removing unnecessary nesting:
    - SeqStmts with only one statement is replaced by that statement
    - OpStmts with only one statement is replaced by that statement
    - Process is applied recursively

    Example transformations:
        SeqStmts([OpStmts([AssignStmt(x, 1)])])
        => AssignStmt(x, 1)

        SeqStmts([OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])])
        => OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])

    Note: This pass does NOT enforce that Function/IfStmt/ForStmt body must be SeqStmts.
    It will flatten them if they contain only a single statement.

    Returns:
        Pass object that flattens single-statement blocks
    """

class NestedCallErrorType(Enum):
    """Nested call verification error types."""

    CALL_IN_CALL_ARGS: int
    CALL_IN_IF_CONDITION: int
    CALL_IN_FOR_RANGE: int
    CALL_IN_BINARY_EXPR: int
    CALL_IN_UNARY_EXPR: int

class DiagnosticSeverity(Enum):
    """Severity level for diagnostics."""

    Error: int
    Warning: int

class Diagnostic:
    """Single diagnostic message from verification.

    Represents a single issue found during IR verification. Contains information
    about the severity, which rule detected it, the specific error code, a
    human-readable message, and the source location where the issue was found.
    """

    severity: DiagnosticSeverity
    rule_name: str
    error_code: int
    message: str
    span: Span

class IRVerifier:
    """IR verification system that manages verification rules.

    IRVerifier collects verification rules and applies them to programs.
    Rules can be enabled/disabled individually.

    Example:
        >>> verifier = IRVerifier.create_default()
        >>> verifier.disable_rule("TypeCheck")
        >>> diagnostics = verifier.verify(program)
        >>> for d in diagnostics:
        ...     print(f"[{d.severity}] {d.rule_name}: {d.message}")
    """

    def __init__(self) -> None:
        """Create an empty verifier with no rules."""

    @staticmethod
    def create_default() -> IRVerifier:
        """Create a verifier with default built-in rules.

        Returns:
            IRVerifier with SSAVerify and TypeCheck rules enabled
        """

    def enable_rule(self, name: str) -> None:
        """Enable a previously disabled rule.

        Args:
            name: Name of the rule to enable
        """

    def disable_rule(self, name: str) -> None:
        """Disable a rule.

        Args:
            name: Name of the rule to disable
        """

    def is_rule_enabled(self, name: str) -> bool:
        """Check if a rule is enabled.

        Args:
            name: Name of the rule to check

        Returns:
            True if the rule is enabled, False otherwise
        """

    def verify(self, program: Program) -> list[Diagnostic]:
        """Verify a program and collect diagnostics.

        This method runs all enabled rules on the program and collects
        diagnostics. It does not throw exceptions even if errors are found.

        Args:
            program: Program to verify

        Returns:
            List of all diagnostics (errors and warnings)
        """

    def verify_or_throw(self, program: Program) -> None:
        """Verify a program and throw on errors.

        This method runs verification and throws a VerificationError if any
        diagnostics with severity Error are found. Warnings do not cause an exception.

        Args:
            program: Program to verify

        Raises:
            VerificationError: If any errors are found
        """

    @staticmethod
    def generate_report(diagnostics: list[Diagnostic]) -> str:
        """Generate a formatted report from diagnostics.

        Args:
            diagnostics: List of diagnostics to format

        Returns:
            Formatted report string
        """

def run_verifier(disabled_rules: list[str] | None = None) -> Pass:
    """Create a verifier pass with configurable rules.

    This pass creates an IRVerifier with default rules and allows disabling
    specific rules. The verifier collects all diagnostics and logs them.

    Args:
        disabled_rules: List of rule names to disable (e.g., ["TypeCheck"])

    Returns:
        Pass that runs IR verification
    """

__all__ = [
    "Pass",
    "init_mem_ref",
    "basic_memory_reuse",
    "insert_sync",
    "add_alloc",
    "VerificationError",
    "SSAErrorType",
    "verify_ssa",
    "TypeCheckErrorType",
    "type_check",
    "convert_to_ssa",
    "flatten_call_expr",
    "normalize_stmt_structure",
    "flatten_single_stmt",
    "NestedCallErrorType",
    "DiagnosticSeverity",
    "Diagnostic",
    "IRVerifier",
    "run_verifier",
]
