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
    Pass objects should be created using factory functions (identity, init_mem_ref, etc.)
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

def identity() -> Pass:
    """Create an identity pass for testing.

    Appends '_identity' to function names to verify pass execution.
    Useful for testing the pass infrastructure.

    Returns:
        Pass object that performs identity transformation
    """

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

    Returns:
        Pass object that inserts synchronization operations
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

__all__ = [
    "Pass",
    "identity",
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
]
