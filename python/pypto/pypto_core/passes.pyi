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
from typing import Union, overload

from pypto.pypto_core.ir import Function, Program, Span

class Pass:
    """Base class for IR transformation passes.

    A Pass represents a transformation that can be applied to a Function or Program.
    Concrete pass implementations should inherit from this class and
    implement the run() method.
    """

    @overload
    def run(self, func: Function) -> Function:
        """Execute the pass on a function.

        Args:
            func: Input Function to transform

        Returns:
            Transformed Function after the pass has been applied
        """

    @overload
    def run(self, program: Program) -> Program:
        """Execute the pass on a program.

        Args:
            program: Input Program to transform

        Returns:
            Transformed Program after the pass has been applied
        """

    def run(self, input_ir: Union[Function, Program]) -> Union[Function, Program]:
        """Execute the pass on a function or program.

        Args:
            input_ir: Input Function or Program to transform

        Returns:
            Transformed Function or Program after the pass has been applied
        """

class IdentityPass(Pass):
    """A pass that appends '_identity' suffix to function name for testing.

    This is a simple pass implementation used primarily for testing the
    pass infrastructure. It modifies the function name by appending
    '_identity' to demonstrate that the pass was executed.
    """

    def __init__(self) -> None:
        """Create an identity pass."""

class InitMemRefPass(Pass):
    """A pass that initializes memref for variables.

    This pass traverses the function and initializes the MemRef field for all
    Var nodes. It sets memory space to UB by default, or DDR for variables
    used in block.load/block.store operations.
    """

    def __init__(self) -> None:
        """Create an InitMemRef pass."""

class BasicMemoryReusePass(Pass):
    """A pass for basic memory reuse based on dependency graph.

    This pass uses DependencyAnalyzer to compute lifetime intervals and
    identifies memory reuse opportunities without execution timing simulation.
    Variables with non-overlapping lifetimes in the same memory space can
    share MemRef objects.
    """

    def __init__(self) -> None:
        """Create a BasicMemoryReuse pass."""

class InsertSyncPass(Pass):
    """A pass that automatically inserts sync and bar operations.

    This pass analyzes data dependencies between operations based on MemRef
    and inserts synchronization instructions (sync_src, sync_dst, bar_v, bar_m)
    to ensure correct execution order across different hardware pipes.
    """

    def __init__(self) -> None:
        """Create an InsertSync pass."""

class AddAllocPass(Pass):
    """A pass that adds alloc operations for all MemRef objects in TileType variables.

    This pass traverses the function and creates alloc operations for each unique
    MemRef object found in TileType variables. The alloc operations are prepended
    to the function body to allocate memory for these MemRef objects.

    The pass:
    1. Identifies all TileType variables in the function
    2. Collects all unique MemRef objects from these variables
    3. Creates an alloc operation for each unique MemRef
    4. Prepends these alloc operations to the function body
    """

    def __init__(self) -> None:
        """Create an AddAlloc pass."""

class SSAErrorType(Enum):
    """SSA verification error types."""

    MULTIPLE_ASSIGNMENT: int
    NAME_SHADOWING: int
    MISSING_YIELD: int
    CONTROL_FLOW_TYPE_MISMATCH: int

class SSAError:
    """SSA verification error information."""

    type: SSAErrorType
    message: str
    span: Span

class VerifySSAPass(Pass):
    """A pass that verifies SSA form of IR.

    This pass checks the following SSA properties:
    1. Each variable is assigned only once (MULTIPLE_ASSIGNMENT)
    2. No variable name shadowing across scopes (NAME_SHADOWING)
    3. ForStmt with iter_args must have YieldStmt as last statement (MISSING_YIELD)
    4. Type consistency in ForStmt: iter_args initValue, yield values, return_vars
       (CONTROL_FLOW_TYPE_MISMATCH)
    5. IfStmt with return_vars must have YieldStmt in both then and else branches
       (MISSING_YIELD)
    6. Type consistency in IfStmt: then yield and else yield
       (CONTROL_FLOW_TYPE_MISMATCH)

    The pass collects all errors and generates a verification report instead of
    throwing exceptions, allowing detection of all issues in a single run.
    """

    def __init__(self) -> None:
        """Create a VerifySSA pass."""

    def has_errors(self) -> bool:
        """Check if any SSA violations were found.

        Returns:
            True if errors exist, False otherwise
        """

    def get_errors(self) -> list[SSAError]:
        """Get list of SSA errors.

        Returns:
            Vector of SSA errors found during verification
        """

    def get_report(self) -> str:
        """Get formatted verification report.

        Returns:
            String containing the formatted report
        """

__all__ = [
    "Pass",
    "IdentityPass",
    "InitMemRefPass",
    "BasicMemoryReusePass",
    "InsertSyncPass",
    "AddAllocPass",
    "SSAErrorType",
    "SSAError",
    "VerifySSAPass",
]
