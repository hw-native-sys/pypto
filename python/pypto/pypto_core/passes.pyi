# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Type stubs for PyPTO IR Pass transformations."""

from pypto.pypto_core.ir import Program

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

__all__ = [
    "Pass",
    "identity",
    "init_mem_ref",
    "basic_memory_reuse",
    "insert_sync",
    "add_alloc",
]
