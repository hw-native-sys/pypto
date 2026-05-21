# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# pylint: disable=unused-argument
"""Code generation module for converting IR to PTO assembly (PTOCodegen)."""

from pypto.pypto_core.ir import CoreType, Expr, Function, Program, Var

class PTOCodegen:
    """Code generator that transforms PyPTO IR to PTO assembly (.pto format).

    Generates PTO ISA instructions from PyPTO IR, supporting:
    - Tile operations (binary, unary, scalar) -> PTO instructions (VADD, VMUL, etc.)
    - Control flow (for loops, if statements) -> FOR/ENDFOR, IF/ENDIF
    - SSA-style variable naming with % prefix
    - Proper type annotations (!pto.tile<...>, !pto.memref<...>)
    """

    def __init__(self) -> None:
        """Create a new PTO code generator."""

    def generate(self, program: Program) -> str:
        """Generate PTO assembly from PyPTO IR Program.

        Args:
            program: Input PyPTO IR Program

        Returns:
            PTO assembly code string (.pto format) with instructions like tmul, tadd, FOR/ENDFOR, etc.

        Example:
            >>> from pypto import codegen
            >>> cg = codegen.PTOCodegen()
            >>> pto_code = cg.generate(program)
        """

class OrchestrationResult:
    """Result of orchestration code generation."""

    @property
    def code(self) -> str:
        """Generated C++ orchestration code."""
        ...

    @property
    def func_name_to_id(self) -> dict[str, int]:
        """Kernel function name to func_id mapping."""
        ...

    @property
    def func_name_to_core_type(self) -> dict[str, CoreType]:
        """Kernel function name to core type mapping."""
        ...

class DistributedCodegen:
    """Distributed codegen for Linqu hierarchy runtime C++ code."""

    def __init__(self) -> None:
        """Create a distributed code generator."""

    def generate(self, program: Program) -> str:
        """Generate distributed C++ code from IR Program.

        Args:
            program: The IR Program (after OutlineHierarchyScopes)

        Returns:
            Complete C++ source code as a string
        """

def generate_orchestration(program: Program, func: Function) -> OrchestrationResult:
    """Generate C++ orchestration code for a function.

    Uses PTO2 runtime API. This is backend-agnostic.

    Args:
        program: The IR Program containing all functions
        func: The orchestration function to generate code for

    Returns:
        OrchestrationResult with generated code and function metadata
    """

def infer_function_core_type(func: Function) -> CoreType:
    """Infer the core type (CUBE or VECTOR) of a function from its operations.

    Args:
        func: The function to infer core type for

    Returns:
        CoreType.CUBE or CoreType.VECTOR
    """

def collect_vars_from_shape_expr(expr: Expr) -> list[Var]:
    """Collect Vars from a tensor-shape expression in first-seen DFS order.

    Used by the Python kernel-wrapper codegen to recover dynamic dims from
    ``tensor->shapes[]``. The same walk drives the trailing ``%argN: index``
    params emitted onto the ``func.func`` signature, so the wrapper and the
    compiled function stay in lockstep by construction (single source of
    truth).

    Args:
        expr: Tensor-shape expression (a dim from ``TensorType.shape``).

    Returns:
        Vars in first-seen DFS order, deduped within this single call.
    """

__all__ = [
    "PTOCodegen",
    "DistributedCodegen",
    "OrchestrationResult",
    "collect_vars_from_shape_expr",
    "generate_orchestration",
    "infer_function_core_type",
]
