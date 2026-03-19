# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# pylint: disable=unused-argument
"""Code generation module for converting IR to pto-isa C++ (PTOCodegen, CCECodegen, TypeConverter)."""

from pypto.pypto_core.ir import CoreType, Function, PipeType, Program

class TypeConverter:
    """Utility for converting IR types to pto-isa C++ types"""

    def __init__(self) -> None:
        """Create a type converter"""

    def ConvertPipeType(self, pipe: PipeType) -> str:
        """Convert PipeType to pto-isa pipe type string

        Args:
            pipe: Pipeline type

        Returns:
            C++ pipe type string with 'PIPE_' prefix (e.g., 'PIPE_MTE1', 'PIPE_V')
        """

    def ConvertEventId(self, event_id: int) -> str:
        """Convert event ID to pto-isa event ID string

        Args:
            event_id: Event ID (must be in range [0, 7])

        Returns:
            C++ event ID string with 'EVENT_ID' prefix (e.g., 'EVENT_ID0')
        """

    def GenerateShapeType(self, dims: list[int]) -> str:
        """Generate Shape type instantiation

        Args:
            dims: Shape dimensions

        Returns:
            Shape type string with 5D padding (e.g., 'Shape<1, 1, 1, 128, 64>')
        """

    def GenerateStrideType(self, shape: list[int]) -> str:
        """Generate Stride type instantiation for row-major layout

        Args:
            shape: Shape dimensions

        Returns:
            Stride type string with 5D padding
        """

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

class CCECodegen:
    """CCE code generator for converting PyPTO IR to pto-isa C++ code."""

    def __init__(self) -> None:
        """Create a code generator."""

    def generate(self, program: Program) -> dict[str, str]:
        """Generate C++ code from a PyPTO IR Program.

        Classifies functions into kernel and orchestration, then generates:
        - Kernel functions -> kernels/<func_name>.cpp (CCE kernel C++ code)
        - Orchestration function -> orchestration/<func_name>.cpp (orchestration C++ code)

        Args:
            program: The IR Program to generate code for

        Returns:
            Dict mapping file path to generated C++ code content

        Example:
            >>> from pypto import codegen
            >>> cg = codegen.CCECodegen()
            >>> files = cg.generate(program)
            >>> kernel_code = files["kernels/my_kernel.cpp"]
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

def generate_orchestration(program: Program, func: Function) -> OrchestrationResult:
    """Generate C++ orchestration code for a function.

    Uses PTO2 runtime API. This is backend-agnostic and works with both CCE and PTO backends.

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

__all__ = [
    "TypeConverter",
    "PTOCodegen",
    "CCECodegen",
    "OrchestrationResult",
    "generate_orchestration",
    "infer_function_core_type",
]
