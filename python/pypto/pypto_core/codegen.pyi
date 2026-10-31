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

class VFHelper:
    """No-Bridge VF helper and Tile adapter with ordered, whole-Tile effects."""

    @property
    def symbol(self) -> str: ...
    @property
    def adapter_symbol(self) -> str: ...
    @property
    def source(self) -> str: ...
    @property
    def parameter_types(self) -> list[str]: ...
    @property
    def caller_parameter_types(self) -> list[str]: ...
    @property
    def parameter_effects(self) -> list[str]: ...
    @property
    def parameter_indices(self) -> list[int]: ...
    @property
    def capture_byte_offsets(self) -> list[int]: ...

class PTOCodegenBundle:
    """Execution artifacts owned by one PTO codegen invocation."""

    @property
    def caller_source(self) -> str: ...
    @property
    def helpers(self) -> list[VFHelper]: ...

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

    def generate_bundle(
        self, program: Program, emit_tile_addr: bool = True, emit_source_loc: bool = True
    ) -> PTOCodegenBundle:
        """Generate an EmitC caller and separate typed-VMI helpers for A5/AIV."""

    def generate(
        self,
        program: Program,
        emit_tile_addr: bool = True,
        emit_source_loc: bool = True,
    ) -> str:
        """Generate PTO assembly from PyPTO IR Program.

        Args:
            program: Input PyPTO IR Program
            emit_tile_addr: When True (default), emit the physical ``addr`` operand
                on ``pto.alloc_tile`` (ptoas ``--pto-level=level3``). When False,
                omit ``addr`` so the ptoas PlanMemory pass allocates instead
                (``--pto-level=level2``).
            emit_source_loc: When True (default), suffix every emitted operation
                with an MLIR ``loc("file":line:col)`` taken from the IR ``Span``,
                so ptoas diagnostics name the user's source instead of a line in
                the generated ``.pto``. When False, emit no locations.

        Returns:
            PTO assembly code string (.pto format) with instructions like tmul, tadd, FOR/ENDFOR, etc.

        Example:
            >>> from pypto import codegen
            >>> cg = codegen.PTOCodegen()
            >>> pto_code = cg.generate(program)
        """

class KernelRegistration:
    """Runtime tensor ABI and ID for a generated code unit."""

    @property
    def func_id(self) -> int: ...
    @property
    def code_name(self) -> str: ...
    @property
    def core_type(self) -> CoreType: ...
    @property
    def signature(self) -> list[str]: ...

class OrchestrationResult:
    """Result of orchestration code generation."""

    @property
    def code(self) -> str:
        """Generated C++ orchestration code."""
        ...

    @property
    def kernel_registrations(self) -> list[KernelRegistration]:
        """Distinct runtime registrations, in func_id order; code may be shared."""
        ...

    @property
    def orchestration_signature(self) -> list[str]:
        """Orchestration entry per-tensor ArgDirection names (scalars excluded), in orch_args tensor order."""
        ...

class BuiltinNextLevelSpec:
    """Builtin chip-callable variant requested by distributed host codegen."""

    @property
    def op_name(self) -> str:
        """Internal builtin op name, e.g. ``builtin.tensor.allreduce``."""
        ...

    @property
    def variant(self) -> str:
        """Runtime callable key and ``next_levels/<variant>`` directory name."""
        ...

    @property
    def entry_symbol(self) -> str:
        """C ABI-safe entry symbol name."""
        ...

    @property
    def template_dir(self) -> str:
        """Package-resource template directory handle."""
        ...

    @property
    def template_vars(self) -> dict[str, str]:
        """Template variables supplied by the builtin op codegen handler."""
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

    def get_builtin_next_level_specs(self) -> list[BuiltinNextLevelSpec]:
        """Return builtin ``next_levels`` variants referenced by the generated host orchestration."""
        ...

def generate_orchestration(program: Program, func: Function) -> OrchestrationResult:
    """Generate C++ orchestration code for a function.

    Uses simpler runtime API. This is backend-agnostic.

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

def collect_tensor_shape_dyn_vars(func: Function) -> list[Var]:
    """Return ordered trailing shape arguments, excluding explicit scalar parameters."""

__all__ = [
    "PTOCodegen",
    "BuiltinNextLevelSpec",
    "DistributedCodegen",
    "OrchestrationResult",
    "KernelRegistration",
    "collect_vars_from_shape_expr",
    "collect_tensor_shape_dyn_vars",
    "generate_orchestration",
    "infer_function_core_type",
]
