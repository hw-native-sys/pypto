# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO IR module with tensor operations.

This module provides:
- Re-exports of all core IR types from pypto_core.ir
- Organized operation namespaces (e.g., op.tensor.create)
- IR Builder for incremental IR construction
- Helper utilities
- Enhanced type constructors (e.g., TensorType with integer shape support)
"""

# Re-export all core IR types and functions from native module
# Re-export DataType for convenience
from pypto.pypto_core import DataType  # noqa: F401
from pypto.pypto_core import ir as _ir_core  # noqa: F401
from pypto.pypto_core.ir import *  # noqa: F401, F403

# Import operation modules
# Import operator overloading with span capture and normalization
# This patches Var and ScalarExpr with Python operators
from . import (
    op,
    operators,  # noqa: F401
)

# Export common DataType values for convenience
FP4 = DataType.FP4
FP8 = DataType.FP8
FP16 = DataType.FP16
FP32 = DataType.FP32
BF16 = DataType.BF16
HF4 = DataType.HF4
HF8 = DataType.HF8
INT4 = DataType.INT4
INT8 = DataType.INT8
INT16 = DataType.INT16
INT32 = DataType.INT32
INT64 = DataType.INT64
UINT4 = DataType.UINT4
UINT8 = DataType.UINT8
UINT16 = DataType.UINT16
UINT32 = DataType.UINT32
UINT64 = DataType.UINT64
BOOL = DataType.BOOL

# Import IR Builder
# Import for Compile function
import os  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Optional  # noqa: E402

from .builder import IRBuilder  # noqa: F401, E402

# Import parser DSL APIs
from .parser import Tensor, function, range, yield_  # noqa: F401, E402

# Import PassManager and OptimizationStrategy
from .pass_manager import OptimizationStrategy, PassManager  # noqa: F401, E402

# Import TensorType and TileType with enhanced __init__ that supports integer shapes
# This patches the native TensorType and TileType classes to accept integer shapes
from .type import TensorType, TileType  # noqa: F401, E402


def compile(
    program,  # type: ignore[misc]
    output_dir: Optional[str] = None,
    strategy: OptimizationStrategy = OptimizationStrategy.Default,
    dump_passes: bool = True,
) -> str:
    """Compile a Program through passes and codegen.

    This function provides a complete compilation pipeline that:
    1. Dumps the original IR
    2. Runs optimization passes via PassManager
    3. Dumps IR after each pass (if dump_passes=True)
    4. Generates PTO assembly code via PTOCodegen
    5. Saves all artifacts to a unified output directory

    Args:
        program: Input Program to compile
        output_dir: Output directory (default: build_output/<program_name>)
        strategy: Optimization strategy to use (default: Default)
        dump_passes: Whether to dump IR after each pass (default: True)

    Returns:
        Path to the output directory containing all artifacts

    Example:
        >>> from pypto import ir, DataType
        >>> # Create program
        >>> program = build_my_program()
        >>> # Compile with Custom2 optimization
        >>> output_dir = ir.compile(
        ...     program,
        ...     strategy=ir.OptimizationStrategy.Custom2,
        ...     dump_passes=True
        ... )
        >>> print(f"Artifacts saved to: {output_dir}")
    """
    # Determine output directory
    if output_dir is None:
        # Generate timestamp in format: YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("build_output", f"{program.name}_{timestamp}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Dump frontend IR (original IR before any passes)
    frontend_path = os.path.join(output_dir, "00_frontend.py")
    with open(frontend_path, "w") as f:
        f.write(python_print(program, prefix="pl"))

    # Step 2: Run passes with PassManager
    pm = PassManager.get_strategy(strategy)

    if dump_passes and len(pm.passes) > 0:
        # Run passes one by one and dump Program state after each pass
        current_program = program
        pass_names = pm.get_pass_names()

        for i, (pass_instance, pass_name) in enumerate(zip(pm.passes, pass_names), start=1):
            # Apply this pass to all functions in the program
            transformed_functions = []
            for global_var, func in current_program.functions.items():
                transformed_func = pass_instance.run(func)
                transformed_functions.append(transformed_func)

            # Create new program with transformed functions
            current_program = _ir_core.Program(
                transformed_functions, current_program.name, current_program.span
            )

            # Dump IR after this pass with sequential numbering (01, 02, ...)
            pass_dump_path = os.path.join(output_dir, f"{i:02d}_after_{pass_name}.py")
            with open(pass_dump_path, "w") as f:
                f.write(python_print(current_program, prefix="pl"))

        transformed_program = current_program
    else:
        # Run all passes at once without dumping
        result = pm.run_passes(program)
        # Since input is a Program, output must be a Program
        transformed_program = result  # type: ignore[assignment]

    # Step 3: Generate PTO assembly code
    codegen = _ir_core.PTOCodegen()
    pto_code = codegen.generate(transformed_program)  # type: ignore[arg-type]

    # Step 4: Save PTO assembly
    pto_path = os.path.join(output_dir, "output.pto")
    with open(pto_path, "w") as f:
        f.write(pto_code)

    return output_dir


def python_print(node, prefix="pl"):  # type: ignore[misc]
    """
    Print IR node or Type object in Python IR syntax.

    This is a unified wrapper that dispatches to the appropriate C++ function
    based on the type of the input object.

    Args:
        node: IR node (Expr, Stmt, Function, Program) or Type object to print
        prefix: Module prefix (default 'pl' for 'import pypto.language as pl')

    Returns:
        str: Python-style string representation
    """
    # Check if node is a Type object
    if isinstance(node, _ir_core.Type):
        # Use the separate function for Type objects
        return _ir_core.python_print_type(node, prefix)
    else:
        # Use the standard function for IRNode objects
        return _ir_core.python_print(node, prefix)


__all__ = [
    "op",
    "IRBuilder",
    "TensorType",
    "TileType",
    "python_print",
    "compile",
    "PassManager",
    "OptimizationStrategy",
    "function",
    "range",
    "yield_",
    "Tensor",
]  # fmt: skip
