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

# Import python_print utility
from .printer import python_print  # noqa: F401, E402

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
    1. Runs optimization passes via PassManager
    2. Optionally dumps IR before and after each pass (if dump_passes=True)
    3. Generates PTO assembly code via PTOCodegen
    4. Saves all artifacts to a unified output directory

    Args:
        program: Input Program to compile
        output_dir: Output directory (default: build_output/<program_name>_<timestamp>)
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

    # Run passes with PassManager
    pm = PassManager.get_strategy(strategy)
    transformed_program = pm.run_passes(program, dump_ir=dump_passes, output_dir=output_dir)

    # Generate PTO assembly code
    codegen = _ir_core.PTOCodegen()
    pto_code = codegen.generate(transformed_program)  # type: ignore[arg-type]

    # Save PTO assembly
    pto_path = os.path.join(output_dir, "output.pto")
    with open(pto_path, "w") as f:
        f.write(pto_code)

    return output_dir


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
