# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""High-level API functions for PyPTO IR compilation."""

import os
from datetime import datetime
from typing import Optional

from pypto.pypto_core import ir as _ir_core

from .pass_manager import OptimizationStrategy, PassManager


def compile(
    program: _ir_core.Program,
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
        >>> # Compile with PTOAS optimization
        >>> output_dir = ir.compile(
        ...     program,
        ...     strategy=ir.OptimizationStrategy.PTOAS,
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
