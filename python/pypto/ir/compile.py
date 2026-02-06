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

from pypto.backend import BackendType
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core import codegen as _codegen_core
from pypto.pypto_core import ir as _ir_core

from .pass_manager import OptimizationStrategy, PassManager


def compile(
    program: _ir_core.Program,
    output_dir: Optional[str] = None,
    strategy: OptimizationStrategy = OptimizationStrategy.Default,
    dump_passes: bool = True,
    backend_type: BackendType = BackendType.PTO,
) -> str:
    """Compile a Program through passes and codegen.

    This function provides a complete compilation pipeline that:
    1. Runs optimization passes via PassManager
    2. Optionally dumps IR before and after each pass (if dump_passes=True)
    3. Generates code via selected backend (PTO or CCE)
    4. Saves all artifacts to a unified output directory

    Args:
        program: Input Program to compile
        output_dir: Output directory (default: build_output/<program_name>_<timestamp>)
        strategy: Optimization strategy to use (default: Default)
        dump_passes: Whether to dump IR after each pass (default: True)
        backend_type: Backend type for passes and codegen (default: PTO)

    Returns:
        Path to the output directory containing all artifacts

    Example:
        >>> from pypto import ir
        >>> from pypto.backend import BackendType
        >>> program = build_my_program()
        >>> output_dir = ir.compile(
        ...     program,
        ...     strategy=ir.OptimizationStrategy.PTOAS,
        ...     dump_passes=True,
        ...     backend_type=BackendType.PTO
        ... )
    """
    # Set the global backend type (idempotent - can be called multiple times)
    _backend_core.set_backend_type(backend_type)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("build_output", f"{program.name}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    pm = PassManager.get_strategy(strategy)
    passes_dump_dir = os.path.join(output_dir, "passes_dump")
    transformed_program = pm.run_passes(program, dump_ir=dump_passes, output_dir=passes_dump_dir)

    if backend_type == BackendType.PTO:
        codegen_instance = _codegen_core.PTOCodegen()
        pto_code = codegen_instance.generate(transformed_program)  # type: ignore[arg-type]
        pto_path = os.path.join(output_dir, "output.pto")
        with open(pto_path, "w") as f:
            f.write(pto_code)
    elif backend_type == BackendType.CCE:
        codegen_instance = _codegen_core.CCECodegen()
        files = codegen_instance.generate(transformed_program)  # type: ignore[arg-type]
        for filepath, content in files.items():
            full_path = os.path.join(output_dir, filepath)
            file_dir = os.path.dirname(full_path)
            if file_dir:
                os.makedirs(file_dir, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")

    return output_dir
