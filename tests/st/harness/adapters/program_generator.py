# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Program code generator for PTO testing framework.

Generates both kernel and orchestration C++ code from PyPTO Programs
using the ir.compile() API.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypto import ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

if TYPE_CHECKING:
    pass


class ProgramCodeGenerator:
    """Generates CCE C++ kernel and orchestration code from PyPTO Programs.

    Pipeline: ir.Program -> ir.compile() -> C++ source files

    Uses PyPTO's ir.compile() to generate both kernel and orchestration files.
    The generated files are organized in the output directory:
      - kernels/*.cpp - Kernel source files
      - orchestration/*.cpp - Orchestration functions (if present in program)
      - pass_results/ - IR dumps from each pass (if dump_passes=True)

    Example:
        from pypto.ir.pass_manager import OptimizationStrategy

        generator = ProgramCodeGenerator(strategy=OptimizationStrategy.PTOAS)
        result = generator.generate(program, output_dir)
        # Returns: {
        #   "kernels": [{"func_id": 0, "source": "path/to/kernels/func.cpp", "core_type": "aiv"}, ...],
        #   "orchestration": {"source": "path/to/orchestration/orch.cpp", "function_name": "BuildGraph"}
        # }
    """

    def __init__(
        self,
        strategy: OptimizationStrategy | None = None,
        backend_type: BackendType | None = None,
    ):
        """Initialize kernel generator.

        Args:
            strategy: Optimization strategy for pass pipeline.
                      If None, uses OptimizationStrategy.Default.
            backend_type: Backend type for code generation.
                          If None, uses BackendType.CCE.
        """
        if strategy is None:
            strategy = OptimizationStrategy.Default
        if backend_type is None:
            backend_type = BackendType.CCE

        self.strategy = strategy
        self.backend_type = backend_type

    def _add_headers_to_orch_file(self, orch_file: Path) -> None:
        """Add required headers to orchestration file if not already present.

        Args:
            orch_file: Path to the orchestration C++ file.
        """
        # Read the file content
        content = orch_file.read_text(encoding="utf-8")

        # Check if headers are already present
        has_runtime_h = '#include "runtime.h"' in content
        has_iostream = "#include <iostream>" in content

        # If both headers are present, no need to modify
        if has_runtime_h and has_iostream:
            return

        # Prepare headers to add
        headers_to_add = []
        if not has_runtime_h:
            headers_to_add.append('#include "runtime.h"')
        if not has_iostream:
            headers_to_add.append("#include <iostream>")

        # Find the insertion point (after comments, before first code line)
        lines = content.splitlines(keepends=True)
        insert_pos = 0

        # Find the first non-comment, non-blank line (usually extern "C" or #include)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                # Stop at first non-comment line
                if (
                    not stripped.startswith("//")
                    and not stripped.startswith("/*")
                    and not stripped.startswith("*")
                ):
                    insert_pos = i
                    break

        # Insert headers before the first non-comment line
        headers_text = "\n".join(headers_to_add) + "\n"
        # Add a blank line after headers for readability
        if insert_pos > 0:
            headers_text += "\n"

        # Insert headers
        lines.insert(insert_pos, headers_text)

        # Write back to file
        orch_file.write_text("".join(lines), encoding="utf-8")

    def generate(
        self,
        program: Any,
        output_dir: Path,
        dump_passes: bool = False,
    ) -> dict[str, Any]:
        """Generate CCE C++ kernel files from a PyPTO Program.

        Args:
            program: PyPTO Program (from @pl.program decorator or ir.Program).
            output_dir: Directory to write kernel files into.
                        ir.compile() will create kernels/ and orchestration/ subdirectories.
            dump_passes: If True, dump intermediate IR after each pass.

        Returns:
            Dict with kernels list and orchestration info:
            {
                "kernels": [{"source": "path/to/kernel.cpp", "core_type": "aiv"}, ...],
                "orchestration": {"source": "path/to/orch.cpp", "function_name": "BuildGraph"} or None
            }
        """
        output_dir = Path(output_dir)

        # Call ir.compile() to generate all code directly in output_dir
        ir.compile(
            program,
            output_dir=str(output_dir),
            strategy=self.strategy,
            dump_passes=dump_passes,
            backend_type=self.backend_type,
        )
        # Files are now in output_dir with structure:
        #   output_dir/kernels/aiv/*.cpp and output_dir/kernels/aic/*.cpp
        #   output_dir/orchestration/*.cpp (if orchestration function exists)
        #   output_dir/kernel_config.py (if orchestration function exists)
        #   output_dir/passes_dump/ (if dump_passes=True)

        # Locate generated kernel files in output_dir/kernels/
        kernels_dir = output_dir / "kernels"
        if not kernels_dir.exists():
            raise ValueError(f"No kernels directory found in {output_dir}")

        # Extract kernel file paths (ir.compile already added headers)
        kernels = []

        # Traverse aiv and aic subdirectories
        for core_type_subdir in ["aiv", "aic"]:
            core_dir = kernels_dir / core_type_subdir
            if core_dir.exists():
                for kernel_file in sorted(core_dir.glob("*.cpp")):
                    kernels.append(
                        {
                            "source": str(kernel_file),
                            "core_type": core_type_subdir,  # Use actual subdirectory name
                        }
                    )

        # Check if orchestration files were generated
        orch_dir = output_dir / "orchestration"
        orch_info = None

        if orch_dir.exists() and list(orch_dir.glob("*.cpp")):
            # Orchestration files are already in the right location
            # Just extract the information
            orch_files = list(orch_dir.glob("*.cpp"))
            if orch_files:
                orch_file = orch_files[0]  # Assuming single orchestration file

                # Add required headers to orchestration file
                self._add_headers_to_orch_file(orch_file)

                orch_info = {
                    "source": str(orch_file),
                    "function_name": "Build" + orch_file.stem,  # Extract from filename
                }

        return {
            "kernels": kernels,
            "orchestration": orch_info,
        }
