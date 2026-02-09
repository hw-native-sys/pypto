# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Example Orchestration Function - Python Implementation

This script builds the orchestration function for the formula: f = (a + b + 1)(a + b + 2)

Task Graph:
  task0: c = a + b          (kernel_add, func_id=0)
  task1: d = c + 1          (kernel_add_scalar, func_id=1)
  task2: e = c + 2          (kernel_add_scalar, func_id=1)
  task3: f = d * e          (kernel_mul, func_id=2)

Dependencies: t0→t1, t0→t2, t1→t3, t2→t3
"""

import os

import pypto.language as pl
from pypto import DataType, ir
from pypto.backend import BackendType


@pl.program
class ExampleOrchProgram:
    """Example orchestration program with InCore kernels."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Adds two tensors element-wise: result = a + b"""
        a_tile: pl.Tile[[16, 16], pl.FP32] = pl.op.load(a, 0, 0, 16, 16)
        b_tile: pl.Tile[[16, 16], pl.FP32] = pl.op.load(b, 0, 0, 16, 16)
        result: pl.Tile[[16, 16], pl.FP32] = pl.op.add(a_tile, b_tile)
        output_new: pl.Tensor[[16, 16], pl.FP32] = pl.op.store(result, 0, 0, 16, 16, output)
        return output_new

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add_scalar(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        scalar: pl.Scalar[pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Adds a scalar to each element: result = a + scalar"""
        x: pl.Tile[[16, 16], pl.FP32] = pl.op.load(a, 0, 0, 16, 16)
        result: pl.Tile[[16, 16], pl.FP32] = pl.op.add(x, scalar)
        output_new: pl.Tensor[[16, 16], pl.FP32] = pl.op.store(result, 0, 0, 16, 16, output)
        return output_new

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_mul(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Multiplies two tensors element-wise: result = a * b"""
        a_tile: pl.Tile[[16, 16], pl.FP32] = pl.op.load(a, 0, 0, 16, 16)
        b_tile: pl.Tile[[16, 16], pl.FP32] = pl.op.load(b, 0, 0, 16, 16)
        result: pl.Tile[[16, 16], pl.FP32] = pl.op.mul(a_tile, b_tile)
        output_new: pl.Tensor[[16, 16], pl.FP32] = pl.op.store(result, 0, 0, 16, 16, output)
        return output_new

    @pl.function(type=pl.FunctionType.Orchestration)
    def BuildExampleGraph(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Build BuildExampleGraph orchestration function.

        Orchestration function for formula: f = (a + b + 1)(a + b + 2)
        Uses load/store pattern: InCore kernels take input + output tensors.

        Calls InCore functions to build the task graph:
          - task0: c = a + b (kernel_add writes to c)
          - task1: d = c + 1 (kernel_add_scalar writes to d)
          - task2: e = c + 2 (kernel_add_scalar writes to e)
          - task3: f = d * e (kernel_mul writes to f)

        Args:
            a: Input tensor A
            b: Input tensor B
            c: Temp buffer for a + b
            d: Temp buffer for c + 1
            e: Temp buffer for c + 2
            output: Final output buffer for d * e

        Returns:
            Final result tensor
        """
        # Task 0: c = a + b (call kernel_add with output buffer c)
        c: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add(a, b)

        # Task 1: d = c + 1 (call kernel_add_scalar with output buffer d)
        d: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add_scalar(c, 1.0)  # type: ignore[reportArgumentType]

        # Task 2: e = c + 2 (call kernel_add_scalar with output buffer e)
        e: pl.Tensor[[16, 16], pl.FP32] = self.kernel_add_scalar(c, 2.0)  # type: ignore[reportArgumentType]

        # Task 3: f = d * e (call kernel_mul with output buffer)
        f_result: pl.Tensor[[16, 16], pl.FP32] = self.kernel_mul(d, e)
        return f_result


def build_example_orch_program(dtype: DataType = DataType.FP32):
    """Build the complete example_orch program.

    Creates a program with:
      - 3 InCore functions (kernel_add, kernel_add_scalar, kernel_mul)
      - 1 Orchestration function (BuildExampleGraph)

    Args:
        dtype: Data type for tensors (currently only FP32 supported)

    Returns:
        Program object
    """
    if dtype != DataType.FP32:
        raise ValueError(f"Only FP32 is currently supported, got {dtype}")

    # The ExampleOrchProgram class decorator already creates the full Program
    # with all functions in the correct order
    print("Building functions using @pl.program decorator...")
    print("✓ InCore functions built: kernel_add, kernel_add_scalar, kernel_mul")
    print("✓ Orchestration function built: BuildExampleGraph")

    return ExampleOrchProgram


def main():
    """Main function - complete compilation workflow."""
    print("=" * 70)
    print("Example Orch Code Generation")
    print("=" * 70)

    # Configuration
    dtype = DataType.FP32
    print(f"\nConfiguration: {dtype}")

    # Step 1: Build IR
    print("\n[1] Building IR...")
    program = build_example_orch_program(dtype)
    print("✓ IR construction complete")
    print(f"  Functions: {[f.name for f in program.functions.values()]}")

    # Step 2: Print IR preview
    print("\n[2] IR Preview (Python syntax):")
    print("-" * 70)
    ir_text = ir.python_print(program)
    lines = ir_text.split("\n")
    preview_lines = min(40, len(lines))
    print("\n".join(lines[:preview_lines]))
    if len(lines) > preview_lines:
        print(f"\n... ({len(lines) - preview_lines} more lines)")
    print("-" * 70)

    # Step 3: Compile (using high-level ir.compile API)
    print("\n[3] Compiling with PassManager and CCECodegen...")
    output_dir = ir.compile(
        program,
        strategy=ir.OptimizationStrategy.Default,
        dump_passes=True,
        backend_type=BackendType.CCE,
    )
    print("✓ Compilation complete")
    print(f"✓ Output directory: {output_dir}")

    # Step 4: Display generated files
    print("\n[4] Generated files:")
    for root, _dirs, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, output_dir)
            file_size = os.path.getsize(filepath)
            print(f"  - {rel_path} ({file_size} bytes)")

    # Step 5: Preview orchestration C++ code (if exists)
    orch_file = os.path.join(output_dir, "orchestration", "BuildExampleGraph.cpp")
    if os.path.exists(orch_file):
        print("\n[5] Generated Orchestration C++ (preview):")
        print("=" * 70)
        with open(orch_file, "r") as f:
            content = f.read()
            lines = content.split("\n")
            preview_lines = min(50, len(lines))
            print("\n".join(lines[:preview_lines]))
            if len(lines) > preview_lines:
                print(f"\n... ({len(lines) - preview_lines} more lines)")
        print("=" * 70)
    else:
        print("\n[5] Warning: orchestration/BuildExampleGraph.cpp not found")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"  Program: {program.name}")
    print(f"  Functions: {len(program.functions)}")
    print("    - kernel_add (InCore)")
    print("    - kernel_add_scalar (InCore)")
    print("    - kernel_mul (InCore)")
    print("    - BuildExampleGraph (Orchestration)")
    print(f"  Output: {output_dir}")
    print(f"  Data type: {dtype}")
    print("  Optimization: XPlatform")
    print("  Formula: f = (a + b + 1)(a + b + 2)")
    print("=" * 70)


if __name__ == "__main__":
    main()
