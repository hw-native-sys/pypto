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

from pypto import DataType, ir
from pypto.ir import IRBuilder


def build_kernel_add(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build kernel_add InCore function.

    Adds two tensors element-wise: result = a + b
    Uses load/store pattern (reference: dynamic_softmax_codegen.py)

    Args:
        ib: IRBuilder instance
        dtype: Data type for tensors

    Returns:
        Function object
    """
    with ib.function("kernel_add", type=ir.FunctionType.InCore) as f:
        # Parameters - TensorType for InCore functions
        input_a = f.param("a", ir.TensorType([16, 16], dtype))
        input_b = f.param("b", ir.TensorType([16, 16], dtype))
        output_tensor = f.param("output", ir.TensorType([16, 16], dtype))
        f.return_type(ir.TensorType([16, 16], dtype))

        # Load tiles from input tensors (use distinct names to avoid shadowing params)
        a_tile = ib.let("a_tile", ir.op.block.load(input_a, 0, 0, 16, 16))
        b_tile = ib.let("b_tile", ir.op.block.load(input_b, 0, 0, 16, 16))

        # Element-wise addition using block.add
        result = ib.let("result", ir.op.block.add(a_tile, b_tile))

        # Store result to output tensor
        output_new = ib.let("output_new", ir.op.block.store(result, 0, 0, 16, 16, output_tensor))

        ib.return_stmt(output_new)

    return f.get_result()


def build_kernel_add_scalar(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build kernel_add_scalar InCore function.

    Adds a scalar to each element: result = a + scalar
    Uses load/store pattern (reference: dynamic_softmax_codegen.py)

    Args:
        ib: IRBuilder instance
        dtype: Data type for tensors

    Returns:
        Function object
    """
    with ib.function("kernel_add_scalar", type=ir.FunctionType.InCore) as f:
        # Parameters - TensorType and ScalarType
        input_tensor = f.param("a", ir.TensorType([16, 16], dtype))
        scalar = f.param("scalar", ir.ScalarType(dtype))
        output_tensor = f.param("output", ir.TensorType([16, 16], dtype))
        f.return_type(ir.TensorType([16, 16], dtype))

        # Load tile from input tensor
        x = ib.let("x", ir.op.block.load(input_tensor, 0, 0, 16, 16))

        # Tile + scalar using block.adds
        result = ib.let("result", ir.op.block.adds(x, scalar))

        # Store result to output tensor
        output_new = ib.let("output_new", ir.op.block.store(result, 0, 0, 16, 16, output_tensor))

        ib.return_stmt(output_new)

    return f.get_result()


def build_kernel_mul(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build kernel_mul InCore function.

    Multiplies two tensors element-wise: result = a * b
    Uses load/store pattern (reference: dynamic_softmax_codegen.py)

    Args:
        ib: IRBuilder instance
        dtype: Data type for tensors

    Returns:
        Function object
    """
    with ib.function("kernel_mul", type=ir.FunctionType.InCore) as f:
        # Parameters - TensorType
        input_a = f.param("a", ir.TensorType([16, 16], dtype))
        input_b = f.param("b", ir.TensorType([16, 16], dtype))
        output_tensor = f.param("output", ir.TensorType([16, 16], dtype))
        f.return_type(ir.TensorType([16, 16], dtype))

        # Load tiles from input tensors (use distinct names to avoid shadowing params)
        a_tile = ib.let("a_tile", ir.op.block.load(input_a, 0, 0, 16, 16))
        b_tile = ib.let("b_tile", ir.op.block.load(input_b, 0, 0, 16, 16))

        # Element-wise multiplication using block.mul
        result = ib.let("result", ir.op.block.mul(a_tile, b_tile))

        # Store result to output tensor
        output_new = ib.let("output_new", ir.op.block.store(result, 0, 0, 16, 16, output_tensor))

        ib.return_stmt(output_new)

    return f.get_result()


def build_example_graph(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build BuildExampleGraph orchestration function.

    Orchestration function for formula: f = (a + b + 1)(a + b + 2)
    Uses load/store pattern: InCore kernels take input + output tensors.

    Calls InCore functions to build the task graph:
      - task0: c = a + b (kernel_add writes to c)
      - task1: d = c + 1 (kernel_add_scalar writes to d)
      - task2: e = c + 2 (kernel_add_scalar writes to e)
      - task3: f = d * e (kernel_mul writes to f)

    Args:
        ib: IRBuilder instance
        dtype: Data type for tensors

    Returns:
        Function object
    """
    with ib.function("BuildExampleGraph", type=ir.FunctionType.Orchestration) as f:
        # Parameters - input tensors and temp/output buffers (reference: dynamic_softmax)
        a = f.param("a", ir.TensorType([16, 16], dtype))
        b = f.param("b", ir.TensorType([16, 16], dtype))
        c = f.param("c", ir.TensorType([16, 16], dtype))  # temp: a + b
        d = f.param("d", ir.TensorType([16, 16], dtype))  # temp: c + 1
        e = f.param("e", ir.TensorType([16, 16], dtype))  # temp: c + 2
        output = f.param("output", ir.TensorType([16, 16], dtype))  # final: d * e
        f.return_type(ir.TensorType([16, 16], dtype))

        # Create scalar constants
        scalar_1 = ir.ConstFloat(1.0, dtype, ir.Span.unknown())
        scalar_2 = ir.ConstFloat(2.0, dtype, ir.Span.unknown())

        # Task 0: c = a + b (call kernel_add with output buffer c)
        kernel_add_op = ir.GlobalVar("kernel_add")
        c_updated = ib.let("c_updated", ir.Call(kernel_add_op, [a, b, c], ir.Span.unknown()))

        # Task 1: d = c + 1 (call kernel_add_scalar with output buffer d)
        kernel_add_scalar_op = ir.GlobalVar("kernel_add_scalar")
        d_updated = ib.let(
            "d_updated", ir.Call(kernel_add_scalar_op, [c_updated, scalar_1, d], ir.Span.unknown())
        )

        # Task 2: e = c + 2 (call kernel_add_scalar with output buffer e)
        e_updated = ib.let(
            "e_updated", ir.Call(kernel_add_scalar_op, [c_updated, scalar_2, e], ir.Span.unknown())
        )

        # Task 3: f = d * e (call kernel_mul with output buffer)
        kernel_mul_op = ir.GlobalVar("kernel_mul")
        f_result = ib.let("f", ir.Call(kernel_mul_op, [d_updated, e_updated, output], ir.Span.unknown()))

        ib.return_stmt(f_result)

    return f.get_result()


def build_example_orch_program(dtype: DataType = DataType.FP32):
    """Build the complete example_orch program.

    Creates a program with:
      - 3 InCore functions (kernel_add, kernel_add_scalar, kernel_mul)
      - 1 Orchestration function (BuildExampleGraph)

    Args:
        dtype: Data type for tensors

    Returns:
        Program object
    """
    ib = IRBuilder()

    # Step 1: Build all InCore functions
    print("Building InCore functions...")
    kernel_add_func = build_kernel_add(ib, dtype)
    kernel_add_scalar_func = build_kernel_add_scalar(ib, dtype)
    kernel_mul_func = build_kernel_mul(ib, dtype)
    print("✓ InCore functions built")

    # Step 2: Build Orchestration function
    print("Building Orchestration function...")
    orch_func = build_example_graph(ib, dtype)
    print("✓ Orchestration function built")

    # Step 3: Create Program (order matters: InCore first, then Orchestration)
    functions = [
        kernel_add_func,  # func_id=0
        kernel_add_scalar_func,  # func_id=1
        kernel_mul_func,  # func_id=2
        orch_func,  # Orchestration function
    ]

    program = ir.Program(functions, "example_orch", ir.Span.unknown())

    return program


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
        program, strategy=ir.OptimizationStrategy.Default, dump_passes=True, codegen=ir.CodegenBackend.CCE
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
