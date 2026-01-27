# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Dynamic Softmax Code Generation Example

Demonstrates:
1. Building dynamic softmax computation using IRBuilder and block operations
2. Creating multiple helper functions (InCore) and orchestration function
3. Using FOR loops and IF statements with function calls
4. Generating PTO assembly (.pto format)

Dynamic Softmax Algorithm:
    1. rowmax: find max value in each row
    2. rowexpandsub: subtract max from each element (for numerical stability)
    3. elem_exp: apply exp to each element
    4. rowsum: sum all elements in each row
    5. rowexpanddiv: divide each element by row sum
"""

import os

from pypto import DataType, ir
from pypto.ir import IRBuilder


def BuildRowMaxFunction(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build rowmax InCore function.

    Computes the maximum value along each row.
    Input: tensor [8, 8], Output: tensor [8, 1]

    Args:
        ib: IR Builder instance
        dtype: Data type for computation

    Returns:
        ir.Function: The rowmax function
    """
    with ib.function("rowmax") as f:
        # Parameters
        input_tensor = f.param("input", ir.TensorType([8, 8], dtype))
        output_tensor = f.param("output", ir.TensorType([8, 1], dtype))
        f.return_type(ir.TensorType([8, 1], dtype))

        # Load tile from input (8x8)
        x = ib.let("x", ir.op.block.load(input_tensor, 0, 0, 8, 8))

        # Compute row max (8x8 -> 8x1)
        result = ib.let("result", ir.op.block.row_max(x))

        # Store result to output (8x1)
        output_new = ib.let("output_new", ir.op.block.store(result, 0, 0, 8, 1, output_tensor))

        # Return the updated output tensor
        ib.return_stmt(output_new)

    return f.get_result()


def BuildRowExpandSubFunction(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build rowexpandsub InCore function.

    Performs row-wise broadcast subtraction: input_x[i,j] - input_row[i,0]

    Args:
        ib: IR Builder instance
        dtype: Data type for computation

    Returns:
        ir.Function: The rowexpandsub function
    """
    with ib.function("rowexpandsub") as f:
        # Parameters
        input_x = f.param("input_x", ir.TensorType([8, 8], dtype))
        input_row = f.param("input_row", ir.TensorType([8, 1], dtype))
        output_tensor = f.param("output", ir.TensorType([8, 8], dtype))
        f.return_type(ir.TensorType([8, 8], dtype))

        # Load tiles
        x = ib.let("x", ir.op.block.load(input_x, 0, 0, 8, 8))
        row_vals = ib.let("row_vals", ir.op.block.load(input_row, 0, 0, 8, 1))

        # Row expand subtract (8x8, 8x1 -> 8x8)
        result = ib.let("result", ir.op.block.row_expand_sub(x, row_vals))

        # Store result
        output_new = ib.let("output_new", ir.op.block.store(result, 0, 0, 8, 8, output_tensor))

        # Return the updated output tensor
        ib.return_stmt(output_new)

    return f.get_result()


def BuildElemExpFunction(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build elem_exp InCore function.

    Applies element-wise exponential operation.

    Args:
        ib: IR Builder instance
        dtype: Data type for computation

    Returns:
        ir.Function: The elem_exp function
    """
    with ib.function("elem_exp") as f:
        # Parameters
        input_tensor = f.param("input", ir.TensorType([8, 8], dtype))
        output_tensor = f.param("output", ir.TensorType([8, 8], dtype))
        f.return_type(ir.TensorType([8, 8], dtype))

        # Load tile
        x = ib.let("x", ir.op.block.load(input_tensor, 0, 0, 8, 8))

        # Apply exp
        result = ib.let("result", ir.op.block.exp(x))

        # Store result
        output_new = ib.let("output_new", ir.op.block.store(result, 0, 0, 8, 8, output_tensor))

        # Return the updated output tensor
        ib.return_stmt(output_new)

    return f.get_result()


def BuildRowSumFunction(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build rowsum InCore function.

    Computes the sum of values along each row.
    Input: tensor [8, 8], Output: tensor [8, 1]

    Args:
        ib: IR Builder instance
        dtype: Data type for computation

    Returns:
        ir.Function: The rowsum function
    """
    with ib.function("rowsum") as f:
        # Parameters
        input_tensor = f.param("input", ir.TensorType([8, 8], dtype))
        output_tensor = f.param("output", ir.TensorType([8, 1], dtype))
        f.return_type(ir.TensorType([8, 1], dtype))

        # Load tile
        x = ib.let("x", ir.op.block.load(input_tensor, 0, 0, 8, 8))

        # Compute row sum (8x8 -> 8x1)
        result = ib.let("result", ir.op.block.row_sum(x))

        # Store result
        output_new = ib.let("output_new", ir.op.block.store(result, 0, 0, 8, 1, output_tensor))

        # Return the updated output tensor
        ib.return_stmt(output_new)

    return f.get_result()


def BuildRowExpandDivFunction(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build rowexpanddiv InCore function.

    Performs row-wise broadcast division: input_x[i,j] / input_row[i,0]

    Args:
        ib: IR Builder instance
        dtype: Data type for computation

    Returns:
        ir.Function: The rowexpanddiv function
    """
    with ib.function("rowexpanddiv") as f:
        # Parameters
        input_x = f.param("input_x", ir.TensorType([8, 8], dtype))
        input_row = f.param("input_row", ir.TensorType([8, 1], dtype))
        output_tensor = f.param("output", ir.TensorType([8, 8], dtype))
        f.return_type(ir.TensorType([8, 8], dtype))

        # Load tiles
        x = ib.let("x", ir.op.block.load(input_x, 0, 0, 8, 8))
        row_vals = ib.let("row_vals", ir.op.block.load(input_row, 0, 0, 8, 1))

        # Row expand divide (8x8, 8x1 -> 8x8)
        result = ib.let("result", ir.op.block.row_expand_div(x, row_vals))

        # Store result
        output_new = ib.let("output_new", ir.op.block.store(result, 0, 0, 8, 8, output_tensor))

        # Return the updated output tensor
        ib.return_stmt(output_new)

    return f.get_result()


def BuildDynamicSoftmaxFunction(ib: IRBuilder, dtype: DataType = DataType.FP32):
    """Build dynamic_softmax Orchestration function.

    Main function that orchestrates the softmax computation using helper functions.
    Includes FOR loop for processing full tiles and IF statement for tail processing.

    Args:
        ib: IR Builder instance
        dtype: Data type for computation

    Returns:
        ir.Function: The dynamic_softmax function
    """
    with ib.function("dynamic_softmax") as f:
        # Parameters - 6 memref tensors
        input_tensor = f.param("input", ir.TensorType([128, 128], dtype))
        output_tensor = f.param("output", ir.TensorType([128, 128], dtype))
        temp_rowmax = f.param("temp_rowmax", ir.TensorType([8, 1], dtype))
        temp_shifted = f.param("temp_shifted", ir.TensorType([8, 8], dtype))
        temp_exp = f.param("temp_exp", ir.TensorType([8, 8], dtype))
        temp_rowsum = f.param("temp_rowsum", ir.TensorType([8, 1], dtype))
        f.return_type(ir.TensorType([128, 128], dtype))

        # Scalar declarations
        num_full_tiles = ib.let("num_full_tiles", ir.ConstInt(4, DataType.INT32, ir.Span.unknown()))
        tail_rows = ib.let("tail_rows", ir.ConstInt(0, DataType.INT32, ir.Span.unknown()))
        zero = ib.let("zero", ir.ConstInt(0, DataType.INT32, ir.Span.unknown()))
        has_tail = ib.let("has_tail", ir.ConstBool(False, ir.Span.unknown()))

        # Create loop variable
        tile_idx = ib.var("tile_idx", ir.ScalarType(DataType.INT32))

        # FOR loop to process full tiles
        with ib.for_loop(tile_idx, 0, num_full_tiles, 1) as loop:
            # Iteration argument: output tensor is carried across iterations
            output_iter = loop.iter_arg("output_iter", output_tensor)
            loop.return_var("output_updated")

            # Inside loop: call helper functions with return value capture for SSA
            # 1. Call rowmax - returns updated temp_rowmax
            rowmax_op = ir.GlobalVar("rowmax")
            rowmax_call = ir.Call(rowmax_op, [input_tensor, temp_rowmax], ir.Span.unknown())
            temp_rowmax_updated = ib.let("temp_rowmax_updated", rowmax_call)

            # 2. Call rowexpandsub - returns updated temp_shifted
            rowexpandsub_op = ir.GlobalVar("rowexpandsub")
            rowexpandsub_call = ir.Call(
                rowexpandsub_op, [input_tensor, temp_rowmax_updated, temp_shifted], ir.Span.unknown()
            )
            temp_shifted_updated = ib.let("temp_shifted_updated", rowexpandsub_call)

            # 3. Call elem_exp - returns updated temp_exp
            elem_exp_op = ir.GlobalVar("elem_exp")
            elem_exp_call = ir.Call(elem_exp_op, [temp_shifted_updated, temp_exp], ir.Span.unknown())
            temp_exp_updated = ib.let("temp_exp_updated", elem_exp_call)

            # 4. Call rowsum - returns updated temp_rowsum
            rowsum_op = ir.GlobalVar("rowsum")
            rowsum_call = ir.Call(rowsum_op, [temp_exp_updated, temp_rowsum], ir.Span.unknown())
            temp_rowsum_updated = ib.let("temp_rowsum_updated", rowsum_call)

            # 5. Call rowexpanddiv - returns updated output
            rowexpanddiv_op = ir.GlobalVar("rowexpanddiv")
            rowexpanddiv_call = ir.Call(
                rowexpanddiv_op, [temp_exp_updated, temp_rowsum_updated, output_iter], ir.Span.unknown()
            )
            output_from_call = ib.let("output_from_call", rowexpanddiv_call)

            # Yield updated output tensor
            ib.emit(ir.YieldStmt([output_from_call], ir.Span.unknown()))

        # Get output after loop
        output_after_loop = loop.output()

        # Check if there's a tail to process
        has_tail = ib.let("has_tail", tail_rows > zero)

        # IF statement for tail processing
        with ib.if_stmt(has_tail) as if_builder:
            if_builder.return_var("output_final", ir.TensorType([128, 128], dtype))

            # Then branch: process tail with SSA return values
            # Call the same sequence of functions for tail
            # 1. Call rowmax - returns updated temp_rowmax
            tail_rowmax_op = ir.GlobalVar("rowmax")
            tail_rowmax_call = ir.Call(tail_rowmax_op, [input_tensor, temp_rowmax], ir.Span.unknown())
            tail_temp_rowmax_updated = ib.let("tail_temp_rowmax_updated", tail_rowmax_call)

            # 2. Call rowexpandsub - returns updated temp_shifted
            tail_rowexpandsub_op = ir.GlobalVar("rowexpandsub")
            tail_rowexpandsub_call = ir.Call(
                tail_rowexpandsub_op,
                [input_tensor, tail_temp_rowmax_updated, temp_shifted],
                ir.Span.unknown(),
            )
            tail_temp_shifted_updated = ib.let("tail_temp_shifted_updated", tail_rowexpandsub_call)

            # 3. Call elem_exp - returns updated temp_exp
            tail_elem_exp_op = ir.GlobalVar("elem_exp")
            tail_elem_exp_call = ir.Call(
                tail_elem_exp_op, [tail_temp_shifted_updated, temp_exp], ir.Span.unknown()
            )
            tail_temp_exp_updated = ib.let("tail_temp_exp_updated", tail_elem_exp_call)

            # 4. Call rowsum - returns updated temp_rowsum
            tail_rowsum_op = ir.GlobalVar("rowsum")
            tail_rowsum_call = ir.Call(
                tail_rowsum_op, [tail_temp_exp_updated, temp_rowsum], ir.Span.unknown()
            )
            tail_temp_rowsum_updated = ib.let("tail_temp_rowsum_updated", tail_rowsum_call)

            # 5. Call rowexpanddiv - returns updated output
            tail_rowexpanddiv_op = ir.GlobalVar("rowexpanddiv")
            tail_rowexpanddiv_call = ir.Call(
                tail_rowexpanddiv_op,
                [tail_temp_exp_updated, tail_temp_rowsum_updated, output_after_loop],
                ir.Span.unknown(),
            )
            output_with_tail = ib.let("output_with_tail", tail_rowexpanddiv_call)

            # Yield updated tensor
            ib.emit(ir.YieldStmt([output_with_tail], ir.Span.unknown()))

            # Else branch: no tail processing
            if_builder.else_()
            ib.emit(ir.YieldStmt([output_after_loop], ir.Span.unknown()))

        # Get final output from if statement
        output_final = if_builder.output()

        # Return final output
        ib.return_stmt(output_final)

    return f.get_result()


def BuildDynamicSoftmaxIR(dtype: DataType = DataType.FP32):
    """Build complete dynamic softmax IR with all functions.

    Creates all InCore helper functions and the main Orchestration function.

    Args:
        dtype: Data type for computation

    Returns:
        ir.Program: The complete program with all functions
    """
    ib = IRBuilder()

    # Build all InCore functions
    print("Building InCore functions...")
    rowmax_func = BuildRowMaxFunction(ib, dtype)
    rowexpandsub_func = BuildRowExpandSubFunction(ib, dtype)
    elem_exp_func = BuildElemExpFunction(ib, dtype)
    rowsum_func = BuildRowSumFunction(ib, dtype)
    rowexpanddiv_func = BuildRowExpandDivFunction(ib, dtype)
    print("✓ InCore functions built")

    # Build Orchestration function
    print("Building Orchestration function...")
    dynamic_softmax_func = BuildDynamicSoftmaxFunction(ib, dtype)
    print("✓ Orchestration function built")

    # Create program with all functions
    functions = [
        rowmax_func,
        rowexpandsub_func,
        elem_exp_func,
        rowsum_func,
        rowexpanddiv_func,
        dynamic_softmax_func,
    ]

    program = ir.Program(functions, "dynamic_softmax_module", ir.Span.unknown())
    return program


def main():
    """Main entry point for dynamic softmax code generation."""

    print("=" * 70)
    print("Dynamic Softmax Code Generation (PTO Assembly)")
    print("=" * 70)

    # Configuration
    dtype = DataType.FP32

    print(f"\nConfiguration: {dtype}")
    print("Generating PTO assembly format (.pto files)")

    # Step 1: Build IR
    print("\n[1] Building IR using IRBuilder and block operations...")
    program = BuildDynamicSoftmaxIR(dtype)
    print("✓ IR construction complete")

    # Step 2: Print original IR (preview)
    print("\n[2] Original IR (Python syntax - preview):")
    print("-" * 70)
    ir_text = ir.python_print(program)
    lines = ir_text.split("\n")
    preview_lines = min(30, len(lines))
    print("\n".join(lines[:preview_lines]))
    if len(lines) > preview_lines:
        print(f"\n... ({len(lines) - preview_lines} more lines)")
    print("-" * 70)

    # Step 3: Compile with passes and codegen
    print("\n[3] Compiling with PassManager and PTOCodegen...")
    print("    - Running optimization passes (XPlatform strategy)")
    print("    - Dumping IR after each pass")
    print("    - Generating PTO assembly")

    output_dir = ir.compile(
        program,
        strategy=ir.OptimizationStrategy.XPlatform,
        dump_passes=True,
    )
    print("✓ Compilation complete")
    print(f"✓ All artifacts saved to: {output_dir}")

    # Step 4: Read and preview the generated PTO assembly
    pto_file = os.path.join(output_dir, "output.pto")
    with open(pto_file, "r") as f:
        pto_code = f.read()

    print("\n[4] Generated PTO Assembly (preview):")
    print("=" * 70)
    # Print first 40 lines as preview
    lines = pto_code.split("\n")
    preview_lines = min(40, len(lines))
    print("\n".join(lines[:preview_lines]))
    if len(lines) > preview_lines:
        print(f"\n... ({len(lines) - preview_lines} more lines)")
    print("=" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("Compilation complete!")
    print("=" * 70)
    print("\nSummary:")
    print("  - Module name: dynamic_softmax_module")
    print(f"  - Number of functions: {len(program.functions)}")
    print(f"  - Output directory: {output_dir}")
    print("  - Output format: PTO assembly (.pto)")
    print(f"  - Data type: {dtype}")
    print("  - Optimization strategy: XPlatform")
    print("\nFunctions implemented:")
    print("  - rowmax (InCore): row-wise maximum reduction")
    print("  - rowexpandsub (InCore): row-wise broadcast subtraction")
    print("  - elem_exp (InCore): element-wise exponential")
    print("  - rowsum (InCore): row-wise sum reduction")
    print("  - rowexpanddiv (InCore): row-wise broadcast division")
    print("  - dynamic_softmax (Orchestration): main softmax orchestration")
    print("\nGenerated artifacts:")
    print(f"  - {os.path.join(output_dir, 'original.py')} - Original IR")
    print(f"  - {os.path.join(output_dir, 'after_*.py')} - IR after each pass")
    print(f"  - {os.path.join(output_dir, 'output.pto')} - Final PTO assembly")
    print("\nThe generated PTO assembly:")
    print("  - Uses SSA-style variable naming with % prefix")
    print("  - Includes type annotations for all operations")
    print("  - Compatible with PTO ISA specification")
    print("  - Entry point: @dynamic_softmax")
    print("=" * 70)


if __name__ == "__main__":
    main()
