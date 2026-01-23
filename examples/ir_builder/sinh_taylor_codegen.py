# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Sinh Taylor Expansion Code Generation Example

Demonstrates:
1. Building sinh computation using IRBuilder and tile operations
2. Using PTOCodegen to generate PTO assembly (.pto format)
3. Printing the generated PTO assembly

sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + ...
        = x + x³/6 + x⁵/120 + x⁷/5040 + ...

Algorithm:
    result = x
    term = x
    x_squared = x * x
    for each term:
        term = term * x_squared / divisor
        result = result + term
"""

import os

from pypto import DataType, ir
from pypto.ir import IRBuilder

# Elements per NEON 128-bit vector
# Keys match DataType.to_string() output format
ARM64_VECTOR_LANES = {
    "fp32": 4,
    "fp16": 8,
    "fp64": 2,
    "int8": 16,
    "int16": 8,
    "int32": 4,
    "int64": 2,
    "uint8": 16,
    "uint16": 8,
    "uint32": 4,
    "uint64": 2,
}

# ARM64 Physical Tile Size
# Physical_Row_Size: Optimal repeat count for vector pipeline performance
ARM64_PHYSICAL_ROW_SIZE = 1  # Optimal repeat count for ARM64

# Ascend vector type mappings (LocalTensor)
# Keys match DataType.to_string() output format
ASCEND_VECTOR_LANES = {
    "fp32": 8,  # 256-bit vector / 32-bit
    "fp16": 16,  # 256-bit vector / 16-bit
    "bfloat16": 16,
    "int32": 8,
    "int8": 32,
    "uint8": 32,
}

ASCEND_PHYSICAL_ROW_SIZE = 32  # Optimal repeat count for Ascend 910B pipeline

# Maximum tile size in bytes (16KB)
MAX_TILE_BYTES = 16 * 1024


def compute_tile_shape(dtype: DataType, target_isa: str = "arm64") -> tuple:
    """
    Compute optimal tile shape based on data type and target ISA.

    Rules:
    1) col should be multiples of VECTOR_LANES
    2) row should be multiple of PHYSICAL_ROW_SIZE
    3) byte size of the TILE should be no greater than 16KB

    Returns:
        (rows, cols) tuple
    """
    dtype_str = dtype.to_string()

    # Get ISA-specific parameters
    if target_isa == "arm64":
        vector_lanes = ARM64_VECTOR_LANES.get(dtype_str, 4)
        physical_row_size = ARM64_PHYSICAL_ROW_SIZE
    elif target_isa == "ascend910b":
        vector_lanes = ASCEND_VECTOR_LANES.get(dtype_str, 8)
        physical_row_size = ASCEND_PHYSICAL_ROW_SIZE
    else:
        # Default to ARM64
        vector_lanes = ARM64_VECTOR_LANES.get(dtype_str, 4)
        physical_row_size = ARM64_PHYSICAL_ROW_SIZE

    # Calculate element size in bytes using get_bit() method
    element_bytes = dtype.get_bit() // 8
    if element_bytes == 0:
        # For sub-byte types (4-bit), round up to 1 byte
        element_bytes = 1

    # Start with col = vector_lanes (minimum aligned column count)
    # Try to maximize cols as multiples of vector_lanes while staying under 16KB

    # Maximum elements that fit in 16KB
    max_elements = MAX_TILE_BYTES // element_bytes

    # Start with a reasonable row count based on physical_row_size
    # For Ascend, we want 32 rows; for ARM64/CUDA, we want 1 row minimum
    # but we'll try to increase rows to fill the tile size

    # Strategy: Use cols as multiple of vector_lanes
    # Compute how many columns we can have
    # cols = N * vector_lanes, where N is a power of 2 for alignment

    # Try different row configurations
    best_rows = physical_row_size
    best_cols = vector_lanes
    best_total = best_rows * best_cols

    for row_mult in [1, 2, 4, 8, 16, 32, 64, 128]:
        rows = physical_row_size * row_mult

        # Compute max cols for this row count
        max_cols_for_rows = max_elements // rows

        # Round down to multiple of vector_lanes
        cols = (max_cols_for_rows // vector_lanes) * vector_lanes

        if cols < vector_lanes:
            break  # Too many rows, can't fit even one vector width

        total = rows * cols
        if total > best_total and total * element_bytes <= MAX_TILE_BYTES:
            best_rows = rows
            best_cols = cols
            best_total = total

    return best_rows, best_cols


def build_sinh_ir(dtype: DataType = DataType.FP32, target_isa: str = "arm64"):
    """Build sinh Taylor expansion IR using IRBuilder and tile operations.

    Includes control flow (for loop) to demonstrate codegen capabilities.

    Args:
        dtype: Data type for tiles
        target_isa: Target ISA for tile shape computation

    Returns:
        ir.Program: The sinh Taylor expansion program
    """
    ib = IRBuilder()

    rows, cols = 32, 128  # compute_tile_shape(dtype, target_isa)
    # tile_elements = rows * cols

    with ib.function("sinh_taylor") as f:
        # Create MemRef for input tensor (global memory)
        input_memref = ir.MemRef(
            ir.MemorySpace.DDR,  # Global memory (DDR)
            ir.ConstInt(0, DataType.INT64, ir.Span.unknown()),  # Base address
            0,  # Size (can be 0 for dynamic)
            0,  # ID
        )

        # Create MemRef for output tensor (global memory)
        output_memref = ir.MemRef(
            ir.MemorySpace.DDR,  # Global memory (DDR)
            ir.ConstInt(0, DataType.INT64, ir.Span.unknown()),  # Base address
            0,  # Size (can be 0 for dynamic)
            1,  # ID
        )

        # Parameters: input tensor and output tensor with MemRef
        input_tensor = f.param("input", ir.TensorType([128, 128], dtype, input_memref))
        output_tensor = f.param("output", ir.TensorType([128, 128], dtype, output_memref))
        f.return_type(ir.TensorType([128, 128], dtype))

        x = ib.var("x", ir.TileType([rows, cols], dtype))
        x_squared = ib.var("x_squared", ir.TileType([rows, cols], dtype))
        term = ib.var("term", ir.TileType([rows, cols], dtype))
        result = ib.var("result", ir.TileType([rows, cols], dtype))

        # Scalar declarations for loop control (similar to pto_isa_sinh.py)
        # total_elements = ib.let("total_elements", ir.ConstInt(1024, DataType.INT32, ir.Span.unknown()))
        # tile_size = ib.let("tile_size", ir.ConstInt(tile_elements, DataType.INT32, ir.Span.unknown()))
        num_full_tiles = ib.let("num_full_tiles", ir.ConstInt(4, DataType.INT32, ir.Span.unknown()))
        tail_elements = ib.let("tail_elements", ir.ConstInt(0, DataType.INT32, ir.Span.unknown()))
        # offset = ib.let("offset", ir.ConstInt(0, DataType.INT32, ir.Span.unknown()))
        zero = ib.let("zero", ir.ConstInt(0, DataType.INT32, ir.Span.unknown()))
        has_tail = ib.let("has_tail", ir.ConstBool(False, ir.Span.unknown()))

        # Create loop variable for iterating over tiles
        tile_idx = ib.var("tile_idx", ir.ScalarType(DataType.INT32))

        # For loop to process multiple tiles
        with ib.for_loop(tile_idx, 0, num_full_tiles, 1):
            # Inside the loop: sinh computation on each tile
            # Load tile from input tensor using loop variable as index
            x = ib.let("x", ir.op.block.load(input_tensor, tile_idx, 0, rows, cols))
            result = ib.let("result", ir.op.block.muls(x, 1.0))
            x_squared = ib.let("x_squared", ir.op.block.mul(x, x))
            term = ib.let("term", ir.op.block.muls(x, 1.0))

            # Taylor expansion terms
            # Term 2: x³/6
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 6.0))
            result = ib.let("result", ir.op.block.add(result, term))

            # Term 3: x⁵/120
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 20.0))
            result = ib.let("result", ir.op.block.add(result, term))

            # Term 4: x⁷/5040
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 42.0))
            result = ib.let("result", ir.op.block.add(result, term))

            # Term 5: x⁹/362880
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 72.0))
            result = ib.let("result", ir.op.block.add(result, term))

            # Term 6: x¹¹/11!
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 110.0))
            result = ib.let("result", ir.op.block.add(result, term))

            # Term 7: x¹³/13!
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 156.0))
            result = ib.let("result", ir.op.block.add(result, term))

            # Store result back to output tensor using loop variable as index
            ib.let("output", ir.op.block.store(result, tile_idx, 0, rows, cols, output_tensor))
        has_tail = ib.let("has_tail", tail_elements > zero)
        with ib.if_stmt(has_tail):
            x = ib.let("x", ir.op.block.load(input_tensor, num_full_tiles, 0, rows, cols))
            result = ib.let("result", ir.op.block.muls(x, 1.0))
            x_squared = ib.let("x_squared", ir.op.block.mul(x, x))
            term = ib.let("term", ir.op.block.muls(x, 1.0))
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 6.0))
            result = ib.let("result", ir.op.block.add(result, term))
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 20.0))
            result = ib.let("result", ir.op.block.add(result, term))
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 42.0))
            result = ib.let("result", ir.op.block.add(result, term))
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 72.0))
            result = ib.let("result", ir.op.block.add(result, term))
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 110.0))
            result = ib.let("result", ir.op.block.add(result, term))
            term = ib.let("term", ir.op.block.mul(term, x_squared))
            term = ib.let("term", ir.op.block.divs(term, 156.0))
            result = ib.let("result", ir.op.block.add(result, term))
            ib.let("output2", ir.op.block.store(result, num_full_tiles, 0, rows, cols, output_tensor))
        ib.return_stmt(output_tensor)

    func = f.get_result()
    program = ir.Program([func], "sinh_taylor", ir.Span.unknown())
    return program


def main():
    """Main entry point for sinh Taylor expansion code generation."""

    print("=" * 70)
    print("Sinh Taylor Expansion Code Generation (PTO Assembly)")
    print("=" * 70)

    # Configuration
    dtype = DataType.FP32
    target_isa = "arm64"

    print(f"\nConfiguration: {dtype} @ {target_isa}")
    print("Generating PTO assembly format (.pto files)")

    # Step 1: Build IR
    print("\n[1] Building IR using IRBuilder and tile operations...")
    program = build_sinh_ir(dtype, target_isa)
    print("✓ IR construction complete")

    # Step 2: Print original IR (preview)
    print("\n[2] Original IR (Python syntax - preview):")
    print("-" * 70)
    ir_text = ir.python_print(program)
    lines = ir_text.split("\n")
    preview_lines = min(20, len(lines))
    print("\n".join(lines[:preview_lines]))
    if len(lines) > preview_lines:
        print(f"\n... ({len(lines) - preview_lines} more lines)")
    print("-" * 70)

    # Step 3: Compile with passes and codegen
    print("\n[3] Compiling with PassManager and PTOCodegen...")
    print("    - Running optimization passes (Custom2 strategy)")
    print("    - Dumping IR after each pass")
    print("    - Generating PTO assembly")

    output_dir = ir.compile(
        program,
        strategy=ir.OptimizationStrategy.Custom2,
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
    # Print first 30 lines as preview
    lines = pto_code.split("\n")
    preview_lines = min(30, len(lines))
    print("\n".join(lines[:preview_lines]))
    if len(lines) > preview_lines:
        print(f"\n... ({len(lines) - preview_lines} more lines)")
    print("=" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("Compilation complete!")
    print("=" * 70)
    print("\nSummary:")
    func = list(program.functions.values())[0]
    print(f"  - Function name: {func.name}")
    print(f"  - Output directory: {output_dir}")
    print("  - Output format: PTO assembly (.pto)")
    print(f"  - Data type: {dtype}")
    print("  - Optimization strategy: Custom2")
    print("  - Taylor terms: 7 terms (up to x¹³/13!)")
    print("  - Operations used: tile.mul, tile.divs, tile.add")
    print("  - Control flow: for loop (4 iterations)")
    print("\nGenerated artifacts:")
    print(f"  - {os.path.join(output_dir, 'original.py')} - Original IR")
    print(f"  - {os.path.join(output_dir, 'after_*.py')} - IR after each pass")
    print(f"  - {os.path.join(output_dir, 'output.pto')} - Final PTO assembly")
    print("\nThe generated PTO assembly:")
    print("  - Uses SSA-style variable naming with % prefix")
    print("  - Includes type annotations for all operations")
    print("  - Can be used as reference for PTO ISA programs")
    print("  - Compatible with PTO assembly syntax")
    print("=" * 70)


if __name__ == "__main__":
    main()
