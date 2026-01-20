# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example: Building block operations using IRBuilder

This example demonstrates how to use block operations from the pypto.ir.op.block module,
including memory operations, element-wise operations, unary operations, and reduction operations.
"""

from pypto.ir.builder import IRBuilder
from pypto.ir.op import block
from pypto.pypto_core import DataType, ir


def build_block_elementwise_example():
    """Build an example function using block element-wise operations.

    This function demonstrates:
    1. Copy data from tensor to unified buffer (tile)
    2. Perform element-wise operations (add, multiply)
    3. Copy results back to tensor
    """
    ib = IRBuilder()

    with ib.function("block_elementwise_example") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
        output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Define tile size and offsets
        tile_height = 32
        tile_width = 32
        row_offset = 0
        col_offset = 0

        # Copy data from tensor to unified buffer
        tile_a = ib.let("tile_a", block.ub_copy_in(input_a, row_offset, col_offset, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.ub_copy_in(input_b, row_offset, col_offset, tile_height, tile_width))

        # Perform element-wise operations
        # tile_c = (tile_a + tile_b) * 2.0
        tile_sum = ib.let("tile_sum", block.add(tile_a, tile_b))
        tile_c = ib.let("tile_c", block.mul(tile_sum, 2.0))

        # Copy results back to tensor
        result = ib.let(
            "result", block.ub_copy_out(tile_c, row_offset, col_offset, tile_height, tile_width, output)
        )

        # Return result
        ib.return_stmt(result)

    return f.get_result()


def build_block_reduction_example():
    """Build an example function using block reduction operations.

    This function demonstrates:
    1. Copy data from tensor to tile
    2. Perform reduction operation (sum)
    3. Copy reduction result back to tensor
    """
    ib = IRBuilder()

    with ib.function("block_reduction_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
        f.return_type(ir.TensorType([128, 1], DataType.FP32))

        # Define tile size
        tile_height = 32
        tile_width = 128

        # Define offsets
        row_offset = 0
        col_offset = 0

        # Copy data from tensor to tile
        tile_in = ib.let(
            "tile_in", block.ub_copy_in(input_tensor, row_offset, col_offset, tile_height, tile_width)
        )

        # Perform reduction sum along the last axis (axis=1)
        tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))

        # Copy reduction result back to tensor
        result = ib.let("result", block.ub_copy_out(tile_sum, row_offset, 0, tile_height, 1, output_tensor))

        # Return result
        ib.return_stmt(result)

    return f.get_result()


def build_block_unary_example():
    """Build an example function using block unary operations.

    This function demonstrates:
    1. Copy data from tensor to tile
    2. Perform unary operation (sqrt)
    3. Copy result back to tensor
    """
    ib = IRBuilder()

    with ib.function("block_unary_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Define tile size
        tile_height = 32
        tile_width = 32

        # Define offsets
        row_offset = 0
        col_offset = 0

        # Copy data from tensor to tile
        tile_in = ib.let(
            "tile_in", block.ub_copy_in(input_tensor, row_offset, col_offset, tile_height, tile_width)
        )

        # Perform unary operation: sqrt
        tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_in))

        # Copy result back to tensor
        result = ib.let(
            "result",
            block.ub_copy_out(tile_sqrt, row_offset, col_offset, tile_height, tile_width, output_tensor),
        )

        # Return result
        ib.return_stmt(result)

    return f.get_result()


def build_complex_block_computation():
    """Build a complex block computation example.

    This function demonstrates the combination of various block operations:
    - Memory operations
    - Element-wise operations
    - Unary operations
    - Reduction operations

    Computation: output = sum(sqrt(a * b + c), axis=1)
    """
    ib = IRBuilder()

    with ib.function("complex_block_computation") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
        input_c = f.param("input_c", ir.TensorType([128, 128], DataType.FP32))
        output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
        f.return_type(ir.TensorType([128, 1], DataType.FP32))

        # Define tile size
        tile_height = 32
        tile_width = 128

        # Define offsets
        row_offset = 0
        col_offset = 0

        # Copy data from tensor to unified buffer
        tile_a = ib.let("tile_a", block.ub_copy_in(input_a, row_offset, col_offset, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.ub_copy_in(input_b, row_offset, col_offset, tile_height, tile_width))
        tile_c = ib.let("tile_c", block.ub_copy_in(input_c, row_offset, col_offset, tile_height, tile_width))

        # Perform computation: a * b + c
        tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
        tile_add = ib.let("tile_add", block.add(tile_mul, tile_c))

        # Perform unary operation: sqrt
        tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_add))

        # Perform reduction: sum(axis=1)
        tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))

        # Copy result back to tensor
        result = ib.let("result", block.ub_copy_out(tile_sum, row_offset, 0, tile_height, 1, output))

        # Return result
        ib.return_stmt(result)

    return f.get_result()


if __name__ == "__main__":
    print("=" * 80)
    print("Block Operations Examples")
    print("=" * 80)

    # Example 1: Element-wise operations
    print("\n1. Block Element-wise Operations Example")
    print("-" * 80)
    func1 = build_block_elementwise_example()
    print(func1)

    # Example 2: Reduction operations
    print("\n2. Block Reduction Operations Example")
    print("-" * 80)
    func2 = build_block_reduction_example()
    print(func2)

    # Example 3: Unary operations
    print("\n3. Block Unary Operations Example")
    print("-" * 80)
    func3 = build_block_unary_example()
    print(func3)

    # Example 4: Complex block computation
    print("\n4. Complex Block Computation Example")
    print("-" * 80)
    func4 = build_complex_block_computation()
    print(func4)

    print("\n" + "=" * 80)
    print("All examples built successfully!")
    print("=" * 80)
