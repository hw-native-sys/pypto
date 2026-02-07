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
        tile_a = ib.let("tile_a", block.load(input_a, row_offset, col_offset, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, row_offset, col_offset, tile_height, tile_width))

        # Perform element-wise operations
        # tile_c = (tile_a + tile_b) * 2.0
        tile_sum = ib.let("tile_sum", block.add(tile_a, tile_b))
        tile_c = ib.let("tile_c", block.muls(tile_sum, 2.0))

        # Copy results back to tensor
        result = ib.let(
            "result", block.store(tile_c, row_offset, col_offset, tile_height, tile_width, output)
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
        tile_in = ib.let("tile_in", block.load(input_tensor, row_offset, col_offset, tile_height, tile_width))

        # Perform reduction sum along the last axis (axis=1)
        tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))

        # Copy reduction result back to tensor
        result = ib.let("result", block.store(tile_sum, row_offset, 0, tile_height, 1, output_tensor))

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
        tile_in = ib.let("tile_in", block.load(input_tensor, row_offset, col_offset, tile_height, tile_width))

        # Perform unary operation: sqrt
        tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_in))

        # Copy result back to tensor
        result = ib.let(
            "result",
            block.store(tile_sqrt, row_offset, col_offset, tile_height, tile_width, output_tensor),
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
        tile_a = ib.let("tile_a", block.load(input_a, row_offset, col_offset, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, row_offset, col_offset, tile_height, tile_width))
        tile_c = ib.let("tile_c", block.load(input_c, row_offset, col_offset, tile_height, tile_width))

        # Perform computation: a * b + c
        tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
        tile_add = ib.let("tile_add", block.add(tile_mul, tile_c))

        # Perform unary operation: sqrt
        tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_add))

        # Perform reduction: sum(axis=1)
        tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))

        # Copy result back to tensor
        result = ib.let("result", block.store(tile_sum, row_offset, 0, tile_height, 1, output))

        # Return result
        ib.return_stmt(result)

    return f.get_result()


def build_block_cast_example():
    """Build an example function using block.cast operation.

    This function demonstrates:
    1. Load BF16 data from tensor
    2. Cast BF16 to FP32 for computation
    3. Perform computation in FP32
    4. Cast back to BF16 for storage
    5. Store result
    """
    ib = IRBuilder()

    with ib.function("block_cast_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.BF16))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.BF16))
        f.return_type(ir.TensorType([128, 128], DataType.BF16))

        # Define tile size
        tile_height = 32
        tile_width = 32

        # Define offsets
        row_offset = 0
        col_offset = 0

        # Load BF16 data from tensor
        tile_bf16 = ib.let(
            "tile_bf16", block.load(input_tensor, row_offset, col_offset, tile_height, tile_width)
        )

        # Cast BF16 to FP32 for computation
        tile_fp32 = ib.let("tile_fp32", block.cast(tile_bf16, DataType.FP32))

        # Perform computation in FP32 (example: multiply by 2.0)
        tile_result_fp32 = ib.let("tile_result_fp32", block.muls(tile_fp32, 2.0))

        # Cast back to BF16 for storage
        tile_result_bf16 = ib.let("tile_result_bf16", block.cast(tile_result_fp32, DataType.BF16))

        # Store result
        result = ib.let(
            "result",
            block.store(tile_result_bf16, row_offset, col_offset, tile_height, tile_width, output_tensor),
        )

        # Return result
        ib.return_stmt(result)

    return f.get_result()


def build_block_zeros_example():
    """Build an example function using block.zeros operation.

    This function demonstrates:
    1. Create zero-initialized tiles with different dimensions
    2. Load data from tensor
    3. Add zeros to data (to demonstrate zeros work correctly)
    4. Store result
    """
    ib = IRBuilder()

    with ib.function("block_zeros_example") as f:
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

        # Create zero-initialized tiles with different dimensions
        # 2D tile
        tile_zeros_2d = ib.let("tile_zeros_2d", block.zeros([tile_height, tile_width], DataType.FP32))

        # Load data from tensor
        tile_data = ib.let(
            "tile_data", block.load(input_tensor, row_offset, col_offset, tile_height, tile_width)
        )

        # Add zeros to data (should return original data)
        tile_result = ib.let("tile_result", block.add(tile_data, tile_zeros_2d))

        # Store result
        result = ib.let(
            "result",
            block.store(tile_result, row_offset, col_offset, tile_height, tile_width, output_tensor),
        )

        # Return result
        ib.return_stmt(result)

    return f.get_result()


def build_block_minimum_example():
    """Build an example function using block.minimum operation.

    This function demonstrates:
    1. Load two tiles from tensors
    2. Compute element-wise minimum
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_minimum_example") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Define tile size
        tile_height = 32
        tile_width = 32

        # Define offsets
        row_offset = 0
        col_offset = 0

        # Load two tiles
        tile_a = ib.let("tile_a", block.load(input_a, row_offset, col_offset, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.load(input_b, row_offset, col_offset, tile_height, tile_width))

        # Compute element-wise minimum
        tile_min = ib.let("tile_min", block.minimum(tile_a, tile_b))

        # Store result
        result = ib.let(
            "result",
            block.store(tile_min, row_offset, col_offset, tile_height, tile_width, output_tensor),
        )

        # Return result
        ib.return_stmt(result)

    return f.get_result()


# ============================================================================
# Phase 1: Core Extension OPs (transpose, reshape, log, abs, relu)
# ============================================================================


def build_block_transpose_example():
    """Build an example function using block.transpose operation.

    This function demonstrates:
    1. Load a 2D tile from tensor
    2. Transpose the tile by swapping axis 0 and axis 1 (rows and columns)
    3. Store the transposed result
    """
    ib = IRBuilder()

    with ib.function("block_transpose_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 64], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([64, 128], DataType.FP32))
        f.return_type(ir.TensorType([64, 128], DataType.FP32))

        # Load tile [32, 64]
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 64))

        # Transpose by swapping axis 0 and axis 1: [32, 64] -> [64, 32]
        tile_t = ib.let("tile_t", block.transpose(tile_in, axis1=0, axis2=1))

        # Store result
        result = ib.let("result", block.store(tile_t, 0, 0, 64, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_reshape_example():
    """Build an example function using block.reshape operation.

    This function demonstrates:
    1. Load a tile from tensor
    2. Reshape through multiple dimensions (2D -> 2D -> 3D -> 2D)
    3. Store the reshaped result
    """
    ib = IRBuilder()

    with ib.function("block_reshape_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile [32, 32] (1024 elements)
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))

        # Reshape to [64, 16] (1024 elements)
        tile_reshaped1 = ib.let("tile_reshaped1", block.reshape(tile_in, [64, 16]))

        # Reshape to 3D: [4, 8, 32] (1024 elements)
        tile_reshaped2 = ib.let("tile_reshaped2", block.reshape(tile_reshaped1, [4, 8, 32]))

        # Reshape back to 2D: [32, 32] (1024 elements)
        tile_result = ib.let("tile_result", block.reshape(tile_reshaped2, [32, 32]))

        # Store result
        result = ib.let("result", block.store(tile_result, 0, 0, 32, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_log_example():
    """Build an example function using block.log operation.

    This function demonstrates:
    1. Load tile from tensor
    2. Compute natural logarithm
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_log_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))

        # Compute natural logarithm
        tile_log = ib.let("tile_log", block.log(tile_in))

        # Store result
        result = ib.let("result", block.store(tile_log, 0, 0, 32, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_abs_example():
    """Build an example function using block.abs operation.

    This function demonstrates:
    1. Load tile from tensor
    2. Compute absolute value
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_abs_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))

        # Compute absolute value
        tile_abs = ib.let("tile_abs", block.abs(tile_in))

        # Store result
        result = ib.let("result", block.store(tile_abs, 0, 0, 32, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_relu_example():
    """Build an example function using block.relu operation.

    This function demonstrates:
    1. Load tile from tensor
    2. Apply ReLU activation (max(0, x))
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_relu_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))

        # Apply ReLU
        tile_relu = ib.let("tile_relu", block.relu(tile_in))

        # Store result
        result = ib.let("result", block.store(tile_relu, 0, 0, 32, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


# ============================================================================
# Phase 2: Reduction and Comparison OPs
# ============================================================================


def build_block_row_min_example():
    """Build an example function using block.min operation with axis=1.

    This function demonstrates:
    1. Load 2D tile from tensor
    2. Compute row-wise min (output shape: [rows, 1])
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_row_min_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
        f.return_type(ir.TensorType([128, 1], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))

        # Compute row-wise min (reduce along axis 1 with keepdim=True)
        tile_min = ib.let("tile_min", block.min(tile_in, axis=1, keepdim=True))

        # Store result
        result = ib.let("result", block.store(tile_min, 0, 0, 32, 1, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_row_max_example():
    """Build an example function using block.max operation with axis=1.

    This function demonstrates:
    1. Load 2D tile from tensor
    2. Compute row-wise max (reduce along axis 1, output shape: [rows, 1])
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_row_max_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
        f.return_type(ir.TensorType([128, 1], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))

        # Compute row-wise max (reduce along axis 1 with keepdim=True)
        tile_max = ib.let("tile_max", block.max(tile_in, axis=1, keepdim=True))

        # Store result
        result = ib.let("result", block.store(tile_max, 0, 0, 32, 1, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_col_min_example():
    """Build an example function using block.min operation with axis=0.

    This function demonstrates:
    1. Load 2D tile from tensor
    2. Compute column-wise min (reduce along axis 0, output shape: [1, cols])
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_col_min_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([1, 128], DataType.FP32))
        f.return_type(ir.TensorType([1, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 128, 32))

        # Compute column-wise min (reduce along axis 0 with keepdim=True)
        tile_min = ib.let("tile_min", block.min(tile_in, axis=0, keepdim=True))

        # Store result
        result = ib.let("result", block.store(tile_min, 0, 0, 1, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_col_max_example():
    """Build an example function using block.max operation with axis=0.

    This function demonstrates:
    1. Load 2D tile from tensor
    2. Compute column-wise max (reduce along axis 0, output shape: [1, cols])
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_col_max_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([1, 128], DataType.FP32))
        f.return_type(ir.TensorType([1, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 128, 32))

        # Compute column-wise max (reduce along axis 0 with keepdim=True)
        tile_max = ib.let("tile_max", block.max(tile_in, axis=0, keepdim=True))

        # Store result
        result = ib.let("result", block.store(tile_max, 0, 0, 1, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_row_sum_example():
    """Build an example function using block.sum operation with axis=1.

    This function demonstrates:
    1. Load 2D tile from tensor
    2. Compute row-wise sum (reduce along axis 1, output shape: [rows, 1])
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_row_sum_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
        f.return_type(ir.TensorType([128, 1], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 128))

        # Compute row-wise sum (reduce along axis 1 with keepdim=True)
        tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))

        # Store result
        result = ib.let("result", block.store(tile_sum, 0, 0, 32, 1, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_col_sum_example():
    """Build an example function using block.sum operation with axis=0.

    This function demonstrates:
    1. Load 2D tile from tensor
    2. Compute column-wise sum (reduce along axis 0, output shape: [1, cols])
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_col_sum_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([1, 128], DataType.FP32))
        f.return_type(ir.TensorType([1, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 128, 32))

        # Compute column-wise sum (reduce along axis 0 with keepdim=True)
        tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=0, keepdim=True))

        # Store result
        result = ib.let("result", block.store(tile_sum, 0, 0, 1, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_cmp_example():
    """Build an example function using block.cmp operation.

    This function demonstrates:
    1. Load two tiles from tensors
    2. Perform element-wise comparison using all comparison types
    3. Store result (boolean tile)
    """
    ib = IRBuilder()

    with ib.function("block_cmp_example") as f:
        # Define input and output parameters
        input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load two tiles
        tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 32))
        tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 32))

        # Compare tiles using all comparison types
        # 0=GT(>), 1=LT(<), 2=GE(>=), 3=LE(<=), 4=EQ(==), 5=NE(!=)
        cmp_types = [0, 1, 2, 3, 4, 5]
        cmp_names = ["gt", "lt", "ge", "le", "eq", "ne"]
        cmp_results = []
        for cmp_type, cmp_name in zip(cmp_types, cmp_names):
            # Create comparison with current type
            result_name = f"tile_cmp_{cmp_name}"
            cmp_result = ib.let(result_name, block.cmp(tile_a, tile_b, cmp_type))
            cmp_results.append(cmp_result)

        # Use the GT result for output (tile_a > tile_b)
        result = ib.let("result", block.store(cmp_results[0], 0, 0, 32, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_cmps_example():
    """Build an example function using block.cmps operation.

    This function demonstrates:
    1. Load tile from tensor
    2. Perform element-wise comparison with scalar using all comparison types
    3. Store result (boolean tile)
    """
    ib = IRBuilder()

    with ib.function("block_cmps_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))

        # Compare with scalar using all comparison types
        # 0=GT(>), 1=LT(<), 2=GE(>=), 3=LE(<=), 4=EQ(==), 5=NE(!=)
        cmp_types = [0, 1, 2, 3, 4, 5]
        cmp_names = ["gt", "lt", "ge", "le", "eq", "ne"]
        cmp_results = []
        for cmp_type, cmp_name in zip(cmp_types, cmp_names):
            # Create comparison with current type
            result_name = f"tile_cmp_{cmp_name}"
            cmp_result = ib.let(result_name, block.cmps(tile_in, 0.0, cmp_type))
            cmp_results.append(cmp_result)

        # Use the GT result for output (tile > 0.0)
        result = ib.let("result", block.store(cmp_results[0], 0, 0, 32, 32, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


# ============================================================================
# Phase 3: Column Expand and Scalar Expand OPs
# ============================================================================


def build_block_col_expand_example():
    """Build an example function using block.col_expand operation.

    This function demonstrates:
    1. Load a target tile and a column vector
    2. Expand column vector to target shape
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_col_expand_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        col_tensor = f.param("col_vec", ir.TensorType([1, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load target tile [32, 64]
        tile_target = ib.let("tile_target", block.load(input_tensor, 0, 0, 32, 64))

        # Load column vector [1, 64]
        tile_col = ib.let("tile_col", block.load(col_tensor, 0, 0, 1, 64))

        # Expand column vector to target shape
        tile_expanded = ib.let("tile_expanded", block.col_expand(tile_target, tile_col))

        # Store result
        result = ib.let("result", block.store(tile_expanded, 0, 0, 32, 64, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_col_expand_mul_example():
    """Build an example function using block.col_expand_mul operation.

    This function demonstrates:
    1. Load a tile and a column vector
    2. Expand and multiply column vector with tile
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_col_expand_mul_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        col_tensor = f.param("col_vec", ir.TensorType([1, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 64))

        # Load column vector
        tile_col = ib.let("tile_col", block.load(col_tensor, 0, 0, 1, 64))

        # Expand and multiply
        tile_result = ib.let("tile_result", block.col_expand_mul(tile_in, tile_col))

        # Store result
        result = ib.let("result", block.store(tile_result, 0, 0, 32, 64, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_col_expand_div_example():
    """Build an example function using block.col_expand_div operation.

    This function demonstrates:
    1. Load a tile and a column vector
    2. Expand column vector and divide tile by it
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_col_expand_div_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        col_tensor = f.param("col_vec", ir.TensorType([1, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 64))

        # Load column vector
        tile_col = ib.let("tile_col", block.load(col_tensor, 0, 0, 1, 64))

        # Expand and divide
        tile_result = ib.let("tile_result", block.col_expand_div(tile_in, tile_col))

        # Store result
        result = ib.let("result", block.store(tile_result, 0, 0, 32, 64, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_col_expand_sub_example():
    """Build an example function using block.col_expand_sub operation.

    This function demonstrates:
    1. Load a tile and a column vector
    2. Expand column vector and subtract from tile
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_col_expand_sub_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        col_tensor = f.param("col_vec", ir.TensorType([1, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 64))

        # Load column vector
        tile_col = ib.let("tile_col", block.load(col_tensor, 0, 0, 1, 64))

        # Expand and subtract
        tile_result = ib.let("tile_result", block.col_expand_sub(tile_in, tile_col))

        # Store result
        result = ib.let("result", block.store(tile_result, 0, 0, 32, 64, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_row_expand_add_example():
    """Build an example function using block.row_expand_add operation.

    This function demonstrates:
    1. Load a tile and a row vector
    2. Expand row vector and add to tile
    3. Store result
    """
    ib = IRBuilder()

    with ib.function("block_row_expand_add_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        row_tensor = f.param("row_vec", ir.TensorType([128, 1], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 64))

        # Load row vector
        tile_row = ib.let("tile_row", block.load(row_tensor, 0, 0, 32, 1))

        # Expand and add
        tile_result = ib.let("tile_result", block.row_expand_add(tile_in, tile_row))

        # Store result
        result = ib.let("result", block.store(tile_result, 0, 0, 32, 64, output_tensor))

        ib.return_stmt(result)

    return f.get_result()


def build_block_expands_example():
    """Build an example function using block.expands operation.

    This function demonstrates:
    1. Load a tile (defines target shape)
    2. Expand scalar to match tile shape
    3. Add expanded scalar to tile
    4. Store result
    """
    ib = IRBuilder()

    with ib.function("block_expands_example") as f:
        # Define input and output parameters
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # Load tile
        tile_in = ib.let("tile_in", block.load(input_tensor, 0, 0, 32, 32))

        # Expand scalar to tile shape
        tile_scalar = ib.let("tile_scalar", block.expands(tile_in, 1.0))

        # Add expanded scalar to tile
        tile_result = ib.let("tile_result", block.add(tile_in, tile_scalar))

        # Store result
        result = ib.let("result", block.store(tile_result, 0, 0, 32, 32, output_tensor))

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

    # Example 5: block.cast operation
    print("\n5. Block Cast Operation Example")
    print("-" * 80)
    func5 = build_block_cast_example()
    print(func5)

    # Example 6: block.zeros operation
    print("\n6. Block Zeros Operation Example")
    print("-" * 80)
    func6 = build_block_zeros_example()
    print(func6)

    # Example 7: block.minimum operation
    print("\n7. Block Minimum Operation Example")
    print("-" * 80)
    func7 = build_block_minimum_example()
    print(func7)

    # Phase 1 Examples: Core Extension OPs

    print("\n8. Block Transpose Operation Example (Phase 1)")
    print("-" * 80)
    func8 = build_block_transpose_example()
    print(func8)

    print("\n9. Block Reshape Operation Example (Phase 1)")
    print("-" * 80)
    func9 = build_block_reshape_example()
    print(func9)

    print("\n10. Block Log Operation Example (Phase 1)")
    print("-" * 80)
    func10 = build_block_log_example()
    print(func10)

    print("\n11. Block Abs Operation Example (Phase 1)")
    print("-" * 80)
    func11 = build_block_abs_example()
    print(func11)

    print("\n12. Block ReLU Operation Example (Phase 1)")
    print("-" * 80)
    func12 = build_block_relu_example()
    print(func12)

    # Phase 2 Examples: Reduction and Comparison OPs
    print("\n13. Block Row Min Operation Example (Phase 2)")
    print("-" * 80)
    func13 = build_block_row_min_example()
    print(func13)

    print("\n14. Block Row Max Operation Example (Phase 2)")
    print("-" * 80)
    func14 = build_block_row_max_example()
    print(func14)

    print("\n15. Block Col Min Operation Example (Phase 2)")
    print("-" * 80)
    func15 = build_block_col_min_example()
    print(func15)

    print("\n16. Block Col Max Operation Example (Phase 2)")
    print("-" * 80)
    func16 = build_block_col_max_example()
    print(func16)

    print("\n17. Block Row Sum Operation Example (Phase 2)")
    print("-" * 80)
    func17_sum = build_block_row_sum_example()
    print(func17_sum)

    print("\n18. Block Col Sum Operation Example (Phase 2)")
    print("-" * 80)
    func18_sum = build_block_col_sum_example()
    print(func18_sum)

    print("\n19. Block Cmp Operation Example (Phase 2)")
    print("-" * 80)
    func19 = build_block_cmp_example()
    print(func19)

    print("\n20. Block Cmps Operation Example (Phase 2)")
    print("-" * 80)
    func20 = build_block_cmps_example()
    print(func20)

    # Phase 3 Examples: Column Expand and Scalar Expand OPs
    print("\n21. Block Col Expand Operation Example (Phase 3)")
    print("-" * 80)
    func21 = build_block_col_expand_example()
    print(func21)

    print("\n22. Block Col Expand Mul Operation Example (Phase 3)")
    print("-" * 80)
    func22 = build_block_col_expand_mul_example()
    print(func22)

    print("\n23. Block Col Expand Div Operation Example (Phase 3)")
    print("-" * 80)
    func23 = build_block_col_expand_div_example()
    print(func23)

    print("\n24. Block Col Expand Sub Operation Example (Phase 3)")
    print("-" * 80)
    func24 = build_block_col_expand_sub_example()
    print(func24)

    print("\n25. Block Row Expand Add Operation Example (Phase 3)")
    print("-" * 80)
    func25 = build_block_row_expand_add_example()
    print(func25)

    print("\n26. Block Expands Operation Example (Phase 3)")
    print("-" * 80)
    func26 = build_block_expands_example()
    print(func26)

    print("\n" + "=" * 80)
    print("All examples built successfully!")
    print("=" * 80)
