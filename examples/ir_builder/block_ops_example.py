# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
示例：使用 IRBuilder 构造 block operations

该示例展示如何使用 pypto.ir.op.block 模块中的 block operations，
包括内存操作、逐元素操作、一元操作和归约操作。
"""

from pypto.ir.builder import IRBuilder
from pypto.ir.op import block
from pypto.pypto_core import DataType, ir


def build_block_elementwise_example():
    """构建一个使用 block 逐元素操作的示例函数。

    该函数演示：
    1. 从 tensor 复制数据到 unified buffer (tile)
    2. 执行逐元素操作（加法、乘法、除法）
    3. 将结果复制回 tensor
    """
    ib = IRBuilder()

    with ib.function("block_elementwise_example") as f:
        # 定义输入和输出参数
        input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
        output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # 定义 tile 大小和偏移量
        tile_height = 32
        tile_width = 32
        row_offset = 0
        col_offset = 0

        # 从 tensor 复制数据到 unified buffer
        tile_a = ib.let("tile_a", block.ub_copy_in(input_a, row_offset, col_offset, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.ub_copy_in(input_b, row_offset, col_offset, tile_height, tile_width))

        # 执行逐元素操作
        # tile_c = (tile_a + tile_b) * 2.0
        tile_sum = ib.let("tile_sum", block.add(tile_a, tile_b))
        tile_c = ib.let("tile_c", block.mul(tile_sum, 2.0))

        # 将结果复制回 tensor
        result = ib.let(
            "result", block.ub_copy_out(tile_c, row_offset, col_offset, tile_height, tile_width, output)
        )

        # 返回结果
        ib.return_stmt(result)

    return f.get_result()


def build_block_reduction_example():
    """构建一个使用 block 归约操作的示例函数。

    该函数演示：
    1. 从 tensor 复制数据到 tile
    2. 执行归约操作（sum）
    3. 将归约结果复制回 tensor
    """
    ib = IRBuilder()

    with ib.function("block_reduction_example") as f:
        # 定义输入和输出参数
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 1], DataType.FP32))
        f.return_type(ir.TensorType([128, 1], DataType.FP32))

        # 定义 tile 大小
        tile_height = 32
        tile_width = 128

        # 定义偏移量
        row_offset = 0
        col_offset = 0

        # 从 tensor 复制数据到 tile
        tile_in = ib.let(
            "tile_in", block.ub_copy_in(input_tensor, row_offset, col_offset, tile_height, tile_width)
        )

        # 沿着最后一个轴（axis=1）进行归约求和
        tile_sum = ib.let("tile_sum", block.sum(tile_in, axis=1, keepdim=True))

        # 将归约结果复制回 tensor
        result = ib.let("result", block.ub_copy_out(tile_sum, row_offset, 0, tile_height, 1, output_tensor))

        # 返回结果
        ib.return_stmt(result)

    return f.get_result()


def build_block_unary_example():
    """构建一个使用 block 一元操作的示例函数。

    该函数演示：
    1. 从 tensor 复制数据到 tile
    2. 执行一元操作（sqrt）
    3. 将结果复制回 tensor
    """
    ib = IRBuilder()

    with ib.function("block_unary_example") as f:
        # 定义输入和输出参数
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TensorType([128, 128], DataType.FP32))

        # 定义 tile 大小
        tile_height = 32
        tile_width = 32

        # 定义偏移量
        row_offset = 0
        col_offset = 0

        # 从 tensor 复制数据到 tile
        tile_in = ib.let(
            "tile_in", block.ub_copy_in(input_tensor, row_offset, col_offset, tile_height, tile_width)
        )

        # 执行一元操作: sqrt
        tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_in))

        # 将结果复制回 tensor
        result = ib.let(
            "result",
            block.ub_copy_out(tile_sqrt, row_offset, col_offset, tile_height, tile_width, output_tensor),
        )

        # 返回结果
        ib.return_stmt(result)

    return f.get_result()


def build_complex_block_computation():
    """构建一个复杂的 block 计算示例。

    该函数演示组合使用多种 block 操作：
    - 内存操作
    - 逐元素操作
    - 一元操作
    - 归约操作

    计算：output = sum(sqrt(a * b + c), axis=1)
    """
    ib = IRBuilder()

    with ib.function("complex_block_computation") as f:
        # 定义输入和输出参数
        input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
        input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
        input_c = f.param("input_c", ir.TensorType([128, 128], DataType.FP32))
        output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
        f.return_type(ir.TensorType([128, 1], DataType.FP32))

        # 定义 tile 大小
        tile_height = 32
        tile_width = 128

        # 定义偏移量
        row_offset = 0
        col_offset = 0

        # 从 tensor 复制数据到 unified buffer
        tile_a = ib.let("tile_a", block.ub_copy_in(input_a, row_offset, col_offset, tile_height, tile_width))
        tile_b = ib.let("tile_b", block.ub_copy_in(input_b, row_offset, col_offset, tile_height, tile_width))
        tile_c = ib.let("tile_c", block.ub_copy_in(input_c, row_offset, col_offset, tile_height, tile_width))

        # 执行计算: a * b + c
        tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
        tile_add = ib.let("tile_add", block.add(tile_mul, tile_c))

        # 执行一元操作: sqrt
        tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_add))

        # 执行归约: sum(axis=1)
        tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))

        # 将结果复制回 tensor
        result = ib.let("result", block.ub_copy_out(tile_sum, row_offset, 0, tile_height, 1, output))

        # 返回结果
        ib.return_stmt(result)

    return f.get_result()


if __name__ == "__main__":
    print("=" * 80)
    print("Block Operations 示例")
    print("=" * 80)

    # 示例1: 逐元素操作
    print("\n1. Block 逐元素操作示例")
    print("-" * 80)
    func1 = build_block_elementwise_example()
    print(func1)

    # 示例2: 归约操作
    print("\n2. Block 归约操作示例")
    print("-" * 80)
    func2 = build_block_reduction_example()
    print(func2)

    # 示例3: 一元操作
    print("\n3. Block 一元操作示例")
    print("-" * 80)
    func3 = build_block_unary_example()
    print(func3)

    # 示例4: 复杂的 block 计算
    print("\n4. 复杂的 Block 计算示例")
    print("-" * 80)
    func4 = build_complex_block_computation()
    print(func4)

    print("\n" + "=" * 80)
    print("所有示例构建完成！")
    print("=" * 80)
