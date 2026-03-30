# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Fused operations: combining multiple ops in a single InCore kernel.

Programs:
  FusedAddScaleProgram   — c = (a + b) * 2.0           (vector only)
  FusedAddReluProgram    — c = relu(a + b)              (vector only)
  FusedMatmulBiasProgram — c = matmul(a, b) + bias      (cube + vector)
  FusedLinearReluProgram — y = relu(matmul(x, w) + bias) (cube + vector)

Concepts introduced:
  - Scalar operations: pl.mul(tile, 2.0)
  - Activation functions: pl.relu
  - Memory spaces: pl.MemorySpace.Mat (L1), Left (L0A), Right (L0B)
  - pl.move for transferring tiles between memory spaces
  - pl.matmul for cube unit matrix multiplication
  - Multi-kernel orchestration: pl.create_tensor for intermediate buffers

Run:  python examples/kernels/02_fused_ops.py
Next: examples/kernels/03_matmul.py
"""

import pypto.language as pl


@pl.program
class FusedAddScaleProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def fused_add_scale(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Fused: load a, b -> add -> scale by 2.0 -> store c."""
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_sum = pl.add(tile_a, tile_b)
        tile_c = pl.mul(tile_sum, 2.0)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c = self.fused_add_scale(a, b, out_c)
        return out_c


@pl.program
class FusedAddReluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def fused_add_relu(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Fused: load a, b -> add -> relu -> store c."""
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_sum = pl.add(tile_a, tile_b)
        tile_c = pl.relu(tile_sum)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c = self.fused_add_relu(a, b, out_c)
        return out_c


@pl.program
class FusedMatmulBiasProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Cube InCore: compute a @ b and store to output."""
        tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.store(tile_c_l0c, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def add_bias_kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        bias: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: add bias to x and store to output."""
        tile_x = pl.load(x, [0, 0], [64, 64])
        tile_bias = pl.load(bias, [0, 0], [64, 64])
        tile_c = pl.add(tile_x, tile_bias)
        out = pl.store(tile_c, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        bias: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Orchestrate: c = matmul(a, b) + bias"""
        mm_out = pl.create_tensor([64, 64], dtype=pl.FP32)
        mm_out = self.matmul_kernel(a, b, mm_out)
        c = self.add_bias_kernel(mm_out, bias, c)
        return c


@pl.program
class FusedLinearReluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        w: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Cube InCore: compute x @ w and store to output."""
        tile_x_l1 = pl.load(x, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_w_l1 = pl.load(w, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_x_l0a = pl.move(tile_x_l1, target_memory=pl.MemorySpace.Left)
        tile_w_l0b = pl.move(tile_w_l1, target_memory=pl.MemorySpace.Right)
        tile_out_l0c = pl.matmul(tile_x_l0a, tile_w_l0b)
        out = pl.store(tile_out_l0c, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def add_bias_relu_kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        bias: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: fused bias add and relu activation."""
        tile_x = pl.load(x, [0, 0], [64, 64])
        tile_bias = pl.load(bias, [0, 0], [64, 64])
        tile_biased = pl.add(tile_x, tile_bias)
        tile_y = pl.relu(tile_biased)
        out = pl.store(tile_y, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        w: pl.Tensor[[64, 64], pl.FP32],
        bias: pl.Tensor[[64, 64], pl.FP32],
        y: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Orchestrate: y = relu(matmul(x, w) + bias)"""
        mm_out = pl.create_tensor([64, 64], dtype=pl.FP32)
        mm_out = self.matmul_kernel(x, w, mm_out)
        y = self.add_bias_relu_kernel(mm_out, bias, y)
        return y


if __name__ == "__main__":
    for name, prog in [
        ("FusedAddScale", FusedAddScaleProgram),
        ("FusedAddRelu", FusedAddReluProgram),
        ("FusedMatmulBias", FusedMatmulBiasProgram),
        ("FusedLinearRelu", FusedLinearReluProgram),
    ]:
        print(f"=== {name} ===")
        print(prog.as_python())
        print()
