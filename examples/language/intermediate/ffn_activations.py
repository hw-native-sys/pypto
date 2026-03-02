# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
FFN module programs using PyPTO language DSL.

Each program implements a full FFN forward pass (gate projection → activation → down
projection) in a single 64x64 tile:

  FFNGeluProgram   — output = GELU(hidden_states @ gate_proj_weight) @ down_proj_weight
  FFNSwigluProgram — output = SwiGLU(gate, up) @ down_proj_weight
                     where gate = hidden_states @ gate_proj_weight
                           up   = hidden_states @ up_proj_weight
  FFNReluProgram   — output = ReLU(hidden_states @ gate_proj_weight) @ down_proj_weight
"""

import pypto.language as pl


@pl.program
class FFNGeluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Cube InCore: compute a @ b and store result to GM."""
        tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.l0c_store(tile_c_l0c, offsets=[0, 0], shapes=[64, 64], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def gelu_kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: apply GELU activation — x * sigmoid(1.702 * x)."""
        tile_x: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        x_scaled: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_x, 1.702)  # type: ignore[reportArgumentType]
        x_neg: pl.Tile[[64, 64], pl.FP32] = pl.mul(x_scaled, -1.0)  # type: ignore[reportArgumentType]
        exp_neg: pl.Tile[[64, 64], pl.FP32] = pl.exp(x_neg)
        denom: pl.Tile[[64, 64], pl.FP32] = pl.add(exp_neg, 1.0)  # type: ignore[reportArgumentType]
        sigmoid: pl.Tile[[64, 64], pl.FP32] = pl.recip(denom)
        result: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_x, sigmoid)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def ffn_gelu_orch(
        self,
        hidden_states: pl.Tensor[[64, 64], pl.FP32],
        gate_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        down_proj_weight: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # gate = hidden_states @ gate_proj_weight
        gate: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        gate = self.matmul_kernel(hidden_states, gate_proj_weight, gate)
        # activated = GELU(gate)
        activated: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        activated = self.gelu_kernel(gate, activated)
        # output = activated @ down_proj_weight
        output: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        output = self.matmul_kernel(activated, down_proj_weight, output)
        return output


@pl.program
class FFNSwigluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Cube InCore: compute a @ b and store result to GM."""
        tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.l0c_store(tile_c_l0c, offsets=[0, 0], shapes=[64, 64], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def swiglu_kernel(
        self,
        gate: pl.Tensor[[64, 64], pl.FP32],
        up: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: apply SwiGLU activation — gate * sigmoid(gate) * up."""
        tile_gate: pl.Tile[[64, 64], pl.FP32] = pl.load(gate, [0, 0], [64, 64])
        tile_up: pl.Tile[[64, 64], pl.FP32] = pl.load(up, [0, 0], [64, 64])
        gate_neg: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_gate, -1.0)  # type: ignore[reportArgumentType]
        exp_neg: pl.Tile[[64, 64], pl.FP32] = pl.exp(gate_neg)
        denom: pl.Tile[[64, 64], pl.FP32] = pl.add(exp_neg, 1.0)  # type: ignore[reportArgumentType]
        sigmoid: pl.Tile[[64, 64], pl.FP32] = pl.recip(denom)
        swish: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_gate, sigmoid)
        result: pl.Tile[[64, 64], pl.FP32] = pl.mul(swish, tile_up)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def ffn_swiglu_orch(
        self,
        hidden_states: pl.Tensor[[64, 64], pl.FP32],
        gate_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        up_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        down_proj_weight: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # gate = hidden_states @ gate_proj_weight
        gate: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        gate = self.matmul_kernel(hidden_states, gate_proj_weight, gate)
        # up = hidden_states @ up_proj_weight
        up: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        up = self.matmul_kernel(hidden_states, up_proj_weight, up)
        # activated = SwiGLU(gate, up)
        activated: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        activated = self.swiglu_kernel(gate, up, activated)
        # output = activated @ down_proj_weight
        output: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        output = self.matmul_kernel(activated, down_proj_weight, output)
        return output


@pl.program
class FFNReluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Cube InCore: compute a @ b and store result to GM."""
        tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.l0c_store(tile_c_l0c, offsets=[0, 0], shapes=[64, 64], output_tensor=output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def relu_kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: apply ReLU activation — max(0, x)."""
        tile_x: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        result: pl.Tile[[64, 64], pl.FP32] = pl.relu(tile_x)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def ffn_relu_orch(
        self,
        hidden_states: pl.Tensor[[64, 64], pl.FP32],
        gate_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        down_proj_weight: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # gate = hidden_states @ gate_proj_weight
        gate: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        gate = self.matmul_kernel(hidden_states, gate_proj_weight, gate)
        # activated = ReLU(gate)
        activated: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        activated = self.relu_kernel(gate, activated)
        # output = activated @ down_proj_weight
        output: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        output = self.matmul_kernel(activated, down_proj_weight, output)
        return output
