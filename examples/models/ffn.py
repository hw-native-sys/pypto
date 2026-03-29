# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
FFN module programs (64x64 tiles).

Each program implements a full FFN forward pass (gate projection -> activation ->
down projection):

  FFNGeluProgram   -- output = GELU(hidden_states @ gate_proj_weight) @ down_proj_weight
  FFNSwigluProgram -- output = SwiGLU(gate, up) @ down_proj_weight
  FFNReluProgram   -- output = ReLU(hidden_states @ gate_proj_weight) @ down_proj_weight

Concepts introduced:
  - Module-level @pl.function: shared kernel reused across multiple programs
  - Multi-kernel orchestration: matmul -> activation -> matmul pipeline
  - Direct call to module-level kernels (no self. prefix)

Run:  python examples/models/ffn.py
Next: examples/models/vector_dag.py
"""

import pypto.language as pl

# ── Shared cube matmul kernel (module-level, reusable across programs) ────────


@pl.function(type=pl.FunctionType.InCore)
def matmul_kernel(
    a: pl.Tensor[[64, 64], pl.FP32],
    b: pl.Tensor[[64, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
) -> pl.Tensor[[64, 64], pl.FP32]:
    """Cube InCore: compute a @ b and store result to GM."""
    tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
    tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
    tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
    out = pl.store(tile_c_l0c, [0, 0], output)
    return out


# ── FFN with GELU activation ─────────────────────────────────────────────────


@pl.program
class FFNGeluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def gelu_kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: apply GELU activation -- x * sigmoid(1.702 * x)."""
        tile_x = pl.load(x, [0, 0], [64, 64])
        x_scaled = pl.mul(tile_x, 1.702)
        x_neg = pl.mul(x_scaled, -1.0)
        exp_neg = pl.exp(x_neg)
        denom = pl.add(exp_neg, 1.0)
        sigmoid = pl.recip(denom)
        result = pl.mul(tile_x, sigmoid)
        out = pl.store(result, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def ffn_gelu_orch(
        self,
        hidden_states: pl.Tensor[[64, 64], pl.FP32],
        gate_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        down_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # gate = hidden_states @ gate_proj_weight
        gate = pl.create_tensor([64, 64], dtype=pl.FP32)
        gate = matmul_kernel(hidden_states, gate_proj_weight, gate)
        # activated = GELU(gate)
        activated = pl.create_tensor([64, 64], dtype=pl.FP32)
        activated = self.gelu_kernel(gate, activated)
        # output = activated @ down_proj_weight
        output = matmul_kernel(activated, down_proj_weight, output)
        return output


# ── FFN with SwiGLU activation ───────────────────────────────────────────────


@pl.program
class FFNSwigluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def swiglu_kernel(
        self,
        gate: pl.Tensor[[64, 64], pl.FP32],
        up: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: apply SwiGLU activation -- gate * sigmoid(gate) * up."""
        tile_gate = pl.load(gate, [0, 0], [64, 64])
        tile_up = pl.load(up, [0, 0], [64, 64])
        gate_neg = pl.mul(tile_gate, -1.0)
        exp_neg = pl.exp(gate_neg)
        denom = pl.add(exp_neg, 1.0)
        sigmoid = pl.recip(denom)
        swish = pl.mul(tile_gate, sigmoid)
        result = pl.mul(swish, tile_up)
        out = pl.store(result, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def ffn_swiglu_orch(
        self,
        hidden_states: pl.Tensor[[64, 64], pl.FP32],
        gate_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        up_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        down_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # gate = hidden_states @ gate_proj_weight
        gate = pl.create_tensor([64, 64], dtype=pl.FP32)
        gate = matmul_kernel(hidden_states, gate_proj_weight, gate)
        # up = hidden_states @ up_proj_weight
        up = pl.create_tensor([64, 64], dtype=pl.FP32)
        up = matmul_kernel(hidden_states, up_proj_weight, up)
        # activated = SwiGLU(gate, up)
        activated = pl.create_tensor([64, 64], dtype=pl.FP32)
        activated = self.swiglu_kernel(gate, up, activated)
        # output = activated @ down_proj_weight
        output = matmul_kernel(activated, down_proj_weight, output)
        return output


# ── FFN with ReLU activation ─────────────────────────────────────────────────


@pl.program
class FFNReluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def relu_kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: apply ReLU activation -- max(0, x)."""
        tile_x = pl.load(x, [0, 0], [64, 64])
        result = pl.relu(tile_x)
        out = pl.store(result, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def ffn_relu_orch(
        self,
        hidden_states: pl.Tensor[[64, 64], pl.FP32],
        gate_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        down_proj_weight: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # gate = hidden_states @ gate_proj_weight
        gate = pl.create_tensor([64, 64], dtype=pl.FP32)
        gate = matmul_kernel(hidden_states, gate_proj_weight, gate)
        # activated = ReLU(gate)
        activated = pl.create_tensor([64, 64], dtype=pl.FP32)
        activated = self.relu_kernel(gate, activated)
        # output = activated @ down_proj_weight
        output = matmul_kernel(activated, down_proj_weight, output)
        return output


if __name__ == "__main__":
    for name, prog in [
        ("FFNGelu", FFNGeluProgram),
        ("FFNSwiglu", FFNSwigluProgram),
        ("FFNRelu", FFNReluProgram),
    ]:
        print(f"=== {name} ===")
        print(prog.as_python())
        print()
