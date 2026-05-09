# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
FFN module JIT entries (64x64 tiles).

Each entry implements a full FFN forward pass (gate projection -> activation ->
down projection):

  ffn_gelu   -- output = GELU(hidden_states @ gate_proj_weight) @ down_proj_weight
  ffn_swiglu -- output = SwiGLU(gate, up) @ down_proj_weight
  ffn_relu   -- output = ReLU(hidden_states @ gate_proj_weight) @ down_proj_weight

Concepts introduced:
  - Module-level @pl.jit.incore: shared kernel reused across multiple JIT entries
  - Multi-kernel orchestration: matmul -> activation -> matmul pipeline
  - Direct call to module-level kernels (no self. prefix)

Run:  python examples/models/01_ffn.py
Next: examples/models/02_vector_dag.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

# ── Shared cube matmul kernel (module-level, reusable across entries) ────────


@pl.jit.incore
def matmul_kernel(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Cube InCore: compute a @ b and store result to GM."""
    tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
    tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
    tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
    pl.store(tile_c_l0c, [0, 0], output)
    return output


# ── Activation kernels (module-level @pl.jit.incore) ─────────────────────────


@pl.jit.incore
def gelu_kernel(x: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Vector InCore: apply GELU activation -- x * sigmoid(1.702 * x)."""
    tile_x = pl.load(x, [0, 0], [64, 64])
    x_scaled = pl.mul(tile_x, 1.702)
    x_neg = pl.mul(x_scaled, -1.0)
    exp_neg = pl.exp(x_neg)
    denom = pl.add(exp_neg, 1.0)
    sigmoid = pl.recip(denom)
    result = pl.mul(tile_x, sigmoid)
    pl.store(result, [0, 0], output)
    return output


@pl.jit.incore
def swiglu_kernel(gate: pl.Tensor, up: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Vector InCore: apply SwiGLU activation -- gate * sigmoid(gate) * up."""
    tile_gate = pl.load(gate, [0, 0], [64, 64])
    tile_up = pl.load(up, [0, 0], [64, 64])
    gate_neg = pl.mul(tile_gate, -1.0)
    exp_neg = pl.exp(gate_neg)
    denom = pl.add(exp_neg, 1.0)
    sigmoid = pl.recip(denom)
    swish = pl.mul(tile_gate, sigmoid)
    result = pl.mul(swish, tile_up)
    pl.store(result, [0, 0], output)
    return output


@pl.jit.incore
def relu_kernel(x: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Vector InCore: apply ReLU activation -- max(0, x)."""
    tile_x = pl.load(x, [0, 0], [64, 64])
    result = pl.relu(tile_x)
    pl.store(result, [0, 0], output)
    return output


# ── FFN orchestration entries (@pl.jit) ───────────────────────────────────────


@pl.jit
def ffn_gelu(
    hidden_states: pl.Tensor,
    gate_proj_weight: pl.Tensor,
    down_proj_weight: pl.Tensor,
    output: pl.Out[pl.Tensor],
):
    """FFN with GELU activation."""
    # gate = hidden_states @ gate_proj_weight
    gate = pl.create_tensor([64, 64], dtype=pl.FP32)
    gate = matmul_kernel(hidden_states, gate_proj_weight, gate)
    # activated = GELU(gate)
    activated = pl.create_tensor([64, 64], dtype=pl.FP32)
    activated = gelu_kernel(gate, activated)
    # output = activated @ down_proj_weight
    output = matmul_kernel(activated, down_proj_weight, output)
    return output


@pl.jit
def ffn_swiglu(
    hidden_states: pl.Tensor,
    gate_proj_weight: pl.Tensor,
    up_proj_weight: pl.Tensor,
    down_proj_weight: pl.Tensor,
    output: pl.Out[pl.Tensor],
):
    """FFN with SwiGLU activation."""
    # gate = hidden_states @ gate_proj_weight
    gate = pl.create_tensor([64, 64], dtype=pl.FP32)
    gate = matmul_kernel(hidden_states, gate_proj_weight, gate)
    # up = hidden_states @ up_proj_weight
    up = pl.create_tensor([64, 64], dtype=pl.FP32)
    up = matmul_kernel(hidden_states, up_proj_weight, up)
    # activated = SwiGLU(gate, up)
    activated = pl.create_tensor([64, 64], dtype=pl.FP32)
    activated = swiglu_kernel(gate, up, activated)
    # output = activated @ down_proj_weight
    output = matmul_kernel(activated, down_proj_weight, output)
    return output


@pl.jit
def ffn_relu(
    hidden_states: pl.Tensor,
    gate_proj_weight: pl.Tensor,
    down_proj_weight: pl.Tensor,
    output: pl.Out[pl.Tensor],
):
    """FFN with ReLU activation."""
    # gate = hidden_states @ gate_proj_weight
    gate = pl.create_tensor([64, 64], dtype=pl.FP32)
    gate = matmul_kernel(hidden_states, gate_proj_weight, gate)
    # activated = ReLU(gate)
    activated = pl.create_tensor([64, 64], dtype=pl.FP32)
    activated = relu_kernel(gate, activated)
    # output = activated @ down_proj_weight
    output = matmul_kernel(activated, down_proj_weight, output)
    return output


if __name__ == "__main__":
    cfg = RunConfig()
    torch.manual_seed(0)

    hidden_states = torch.randn(64, 64, dtype=torch.float32)
    gate_proj_weight = torch.randn(64, 64, dtype=torch.float32)
    up_proj_weight = torch.randn(64, 64, dtype=torch.float32)
    down_proj_weight = torch.randn(64, 64, dtype=torch.float32)

    # FFN + GELU: GELU(hidden @ gate_proj) @ down_proj, GELU = x * sigmoid(1.702 * x)
    output = torch.zeros(64, 64, dtype=torch.float32)
    ffn_gelu(hidden_states, gate_proj_weight, down_proj_weight, output, config=cfg)
    gate = hidden_states @ gate_proj_weight
    expected_gelu = (gate * torch.sigmoid(1.702 * gate)) @ down_proj_weight
    assert torch.allclose(output, expected_gelu, rtol=3e-3, atol=3e-3), (
        f"ffn_gelu failed: max diff = {(output - expected_gelu).abs().max().item()}"
    )

    # FFN + SwiGLU: SwiGLU(gate, up) @ down_proj, SwiGLU = gate * sigmoid(gate) * up
    output = torch.zeros(64, 64, dtype=torch.float32)
    ffn_swiglu(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight, output, config=cfg)
    gate = hidden_states @ gate_proj_weight
    up = hidden_states @ up_proj_weight
    expected_swiglu = (gate * torch.sigmoid(gate) * up) @ down_proj_weight
    assert torch.allclose(output, expected_swiglu, rtol=3e-3, atol=3e-3), (
        f"ffn_swiglu failed: max diff = {(output - expected_swiglu).abs().max().item()}"
    )

    # FFN + ReLU: ReLU(hidden @ gate_proj) @ down_proj
    output = torch.zeros(64, 64, dtype=torch.float32)
    ffn_relu(hidden_states, gate_proj_weight, down_proj_weight, output, config=cfg)
    gate = hidden_states @ gate_proj_weight
    expected_relu = torch.relu(gate) @ down_proj_weight
    assert torch.allclose(output, expected_relu, rtol=3e-3, atol=3e-3), (
        f"ffn_relu failed: max diff = {(output - expected_relu).abs().max().item()}"
    )

    print("OK")
