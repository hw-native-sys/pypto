# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Activation functions.

Kernels:
  fused_add_relu -- relu(a + b)                                (128x128, vector only)
  silu           -- SiLU:   output = x * sigmoid(x)             = x / (1 + exp(-x))  (32x128)
  gelu           -- GELU:   output = x * sigmoid(1.702 * x)     (fast approximation) (32x128)
  swiglu         -- SwiGLU: output = gate * sigmoid(gate) * up                       (32x128)
  geglu          -- GeGLU:  output = gate * sigmoid(1.702 * gate) * up               (32x128)

Concepts introduced:
  - Activation functions: pl.relu
  - pl.exp, pl.recip for building sigmoid from primitives
  - Chaining element-wise ops for complex activation functions
  - Two-input activations (SwiGLU, GeGLU) with gate and up projections

Run:  python examples/beginner/04_activation.py
Next: examples/beginner/05_matmul.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def fused_add_relu(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Fused: load a, b -> add -> relu -> store c."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_sum = pl.add(tile_a, tile_b)
        tile_c = pl.relu(tile_sum)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def silu(x: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        tile_x = pl.load(x, [0, 0], [32, 128])
        x_neg = pl.mul(tile_x, -1.0)
        exp_neg = pl.exp(x_neg)
        denom = pl.add(exp_neg, 1.0)
        sigmoid = pl.recip(denom)
        result = pl.mul(tile_x, sigmoid)
        pl.store(result, [0, 0], output)
    return output


@pl.jit
def gelu(x: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        # GELU(x) = x * sigmoid(1.702 * x)  (fast approximation)
        tile_x = pl.load(x, [0, 0], [32, 128])
        x_scaled = pl.mul(tile_x, 1.702)
        x_neg = pl.mul(x_scaled, -1.0)
        exp_neg = pl.exp(x_neg)
        denom = pl.add(exp_neg, 1.0)
        sigmoid = pl.recip(denom)
        result = pl.mul(tile_x, sigmoid)
        pl.store(result, [0, 0], output)
    return output


@pl.jit
def swiglu(gate: pl.Tensor, up: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        # SwiGLU(gate, up) = Swish(gate) * up = gate * sigmoid(gate) * up
        tile_gate = pl.load(gate, [0, 0], [32, 128])
        tile_up = pl.load(up, [0, 0], [32, 128])
        gate_neg = pl.mul(tile_gate, -1.0)
        exp_neg = pl.exp(gate_neg)
        denom = pl.add(exp_neg, 1.0)
        sigmoid = pl.recip(denom)
        swish = pl.mul(tile_gate, sigmoid)
        result = pl.mul(swish, tile_up)
        pl.store(result, [0, 0], output)
    return output


@pl.jit
def geglu(gate: pl.Tensor, up: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        # GeGLU(gate, up) = GELU(gate) * up
        # GELU approximation: gate * sigmoid(1.702 * gate)
        tile_gate = pl.load(gate, [0, 0], [32, 128])
        tile_up = pl.load(up, [0, 0], [32, 128])
        gate_scaled = pl.mul(tile_gate, 1.702)
        gate_neg = pl.mul(gate_scaled, -1.0)
        exp_neg = pl.exp(gate_neg)
        denom = pl.add(exp_neg, 1.0)
        sigmoid = pl.recip(denom)
        gelu_gate = pl.mul(tile_gate, sigmoid)
        result = pl.mul(gelu_gate, tile_up)
        pl.store(result, [0, 0], output)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    config = RunConfig()

    # fused_add_relu
    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    c = torch.zeros((128, 128), dtype=torch.float32)
    fused_add_relu(a, b, c, config=config)
    assert torch.allclose(c, torch.relu(a + b), rtol=1e-5, atol=1e-5)

    # SiLU
    x = torch.randn(32, 128, dtype=torch.float32)
    out = torch.zeros_like(x)
    silu(x, out, config=config)
    expected = x * torch.sigmoid(x)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
        f"silu failed: max diff = {(out - expected).abs().max().item()}"
    )

    # GELU
    x = torch.randn(32, 128, dtype=torch.float32)
    out = torch.zeros_like(x)
    gelu(x, out, config=config)
    expected = x * torch.sigmoid(1.702 * x)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
        f"gelu failed: max diff = {(out - expected).abs().max().item()}"
    )

    # SwiGLU
    gate = torch.randn(32, 128, dtype=torch.float32)
    up = torch.randn(32, 128, dtype=torch.float32)
    out = torch.zeros_like(gate)
    swiglu(gate, up, out, config=config)
    expected = gate * torch.sigmoid(gate) * up
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
        f"swiglu failed: max diff = {(out - expected).abs().max().item()}"
    )

    # GeGLU
    gate = torch.randn(32, 128, dtype=torch.float32)
    up = torch.randn(32, 128, dtype=torch.float32)
    out = torch.zeros_like(gate)
    geglu(gate, up, out, config=config)
    expected = gate * torch.sigmoid(1.702 * gate) * up
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
        f"geglu failed: max diff = {(out - expected).abs().max().item()}"
    )

    print("OK")
