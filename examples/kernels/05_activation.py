# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Activation functions (32x128 tiles).

Kernels:
  silu    -- SiLU:   output = x * sigmoid(x)                    = x / (1 + exp(-x))
  gelu    -- GELU:   output = x * sigmoid(1.702 * x)            (fast approximation)
  swiglu  -- SwiGLU: output = gate * sigmoid(gate) * up
  geglu   -- GeGLU:  output = gate * sigmoid(1.702 * gate) * up

Concepts introduced:
  - pl.exp, pl.recip for building sigmoid from primitives
  - Chaining element-wise ops for complex activation functions
  - Two-input activations (SwiGLU, GeGLU) with gate and up projections

Run:  python examples/kernels/05_activation.py
Next: examples/kernels/06_softmax.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def silu(x: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.incore():
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
    with pl.incore():
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
    with pl.incore():
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
    with pl.incore():
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
