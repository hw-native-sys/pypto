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

Kernels:
  fused_add_scale    — c = (a + b) * 2.0           (vector only)
  fused_add_relu     — c = relu(a + b)              (vector only)
  fused_matmul_bias  — c = matmul(a, b) + bias      (cube + vector)
  fused_linear_relu  — y = relu(matmul(x, w) + bias) (cube + vector)

Concepts introduced:
  - Scalar operations: pl.mul(tile, 2.0)
  - Activation functions: pl.relu
  - Memory spaces: pl.MemorySpace.Mat (L1), Left (L0A), Right (L0B)
  - pl.move for transferring tiles between memory spaces
  - pl.matmul for cube unit matrix multiplication
  - Multi-kernel orchestration: @pl.jit.incore helpers + pl.create_tensor for intermediate buffers

Run:  python examples/kernels/02_fused_ops.py
Next: examples/kernels/03_matmul.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def fused_add_scale(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Fused: load a, b -> add -> scale by 2.0 -> store c."""
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_sum = pl.add(tile_a, tile_b)
        tile_c = pl.mul(tile_sum, 2.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def fused_add_relu(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Fused: load a, b -> add -> relu -> store c."""
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_sum = pl.add(tile_a, tile_b)
        tile_c = pl.relu(tile_sum)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit.incore
def _matmul_kernel_64x64(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Cube InCore: compute a @ b and store to output."""
    tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
    tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
    tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
    pl.store(tile_c_l0c, [0, 0], output)
    return output


@pl.jit.incore
def _add_bias_kernel_64x64(x: pl.Tensor, bias: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Vector InCore: add bias to x and store to output."""
    tile_x = pl.load(x, [0, 0], [64, 64])
    tile_bias = pl.load(bias, [0, 0], [64, 64])
    tile_c = pl.add(tile_x, tile_bias)
    pl.store(tile_c, [0, 0], output)
    return output


@pl.jit.incore
def _add_bias_relu_kernel_64x64(x: pl.Tensor, bias: pl.Tensor, output: pl.Out[pl.Tensor]):
    """Vector InCore: fused bias add and relu activation."""
    tile_x = pl.load(x, [0, 0], [64, 64])
    tile_bias = pl.load(bias, [0, 0], [64, 64])
    tile_biased = pl.add(tile_x, tile_bias)
    tile_y = pl.relu(tile_biased)
    pl.store(tile_y, [0, 0], output)
    return output


@pl.jit
def fused_matmul_bias(a: pl.Tensor, b: pl.Tensor, bias: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Orchestrate: c = matmul(a, b) + bias"""
    mm_out = pl.create_tensor([64, 64], dtype=pl.FP32)
    mm_out = _matmul_kernel_64x64(a, b, mm_out)
    c = _add_bias_kernel_64x64(mm_out, bias, c)
    return c


@pl.jit
def fused_linear_relu(x: pl.Tensor, w: pl.Tensor, bias: pl.Tensor, y: pl.Out[pl.Tensor]):
    """Orchestrate: y = relu(matmul(x, w) + bias)"""
    mm_out = pl.create_tensor([64, 64], dtype=pl.FP32)
    mm_out = _matmul_kernel_64x64(x, w, mm_out)
    y = _add_bias_relu_kernel_64x64(mm_out, bias, y)
    return y


if __name__ == "__main__":
    cfg = RunConfig()
    torch.manual_seed(0)

    # fused_add_scale
    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    c = torch.zeros((128, 128), dtype=torch.float32)
    fused_add_scale(a, b, c, config=cfg)
    assert torch.allclose(c, (a + b) * 2.0, rtol=1e-5, atol=1e-5)

    # fused_add_relu
    c = torch.zeros((128, 128), dtype=torch.float32)
    fused_add_relu(a, b, c, config=cfg)
    assert torch.allclose(c, torch.relu(a + b), rtol=1e-5, atol=1e-5)

    # fused_matmul_bias
    a64 = torch.full((64, 64), 2.0, dtype=torch.float32)
    b64 = torch.full((64, 64), 3.0, dtype=torch.float32)
    bias = torch.randn(64, 64, dtype=torch.float32)
    c64 = torch.zeros((64, 64), dtype=torch.float32)
    fused_matmul_bias(a64, b64, bias, c64, config=cfg)
    assert torch.allclose(c64, torch.matmul(a64, b64) + bias, rtol=1e-3, atol=1e-3)

    # fused_linear_relu
    y = torch.zeros((64, 64), dtype=torch.float32)
    fused_linear_relu(a64, b64, bias, y, config=cfg)
    assert torch.allclose(y, torch.relu(torch.matmul(a64, b64) + bias), rtol=1e-3, atol=1e-3)

    print("OK")
