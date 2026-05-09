# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Normalization layers: RMSNorm and LayerNorm (32x64 input).

Kernels:
  rms_norm   -- output = x / sqrt(mean(x^2) + eps) * gamma
  layer_norm -- output = (x - mean) / sqrt(var + eps) * gamma + beta

Concepts introduced:
  - pl.reshape for transposing [32,1] -> [1,32] (ColMajor -> RowMajor workaround)
  - pl.row_sum for row-wise reduction
  - pl.row_expand_sub, pl.row_expand_div for broadcasting row vectors
  - pl.col_expand_mul, pl.col_expand for broadcasting column vectors
  - pl.sqrt for square root

Run:  python examples/kernels/07_normalization.py
Next: examples/kernels/08_assemble.py
"""

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


@pl.jit
def rms_norm(x: pl.Tensor, gamma: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.incore():
        tile_x = pl.load(x, [0, 0], [32, 64])
        tile_gamma = pl.load(gamma, [0, 0], [1, 64])

        # [32, 1] tiles are ColMajor; scalar ops (mul/add) need RowMajor.
        # Workaround: reshape [32, 1] -> [1, 32], apply op, reshape back.

        # squared = x * x
        squared = pl.mul(tile_x, tile_x)

        # mean_sq = sum(x^2, dim=-1, keepdim=True) / hidden_size
        tmp = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        mean_sq: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(squared, tmp)
        mean_sq_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean_sq, [1, 32])
        mean_sq_T = pl.mul(mean_sq_T, 0.015625)  # 1.0 / 64
        mean_sq = pl.reshape(mean_sq_T, [32, 1])

        # rms = sqrt(mean_sq + eps)
        mean_sq_T = pl.reshape(mean_sq, [1, 32])
        rms_T = pl.add(mean_sq_T, 1e-5)
        rms_T = pl.sqrt(rms_T)
        rms = pl.reshape(rms_T, [32, 1])

        # normalized = x / rms (broadcast rms across hidden dim)
        normalized = pl.row_expand_div(tile_x, rms)

        # result = normalized * gamma (broadcast gamma across batch)
        result = pl.col_expand_mul(normalized, tile_gamma)

        pl.store(result, [0, 0], output)
    return output


@pl.jit
def layer_norm(
    x: pl.Tensor,
    gamma: pl.Tensor,
    beta: pl.Tensor,
    output: pl.Out[pl.Tensor],
):
    with pl.incore():
        tile_x = pl.load(x, [0, 0], [32, 64])
        tile_gamma = pl.load(gamma, [0, 0], [1, 64])
        tile_beta = pl.load(beta, [0, 0], [1, 64])

        # mean = sum(x, dim=-1, keepdim=True) / hidden_size
        tmp = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        mean: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_x, tmp)
        mean_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean, [1, 32])
        mean_T = pl.mul(mean_T, 0.015625)  # 1.0 / 64
        mean = pl.reshape(mean_T, [32, 1])

        # centered = x - mean (broadcast mean across hidden dim)
        centered = pl.row_expand_sub(tile_x, mean)

        # var = sum(centered^2, dim=-1, keepdim=True) / hidden_size
        squared = pl.mul(centered, centered)
        tmp2 = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        var: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(squared, tmp2)
        var_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(var, [1, 32])
        var_T = pl.mul(var_T, 0.015625)  # 1.0 / 64
        var = pl.reshape(var_T, [32, 1])

        # std = sqrt(var + eps)
        var_T = pl.reshape(var, [1, 32])
        var_eps_T = pl.add(var_T, 1e-5)
        std_T = pl.sqrt(var_eps_T)
        std = pl.reshape(std_T, [32, 1])

        # normalized = centered / std (broadcast std across hidden dim)
        normalized = pl.row_expand_div(centered, std)

        # scaled = normalized * gamma (broadcast gamma across batch)
        scaled = pl.col_expand_mul(normalized, tile_gamma)

        # result = scaled + beta (broadcast beta across batch)
        beta_full = pl.col_expand(scaled, tile_beta)
        result = pl.add(scaled, beta_full)

        pl.store(result, [0, 0], output)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    config = RunConfig()
    eps = 1e-5
    hidden_size = 64

    # RMSNorm
    x = torch.randn(32, 64, dtype=torch.float32)
    gamma = torch.randn(1, 64, dtype=torch.float32)
    out = torch.zeros_like(x)
    rms_norm(x, gamma, out, config=config)
    mean_sq = (x**2).sum(dim=-1, keepdim=True) / hidden_size
    rms_ref = torch.sqrt(mean_sq + eps)
    expected = (x / rms_ref) * gamma
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
        f"rms_norm failed: max diff = {(out - expected).abs().max().item()}"
    )

    # LayerNorm
    x = torch.randn(32, 64, dtype=torch.float32)
    gamma = torch.randn(1, 64, dtype=torch.float32)
    beta = torch.randn(1, 64, dtype=torch.float32)
    out = torch.zeros_like(x)
    layer_norm(x, gamma, beta, out, config=config)
    mean = x.sum(dim=-1, keepdim=True) / hidden_size
    centered = x - mean
    var = (centered**2).sum(dim=-1, keepdim=True) / hidden_size
    std_ref = torch.sqrt(var + eps)
    expected = (centered / std_ref) * gamma + beta
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
        f"layer_norm failed: max diff = {(out - expected).abs().max().item()}"
    )

    print("OK")
