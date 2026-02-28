# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Layer normalization using PyPTO language DSL.

Formula: output = (x - mean) / sqrt(var + eps) * gamma + beta

Computed on 32x64 input, normalizing across the hidden dimension (64).
"""

import pypto.language as pl


@pl.program
class LayerNormProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_layer_norm(
        self,
        x: pl.Tensor[[32, 64], pl.FP32],
        gamma: pl.Tensor[[1, 64], pl.FP32],
        beta: pl.Tensor[[1, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
    ) -> pl.Tensor[[32, 64], pl.FP32]:
        tile_x: pl.Tile[[32, 64], pl.FP32] = pl.load(x, [0, 0], [32, 64])
        tile_gamma: pl.Tile[[1, 64], pl.FP32] = pl.load(gamma, [0, 0], [1, 64])
        tile_beta: pl.Tile[[1, 64], pl.FP32] = pl.load(beta, [0, 0], [1, 64])

        # [32, 1] tiles are ColMajor; scalar ops (mul/add) need RowMajor.
        # Workaround: reshape [32, 1] -> [1, 32], apply op, reshape back.

        # mean = sum(x, dim=-1, keepdim=True) / hidden_size
        tmp: pl.Tile[[32, 64], pl.FP32] = pl.create_tile(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        mean: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_x, tmp)
        mean_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean, [1, 32])
        mean_T = pl.mul(mean_T, 0.015625)  # 1.0 / 64  # type: ignore[reportArgumentType]
        mean = pl.reshape(mean_T, [32, 1])

        # centered = x - mean (broadcast mean across hidden dim)
        centered: pl.Tile[[32, 64], pl.FP32] = pl.row_expand_sub(tile_x, mean)

        # var = sum(centered^2, dim=-1, keepdim=True) / hidden_size
        squared: pl.Tile[[32, 64], pl.FP32] = pl.mul(centered, centered)
        tmp2: pl.Tile[[32, 64], pl.FP32] = pl.create_tile(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        var: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(squared, tmp2)
        var_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(var, [1, 32])
        var_T = pl.mul(var_T, 0.015625)  # 1.0 / 64  # type: ignore[reportArgumentType]
        var = pl.reshape(var_T, [32, 1])

        # std = sqrt(var + eps)
        var_T2: pl.Tile[[1, 32], pl.FP32] = pl.reshape(var, [1, 32])
        var_eps_T: pl.Tile[[1, 32], pl.FP32] = pl.add(var_T2, 1e-5)  # type: ignore[reportArgumentType]
        std_T: pl.Tile[[1, 32], pl.FP32] = pl.sqrt(var_eps_T)
        std: pl.Tile[[32, 1], pl.FP32] = pl.reshape(std_T, [32, 1])

        # normalized = centered / std (broadcast std across hidden dim)
        normalized: pl.Tile[[32, 64], pl.FP32] = pl.row_expand_div(centered, std)

        # scaled = normalized * gamma (broadcast gamma across batch)
        scaled: pl.Tile[[32, 64], pl.FP32] = pl.col_expand_mul(normalized, tile_gamma)

        # result = scaled + beta (broadcast beta across batch)
        beta_full: pl.Tile[[32, 64], pl.FP32] = pl.col_expand(scaled, tile_beta)
        result: pl.Tile[[32, 64], pl.FP32] = pl.add(scaled, beta_full)

        out: pl.Tensor[[32, 64], pl.FP32] = pl.store(result, [0, 0], [32, 64], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def layer_norm_orch(
        self,
        x: pl.Tensor[[32, 64], pl.FP32],
        gamma: pl.Tensor[[1, 64], pl.FP32],
        beta: pl.Tensor[[1, 64], pl.FP32],
    ) -> pl.Tensor[[32, 64], pl.FP32]:
        output: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
        output = self.kernel_layer_norm(x, gamma, beta, output)
        return output
