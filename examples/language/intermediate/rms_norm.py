# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
RMS normalization using PyPTO language DSL.

Formula: output = x / sqrt(mean(x^2) + eps) * gamma

Computed on 32x64 input, normalizing across the hidden dimension (64).
"""

import pypto.language as pl


@pl.program
class RMSNormProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_rms_norm(
        self,
        x: pl.Tensor[[32, 64], pl.FP32],
        gamma: pl.Tensor[[1, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
    ) -> pl.Tensor[[32, 64], pl.FP32]:
        tile_x: pl.Tile[[32, 64], pl.FP32] = pl.load(x, [0, 0], [32, 64])
        tile_gamma: pl.Tile[[1, 64], pl.FP32] = pl.load(gamma, [0, 0], [1, 64])

        # [32, 1] tiles are ColMajor; scalar ops (mul/add) need RowMajor.
        # Workaround: reshape [32, 1] -> [1, 32], apply op, reshape back.

        # squared = x * x
        squared: pl.Tile[[32, 64], pl.FP32] = pl.mul(tile_x, tile_x)

        # mean_sq = sum(x^2, dim=-1, keepdim=True) / hidden_size
        tmp: pl.Tile[[32, 64], pl.FP32] = pl.create_tile(
            [32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        mean_sq: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(squared, tmp)
        mean_sq_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean_sq, [1, 32])
        mean_sq_T = pl.mul(mean_sq_T, 0.015625)  # 1.0 / 64  # type: ignore[reportArgumentType]
        mean_sq = pl.reshape(mean_sq_T, [32, 1])

        # rms = sqrt(mean_sq + eps)
        mean_sq_T2: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean_sq, [1, 32])
        rms_T: pl.Tile[[1, 32], pl.FP32] = pl.add(mean_sq_T2, 1e-5)  # type: ignore[reportArgumentType]
        rms_T = pl.sqrt(rms_T)
        rms: pl.Tile[[32, 1], pl.FP32] = pl.reshape(rms_T, [32, 1])

        # normalized = x / rms (broadcast rms across hidden dim)
        normalized: pl.Tile[[32, 64], pl.FP32] = pl.row_expand_div(tile_x, rms)

        # result = normalized * gamma (broadcast gamma across batch)
        result: pl.Tile[[32, 64], pl.FP32] = pl.col_expand_mul(normalized, tile_gamma)

        out: pl.Tensor[[32, 64], pl.FP32] = pl.store(result, [0, 0], [32, 64], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def rms_norm_orch(
        self,
        x: pl.Tensor[[32, 64], pl.FP32],
        gamma: pl.Tensor[[1, 64], pl.FP32],
    ) -> pl.Tensor[[32, 64], pl.FP32]:
        output: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
        output = self.kernel_rms_norm(x, gamma, output)
        return output
