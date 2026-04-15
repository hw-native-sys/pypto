# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Row-wise softmax (64x64 input).

Formula: output[i] = exp(a[i] - max(a[i])) / sum(exp(a[i] - max(a[i])))

Concepts introduced:
  - pl.create_tile for scratch tiles in a specific memory space
  - Row reduction ops: pl.row_max, pl.row_sum
  - Row broadcast ops: pl.row_expand_sub, pl.row_expand_div
  - Numerically stable softmax algorithm

Run:  python examples/kernels/06_softmax.py
Next: examples/kernels/07_normalization.py
"""

import pypto.language as pl


@pl.program
class TileSoftmaxProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_softmax(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])

        # Step 1: row-wise max for numerical stability
        max_tmp = pl.create_tile([64, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_max: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)

        # Step 2: subtract row max from each row: x - max(x)
        shifted = pl.row_expand_sub(tile_a, row_max)

        # Step 3: exp(x - max(x))
        exp_shifted = pl.exp(shifted)

        # Step 4: row-wise sum of exp values
        sum_tmp = pl.create_tile([64, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(exp_shifted, sum_tmp)

        # Step 5: divide each row by its sum
        result = pl.row_expand_div(exp_shifted, row_sum)

        out = pl.store(result, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output_ret = self.tile_softmax(a, output)
        return output_ret


if __name__ == "__main__":
    print(TileSoftmaxProgram.as_python())
