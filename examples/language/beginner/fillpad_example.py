# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example demonstrating the fillpad operation in PyPTO.

The fillpad operation fills remaining elements of a tile with padding values.
This is useful when loading partial tiles from tensors where the loaded region
may be smaller than the tile capacity, and the remaining elements need to be
filled with a consistent padding value for correct computation.

Typical use case:
  1. Load a partial tile from a tensor (e.g., at boundary regions)
  2. Apply fillpad to fill remaining elements with padding
  3. Perform computation on the padded tile
  4. Store the result back to the output tensor
"""

import pypto.language as pl


@pl.program
class FillpadExampleProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def fillpad_kernel(
        self,
        input_tensor: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Load tile, apply fillpad, then store.

        This kernel demonstrates the basic fillpad pattern:
        1. Load a tile from the input tensor
        2. Apply fillpad to ensure all elements are valid
        3. Perform an operation (here, add with self)
        4. Store the result
        """
        tile: pl.Tile[[128, 128], pl.FP32] = pl.load(
            input_tensor, offsets=[0, 0], shapes=[128, 128]
        )
        padded_tile: pl.Tile[[128, 128], pl.FP32] = pl.fillpad(tile)
        result_tile: pl.Tile[[128, 128], pl.FP32] = pl.add(padded_tile, padded_tile)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(
            result_tile, offsets=[0, 0], shapes=[128, 128], output_tensor=output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Orchestrate the fillpad kernel execution."""
        output: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        output = self.fillpad_kernel(input_tensor, output)
        return output
