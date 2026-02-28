# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Vector DAG computation using PyPTO language DSL.

Implements: f = (a + b + 1)(a + b + 2) + (a + b)

Task graph:
  t0: c = kernel_add(a, b)
  t1: d = kernel_add_scalar(c, 1.0)
  t2: e = kernel_add_scalar(c, 2.0)
  t3: g = kernel_mul(d, e)
  t4: f = kernel_add(g, c)
"""

import pypto.language as pl


@pl.program
class VectorDAGProgram:
    """Vector example program with 3 InCore kernels and 1 Orchestration function."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Adds two tensors element-wise: result = a + b"""
        a_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        b_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.add(a_tile, b_tile)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(result, [0, 0], [128, 128], output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add_scalar(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        scalar: pl.Scalar[pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Adds a scalar to each element: result = a + scalar"""
        x: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.add(x, scalar)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(result, [0, 0], [128, 128], output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_mul(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Multiplies two tensors element-wise: result = a * b"""
        a_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        b_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.mul(a_tile, b_tile)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(result, [0, 0], [128, 128], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch_vector(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Orchestration for formula: f = (a + b + 1)(a + b + 2) + (a + b)

        Task graph:
        t0: c = kernel_add(a, b)
        t1: d = kernel_add_scalar(c, 1.0)
        t2: e = kernel_add_scalar(c, 2.0)
        t3: g = kernel_mul(d, e)
        t4: f = kernel_add(g, c)
        """
        c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        c = self.kernel_add(a, b, c)
        d: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        d = self.kernel_add_scalar(c, 1.0, d)  # type: ignore[reportArgumentType]
        e: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        e = self.kernel_add_scalar(c, 2.0, e)  # type: ignore[reportArgumentType]
        g: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        g = self.kernel_mul(d, e, g)
        f: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        f = self.kernel_add(g, c, f)
        return f
