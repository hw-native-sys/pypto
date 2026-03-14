# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Matrix multiplication with transpose operations.

This example demonstrates three transpose variants:
1. C = A @ B^T  (MatmulTransBProgram)
2. C = A^T @ B  (MatmulTransAProgram)
3. C = A^T @ B^T (MatmulTransABProgram)

Each variant uses:
- Vector kernel(s) for transpose via pl.transpose
- Cube kernel for matmul

Kernels are separated because Cube and Vector operations cannot mix in one function.
"""

import pypto.language as pl


# =============================================================================
# C = A @ B^T
# =============================================================================


@pl.program
class MatmulTransBProgram:
    """Compute C = A @ B^T where A:[64,64], B:[64,64], C:[64,64]."""

    @pl.function
    def transpose_kernel(
        self,
        b: pl.Tensor[[64, 64], pl.FP32],
        b_t: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Transpose B: b_t = b^T (Vector kernel)."""
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64])
        tile_b_T: pl.Tile[[64, 64], pl.FP32] = pl.transpose(tile_b, axis1=0, axis2=1)
        out_b_t: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b_T, [0, 0], [64, 64], b_t)
        return out_b_t

    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b_t: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Matmul: c = a @ b_t (Cube kernel)."""
        tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b_t, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out_c = pl.l0c_store(tile_c_l0c, [0, 0], [64, 64], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Orchestrate: C = A @ B^T."""
        b_t: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        out_c: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        b_t = self.transpose_kernel(b, b_t)
        out_c = self.matmul_kernel(a, b_t, out_c)
        return out_c


# =============================================================================
# C = A^T @ B
# =============================================================================


@pl.program
class MatmulTransAProgram:
    """Compute C = A^T @ B where A:[64,64], B:[64,64], C:[64,64]."""

    @pl.function
    def transpose_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        a_t: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Transpose A: a_t = a^T (Vector kernel)."""
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
        tile_a_T: pl.Tile[[64, 64], pl.FP32] = pl.transpose(tile_a, axis1=0, axis2=1)
        out_a_t: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a_T, [0, 0], [64, 64], a_t)
        return out_a_t

    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        a_t: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Matmul: c = a_t @ b (Cube kernel)."""
        tile_a_l1 = pl.load(a_t, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out_c = pl.l0c_store(tile_c_l0c, [0, 0], [64, 64], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Orchestrate: C = A^T @ B."""
        a_t: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        out_c: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        a_t = self.transpose_kernel(a, a_t)
        out_c = self.matmul_kernel(a_t, b, out_c)
        return out_c


# =============================================================================
# C = A^T @ B^T
# =============================================================================


@pl.program
class MatmulTransABProgram:
    """Compute C = A^T @ B^T where A:[64,64], B:[64,64], C:[64,64]."""

    @pl.function
    def transpose_a_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        a_t: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Transpose A: a_t = a^T (Vector kernel)."""
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
        tile_a_T: pl.Tile[[64, 64], pl.FP32] = pl.transpose(tile_a, axis1=0, axis2=1)
        out_a_t: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a_T, [0, 0], [64, 64], a_t)
        return out_a_t

    @pl.function
    def transpose_b_kernel(
        self,
        b: pl.Tensor[[64, 64], pl.FP32],
        b_t: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Transpose B: b_t = b^T (Vector kernel)."""
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64])
        tile_b_T: pl.Tile[[64, 64], pl.FP32] = pl.transpose(tile_b, axis1=0, axis2=1)
        out_b_t: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b_T, [0, 0], [64, 64], b_t)
        return out_b_t

    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        self,
        a_t: pl.Tensor[[64, 64], pl.FP32],
        b_t: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Matmul: c = a_t @ b_t (Cube kernel)."""
        tile_a_l1 = pl.load(a_t, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b_t, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out_c = pl.l0c_store(tile_c_l0c, [0, 0], [64, 64], c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Orchestrate: C = A^T @ B^T."""
        a_t: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        b_t: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        out_c: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        a_t = self.transpose_a_kernel(a, a_t)
        b_t = self.transpose_b_kernel(b, b_t)
        out_c = self.matmul_kernel(a_t, b_t, out_c)
        return out_c
