# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
System tests for tile.batch_matmul operation.

This test validates the tile.batch_matmul operation through the complete
compilation and execution pipeline, comparing results against PyTorch reference.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec


class TestBatchMatmulTile(PTOTestCase):
    """Tile-level batch matmul: explicit load → batch_matmul → store.

    Uses tile.batch_matmul directly, mirroring how TestMatmul uses tile.matmul.
    """

    __test__ = False

    def __init__(self, batch: int = 2, m: int = 64, k: int = 64, n: int = 64, config=None):
        super().__init__(config)
        self.batch = batch
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"batch_matmul_tile_{self.batch}x{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.batch, self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.batch, self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.batch, self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, M, K, N = self.batch, self.M, self.K, self.N

        @pl.program
        class BatchMatmulTileProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def batch_matmul_tile(
                self,
                a: pl.Tensor[[B, M, K], pl.FP32],
                b: pl.Tensor[[B, K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
            ) -> pl.Tensor[[B, M, N], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0, 0], shapes=[B, M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, offsets=[0, 0, 0], shapes=[B, K, N], target_memory=pl.MemorySpace.Mat)
                tile_c = pl.batch_matmul(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[B, M, K], pl.FP32],
                b: pl.Tensor[[B, K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
            ) -> pl.Tensor[[B, M, N], pl.FP32]:
                out_c = self.batch_matmul_tile(a, b, c)
                return out_c

        return BatchMatmulTileProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.bmm(tensors["a"], tensors["b"])


class TestBatchMatmulBTranspose(PTOTestCase):
    """Tile-level batch matmul with B transposed: C = A @ B^T.

    B is stored as [batch, N, K] and transposed inline before batch_matmul.
    """

    __test__ = False

    def __init__(self, batch: int = 2, m: int = 64, k: int = 64, n: int = 64, config=None):
        super().__init__(config)
        self.batch = batch
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"batch_matmul_b_trans_{self.batch}x{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.batch, self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.batch, self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.batch, self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, M, K, N = self.batch, self.M, self.K, self.N

        @pl.program
        class BatchMatmulBTransProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def batch_matmul_bt(
                self,
                a: pl.Tensor[[B, M, K], pl.FP32],
                b: pl.Tensor[[B, N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
            ) -> pl.Tensor[[B, M, N], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0, 0], shapes=[B, M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(
                    b, offsets=[0, 0, 0], shapes=[B, N, K], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_c = pl.batch_matmul(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[B, M, K], pl.FP32],
                b: pl.Tensor[[B, N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
            ) -> pl.Tensor[[B, M, N], pl.FP32]:
                out_c = self.batch_matmul_bt(a, b, c)
                return out_c

        return BatchMatmulBTransProgram

    def compute_expected(self, tensors, params=None):
        # B is [batch, N, K], transpose last two dims → [batch, K, N]
        tensors["c"][:] = torch.bmm(tensors["a"], tensors["b"].transpose(-2, -1))


class TestBatchMatmulATranspose(PTOTestCase):
    """Tile-level batch matmul with A transposed: C = A^T @ B.

    A is stored as [batch, K, M] and transposed inline before batch_matmul.
    """

    __test__ = False

    def __init__(self, batch: int = 2, m: int = 64, k: int = 64, n: int = 64, config=None):
        super().__init__(config)
        self.batch = batch
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"batch_matmul_a_trans_{self.batch}x{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.batch, self.K, self.M], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.batch, self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.batch, self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, M, K, N = self.batch, self.M, self.K, self.N

        @pl.program
        class BatchMatmulATransProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def batch_matmul_at(
                self,
                a: pl.Tensor[[B, K, M], pl.FP32],
                b: pl.Tensor[[B, K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
            ) -> pl.Tensor[[B, M, N], pl.FP32]:
                tile_a = pl.load(
                    a, offsets=[0, 0, 0], shapes=[B, K, M], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_b = pl.load(b, offsets=[0, 0, 0], shapes=[B, K, N], target_memory=pl.MemorySpace.Mat)
                tile_c = pl.batch_matmul(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[B, K, M], pl.FP32],
                b: pl.Tensor[[B, K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
            ) -> pl.Tensor[[B, M, N], pl.FP32]:
                out_c = self.batch_matmul_at(a, b, c)
                return out_c

        return BatchMatmulATransProgram

    def compute_expected(self, tensors, params=None):
        # A is [batch, K, M], transpose last two dims → [batch, M, K]
        tensors["c"][:] = torch.bmm(tensors["a"].transpose(-2, -1), tensors["b"])


class TestBatchMatmulOperations:
    """Test suite for tile-level batch matrix multiplication.

    Covers both rank>2 tensor inputs and transpose-driven DN-layout codegen.
    """

    @pytest.mark.parametrize(
        "batch,m,k,n",
        [
            (2, 64, 64, 64),
            (4, 32, 32, 32),
            (1, 128, 64, 128),
            # Non-square rank>2 coverage on the non-DN path (no transpose).
            (2, 64, 32, 64),
            (2, 32, 64, 128),
        ],
    )
    def test_batch_matmul_tile(self, test_runner, batch, m, k, n):
        """Test tile.batch_matmul with explicit load/store on the rank>2 non-DN path."""
        test_case = TestBatchMatmulTile(batch=batch, m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "batch,m,k,n",
        [
            (2, 64, 64, 64),
            (4, 32, 32, 32),
            # Non-square DN-layout coverage: B is [batch, N, K] and transpose=True.
            (2, 64, 32, 128),
            (2, 32, 64, 48),
        ],
    )
    def test_batch_matmul_b_transpose(self, test_runner, batch, m, k, n):
        """Test tile.batch_matmul with B transposed on the DN-layout codegen path."""
        test_case = TestBatchMatmulBTranspose(batch=batch, m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (B-trans): {result.error}"

    @pytest.mark.parametrize(
        "batch,m,k,n",
        [
            (2, 64, 64, 64),
            (4, 32, 32, 32),
            # Non-square DN-layout coverage: A is [batch, K, M] and transpose=True.
            (2, 128, 32, 64),
            (2, 48, 64, 32),
        ],
    )
    def test_batch_matmul_a_transpose(self, test_runner, batch, m, k, n):
        """Test tile.batch_matmul with A transposed on the DN-layout codegen path."""
        test_case = TestBatchMatmulATranspose(batch=batch, m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (A-trans): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--forked"])
