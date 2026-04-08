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
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


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


class TestBatchMatmulTilePTO(TestBatchMatmulTile):
    """Tile-level batch matmul with PTO backend."""

    __test__ = False

    def get_name(self) -> str:
        return f"batch_matmul_tile_pto_{self.batch}x{self.M}x{self.K}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class TestBatchMatmulOperations:
    """Test suite for tile-level batch matrix multiplication."""

    @pytest.mark.parametrize(
        "batch,m,k,n",
        [
            (2, 64, 64, 64),
            (4, 32, 32, 32),
            (1, 128, 64, 128),
        ],
    )
    def test_batch_matmul_tile(self, test_runner, batch, m, k, n):
        """Test tile.batch_matmul with explicit load/store."""
        test_case = TestBatchMatmulTile(batch=batch, m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "batch,m,k,n",
        [
            (2, 64, 64, 64),
            (4, 32, 32, 32),
            (1, 128, 64, 128),
        ],
    )
    def test_batch_matmul_tile_pto(self, test_runner, batch, m, k, n):
        """Test tile.batch_matmul with PTO backend."""
        test_case = TestBatchMatmulTilePTO(batch=batch, m=m, k=k, n=n)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed (PTO): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--forked"])
