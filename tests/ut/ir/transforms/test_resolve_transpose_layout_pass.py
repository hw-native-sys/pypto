# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ResolveTransposeLayout pass."""

import pypto.language as pl
import pytest
from pypto import ir, passes


class TestResolveTransposeLayoutBTranspose:
    """Test B transpose cases: C = A @ B^T."""

    def test_btranspose_basic(self):
        """B stored as [N, K], loaded with transpose=True -> param becomes [K, N] + DN."""
        M, K, N = 64, 64, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32, pl.DN],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32, pl.DN]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_btranspose_non_square(self):
        """Non-square dimensions: M=128, K=64, N=32."""
        M, K, N = 128, 64, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32, pl.DN],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32, pl.DN]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestResolveTransposeLayoutATranspose:
    """Test A transpose cases: C = A^T @ B."""

    def test_atranspose_basic(self):
        """A stored as [K, M], loaded with transpose=True -> param becomes [M, K] + DN."""
        M, K, N = 64, 64, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[K, M], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32, pl.DN],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32, pl.DN], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestResolveTransposeLayoutABTranspose:
    """Test both A and B transposed: C = A^T @ B^T."""

    def test_abtranspose_basic(self):
        """Both A and B transposed -> both params get DN layout."""
        M, K, N = 64, 64, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[K, M], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32, pl.DN],
                b: pl.Tensor[[K, N], pl.FP32, pl.DN],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32, pl.DN], b: pl.Tensor[[K, N], pl.FP32, pl.DN]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_abtranspose_non_square(self):
        """Both transposed with non-square dimensions: M=32, K=128, N=64."""
        M, K, N = 32, 128, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[K, M], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32, pl.DN],
                b: pl.Tensor[[K, N], pl.FP32, pl.DN],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32, pl.DN], b: pl.Tensor[[K, N], pl.FP32, pl.DN]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestResolveTransposeLayoutNoOp:
    """Test cases where the pass should be a no-op."""

    def test_no_transpose_unchanged(self):
        """No transpose=True loads -> program unchanged."""
        M, K, N = 64, 64, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Before)

    def test_already_dn_layout_unchanged(self):
        """Parameter already has DN layout -> pass is idempotent."""
        M, K, N = 64, 64, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32, pl.DN],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32, pl.DN]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Before)

    def test_elementwise_no_transpose(self):
        """Simple elementwise with no transpose -> unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def add_incore(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                x_tile: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
                y_tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0: pl.Tensor[[64, 64], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
                out_0: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
                y: pl.Tensor[[64, 64], pl.FP32] = self.add_incore(x, out_0)
                return y

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Before)

    def test_transpose_false_explicit(self):
        """Explicit transpose=False -> no change."""
        M, K, N = 64, 64, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=False)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=False)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Before)


class TestResolveTransposeLayoutMixed:
    """Test mixed scenarios with one transpose and one non-transpose param."""

    def test_only_second_param_transposed(self):
        """Only second param has transpose -> only second param changes."""
        M, K, N = 64, 128, 64

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32, pl.DN],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32, pl.DN]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_only_first_param_transposed(self):
        """Only first param has transpose -> only first param changes."""
        M, K, N = 64, 64, 128

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[K, M], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32, pl.DN],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32, pl.DN], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestResolveTransposeLayoutPartialLoad:
    """Test cases where tile.load reads a subset of the tensor (partial load)."""

    def test_partial_load_square_tensor(self):
        """Tensor [128, 128] with partial tile.load [128, 64] transpose -> shape stays [128, 128] + DN.

        Regression test for #606: paged attention key_cache tensor shape was incorrectly
        changed from [128, 128] to [128, 64] because the pass used the tile load shape
        instead of transposing the original tensor shape.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
                tile_k = pl.load(
                    key_cache, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_a_l0 = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_k_l0 = pl.move(tile_k, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0, tile_k_l0)
                out_store = pl.store(tile_c, [0, 0], out)
                return out_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
                out_result = self.kernel(a, key_cache, out)
                return out_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16, pl.DN],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
                tile_k = pl.load(
                    key_cache, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_a_l0 = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_k_l0 = pl.move(tile_k, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0, tile_k_l0)
                out_store = pl.store(tile_c, [0, 0], out)
                return out_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16, pl.DN],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
                out_result = self.kernel(a, key_cache, out)
                return out_result

        After = passes.resolve_transpose_layout()(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
