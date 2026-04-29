# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for Mat slice → Left/Right matmul pattern.

Validates the pattern where a wide BF16 tile is loaded into Mat (L1) to satisfy
the >= 512 B GM row-alignment constraint, then sliced into two K-chunks that are
moved to Left for matmul.  This pattern appears in Qwen3 decode kernels (q_proj,
kv_proj, out_proj, gate_proj, up_proj, down_proj) where K_CHUNK=128 with BF16
yields only 256 B per row — below the 512 B minimum.  Merging two adjacent
K-chunks into one wider GM load (row = 2 x 128 x 2 = 512 B) and splitting
on-chip is the fix.

Codegen expectation:
  pl.slice on a Mat tile → pto.subview (zero-copy view, not pto.textract)
  pl.move from subview   → pto.tmov (Mat subview → Left)
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
K = 128
N = 64


class TestMatSliceToLeft(PTOTestCase):
    """Wide Mat load → slice → move to Left → matmul + matmul_acc.

    C[M, N] = A[M, 2K] @ B[2K, N]  (BF16 inputs, FP32 accumulator)

    The A matrix is loaded as a single [M, 2K] tile into Mat, then sliced into
    two [M, K] halves that are moved to Left.  B is split into two [K, N]
    weight tensors loaded into Mat → Right.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"mat_slice_to_left_{M}x{2 * K}x{N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, 2 * K], DataType.BF16, init_value=torch.randn),
            TensorSpec("b0", [K, N], DataType.BF16, init_value=torch.randn),
            TensorSpec("b1", [K, N], DataType.BF16, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class MatSliceToLeftProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def mat_slice_to_left(
                self,
                a: pl.Tensor[[M, 2 * K], pl.BF16],
                b0: pl.Tensor[[K, N], pl.BF16],
                b1: pl.Tensor[[K, N], pl.BF16],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                # Wide GM→Mat load: row = 2*128*2 = 512 B (satisfies alignment)
                tile_a_pair = pl.load(a, offsets=[0, 0], shapes=[M, 2 * K], target_memory=pl.MemorySpace.Mat)

                # Slice into two K-chunks (stays in Mat as subview)
                tile_a_0 = pl.slice(tile_a_pair, [M, K], [0, 0])
                tile_a_1 = pl.slice(tile_a_pair, [M, K], [0, K])

                # Move slices to Left for matmul
                tile_a_0_left = pl.move(tile_a_0, target_memory=pl.MemorySpace.Left)
                tile_a_1_left = pl.move(tile_a_1, target_memory=pl.MemorySpace.Left)

                # Load B tiles to Mat → Right
                tile_b_0_mat = pl.load(b0, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_b_1_mat = pl.load(b1, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_b_0_right = pl.move(tile_b_0_mat, target_memory=pl.MemorySpace.Right)
                tile_b_1_right = pl.move(tile_b_1_mat, target_memory=pl.MemorySpace.Right)

                # Matmul with K-split accumulation
                acc = pl.matmul(tile_a_0_left, tile_b_0_right)
                acc = pl.matmul_acc(acc, tile_a_1_left, tile_b_1_right)

                out_c = pl.store(acc, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, 2 * K], pl.BF16],
                b0: pl.Tensor[[K, N], pl.BF16],
                b1: pl.Tensor[[K, N], pl.BF16],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.mat_slice_to_left(a, b0, b1, out_c)
                return out_c

        return MatSliceToLeftProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].float()
        b0 = tensors["b0"].float()
        b1 = tensors["b1"].float()
        # A[:, :K] @ B0 + A[:, K:] @ B1
        tensors["c"][:] = torch.matmul(a[:, :K], b0) + torch.matmul(a[:, K:], b1)


# =============================================================================
# pytest test suite
# =============================================================================


class TestMatSliceToLeftSuite:
    """Test suite for Mat slice → Left matmul pattern."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_mat_slice_to_left(self, test_runner, platform):
        """Wide Mat load sliced into two Left tiles for K-split matmul."""
        result = test_runner.run(TestMatSliceToLeft(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
