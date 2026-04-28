# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST tests for pl.tile.slice with the target_memory kwarg (issue #1198).

Exercises the Qwen3-style pattern: load a merged tile into Mat (L1) once, then
slice it directly into Left/Right (L0A/L0B) to feed matmul, replacing the old
``slice → move`` two-op chain with a single textract.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec


class TestSliceTargetMemoryLeft(PTOTestCase):
    """Merged Mat load + sliced Left tile feeding matmul.

    A is shaped [M, 2K] and loaded into Mat in one DMA, then sliced with
    ``target_memory=Left`` to take the first K columns directly into L0A.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"slice_target_left_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, 2 * self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N
        K2 = 2 * K

        @pl.program
        class SliceTargetLeftProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_sliced(
                self,
                a: pl.Tensor[[M, K2], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[M, K2], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.tile.slice(
                    tile_a_l1, shape=[M, K], offset=[0, 0], target_memory=pl.MemorySpace.Left
                )
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K2], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul_sliced(a, b, out_c)
                return out_c

        return SliceTargetLeftProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"]
        b = tensors["b"]
        k = b.shape[0]
        tensors["c"][:] = torch.matmul(a[:, :k], b)


class TestSliceTargetMemoryRight(PTOTestCase):
    """Merged Mat load + sliced Right tile feeding matmul.

    B is shaped [K, 2N] and loaded into Mat in one DMA, then sliced with
    ``target_memory=Right`` to take the first N columns directly into L0B.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"slice_target_right_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, 2 * self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N
        N2 = 2 * N

        @pl.program
        class SliceTargetRightProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_sliced(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N2], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[K, N2], target_memory=pl.MemorySpace.Mat)
                tile_b_l0b = pl.tile.slice(
                    tile_b_l1, shape=[K, N], offset=[0, 0], target_memory=pl.MemorySpace.Right
                )
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N2], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul_sliced(a, b, out_c)
                return out_c

        return SliceTargetRightProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"]
        b = tensors["b"]
        n = b.shape[1] // 2
        tensors["c"][:] = torch.matmul(a, b[:, :n])


_SHAPES = [(64, 64, 64), (128, 64, 128)]


class TestSliceTargetMemory:
    """Test suite for pl.tile.slice with target_memory kwarg."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _SHAPES)
    def test_slice_target_left(self, test_runner, platform, m, k, n):
        """Slice a Mat tile directly into Left and feed it to matmul."""
        result = test_runner.run(TestSliceTargetMemoryLeft(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _SHAPES)
    def test_slice_target_right(self, test_runner, platform, m, k, n):
        """Slice a Mat tile directly into Right and feed it to matmul."""
        result = test_runner.run(TestSliceTargetMemoryRight(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
