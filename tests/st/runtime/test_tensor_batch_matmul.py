# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
System tests for ND tensor.matmul / tensor.matmul_acc lowering.

These tests exercise the high-level ``pl.matmul`` / ``pl.matmul_acc`` DSL
entry points with rank>2 tensor inputs, validating that the compiler:

* dispatches ``tensor.matmul`` / ``tensor.matmul_acc`` to the batched tile
  ops (``tile.batch_matmul`` / ``tile.batch_matmul_acc``) when any operand
  has rank > 2 (see ``op_conversion_registry.cpp``);
* propagates ``b_trans=True`` to the producer ``pl.load`` via the
  ``InputSpaceReq`` mechanism (no explicit ``pl.transpose`` is emitted);
* unrolls the batched tile op via ``FlattenTileNdTo2D`` into per-batch 2D
  tile ops and produces correct numerical results.

This is the end-to-end DeepSeek-V4 MoE W2 matmul shape (``2D @ 3D``) the
``deepseek_v4_decode_moe_w2_matmul`` kernel relies on.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec


class TestTensorMatmul2dBy3d(PTOTestCase):
    """``pl.matmul(2D, 3D, b_trans=True)`` — single-batch broadcast.

    Validates the rank-based dispatch in ConvertTensorToTileOps and the
    batch-1 fast path in FlattenTileNdTo2D's batch_matmul lowering.
    """

    __test__ = False

    def __init__(
        self,
        m: int = 16,
        k: int = 64,
        n: int = 64,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"tensor_matmul_2d_by_3d_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [1, self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [1, self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class TensorMatmul2dBy3dProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tensor_matmul_2d_by_3d(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[1, N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[1, M, N], pl.FP32]],
            ) -> pl.Tensor[[1, M, N], pl.FP32]:
                # ND tensor-level matmul: lowers to tile.batch_matmul because b is rank-3.
                matmul_out: pl.Tensor[[1, M, N], pl.FP32] = pl.matmul(a, b, b_trans=True, out_dtype=pl.FP32)
                # Explicitly write back into the Out tensor `c` so the runtime sees the
                # result on the bound output buffer (returning a fresh tensor would
                # otherwise leak a separate result slot and leave `c` uninitialised).
                c = pl.assemble(c, matmul_out, offset=[0, 0, 0])
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[1, N, K], pl.FP32],
                out_c: pl.Out[pl.Tensor[[1, M, N], pl.FP32]],
            ) -> pl.Tensor[[1, M, N], pl.FP32]:
                out_c = self.tensor_matmul_2d_by_3d(a, b, out_c)
                return out_c

        return TensorMatmul2dBy3dProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["c"][:] = torch.matmul(a.unsqueeze(0), b.transpose(-2, -1))


class TestTensorMatmulAcc2dBy3d(PTOTestCase):
    """``pl.matmul`` + ``pl.matmul_acc`` chained on (2D, 3D) inputs.

    K is split into two halves: the first chunk feeds ``pl.matmul`` and the
    second chunk feeds ``pl.matmul_acc``. This is the DeepSeek-V4 MoE W2
    pattern from ``deepseek_v4_decode_moe_w2_matmul`` (3D expert weights
    consumed without manual 2D slicing).
    """

    __test__ = False

    def __init__(
        self,
        m: int = 16,
        k: int = 64,
        n: int = 64,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n
        assert self.K % 2 == 0, "K must split evenly for two-chunk accumulation"

    def get_name(self) -> str:
        return f"tensor_matmul_acc_2d_by_3d_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [1, self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [1, self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N
        K_HALF = K // 2

        @pl.program
        class TensorMatmulAcc2dBy3dProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tensor_matmul_acc_2d_by_3d(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[1, N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[1, M, N], pl.FP32]],
            ) -> pl.Tensor[[1, M, N], pl.FP32]:
                a0: pl.Tensor[[M, K_HALF], pl.FP32] = pl.slice(a, [M, K_HALF], [0, 0])
                a1: pl.Tensor[[M, K_HALF], pl.FP32] = pl.slice(a, [M, K_HALF], [0, K_HALF])
                b0: pl.Tensor[[1, N, K_HALF], pl.FP32] = pl.slice(b, [1, N, K_HALF], [0, 0, 0])
                b1: pl.Tensor[[1, N, K_HALF], pl.FP32] = pl.slice(b, [1, N, K_HALF], [0, 0, K_HALF])
                acc: pl.Tensor[[1, M, N], pl.FP32] = pl.matmul(a0, b0, b_trans=True, out_dtype=pl.FP32)
                acc_final: pl.Tensor[[1, M, N], pl.FP32] = pl.matmul_acc(acc, a1, b1, b_trans=True)
                # Write back into the Out tensor `c` (see TestTensorMatmul2dBy3d for why).
                c = pl.assemble(c, acc_final, offset=[0, 0, 0])
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[1, N, K], pl.FP32],
                out_c: pl.Out[pl.Tensor[[1, M, N], pl.FP32]],
            ) -> pl.Tensor[[1, M, N], pl.FP32]:
                out_c = self.tensor_matmul_acc_2d_by_3d(a, b, out_c)
                return out_c

        return TensorMatmulAcc2dBy3dProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["c"][:] = torch.matmul(a.unsqueeze(0), b.transpose(-2, -1))


_ND_SHAPES = [(16, 64, 64), (16, 128, 64)]


class TestTensorBatchMatmulOperations:
    """Test suite for ND ``pl.matmul`` / ``pl.matmul_acc`` lowering."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _ND_SHAPES)
    def test_tensor_matmul_2d_by_3d(self, test_runner, platform, m, k, n):
        """``pl.matmul(2D, 3D, b_trans=True)`` end-to-end."""
        result = test_runner.run(TestTensorMatmul2dBy3d(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _ND_SHAPES)
    def test_tensor_matmul_acc_2d_by_3d(self, test_runner, platform, m, k, n):
        """Chained ``pl.matmul`` + ``pl.matmul_acc`` on (2D, 3D) inputs."""
        result = test_runner.run(TestTensorMatmulAcc2dBy3d(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
