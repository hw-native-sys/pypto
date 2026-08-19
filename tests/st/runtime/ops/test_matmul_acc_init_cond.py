# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device tests for ``matmul_acc(init_cond=...)`` and accumulation into a
window of a shared ``Acc`` tile.

Both behaviours are numeric, not structural: the unit tests assert the emitted
``pto.tmatmul`` / ``pto.tmatmul.acc`` forms, but only a device run shows that the
first K step actually *overwrites* the accumulator (rather than accumulating onto
whatever L0C held) and that a window write lands where the parent expects it.

``TestMatmulAccWindowInitCond`` is the shape that motivated the feature: one
accumulator shared by several output column tiles, each accumulating its own K
reduction in place. Note the accumulator is sliced along **columns** — a row
window of a multi-block-column ``Acc`` tile is rejected by
``CanonicalizeTileSlice``, because the MAD has no destination stride
(hw-native-sys/pto-isa#253).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec

# Same rationale as tests/st/runtime/ops/test_matmul.py: the cube reduces K in a
# different order than torch's single-pass BLAS, so FP32 goldens drift by a few
# ULP of the partial sums. Reuse that file's calibrated floor.
_FP32_MATMUL_RTOL = 1e-4
_FP32_MATMUL_ATOL = 1e-4


class TestMatmulAccInitCond(PTOTestCase):
    """Split-K driven by ``init_cond`` instead of a peeled first step.

    The accumulator is never zeroed, so if ``init_cond`` failed to select the
    overwriting form on ``k0 == 0`` the result would carry stale L0C content and
    the comparison against ``a @ b`` would fail.
    """

    __test__ = False

    def __init__(
        self, m: int = 64, k: int = 256, n: int = 64, k_tile: int = 64, *, platform=None, config=None
    ):
        super().__init__(config, platform=platform)
        if config is None:
            self.config.rtol = _FP32_MATMUL_RTOL
            self.config.atol = _FP32_MATMUL_ATOL
        self.M, self.K, self.N, self.K_TILE = m, k, n, k_tile

    def get_name(self) -> str:
        return f"matmul_acc_init_cond_{self.M}x{self.K}x{self.N}_kt{self.K_TILE}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N, KT = self.M, self.K, self.N, self.K_TILE

        @pl.program
        class MatmulAccInitCondProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_acc_split_k(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                acc = pl.tile.create([M, N], pl.FP32, target_memory=pl.MemorySpace.Acc)
                for k0 in pl.range(0, K, KT):
                    a_l1 = pl.load(a, offsets=[0, k0], shapes=[M, KT], target_memory=pl.MemorySpace.Mat)
                    b_l1 = pl.load(b, offsets=[k0, 0], shapes=[KT, N], target_memory=pl.MemorySpace.Mat)
                    a_l0 = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
                    b_l0 = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
                    acc = pl.matmul_acc(acc, a_l0, b_l0, init_cond=(k0 == 0))
                out_c = pl.store(acc, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul_acc_split_k(a, b, out_c)
                return out_c

        return MatmulAccInitCondProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


class TestMatmulAccWindowInitCond(PTOTestCase):
    """Several output column tiles accumulating into windows of one ``Acc`` tile.

    Each ``[M, N_TILE]`` window runs its own K reduction in place, so this covers
    the destination aliasing (the MAD must write the window's ``pto.subview``,
    not a private L0C buffer) together with ``init_cond``.
    """

    __test__ = False

    def __init__(
        self,
        m: int = 16,
        k: int = 128,
        n_tile: int = 64,
        tiles: int = 2,
        k_tile: int = 64,
        *,
        platform=None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        if config is None:
            self.config.rtol = _FP32_MATMUL_RTOL
            self.config.atol = _FP32_MATMUL_ATOL
        self.M, self.K, self.N_TILE, self.TILES, self.K_TILE = m, k, n_tile, tiles, k_tile
        self.N = n_tile * tiles

    def get_name(self) -> str:
        return f"matmul_acc_window_{self.M}x{self.K}x{self.N}_t{self.TILES}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N, NT, TILES, KT = self.M, self.K, self.N, self.N_TILE, self.TILES, self.K_TILE

        @pl.program
        class MatmulAccWindowProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_acc_window(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                acc = pl.tile.create([M, N], pl.FP32, target_memory=pl.MemorySpace.Acc)
                for t in pl.range(TILES):
                    for k0 in pl.range(0, K, KT):
                        a_l1 = pl.load(a, offsets=[0, k0], shapes=[M, KT], target_memory=pl.MemorySpace.Mat)
                        b_l1 = pl.load(
                            b, offsets=[k0, t * NT], shapes=[KT, NT], target_memory=pl.MemorySpace.Mat
                        )
                        a_l0 = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
                        b_l0 = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
                        # Column window: spans the parent's full row extent, so it
                        # is contiguous in L0C and legal as a MAD destination.
                        win = pl.tile.slice(acc, [M, NT], [0, t * NT])
                        win = pl.matmul_acc(win, a_l0, b_l0, init_cond=(k0 == 0))
                out_c = pl.store(acc, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul_acc_window(a, b, out_c)
                return out_c

        return MatmulAccWindowProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


_INIT_COND_SHAPES = [(64, 256, 64, 64), (32, 128, 64, 32)]


class TestMatmulAccInitCondOperations:
    """Device tests for conditional accumulator initialization."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n,k_tile", _INIT_COND_SHAPES)
    def test_matmul_acc_init_cond(self, test_runner, platform, m, k, n, k_tile):
        """Split-K where ``init_cond=(k0 == 0)`` replaces the peeled first step."""
        result = test_runner.run(TestMatmulAccInitCond(m=m, k=k, n=n, k_tile=k_tile, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_matmul_acc_column_window(self, test_runner, platform):
        """Column windows of one shared ``Acc`` tile, each accumulating over K."""
        result = test_runner.run(TestMatmulAccWindowInitCond(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
