# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bisection ST for issue #1525: chained matmul -> cast -> matmul all-zero output.

Two cases share one CORE_GROUP scope with split=NONE:

  CubeVecCast   : matmul -> cast -> cast  (cube->vec only). If this is all-zero,
                  the cube->vec transfer drops data.
  ChainedMatmul : matmul -> cast -> matmul (cube->vec->cube). The full repro.
                  If CubeVecCast passes but this fails, vec->cube is at fault.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec

M = K = N = 64


class CubeVecCast(PTOTestCase):
    """matmul -> cast(BF16) -> cast(FP32): cube->vec only, no vec->cube."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "chained_cubevec_cast"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.BF16, init_value=torch.randn),
            TensorSpec("w", [K, N], DataType.BF16, init_value=torch.randn),
            TensorSpec("y", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                w: pl.Tensor[[K, N], pl.BF16],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.NONE):
                    t1 = pl.matmul(a, w, out_dtype=pl.FP32)
                    t1b = pl.cast(t1, target_type=pl.BF16, mode="rint")
                    y = pl.cast(t1b, target_type=pl.FP32)
                return y

        return Prog

    def compute_expected(self, tensors, params=None):
        tensors["y"][:] = torch.matmul(tensors["a"].to(torch.float32), tensors["w"].to(torch.float32))


class ChainedMatmul(PTOTestCase):
    """matmul -> cast(BF16) -> matmul: cube->vec->cube full repro."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "chained_matmul_cast_matmul"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.BF16, init_value=torch.randn),
            TensorSpec("w", [K, N], DataType.BF16, init_value=torch.randn),
            TensorSpec("w2", [K, N], DataType.BF16, init_value=torch.randn),
            TensorSpec("y", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                w: pl.Tensor[[K, N], pl.BF16],
                w2: pl.Tensor[[K, N], pl.BF16],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.NONE):
                    t1 = pl.matmul(a, w, out_dtype=pl.FP32)
                    t1b = pl.cast(t1, target_type=pl.BF16, mode="rint")
                    y = pl.matmul(t1b, w2, out_dtype=pl.FP32, b_trans=True)
                return y

        return Prog

    def compute_expected(self, tensors, params=None):
        t1 = torch.matmul(tensors["a"].to(torch.float32), tensors["w"].to(torch.float32))
        t1b = t1.to(torch.bfloat16).to(torch.float32)
        tensors["y"][:] = torch.matmul(t1b, tensors["w2"].to(torch.float32).t())


class TestChainedMatmulCast:
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_cube_vec_cast(self, test_runner, platform):
        """cube->vec only: should pass; failure means cube->vec drops data."""
        result = test_runner.run(CubeVecCast(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_chained_matmul(self, test_runner, platform):
        """cube->vec->cube full repro (issue #1525)."""
        result = test_runner.run(ChainedMatmul(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
