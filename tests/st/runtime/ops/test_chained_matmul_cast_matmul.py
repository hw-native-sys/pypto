# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device test: chained matmul -> cast(bf16) -> matmul fused in one CORE_GROUP scope.

The chained fusion (``SplitMode.NONE``) runs correctly on a2a3 as long as the
output is committed idiomatically (writing into the ``pl.Out`` buffer via
``pl.assemble``), rather than via a bare reassignment ``y = pl.matmul(...)`` that
rebinds the name to a detached value. The bare form is now rejected at compile
time by the OutParamNotShadowed verifier (see issue #1525), so this device test
exercises the supported idiom end to end.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

M = K = N = 64


class ChainedMatmulCastMatmulTest(PTOTestCase):
    """matmul -> cast(bf16) -> matmul(b_trans) chained in one CORE_GROUP."""

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
        class Repro:
            @pl.function(type=pl.FunctionType.Opaque)
            def fused(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                w: pl.Tensor[[K, N], pl.BF16],
                w2: pl.Tensor[[K, N], pl.BF16],
                y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.NONE)]):
                    t1 = pl.matmul(a, w, out_dtype=pl.FP32)
                    t1b = pl.cast(t1, target_type=pl.BF16, mode="rint")
                    t2 = pl.matmul(t1b, w2, out_dtype=pl.FP32, b_trans=True)
                    # Commit into the pl.Out buffer idiomatically (do NOT bare-reassign
                    # `y = pl.matmul(...)` — that detaches y and silently zeros, see #1525).
                    y = pl.assemble(y, t2, [0, 0])
                return y

        return Repro

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].float()
        w = tensors["w"].float()
        w2 = tensors["w2"].float()
        t1 = a @ w
        t1b = t1.bfloat16().float()
        tensors["y"][:] = t1b @ w2.t()


class TestChainedMatmulCastMatmul:
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_chained_matmul_cast_matmul(self, test_runner, platform):
        cfg = RunConfig(platform=platform, rtol=1e-2, atol=1e-2)
        result = test_runner.run(ChainedMatmulCastMatmulTest(platform=platform, config=cfg))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
