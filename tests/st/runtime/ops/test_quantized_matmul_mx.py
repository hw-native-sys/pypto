# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 quant_mx-to-matmul_mx transfer through GM between two kernels."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

M, K, N = 64, 256, 64
GROUPS = K // 32


def _input_tensor_specs() -> list[TensorSpec]:
    a = torch.zeros((M, K), dtype=torch.float32)
    for row in range(M):
        box = row // 16
        row_in_box = row % 16
        for group in range(GROUPS):
            code = 100 + (box * 13 + row_in_box * 7 + group * 3) % 64
            a[row, group * 32 : (group + 1) * 32] = float(2.0 ** (code - 127))
    b = torch.zeros((K, N), dtype=torch.float32)
    for group in range(GROUPS):
        b[group * 32 : (group + 1) * 32, group] = 1.0
    unit_scale = torch.full((1, GROUPS * N), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return [
        TensorSpec("a", [M, K], DataType.FP32, init_value=a),
        TensorSpec("b", [K, N], DataType.FP8E4M3FN, init_value=b.to(torch.float8_e4m3fn)),
        TensorSpec("b_scale", [1, GROUPS * N], DataType.FP8E8M0, init_value=unit_scale),
    ]


def _compute_expected(tensors, params=None):
    tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"].to(torch.float32))


# ======================================================================
# AIV writes GM and AIC reads GM without cross-core pipes
# ======================================================================


@pl.program
class GmProgram:
    @pl.function(type=pl.FunctionType.AIV)
    def vector_quantize(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        a_quant: pl.Out[pl.Tensor[[M, K], pl.FP8E4M3FN]],
        a_scale: pl.Out[pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]],
    ) -> tuple[pl.Tensor[[M, K], pl.FP8E4M3FN], pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]]:
        quant, scale = pl.quant_mx(pl.load(a, [0, 0], [M, K]), layout=pl.MX_A_ZZ)
        a_quant = pl.store(quant, [0, 0], a_quant)
        a_scale = pl.store(pl.reshape(scale, [1, M * GROUPS]), [0, 0], a_scale)
        return a_quant, a_scale

    @pl.function(type=pl.FunctionType.AIC)
    def cube_matmul(
        self,
        a_quant: pl.Tensor[[M, K], pl.FP8E4M3FN],
        a_scale: pl.Tensor[[1, M * GROUPS], pl.FP8E8M0],
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        a_scale_mx = pl.tensor.view(a_scale, [M, GROUPS], layout=pl.MX_A_ZZ)
        lhs = pl.move(
            pl.load(a_quant, [0, 0], [M, K], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Left,
        )
        lhs_scale = pl.move(
            pl.load(a_scale_mx, [0, 0], [M, GROUPS], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.LeftScale,
        )
        b_scale_mx = pl.tensor.view(b_scale, [GROUPS, N], layout=pl.MX_B_NN)
        rhs = pl.move(pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right)
        rhs_scale = pl.move(
            pl.load(b_scale_mx, [0, 0], [GROUPS, N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.RightScale,
        )
        return pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        a_quant: pl.Out[pl.Tensor[[M, K], pl.FP8E4M3FN]],
        a_scale: pl.Out[pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        a_quant, a_scale = self.vector_quantize(a, a_quant, a_scale)
        return self.cube_matmul(a_quant, a_scale, b, b_scale, out)


class GmCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform=None):
        super().__init__(RunConfig(rtol=1e-3, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_gm_64x256x64"

    def get_program(self) -> Any:
        return GmProgram

    def define_tensors(self) -> list[TensorSpec]:
        return [
            *_input_tensor_specs(),
            TensorSpec(
                "a_quant",
                [M, K],
                DataType.FP8E4M3FN,
                init_value=torch.zeros((M, K), dtype=torch.float8_e4m3fn),
            ),
            TensorSpec(
                "a_scale",
                [1, M * GROUPS],
                DataType.FP8E8M0,
                init_value=torch.zeros((1, M * GROUPS), dtype=torch.uint8).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    compute_expected = staticmethod(_compute_expected)


# ======================================================================
# Supported GM-staged entry point
# ======================================================================


@pytest.mark.platforms("a5")
class TestQuantizedMatmulMx:
    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_gm(self, test_runner, platform):
        result = test_runner.run(GmCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
