# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 numerical coverage for staged and mixed-kernel MX quantization and matmul."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

M, K, N = 64, 256, 64
GROUPS = K // 32


def _exact_scale_source(rows: int) -> torch.Tensor:
    """Build values that are represented exactly by MX exponent scales."""
    source = torch.zeros((rows, K), dtype=torch.float32)
    for row in range(rows):
        box = row // 16
        row_in_box = row % 16
        for group in range(GROUPS):
            code = 100 + (box * 13 + row_in_box * 7 + group * 3) % 64
            source[row, group * 32 : (group + 1) * 32] = float(2.0 ** (code - 127))
    return source


def _mx_matmul_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = _exact_scale_source(M)
    b = torch.zeros((K, N), dtype=torch.float32)
    for group in range(GROUPS):
        b[group * 32 : (group + 1) * 32, group] = 1.0
    unit_scale = torch.full((1, GROUPS * N), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return a, b.to(torch.float8_e4m3fn), unit_scale


def _compute_expected(tensors, params=None):
    tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"].to(torch.float32))


def _mx_rhs_matmul_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct independent lhs and exact-scale RHS inputs for B-side quantization."""
    b_source = _exact_scale_source(N)
    lhs = torch.zeros((M, K), dtype=torch.float32)
    for group in range(min(M, GROUPS)):
        lhs[group, group * 32 : (group + 1) * 32] = 1.0
    lhs_scale = torch.full((1, M * GROUPS), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return lhs.to(torch.float8_e4m3fn), lhs_scale, b_source


def _compute_expected_rhs(tensors, params=None):
    tensors["out"][:] = torch.matmul(tensors["a"].to(torch.float32), tensors["b"].transpose(0, 1))


@pl.jit.incore
def staged_vector_quantize(
    a: pl.Tensor[[M, K], pl.FP32],
    a_quant: pl.Out[pl.Tensor[[M, K], pl.FP8E4M3FN]],
    a_scale: pl.Out[pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]],
) -> tuple[pl.Tensor[[M, K], pl.FP8E4M3FN], pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]]:
    """Run quant_mx on AIV and store its outputs in GM."""
    quant, scale = pl.quant_mx(pl.load(a, [0, 0], [M, K]), group_axis=1)
    a_quant = pl.store(quant, [0, 0], a_quant)
    a_scale = pl.store(pl.reshape(scale, [1, M * GROUPS]), [0, 0], a_scale)
    return a_quant, a_scale


@pl.jit.incore
def staged_cube_matmul(
    a_quant: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[1, M * GROUPS], pl.FP8E8M0],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """Run matmul_mx on AIC using the GM-staged quantization outputs."""
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


@pl.jit
def staged_quant_matmul_mx(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
    a_quant: pl.Out[pl.Tensor[[M, K], pl.FP8E4M3FN]],
    a_scale: pl.Out[pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """Run quant_mx on AIV and matmul_mx on AIC with GM staging."""
    a_quant, a_scale = staged_vector_quantize(a, a_quant, a_scale)
    return staged_cube_matmul(a_quant, a_scale, b, b_scale, out)


class GmCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform=None):
        super().__init__(RunConfig(rtol=1e-3, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_gm_64x256x64"

    def get_program(self) -> Any:
        return staged_quant_matmul_mx.specialize()

    def define_tensors(self) -> list[TensorSpec]:
        a, b, b_scale = _mx_matmul_inputs()
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=a),
            TensorSpec("b", [K, N], DataType.FP8E4M3FN, init_value=b),
            TensorSpec("b_scale", [1, GROUPS * N], DataType.FP8E8M0, init_value=b_scale),
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


@pl.jit.incore
def mixed_quant_matmul_mx_kernel(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """Carry quantized A data and its scale directly from AIV to AIC."""
    quant, scale = pl.quant_mx(pl.load(a, [0, 0], [M, K]), group_axis=1)
    lhs = pl.move(
        pl.move(
            quant,
            target_memory=pl.Mem.Mat,
            blayout=pl.TileLayout.col_major,
            slayout=pl.TileLayout.row_major,
        ),
        target_memory=pl.Mem.Left,
    )
    lhs_scale = pl.move(
        pl.move(
            scale,
            target_memory=pl.Mem.Mat,
            blayout=pl.TileLayout.row_major,
            slayout=pl.TileLayout.row_major,
        ),
        target_memory=pl.Mem.LeftScale,
    )

    b_scale_mx = pl.tensor.view(b_scale, [GROUPS, N], layout=pl.MX_B_NN)
    rhs = pl.move(
        pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Right,
    )
    rhs_scale = pl.move(
        pl.load(b_scale_mx, [0, 0], [GROUPS, N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.RightScale,
    )
    out = pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)
    return out


@pl.jit
def mixed_quant_matmul_mx(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    return mixed_quant_matmul_mx_kernel(a, b, b_scale, out)


class MixCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform=None):
        super().__init__(RunConfig(rtol=1e-3, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_mix_64x256x64"

    def get_program(self) -> Any:
        return mixed_quant_matmul_mx.specialize()

    def define_tensors(self) -> list[TensorSpec]:
        scale_boxes = (M // 16, GROUPS // 2)
        assert scale_boxes == (4, 4)
        a, b, b_scale = _mx_matmul_inputs()
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=a),
            TensorSpec("b", [K, N], DataType.FP8E4M3FN, init_value=b),
            TensorSpec("b_scale", [1, GROUPS * N], DataType.FP8E8M0, init_value=b_scale),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    compute_expected = staticmethod(_compute_expected)


@pl.jit.incore
def mixed_quant_rhs_matmul_mx_kernel(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[1, M * GROUPS], pl.FP8E8M0],
    b: pl.Tensor[[N, K], pl.FP32],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """Carry group_axis=0 quantized B data and col/col scale from AIV to AIC."""
    quant, scale = pl.quant_mx(pl.load(b, [0, 0], [N, K]), group_axis=0)
    rhs = pl.move(
        pl.move(
            quant,
            target_memory=pl.Mem.Mat,
            blayout=pl.TileLayout.col_major,
            slayout=pl.TileLayout.row_major,
        ),
        target_memory=pl.Mem.Right,
    )
    rhs_scale = pl.move(
        pl.move(
            scale,
            target_memory=pl.Mem.Mat,
            blayout=pl.TileLayout.col_major,
            slayout=pl.TileLayout.col_major,
        ),
        target_memory=pl.Mem.RightScale,
    )

    a_scale_mx = pl.tensor.view(a_scale, [M, GROUPS], layout=pl.MX_A_ZZ)
    lhs = pl.move(
        pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Left,
    )
    lhs_scale = pl.move(
        pl.load(a_scale_mx, [0, 0], [M, GROUPS], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.LeftScale,
    )
    return pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)


@pl.jit
def mixed_quant_rhs_matmul_mx(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[1, M * GROUPS], pl.FP8E8M0],
    b: pl.Tensor[[N, K], pl.FP32],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    return mixed_quant_rhs_matmul_mx_kernel(a, a_scale, b, out)


class MixRhsCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform=None):
        super().__init__(RunConfig(rtol=1e-3, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_mix_rhs_64x256x64"

    def get_program(self) -> Any:
        return mixed_quant_rhs_matmul_mx.specialize()

    def define_tensors(self) -> list[TensorSpec]:
        a, a_scale, b = _mx_rhs_matmul_inputs()
        return [
            TensorSpec("a", [M, K], DataType.FP8E4M3FN, init_value=a),
            TensorSpec("a_scale", [1, M * GROUPS], DataType.FP8E8M0, init_value=a_scale),
            TensorSpec("b", [N, K], DataType.FP32, init_value=b),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    compute_expected = staticmethod(_compute_expected_rhs)


@pytest.mark.platforms("a5")
class TestQuantizedMatmulMx:
    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_mix(self, test_runner, platform):
        result = test_runner.run(MixCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_mix_rhs(self, test_runner, platform):
        result = test_runner.run(MixRhsCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_gm(self, test_runner, platform):
        result = test_runner.run(GmCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
