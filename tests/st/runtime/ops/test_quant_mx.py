# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board A5 runtime tests for MX quantization."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

MX_GROUP_SIZE = 32


def _case_shapes(group_axis: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((16, 64), (16, 64)) if group_axis == 1 else ((32, 64), (64, 32))


def _pack_scale_codes(codes: torch.Tensor, group_axis: int) -> torch.Tensor:
    """Pack logical scale codes into the public A-ZZ or B-NN byte order."""
    if group_axis == 1:
        rows, groups = codes.shape
        packed = codes.reshape(rows // 16, 16, groups // 2, 2).permute(0, 2, 1, 3)
    else:
        groups, cols = codes.shape
        packed = codes.reshape(groups // 2, 2, cols // 16, 16).permute(2, 0, 3, 1)
    return packed.contiguous().reshape(1, -1).view(torch.float8_e8m0fnu)


def _quant_inputs_and_golden(group_axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact MXFP8 blocks with known quantized values and E8M0 scales."""
    fp8_values = torch.tensor(
        [
            -448,
            -256,
            -128,
            -64,
            -32,
            -16,
            -8,
            -4,
            -2,
            -1,
            -0.5,
            -0.25,
            -0.125,
            -0.0625,
            -0.015625,
            0,
            0.015625,
            0.0625,
            0.125,
            0.25,
            0.5,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            448,
            0,
        ],
        dtype=torch.float32,
    )
    src_shape, quant_shape = _case_shapes(group_axis)
    groups = src_shape[1] // MX_GROUP_SIZE
    exponent_shape = (src_shape[0], groups) if group_axis == 1 else (groups, src_shape[0])
    group_exponents = (torch.arange(exponent_shape[0] * exponent_shape[1]) % 4 - 1).reshape(exponent_shape)
    scales = torch.pow(2.0, group_exponents).to(torch.float32)
    if group_axis == 1:
        src = (fp8_values.reshape(1, 1, MX_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(src_shape)
        quantized = fp8_values.to(torch.float8_e4m3fn).repeat(src_shape[0], groups)
    else:
        src_kn = (fp8_values.reshape(1, MX_GROUP_SIZE, 1) * scales.unsqueeze(1)).reshape(quant_shape)
        src = src_kn.transpose(0, 1)
        quantized = (
            fp8_values.to(torch.float8_e4m3fn)
            .reshape(1, MX_GROUP_SIZE, 1)
            .expand(groups, MX_GROUP_SIZE, src_shape[0])
            .reshape(quant_shape)
        )
    scale = _pack_scale_codes((group_exponents + 127).to(torch.uint8), group_axis)
    return src.contiguous(), quantized.contiguous(), scale.contiguous()


class TestQuantMx(PTOTestCase):
    """Quantize FP32 blocks through either public MX grouping path."""

    __test__ = False

    def __init__(self, group_axis: int, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=0, atol=0), platform=platform)
        self.group_axis = group_axis
        self.src_shape, self.quant_shape = _case_shapes(group_axis)
        self.scale_elements = self.src_shape[0] * self.src_shape[1] // MX_GROUP_SIZE

    def get_name(self) -> str:
        return f"quant_mx_axis{self.group_axis}_{self.src_shape[0]}x{self.src_shape[1]}"

    def define_tensors(self) -> list[TensorSpec]:
        src, _expected_quant, _expected_scale = _quant_inputs_and_golden(self.group_axis)
        return [
            TensorSpec("src", list(self.src_shape), DataType.FP32, init_value=src),
            TensorSpec("out_quant", list(self.quant_shape), DataType.FP8E4M3FN, is_output=True),
            TensorSpec(
                "out_scale",
                [1, self.scale_elements],
                DataType.FP8E8M0,
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        src_rows, src_cols = self.src_shape
        quant_rows, quant_cols = self.quant_shape
        scale_elements = self.scale_elements
        group_axis = self.group_axis

        @pl.program
        class QuantMxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def quant_mx(
                self,
                src: pl.Tensor[[src_rows, src_cols], pl.FP32],
                out_quant: pl.Out[pl.Tensor[[quant_rows, quant_cols], pl.FP8E4M3FN]],
                out_scale: pl.Out[pl.Tensor[[1, scale_elements], pl.FP8E8M0]],
            ):
                src_tile = pl.load(src, [0, 0], [src_rows, src_cols])
                quantized, scale = pl.quant_mx(src_tile, group_axis=group_axis)
                out_quant = pl.store(quantized, [0, 0], out_quant)
                out_scale = pl.store(pl.reshape(scale, [1, scale_elements]), [0, 0], out_scale)
                return out_quant, out_scale

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[src_rows, src_cols], pl.FP32],
                out_quant: pl.Out[pl.Tensor[[quant_rows, quant_cols], pl.FP8E4M3FN]],
                out_scale: pl.Out[pl.Tensor[[1, scale_elements], pl.FP8E8M0]],
            ):
                out_quant, out_scale = self.quant_mx(src, out_quant, out_scale)
                return out_quant, out_scale

        return QuantMxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _src, expected_quant, expected_scale = _quant_inputs_and_golden(self.group_axis)
        tensors["out_quant"][:] = expected_quant
        tensors["out_scale"][:] = expected_scale


@pytest.mark.platforms("a5")
class TestQuantMxOperations:
    """Numerical execution coverage for the Ascend950-only quantization ops."""

    @pytest.mark.parametrize(
        ("platform", "group_axis"),
        [pytest.param("a5", 1, id="a5-axis1"), pytest.param("a5", 0, id="a5-axis0")],
    )
    def test_quant_mx(self, test_runner, platform, group_axis):
        result = test_runner.run(TestQuantMx(group_axis, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
