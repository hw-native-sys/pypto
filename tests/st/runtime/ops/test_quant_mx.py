# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board A5 runtime tests for MX quantization and affine dequantization."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

ROWS, COLS = 16, 64
MX_GROUP_SIZE = 32


def _quant_inputs_and_golden() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    group_exponents = (torch.arange(ROWS * COLS // MX_GROUP_SIZE) % 4 - 1).reshape(ROWS, 2)
    scales = torch.pow(2.0, group_exponents).to(torch.float32)
    src = (fp8_values.reshape(1, 1, MX_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(ROWS, COLS)
    quantized = fp8_values.to(torch.float8_e4m3fn).repeat(ROWS, 2)
    scale = (group_exponents + 127).to(torch.uint8).reshape(1, -1).view(torch.float8_e8m0fnu)
    return src.contiguous(), quantized.contiguous(), scale.contiguous()


def _tdequant_src() -> torch.Tensor:
    return (torch.arange(ROWS * COLS).reshape(ROWS, COLS) % 31 - 15).to(torch.int8)


def _tdequant_scale() -> torch.Tensor:
    return torch.pow(2.0, (torch.arange(ROWS) % 4 - 2).to(torch.float32)).reshape(ROWS, 1)


def _tdequant_offset() -> torch.Tensor:
    return (torch.arange(ROWS) % 5 - 2).to(torch.float32).reshape(ROWS, 1)


class TestQuantMx(PTOTestCase):
    """Quantize FP32 blocks through the public packed-A MX interface."""

    __test__ = False

    def __init__(self, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=0, atol=0), platform=platform)

    def get_name(self) -> str:
        return "quant_mx_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        src, expected_quant, expected_scale = _quant_inputs_and_golden()
        return [
            TensorSpec("src", [ROWS, COLS], DataType.FP32, init_value=src),
            TensorSpec("out_quant", [ROWS, COLS], DataType.FP8E4M3FN, is_output=True),
            TensorSpec(
                "out_scale",
                [1, ROWS * COLS // MX_GROUP_SIZE],
                DataType.FP8E8M0,
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        @pl.program
        class QuantMxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def quant_mx(
                self,
                src: pl.Tensor[[ROWS, COLS], pl.FP32],
                out_quant: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP8E4M3FN]],
                out_scale: pl.Out[pl.Tensor[[1, ROWS * COLS // MX_GROUP_SIZE], pl.FP8E8M0]],
            ):
                src_tile = pl.load(src, [0, 0], [ROWS, COLS])
                quantized, scale = pl.quant_mx(src_tile, layout=pl.MX_A_ZZ)
                out_quant = pl.store(quantized, [0, 0], out_quant)
                out_scale = pl.store(scale, [0, 0], out_scale)
                return out_quant, out_scale

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[ROWS, COLS], pl.FP32],
                out_quant: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP8E4M3FN]],
                out_scale: pl.Out[pl.Tensor[[1, ROWS * COLS // MX_GROUP_SIZE], pl.FP8E8M0]],
            ):
                out_quant, out_scale = self.quant_mx(src, out_quant, out_scale)
                return out_quant, out_scale

        return QuantMxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _src, expected_quant, expected_scale = _quant_inputs_and_golden()
        tensors["out_quant"][:] = expected_quant
        tensors["out_scale"][:] = expected_scale


class TestTDequant(PTOTestCase):
    """Apply per-row affine dequantization to an INT8 tile."""

    __test__ = False

    def __init__(self, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=0, atol=0), platform=platform)

    def get_name(self) -> str:
        return "tdequant_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [ROWS, COLS], DataType.INT8, init_value=_tdequant_src),
            TensorSpec("scale", [ROWS, 1], DataType.FP32, init_value=_tdequant_scale),
            TensorSpec("offset", [ROWS, 1], DataType.FP32, init_value=_tdequant_offset),
            TensorSpec("out", [ROWS, COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TDequantProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tdequant(
                self,
                src: pl.Tensor[[ROWS, COLS], pl.INT8],
                scale: pl.Tensor[[ROWS, 1], pl.FP32],
                offset: pl.Tensor[[ROWS, 1], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ):
                src_tile = pl.load(src, [0, 0], [ROWS, COLS])
                scale_tile = pl.load(scale, [0, 0], [ROWS, 1])
                offset_tile = pl.load(offset, [0, 0], [ROWS, 1])
                result = pl.tdequant(src_tile, scale_tile, offset_tile)
                out = pl.store(result, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[ROWS, COLS], pl.INT8],
                scale: pl.Tensor[[ROWS, 1], pl.FP32],
                offset: pl.Tensor[[ROWS, 1], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ):
                out = self.tdequant(src, scale, offset, out)
                return out

        return TDequantProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = (tensors["src"].to(torch.float32) - tensors["offset"]) * tensors["scale"]


@pytest.mark.platforms("a5")
class TestQuantMxOperations:
    """Numerical execution coverage for the Ascend950-only quantization ops."""

    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_quant_mx(self, test_runner, platform):
        result = test_runner.run(TestQuantMx(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_tdequant(self, test_runner, platform):
        result = test_runner.run(TestTDequant(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
