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

ROWS, COLS = 16, 64
MX_GROUP_SIZE = 32
FP4_B_ROWS = 64


def _mxfp4_source(layout: pl.TensorLayout, dtype: torch.dtype) -> torch.Tensor:
    """Build FP16/BF16 values that are exactly representable as scaled E2M1."""
    e2m1_values = torch.tensor([0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
    pattern = e2m1_values.repeat((MX_GROUP_SIZE + e2m1_values.numel() - 1) // e2m1_values.numel())[
        :MX_GROUP_SIZE
    ]
    if layout == pl.MX_A_ZZ:
        exponents = (torch.arange(ROWS * (COLS // MX_GROUP_SIZE)) % 4 - 2).reshape(
            ROWS, COLS // MX_GROUP_SIZE
        )
        scales = torch.pow(2.0, exponents).to(torch.float32)
        return (pattern.reshape(1, 1, MX_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(ROWS, COLS).to(dtype)

    exponents = (torch.arange((COLS // MX_GROUP_SIZE) * FP4_B_ROWS) % 4 - 2).reshape(
        COLS // MX_GROUP_SIZE, FP4_B_ROWS
    )
    scales = torch.pow(2.0, exponents).to(torch.float32)
    logical_kn = pattern.reshape(1, MX_GROUP_SIZE, 1) * scales.unsqueeze(1)
    return logical_kn.reshape(COLS, FP4_B_ROWS).transpose(0, 1).contiguous().to(dtype)


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


class TestQuantMx(PTOTestCase):
    """Quantize FP32 blocks through the public packed-A MX interface."""

    __test__ = False

    def __init__(self, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=0, atol=0), platform=platform)

    def get_name(self) -> str:
        return "quant_mx_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        src, _expected_quant, _expected_scale = _quant_inputs_and_golden()
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
                out_scale = pl.store(pl.reshape(scale, [1, ROWS * COLS // MX_GROUP_SIZE]), [0, 0], out_scale)
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


class TestQuantMxFp4(PTOTestCase):
    """Quantize FP16/BF16 through native MXFP4 A/B layouts."""

    __test__ = False

    def __init__(self, layout: pl.TensorLayout, source_dtype: DataType, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=0, atol=0), platform=platform)
        self.layout = layout
        self.source_dtype = source_dtype

    def get_name(self) -> str:
        layout_name = "a_zz" if self.layout == pl.MX_A_ZZ else "b_nn"
        return f"quant_mxfp4_{layout_name}_{self.source_dtype.value}"

    def define_tensors(self) -> list[TensorSpec]:
        src_rows = ROWS if self.layout == pl.MX_A_ZZ else FP4_B_ROWS
        out_rows = ROWS if self.layout == pl.MX_A_ZZ else COLS
        out_cols = COLS if self.layout == pl.MX_A_ZZ else FP4_B_ROWS
        groups = src_rows * COLS // MX_GROUP_SIZE
        src = _mxfp4_source(self.layout, self.source_dtype.torch_dtype)
        return [
            TensorSpec("src", [src_rows, COLS], self.source_dtype, init_value=src),
            TensorSpec("out_quant", [out_rows, out_cols // 2], DataType.FP4, is_output=True),
            TensorSpec("out_scale", [1, groups], DataType.FP8E8M0, is_output=True),
        ]

    def get_program(self) -> Any:
        layout = self.layout
        src_dtype = pl.FP16 if self.source_dtype == DataType.FP16 else pl.BF16
        src_rows = ROWS if layout == pl.MX_A_ZZ else FP4_B_ROWS
        out_rows = ROWS if layout == pl.MX_A_ZZ else COLS
        out_cols = COLS if layout == pl.MX_A_ZZ else FP4_B_ROWS
        groups = src_rows * COLS // MX_GROUP_SIZE

        @pl.program
        class QuantMxFp4Program:
            @pl.function(type=pl.FunctionType.InCore)
            def quant_mx_fp4(
                self,
                src: pl.Tensor[[src_rows, COLS], src_dtype],
                out_quant: pl.Out[pl.Tensor[[out_rows, out_cols], pl.FP4]],
                out_scale: pl.Out[pl.Tensor[[1, groups], pl.FP8E8M0]],
            ):
                quantized, scale = pl.quant_mx(
                    pl.load(src, [0, 0], [src_rows, COLS]), layout=layout, dtype=pl.FP4
                )
                out_quant = pl.store(quantized, [0, 0], out_quant)
                out_scale = pl.store(pl.reshape(scale, [1, groups]), [0, 0], out_scale)
                return out_quant, out_scale

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[src_rows, COLS], src_dtype],
                out_quant: pl.Out[pl.Tensor[[out_rows, out_cols], pl.FP4]],
                out_scale: pl.Out[pl.Tensor[[1, groups], pl.FP8E8M0]],
            ):
                return self.quant_mx_fp4(src, out_quant, out_scale)

        return QuantMxFp4Program

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        # Reconstruct the logical quantization result from the exact source.
        # A source is [M,K]; B source is [N,K] and quant_mx returns [K,N].
        src = tensors["src"].to(torch.float32)
        logical = src if src.shape[0] == ROWS else src.transpose(0, 1).contiguous()
        is_a = logical.shape[0] == ROWS
        if is_a:
            grouped = logical.reshape(ROWS, COLS // MX_GROUP_SIZE, MX_GROUP_SIZE)
            max_abs = grouped.abs().amax(dim=2)
            scale_exp = torch.log2(max_abs / 6.0).round().to(torch.int64)
            normalized = grouped / torch.pow(2.0, scale_exp).to(torch.float32).unsqueeze(-1)
            normalized = normalized.reshape(ROWS, COLS)
        else:
            grouped = logical.reshape(COLS // MX_GROUP_SIZE, MX_GROUP_SIZE, FP4_B_ROWS)
            max_abs = grouped.abs().amax(dim=1)
            scale_exp = torch.log2(max_abs / 6.0).round().to(torch.int64)
            normalized = grouped / torch.pow(2.0, scale_exp).to(torch.float32).unsqueeze(1)
            normalized = normalized.reshape(COLS, FP4_B_ROWS)

        e2m1_values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
        )
        codes = (normalized.unsqueeze(-1) == e2m1_values).to(torch.uint8).argmax(dim=-1).to(torch.uint8)
        packed = ((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)
        tensors["out_quant"].view(torch.uint8)[:] = packed

        scale_codes = (scale_exp + 127).to(torch.uint8)
        if is_a:
            packed_scale = scale_codes.reshape(ROWS // 16, 16, COLS // 64, 2).permute(0, 2, 1, 3).contiguous()
        else:
            packed_scale = (
                scale_codes.reshape(COLS // 64, 2, FP4_B_ROWS // 16, 16).permute(2, 0, 3, 1).contiguous()
            )
        tensors["out_scale"].view(torch.uint8)[:] = packed_scale.reshape(1, -1)


@pytest.mark.platforms("a5")
class TestQuantMxOperations:
    """Numerical execution coverage for the Ascend950-only quantization ops."""

    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_quant_mx(self, test_runner, platform):
        result = test_runner.run(TestQuantMx(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("source_dtype", [DataType.FP16, DataType.BF16])
    @pytest.mark.parametrize("layout", [pl.MX_A_ZZ, pl.MX_B_NN])
    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_quant_mxfp4(self, test_runner, platform, layout, source_dtype):
        result = test_runner.run(TestQuantMxFp4(layout, source_dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
