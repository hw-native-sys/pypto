# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device coverage for whole-function Ascend910B vector AutoTile.

Every program marks one fixed tensor DAG with ``attrs={"auto_tile": True}``.
The compiler must either realize that complete DAG as one SPMD/AIV kernel or
fail; these positive cases therefore check both the all-or-nothing admission
contract and numerical execution of each supported schedule family.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_FP16_TOL = RunConfig(rtol=1e-2, atol=1e-2)
_BF16_TOL = RunConfig(rtol=2e-2, atol=2e-2)
_RSQRT_TOL = RunConfig(rtol=1e-2, atol=1e-2)
_NORM_TOL = RunConfig(rtol=1e-2, atol=1e-2)


class _AutoTileCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)


@pl.program
class RaggedPointwiseProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[130, 66], pl.FP32]) -> pl.Tensor[[130, 66], pl.FP32]:
        absolute: pl.Tensor[[130, 66], pl.FP32] = pl.abs(x)
        out: pl.Tensor[[130, 66], pl.FP32] = pl.add(absolute, x)
        return out


class RaggedPointwiseCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_ragged_pointwise"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [130, 66], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [130, 66], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RaggedPointwiseProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.abs(tensors["x"]) + tensors["x"]


@pl.program
class MultiOutputProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self, x: pl.Tensor[[128, 8192], pl.FP32]
    ) -> tuple[pl.Tensor[[128, 8192], pl.FP32], pl.Tensor[[128, 8192], pl.FP32]]:
        exponent: pl.Tensor[[128, 8192], pl.FP32] = pl.exp(x)
        out: pl.Tensor[[128, 8192], pl.FP32] = pl.add(exponent, 1.0)
        return exponent, out


class MultiOutputCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_multi_output"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [128, 8192], DataType.FP32, init_value=torch.randn),
            TensorSpec("exponent", [128, 8192], DataType.FP32, is_output=True),
            TensorSpec("out", [128, 8192], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MultiOutputProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["exponent"][:] = torch.exp(tensors["x"])
        tensors["out"][:] = tensors["exponent"] + 1.0


@pl.program
class SoftmaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[32, 8192], pl.FP32]) -> pl.Tensor[[32, 8192], pl.FP32]:
        maximum: pl.Tensor[[32, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[32, 8192], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[32, 8192], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[32, 1], pl.FP32] = pl.row_sum(exponent)
        out: pl.Tensor[[32, 8192], pl.FP32] = pl.div(exponent, total)
        return out


class SoftmaxCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_softmax"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [32, 8192], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [32, 8192], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SoftmaxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["x"], dim=1)


@pl.program
class CapacityFitSoftmaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[512, 256], pl.FP32]) -> pl.Tensor[[512, 256], pl.FP32]:
        maximum: pl.Tensor[[512, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[512, 256], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[512, 256], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[512, 1], pl.FP32] = pl.row_sum(exponent)
        out: pl.Tensor[[512, 256], pl.FP32] = pl.div(exponent, total)
        return out


class CapacityFitSoftmaxCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_softmax_capacity_fit"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [512, 256], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [512, 256], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return CapacityFitSoftmaxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["x"], dim=1)


@pl.program
class IntermediateSoftmaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[128, 1024], pl.FP32]) -> pl.Tensor[[128, 1024], pl.FP32]:
        maximum: pl.Tensor[[128, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[128, 1024], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[128, 1024], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(exponent)
        out: pl.Tensor[[128, 1024], pl.FP32] = pl.div(exponent, total)
        return out


class IntermediateSoftmaxCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_softmax_intermediate"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [128, 1024], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [128, 1024], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return IntermediateSoftmaxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["x"], dim=1)


@pl.program
class RaggedSoftmaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[130, 272], pl.FP32]) -> pl.Tensor[[130, 272], pl.FP32]:
        maximum: pl.Tensor[[130, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[130, 272], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[130, 272], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[130, 1], pl.FP32] = pl.row_sum(exponent)
        out: pl.Tensor[[130, 272], pl.FP32] = pl.div(exponent, total)
        return out


class RaggedSoftmaxCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_softmax_ragged"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [130, 272], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [130, 272], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RaggedSoftmaxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.softmax(tensors["x"], dim=1)


@pl.program
class RmsProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.FP32]:
        square: pl.Tensor[[128, 8192], pl.FP32] = pl.mul(x, x)
        total: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(square)
        mean: pl.Tensor[[128, 1], pl.FP32] = pl.mul(total, 1.0 / 8192.0)
        inverse: pl.Tensor[[128, 1], pl.FP32] = pl.rsqrt(pl.add(mean, 1.0e-6))
        out: pl.Tensor[[128, 8192], pl.FP32] = pl.mul(x, inverse)
        return out


class RmsCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_rms"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [128, 8192], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [128, 8192], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RmsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        tensors["out"][:] = x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1.0e-6)


@pl.program
class LayerNormProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[512, 256], pl.FP32],
        gamma: pl.Tensor[[1, 256], pl.FP32],
        beta: pl.Tensor[[1, 256], pl.FP32],
    ) -> pl.Tensor[[512, 256], pl.FP32]:
        scaled_x: pl.Tensor[[512, 256], pl.FP32] = pl.mul(x, 1.0 / 256.0)
        mean: pl.Tensor[[512, 1], pl.FP32] = pl.row_sum(scaled_x)
        centered: pl.Tensor[[512, 256], pl.FP32] = pl.sub(x, mean)
        squared: pl.Tensor[[512, 256], pl.FP32] = pl.mul(centered, centered)
        square_sum: pl.Tensor[[512, 1], pl.FP32] = pl.row_sum(squared)
        variance: pl.Tensor[[512, 1], pl.FP32] = pl.mul(square_sum, 1.0 / 256.0)
        std: pl.Tensor[[512, 1], pl.FP32] = pl.sqrt(pl.add(variance, 1.0e-5))
        normalized: pl.Tensor[[512, 256], pl.FP32] = pl.div(centered, std)
        scaled: pl.Tensor[[512, 256], pl.FP32] = pl.mul(normalized, gamma)
        out: pl.Tensor[[512, 256], pl.FP32] = pl.add(scaled, beta)
        return out


class LayerNormCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_layer_norm"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [512, 256], DataType.FP32, init_value=torch.randn),
            TensorSpec("gamma", [1, 256], DataType.FP32, init_value=torch.randn),
            TensorSpec("beta", [1, 256], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [512, 256], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return LayerNormProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        x = tensors["x"]
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        tensors["out"][:] = (x - mean) / torch.sqrt(variance + 1.0e-5) * tensors["gamma"] + tensors["beta"]


@pl.program
class SiluProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[512, 256], pl.FP32]) -> pl.Tensor[[512, 256], pl.FP32]:
        negative: pl.Tensor[[512, 256], pl.FP32] = pl.mul(x, -1.0)
        exponential: pl.Tensor[[512, 256], pl.FP32] = pl.exp(negative)
        denominator: pl.Tensor[[512, 256], pl.FP32] = pl.add(exponential, 1.0)
        sigmoid: pl.Tensor[[512, 256], pl.FP32] = pl.recip(denominator)
        out: pl.Tensor[[512, 256], pl.FP32] = pl.mul(x, sigmoid)
        return out


class SiluCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_silu"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [512, 256], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [512, 256], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SiluProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.nn.functional.silu(tensors["x"])


@pl.program
class RowMaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[64, 4096], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
        out: pl.Tensor[[64, 1], pl.FP32] = pl.row_max(x)
        return out


class RowMaxCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_row_max"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [64, 4096], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [64, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return RowMaxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"].amax(dim=1, keepdim=True)


@pl.program
class NarrowRowSumProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[16384, 16], pl.FP32]) -> pl.Tensor[[16384, 1], pl.FP32]:
        out: pl.Tensor[[16384, 1], pl.FP32] = pl.row_sum(x)
        return out


class NarrowRowSumCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_narrow_row_sum"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [16384, 16], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [16384, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return NarrowRowSumProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"].sum(dim=1, keepdim=True)


@pl.program
class ColSumProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[2048, 8], pl.FP32]) -> pl.Tensor[[1, 8], pl.FP32]:
        out: pl.Tensor[[1, 8], pl.FP32] = pl.col_sum(x)
        return out


class ColSumCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_col_sum_split"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [2048, 8], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [1, 8], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColSumProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"].sum(dim=0, keepdim=True)


@pl.program
class ColMaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[2048, 64], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
        out: pl.Tensor[[1, 64], pl.FP32] = pl.col_max(x)
        return out


class ColMaxCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_col_max"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [2048, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ColMaxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"].amax(dim=0, keepdim=True)


@pl.program
class Fp16Program:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[128, 512], pl.FP16],
        y: pl.Tensor[[128, 512], pl.FP16],
    ) -> pl.Tensor[[128, 512], pl.FP16]:
        out: pl.Tensor[[128, 512], pl.FP16] = pl.mul(x, y)
        return out


class Fp16Case(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_fp16"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [128, 512], DataType.FP16, init_value=torch.randn),
            TensorSpec("y", [128, 512], DataType.FP16, init_value=torch.randn),
            TensorSpec("out", [128, 512], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return Fp16Program

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"] * tensors["y"]


@pl.program
class Bf16ToFp16CastProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[48, 47000], pl.BF16]) -> pl.Tensor[[48, 47000], pl.FP16]:
        out: pl.Tensor[[48, 47000], pl.FP16] = pl.cast(x, pl.FP16)
        return out


class Bf16ToFp16CastCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_bf16_to_fp16_cast"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [48, 47000], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [48, 47000], DataType.FP16, is_output=True),
        ]

    def get_program(self) -> Any:
        return Bf16ToFp16CastProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"].to(torch.float16)


@pl.program
class Fp16ToBf16CastProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[128, 512], pl.FP16]) -> pl.Tensor[[128, 512], pl.BF16]:
        out: pl.Tensor[[128, 512], pl.BF16] = pl.cast(x, pl.BF16)
        return out


class Fp16ToBf16CastCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_fp16_to_bf16_cast"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [128, 512], DataType.FP16, init_value=torch.randn),
            TensorSpec("out", [128, 512], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return Fp16ToBf16CastProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["x"].to(torch.bfloat16)


def _small_integral_input(shape: tuple[int, int]) -> torch.Tensor:
    return torch.randint(-32, 32, shape, dtype=torch.int32).to(torch.float32)


@pl.program
class Int8OutputProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.INT8]:
        incremented: pl.Tensor[[128, 8192], pl.FP32] = pl.add(x, 1.0)
        out: pl.Tensor[[128, 8192], pl.INT8] = pl.cast(incremented, pl.INT8)
        return out


class Int8OutputCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_int8_output"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "x", [128, 8192], DataType.FP32, init_value=lambda: _small_integral_input((128, 8192))
            ),
            TensorSpec("out", [128, 8192], DataType.INT8, is_output=True),
        ]

    def get_program(self) -> Any:
        return Int8OutputProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = (tensors["x"] + 1.0).to(torch.int8)


@pl.program
class RaggedInt8OutputProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(self, x: pl.Tensor[[1, 33], pl.FP32]) -> pl.Tensor[[1, 33], pl.INT8]:
        incremented: pl.Tensor[[1, 33], pl.FP32] = pl.add(x, 1.0)
        out: pl.Tensor[[1, 33], pl.INT8] = pl.cast(incremented, pl.INT8)
        return out


class RaggedInt8OutputCase(_AutoTileCase):
    def get_name(self) -> str:
        return "auto_tile_vector_ragged_int8_output"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [1, 33], DataType.FP32, init_value=lambda: _small_integral_input((1, 33))),
            TensorSpec("out", [1, 33], DataType.INT8, is_output=True),
        ]

    def get_program(self) -> Any:
        return RaggedInt8OutputProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = (tensors["x"] + 1.0).to(torch.int8)


class TestAutoTileVector:
    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        ("case_type", "config"),
        [
            (RaggedPointwiseCase, None),
            (MultiOutputCase, None),
            (SoftmaxCase, None),
            (CapacityFitSoftmaxCase, None),
            (IntermediateSoftmaxCase, None),
            (RaggedSoftmaxCase, None),
            (RmsCase, _RSQRT_TOL),
            (LayerNormCase, _NORM_TOL),
            (SiluCase, None),
            (RowMaxCase, None),
            (NarrowRowSumCase, None),
            (ColSumCase, None),
            (ColMaxCase, None),
            (Fp16Case, _FP16_TOL),
            (Bf16ToFp16CastCase, _FP16_TOL),
            (Fp16ToBf16CastCase, _BF16_TOL),
            (Int8OutputCase, None),
            (RaggedInt8OutputCase, None),
        ],
    )
    @pytest.mark.platforms("a2a3")
    def test_auto_tile_vector(self, test_runner, platform, case_type, config):
        result = test_runner.run(case_type(platform=platform, config=config))
        assert result.passed, f"{case_type.__name__} failed: {result.error}"
