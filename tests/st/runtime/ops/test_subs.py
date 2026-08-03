# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for ``pto.tsubs``.

The tile path calls ``pl.tile.subs`` directly, so the generated PTO program must
contain the exact ``pto.tsubs`` op.  The tensor path separately covers
``tensor.subs -> tile.subs -> pto.tsubs`` lowering.

The matrix follows the current PTOAS contract:

* A2/A3: ``i16``, ``i32``, ``f16``, and ``f32``;
* A5: the common dtypes plus ``i8`` and ``bf16``;
* negative, zero, and positive scalar values;
* supported cross-dtype scalar operands in both integer-to-float and
  float-to-integer directions;
* full tiles and independent row, column, and row+column ``valid_shape`` tails.

The output is an initialized InOut tensor.  A partial store must update only the
valid prefix and leave the invalid region unchanged.

Although the PTOAS verifier accepts a non-row-major tile, both pinned A2/A3 and
A5 simulator backends miscompute that form.  This stable numerical ST therefore
uses the row-major public path; the verifier/backend mismatch is tracked in the
B01 implementation record rather than hidden behind an xfail.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_PL_DT = {
    DataType.BF16: pl.BF16,
    DataType.FP32: pl.FP32,
    DataType.FP16: pl.FP16,
    DataType.INT32: pl.INT32,
    DataType.INT16: pl.INT16,
    DataType.INT8: pl.INT8,
}

_A5_PLATFORMS = [pytest.param("a5", id="a5"), pytest.param("a5sim", id="a5sim")]

_FP16_CONFIG = RunConfig(rtol=2e-3, atol=2e-3)
_BF16_CONFIG = RunConfig(rtol=2e-2, atol=2e-2)


def _run_config(dtype: DataType) -> RunConfig | None:
    if dtype == DataType.FP16:
        return _FP16_CONFIG
    if dtype == DataType.BF16:
        return _BF16_CONFIG
    return None


def _input_data(m: int, n: int, dtype: DataType) -> torch.Tensor:
    """Small signed values avoid integer overflow while covering both signs."""
    values = torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(17) - 8
    if dtype in (DataType.FP16, DataType.FP32, DataType.BF16):
        values = values / 2
    return values.to(dtype.torch_dtype).contiguous()


class TileSubsCase(PTOTestCase):
    """Direct tile-level ``pl.tile.subs`` case."""

    __test__ = False

    def __init__(
        self,
        *,
        m: int = 32,
        n: int = 64,
        dtype: DataType = DataType.FP32,
        scalar: int | float = 2,
        valid_shape: tuple[int, int] | None = None,
        platform: str,
    ):
        super().__init__(_run_config(dtype), platform=platform)
        self.m = m
        self.n = n
        self.dtype = dtype
        self.scalar = scalar
        self.valid_shape = valid_shape

    def get_name(self) -> str:
        valid = self.valid_shape or (self.m, self.n)
        return f"tile_subs_{self.dtype.value}_{self.m}x{self.n}_v{valid[0]}x{valid[1]}_s{self.scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _input_data(self.m, self.n, self.dtype),
            ),
            TensorSpec(
                "out",
                [self.m, self.n],
                self.dtype,
                init_value=torch.zeros,
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        m, n = self.m, self.n
        dtype = _PL_DT[self.dtype]
        scalar = self.scalar
        valid_shape = list(self.valid_shape or (m, n))

        @pl.program
        class SubsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                src_tile: pl.Tile[[m, n], dtype] = pl.load(src, [0, 0], [m, n], valid_shape=valid_shape)
                result: pl.Tile[[m, n], dtype] = pl.tile.subs(src_tile, scalar)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                return self.kernel(src, out)

        return SubsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows, valid_cols = self.valid_shape or (self.m, self.n)
        expected = torch.zeros_like(tensors["out"])
        expected[:valid_rows, :valid_cols] = tensors["src"][:valid_rows, :valid_cols] - self.scalar
        tensors["out"][:] = expected


class TensorSubsCase(PTOTestCase):
    """Tensor-level path lowered to the exact tile/PTO op."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType,
        scalar: int | float,
        platform: str,
        m: int = 32,
        n: int = 64,
    ):
        super().__init__(_run_config(dtype), platform=platform)
        if dtype == DataType.BF16 and scalar != 1.5:
            raise ValueError(
                "TensorSubsCase BF16 requires scalar=1.5 because pl.const() requires a numeric literal"
            )
        self.m = m
        self.n = n
        self.dtype = dtype
        self.scalar = scalar

    def get_name(self) -> str:
        return f"tensor_subs_{self.dtype.value}_{self.m}x{self.n}_s{self.scalar}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _input_data(self.m, self.n, self.dtype),
            ),
            TensorSpec("out", [self.m, self.n], self.dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self.m, self.n
        dtype = _PL_DT[self.dtype]
        scalar = self.scalar

        if self.dtype == DataType.BF16:

            @pl.program
            class TensorSubsBf16Program:
                @pl.function(type=pl.FunctionType.Opaque)
                def main(
                    self,
                    src: pl.Tensor[[m, n], pl.BF16],
                    out: pl.Out[pl.Tensor[[m, n], pl.BF16]],
                ) -> pl.Tensor[[m, n], pl.BF16]:
                    with pl.at(level=pl.Level.CORE_GROUP):
                        # The parser requires a source literal; __init__ guards the matching oracle value.
                        result: pl.Tensor[[m, n], pl.BF16] = pl.tensor.subs(src, pl.const(1.5, pl.BF16))
                        out = pl.assemble(out, result, [0, 0])
                    return out

            return TensorSubsBf16Program

        @pl.program
        class TensorSubsProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.Out[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[m, n], dtype] = pl.tensor.subs(src, scalar)
                    out = pl.assemble(out, result, [0, 0])
                return out

        return TensorSubsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["src"] - self.scalar


class TileSubsInt32ScalarCase(PTOTestCase):
    """Partial FP32 tile minus an explicitly typed INT32 scalar."""

    __test__ = False

    def __init__(self, *, platform: str):
        super().__init__(platform=platform)

    def get_name(self) -> str:
        return "tile_subs_fp32_int32_scalar_v20x47"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [32, 64],
                DataType.FP32,
                init_value=lambda: _input_data(32, 64, DataType.FP32),
            ),
            TensorSpec("out", [32, 64], DataType.FP32, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class SubsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 64], pl.FP32],
                out: pl.InOut[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                src_tile: pl.Tile[[32, 64], pl.FP32] = pl.load(src, [0, 0], [32, 64], valid_shape=[20, 47])
                result: pl.Tile[[32, 64], pl.FP32] = pl.tile.subs(src_tile, pl.const(2, pl.INT32))
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[32, 64], pl.FP32],
                out: pl.InOut[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                return self.kernel(src, out)

        return SubsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = torch.tensor(2, dtype=torch.int32).to(tensors["src"].dtype)
        expected = torch.zeros_like(tensors["out"])
        expected[:20, :47] = tensors["src"][:20, :47] - scalar
        tensors["out"][:] = expected


class TensorSubsFp32ScalarCase(PTOTestCase):
    """INT16 tensor minus an explicitly typed FP32 scalar."""

    __test__ = False

    def __init__(self, *, platform: str):
        super().__init__(platform=platform)

    def get_name(self) -> str:
        return "tensor_subs_int16_fp32_scalar"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [32, 64],
                DataType.INT16,
                init_value=lambda: _input_data(32, 64, DataType.INT16),
            ),
            TensorSpec("out", [32, 64], DataType.INT16, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class SubsProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                src: pl.Tensor[[32, 64], pl.INT16],
                out: pl.Out[pl.Tensor[[32, 64], pl.INT16]],
            ) -> pl.Tensor[[32, 64], pl.INT16]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[32, 64], pl.INT16] = pl.tensor.subs(src, pl.const(2.0, pl.FP32))
                    out = pl.assemble(out, result, [0, 0])
                return out

        return SubsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = torch.tensor(2.0, dtype=torch.float32).to(tensors["src"].dtype)
        tensors["out"][:] = tensors["src"] - scalar


_COMMON_DTYPE_CASES = [
    pytest.param(DataType.FP32, -2.5, 32, 64, id="f32-negative"),
    pytest.param(DataType.FP16, 0.0, 16, 128, id="f16-zero"),
    pytest.param(DataType.INT32, 7, 31, 64, id="i32-positive"),
    pytest.param(DataType.INT16, -3, 15, 192, id="i16-negative"),
]

_A5_DTYPE_CASES = [
    pytest.param(DataType.INT8, 3, id="i8"),
    pytest.param(DataType.BF16, -1.5, id="bf16"),
]

_VALID_SHAPE_CASES = [
    pytest.param((20, 64), id="row-tail"),
    pytest.param((32, 47), id="column-tail"),
    pytest.param((20, 47), id="row-column-tail"),
]


class TestTileSubs:
    """All PTOAS dtype, scalar-sign, architecture, and valid-region branches."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("dtype,scalar,m,n", _COMMON_DTYPE_CASES)
    def test_common_dtypes(self, test_runner, platform, dtype, scalar, m, n):
        result = test_runner.run(TileSubsCase(m=m, n=n, dtype=dtype, scalar=scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_PLATFORMS)
    @pytest.mark.parametrize("dtype,scalar", _A5_DTYPE_CASES)
    def test_a5_extra_dtypes(self, test_runner, platform, dtype, scalar):
        result = test_runner.run(TileSubsCase(dtype=dtype, scalar=scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("valid_shape", _VALID_SHAPE_CASES)
    def test_valid_shape_tails(self, test_runner, platform, valid_shape):
        result = test_runner.run(TileSubsCase(scalar=1.25, valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestTensorSubs:
    """Tensor frontend and Tensor-to-Tile conversion coverage."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize(
        "dtype,scalar",
        [
            pytest.param(DataType.FP32, -2.5, id="f32"),
            pytest.param(DataType.INT16, 3, id="i16"),
        ],
    )
    def test_tensor_lowering(self, test_runner, platform, dtype, scalar):
        result = test_runner.run(TensorSubsCase(dtype=dtype, scalar=scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_PLATFORMS)
    def test_tensor_bf16_lowering(self, test_runner, platform):
        result = test_runner.run(TensorSubsCase(dtype=DataType.BF16, scalar=1.5, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestMixedScalarSubs:
    """The supported scalar dtype may differ from the tile dtype."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fp32_tile_int32_scalar(self, test_runner, platform):
        result = test_runner.run(TileSubsInt32ScalarCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_int16_tensor_fp32_scalar(self, test_runner, platform):
        result = test_runner.run(TensorSubsFp32ScalarCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
