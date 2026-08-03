# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for ``pto.tdiv``.

The tile path calls ``pl.tile.div`` with two tiles, so it exercises the exact
``pto.tdiv`` op rather than the scalar ``pto.tdivs`` variant.  The tensor path
separately covers ``tensor.div -> tile.div -> pto.tdiv`` lowering.

The matrix follows the current PTOAS contract:

* A2/A3: ``f16`` and ``f32``;
* A5: the common floating-point dtypes plus ``i16`` and ``i32``;
* ``default`` and ``high_precision`` for every supported floating-point dtype;
* full tiles, independent row/column tails, a combined tail, and a small
  boundary shape.

Every divisor element is non-zero, including elements outside a narrowed valid
region.  Partial stores target a zero-initialized InOut tensor, so the invalid
region has a reliable zero oracle instead of depending on uninitialized memory.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_PL_DT = {
    DataType.FP32: pl.FP32,
    DataType.FP16: pl.FP16,
    DataType.INT32: pl.INT32,
    DataType.INT16: pl.INT16,
}

_A5_PLATFORMS = [
    pytest.param("a5sim", id="a5sim"),
    pytest.param("a5", id="a5"),
]

_INTEGER_DTYPES = (DataType.INT16, DataType.INT32)


def _run_config(dtype: DataType) -> RunConfig | None:
    # FP16 division is rounded at FP16 precision; the strict 1e-5 default is
    # below one FP16 ULP around 1.0.
    if dtype == DataType.FP16:
        return RunConfig(rtol=2e-3, atol=2e-3)
    return None


def _dividend(m: int, n: int, dtype: DataType) -> torch.Tensor:
    """Deterministic signed dividend with zero and non-integral float values."""
    values = torch.arange(m * n, dtype=torch.int64).reshape(m, n).remainder(31) - 15
    if dtype not in _INTEGER_DTYPES:
        values = values.to(torch.float32) / 3.0
    return values.to(dtype.torch_dtype).contiguous()


def _divisor(m: int, n: int, dtype: DataType) -> torch.Tensor:
    """Deterministic signed divisor whose physical tile contains no zero."""
    index = torch.arange(m * n, dtype=torch.int64).reshape(m, n)
    magnitude = index.remainder(7) + 1
    sign = torch.where(index.remainder(2) == 0, 1, -1)
    values = magnitude * sign
    if dtype not in _INTEGER_DTYPES:
        values = values.to(torch.float32)
    result = values.to(dtype.torch_dtype).contiguous()
    assert torch.count_nonzero(result) == result.numel()
    return result


class TileDivCase(PTOTestCase):
    """Direct two-tile ``pl.tile.div`` case."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType,
        high_precision: bool,
        platform: str,
        m: int = 16,
        n: int = 64,
        valid_shape: tuple[int, int] | None = None,
    ):
        super().__init__(_run_config(dtype), platform=platform)
        self.m = m
        self.n = n
        self.dtype = dtype
        self.high_precision = high_precision
        self.valid_shape = valid_shape

    def get_name(self) -> str:
        valid_rows, valid_cols = self.valid_shape or (self.m, self.n)
        precision = "high_precision" if self.high_precision else "default"
        return f"tile_div_{self.dtype.value}_{precision}_{self.m}x{self.n}_v{valid_rows}x{valid_cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "lhs",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _dividend(self.m, self.n, self.dtype),
            ),
            TensorSpec(
                "rhs",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _divisor(self.m, self.n, self.dtype),
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
        high_precision = self.high_precision
        valid_shape = list(self.valid_shape or (m, n))

        @pl.program
        class DivProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[m, n], dtype],
                rhs: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                lhs_tile: pl.Tile[[m, n], dtype] = pl.load(lhs, [0, 0], [m, n], valid_shape=valid_shape)
                rhs_tile: pl.Tile[[m, n], dtype] = pl.load(rhs, [0, 0], [m, n], valid_shape=valid_shape)
                result: pl.Tile[[m, n], dtype] = pl.tile.div(
                    lhs_tile, rhs_tile, high_precision=high_precision
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[m, n], dtype],
                rhs: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                return self.kernel(lhs, rhs, out)

        return DivProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows, valid_cols = self.valid_shape or (self.m, self.n)
        lhs = tensors["lhs"][:valid_rows, :valid_cols]
        rhs = tensors["rhs"][:valid_rows, :valid_cols]
        expected = torch.zeros_like(tensors["out"])
        if self.dtype in _INTEGER_DTYPES:
            expected[:valid_rows, :valid_cols] = torch.div(lhs, rhs, rounding_mode="trunc")
        else:
            expected[:valid_rows, :valid_cols] = torch.div(lhs, rhs)
        tensors["out"][:] = expected


class TensorDivCase(PTOTestCase):
    """Tensor-level division lowered to the exact tile/PTO op."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType,
        high_precision: bool,
        platform: str,
        m: int = 32,
        n: int = 64,
    ):
        super().__init__(_run_config(dtype), platform=platform)
        self.m = m
        self.n = n
        self.dtype = dtype
        self.high_precision = high_precision

    def get_name(self) -> str:
        precision = "high_precision" if self.high_precision else "default"
        return f"tensor_div_{self.dtype.value}_{precision}_{self.m}x{self.n}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "lhs",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _dividend(self.m, self.n, self.dtype),
            ),
            TensorSpec(
                "rhs",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _divisor(self.m, self.n, self.dtype),
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
        high_precision = self.high_precision

        @pl.program
        class TensorDivProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                lhs: pl.Tensor[[m, n], dtype],
                rhs: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[m, n], dtype] = pl.div(lhs, rhs, high_precision=high_precision)
                    out = pl.assemble(out, result, [0, 0])
                return out

        return TensorDivProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.div(tensors["lhs"], tensors["rhs"])


_FLOAT_PRECISION_CASES = [
    pytest.param(DataType.FP32, False, id="f32-default"),
    pytest.param(DataType.FP16, False, id="f16-default"),
    pytest.param(DataType.FP32, True, id="f32-high-precision"),
    pytest.param(DataType.FP16, True, id="f16-high-precision"),
]

_VALID_SHAPE_CASES = [
    pytest.param(32, 64, (19, 64), id="row-tail"),
    pytest.param(32, 64, (32, 37), id="column-tail"),
    pytest.param(32, 64, (19, 37), id="row-column-tail"),
    pytest.param(2, 16, (2, 16), id="small-boundary"),
]

_A5_INTEGER_CASES = [
    pytest.param(DataType.INT16, 16, 64, (16, 64), id="i16-full"),
    pytest.param(DataType.INT32, 16, 64, (11, 37), id="i32-tail"),
]


class TestTileDiv:
    """Direct ``pto.tdiv`` dtype, precision, shape, and valid-region coverage."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("dtype,high_precision", _FLOAT_PRECISION_CASES)
    def test_float_dtype_and_precision(self, test_runner, platform, dtype, high_precision):
        result = test_runner.run(
            TileDivCase(
                dtype=dtype,
                high_precision=high_precision,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,n,valid_shape", _VALID_SHAPE_CASES)
    def test_valid_shape_and_boundary(self, test_runner, platform, m, n, valid_shape):
        result = test_runner.run(
            TileDivCase(
                dtype=DataType.FP32,
                high_precision=False,
                platform=platform,
                m=m,
                n=n,
                valid_shape=valid_shape,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(DataType.FP32, id="f32"),
            pytest.param(DataType.FP16, id="f16"),
        ],
    )
    def test_high_precision_with_combined_tail(self, test_runner, platform, dtype):
        result = test_runner.run(
            TileDivCase(
                dtype=dtype,
                high_precision=True,
                platform=platform,
                m=32,
                n=64,
                valid_shape=(19, 37),
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_PLATFORMS)
    @pytest.mark.parametrize("dtype,m,n,valid_shape", _A5_INTEGER_CASES)
    def test_a5_integer_dtypes(self, test_runner, platform, dtype, m, n, valid_shape):
        result = test_runner.run(
            TileDivCase(
                dtype=dtype,
                high_precision=False,
                platform=platform,
                m=m,
                n=n,
                valid_shape=valid_shape,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


class TestTensorDiv:
    """Tensor frontend and Tensor-to-Tile conversion coverage."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize(
        "dtype,high_precision",
        [
            pytest.param(DataType.FP32, False, id="f32-default"),
            pytest.param(DataType.FP16, True, id="f16-high-precision"),
        ],
    )
    def test_tensor_lowering(self, test_runner, platform, dtype, high_precision):
        result = test_runner.run(
            TensorDivCase(
                dtype=dtype,
                high_precision=high_precision,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
