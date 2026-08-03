# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for ``pto.tlog``.

The tile path calls ``pl.tile.log`` directly, and the tensor path covers
``tensor.log -> tile.log -> pto.tlog`` lowering.  Both ``f16`` and ``f32`` run
in ``default`` and ``high_precision`` modes on A2/A3 and A5.

The shape matrix covers a full tile, independent row/column tails, a combined
tail, and a small boundary shape.  Inputs are strictly positive throughout the
physical tile, including the invalid tail.  Partial stores target a
zero-initialized InOut tensor so invalid output elements have a defined oracle.
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
}


def _run_config(dtype: DataType) -> RunConfig | None:
    # FP16 logarithm is a transcendental approximation rounded to FP16; the
    # strict 1e-5 default is below one FP16 ULP around 1.0.
    if dtype == DataType.FP16:
        return RunConfig(rtol=2e-3, atol=2e-3)
    return None


def _positive_input(m: int, n: int, dtype: DataType) -> torch.Tensor:
    """Values span both sides of 1.0 while remaining strictly in log's domain."""
    index = torch.arange(m * n, dtype=torch.float32).reshape(m, n)
    values = (index.remainder(127) + 1.0) / 8.0
    result = values.to(dtype.torch_dtype).contiguous()
    assert torch.all(result > 0)
    return result


class TileLogCase(PTOTestCase):
    """Direct ``pl.tile.log`` case."""

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
        return f"tile_log_{self.dtype.value}_{precision}_{self.m}x{self.n}_v{valid_rows}x{valid_cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _positive_input(self.m, self.n, self.dtype),
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
        class LogProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                src_tile: pl.Tile[[m, n], dtype] = pl.load(src, [0, 0], [m, n], valid_shape=valid_shape)
                result: pl.Tile[[m, n], dtype] = pl.tile.log(src_tile, high_precision=high_precision)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                return self.kernel(src, out)

        return LogProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows, valid_cols = self.valid_shape or (self.m, self.n)
        expected = torch.zeros_like(tensors["out"])
        expected[:valid_rows, :valid_cols] = torch.log(tensors["src"][:valid_rows, :valid_cols])
        tensors["out"][:] = expected


class TensorLogCase(PTOTestCase):
    """Tensor-level logarithm lowered to the exact tile/PTO op."""

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
        return f"tensor_log_{self.dtype.value}_{precision}_{self.m}x{self.n}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _positive_input(self.m, self.n, self.dtype),
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
        class TensorLogProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[m, n], dtype] = pl.log(src, high_precision=high_precision)
                    out = pl.assemble(out, result, [0, 0])
                return out

        return TensorLogProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.log(tensors["src"])


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


class TestTileLog:
    """Direct ``pto.tlog`` dtype, precision, shape, and valid-region coverage."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("dtype,high_precision", _FLOAT_PRECISION_CASES)
    def test_float_dtype_and_precision(self, test_runner, platform, dtype, high_precision):
        result = test_runner.run(
            TileLogCase(
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
            TileLogCase(
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
            TileLogCase(
                dtype=dtype,
                high_precision=True,
                platform=platform,
                m=32,
                n=64,
                valid_shape=(19, 37),
            )
        )
        assert result.passed, f"Test failed: {result.error}"


class TestTensorLog:
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
            TensorLogCase(
                dtype=dtype,
                high_precision=high_precision,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
