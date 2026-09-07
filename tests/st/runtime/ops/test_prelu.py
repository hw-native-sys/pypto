# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for the three-input ``pto.tprelu`` chain."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_PL_DT = {
    DataType.INT8: pl.INT8,
    DataType.UINT8: pl.UINT8,
    DataType.INT32: pl.INT32,
    DataType.FP16: pl.FP16,
    DataType.FP32: pl.FP32,
}
_TORCH_DT = {DataType.FP16: torch.float16, DataType.FP32: torch.float32}


def _source(m: int, n: int, dtype: DataType) -> torch.Tensor:
    values = torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(23) - 11
    return values.to(_TORCH_DT[dtype])


def _slope(m: int, n: int, dtype: DataType) -> torch.Tensor:
    choices = torch.tensor([-1.0, 0.0, 0.25, 1.5], dtype=torch.float32)
    indices = torch.arange(m * n).reshape(m, n).remainder(len(choices))
    return choices[indices].to(_TORCH_DT[dtype])


class TilePreluTestCase(PTOTestCase):
    """Execute TPRELU with portable UINT8 scratch."""

    __test__ = False

    def __init__(
        self,
        *,
        m: int = 16,
        n: int = 64,
        valid_shape: tuple[int, int] | None = None,
        dtype: DataType = DataType.FP32,
        tmp_dtype: DataType = DataType.UINT8,
        oversized_tmp: bool = False,
        a5_placeholder_tmp: bool = False,
        platform: str | None = None,
    ):
        config = RunConfig(rtol=2e-3, atol=2e-3) if dtype == DataType.FP16 else None
        super().__init__(config, platform=platform)
        self._m = m
        self._n = n
        self._valid_shape = valid_shape
        self._dtype = dtype
        self._tmp_dtype = tmp_dtype
        self._oversized_tmp = oversized_tmp
        self._a5_placeholder_tmp = a5_placeholder_tmp

    def get_name(self) -> str:
        valid = self._valid_shape or (self._m, self._n)
        if self._a5_placeholder_tmp:
            tmp = "a5_placeholder"
        elif self._oversized_tmp:
            tmp = "oversized"
        else:
            tmp = "minimum"
        tmp_dtype = DataType.INT32 if self._a5_placeholder_tmp else self._tmp_dtype
        return (
            f"tile_prelu_{self._dtype.value}_{self._m}x{self._n}_v{valid[0]}x{valid[1]}_"
            f"{tmp}_{tmp_dtype.value}_tmp"
        )

    def define_tensors(self) -> list[TensorSpec]:
        valid_rows, valid_cols = self._valid_shape or (self._m, self._n)
        if self._a5_placeholder_tmp:
            tmp_shape = [1, 8]
            tmp_dtype = DataType.INT32
        else:
            tmp_shape = [valid_rows + (8 if self._oversized_tmp else 1), 64 if self._oversized_tmp else 32]
            tmp_dtype = self._tmp_dtype
        return [
            TensorSpec(
                "src",
                [self._m, self._n],
                self._dtype,
                init_value=lambda: _source(self._m, self._n, self._dtype),
            ),
            TensorSpec(
                "slope",
                [self._m, self._n],
                self._dtype,
                init_value=lambda: _slope(self._m, self._n, self._dtype),
            ),
            TensorSpec(
                "tmp",
                tmp_shape,
                tmp_dtype,
                init_value=torch.zeros,
            ),
            TensorSpec(
                "out",
                [self._m, self._n],
                self._dtype,
                is_output=True,
                init_value=torch.zeros,
            ),
        ]

    def get_program(self) -> Any:
        m, n = self._m, self._n
        valid_shape = list(self._valid_shape or (m, n))
        valid_rows, valid_cols = valid_shape
        dtype = _PL_DT[self._dtype]
        if self._a5_placeholder_tmp:
            tmp_rows, tmp_cols = 1, 8
            tmp_valid_shape = [1, 1]
            tmp_dtype = pl.INT32
        else:
            tmp_rows = valid_rows + (8 if self._oversized_tmp else 1)
            tmp_cols = 64 if self._oversized_tmp else 32
            tmp_valid_shape = [valid_rows, (valid_cols + 7) // 8]
            tmp_dtype = _PL_DT[self._tmp_dtype]

        @pl.program
        class PreluProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[m, n], dtype],
                slope: pl.Tensor[[m, n], dtype],
                tmp_in: pl.Tensor[[tmp_rows, tmp_cols], tmp_dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                src_tile: pl.Tile[[m, n], dtype] = pl.load(src, [0, 0], [m, n], valid_shape=valid_shape)
                slope_tile: pl.Tile[[m, n], dtype] = pl.load(slope, [0, 0], [m, n], valid_shape=valid_shape)
                tmp: pl.Tile[[tmp_rows, tmp_cols], tmp_dtype] = pl.load(
                    tmp_in,
                    [0, 0],
                    [tmp_rows, tmp_cols],
                    valid_shape=tmp_valid_shape,
                )
                result: pl.Tile[[m, n], dtype] = pl.tile.prelu(src_tile, slope_tile, tmp)
                out = pl.store(result, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[m, n], dtype],
                slope: pl.Tensor[[m, n], dtype],
                tmp_in: pl.Tensor[[tmp_rows, tmp_cols], tmp_dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                out = self.kernel(src, slope, tmp_in, out)
                return out

        return PreluProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        src = tensors["src"]
        slope = tensors["slope"]
        valid_rows, valid_cols = self._valid_shape or (self._m, self._n)
        expected = torch.zeros_like(tensors["out"])
        valid_src = src[:valid_rows, :valid_cols]
        valid_slope = slope[:valid_rows, :valid_cols]
        expected[:valid_rows, :valid_cols] = torch.where(
            valid_src > 0,
            valid_src,
            valid_src * valid_slope,
        )
        tensors["out"][:] = expected


_BOUNDARY_PHYSICAL_SHAPES = ((1, 256), (64, 16))


class TestTilePrelu:
    """TPRELU dtype, scratch, and valid-shape branches on hardware."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(DataType.FP16, id="fp16"),
            pytest.param(DataType.FP32, id="fp32"),
        ],
    )
    def test_dtypes(self, test_runner, platform, dtype):
        result = test_runner.run(TilePreluTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "valid_shape",
        [
            pytest.param(None, id="full"),
            pytest.param((9, 64), id="row-tail"),
            pytest.param((16, 37), id="col-tail"),
            pytest.param((9, 37), id="row-col-tail"),
        ],
    )
    def test_valid_shape(self, test_runner, platform, valid_shape):
        result = test_runner.run(TilePreluTestCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_oversized_tmp(self, test_runner, platform):
        result = test_runner.run(TilePreluTestCase(oversized_tmp=True, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
    def test_a2a3_uint8_tmp(self, test_runner, platform):
        result = test_runner.run(TilePreluTestCase(tmp_dtype=DataType.UINT8, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5")
    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_a5_unused_placeholder_tmp(self, test_runner, platform):
        result = test_runner.run(TilePreluTestCase(a5_placeholder_tmp=True, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    # Same reason as test_sels: indexing a parametrize value put these past what
    # collection can evaluate from source, so they compiled one at a time. The
    # marker holds the harness matrix to the on-board ids the explicit
    # ``ONBOARD_PLATFORMS`` parametrize used to name.
    @pytest.mark.platforms(
        "a2a3",
        "a5",
        reason="a physical-shape boundary is an on-board packing property; the simulator does not model it",
    )
    @st.cases(*(st.from_legacy(TilePreluTestCase(m=m, n=n)) for m, n in _BOUNDARY_PHYSICAL_SHAPES))
    def test_boundary_physical_shapes(self, case_run):
        case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
