# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for ``pto.tsels``.

The mask is produced by ``tile.cmps`` and feeds the canonical four-input
``tile.sels(mask, src, tmp, scalar)`` chain. Coverage includes every comparison
mode, target-supported scalar types, and row/column/combined valid tails.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

_PL_DT = {
    DataType.INT8: pl.INT8,
    DataType.UINT8: pl.UINT8,
    DataType.INT16: pl.INT16,
    DataType.UINT16: pl.UINT16,
    DataType.INT32: pl.INT32,
    DataType.UINT32: pl.UINT32,
    DataType.FP16: pl.FP16,
    DataType.FP32: pl.FP32,
}
_TORCH_DT = {
    DataType.INT8: torch.int8,
    DataType.UINT8: torch.uint8,
    DataType.INT16: torch.int16,
    DataType.UINT16: torch.int16,
    DataType.INT32: torch.int32,
    DataType.UINT32: torch.int32,
    DataType.FP16: torch.float16,
    DataType.FP32: torch.float32,
}
_CMP = {
    0: torch.eq,
    1: torch.ne,
    2: torch.lt,
    3: torch.le,
    4: torch.gt,
    5: torch.ge,
}
_A5_ONBOARD_PLATFORMS = [pytest.param("a5", id="a5")]
_A2A3_ONBOARD_PLATFORMS = [pytest.param("a2a3", id="a2a3")]
_UNSIGNED_DTYPES = {DataType.UINT8, DataType.UINT16, DataType.UINT32}
_INTEGER_BYTES = {
    DataType.INT8: 1,
    DataType.UINT8: 1,
    DataType.INT16: 2,
    DataType.UINT16: 2,
    DataType.INT32: 4,
    DataType.UINT32: 4,
}
_DTYPE_BYTES = {
    **_INTEGER_BYTES,
    DataType.FP16: 2,
    DataType.FP32: 4,
}
_ALTERNATING_MASK_VALUE = {
    DataType.INT8: -86,
    DataType.UINT8: 0xAA,
    DataType.INT16: -21846,
    DataType.UINT16: -21846,
    DataType.INT32: -1431655766,
    DataType.UINT32: -1431655766,
}


def _source(m: int, n: int, dtype: DataType) -> torch.Tensor:
    values = torch.arange(m * n, dtype=torch.int64).reshape(m, n).remainder(17)
    if dtype in _UNSIGNED_DTYPES:
        values += 1 << (_DTYPE_BYTES[dtype] * 8 - 1)
    else:
        values = values - 8
    return values.to(_TORCH_DT[dtype])


class TileSelsTestCase(PTOTestCase):
    """Execute one canonical TSELS branch on hardware."""

    __test__ = False

    def __init__(
        self,
        *,
        m: int = 16,
        n: int = 64,
        valid_shape: tuple[int, int] | None = None,
        dtype: DataType = DataType.FP32,
        cmp_type: int = 4,
        threshold: int | float = 0,
        scalar: int | float = -3,
        tmp_dtype: DataType | None = None,
        minimal_tmp: bool = False,
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._m = m
        self._n = n
        self._valid_shape = valid_shape
        self._dtype = dtype
        self._cmp_type = cmp_type
        self._threshold = threshold
        self._scalar = scalar
        self._tmp_dtype = tmp_dtype or (dtype if platform == "a2a3" else DataType.UINT8)
        self._minimal_tmp = minimal_tmp
        self._platform = platform

    def get_name(self) -> str:
        valid = self._valid_shape or (self._m, self._n)
        tmp_suffix = f"_{self._tmp_dtype.value}_{'min' if self._minimal_tmp else 'default'}_tmp"
        return (
            f"tile_sels_{self._dtype.value}_{self._m}x{self._n}_v{valid[0]}x{valid[1]}_"
            f"cmp{self._cmp_type}{tmp_suffix}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self._m, self._n],
                self._dtype,
                init_value=lambda: _source(self._m, self._n, self._dtype),
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
        dtype = _PL_DT[self._dtype]
        cmp_type = self._cmp_type
        threshold = self._threshold
        scalar = self._scalar
        mask_cols = ((n + 7) // 8 + 31) // 32 * 32
        tmp_dtype = _PL_DT[self._tmp_dtype]
        aligned_minimum_cols = 32 // _DTYPE_BYTES[self._tmp_dtype]
        if self._platform == "a2a3":
            # PTOAS v0.60 verifies the A2/A3 scratch against one complete
            # physical source row, in the source dtype.
            tmp_rows, tmp_cols = 1, n
        else:
            tmp_rows, tmp_cols = (1, aligned_minimum_cols) if self._minimal_tmp else (1, 32)

        @pl.program
        class SelsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                src_tile: pl.Tile[[m, n], dtype] = pl.load(src, [0, 0], [m, n], valid_shape=valid_shape)
                mask: pl.Tile[[m, mask_cols], pl.UINT8] = pl.tile.cmps(src_tile, threshold, cmp_type=cmp_type)
                tmp: pl.Tile[[tmp_rows, tmp_cols], tmp_dtype] = pl.tile.create(
                    [tmp_rows, tmp_cols],
                    dtype=tmp_dtype,
                )
                result: pl.Tile[[m, n], dtype] = pl.tile.sels(mask, src_tile, tmp, scalar)
                out = pl.store(result, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[m, n], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                out = self.kernel(src, out)
                return out

        return SelsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        src = tensors["src"]
        valid_rows, valid_cols = self._valid_shape or (self._m, self._n)
        expected = torch.zeros_like(tensors["out"])
        valid_src = src[:valid_rows, :valid_cols]
        mask = _CMP[self._cmp_type](valid_src, self._threshold)
        expected[:valid_rows, :valid_cols] = torch.where(
            mask,
            valid_src,
            torch.as_tensor(self._scalar, dtype=valid_src.dtype),
        )
        tensors["out"][:] = expected


class TileSelsMaskCarrierTestCase(PTOTestCase):
    """Execute TSELS with an explicitly loaded 8/16/32-bit mask carrier."""

    __test__ = False

    def __init__(
        self,
        mask_dtype: DataType,
        platform: str,
        *,
        src_dtype: DataType = DataType.FP32,
        scalar: int | float = -3.0,
    ):
        super().__init__(platform=platform)
        self._mask_dtype = mask_dtype
        self._platform = platform
        self._src_dtype = src_dtype
        self._scalar = scalar

    def get_name(self) -> str:
        return f"tile_sels_{self._src_dtype.value}_mask_{self._mask_dtype.value}"

    def define_tensors(self) -> list[TensorSpec]:
        mask_cols = 32 // _INTEGER_BYTES[self._mask_dtype]
        mask_value = _ALTERNATING_MASK_VALUE[self._mask_dtype]
        return [
            TensorSpec(
                "src",
                [2, 16],
                self._src_dtype,
                init_value=lambda: _source(2, 16, self._src_dtype),
            ),
            TensorSpec(
                "mask",
                [2, mask_cols],
                self._mask_dtype,
                init_value=lambda: torch.full(
                    (2, mask_cols),
                    mask_value,
                    dtype=_TORCH_DT[self._mask_dtype],
                ),
            ),
            TensorSpec("out", [2, 16], self._src_dtype, is_output=True, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        mask_dtype = _PL_DT[self._mask_dtype]
        mask_cols = 32 // _INTEGER_BYTES[self._mask_dtype]
        src_dtype = _PL_DT[self._src_dtype]
        scalar = self._scalar
        tmp_dtype = src_dtype if self._platform == "a2a3" else pl.UINT8
        tmp_cols = 16 if self._platform == "a2a3" else 32

        @pl.program
        class SelsMaskProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[2, 16], src_dtype],
                mask_in: pl.Tensor[[2, mask_cols], mask_dtype],
                out: pl.InOut[pl.Tensor[[2, 16], src_dtype]],
            ) -> pl.Tensor[[2, 16], src_dtype]:
                src_tile: pl.Tile[[2, 16], src_dtype] = pl.load(src, [0, 0], [2, 16])
                mask: pl.Tile[[2, mask_cols], mask_dtype] = pl.load(
                    mask_in,
                    [0, 0],
                    [2, mask_cols],
                )
                tmp: pl.Tile[[1, tmp_cols], tmp_dtype] = pl.tile.create([1, tmp_cols], dtype=tmp_dtype)
                result: pl.Tile[[2, 16], src_dtype] = pl.tile.sels(mask, src_tile, tmp, scalar)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[2, 16], src_dtype],
                mask_in: pl.Tensor[[2, mask_cols], mask_dtype],
                out: pl.InOut[pl.Tensor[[2, 16], src_dtype]],
            ) -> pl.Tensor[[2, 16], src_dtype]:
                return self.kernel(src, mask_in, out)

        return SelsMaskProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        scalar = self._scalar
        if self._src_dtype in _UNSIGNED_DTYPES and isinstance(scalar, int):
            bits = _DTYPE_BYTES[self._src_dtype] * 8
            if scalar >= 1 << (bits - 1):
                scalar -= 1 << bits
        expected = torch.full_like(tensors["out"], scalar)
        expected[:, 1::2] = tensors["src"][:, 1::2]
        tensors["out"][:] = expected


class TestTileSels:
    """TSELS semantic branches on every onboard platform."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("cmp_type", range(6), ids=("eq", "ne", "lt", "le", "gt", "ge"))
    def test_comparison_modes(self, test_runner, platform, cmp_type):
        result = test_runner.run(TileSelsTestCase(cmp_type=cmp_type, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "dtype,scalar",
        [
            pytest.param(DataType.FP16, -0.5, id="fp16"),
            pytest.param(DataType.FP32, 1.25, id="fp32"),
        ],
    )
    def test_scalar_dtypes(self, test_runner, platform, dtype, scalar):
        result = test_runner.run(TileSelsTestCase(dtype=dtype, scalar=scalar, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A2A3_ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "dtype,scalar",
        [
            pytest.param(DataType.INT16, -7, id="int16"),
            pytest.param(DataType.UINT16, 0x8007, id="uint16-high-bit"),
            pytest.param(DataType.INT32, -11, id="int32"),
            pytest.param(DataType.UINT32, 0x8000000B, id="uint32-high-bit"),
        ],
    )
    def test_a2a3_integer_scalar_dtypes(self, test_runner, platform, dtype, scalar):
        result = test_runner.run(
            TileSelsMaskCarrierTestCase(
                DataType.UINT8,
                platform,
                src_dtype=dtype,
                scalar=scalar,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "dtype,scalar",
        [
            pytest.param(DataType.INT8, -2, id="int8"),
            pytest.param(DataType.UINT8, 0x82, id="uint8-high-bit"),
            pytest.param(DataType.INT16, -7, id="int16"),
            pytest.param(DataType.UINT16, 0x8007, id="uint16-high-bit"),
            pytest.param(DataType.INT32, 11, id="int32"),
            pytest.param(DataType.UINT32, 0x8000000B, id="uint32-high-bit"),
        ],
    )
    def test_a5_integer_scalar_dtypes(self, test_runner, platform, dtype, scalar):
        result = test_runner.run(
            TileSelsMaskCarrierTestCase(
                DataType.UINT8,
                platform,
                src_dtype=dtype,
                scalar=scalar,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "mask_dtype",
        [
            pytest.param(DataType.INT8, id="int8"),
            pytest.param(DataType.UINT8, id="uint8"),
            pytest.param(DataType.INT16, id="int16"),
            pytest.param(DataType.UINT16, id="uint16"),
            pytest.param(DataType.INT32, id="int32"),
            pytest.param(DataType.UINT32, id="uint32"),
        ],
    )
    def test_mask_carrier_dtypes(self, test_runner, platform, mask_dtype):
        result = test_runner.run(TileSelsMaskCarrierTestCase(mask_dtype, platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(
        "platform,src_dtype,mask_dtype",
        [
            pytest.param("a2a3", DataType.INT16, DataType.INT16, id="a2a3-i16-mask-i16"),
            pytest.param("a2a3", DataType.UINT32, DataType.UINT32, id="a2a3-u32-mask-u32"),
            pytest.param("a5", DataType.INT8, DataType.INT16, id="a5-i8-mask-i16"),
            pytest.param("a5", DataType.UINT16, DataType.UINT32, id="a5-u16-mask-u32"),
        ],
    )
    def test_source_and_mask_width_interactions(self, test_runner, platform, src_dtype, mask_dtype):
        result = test_runner.run(TileSelsMaskCarrierTestCase(mask_dtype, platform, src_dtype=src_dtype))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", _A2A3_ONBOARD_PLATFORMS)
    def test_a2a3_minimum_typed_tmp(self, test_runner, platform):
        result = test_runner.run(TileSelsTestCase(minimal_tmp=True, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5")
    @pytest.mark.parametrize("platform", _A5_ONBOARD_PLATFORMS)
    def test_a5_unrestricted_tmp_placeholder(self, test_runner, platform):
        result = test_runner.run(
            TileSelsTestCase(
                tmp_dtype=DataType.INT32,
                minimal_tmp=True,
                platform=platform,
            )
        )
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
        result = test_runner.run(TileSelsTestCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "physical_shape,valid_shape",
        [
            pytest.param((1, 64), None, id="one-row"),
            pytest.param((64, 16), None, id="tall-narrow"),
            pytest.param((2, 256), None, id="packed-32-byte-boundary"),
            pytest.param((2, 264), (2, 257), id="packed-33-byte-boundary"),
        ],
    )
    def test_boundary_physical_shapes(self, test_runner, platform, physical_shape, valid_shape):
        result = test_runner.run(
            TileSelsTestCase(
                m=physical_shape[0],
                n=physical_shape[1],
                valid_shape=valid_shape,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
