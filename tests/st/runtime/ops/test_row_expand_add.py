# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for ``pto.trowexpandadd``.

For a DN ``[M, 1]`` carrier, semantics are
``dst[row, col] = src0[row, col] + src1[row, 0]``.  The raw row-major carrier
holds one 32-byte block per row and repeats its lanes across the destination
row.  The direct tile path covers both PTOAS signatures, with and without
optional ``tmp``.  The tensor path covers compiler lowering to the two-input
DN form.

The current PTOAS architecture matrix is:

* A2/A3: ``i16``, ``i32``, ``f16``, and ``f32``;
* A5: the common dtypes plus ``i8``.

For every partial main-tile valid region the per-row vector uses the same valid
row count, as required by PTOAS.  Full, row-tail, column-tail, and combined-tail
regions are all represented.
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
    DataType.INT8: pl.INT8,
}

_A5_PLATFORMS = [pytest.param("a5", id="a5"), pytest.param("a5sim", id="a5sim")]
_A5_DEVICE_PLATFORMS = [pytest.param("a5", id="a5")]
_FP16_CONFIG = RunConfig(rtol=2e-3, atol=2e-3)


def _run_config(dtype: DataType) -> RunConfig | None:
    return _FP16_CONFIG if dtype == DataType.FP16 else None


def _main_data(m: int, n: int, dtype: DataType) -> torch.Tensor:
    values = torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(23) - 11
    if dtype in (DataType.FP16, DataType.FP32):
        values = values / 4
    return values.to(dtype.torch_dtype).contiguous()


def _row_data(m: int, dtype: DataType, carrier_cols: int = 1) -> torch.Tensor:
    values = torch.arange(m, dtype=torch.float32).reshape(m, 1).remainder(9) - 4
    if dtype in (DataType.FP16, DataType.FP32):
        values = values / 2
    carrier = values.repeat(1, carrier_cols)
    if carrier_cols > 1:
        # Distinct, in-range values in the remaining packed lanes verify that
        # the complete 32-byte lane block repeats across destination columns.
        lane_values = 17 + torch.arange(carrier_cols - 1, dtype=torch.float32).remainder(7)
        carrier[:, 1:] = lane_values
    return carrier.to(dtype.torch_dtype).contiguous()


class TileRowExpandAddCase(PTOTestCase):
    """Direct exact-op case using either legal row-vector carrier form."""

    __test__ = False

    def __init__(
        self,
        *,
        m: int = 32,
        n: int = 64,
        dtype: DataType = DataType.FP32,
        valid_shape: tuple[int, int] | None = None,
        use_tmp: bool = False,
        packed_row_vector: bool = False,
        tmp_shape: tuple[int, int] | None = None,
        tmp_dtype: DataType | None = None,
        platform: str,
    ):
        super().__init__(_run_config(dtype), platform=platform)
        self.m = m
        self.n = n
        self.dtype = dtype
        self.valid_shape = valid_shape
        self.use_tmp = use_tmp
        self.packed_row_vector = packed_row_vector
        self.tmp_shape = tmp_shape
        self.tmp_dtype = tmp_dtype
        self._platform = platform

    @property
    def row_vector_cols(self) -> int:
        if not self.packed_row_vector:
            return 1
        return 32 // self.dtype.torch_dtype.itemsize

    def get_name(self) -> str:
        valid = self.valid_shape or (self.m, self.n)
        signature = "with_tmp" if self.use_tmp else "without_tmp"
        carrier = "packed" if self.packed_row_vector else "column"
        tmp_kind = ""
        if self.use_tmp and (self.tmp_shape is not None or self.tmp_dtype is not None):
            tmp_shape = self.tmp_shape or (self.m, self.n)
            tmp_dtype = self.tmp_dtype or self.dtype
            tmp_kind = f"_tmp_{tmp_dtype.value}_{tmp_shape[0]}x{tmp_shape[1]}"
        return (
            f"tile_row_expand_add_{self.dtype.value}_{self.m}x{self.n}"
            f"_v{valid[0]}x{valid[1]}_{carrier}_{signature}{tmp_kind}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _main_data(self.m, self.n, self.dtype),
            ),
            TensorSpec(
                "row_vec",
                [self.m, self.row_vector_cols],
                self.dtype,
                init_value=lambda: _row_data(self.m, self.dtype, self.row_vector_cols),
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
        valid_shape = list(self.valid_shape or (m, n))
        valid_rows = valid_shape[0]
        use_tmp = self.use_tmp
        row_cols = self.row_vector_cols
        tmp_m, tmp_n = self.tmp_shape or (m, n)
        if use_tmp and self.tmp_shape is None and self._platform == "a2a3":
            # PTOAS v0.60's A2/A3 tmp-form reserves an 8 KiB workspace.
            elem_bytes = self.dtype.torch_dtype.itemsize
            tmp_n = max(tmp_n, (8192 + m * elem_bytes - 1) // (m * elem_bytes))
        tmp_dtype = _PL_DT[self.tmp_dtype or self.dtype]

        if use_tmp:

            @pl.program
            class RowExpandAddWithTmpProgram:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    src: pl.Tensor[[m, n], dtype],
                    row_vec: pl.Tensor[[m, row_cols], dtype],
                    out: pl.InOut[pl.Tensor[[m, n], dtype]],
                ) -> pl.Tensor[[m, n], dtype]:
                    src_tile: pl.Tile[[m, n], dtype] = pl.load(src, [0, 0], [m, n], valid_shape=valid_shape)
                    row_tile: pl.Tile[[m, row_cols], dtype] = pl.load(
                        row_vec,
                        [0, 0],
                        [m, row_cols],
                        valid_shape=[valid_rows, row_cols],
                    )
                    tmp: pl.Tile[[tmp_m, tmp_n], tmp_dtype] = pl.tile.create(
                        [tmp_m, tmp_n], dtype=tmp_dtype, target_memory=pl.MemorySpace.Vec
                    )
                    result: pl.Tile[[m, n], dtype] = pl.tile.row_expand_add(src_tile, row_tile, tmp=tmp)
                    return pl.store(result, [0, 0], out)

                @pl.function(type=pl.FunctionType.Orchestration)
                def orchestrator(
                    self,
                    src: pl.Tensor[[m, n], dtype],
                    row_vec: pl.Tensor[[m, row_cols], dtype],
                    out: pl.InOut[pl.Tensor[[m, n], dtype]],
                ) -> pl.Tensor[[m, n], dtype]:
                    return self.kernel(src, row_vec, out)

            return RowExpandAddWithTmpProgram

        @pl.program
        class RowExpandAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[m, n], dtype],
                row_vec: pl.Tensor[[m, row_cols], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                src_tile: pl.Tile[[m, n], dtype] = pl.load(src, [0, 0], [m, n], valid_shape=valid_shape)
                row_tile: pl.Tile[[m, row_cols], dtype] = pl.load(
                    row_vec,
                    [0, 0],
                    [m, row_cols],
                    valid_shape=[valid_rows, row_cols],
                )
                result: pl.Tile[[m, n], dtype] = pl.tile.row_expand_add(src_tile, row_tile)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[m, n], dtype],
                row_vec: pl.Tensor[[m, row_cols], dtype],
                out: pl.InOut[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                return self.kernel(src, row_vec, out)

        return RowExpandAddProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows, valid_cols = self.valid_shape or (self.m, self.n)
        expected = torch.zeros_like(tensors["out"])
        row_values = tensors["row_vec"][:valid_rows]
        if self.packed_row_vector:
            repeats = (valid_cols + self.row_vector_cols - 1) // self.row_vector_cols
            row_values = row_values.repeat(1, repeats)[:, :valid_cols]
        expected[:valid_rows, :valid_cols] = tensors["src"][:valid_rows, :valid_cols] + row_values
        tensors["out"][:] = expected


class TensorRowExpandAddCase(PTOTestCase):
    """Tensor frontend lowered to ``tile.row_expand_add``."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType,
        platform: str,
        m: int = 32,
        n: int = 64,
    ):
        super().__init__(_run_config(dtype), platform=platform)
        self.m = m
        self.n = n
        self.dtype = dtype

    def get_name(self) -> str:
        return f"tensor_row_expand_add_{self.dtype.value}_{self.m}x{self.n}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src",
                [self.m, self.n],
                self.dtype,
                init_value=lambda: _main_data(self.m, self.n, self.dtype),
            ),
            TensorSpec(
                "row_vec",
                [self.m, 1],
                self.dtype,
                init_value=lambda: _row_data(self.m, self.dtype),
            ),
            TensorSpec("out", [self.m, self.n], self.dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        m, n = self.m, self.n
        dtype = _PL_DT[self.dtype]

        @pl.program
        class TensorRowExpandAddProgram:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                src: pl.Tensor[[m, n], dtype],
                row_vec: pl.Tensor[[m, 1], dtype],
                out: pl.Out[pl.Tensor[[m, n], dtype]],
            ) -> pl.Tensor[[m, n], dtype]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[m, n], dtype] = pl.tensor.row_expand_add(src, row_vec)
                    out = pl.assemble(out, result, [0, 0])
                return out

        return TensorRowExpandAddProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = tensors["src"] + tensors["row_vec"]


_COMMON_DTYPE_CASES = [
    pytest.param(DataType.FP32, 32, 64, id="f32-32x64"),
    pytest.param(DataType.FP16, 16, 128, id="f16-16x128"),
    pytest.param(DataType.INT32, 32, 64, id="i32-32x64"),
    pytest.param(DataType.INT16, 16, 192, id="i16-16x192"),
]

_VALID_SHAPE_CASES = [
    pytest.param((20, 64), id="row-tail"),
    pytest.param((32, 47), id="column-tail"),
    pytest.param((20, 47), id="row-column-tail"),
]


class TestTileRowExpandAdd:
    """All dtype, architecture, signature, and valid-region branches."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("dtype,m,n", _COMMON_DTYPE_CASES)
    def test_common_dtypes_without_tmp(self, test_runner, platform, dtype, m, n):
        result = test_runner.run(TileRowExpandAddCase(m=m, n=n, dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_DEVICE_PLATFORMS)
    def test_a5_int8_without_tmp(self, test_runner, platform):
        # Latest PTOAS/A5 device supports i8. The shared CPU simulator stub
        # rejects it at C++ compile time, so this is intentionally device-only.
        result = test_runner.run(TileRowExpandAddCase(dtype=DataType.INT8, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("dtype,m,n", _COMMON_DTYPE_CASES)
    def test_packed_row_vector_without_tmp(self, test_runner, platform, dtype, m, n):
        result = test_runner.run(
            TileRowExpandAddCase(
                m=m,
                n=n,
                dtype=dtype,
                packed_row_vector=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_packed_row_vector_without_tmp_with_combined_tail(self, test_runner, platform):
        result = test_runner.run(
            TileRowExpandAddCase(
                valid_shape=(20, 47),
                packed_row_vector=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_DEVICE_PLATFORMS)
    def test_a5_int8_packed_without_tmp(self, test_runner, platform):
        result = test_runner.run(
            TileRowExpandAddCase(
                dtype=DataType.INT8,
                packed_row_vector=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("dtype,m,n", _COMMON_DTYPE_CASES)
    def test_optional_tmp_common_dtypes(self, test_runner, platform, dtype, m, n):
        result = test_runner.run(
            TileRowExpandAddCase(
                m=m,
                n=n,
                dtype=dtype,
                use_tmp=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_DEVICE_PLATFORMS)
    def test_a5_int8_optional_tmp(self, test_runner, platform):
        result = test_runner.run(
            TileRowExpandAddCase(
                dtype=DataType.INT8,
                use_tmp=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("valid_shape", _VALID_SHAPE_CASES)
    def test_valid_shape_tails(self, test_runner, platform, valid_shape):
        result = test_runner.run(TileRowExpandAddCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_optional_tmp_with_combined_tail(self, test_runner, platform):
        result = test_runner.run(TileRowExpandAddCase(valid_shape=(20, 47), use_tmp=True, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_PLATFORMS)
    @pytest.mark.parametrize(
        "dtype,m,n",
        _COMMON_DTYPE_CASES,
    )
    def test_a5_packed_optional_tmp(self, test_runner, platform, dtype, m, n):
        # The A5 overload ignores tmp, so both physical carrier forms are legal.
        # A2/A3's tmp-taking backend form is intentionally covered only with the
        # non-row-major [M, 1] carrier above.
        result = test_runner.run(
            TileRowExpandAddCase(
                m=m,
                n=n,
                dtype=dtype,
                packed_row_vector=True,
                use_tmp=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_PLATFORMS)
    def test_a5_packed_optional_tmp_with_combined_tail(self, test_runner, platform):
        result = test_runner.run(
            TileRowExpandAddCase(
                valid_shape=(20, 47),
                packed_row_vector=True,
                use_tmp=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_DEVICE_PLATFORMS)
    def test_a5_int8_packed_optional_tmp(self, test_runner, platform):
        result = test_runner.run(
            TileRowExpandAddCase(
                dtype=DataType.INT8,
                packed_row_vector=True,
                use_tmp=True,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_PLATFORMS)
    def test_a5_small_mismatched_tmp_placeholder(self, test_runner, platform):
        # PTOAS keeps tmp only for ABI compatibility on A5. A small, differently
        # typed but well-formed Vec tile proves that the backend does not consume
        # it as A2/A3 workspace.
        result = test_runner.run(
            TileRowExpandAddCase(
                dtype=DataType.FP32,
                use_tmp=True,
                tmp_shape=(16, 1),
                tmp_dtype=DataType.INT16,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


class TestTensorRowExpandAdd:
    """Tensor frontend and conversion coverage."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(DataType.FP32, id="f32"),
            pytest.param(DataType.INT16, id="i16"),
        ],
    )
    def test_tensor_lowering(self, test_runner, platform, dtype):
        result = test_runner.run(TensorRowExpandAddCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", _A5_DEVICE_PLATFORMS)
    def test_tensor_int8_lowering(self, test_runner, platform):
        result = test_runner.run(TensorRowExpandAddCase(dtype=DataType.INT8, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
