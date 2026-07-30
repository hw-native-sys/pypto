# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 coverage for the cumulative ``thistogram`` instruction."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 32
N = 32


def _src16() -> torch.Tensor:
    rows = torch.arange(M, dtype=torch.int32).reshape(M, 1)
    cols = torch.arange(N, dtype=torch.int32).reshape(1, N)
    return ((rows << 8) | ((cols * 17 + rows) & 0xFF)).to(torch.uint16).contiguous()


def _idx16() -> torch.Tensor:
    return torch.arange(M, dtype=torch.uint8).reshape(1, M).contiguous()


def _src32() -> torch.Tensor:
    rows = torch.arange(M, dtype=torch.int64).reshape(M, 1)
    cols = torch.arange(N, dtype=torch.int64).reshape(1, N)
    low = (cols * 13 + rows * 7 + 5) & 0xFF
    high = torch.where(cols.remainder(3) == 0, 0x99, 0x12)
    return ((high << 24) | (0x34 << 16) | (0x56 << 8) | low).to(torch.uint32).contiguous()


def _idx32(rows: int) -> torch.Tensor:
    values = torch.tensor([0x12, 0x34, 0x56], dtype=torch.uint8).reshape(3, 1)
    return values[:rows].expand(rows, N).contiguous()


def _histogram16(byte: int, valid_shape: tuple[int, int]):
    valid_rows, valid_cols = valid_shape

    @pl.program
    class Histogram16:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[M, N], pl.UINT16],
            idx: pl.Tensor[[1, M], pl.UINT8],
            out: pl.Out[pl.Tensor[[M, 256], pl.UINT32]],
        ) -> pl.Tensor[[M, 256], pl.UINT32]:
            src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=[valid_rows, valid_cols])
            idx_row = pl.load(idx, [0, 0], [1, M], valid_shapes=[1, valid_rows])
            idx_col = pl.tile.reshape(idx_row, [M, 1])
            result = pl.tile.histogram(src_tile, idx_col, byte=byte)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[M, N], pl.UINT16],
            idx: pl.Tensor[[1, M], pl.UINT8],
            out: pl.Out[pl.Tensor[[M, 256], pl.UINT32]],
        ) -> pl.Tensor[[M, 256], pl.UINT32]:
            return self.kernel(src, idx, out)

    return Histogram16


def _histogram32(byte: int, idx_rows: int, valid_shape: tuple[int, int]):
    valid_rows, valid_cols = valid_shape

    @pl.program
    class Histogram32:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[M, N], pl.UINT32],
            idx: pl.Tensor[[idx_rows, N], pl.UINT8],
            out: pl.Out[pl.Tensor[[M, 256], pl.UINT32]],
        ) -> pl.Tensor[[M, 256], pl.UINT32]:
            src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=[valid_rows, valid_cols])
            idx_tile = pl.load(
                idx,
                [0, 0],
                [idx_rows, N],
                valid_shapes=[idx_rows, valid_cols],
            )
            result = pl.tile.histogram(src_tile, idx_tile, byte=byte)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[M, N], pl.UINT32],
            idx: pl.Tensor[[idx_rows, N], pl.UINT8],
            out: pl.Out[pl.Tensor[[M, 256], pl.UINT32]],
        ) -> pl.Tensor[[M, 256], pl.UINT32]:
            return self.kernel(src, idx, out)

    return Histogram32


def _cumulative(values: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(values.to(torch.int64), minlength=256)
    return torch.cumsum(counts, dim=0).to(torch.uint32)


class HistogramTestCase(PTOTestCase):
    __test__ = False

    def __init__(
        self,
        dtype: DataType,
        byte: int,
        valid_shape: tuple[int, int],
        *,
        platform=None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self._dtype = dtype
        self._byte = byte
        self._valid_shape = valid_shape

    def get_name(self) -> str:
        dtype_name = "uint16" if self._dtype == DataType.UINT16 else "uint32"
        valid_tag = f"v{self._valid_shape[0]}x{self._valid_shape[1]}"
        return f"histogram_{dtype_name}_byte{self._byte}_{valid_tag}"

    def define_tensors(self) -> list[TensorSpec]:
        if self._dtype == DataType.UINT16:
            return [
                TensorSpec("src", [M, N], DataType.UINT16, init_value=_src16),
                TensorSpec("idx", [1, M], DataType.UINT8, init_value=_idx16),
                TensorSpec("out", [M, 256], DataType.UINT32, is_output=True),
            ]
        rows = 3 if self._byte == 0 else 2 if self._byte == 1 else 1
        return [
            TensorSpec("src", [M, N], DataType.UINT32, init_value=_src32),
            TensorSpec("idx", [rows, N], DataType.UINT8, init_value=lambda: _idx32(rows)),
            TensorSpec("out", [M, 256], DataType.UINT32, is_output=True),
        ]

    def get_program(self) -> Any:
        if self._dtype == DataType.UINT16:
            return _histogram16(self._byte, self._valid_shape)
        rows = 3 if self._byte == 0 else 2 if self._byte == 1 else 1
        return _histogram32(self._byte, rows, self._valid_shape)

    def compute_expected(self, tensors, params=None):
        valid_rows, valid_cols = self._valid_shape
        src = tensors["src"].to(torch.int64)
        tensors["out"].zero_()
        if self._dtype == DataType.UINT16:
            for row in range(valid_rows):
                row_src = src[row, :valid_cols]
                values = (row_src >> (8 * self._byte)) & 0xFF
                if self._byte == 0:
                    values = values[((row_src >> 8) & 0xFF) == row]
                tensors["out"][row] = _cumulative(values)
            return

        for row in range(valid_rows):
            row_src = src[row, :valid_cols]
            values = (row_src >> (8 * self._byte)) & 0xFF
            if self._byte < 3:
                selected = torch.ones(valid_cols, dtype=torch.bool)
                for filter_byte in range(self._byte + 1, 4):
                    idx_row = 3 - filter_byte
                    selected &= ((row_src >> (8 * filter_byte)) & 0xFF) == tensors["idx"][
                        idx_row, 0
                    ].to(torch.int64)
                values = values[selected]
            tensors["out"][row] = _cumulative(values)


_FULL = (M, N)
_ROW_TAIL = (11, N)
_COL_TAIL = (M, 23)
_COMBINED_TAIL = (11, 23)
_CASES = [
    *[
        pytest.param(dtype, byte, valid_shape, id=f"{dtype.value}-byte{byte}-{shape_id}")
        for dtype, bytes_ in (
            (DataType.UINT16, (0, 1)),
            (DataType.UINT32, (0, 1, 2, 3)),
        )
        for byte in bytes_
        for valid_shape, shape_id in (
            (_FULL, "full"),
            (_COMBINED_TAIL, "combined-tail"),
        )
    ],
    *[
        pytest.param(dtype, 0, valid_shape, id=f"{dtype.value}-byte0-{shape_id}")
        for dtype in (DataType.UINT16, DataType.UINT32)
        for valid_shape, shape_id in (
            (_ROW_TAIL, "row-tail"),
            (_COL_TAIL, "col-tail"),
        )
    ],
]


@pytest.mark.platforms("a5", "a5sim")
@pytest.mark.parametrize(
    "platform",
    [pytest.param("a5", id="a5"), pytest.param("a5sim", id="a5sim")],
)
@pytest.mark.parametrize("dtype,byte,valid_shape", _CASES)
def test_histogram(test_runner, platform, dtype, byte, valid_shape):
    result = test_runner.run(
        HistogramTestCase(dtype, byte, valid_shape, platform=platform)
    )
    assert result.passed, f"Test failed: {result.error}"
