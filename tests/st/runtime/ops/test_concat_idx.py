# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTOAS contract coverage for indexed per-row ``tconcatidx``."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 8
SRC_N = 32
DST_N = 48
IDX_N = 8

FULL = (M, DST_N)
ROW_TAIL = (5, DST_N)
COL_TAIL = (M, 24)
COMBINED_TAIL = (5, 24)

_PL_DT = {
    DataType.INT8: pl.INT8,
    DataType.UINT8: pl.UINT8,
    DataType.INT16: pl.INT16,
    DataType.UINT16: pl.UINT16,
    DataType.INT32: pl.INT32,
    DataType.UINT32: pl.UINT32,
    DataType.FP16: pl.FP16,
    DataType.BF16: pl.BF16,
    DataType.FP32: pl.FP32,
}
_TORCH_DT = {
    DataType.INT8: torch.int8,
    DataType.UINT8: torch.uint8,
    DataType.INT16: torch.int16,
    DataType.UINT16: torch.uint16,
    DataType.INT32: torch.int32,
    DataType.UINT32: torch.uint32,
    DataType.FP16: torch.float16,
    DataType.BF16: torch.bfloat16,
    DataType.FP32: torch.float32,
}
_DATA_DTYPES = [
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
    DataType.FP16,
    DataType.BF16,
    DataType.FP32,
]
_INDEX_DTYPES = [
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
]


def _data(dtype: DataType, offset: int) -> torch.Tensor:
    values = (
        torch.arange(M * SRC_N, dtype=torch.int64).reshape(M, SRC_N).remainder(53)
        + offset
    )
    return values.to(_TORCH_DT[dtype]).contiguous()


def _destination(dtype: DataType) -> torch.Tensor:
    return torch.full((M, DST_N), 3, dtype=_TORCH_DT[dtype])


def _index_values(
    dtype: DataType,
    source: int,
    pattern: str,
) -> torch.Tensor:
    if pattern == "zero":
        counts0 = [0, 4, 8, 12, 16, 20, 6, 10]
        counts1 = [12, 0, 8, 4, 20, 6, 10, 14]
    elif pattern == "overflow":
        counts0 = [20, 18, 16, 14, 12, 10, 8, 6]
        counts1 = [20, 18, 16, 14, 12, 10, 8, 6]
    else:
        counts0 = [4, 8, 12, 16, 20, 6, 10, 14]
        counts1 = [12, 8, 4, 16, 6, 20, 14, 10]
    counts = counts0 if source == 0 else counts1
    element_bytes = torch.empty((), dtype=_TORCH_DT[dtype]).element_size()
    encoded = torch.tensor(counts, dtype=torch.int64) * element_bytes
    result = torch.zeros((M, IDX_N), dtype=_TORCH_DT[dtype])
    result[:, 0] = encoded.to(_TORCH_DT[dtype])
    return result.contiguous()


def _make_program(
    data_dtype: DataType,
    index_dtype: DataType,
    valid_shape: tuple[int, int],
):
    pl_data_dtype = _PL_DT[data_dtype]
    pl_index_dtype = _PL_DT[index_dtype]
    valid_rows, valid_cols = valid_shape
    src_valid_cols = min(SRC_N, valid_cols)

    @pl.program
    class ConcatIdxProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, SRC_N], pl_data_dtype],
            src1: pl.Tensor[[M, SRC_N], pl_data_dtype],
            idx0: pl.Tensor[[M, IDX_N], pl_index_dtype],
            idx1: pl.Tensor[[M, IDX_N], pl_index_dtype],
            out: pl.InOut[pl.Tensor[[M, DST_N], pl_data_dtype]],
        ) -> pl.Tensor[[M, DST_N], pl_data_dtype]:
            value0 = pl.load(
                src0,
                [0, 0],
                [M, SRC_N],
                valid_shapes=[valid_rows, src_valid_cols],
            )
            value1 = pl.load(
                src1,
                [0, 0],
                [M, SRC_N],
                valid_shapes=[valid_rows, src_valid_cols],
            )
            count0 = pl.load(
                idx0,
                [0, 0],
                [M, IDX_N],
                valid_shapes=[valid_rows, 1],
            )
            count1 = pl.load(
                idx1,
                [0, 0],
                [M, IDX_N],
                valid_shapes=[valid_rows, 1],
            )
            dst = pl.load(
                out,
                [0, 0],
                [M, DST_N],
                valid_shapes=[valid_rows, valid_cols],
            )
            result = pl.tile.concat_idx(value0, value1, count0, count1, dst)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, SRC_N], pl_data_dtype],
            src1: pl.Tensor[[M, SRC_N], pl_data_dtype],
            idx0: pl.Tensor[[M, IDX_N], pl_index_dtype],
            idx1: pl.Tensor[[M, IDX_N], pl_index_dtype],
            out: pl.InOut[pl.Tensor[[M, DST_N], pl_data_dtype]],
        ) -> pl.Tensor[[M, DST_N], pl_data_dtype]:
            return self.kernel(src0, src1, idx0, idx1, out)

    return ConcatIdxProgram


class ConcatIdxCase(PTOTestCase):
    __test__ = False

    def __init__(
        self,
        data_dtype: DataType,
        index_dtype: DataType,
        valid_shape: tuple[int, int],
        pattern: str,
        *,
        platform: str,
    ):
        super().__init__(platform=platform)
        self.data_dtype = data_dtype
        self.index_dtype = index_dtype
        self.valid_shape = valid_shape
        self.pattern = pattern

    def get_name(self) -> str:
        valid_tag = f"v{self.valid_shape[0]}x{self.valid_shape[1]}"
        return (
            f"concat_idx_{self.data_dtype.value}_{self.index_dtype.value}"
            f"_{valid_tag}_{self.pattern}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "src0",
                [M, SRC_N],
                self.data_dtype,
                init_value=lambda: _data(self.data_dtype, 1),
            ),
            TensorSpec(
                "src1",
                [M, SRC_N],
                self.data_dtype,
                init_value=lambda: _data(self.data_dtype, 61),
            ),
            TensorSpec(
                "idx0",
                [M, IDX_N],
                self.index_dtype,
                init_value=lambda: _index_values(self.index_dtype, 0, self.pattern),
            ),
            TensorSpec(
                "idx1",
                [M, IDX_N],
                self.index_dtype,
                init_value=lambda: _index_values(self.index_dtype, 1, self.pattern),
            ),
            TensorSpec(
                "out",
                [M, DST_N],
                self.data_dtype,
                init_value=lambda: _destination(self.data_dtype),
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        return _make_program(self.data_dtype, self.index_dtype, self.valid_shape)

    def compute_expected(self, tensors, params=None):
        valid_rows, valid_cols = self.valid_shape
        element_bytes = tensors["idx0"].element_size()
        expected = _destination(self.data_dtype)
        for row in range(valid_rows):
            count0 = min(int(tensors["idx0"][row, 0].item()) // element_bytes, valid_cols)
            remaining = valid_cols - count0
            count1 = min(int(tensors["idx1"][row, 0].item()) // element_bytes, remaining)
            expected[row, :count0] = tensors["src0"][row, :count0]
            expected[row, count0 : count0 + count1] = tensors["src1"][row, :count1]
        tensors["out"][:] = expected


def _case(
    platform: str,
    data_dtype: DataType,
    index_dtype: DataType,
    valid_shape: tuple[int, int],
    pattern: str = "standard",
):
    return pytest.param(
        platform,
        data_dtype,
        index_dtype,
        valid_shape,
        pattern,
        id=(
            f"{platform}-{data_dtype.value}-{index_dtype.value}"
            f"-v{valid_shape[0]}x{valid_shape[1]}-{pattern}"
        ),
    )


_CASES = []
for _platform in ("a2a3", "a5"):
    for _data_dtype in _DATA_DTYPES:
        _CASES.append(
            _case(
                _platform,
                _data_dtype,
                DataType.INT32,
                COMBINED_TAIL,
            )
        )
    for _index_dtype in _INDEX_DTYPES:
        _CASES.append(
            _case(
                _platform,
                DataType.FP32,
                _index_dtype,
                COMBINED_TAIL,
            )
        )
    for _valid_shape in (FULL, ROW_TAIL, COL_TAIL):
        _CASES.append(
            _case(
                _platform,
                DataType.FP32,
                DataType.INT32,
                _valid_shape,
            )
        )
    _CASES.append(
        _case(
            _platform,
            DataType.FP32,
            DataType.INT32,
            COL_TAIL,
            "overflow",
        )
    )

# Zero-length indexed segments are valid on A5. The pinned A2/A3 implementation
# underflows its repeat count for a zero segment, so that upstream defect is
# documented rather than represented by a knowingly crashing A2/A3 case.
_CASES.append(
    _case(
        "a5",
        DataType.FP32,
        DataType.INT32,
        COMBINED_TAIL,
        "zero",
    )
)


@pytest.mark.platforms("a2a3", "a5")
@pytest.mark.parametrize(
    "platform,data_dtype,index_dtype,valid_shape,pattern",
    _CASES,
)
def test_concat_idx(
    test_runner,
    platform,
    data_dtype,
    index_dtype,
    valid_shape,
    pattern,
):
    result = test_runner.run(
        ConcatIdxCase(
            data_dtype,
            index_dtype,
            valid_shape,
            pattern,
            platform=platform,
        )
    )
    assert result.passed, f"Test failed: {result.error}"
