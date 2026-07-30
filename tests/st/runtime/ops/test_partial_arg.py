# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime coverage for ``tpartargmax`` and ``tpartargmin``.

The source valid regions cover equal, source-0-dominant, and
source-1-dominant cases. Within their overlap, source 0 wins ties on the
pinned A2/A3 and A5 implementations; outside it, the only valid source is
copied together with its paired index.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 16

_PL_VALUE_DTYPES = {
    DataType.FP16: pl.FP16,
    DataType.FP32: pl.FP32,
}
_PL_INDEX_DTYPES = {
    DataType.INT32: pl.INT32,
    DataType.UINT32: pl.UINT32,
}


def _torch_value_dtype(dtype: DataType) -> torch.dtype:
    return torch.float16 if dtype == DataType.FP16 else torch.float32


def _torch_index_dtype(dtype: DataType) -> torch.dtype:
    return torch.int32 if dtype == DataType.INT32 else torch.uint32


def _src0(dtype: DataType) -> torch.Tensor:
    return (
        (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(11) - 5)
        .to(_torch_value_dtype(dtype))
        .contiguous()
    )


def _src1(dtype: DataType) -> torch.Tensor:
    values = (torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(7) - 3).to(
        _torch_value_dtype(dtype)
    )
    values[:, ::5] = _src0(dtype)[:, ::5]
    return values.contiguous()


def _idx0(dtype: DataType) -> torch.Tensor:
    return (
        torch.arange(M * N, dtype=torch.int64)
        .reshape(M, N)
        .to(_torch_index_dtype(dtype))
        .contiguous()
    )


def _idx1(dtype: DataType) -> torch.Tensor:
    return (
        (1000 + torch.arange(M * N, dtype=torch.int64))
        .reshape(M, N)
        .to(_torch_index_dtype(dtype))
        .contiguous()
    )


def _part_argmax(
    value_dtype: DataType,
    index_dtype: DataType,
    src0_valid: tuple[int, int],
    src1_valid: tuple[int, int],
):
    pl_value_dtype = _PL_VALUE_DTYPES[value_dtype]
    pl_index_dtype = _PL_INDEX_DTYPES[index_dtype]
    src0_valid_shape = list(src0_valid)
    src1_valid_shape = list(src1_valid)

    @pl.program
    class PartArgMax:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl_value_dtype],
            src1: pl.Tensor[[M, N], pl_value_dtype],
            idx0: pl.Tensor[[M, N], pl_index_dtype],
            idx1: pl.Tensor[[M, N], pl_index_dtype],
            value_out: pl.Out[pl.Tensor[[M, N], pl_value_dtype]],
            index_out: pl.Out[pl.Tensor[[M, N], pl_index_dtype]],
        ) -> tuple[pl.Tensor[[M, N], pl_value_dtype], pl.Tensor[[M, N], pl_index_dtype]]:
            value0 = pl.load(src0, [0, 0], [M, N], valid_shapes=src0_valid_shape)
            value1 = pl.load(src1, [0, 0], [M, N], valid_shapes=src1_valid_shape)
            index0 = pl.load(idx0, [0, 0], [M, N], valid_shapes=src0_valid_shape)
            index1 = pl.load(idx1, [0, 0], [M, N], valid_shapes=src1_valid_shape)
            value, index = pl.tile.part_argmax(value0, value1, index0, index1)
            value_out = pl.store(value, [0, 0], value_out)
            index_out = pl.store(index, [0, 0], index_out)
            return value_out, index_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl_value_dtype],
            src1: pl.Tensor[[M, N], pl_value_dtype],
            idx0: pl.Tensor[[M, N], pl_index_dtype],
            idx1: pl.Tensor[[M, N], pl_index_dtype],
            value_out: pl.Out[pl.Tensor[[M, N], pl_value_dtype]],
            index_out: pl.Out[pl.Tensor[[M, N], pl_index_dtype]],
        ) -> tuple[pl.Tensor[[M, N], pl_value_dtype], pl.Tensor[[M, N], pl_index_dtype]]:
            return self.kernel(src0, src1, idx0, idx1, value_out, index_out)

    return PartArgMax


def _part_argmin(
    value_dtype: DataType,
    index_dtype: DataType,
    src0_valid: tuple[int, int],
    src1_valid: tuple[int, int],
):
    pl_value_dtype = _PL_VALUE_DTYPES[value_dtype]
    pl_index_dtype = _PL_INDEX_DTYPES[index_dtype]
    src0_valid_shape = list(src0_valid)
    src1_valid_shape = list(src1_valid)

    @pl.program
    class PartArgMin:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl_value_dtype],
            src1: pl.Tensor[[M, N], pl_value_dtype],
            idx0: pl.Tensor[[M, N], pl_index_dtype],
            idx1: pl.Tensor[[M, N], pl_index_dtype],
            value_out: pl.Out[pl.Tensor[[M, N], pl_value_dtype]],
            index_out: pl.Out[pl.Tensor[[M, N], pl_index_dtype]],
        ) -> tuple[pl.Tensor[[M, N], pl_value_dtype], pl.Tensor[[M, N], pl_index_dtype]]:
            value0 = pl.load(src0, [0, 0], [M, N], valid_shapes=src0_valid_shape)
            value1 = pl.load(src1, [0, 0], [M, N], valid_shapes=src1_valid_shape)
            index0 = pl.load(idx0, [0, 0], [M, N], valid_shapes=src0_valid_shape)
            index1 = pl.load(idx1, [0, 0], [M, N], valid_shapes=src1_valid_shape)
            value, index = pl.tile.part_argmin(value0, value1, index0, index1)
            value_out = pl.store(value, [0, 0], value_out)
            index_out = pl.store(index, [0, 0], index_out)
            return value_out, index_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl_value_dtype],
            src1: pl.Tensor[[M, N], pl_value_dtype],
            idx0: pl.Tensor[[M, N], pl_index_dtype],
            idx1: pl.Tensor[[M, N], pl_index_dtype],
            value_out: pl.Out[pl.Tensor[[M, N], pl_value_dtype]],
            index_out: pl.Out[pl.Tensor[[M, N], pl_index_dtype]],
        ) -> tuple[pl.Tensor[[M, N], pl_value_dtype], pl.Tensor[[M, N], pl_index_dtype]]:
            return self.kernel(src0, src1, idx0, idx1, value_out, index_out)

    return PartArgMin


class PartialArgTestCase(PTOTestCase):
    __test__ = False

    def __init__(
        self,
        op_name: str,
        value_dtype: DataType,
        index_dtype: DataType,
        src0_valid: tuple[int, int],
        src1_valid: tuple[int, int],
        *,
        platform=None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self._op_name = op_name
        self._value_dtype = value_dtype
        self._index_dtype = index_dtype
        self._src0_valid = src0_valid
        self._src1_valid = src1_valid

    def get_name(self) -> str:
        src0_tag = f"{self._src0_valid[0]}x{self._src0_valid[1]}"
        src1_tag = f"{self._src1_valid[0]}x{self._src1_valid[1]}"
        return (
            f"{self._op_name}_{self._value_dtype.value}_{self._index_dtype.value}"
            f"_s0-{src0_tag}_s1-{src1_tag}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src0", [M, N], self._value_dtype, init_value=lambda: _src0(self._value_dtype)),
            TensorSpec("src1", [M, N], self._value_dtype, init_value=lambda: _src1(self._value_dtype)),
            TensorSpec("idx0", [M, N], self._index_dtype, init_value=lambda: _idx0(self._index_dtype)),
            TensorSpec("idx1", [M, N], self._index_dtype, init_value=lambda: _idx1(self._index_dtype)),
            TensorSpec("value_out", [M, N], self._value_dtype, is_output=True),
            TensorSpec("index_out", [M, N], self._index_dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        factory = _part_argmax if self._op_name == "part_argmax" else _part_argmin
        return factory(
            self._value_dtype,
            self._index_dtype,
            self._src0_valid,
            self._src1_valid,
        )

    def compute_expected(self, tensors, params=None):
        src0 = tensors["src0"]
        src1 = tensors["src1"]
        rows = torch.arange(M).reshape(M, 1)
        cols = torch.arange(N).reshape(1, N)
        valid0 = (rows < self._src0_valid[0]) & (cols < self._src0_valid[1])
        valid1 = (rows < self._src1_valid[0]) & (cols < self._src1_valid[1])
        preferred0 = src0 >= src1 if self._op_name == "part_argmax" else src0 <= src1
        choose0 = valid0 & (~valid1 | preferred0)

        value = torch.zeros_like(src0)
        index = torch.zeros_like(tensors["idx0"])
        value[valid1] = src1[valid1]
        index[valid1] = tensors["idx1"][valid1]
        value[choose0] = src0[choose0]
        index[choose0] = tensors["idx0"][choose0]
        tensors["value_out"][:] = value
        tensors["index_out"][:] = index


_DTYPE_PAIRS = [
    (DataType.FP16, DataType.INT32),
    (DataType.FP16, DataType.UINT32),
    (DataType.FP32, DataType.INT32),
    (DataType.FP32, DataType.UINT32),
]
_VALID_SCENARIOS = [
    ((M, N), (M, N), "full"),
    ((11, N), (11, N), "row-tail"),
    ((M, 11), (M, 11), "col-tail"),
    ((11, 13), (11, 13), "combined-tail"),
    ((M, N), (11, 13), "src0-dominant"),
    ((11, 13), (M, N), "src1-dominant"),
]
_CASES = [
    *[
        pytest.param(
            op_name,
            value_dtype,
            index_dtype,
            (11, 13),
            (11, 13),
            id=f"{op_name}-{value_dtype.value}-{index_dtype.value}-combined",
        )
        for op_name in ("part_argmax", "part_argmin")
        for value_dtype, index_dtype in _DTYPE_PAIRS
    ],
    *[
        pytest.param(
            op_name,
            DataType.FP32,
            DataType.INT32,
            src0_valid,
            src1_valid,
            id=f"{op_name}-fp32-int32-{scenario}",
        )
        for op_name in ("part_argmax", "part_argmin")
        for src0_valid, src1_valid, scenario in _VALID_SCENARIOS
        if scenario != "combined-tail"
    ],
]


@pytest.mark.platforms("a2a3", "a5")
@pytest.mark.parametrize(
    "platform",
    [pytest.param("a2a3", id="a2a3"), pytest.param("a5", id="a5")],
)
@pytest.mark.parametrize(
    "op_name,value_dtype,index_dtype,src0_valid,src1_valid",
    _CASES,
)
def test_partial_arg(
    test_runner,
    platform,
    op_name,
    value_dtype,
    index_dtype,
    src0_valid,
    src1_valid,
):
    result = test_runner.run(
        PartialArgTestCase(
            op_name,
            value_dtype,
            index_dtype,
            src0_valid,
            src1_valid,
            platform=platform,
        )
    )
    assert result.passed, f"Test failed: {result.error}"
