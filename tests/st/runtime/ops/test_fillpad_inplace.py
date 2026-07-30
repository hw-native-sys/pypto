# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTOAS mode, dtype, alias, and valid-shape coverage for TFILLPAD_INPLACE."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 64

FULL = (M, N)
ROW_TAIL = (11, N)
COL_TAIL = (M, 47)
COMBINED_TAIL = (11, 47)
MIN_BOUNDARY = (1, 1)

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
_DTYPES = list(_PL_DT)
_PAD_MODES = {
    "zero": pl.PadValue.zero,
    "max": pl.PadValue.max,
    "min": pl.PadValue.min,
}


def _input(dtype: DataType) -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(31) + 1
    return values.to(_TORCH_DT[dtype]).contiguous()


def _pad_scalar(dtype: DataType, mode: str) -> int | float:
    if mode == "zero":
        return 0
    torch_dtype = _TORCH_DT[dtype]
    if torch_dtype.is_floating_point:
        return float("inf") if mode == "max" else float("-inf")
    limits = torch.iinfo(torch_dtype)
    return limits.max if mode == "max" else limits.min


def _make_program(
    dtype: DataType,
    mode: str,
    valid_shape: tuple[int, int],
):
    pl_dtype = _PL_DT[dtype]
    pad_mode = _PAD_MODES[mode]
    valid = list(valid_shape)

    @pl.program
    class FillpadInplaceProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            input_tensor: pl.Tensor[[M, N], pl_dtype],
            output: pl.Out[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            tile = pl.load(
                input_tensor,
                offsets=[0, 0],
                shapes=[M, N],
                valid_shapes=valid,
            )
            padded = pl.tile.fillpad_inplace(tile, pad_value=pad_mode)
            return pl.store(padded, offsets=[0, 0], output_tensor=output)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            input_tensor: pl.Tensor[[M, N], pl_dtype],
            output: pl.Out[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            return self.kernel(input_tensor, output)

    return FillpadInplaceProgram


class FillpadInplaceCase(PTOTestCase):
    __test__ = False

    def __init__(
        self,
        dtype: DataType,
        mode: str,
        valid_shape: tuple[int, int],
        *,
        platform: str,
    ):
        super().__init__(platform=platform)
        self.dtype = dtype
        self.mode = mode
        self.valid_shape = valid_shape

    def get_name(self) -> str:
        valid_tag = f"v{self.valid_shape[0]}x{self.valid_shape[1]}"
        return f"fillpad_inplace_{self.dtype.value}_{self.mode}_{valid_tag}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "input_tensor",
                [M, N],
                self.dtype,
                init_value=lambda: _input(self.dtype),
            ),
            TensorSpec("output", [M, N], self.dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_program(self.dtype, self.mode, self.valid_shape)

    def compute_expected(self, tensors, params=None):
        valid_rows, valid_cols = self.valid_shape
        fill = _pad_scalar(self.dtype, self.mode)
        expected = torch.full(
            (M, N),
            fill,
            dtype=_TORCH_DT[self.dtype],
        )
        expected[:valid_rows, :valid_cols] = tensors["input_tensor"][
            :valid_rows, :valid_cols
        ]
        tensors["output"][:] = expected


def _case(
    platform: str,
    dtype: DataType,
    mode: str,
    valid_shape: tuple[int, int],
):
    return pytest.param(
        platform,
        dtype,
        mode,
        valid_shape,
        id=f"{platform}-{dtype.value}-{mode}-v{valid_shape[0]}x{valid_shape[1]}",
    )


_CASES = []
for _platform in ("a2a3", "a5"):
    for _dtype in _DTYPES:
        for _mode in _PAD_MODES:
            _CASES.append(
                _case(
                    _platform,
                    _dtype,
                    _mode,
                    COMBINED_TAIL,
                )
            )
    for _mode in _PAD_MODES:
        for _valid_shape in (FULL, ROW_TAIL, COL_TAIL):
            _CASES.append(
                _case(
                    _platform,
                    DataType.FP32,
                    _mode,
                    _valid_shape,
                )
            )
    _CASES.append(
        _case(
            _platform,
            DataType.FP32,
            "zero",
            MIN_BOUNDARY,
        )
    )


@pytest.mark.platforms("a2a3", "a5")
@pytest.mark.parametrize("platform,dtype,mode,valid_shape", _CASES)
def test_fillpad_inplace(
    test_runner,
    platform,
    dtype,
    mode,
    valid_shape,
):
    result = test_runner.run(
        FillpadInplaceCase(
            dtype,
            mode,
            valid_shape,
            platform=platform,
        )
    )
    assert result.passed, f"Test failed: {result.error}"
