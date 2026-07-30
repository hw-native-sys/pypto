# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact-op system tests for the tile shift families."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 32
N = 32

_VALID_SHAPES = {
    "full": (M, N),
    "rows": (17, N),
    "cols": (M, 23),
    "combined": (17, 23),
}

_PL_DT = {
    DataType.INT8: pl.INT8,
    DataType.UINT8: pl.UINT8,
    DataType.INT16: pl.INT16,
    DataType.UINT16: pl.UINT16,
    DataType.INT32: pl.INT32,
    DataType.UINT32: pl.UINT32,
}

_WIDTH = {
    DataType.INT8: 8,
    DataType.UINT8: 8,
    DataType.INT16: 16,
    DataType.UINT16: 16,
    DataType.INT32: 32,
    DataType.UINT32: 32,
}

_SIGNED_DTYPES = {DataType.INT8, DataType.INT16, DataType.INT32}
_ALL_DTYPES = tuple(_PL_DT)
_A2A3_SCALAR_DTYPES = (
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
)


def _values(dtype: DataType) -> torch.Tensor:
    width = _WIDTH[dtype]
    if dtype in _SIGNED_DTYPES:
        values = [-(1 << (width - 1)), -17, -1, 0, 1, 17, (1 << (width - 1)) - 1]
    else:
        values = [0, 1, 17, 1 << (width - 1), (1 << width) - 1]
    index = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(len(values))
    result = torch.zeros((M, N), dtype=torch.int64)
    for i, value in enumerate(values):
        result[index == i] = value
    return result.to(dtype.torch_dtype).contiguous()


def _shift_counts(dtype: DataType) -> torch.Tensor:
    width = _WIDTH[dtype]
    values = [0, 1, width - 1]
    index = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(len(values))
    result = torch.zeros((M, N), dtype=torch.int64)
    for i, value in enumerate(values):
        result[index == i] = value
    return result.to(dtype.torch_dtype).contiguous()


def _make_program(op_name: str, dtype: DataType, scalar: int | None, valid_shape: tuple[int, int]):
    pl_dtype = _PL_DT[dtype]
    valid = list(valid_shape)

    if op_name == "shl":

        @pl.program
        class ShlProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                shift: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=valid)
                shift_tile = pl.load(shift, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.shl(src_tile, shift_tile), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                shift: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src, shift, out)

        return ShlProgram

    if op_name == "shr":

        @pl.program
        class ShrProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                shift: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=valid)
                shift_tile = pl.load(shift, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.shr(src_tile, shift_tile), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                shift: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src, shift, out)

        return ShrProgram

    assert scalar is not None

    if op_name == "shls":

        @pl.program
        class ShlsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.shls(src_tile, scalar), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src, out)

        return ShlsProgram

    assert op_name == "shrs"

    @pl.program
    class ShrsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            src_tile = pl.load(src, [0, 0], [M, N], valid_shapes=valid)
            return pl.store(pl.tile.shrs(src_tile, scalar), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            return self.kernel(src, out)

    return ShrsProgram


class ShiftCase(PTOTestCase):
    """One direct shift-instruction case."""

    __test__ = False

    def __init__(
        self,
        *,
        op_name: str,
        dtype: DataType,
        valid_name: str,
        valid_shape: tuple[int, int],
        scalar: int | None = None,
    ):
        super().__init__()
        self.op_name = op_name
        self.dtype = dtype
        self.valid_name = valid_name
        self.valid_shape = valid_shape
        self.scalar = scalar

    def get_name(self) -> str:
        scalar_tag = f"_s{self.scalar}" if self.scalar is not None else ""
        rows, cols = self.valid_shape
        return f"tile_{self.op_name}_{self.dtype.value}_{self.valid_name}_v{rows}x{cols}{scalar_tag}"

    def define_tensors(self) -> list[TensorSpec]:
        specs = [TensorSpec("src", [M, N], self.dtype, init_value=lambda: _values(self.dtype))]
        if self.scalar is None:
            specs.append(
                TensorSpec("shift", [M, N], self.dtype, init_value=lambda: _shift_counts(self.dtype))
            )
        specs.append(TensorSpec("out", [M, N], self.dtype, init_value=torch.zeros, is_output=True))
        return specs

    def get_program(self) -> Any:
        return _make_program(self.op_name, self.dtype, self.scalar, self.valid_shape)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        rows, cols = self.valid_shape
        width = _WIDTH[self.dtype]
        src = tensors["src"][:rows, :cols].to(torch.int64)
        shift: torch.Tensor | int
        if self.scalar is None:
            shift = tensors["shift"][:rows, :cols].to(torch.int64)
        else:
            shift = self.scalar

        if self.op_name in {"shl", "shls"}:
            mask = (1 << width) - 1
            value = torch.bitwise_left_shift(src, shift) & mask
            if self.dtype in _SIGNED_DTYPES:
                sign = 1 << (width - 1)
                value = (value ^ sign) - sign
        else:
            if self.dtype not in _SIGNED_DTYPES:
                src = src & ((1 << width) - 1)
            value = torch.bitwise_right_shift(src, shift)

        expected = torch.zeros_like(tensors["out"])
        expected[:rows, :cols] = value.to(self.dtype.torch_dtype)
        tensors["out"][:] = expected


_TILE_CASES = [
    pytest.param(op_name, dtype, valid_name, valid_shape, id=f"t{op_name}-{dtype.value}-{valid_name}")
    for op_name in ("shl", "shr")
    for dtype in _ALL_DTYPES
    for valid_name, valid_shape in _VALID_SHAPES.items()
]

_A5_SCALAR_CASES = [
    pytest.param(
        op_name,
        dtype,
        valid_name,
        valid_shape,
        scalar,
        id=f"t{op_name}-{dtype.value}-{valid_name}-s{scalar}",
    )
    for op_name in ("shls", "shrs")
    for dtype in _ALL_DTYPES
    for valid_name, valid_shape in _VALID_SHAPES.items()
    for scalar in (0, _WIDTH[dtype] - 1)
]

# Pinned pto-isa a2/a3 TShiftCheck compares dst valid rows with src valid
# columns. Until that upstream typo is fixed, only square valid regions can be
# exercised by scalar shifts. The full and square-subview cases retain A2/A3
# coverage without claiming the blocked row-only/col-only shapes pass.
_A2A3_SAFE_VALID_SHAPES = {
    "full": (M, N),
    "square_combined": (17, 17),
}

_A2A3_SCALAR_CASES = [
    pytest.param(
        op_name,
        dtype,
        valid_name,
        valid_shape,
        scalar,
        id=f"t{op_name}-{dtype.value}-{valid_name}-s{scalar}",
    )
    for op_name in ("shls", "shrs")
    for dtype in _A2A3_SCALAR_DTYPES
    for valid_name, valid_shape in _A2A3_SAFE_VALID_SHAPES.items()
    for scalar in (0, _WIDTH[dtype] - 1)
]


class TestShiftTileFamily:
    """A2/A3 and A5 coverage for tile-tile shifts."""

    @pytest.mark.platforms("a2a3", "a5")
    @pytest.mark.parametrize("op_name,dtype,valid_name,valid_shape", _TILE_CASES)
    def test_shift(self, test_runner, op_name, dtype, valid_name, valid_shape):
        result = test_runner.run(
            ShiftCase(
                op_name=op_name,
                dtype=dtype,
                valid_name=valid_name,
                valid_shape=valid_shape,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


class TestShiftScalarA5Family:
    """A5 coverage for every scalar-shift width and valid-shape mode."""

    @pytest.mark.platforms("a5")
    @pytest.mark.parametrize("op_name,dtype,valid_name,valid_shape,scalar", _A5_SCALAR_CASES)
    def test_shift(self, test_runner, op_name, dtype, valid_name, valid_shape, scalar):
        result = test_runner.run(
            ShiftCase(
                op_name=op_name,
                dtype=dtype,
                valid_name=valid_name,
                valid_shape=valid_shape,
                scalar=scalar,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


class TestShiftScalarA2A3Family:
    """A2/A3 scalar coverage limited to shapes accepted by pinned pto-isa."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("op_name,dtype,valid_name,valid_shape,scalar", _A2A3_SCALAR_CASES)
    def test_shift(self, test_runner, op_name, dtype, valid_name, valid_shape, scalar):
        result = test_runner.run(
            ShiftCase(
                op_name=op_name,
                dtype=dtype,
                valid_name=valid_name,
                valid_shape=valid_shape,
                scalar=scalar,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
