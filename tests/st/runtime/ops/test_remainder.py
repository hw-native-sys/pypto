# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact-op system tests for the PTO remainder family.

The four direct tile APIs must emit ``pto.trem``, ``pto.trems``,
``pto.tfmod``, and ``pto.tfmods`` respectively. A2/A3 coverage follows the
current pinned PTO-ISA contract:

* TREM/TREMS: FP32 and INT32;
* TFMOD/TFMODS: FP32;
* TREM uses a same-dtype scratch tile with two valid rows;
* TREMS uses a same-dtype scratch tile with one valid row.
* TREM/TFMOD cover default and ``high_precision`` PTOAS attributes; A2/A3
  accepts but ignores the high-precision selection, as specified by PTO-ISA.
* For A2/A3 INT32 TREM/TREMS, every source element and scalar must stay in
  the inclusive PTO-ISA domain ``[-2**24, 2**24]``.

Inputs combine positive, negative, and zero dividends with positive and
negative, strictly non-zero divisors. Full, row-tail, column-tail, and combined
row-and-column-tail valid regions are covered. Invalid output cells remain zero
through a zero-initialized InOut tensor, so the tests never rely on
uninitialized tile contents.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 64
INT32_REMAINDER_LIMIT = 2**24
ROW_TAIL = (11, N)
COL_TAIL = (M, 47)
COMBINED_TAIL = (11, 47)

_PL_DT = {
    DataType.FP32: pl.FP32,
    DataType.INT32: pl.INT32,
}


def _repeat_int32_pattern(values: list[int]) -> torch.Tensor:
    pattern = torch.tensor(values, dtype=torch.int64)
    repeats = (M * N + pattern.numel() - 1) // pattern.numel()
    return pattern.repeat(repeats)[: M * N].reshape(M, N)


def _dividend(dtype: DataType, boundary_values: bool = False) -> torch.Tensor:
    if boundary_values:
        assert dtype == DataType.INT32
        return _repeat_int32_pattern(
            [
                INT32_REMAINDER_LIMIT,
                INT32_REMAINDER_LIMIT - 1,
                -INT32_REMAINDER_LIMIT,
                -(INT32_REMAINDER_LIMIT - 1),
                0,
                1,
                -1,
            ]
        ).to(torch.int32)
    values = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(23) - 11
    if dtype == DataType.FP32:
        values = values.to(torch.float32) / 2.0
    return values.to(dtype.torch_dtype).contiguous()


def _divisor(dtype: DataType, boundary_values: bool = False) -> torch.Tensor:
    if boundary_values:
        assert dtype == DataType.INT32
        return _repeat_int32_pattern(
            [
                INT32_REMAINDER_LIMIT,
                INT32_REMAINDER_LIMIT - 1,
                -INT32_REMAINDER_LIMIT,
                -(INT32_REMAINDER_LIMIT - 1),
                1,
                -1,
                2,
                -2,
            ]
        ).to(torch.int32)
    index = torch.arange(M * N, dtype=torch.int64).reshape(M, N)
    magnitude = index.remainder(5) + 1
    sign = torch.where(index.remainder(3) == 0, -1, 1)
    result = (magnitude * sign).to(dtype.torch_dtype).contiguous()
    assert torch.count_nonzero(result) == result.numel()
    return result


def _make_program(
    op_name: str,
    dtype: DataType,
    valid_shape: tuple[int, int],
    scalar: int | float | None,
    high_precision: bool,
):
    pl_dtype = _PL_DT[dtype]
    valid = list(valid_shape)

    if op_name == "rem":

        @pl.program
        class RemProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shape=valid)
                rhs_tile = pl.load(rhs, [0, 0], [M, N], valid_shape=valid)
                tmp: pl.Tile[[2, N], pl_dtype] = pl.tile.create(
                    [2, N], dtype=pl_dtype, target_memory=pl.MemorySpace.Vec
                )
                result = pl.tile.rem(lhs_tile, rhs_tile, tmp, high_precision=high_precision)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, rhs, out)

        return RemProgram

    if op_name == "rems":
        assert scalar is not None

        @pl.program
        class RemsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shape=valid)
                tmp: pl.Tile[[1, N], pl_dtype] = pl.tile.create(
                    [1, N], dtype=pl_dtype, target_memory=pl.MemorySpace.Vec
                )
                result = pl.tile.rems(lhs_tile, scalar, tmp)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, out)

        return RemsProgram

    if op_name == "fmod":

        @pl.program
        class FmodProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shape=valid)
                rhs_tile = pl.load(rhs, [0, 0], [M, N], valid_shape=valid)
                result = pl.tile.fmod(lhs_tile, rhs_tile, high_precision=high_precision)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, rhs, out)

        return FmodProgram

    assert op_name == "fmods"
    assert scalar is not None

    @pl.program
    class FmodsProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shape=valid)
            result = pl.tile.fmods(lhs_tile, scalar)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            lhs: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            return self.kernel(lhs, out)

    return FmodsProgram


class RemainderCase(PTOTestCase):
    """One direct tile remainder instruction case."""

    __test__ = False

    def __init__(
        self,
        *,
        op_name: str,
        dtype: DataType,
        valid_shape: tuple[int, int],
        scalar: int | float | None = None,
        high_precision: bool = False,
        boundary_values: bool = False,
    ):
        super().__init__()
        self.op_name = op_name
        self.dtype = dtype
        self.valid_shape = valid_shape
        self.scalar = scalar
        self.high_precision = high_precision
        self.boundary_values = boundary_values

    def get_name(self) -> str:
        scalar_tag = f"_s{self.scalar}" if self.scalar is not None else ""
        precision_tag = "_high_precision" if self.high_precision else ""
        boundary_tag = "_int32_boundary" if self.boundary_values else ""
        return (
            f"tile_{self.op_name}_{self.dtype.value}_v{self.valid_shape[0]}x{self.valid_shape[1]}"
            f"{scalar_tag}{precision_tag}{boundary_tag}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        specs = [
            TensorSpec(
                "lhs",
                [M, N],
                self.dtype,
                init_value=lambda: _dividend(self.dtype, self.boundary_values),
            ),
        ]
        if self.op_name in {"rem", "fmod"}:
            specs.append(
                TensorSpec(
                    "rhs",
                    [M, N],
                    self.dtype,
                    init_value=lambda: _divisor(self.dtype, self.boundary_values),
                )
            )
        specs.append(TensorSpec("out", [M, N], self.dtype, init_value=torch.zeros, is_output=True))
        return specs

    def get_program(self) -> Any:
        return _make_program(
            self.op_name,
            self.dtype,
            self.valid_shape,
            self.scalar,
            self.high_precision,
        )

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        valid_rows, valid_cols = self.valid_shape
        lhs = tensors["lhs"][:valid_rows, :valid_cols]
        rhs: torch.Tensor | int | float
        if self.scalar is None:
            rhs = tensors["rhs"][:valid_rows, :valid_cols]
        else:
            rhs = self.scalar
        expected = torch.zeros_like(tensors["out"])
        if self.op_name in {"rem", "rems"}:
            expected[:valid_rows, :valid_cols] = torch.remainder(lhs, rhs)
        else:
            expected[:valid_rows, :valid_cols] = torch.fmod(lhs, rhs)
        tensors["out"][:] = expected


_CASES = [
    pytest.param("rem", DataType.FP32, (M, N), None, False, False, id="trem-f32-full"),
    pytest.param("rem", DataType.FP32, ROW_TAIL, None, False, False, id="trem-f32-row-tail"),
    pytest.param("rem", DataType.FP32, COL_TAIL, None, False, False, id="trem-f32-col-tail"),
    pytest.param("rem", DataType.FP32, COMBINED_TAIL, None, False, False, id="trem-f32-combined-tail"),
    pytest.param("rem", DataType.FP32, (M, N), None, True, False, id="trem-f32-high-precision-full"),
    pytest.param(
        "rem", DataType.FP32, COMBINED_TAIL, None, True, False, id="trem-f32-high-precision-combined-tail"
    ),
    pytest.param("rem", DataType.INT32, (M, N), None, False, False, id="trem-i32-full"),
    pytest.param("rem", DataType.INT32, ROW_TAIL, None, False, False, id="trem-i32-row-tail"),
    pytest.param("rem", DataType.INT32, COL_TAIL, None, False, False, id="trem-i32-col-tail"),
    pytest.param("rem", DataType.INT32, COMBINED_TAIL, None, False, False, id="trem-i32-combined-tail"),
    pytest.param("rem", DataType.INT32, (M, N), None, False, True, id="trem-i32-domain-boundaries"),
    pytest.param("rems", DataType.FP32, (M, N), 3.0, False, False, id="trems-f32-positive"),
    pytest.param("rems", DataType.FP32, ROW_TAIL, -3.0, False, False, id="trems-f32-negative-row-tail"),
    pytest.param("rems", DataType.FP32, COL_TAIL, 3.0, False, False, id="trems-f32-positive-col-tail"),
    pytest.param(
        "rems", DataType.FP32, COMBINED_TAIL, -3.0, False, False, id="trems-f32-negative-combined-tail"
    ),
    pytest.param("rems", DataType.INT32, (M, N), 3, False, False, id="trems-i32-positive"),
    pytest.param("rems", DataType.INT32, ROW_TAIL, -3, False, False, id="trems-i32-negative-row-tail"),
    pytest.param("rems", DataType.INT32, COL_TAIL, 3, False, False, id="trems-i32-positive-col-tail"),
    pytest.param(
        "rems", DataType.INT32, COMBINED_TAIL, -3, False, False, id="trems-i32-negative-combined-tail"
    ),
    pytest.param(
        "rems", DataType.INT32, (M, N), INT32_REMAINDER_LIMIT, False, True, id="trems-i32-positive-limit"
    ),
    pytest.param(
        "rems",
        DataType.INT32,
        COMBINED_TAIL,
        -INT32_REMAINDER_LIMIT,
        False,
        True,
        id="trems-i32-negative-limit",
    ),
    pytest.param("fmod", DataType.FP32, (M, N), None, False, False, id="tfmod-full"),
    pytest.param("fmod", DataType.FP32, ROW_TAIL, None, False, False, id="tfmod-row-tail"),
    pytest.param("fmod", DataType.FP32, COL_TAIL, None, False, False, id="tfmod-col-tail"),
    pytest.param("fmod", DataType.FP32, COMBINED_TAIL, None, False, False, id="tfmod-combined-tail"),
    pytest.param("fmod", DataType.FP32, (M, N), None, True, False, id="tfmod-high-precision-full"),
    pytest.param(
        "fmod", DataType.FP32, COMBINED_TAIL, None, True, False, id="tfmod-high-precision-combined-tail"
    ),
    pytest.param("fmods", DataType.FP32, (M, N), 3.0, False, False, id="tfmods-positive"),
    pytest.param("fmods", DataType.FP32, ROW_TAIL, -3.0, False, False, id="tfmods-negative-row-tail"),
    pytest.param("fmods", DataType.FP32, COL_TAIL, 3.0, False, False, id="tfmods-positive-col-tail"),
    pytest.param(
        "fmods", DataType.FP32, COMBINED_TAIL, -3.0, False, False, id="tfmods-negative-combined-tail"
    ),
]


class TestRemainderFamily:
    """A2/A3 exact-op coverage for TREM, TREMS, TFMOD, and TFMODS."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("op_name,dtype,valid_shape,scalar,high_precision,boundary_values", _CASES)
    def test_remainder(
        self, test_runner, op_name, dtype, valid_shape, scalar, high_precision, boundary_values
    ):
        result = test_runner.run(
            RemainderCase(
                op_name=op_name,
                dtype=dtype,
                valid_shape=valid_shape,
                scalar=scalar,
                high_precision=high_precision,
                boundary_values=boundary_values,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
