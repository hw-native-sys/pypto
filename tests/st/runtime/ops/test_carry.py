# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact-op system tests for TADDC, TSUBC, TADDSC, and TSUBSC.

Every op runs on the full PTOAS dtype union and on full, row-tail, column-tail,
and combined-tail valid regions. Both result/src0-alias-eligible and forced
non-alias paths execute on hardware. The carry tile alternates between zero and
one. Signed integer inputs include both limits so addition overflow and
subtraction borrow/wrap behavior are checked; cells outside the valid region
remain zero.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 64

_PL_DT = {
    DataType.FP16: pl.FP16,
    DataType.FP32: pl.FP32,
    DataType.INT16: pl.INT16,
    DataType.INT32: pl.INT32,
}
_DTYPES = (DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32)
_VALID_SHAPES = (
    ((M, N), "full"),
    ((11, N), "row-tail"),
    ((M, 47), "column-tail"),
    ((11, 47), "combined-tail"),
)
_OPS = ("addc", "subc", "addsc", "subsc")
_REUSE_MODES = ("alias", "nonalias")


def _src0(dtype: DataType) -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(31) - 15
    if dtype in {DataType.INT16, DataType.INT32}:
        info = torch.iinfo(dtype.torch_dtype)
        values[0, 0] = info.max  # carry = 0
        values[0, 1] = info.max  # carry = 1
        values[0, 2] = info.min  # carry = 0
        values[0, 3] = info.min  # carry = 1
    else:
        values = values.to(torch.float32) / 4.0
    return values.to(dtype.torch_dtype).contiguous()


def _src1(dtype: DataType) -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(7) + 1
    if dtype in {DataType.FP16, DataType.FP32}:
        values = values.to(torch.float32) / 2.0
    return values.to(dtype.torch_dtype).contiguous()


def _carry(dtype: DataType) -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(2)
    return values.to(dtype.torch_dtype).contiguous()


def _make_program(
    op_name: str,
    dtype: DataType,
    valid_shape: tuple[int, int],
    scalar: int | float | None,
    reuse_mode: str,
):
    pl_dtype = _PL_DT[dtype]
    valid = list(valid_shape)

    if reuse_mode == "nonalias":
        return _make_nonalias_program(op_name, pl_dtype, valid, scalar)

    if op_name == "addc":

        @pl.program
        class TileAddcProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                src1: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src0_tile = pl.load(src0, [0, 0], [M, N], valid_shape=valid)
                src1_tile = pl.load(src1, [0, 0], [M, N], valid_shape=valid)
                carry_tile = pl.load(carry, [0, 0], [M, N], valid_shape=valid)
                result = pl.tile.addc(src0_tile, src1_tile, carry_tile)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                src1: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src0, src1, carry, out)

        return TileAddcProgram

    if op_name == "subc":

        @pl.program
        class TileSubcProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                src1: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src0_tile = pl.load(src0, [0, 0], [M, N], valid_shape=valid)
                src1_tile = pl.load(src1, [0, 0], [M, N], valid_shape=valid)
                carry_tile = pl.load(carry, [0, 0], [M, N], valid_shape=valid)
                result = pl.tile.subc(src0_tile, src1_tile, carry_tile)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                src1: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src0, src1, carry, out)

        return TileSubcProgram

    assert scalar is not None

    if op_name == "addsc":

        @pl.program
        class TileAddscProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                src0_tile = pl.load(src0, [0, 0], [M, N], valid_shape=valid)
                carry_tile = pl.load(carry, [0, 0], [M, N], valid_shape=valid)
                scalar_value: pl.Scalar[pl_dtype] = pl.cast(scalar, pl_dtype)
                result = pl.tile.addsc(src0_tile, scalar_value, carry_tile)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(src0, carry, out)

        return TileAddscProgram

    assert op_name == "subsc"

    @pl.program
    class TileSubscProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl_dtype],
            carry: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            src0_tile = pl.load(src0, [0, 0], [M, N], valid_shape=valid)
            carry_tile = pl.load(carry, [0, 0], [M, N], valid_shape=valid)
            scalar_value: pl.Scalar[pl_dtype] = pl.cast(scalar, pl_dtype)
            result = pl.tile.subsc(src0_tile, scalar_value, carry_tile)
            return pl.store(result, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl_dtype],
            carry: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            return self.kernel(src0, carry, out)

    return TileSubscProgram


def _make_nonalias_program(
    op_name: str,
    pl_dtype: Any,
    valid: list[int],
    scalar: int | float | None,
):
    """Keep src0 live after the carry op so its result cannot reuse src0 storage."""
    if op_name == "addc":

        @pl.program
        class TileAddcNonaliasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                src1: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
                src0_after: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> tuple[pl.Tensor[[M, N], pl_dtype], pl.Tensor[[M, N], pl_dtype]]:
                src0_tile = pl.load(src0, [0, 0], [M, N], valid_shape=valid)
                src1_tile = pl.load(src1, [0, 0], [M, N], valid_shape=valid)
                carry_tile = pl.load(carry, [0, 0], [M, N], valid_shape=valid)
                result = pl.tile.addc(src0_tile, src1_tile, carry_tile)
                out = pl.store(result, [0, 0], out)
                src0_after = pl.store(src0_tile, [0, 0], src0_after)
                return out, src0_after

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                src1: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
                src0_after: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> tuple[pl.Tensor[[M, N], pl_dtype], pl.Tensor[[M, N], pl_dtype]]:
                return self.kernel(src0, src1, carry, out, src0_after)

        return TileAddcNonaliasProgram

    if op_name == "subc":

        @pl.program
        class TileSubcNonaliasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                src1: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
                src0_after: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> tuple[pl.Tensor[[M, N], pl_dtype], pl.Tensor[[M, N], pl_dtype]]:
                src0_tile = pl.load(src0, [0, 0], [M, N], valid_shape=valid)
                src1_tile = pl.load(src1, [0, 0], [M, N], valid_shape=valid)
                carry_tile = pl.load(carry, [0, 0], [M, N], valid_shape=valid)
                result = pl.tile.subc(src0_tile, src1_tile, carry_tile)
                out = pl.store(result, [0, 0], out)
                src0_after = pl.store(src0_tile, [0, 0], src0_after)
                return out, src0_after

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                src1: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
                src0_after: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> tuple[pl.Tensor[[M, N], pl_dtype], pl.Tensor[[M, N], pl_dtype]]:
                return self.kernel(src0, src1, carry, out, src0_after)

        return TileSubcNonaliasProgram

    assert scalar is not None

    if op_name == "addsc":

        @pl.program
        class TileAddscNonaliasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
                src0_after: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> tuple[pl.Tensor[[M, N], pl_dtype], pl.Tensor[[M, N], pl_dtype]]:
                src0_tile = pl.load(src0, [0, 0], [M, N], valid_shape=valid)
                carry_tile = pl.load(carry, [0, 0], [M, N], valid_shape=valid)
                scalar_value: pl.Scalar[pl_dtype] = pl.cast(scalar, pl_dtype)
                result = pl.tile.addsc(src0_tile, scalar_value, carry_tile)
                out = pl.store(result, [0, 0], out)
                src0_after = pl.store(src0_tile, [0, 0], src0_after)
                return out, src0_after

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_dtype],
                carry: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
                src0_after: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> tuple[pl.Tensor[[M, N], pl_dtype], pl.Tensor[[M, N], pl_dtype]]:
                return self.kernel(src0, carry, out, src0_after)

        return TileAddscNonaliasProgram

    assert op_name == "subsc"

    @pl.program
    class TileSubscNonaliasProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl_dtype],
            carry: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            src0_after: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> tuple[pl.Tensor[[M, N], pl_dtype], pl.Tensor[[M, N], pl_dtype]]:
            src0_tile = pl.load(src0, [0, 0], [M, N], valid_shape=valid)
            carry_tile = pl.load(carry, [0, 0], [M, N], valid_shape=valid)
            scalar_value: pl.Scalar[pl_dtype] = pl.cast(scalar, pl_dtype)
            result = pl.tile.subsc(src0_tile, scalar_value, carry_tile)
            out = pl.store(result, [0, 0], out)
            src0_after = pl.store(src0_tile, [0, 0], src0_after)
            return out, src0_after

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl_dtype],
            carry: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            src0_after: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> tuple[pl.Tensor[[M, N], pl_dtype], pl.Tensor[[M, N], pl_dtype]]:
            return self.kernel(src0, carry, out, src0_after)

    return TileSubscNonaliasProgram


class CarryCase(PTOTestCase):
    """One direct tile carry-in instruction case."""

    __test__ = False

    def __init__(
        self,
        *,
        op_name: str,
        dtype: DataType,
        valid_shape: tuple[int, int],
        reuse_mode: str,
        scalar: int | float | None = None,
    ):
        super().__init__()
        self.op_name = op_name
        self.dtype = dtype
        self.valid_shape = valid_shape
        self.reuse_mode = reuse_mode
        self.scalar = scalar

    def get_name(self) -> str:
        scalar_tag = f"_s{self.scalar}" if self.scalar is not None else ""
        valid_tag = f"v{self.valid_shape[0]}x{self.valid_shape[1]}"
        return f"tile_{self.op_name}_{self.dtype.value}_{valid_tag}_{self.reuse_mode}{scalar_tag}"

    def define_tensors(self) -> list[TensorSpec]:
        specs = [
            TensorSpec("src0", [M, N], self.dtype, init_value=lambda: _src0(self.dtype)),
        ]
        if self.op_name in {"addc", "subc"}:
            specs.append(TensorSpec("src1", [M, N], self.dtype, init_value=lambda: _src1(self.dtype)))
        specs.extend(
            [
                TensorSpec("carry", [M, N], self.dtype, init_value=lambda: _carry(self.dtype)),
                TensorSpec("out", [M, N], self.dtype, init_value=torch.zeros, is_output=True),
            ]
        )
        if self.reuse_mode == "nonalias":
            specs.append(TensorSpec("src0_after", [M, N], self.dtype, init_value=torch.zeros, is_output=True))
        return specs

    def get_program(self) -> Any:
        return _make_program(self.op_name, self.dtype, self.valid_shape, self.scalar, self.reuse_mode)

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        rows, cols = self.valid_shape
        src0 = tensors["src0"][:rows, :cols]
        carry = tensors["carry"][:rows, :cols]
        expected = torch.zeros_like(tensors["out"])
        if self.op_name == "addc":
            value = src0 + tensors["src1"][:rows, :cols] + carry
        elif self.op_name == "subc":
            value = src0 - tensors["src1"][:rows, :cols] + carry
        elif self.op_name == "addsc":
            value = src0 + self.scalar + carry
        else:
            value = src0 - self.scalar + carry
        expected[:rows, :cols] = value
        tensors["out"][:] = expected
        if self.reuse_mode == "nonalias":
            src0_after = torch.zeros_like(tensors["src0_after"])
            src0_after[:rows, :cols] = src0
            tensors["src0_after"][:] = src0_after


_CASES = [
    pytest.param(
        op_name,
        dtype,
        valid_shape,
        (1.5 if dtype in {DataType.FP16, DataType.FP32} else 1) if op_name.endswith("sc") else None,
        reuse_mode,
        marks=(
            pytest.mark.platform_xfail(
                "a2a3",
                reason=(
                    "PTO ISA v0.57 TADDSC/TSUBSC full-column row-tail fast path computes incorrect results"
                ),
            )
            if op_name in {"addsc", "subsc"} and valid_tag == "row-tail"
            else ()
        ),
        id=f"t{op_name}-{dtype.value}-{valid_tag}-{reuse_mode}",
    )
    for op_name in _OPS
    for dtype in _DTYPES
    for valid_shape, valid_tag in _VALID_SHAPES
    for reuse_mode in _REUSE_MODES
]


class TestCarryFamily:
    """A2/A3 exact-op coverage for the four carry-in instructions."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("op_name,dtype,valid_shape,scalar,reuse_mode", _CASES)
    def test_carry(self, test_runner, op_name, dtype, valid_shape, scalar, reuse_mode):
        result = test_runner.run(
            CarryCase(
                op_name=op_name,
                dtype=dtype,
                valid_shape=valid_shape,
                reuse_mode=reuse_mode,
                scalar=scalar,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
