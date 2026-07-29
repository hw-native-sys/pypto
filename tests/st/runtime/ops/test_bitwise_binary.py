# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact-op system tests for the tile bitwise binary and scalar families."""

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


def _pattern(dtype: DataType, phase: int) -> torch.Tensor:
    width = _WIDTH[dtype]
    mask = (1 << width) - 1
    alternating = int("55" * (width // 8), 16)
    inverse = alternating ^ mask
    values = [0, mask, alternating, inverse]
    index = (torch.arange(M * N, dtype=torch.int64).reshape(M, N) + phase).remainder(4)
    result = torch.zeros((M, N), dtype=torch.int64)
    for i, value in enumerate(values):
        result[index == i] = value
    return result.to(dtype.torch_dtype).contiguous()


def _scalar(dtype: DataType) -> int:
    return int("55" * (_WIDTH[dtype] // 8), 16)


def _make_program(
    op_name: str,
    dtype: DataType,
    valid_shape: tuple[int, int],
    scalar: int | None,
    scalar_encoding: str,
):
    pl_dtype = _PL_DT[dtype]
    valid = list(valid_shape)
    valid_rows, valid_cols = valid_shape

    if op_name == "and":

        @pl.program
        class AndProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
                rhs_tile = pl.load(rhs, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.and_(lhs_tile, rhs_tile), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, rhs, out)

        return AndProgram

    if op_name == "or":

        @pl.program
        class OrProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
                rhs_tile = pl.load(rhs, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.or_(lhs_tile, rhs_tile), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, rhs, out)

        return OrProgram

    if op_name == "xor":

        @pl.program
        class XorProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
                rhs_tile = pl.load(rhs, [0, 0], [M, N], valid_shapes=valid)
                tmp: pl.Tile[
                    [M, N],
                    pl_dtype,
                    pl.MemorySpace.Vec,
                    pl.TileView(valid_shape=[valid_rows, valid_cols]),
                ] = pl.tile.create([M, N], dtype=pl_dtype, target_memory=pl.MemorySpace.Vec)
                return pl.store(pl.tile.xor(lhs_tile, rhs_tile, tmp), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                rhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, rhs, out)

        return XorProgram

    assert scalar is not None

    if op_name == "ands" and scalar_encoding == "immediate":

        @pl.program
        class AndsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.ands(lhs_tile, scalar), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, out)

        return AndsProgram

    if op_name == "ands":

        @pl.program
        class AndsSSAProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                scalar_value: pl.Scalar[pl_dtype] = pl.read(lhs, [0, 2])
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.ands(lhs_tile, scalar_value), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, out)

        return AndsSSAProgram

    if op_name == "ors" and scalar_encoding == "immediate":

        @pl.program
        class OrsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.ors(lhs_tile, scalar), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, out)

        return OrsProgram

    if op_name == "ors":

        @pl.program
        class OrsSSAProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                scalar_value: pl.Scalar[pl_dtype] = pl.read(lhs, [0, 2])
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.ors(lhs_tile, scalar_value), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, out)

        return OrsSSAProgram

    assert op_name == "xors"

    if scalar_encoding == "immediate":

        @pl.program
        class XorsProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
                tmp: pl.Tile[
                    [M, N],
                    pl_dtype,
                    pl.MemorySpace.Vec,
                    pl.TileView(valid_shape=[valid_rows, valid_cols]),
                ] = pl.tile.create([M, N], dtype=pl_dtype, target_memory=pl.MemorySpace.Vec)
                return pl.store(pl.tile.xors(lhs_tile, scalar, tmp), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                lhs: pl.Tensor[[M, N], pl_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
            ) -> pl.Tensor[[M, N], pl_dtype]:
                return self.kernel(lhs, out)

        return XorsProgram

    @pl.program
    class XorsSSAProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            scalar_value: pl.Scalar[pl_dtype] = pl.read(lhs, [0, 2])
            lhs_tile = pl.load(lhs, [0, 0], [M, N], valid_shapes=valid)
            tmp: pl.Tile[
                [M, N],
                pl_dtype,
                pl.MemorySpace.Vec,
                pl.TileView(valid_shape=[valid_rows, valid_cols]),
            ] = pl.tile.create([M, N], dtype=pl_dtype, target_memory=pl.MemorySpace.Vec)
            return pl.store(pl.tile.xors(lhs_tile, scalar_value, tmp), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            lhs: pl.Tensor[[M, N], pl_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_dtype]],
        ) -> pl.Tensor[[M, N], pl_dtype]:
            return self.kernel(lhs, out)

    return XorsSSAProgram


class BitwiseCase(PTOTestCase):
    """One direct bitwise instruction case."""

    __test__ = False

    def __init__(
        self,
        *,
        op_name: str,
        dtype: DataType,
        valid_shape: tuple[int, int],
        scalar: int | None = None,
        scalar_encoding: str = "immediate",
        platform: str = "a2a3",
    ):
        super().__init__(platform=platform)
        self.op_name = op_name
        self.dtype = dtype
        self.valid_shape = valid_shape
        self.scalar = scalar
        self.scalar_encoding = scalar_encoding

    def get_name(self) -> str:
        scalar_tag = f"_s{self.scalar}" if self.scalar is not None else ""
        valid_tag = f"v{self.valid_shape[0]}x{self.valid_shape[1]}"
        return f"tile_{self.op_name}_{self.dtype.value}_{valid_tag}_{self.scalar_encoding}{scalar_tag}"

    def define_tensors(self) -> list[TensorSpec]:
        specs = [
            TensorSpec("lhs", [M, N], self.dtype, init_value=lambda: _pattern(self.dtype, 0)),
        ]
        if self.op_name in {"and", "or", "xor"}:
            specs.append(TensorSpec("rhs", [M, N], self.dtype, init_value=lambda: _pattern(self.dtype, 1)))
        specs.append(TensorSpec("out", [M, N], self.dtype, init_value=torch.zeros, is_output=True))
        return specs

    def get_program(self) -> Any:
        return _make_program(
            self.op_name,
            self.dtype,
            self.valid_shape,
            self.scalar,
            self.scalar_encoding,
        )

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        rows, cols = self.valid_shape
        lhs = tensors["lhs"][:rows, :cols]
        rhs: torch.Tensor | int
        if self.scalar is None:
            rhs = tensors["rhs"][:rows, :cols]
        elif self.scalar_encoding == "ssa":
            rhs = tensors["lhs"][0, 2].item()
        else:
            rhs = self.scalar
        if self.op_name in {"and", "ands"}:
            value = torch.bitwise_and(lhs, rhs)
        elif self.op_name in {"or", "ors"}:
            value = torch.bitwise_or(lhs, rhs)
        else:
            value = torch.bitwise_xor(lhs, rhs)
        expected = torch.zeros_like(tensors["out"])
        expected[:rows, :cols] = value
        tensors["out"][:] = expected


_A2A3_TILE_DTYPES = [
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
]
_A2A3_SCALAR_DTYPES = [DataType.INT8, DataType.INT16]

_A5_TILE_DTYPES = [
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
]
_A5_SCALAR_DTYPES = [DataType.INT8, DataType.INT16, DataType.INT32]


def _cases(
    tile_dtypes: list[DataType],
    scalar_dtypes: list[DataType],
) -> list[Any]:
    cases = [
        *[
            pytest.param(
                op_name,
                dtype,
                COMBINED_TAIL,
                None,
                "immediate",
                id=f"t{op_name}-{dtype.value}-combined-tail",
            )
            for op_name in ("and", "or", "xor")
            for dtype in tile_dtypes
        ],
        *[
            pytest.param(
                op_name,
                dtype,
                COMBINED_TAIL,
                _scalar(dtype),
                "immediate",
                id=f"t{op_name}-{dtype.value}-combined-tail-immediate",
            )
            for op_name in ("ands", "ors", "xors")
            for dtype in scalar_dtypes
        ],
    ]

    representative_dtype = scalar_dtypes[-1]
    cases.extend(
        pytest.param(
            op_name,
            representative_dtype,
            valid_shape,
            _scalar(representative_dtype) if op_name.endswith("s") else None,
            "immediate",
            id=f"t{op_name}-{representative_dtype.value}-{valid_tag}",
        )
        for op_name in ("and", "or", "xor", "ands", "ors", "xors")
        for valid_shape, valid_tag in (
            (FULL, "full"),
            (ROW_TAIL, "row-tail"),
            (COL_TAIL, "col-tail"),
        )
    )
    cases.extend(
        pytest.param(
            op_name,
            representative_dtype,
            COMBINED_TAIL,
            _scalar(representative_dtype),
            "ssa",
            id=f"t{op_name}-{representative_dtype.value}-combined-tail-ssa",
        )
        for op_name in ("ands", "ors", "xors")
    )
    return cases


_A2A3_CASES = _cases(_A2A3_TILE_DTYPES, _A2A3_SCALAR_DTYPES)
_A5_CASES = _cases(_A5_TILE_DTYPES, _A5_SCALAR_DTYPES)


class TestBitwiseBinaryFamily:
    """A2/A3 same-name coverage for six tile bitwise instructions."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("op_name,dtype,valid_shape,scalar,scalar_encoding", _A2A3_CASES)
    def test_bitwise_binary(self, test_runner, op_name, dtype, valid_shape, scalar, scalar_encoding):
        result = test_runner.run(
            BitwiseCase(
                op_name=op_name,
                dtype=dtype,
                valid_shape=valid_shape,
                scalar=scalar,
                scalar_encoding=scalar_encoding,
                platform="a2a3",
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5")
    @pytest.mark.parametrize("op_name,dtype,valid_shape,scalar,scalar_encoding", _A5_CASES)
    def test_bitwise_binary_a5(self, test_runner, op_name, dtype, valid_shape, scalar, scalar_encoding):
        result = test_runner.run(
            BitwiseCase(
                op_name=op_name,
                dtype=dtype,
                valid_shape=valid_shape,
                scalar=scalar,
                scalar_encoding=scalar_encoding,
                platform="a5",
            )
        )
        assert result.passed, f"Test failed: {result.error}"
