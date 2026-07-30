# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime coverage for executable TAXPY, TPOW, and TPOWS contracts.

``taddrelu`` remains codegen-only because the pinned PTOAS does not define the
op and the newer A2/A3 legalization path still rejects it. That blocker is
recorded in the PTOAS status documentation rather than hidden by a skipped ST.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 16
N = 16

FULL = (M, N)
ROW_TAIL = (11, N)
COL_TAIL = (M, 13)
COMBINED_TAIL = (11, 13)

_PL_DT = {
    DataType.INT8: pl.INT8,
    DataType.UINT8: pl.UINT8,
    DataType.INT16: pl.INT16,
    DataType.UINT16: pl.UINT16,
    DataType.INT32: pl.INT32,
    DataType.UINT32: pl.UINT32,
    DataType.FP16: pl.FP16,
    DataType.FP32: pl.FP32,
    DataType.BF16: pl.BF16,
}
_TORCH_DT = {
    DataType.INT8: torch.int8,
    DataType.UINT8: torch.uint8,
    DataType.INT16: torch.int16,
    DataType.UINT16: torch.uint16,
    DataType.INT32: torch.int32,
    DataType.UINT32: torch.uint32,
    DataType.FP16: torch.float16,
    DataType.FP32: torch.float32,
    DataType.BF16: torch.bfloat16,
}
_FLOAT_DTYPES = {DataType.FP16, DataType.FP32, DataType.BF16}
_INTEGER_DTYPES = [
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
]


def _base(dtype: DataType) -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(4) + 1
    return values.to(_TORCH_DT[dtype]).contiguous()


def _exponent(dtype: DataType) -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.int64).reshape(M, N).remainder(4)
    return values.to(_TORCH_DT[dtype]).contiguous()


def _accumulator(dtype: DataType) -> torch.Tensor:
    values = torch.arange(M * N, dtype=torch.float32).reshape(M, N).remainder(13) / 3 - 2
    return values.to(_TORCH_DT[dtype]).contiguous()


def _make_program(
    op_name: str,
    src_dtype: DataType,
    dst_dtype: DataType,
    valid_shape: tuple[int, int],
    scalar: int | float,
    high_precision: bool,
    scalar_encoding: str,
):
    pl_src_dtype = _PL_DT[src_dtype]
    pl_dst_dtype = _PL_DT[dst_dtype]
    valid = list(valid_shape)
    valid_rows, valid_cols = valid_shape

    if op_name == "axpy":

        @pl.program
        class AxpyProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_dst_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dst_dtype]],
            ) -> pl.Tensor[[M, N], pl_dst_dtype]:
                src = pl.load(src0, [0, 0], [M, N], valid_shapes=valid)
                dst = pl.load(src1, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.axpy(src, scalar, dst), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_dst_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_dst_dtype]],
            ) -> pl.Tensor[[M, N], pl_dst_dtype]:
                return self.kernel(src0, src1, out)

        return AxpyProgram

    if op_name == "pow" and src_dtype in _FLOAT_DTYPES:

        @pl.program
        class FloatPowProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                base = pl.load(src0, [0, 0], [M, N], valid_shapes=valid)
                exp = pl.load(src1, [0, 0], [M, N], valid_shapes=valid)
                tmp_raw = pl.tile.create(
                    [M, N],
                    dtype=pl_src_dtype,
                    target_memory=pl.MemorySpace.Vec,
                )
                tmp = pl.tile.set_validshape(tmp_raw, valid_rows, valid_cols)
                result = pl.tile.pow(
                    base,
                    exp,
                    tmp,
                    high_precision=high_precision,
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                return self.kernel(src0, src1, out)

        return FloatPowProgram

    if op_name == "pow":

        @pl.program
        class IntegerPowProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                base = pl.load(src0, [0, 0], [M, N], valid_shapes=valid)
                exp = pl.load(src1, [0, 0], [M, N], valid_shapes=valid)
                return pl.store(pl.tile.pow(base, exp), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                return self.kernel(src0, src1, out)

        return IntegerPowProgram

    if src_dtype in _FLOAT_DTYPES and scalar_encoding == "ssa":

        @pl.program
        class FloatPowsSSAProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                base = pl.load(src0, [0, 0], [M, N], valid_shapes=valid)
                exp = pl.read(src1, [0, 0])
                tmp_raw = pl.tile.create(
                    [M, N],
                    dtype=pl_src_dtype,
                    target_memory=pl.MemorySpace.Vec,
                )
                tmp = pl.tile.set_validshape(tmp_raw, valid_rows, valid_cols)
                result = pl.tile.pows(
                    base,
                    exp,
                    tmp,
                    high_precision=high_precision,
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                return self.kernel(src0, src1, out)

        return FloatPowsSSAProgram

    if src_dtype in _FLOAT_DTYPES:

        @pl.program
        class FloatPowsImmediateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                base = pl.load(src0, [0, 0], [M, N], valid_shapes=valid)
                tmp_raw = pl.tile.create(
                    [M, N],
                    dtype=pl_src_dtype,
                    target_memory=pl.MemorySpace.Vec,
                )
                tmp = pl.tile.set_validshape(tmp_raw, valid_rows, valid_cols)
                result = pl.tile.pows(
                    base,
                    scalar,
                    tmp,
                    high_precision=high_precision,
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                return self.kernel(src0, src1, out)

        return FloatPowsImmediateProgram

    if scalar_encoding == "ssa":

        @pl.program
        class IntegerPowsSSAProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                base = pl.load(src0, [0, 0], [M, N], valid_shapes=valid)
                exp = pl.read(src1, [0, 0])
                return pl.store(pl.tile.pows(base, exp), [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src0: pl.Tensor[[M, N], pl_src_dtype],
                src1: pl.Tensor[[M, N], pl_src_dtype],
                out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
            ) -> pl.Tensor[[M, N], pl_src_dtype]:
                return self.kernel(src0, src1, out)

        return IntegerPowsSSAProgram

    @pl.program
    class IntegerPowsImmediateProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[M, N], pl_src_dtype],
            src1: pl.Tensor[[M, N], pl_src_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
        ) -> pl.Tensor[[M, N], pl_src_dtype]:
            base = pl.load(src0, [0, 0], [M, N], valid_shapes=valid)
            return pl.store(pl.tile.pows(base, scalar), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[M, N], pl_src_dtype],
            src1: pl.Tensor[[M, N], pl_src_dtype],
            out: pl.InOut[pl.Tensor[[M, N], pl_src_dtype]],
        ) -> pl.Tensor[[M, N], pl_src_dtype]:
            return self.kernel(src0, src1, out)

    return IntegerPowsImmediateProgram


class MathFusedCase(PTOTestCase):
    __test__ = False

    def __init__(
        self,
        op_name: str,
        src_dtype: DataType,
        dst_dtype: DataType,
        valid_shape: tuple[int, int],
        scalar: int | float,
        high_precision: bool,
        scalar_encoding: str,
        *,
        platform: str,
    ):
        super().__init__(platform=platform)
        self.op_name = op_name
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
        self.valid_shape = valid_shape
        self.scalar = scalar
        self.high_precision = high_precision
        self.scalar_encoding = scalar_encoding

    def get_name(self) -> str:
        valid_tag = f"v{self.valid_shape[0]}x{self.valid_shape[1]}"
        precision_tag = "high" if self.high_precision else "default"
        return (
            f"math_fused_{self.op_name}_{self.src_dtype.value}_to_{self.dst_dtype.value}"
            f"_{valid_tag}_{precision_tag}_{self.scalar_encoding}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        src1_dtype = self.dst_dtype if self.op_name == "axpy" else self.src_dtype
        src1_init = _accumulator if self.op_name == "axpy" else _exponent
        return [
            TensorSpec("src0", [M, N], self.src_dtype, init_value=lambda: _base(self.src_dtype)),
            TensorSpec("src1", [M, N], src1_dtype, init_value=lambda: src1_init(src1_dtype)),
            TensorSpec(
                "out",
                [M, N],
                self.dst_dtype,
                init_value=torch.zeros,
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        return _make_program(
            self.op_name,
            self.src_dtype,
            self.dst_dtype,
            self.valid_shape,
            self.scalar,
            self.high_precision,
            self.scalar_encoding,
        )

    def compute_expected(self, tensors, params=None):
        valid_rows, valid_cols = self.valid_shape
        src0 = tensors["src0"][:valid_rows, :valid_cols]
        src1 = tensors["src1"][:valid_rows, :valid_cols]
        if self.op_name == "axpy":
            expected = src1 + src0 * self.scalar
        elif self.op_name == "pow":
            if self.src_dtype in _FLOAT_DTYPES:
                expected = torch.pow(src0, src1)
            else:
                expected = torch.pow(src0.to(torch.int64), src1.to(torch.int64))
        else:
            exp = tensors["src1"][0, 0].item() if self.scalar_encoding == "ssa" else self.scalar
            if self.src_dtype in _FLOAT_DTYPES:
                expected = torch.pow(src0, exp)
            else:
                expected = torch.pow(src0.to(torch.int64), int(exp))
        tensors["out"].zero_()
        tensors["out"][:valid_rows, :valid_cols] = expected.to(_TORCH_DT[self.dst_dtype])


def _case(
    platform: str,
    op_name: str,
    src_dtype: DataType,
    dst_dtype: DataType,
    valid_shape: tuple[int, int],
    scalar: int | float,
    high_precision: bool = False,
    scalar_encoding: str = "immediate",
):
    valid_tag = f"{valid_shape[0]}x{valid_shape[1]}"
    precision_tag = "high" if high_precision else "default"
    return pytest.param(
        platform,
        op_name,
        src_dtype,
        dst_dtype,
        valid_shape,
        scalar,
        high_precision,
        scalar_encoding,
        id=(
            f"{platform}-{op_name}-{src_dtype.value}-to-{dst_dtype.value}"
            f"-{valid_tag}-{precision_tag}-{scalar_encoding}"
        ),
    )


_CASES = []
for _platform in ("a2a3", "a5"):
    for _src_dtype, _dst_dtype, _scalar in (
        (DataType.FP16, DataType.FP16, -1.5),
        (DataType.FP16, DataType.FP32, 0.0),
        (DataType.FP32, DataType.FP32, 2.0),
    ):
        _CASES.append(
            _case(
                _platform,
                "axpy",
                _src_dtype,
                _dst_dtype,
                COMBINED_TAIL,
                _scalar,
            )
        )
    for _valid_shape in (FULL, ROW_TAIL, COL_TAIL):
        _CASES.append(
            _case(
                _platform,
                "axpy",
                DataType.FP32,
                DataType.FP32,
                _valid_shape,
                2.0,
            )
        )

_A2A3_POW_VARIANTS = [*[(dtype, False) for dtype in _INTEGER_DTYPES], (DataType.FP32, False)]
_A5_POW_VARIANTS = [
    *[(dtype, False) for dtype in _INTEGER_DTYPES],
    (DataType.FP16, False),
    (DataType.FP32, False),
    (DataType.FP16, True),
    (DataType.FP32, True),
    (DataType.BF16, True),
]
for _platform, _variants in (
    ("a2a3", _A2A3_POW_VARIANTS),
    ("a5", _A5_POW_VARIANTS),
):
    for _op_name in ("pow", "pows"):
        for _dtype, _high_precision in _variants:
            _scalar = 3 if _dtype not in _FLOAT_DTYPES else 2.0
            _CASES.append(
                _case(
                    _platform,
                    _op_name,
                    _dtype,
                    _dtype,
                    COMBINED_TAIL,
                    _scalar,
                    _high_precision,
                )
            )
        for _valid_shape in (FULL, ROW_TAIL, COL_TAIL):
            _CASES.append(
                _case(
                    _platform,
                    _op_name,
                    DataType.FP32,
                    DataType.FP32,
                    _valid_shape,
                    2.0,
                )
            )
    _CASES.append(
        _case(
            _platform,
            "pows",
            DataType.FP32,
            DataType.FP32,
            COMBINED_TAIL,
            2.0,
            scalar_encoding="ssa",
        )
    )


@pytest.mark.platforms("a2a3", "a5")
@pytest.mark.parametrize(
    (
        "platform,op_name,src_dtype,dst_dtype,valid_shape,scalar,"
        "high_precision,scalar_encoding"
    ),
    _CASES,
)
def test_math_fused(
    test_runner,
    platform,
    op_name,
    src_dtype,
    dst_dtype,
    valid_shape,
    scalar,
    high_precision,
    scalar_encoding,
):
    result = test_runner.run(
        MathFusedCase(
            op_name,
            src_dtype,
            dst_dtype,
            valid_shape,
            scalar,
            high_precision,
            scalar_encoding,
            platform=platform,
        )
    )
    assert result.passed, f"Test failed: {result.error}"
