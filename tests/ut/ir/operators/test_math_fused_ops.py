# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type contracts for TAXPY, TADDRELU, TPOW, and TPOWS."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.op import tile_ops as tile
from pypto.pypto_core import DataType


def _tile(name, dtype=DataType.FP32, valid_shape=(7, 13)):
    view = ir.TileView(
        valid_shape=list(valid_shape),
        blayout=ir.TileLayout.row_major,
        slayout=ir.TileLayout.none_box,
    )
    return ir.Var(name, ir.TileType([8, 16], dtype, tile_view=view), ir.Span.unknown())


def _const_values(shape):
    values = []
    for dim in shape:
        assert isinstance(dim, ir.ConstInt)
        values.append(dim.value)
    return values


def test_axpy_and_add_relu_preserve_destination_contract():
    src = _tile("src", DataType.FP16)
    dst = _tile("dst", DataType.FP32)

    axpy = tile.axpy(src, 2.0, dst)
    fused = tile.add_relu(dst, _tile("rhs"))

    assert isinstance(axpy.type, ir.TileType)
    assert isinstance(fused.type, ir.TileType)
    assert axpy.type.dtype == DataType.FP32
    assert _const_values(axpy.type.get_effective_tile_view().valid_shape) == [7, 13]
    assert fused.type.dtype == DataType.FP32


@pytest.mark.parametrize("high_precision", [False, True])
def test_float_pow_forms_require_and_accept_tmp(high_precision):
    base = _tile("base")
    exp = _tile("exp")
    tmp = _tile("tmp")

    power = tile.pow(base, exp, tmp, high_precision=high_precision)
    scalar_power = tile.pows(base, 2.0, tmp, high_precision=high_precision)

    assert isinstance(power.type, ir.TileType)
    assert isinstance(scalar_power.type, ir.TileType)
    assert power.type.dtype == DataType.FP32
    assert scalar_power.type.dtype == DataType.FP32


@pytest.mark.parametrize(
    "dtype",
    [
        DataType.INT8,
        DataType.UINT8,
        DataType.INT16,
        DataType.UINT16,
        DataType.INT32,
        DataType.UINT32,
    ],
)
def test_integer_pow_forms_omit_tmp(dtype):
    base = _tile("base", dtype)
    exp = _tile("exp", dtype)

    power = tile.pow(base, exp)
    scalar_power = tile.pows(base, 3)

    assert isinstance(power.type, ir.TileType)
    assert isinstance(scalar_power.type, ir.TileType)
    assert power.type.dtype == dtype
    assert scalar_power.type.dtype == dtype


def test_pow_rejects_wrong_tmp_contract():
    base = _tile("base")
    exp = _tile("exp")

    with pytest.raises(ValueError, match="requires tmp"):
        tile.pow(base, exp)
    with pytest.raises(ValueError, match="forbids tmp"):
        tile.pow(_tile("ibase", DataType.INT32), _tile("iexp", DataType.INT32), _tile("tmp", DataType.INT32))
    with pytest.raises(ValueError, match="high_precision"):
        tile.pows(_tile("ibase", DataType.INT32), 2, high_precision=True)


def test_add_relu_emits_exact_pto_op(tmp_path):
    @pl.program
    class AddReluProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src0: pl.Tensor[[16, 16], pl.FP32],
            src1: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            lhs = pl.load(src0, [0, 0], [16, 16])
            rhs = pl.load(src1, [0, 0], [16, 16])
            return pl.store(pl.tile.add_relu(lhs, rhs), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src0: pl.Tensor[[16, 16], pl.FP32],
            src1: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return self.kernel(src0, src1, out)

    ir.compile(AddReluProgram, output_dir=str(tmp_path), skip_ptoas=True, platform="a2a3")
    pto_files = list(tmp_path.rglob("*.pto"))
    assert pto_files
    assert any("pto.taddrelu" in path.read_text() for path in pto_files)
