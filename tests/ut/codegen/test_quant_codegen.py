# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public tile quantization and the A5 TQUANT ABI."""

from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir
from pypto.runtime import RunConfig


@pytest.mark.parametrize(
    ("offset_shape", "offset_dtype", "message"),
    [
        ((16, 2), pl.FP32, r"unboxed \[rows, 1\]"),
        ((8, 1), pl.FP32, "same valid row count"),
        ((16, 1), pl.FP16, "rank-2 FP32 offset"),
    ],
)
def test_quant_rejects_invalid_offset(offset_shape, offset_dtype, message):
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([16, 64], pl.FP32), span)
    scale = ir.Var("scale", ir.TileType([16, 1], pl.FP32), span)
    offset = ir.Var("offset", ir.TileType(list(offset_shape), offset_dtype), span)

    with pytest.raises(ValueError, match=message):
        ir.op.tile.quant(src, scale, offset)


def test_quant_preserves_valid_shape():
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([16, 64], pl.FP32, tile_view=ir.TileView(valid_shape=[10, 47])), span)
    scale = ir.Var("scale", ir.TileType([16, 1], pl.FP32, tile_view=ir.TileView(valid_shape=[10, 1])), span)
    result_type = ir.op.tile.quant(src, scale).type

    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == pl.INT8
    assert [dim.value for dim in result_type.shape] == [16, 64]
    assert result_type.tile_view is not None
    assert [dim.value for dim in result_type.tile_view.valid_shape] == [10, 47]


@pytest.mark.parametrize(
    ("physical_cols", "valid_cols", "safe"),
    [
        (32, 32, False),
        (64, 47, True),
        (96, 64, True),
        (96, 65, False),
        (96, None, False),
        (128, None, True),
    ],
)
def test_quant_requires_source_pitch_for_full_vector_reads(physical_cols, valid_cols, safe):
    span = ir.Span.unknown()
    rows = ir.ConstInt(16, pl.INDEX, span)
    cols = (
        ir.Var("valid_cols", ir.ScalarType(pl.INDEX), span)
        if valid_cols is None
        else ir.ConstInt(valid_cols, pl.INDEX, span)
    )
    source_view = ir.TileView(valid_shape=[rows, cols])
    src = ir.Var("src", ir.TileType([16, physical_cols], pl.FP32, tile_view=source_view), span)
    scale = ir.Var("scale", ir.TileType([16, 1], pl.FP32), span)

    if not safe:
        with pytest.raises(ValueError, match="cover the valid columns rounded up to 64"):
            ir.op.tile.quant(src, scale)
        return

    result_type = ir.op.tile.quant(src, scale).type
    assert isinstance(result_type, ir.TileType)
    assert result_type.shape[1].value == physical_cols


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    ("asymmetric", "dst_dtype", "mode"), [(False, pl.INT8, "INT8_SYM"), (True, pl.UINT8, "INT8_ASYM")]
)
def test_quant_assembles_a5(asymmetric, dst_dtype, mode, tmp_path):
    @pl.jit.incore
    def quant_kernel(
        src: pl.Tensor[[16, 64], pl.FP32],
        scale: pl.Tensor[[16, 1], pl.FP32],
        offset: pl.Tensor[[16, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 64], dst_dtype]],
    ) -> pl.Tensor[[16, 64], dst_dtype]:
        values = pl.tile.load(src, [0, 0], [16, 64])
        scales = pl.tile.load(scale, [0, 0], [16, 1])
        if asymmetric:
            offsets = pl.tile.load(offset, [0, 0], [16, 1])
            quantized = pl.tile.quant(values, scales, offsets)
        else:
            quantized = pl.tile.quant(values, scales)
        return pl.tile.store(quantized, [0, 0], out)

    @pl.jit
    def quant_program(
        src: pl.Tensor[[16, 64], pl.FP32],
        scale: pl.Tensor[[16, 1], pl.FP32],
        offset: pl.Tensor[[16, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 64], dst_dtype]],
    ) -> pl.Tensor[[16, 64], dst_dtype]:
        return quant_kernel(src, scale, offset, out)

    compiled = quant_program.compile(config=RunConfig(platform="a5", save_kernels_dir=str(tmp_path)))
    mlir = "\n".join(path.read_text() for path in Path(compiled.output_dir).rglob("*.pto"))
    assert "pto.tquant ins(" in mlir
    assert f"quant_type = #pto<quant_type {mode}>" in mlir
