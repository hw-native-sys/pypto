# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check the A5 integer dequantization type and native PTO lowering."""

from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@pytest.mark.parametrize(
    ("src_dtype", "src_cols", "valid_cols"),
    [(pl.INT8, 64, 23), (pl.INT8, 96, 64), (pl.INT16, 128, 95), (pl.INT16, 128, None)],
)
def test_dequant_result_preserves_tile_shape(src_dtype, src_cols, valid_cols):
    span = ir.Span.unknown()
    if valid_cols is None:
        valid_cols = ir.Var("valid_cols", ir.ScalarType(pl.INDEX), span)
    src = ir.Var("src", ir.TileType([16, src_cols], src_dtype, tile_view=ir.TileView([10, valid_cols])), span)
    params = ir.TileType([16, 1], pl.FP32, tile_view=ir.TileView([10, 1], blayout=ir.TileLayout.col_major))
    result_type = ir.op.tile.dequant(src, ir.Var("scale", params, span), ir.Var("offset", params, span)).type

    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == pl.FP32
    assert all(isinstance(dim, ir.ConstInt) for dim in result_type.shape)
    assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [16, src_cols]
    assert result_type.tile_view is not None
    rows, cols = result_type.tile_view.valid_shape
    assert isinstance(rows, ir.ConstInt) and rows.value == 10
    if isinstance(valid_cols, int):
        assert isinstance(cols, ir.ConstInt) and cols.value == valid_cols
    else:
        assert isinstance(cols, ir.Var) and cols.name_hint == valid_cols.name_hint


@pytest.mark.parametrize(
    ("src_dtype", "offset_shape", "offset_dtype", "message"),
    [
        (pl.UINT8, [16, 1], pl.FP32, "signed INT8 or INT16"),
        (pl.INT8, [16, 1], pl.FP16, "rank-2 FP32 offset"),
        (pl.INT8, [16, 2], pl.FP32, "one valid column"),
        (pl.INT8, [8, 1], pl.FP32, "same valid row count"),
    ],
)
def test_dequant_rejects_invalid_operands(src_dtype, offset_shape, offset_dtype, message):
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([16, 64], src_dtype), span)
    scale = ir.Var("scale", ir.TileType([16, 1], pl.FP32), span)
    offset = ir.Var("offset", ir.TileType(offset_shape, offset_dtype), span)

    with pytest.raises(ValueError, match=message):
        ir.op.tile.dequant(src, scale, offset)


def test_dequant_rejects_mismatched_parameter_descriptors():
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([16, 64], pl.INT8), span)
    scale = ir.Var("scale", ir.TileType([16, 1], pl.FP32), span)
    offset_type = ir.TileType([16, 8], pl.FP32, tile_view=ir.TileView(valid_shape=[16, 1]))
    offset = ir.Var("offset", offset_type, span)

    with pytest.raises(ValueError, match="matching physical shapes and layouts"):
        ir.op.tile.dequant(src, scale, offset)


@pytest.mark.parametrize(
    ("src_dtype", "src_cols", "valid_cols"),
    [(pl.INT8, 32, 32), (pl.INT16, 96, 96), (pl.INT16, 96, None)],
)
def test_dequant_rejects_incomplete_source_vectors(src_dtype, src_cols, valid_cols):
    span = ir.Span.unknown()
    if valid_cols is None:
        valid_cols = ir.Var("valid_cols", ir.ScalarType(pl.INDEX), span)
    src = ir.Var("src", ir.TileType([16, src_cols], src_dtype, tile_view=ir.TileView([16, valid_cols])), span)
    params = ir.TileType([16, 1], pl.FP32)

    with pytest.raises(ValueError, match="physical source columns to cover every 64-element vector"):
        ir.op.tile.dequant(src, ir.Var("scale", params, span), ir.Var("offset", params, span))


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    ("src_dtype", "pto_dtype", "width", "offset_rows", "coeff_cols", "valid_rows"),
    [
        (pl.INT8, "i8", 32, 16, 1, 16),
        (pl.INT16, "i16", 96, 16, 1, 16),
        (pl.INT8, "i8", 64, 32, 1, 16),
        (pl.INT8, "i8", 64, 32, 1, 10),
        (pl.INT8, "i8", 64, 16, 8, 16),
        (pl.INT8, "i8", 64, 16, 8, 9),
        (pl.INT8, "i8", 64, 1, 1, 16),
    ],
)
def test_dequant_assembles_a5(src_dtype, pto_dtype, width, offset_rows, coeff_cols, valid_rows, tmp_path):
    physical_width = (width + 63) // 64 * 64
    offset_cols = 16 if offset_rows == 1 else coeff_cols

    @pl.jit.incore
    def dequant_kernel(
        src: pl.Tensor[[valid_rows, width], src_dtype],
        scale: pl.Tensor[[16, coeff_cols], pl.FP32],
        offset: pl.Tensor[[offset_rows, offset_cols], pl.FP32],
        out: pl.Out[pl.Tensor[[valid_rows, width], pl.FP32]],
    ) -> pl.Tensor[[valid_rows, width], pl.FP32]:
        values = pl.tile.load(src, [0, 0], [16, physical_width], valid_shape=[valid_rows, width])
        if coeff_cols == 1:
            scales = pl.tile.load(scale, [0, 0], [16, 1], valid_shape=[valid_rows, 1])
            offsets = pl.tile.load(offset, [0, 0], [offset_rows, offset_cols])
            if offset_rows == 1:
                offsets = pl.tile.reshape(offsets, [16, 1])
            elif offset_rows != 16:
                if valid_rows == 16:
                    offsets = pl.tile.slice(offsets, [16, 1], [0, 0])
                else:
                    offsets = pl.tile.slice(offsets, [16, 1], [0, 0], valid_shape=[valid_rows, 1])
        elif valid_rows == 16:
            scales = pl.tile.load(scale, [0, 0], [16, coeff_cols], valid_shape=[16, 1])
            offsets = pl.tile.load(offset, [0, 0], [offset_rows, coeff_cols], valid_shape=[16, 1])
            scales = pl.tile.slice(scales, [16, 1], [0, 0])
            offsets = pl.tile.slice(offsets, [16, 1], [0, 0])
        else:
            scales = pl.tile.load(scale, [0, 0], [16, coeff_cols])
            offsets = pl.tile.load(offset, [0, 0], [offset_rows, coeff_cols])
            scales = pl.tile.slice(scales, [valid_rows, 1], [1, 1])
            offsets = pl.tile.slice(offsets, [valid_rows, 1], [1, 1])
        return pl.tile.store(pl.tile.dequant(values, scales, offsets), [0, 0], out)

    @pl.jit
    def dequant_program(
        src: pl.Tensor[[valid_rows, width], src_dtype],
        scale: pl.Tensor[[16, coeff_cols], pl.FP32],
        offset: pl.Tensor[[offset_rows, offset_cols], pl.FP32],
        out: pl.Out[pl.Tensor[[valid_rows, width], pl.FP32]],
    ) -> pl.Tensor[[valid_rows, width], pl.FP32]:
        return dequant_kernel(src, scale, offset, out)

    compiled = dequant_program.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            save_kernels_dir=str(tmp_path),
        )
    )
    mlir = "\n".join(path.read_text() for path in Path(compiled.output_dir).rglob("*.pto"))
    dequant = next(line for line in mlir.splitlines() if "pto.tdequant ins(" in line)
    inputs, output = dequant.split(" outs(", 1)
    src_type = f"dtype={pto_dtype}, rows=16, cols={physical_width}"
    result_type = f"dtype=f32, rows=16, cols={physical_width}"
    assert src_type in inputs
    assert result_type in output
    for tile_type in (src_type, result_type):
        allocation = next(
            line for line in mlir.splitlines() if "pto.alloc_tile" in line and tile_type in line
        )
        assert f"valid_col = %c{width}_index" in allocation
    assert f"!pto.partition_tensor_view<{valid_rows}x{width}x{pto_dtype}>" in mlir
    assert f"!pto.partition_tensor_view<{valid_rows}x{width}xf32>" in mlir
    if offset_rows != 16 or coeff_cols != 1:
        assert "pto.tgetval" in mlir
        assert "pto.tsetval" in mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
