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
    ("src_dtype", "src_cols", "valid_cols", "out_cols"),
    [(pl.INT8, 32, 23, 32), (pl.INT16, 96, 95, 128)],
)
def test_dequant_result_preserves_valid_shape(src_dtype, src_cols, valid_cols, out_cols):
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([16, src_cols], src_dtype, tile_view=ir.TileView([10, valid_cols])), span)
    params = ir.TileType([16, 1], pl.FP32, tile_view=ir.TileView([10, 1], blayout=ir.TileLayout.col_major))
    result_type = ir.op.tile.dequant(src, ir.Var("scale", params, span), ir.Var("offset", params, span)).type

    assert isinstance(result_type, ir.TileType)
    assert result_type.dtype == pl.FP32
    assert [dim.value for dim in result_type.shape] == [16, out_cols]
    assert result_type.tile_view is not None
    assert [dim.value for dim in result_type.tile_view.valid_shape] == [10, valid_cols]


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
    src = ir.Var("src", ir.TileType([16, 32], src_dtype), span)
    scale = ir.Var("scale", ir.TileType([16, 1], pl.FP32), span)
    offset = ir.Var("offset", ir.TileType(offset_shape, offset_dtype), span)

    with pytest.raises(ValueError, match=message):
        ir.op.tile.dequant(src, scale, offset)


def test_dequant_rejects_mismatched_parameter_descriptors():
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([16, 32], pl.INT8), span)
    scale = ir.Var("scale", ir.TileType([16, 1], pl.FP32), span)
    offset_type = ir.TileType([16, 8], pl.FP32, tile_view=ir.TileView(valid_shape=[16, 1]))
    offset = ir.Var("offset", offset_type, span)

    with pytest.raises(ValueError, match="matching physical shapes and layouts"):
        ir.op.tile.dequant(src, scale, offset)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(("src_dtype", "width"), [(pl.INT8, 32), (pl.INT16, 96)])
def test_dequant_assembles_a5(src_dtype, width, tmp_path):
    @pl.jit.incore
    def dequant_kernel(
        src: pl.Tensor[[16, width], src_dtype],
        scale: pl.Tensor[[16, 1], pl.FP32],
        offset: pl.Tensor[[16, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[16, width], pl.FP32]],
    ) -> pl.Tensor[[16, width], pl.FP32]:
        values = pl.tile.load(src, [0, 0], [16, width])
        scales = pl.tile.load(scale, [0, 0], [16, 1])
        offsets = pl.tile.load(offset, [0, 0], [16, 1])
        return pl.tile.store(pl.tile.dequant(values, scales, offsets), [0, 0], out)

    @pl.jit
    def dequant_program(
        src: pl.Tensor[[16, width], src_dtype],
        scale: pl.Tensor[[16, 1], pl.FP32],
        offset: pl.Tensor[[16, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[16, width], pl.FP32]],
    ) -> pl.Tensor[[16, width], pl.FP32]:
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
    if width == 96:
        # A5 repeats the full first predicate when storing the 32-column tail.
        result_type = "dtype=f32, rows=16, cols=128"
        assert result_type in dequant.split(" outs(", 1)[1]
        allocation = next(
            line for line in mlir.splitlines() if "pto.alloc_tile" in line and result_type in line
        )
        assert "valid_col = %c96_index" in allocation
        assert "!pto.partition_tensor_view<16x96xf32>" in mlir
