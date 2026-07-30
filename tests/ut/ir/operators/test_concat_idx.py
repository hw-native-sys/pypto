# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type contract tests for indexed tile concatenation."""

import pytest
from pypto import ir
from pypto.ir.op import tile_ops as tile
from pypto.pypto_core import DataType


def _tile(name, shape, dtype, valid_shape):
    view = ir.TileView(
        valid_shape=list(valid_shape),
        blayout=ir.TileLayout.row_major,
        slayout=ir.TileLayout.none_box,
    )
    return ir.Var(name, ir.TileType(list(shape), dtype, tile_view=view), ir.Span.unknown())


@pytest.mark.parametrize(
    "data_dtype",
    [
        DataType.INT8,
        DataType.UINT8,
        DataType.INT16,
        DataType.UINT16,
        DataType.INT32,
        DataType.UINT32,
        DataType.FP16,
        DataType.BF16,
        DataType.FP32,
    ],
)
@pytest.mark.parametrize(
    "index_dtype",
    [
        DataType.INT8,
        DataType.UINT8,
        DataType.INT16,
        DataType.UINT16,
        DataType.INT32,
        DataType.UINT32,
    ],
)
def test_concat_idx_returns_destination_type(data_dtype, index_dtype):
    src0 = _tile("src0", (8, 64), data_dtype, (8, 64))
    src1 = _tile("src1", (8, 64), data_dtype, (8, 48))
    idx0 = _tile("idx0", (8, 8), index_dtype, (8, 1))
    idx1 = _tile("idx1", (8, 8), index_dtype, (8, 1))
    dst = _tile("dst", (8, 64), data_dtype, (8, 64))

    call = tile.concat_idx(src0, src1, idx0, idx1, dst)

    assert isinstance(call.type, ir.TileType)
    assert isinstance(dst.type, ir.TileType)
    assert call.type.dtype == data_dtype
    assert call.type.shape == dst.type.shape


def test_concat_idx_rejects_invalid_index_contract():
    src = _tile("src", (8, 64), DataType.FP32, (8, 64))
    dst = _tile("dst", (8, 64), DataType.FP32, (8, 64))
    bad_dtype = _tile("bad", (8, 8), DataType.FP32, (8, 1))
    bad_cols = _tile("wide", (8, 8), DataType.INT32, (8, 2))
    good = _tile("good", (8, 8), DataType.INT32, (8, 1))

    with pytest.raises(ValueError, match="integer index"):
        tile.concat_idx(src, src, bad_dtype, good, dst)
    with pytest.raises(ValueError, match="columns equal to 1"):
        tile.concat_idx(src, src, bad_cols, good, dst)
