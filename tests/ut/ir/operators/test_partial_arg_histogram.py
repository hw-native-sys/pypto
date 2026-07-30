# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type contracts for TPARTARGMAX/MIN and THISTOGRAM."""

import pytest
from pypto import ir
from pypto.ir.op import tile_ops as tile
from pypto.pypto_core import DataType


def _view(valid_shape, layout=ir.TileLayout.row_major):
    return ir.TileView(
        valid_shape=valid_shape,
        blayout=layout,
        slayout=ir.TileLayout.none_box,
    )


def _tile(name, dtype, valid_shape, layout=ir.TileLayout.row_major, shape=(8, 16)):
    return ir.Var(
        name,
        ir.TileType(list(shape), dtype, tile_view=_view(list(valid_shape), layout)),
        ir.Span.unknown(),
    )


def _const_values(shape):
    values = []
    for dim in shape:
        assert isinstance(dim, ir.ConstInt)
        values.append(dim.value)
    return values


@pytest.mark.parametrize("op", [tile.part_argmax, tile.part_argmin])
@pytest.mark.parametrize("value_dtype", [DataType.FP16, DataType.FP32])
@pytest.mark.parametrize("index_dtype", [DataType.INT32, DataType.UINT32])
def test_partial_arg_contract_returns_value_and_index_tiles(op, value_dtype, index_dtype):
    src0 = _tile("src0", value_dtype, (8, 16))
    src1 = _tile("src1", value_dtype, (7, 13))
    idx0 = _tile("idx0", index_dtype, (8, 16))
    idx1 = _tile("idx1", index_dtype, (7, 13))

    call = op(src0, src1, idx0, idx1)

    assert isinstance(call.type, ir.TupleType)
    value_type, index_type = call.type.types
    assert isinstance(value_type, ir.TileType)
    assert isinstance(index_type, ir.TileType)
    assert value_type.dtype == value_dtype
    assert index_type.dtype == index_dtype
    assert _const_values(value_type.get_effective_tile_view().valid_shape) == [8, 16]
    assert _const_values(index_type.get_effective_tile_view().valid_shape) == [8, 16]


def test_partial_arg_result_layout_follows_dominating_source_pair():
    src0 = _tile("src0", DataType.FP32, (7, 13), ir.TileLayout.col_major)
    src1 = _tile("src1", DataType.FP32, (8, 16))
    idx0 = _tile("idx0", DataType.INT32, (7, 13), ir.TileLayout.col_major)
    idx1 = _tile("idx1", DataType.INT32, (8, 16))

    result_type = tile.part_argmax(src0, src1, idx0, idx1).type
    assert isinstance(result_type, ir.TupleType)
    value_type, index_type = result_type.types
    assert isinstance(value_type, ir.TileType)
    assert isinstance(index_type, ir.TileType)

    assert value_type.get_effective_tile_view().blayout == ir.TileLayout.row_major
    assert index_type.get_effective_tile_view().blayout == ir.TileLayout.row_major


@pytest.mark.parametrize("op", [tile.part_argmax, tile.part_argmin])
def test_partial_arg_contract_rejects_mismatched_pairs_and_crossing_valid_shapes(op):
    src0 = _tile("src0", DataType.FP32, (8, 12))
    src1 = _tile("src1", DataType.FP32, (7, 16))
    idx0 = _tile("idx0", DataType.INT32, (8, 12))
    idx1 = _tile("idx1", DataType.INT32, (7, 16))

    with pytest.raises(ValueError, match="contain the other"):
        op(src0, src1, idx0, idx1)

    full = _tile("full", DataType.FP32, (8, 16))
    bad_idx = _tile("bad_idx", DataType.INT32, (7, 16))
    with pytest.raises(ValueError, match="src0 and src0_idx"):
        op(full, full, bad_idx, _tile("idx", DataType.INT32, (8, 16)))


@pytest.mark.parametrize("byte", [0, 1])
def test_histogram_contract_uint16(byte):
    src = _tile("src", DataType.UINT16, (7, 13))
    idx = _tile("idx", DataType.UINT8, (7, 1), ir.TileLayout.col_major, shape=(8, 1))

    call = tile.histogram(src, idx, byte=byte)

    assert isinstance(call.type, ir.TileType)
    assert call.type.dtype == DataType.UINT32
    assert _const_values(call.type.shape) == [8, 256]
    assert _const_values(call.type.get_effective_tile_view().valid_shape) == [7, 256]


@pytest.mark.parametrize("byte,rows", [(0, 3), (1, 2), (2, 1), (3, 1)])
def test_histogram_contract_uint32(byte, rows):
    src = _tile("src", DataType.UINT32, (7, 13))
    idx = _tile("idx", DataType.UINT8, (rows, 13), shape=(rows, 16))

    call = tile.histogram(src, idx, byte=byte)

    assert isinstance(call.type, ir.TileType)
    assert call.type.dtype == DataType.UINT32
    assert _const_values(call.type.shape) == [8, 256]


def test_histogram_contract_rejects_invalid_byte_and_index_layout():
    src = _tile("src", DataType.UINT16, (7, 13))
    row_major_idx = _tile("idx", DataType.UINT8, (7, 1), shape=(8, 1))

    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        tile.histogram(src, row_major_idx, byte=4)
    with pytest.raises(ValueError, match="col_major"):
        tile.histogram(src, row_major_idx, byte=1)
