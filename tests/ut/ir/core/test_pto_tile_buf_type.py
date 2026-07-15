# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the internal PTO mutable tile-buffer handle type."""

import pytest
from pypto.pypto_core import DataType, ir


def make_type(
    *,
    memory_space: ir.MemorySpace = ir.MemorySpace.Vec,
    dtype: DataType = DataType.FP32,
    rows: int = 32,
    cols: int = 64,
    blayout: ir.TileLayout = ir.TileLayout.row_major,
    slayout: ir.TileLayout = ir.TileLayout.none_box,
    fractal: int = 512,
    pad: ir.PadValue = ir.PadValue.null,
    valid_rows: int | None = None,
    valid_cols: int | None = None,
) -> ir.PTOTileBufType:
    return ir.PTOTileBufType(
        memory_space,
        dtype,
        rows,
        cols,
        blayout,
        slayout,
        fractal,
        pad,
        valid_rows,
        valid_cols,
    )


def test_fields_model_complete_pto_tile_buf_type():
    tile_buf = make_type(
        memory_space=ir.MemorySpace.Acc,
        dtype=DataType.FP16,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=256,
        pad=ir.PadValue.zero,
        valid_rows=8,
        valid_cols=48,
    )

    assert tile_buf.memory_space == ir.MemorySpace.Acc
    assert tile_buf.dtype == DataType.FP16
    assert (tile_buf.rows, tile_buf.cols) == (32, 64)
    assert tile_buf.blayout == ir.TileLayout.col_major
    assert tile_buf.slayout == ir.TileLayout.row_major
    assert tile_buf.fractal == 256
    assert tile_buf.pad == ir.PadValue.zero
    assert tile_buf.valid_rows == 8
    assert tile_buf.valid_cols == 48


@pytest.mark.parametrize(
    "changed",
    [
        make_type(memory_space=ir.MemorySpace.Mat),
        make_type(dtype=DataType.FP16),
        make_type(rows=16),
        make_type(cols=32),
        make_type(blayout=ir.TileLayout.col_major),
        make_type(slayout=ir.TileLayout.row_major),
        make_type(fractal=256),
        make_type(pad=ir.PadValue.min),
        make_type(valid_rows=16),
        make_type(valid_cols=48),
    ],
)
def test_structural_identity_includes_every_target_field(changed: ir.PTOTileBufType):
    baseline = make_type()

    assert not ir.structural_equal(baseline, changed)
    assert baseline != changed


def test_structural_equal_and_hash_are_consistent():
    lhs = make_type()
    rhs = make_type()

    assert ir.structural_equal(lhs, rhs)
    assert ir.structural_hash(lhs) == ir.structural_hash(rhs)
    assert hash(lhs) == hash(rhs)
    assert lhs in {rhs}


def test_serialization_roundtrip_preserves_precise_type_and_fields():
    tile_buf = make_type(
        memory_space=ir.MemorySpace.Right,
        dtype=DataType.BF16,
        rows=16,
        cols=128,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=256,
        pad=ir.PadValue.max,
        valid_rows=8,
    )
    var = ir.Var("buf", tile_buf, ir.Span.unknown())

    restored = ir.deserialize(ir.serialize(var))

    assert isinstance(restored, ir.Var)
    assert isinstance(restored.type, ir.PTOTileBufType)
    ir.assert_structural_equal(tile_buf, restored.type)


def test_debug_print_is_distinct_from_logical_tile_type():
    text = str(make_type())

    assert text.startswith("PTOTileBufType<")
    assert "memory_space=Vec" in text
    assert "blayout=row_major" in text
    assert "valid_rows=?" in text
    assert not text.startswith("pl.Tile[")


@pytest.mark.parametrize(("rows", "cols", "fractal"), [(0, 32, 512), (32, 0, 512), (32, 32, 0)])
def test_rejects_non_positive_physical_extents(rows: int, cols: int, fractal: int):
    with pytest.raises(Exception, match="must be positive"):
        make_type(rows=rows, cols=cols, fractal=fractal)


@pytest.mark.parametrize(("valid_rows", "valid_cols"), [(0, None), (33, None), (None, 65)])
def test_rejects_invalid_static_valid_extents(valid_rows: int | None, valid_cols: int | None):
    with pytest.raises(Exception, match="must be in"):
        make_type(valid_rows=valid_rows, valid_cols=valid_cols)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
