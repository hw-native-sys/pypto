# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for explicit tile buffer-set IR operations."""

import pytest
from pypto import DataType, ir
from pypto.ir.op import tile


def test_create_buffer_set_returns_first_class_storage_type():
    buffer_set = tile.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)

    assert buffer_set.op.name == "tile.create_buffer_set"
    assert isinstance(buffer_set.type, ir.TileBufferSetType)
    assert buffer_set.type.count == 2
    assert buffer_set.type.memory_space == ir.MemorySpace.Acc


def test_buffer_slot_accepts_dynamic_integer_index():
    buffer_set = tile.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)
    index = ir.Var("index", ir.ScalarType(DataType.INDEX), ir.Span.unknown())

    slot = tile.buffer_slot(buffer_set, index)

    assert slot.op.name == "tile.buffer_slot"
    assert isinstance(slot.type, ir.TileType)
    assert slot.type.memory_space == ir.MemorySpace.Acc
    assert slot.type.dtype == DataType.FP32


@pytest.mark.parametrize("index", [-1, 2])
def test_buffer_slot_rejects_constant_out_of_range_index(index):
    buffer_set = tile.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)

    with pytest.raises(ValueError, match="index.*out of range"):
        tile.buffer_slot(buffer_set, index)


def test_buffer_slot_rejects_non_integer_dynamic_index():
    buffer_set = tile.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)
    index = ir.Var("index", ir.ScalarType(DataType.FP32), ir.Span.unknown())

    with pytest.raises(ValueError, match="index.*integer"):
        tile.buffer_slot(buffer_set, index)


def test_buffer_slot_rejects_ordinary_tile():
    ordinary_tile = tile.create([16, 128], DataType.FP32, ir.MemorySpace.Acc)

    with pytest.raises(ValueError, match="first argument.*TileBufferSetType"):
        tile.buffer_slot(ordinary_tile, 0)


def test_release_accepts_selected_slot_and_returns_lifetime_marker():
    buffer_set = tile.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)
    slot = tile.buffer_slot(buffer_set, 0)

    marker = tile.release(slot)

    assert marker.op.name == "tile.release"
    assert isinstance(marker.type, ir.ScalarType)
    assert marker.type.dtype == DataType.BOOL


def test_release_accepts_ssa_variable_with_selected_slot_type():
    buffer_set = tile.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)
    slot = tile.buffer_slot(buffer_set, 0)
    slot_var = ir.Var("slot", slot.type, ir.Span.unknown())

    marker = tile.release(slot_var)

    assert marker.op.name == "tile.release"


def test_release_rejects_non_slot_tile():
    ordinary_tile = tile.create([16, 128], DataType.FP32, ir.MemorySpace.Acc)

    with pytest.raises(ValueError, match="selected by tile.buffer_slot"):
        tile.release(ordinary_tile)
