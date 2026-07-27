# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the first-class tile buffer-set storage type."""

import pytest
from pypto import DataType, ir


def test_tile_buffer_set_type_exposes_homogeneous_slot_contract():
    """Changing the type to lose count or slot metadata must fail this test."""
    buffer_set_type = ir.TileBufferSetType(
        [16, 128],
        DataType.FP32,
        2,
        None,
        None,
        ir.MemorySpace.Acc,
    )

    assert buffer_set_type.count == 2
    assert isinstance(buffer_set_type.shape[0], ir.ConstInt)
    assert isinstance(buffer_set_type.shape[1], ir.ConstInt)
    assert [buffer_set_type.shape[0].value, buffer_set_type.shape[1].value] == [16, 128]
    assert buffer_set_type.dtype == DataType.FP32
    assert buffer_set_type.memory_space == ir.MemorySpace.Acc


@pytest.mark.parametrize("count", [1, 17])
def test_tile_buffer_set_type_rejects_invalid_count(count):
    """Changing the hardware-supported count bounds must fail this test."""
    with pytest.raises(ValueError, match=r"count.*must be in \[2, 16\]"):
        ir.TileBufferSetType([16, 128], DataType.FP32, count, None, None, ir.MemorySpace.Acc)


@pytest.mark.parametrize("shape", [[], [0, 128], [-1, 128]])
def test_tile_buffer_set_type_rejects_non_positive_static_shape(shape):
    """Allowing an empty or non-positive physical slot must fail this test."""
    with pytest.raises(ValueError, match="shape.*non-empty.*positive"):
        ir.TileBufferSetType(shape, DataType.FP32, 2, None, None, ir.MemorySpace.Acc)


def test_tile_buffer_set_type_rejects_dynamic_shape():
    """Allowing a runtime-sized physical slot must fail this test."""
    dim = ir.Var("dim", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    with pytest.raises(ValueError, match="shape.*static"):
        ir.TileBufferSetType(
            [dim, ir.ConstInt(128, DataType.INDEX, ir.Span.unknown())],
            DataType.FP32,
            2,
            None,
            None,
            ir.MemorySpace.Acc,
        )


@pytest.mark.parametrize("memory_space", [None, ir.MemorySpace.DDR, ir.MemorySpace.ScalarLocal])
def test_tile_buffer_set_type_rejects_non_tile_memory_space(memory_space):
    """Allowing non-tile storage for a physical slot group must fail this test."""
    with pytest.raises(ValueError, match="memory_space.*on-chip tile memory"):
        ir.TileBufferSetType([16, 128], DataType.FP32, 2, None, None, memory_space)


def test_tile_buffer_set_type_count_participates_in_structural_identity():
    """Ignoring count in equality or hashing must fail this test."""
    two_slots = ir.TileBufferSetType([16, 128], DataType.FP32, 2, None, None, ir.MemorySpace.Acc)
    same_two_slots = ir.TileBufferSetType([16, 128], DataType.FP32, 2, None, None, ir.MemorySpace.Acc)
    three_slots = ir.TileBufferSetType([16, 128], DataType.FP32, 3, None, None, ir.MemorySpace.Acc)

    assert ir.structural_equal(two_slots, same_two_slots)
    assert ir.structural_hash(two_slots) == ir.structural_hash(same_two_slots)
    assert not ir.structural_equal(two_slots, three_slots)
    assert ir.structural_hash(two_slots) != ir.structural_hash(three_slots)


def test_tile_buffer_set_type_serialization_round_trip():
    """Dropping any storage field during serialization must fail this test."""
    original_type = ir.TileBufferSetType([16, 128], DataType.FP32, 3, None, None, ir.MemorySpace.Right)
    original = ir.Var("buffers", original_type, ir.Span.unknown())

    restored = ir.deserialize(ir.serialize(original))

    assert isinstance(restored, ir.Var)
    assert isinstance(restored.type, ir.TileBufferSetType)
    assert restored.type.count == 3
    ir.assert_structural_equal(original, restored, enable_auto_mapping=True)


def test_tile_buffer_set_type_python_annotation_is_stable():
    """Changing the public annotation order or omitting count must fail this test."""
    buffer_set_type = ir.TileBufferSetType([16, 128], DataType.FP32, 2, None, None, ir.MemorySpace.Acc)

    assert ir.python_print(buffer_set_type) == "pl.TileBufferSet[[16, 128], pl.FP32, 2, pl.Mem.Acc]"
