# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural comparison and persistence of final buffer descriptors."""

from typing import Any

import pytest
from pypto import DataType, ir


def _buffer_type(**overrides: Any) -> ir.BufferType:
    fields = {"shape": [16, 32], "dtype": DataType.FP16, "memory_space": ir.MemorySpace.Vec}
    return ir.BufferType(**(fields | overrides))


def _round_trip_type(type_: ir.Type) -> ir.Type:
    value = ir.Var("buffer", type_, ir.Span.unknown())
    restored = ir.deserialize(ir.serialize(value))
    assert isinstance(restored, ir.Var)
    ir.assert_structural_equal(value, restored, enable_auto_mapping=True)
    assert ir.structural_hash(type_) == ir.structural_hash(restored.type)
    return restored.type


def _buffer_fields() -> dict[str, bytes]:
    """Raw MessagePack fields for the default descriptor used in corruption tests."""
    return {
        "type_kind": b"\xaaBufferType",
        "physical_shape": b"\x92\x10\x20",
        "dtype": bytes([DataType.FP16.code()]),
        "memory_space": bytes([ir.MemorySpace.Vec.value]),
        "valid_shape": b"\x92\x10\x20",
        "blayout": bytes([ir.TileLayout.row_major.value]),
        "slayout": bytes([ir.TileLayout.none_box.value]),
        "fractal": b"\xcd\x02\x00",
        "pad": bytes([ir.PadValue.null.value]),
        "compact": bytes([ir.CompactMode.null.value]),
    }


def _fixed_type_map(fields: list[tuple[str, bytes]]) -> bytes:
    """Build a small descriptor fixture, preserving duplicate keys for validation."""
    assert len(fields) < 16
    entries = []
    for key, value in fields:
        assert len(key) < 32
        entries.append(bytes([0xA0 + len(key)]) + key.encode() + value)
    return bytes([0x80 + len(fields)]) + b"".join(entries)


def _replace_buffer_descriptor(fields: list[tuple[str, bytes]]) -> bytes:
    value = ir.Var("buffer", _buffer_type(), ir.Span.unknown())
    payload = ir.serialize(value)
    original = _fixed_type_map(sorted(_buffer_fields().items()))
    assert payload.count(original) == 1
    return payload.replace(original, _fixed_type_map(fields))


def test_buffer_type_round_trip_preserves_all_descriptors():
    type_ = _buffer_type(
        valid_shape=[-1, 24],
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=256,
        pad=ir.PadValue.zero,
        compact=ir.CompactMode.normal,
    )
    restored = _round_trip_type(type_)
    assert isinstance(restored, ir.BufferType)
    assert restored.shape == [16, 32]
    assert restored.valid_shape == [-1, 24]
    assert restored.dtype == DataType.FP16
    assert restored.memory_space == ir.MemorySpace.Vec
    assert restored.blayout == ir.TileLayout.col_major
    assert restored.slayout == ir.TileLayout.row_major
    assert restored.fractal == 256
    assert restored.pad == ir.PadValue.zero
    assert restored.compact == ir.CompactMode.normal


@pytest.mark.parametrize("valid_shape", [[], [16, 32], [-1, -1], [0, 0]])
def test_buffer_valid_shape_round_trip(valid_shape):
    type_ = _buffer_type(valid_shape=valid_shape)
    restored = _round_trip_type(type_)
    assert isinstance(restored, ir.BufferType)
    assert restored.valid_shape == (valid_shape or [16, 32])


def test_buffer_implicit_and_explicit_full_valid_shapes_are_equal():
    implicit = _buffer_type()
    explicit = _buffer_type(valid_shape=[16, 32])
    ir.assert_structural_equal(implicit, explicit)
    assert ir.structural_hash(implicit) == ir.structural_hash(explicit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [16, 64]),
        ("dtype", DataType.FP32),
        ("memory_space", ir.MemorySpace.Mat),
        ("valid_shape", [-1, 32]),
        ("blayout", ir.TileLayout.col_major),
        ("slayout", ir.TileLayout.row_major),
        ("fractal", 256),
        ("pad", ir.PadValue.zero),
        ("compact", ir.CompactMode.normal),
    ],
)
def test_buffer_descriptor_fields_affect_equality_and_hash(field, value):
    original = _buffer_type()
    changed = _buffer_type(**{field: value})
    assert not ir.structural_equal(original, changed)
    assert ir.structural_hash(original) != ir.structural_hash(changed)


def test_multi_buffer_round_trip_preserves_element_and_slot_count():
    element = _buffer_type(valid_shape=[-1, 32])
    restored = _round_trip_type(ir.MultiBufferType(element, 3))
    assert isinstance(restored, ir.MultiBufferType)
    assert restored.slot_count == 3
    ir.assert_structural_equal(restored.element_type, element)


def test_multi_buffer_equality_and_hash_include_element_and_slot_count():
    original = ir.MultiBufferType(_buffer_type(), 2)
    for changed in [
        ir.MultiBufferType(_buffer_type(), 3),
        ir.MultiBufferType(_buffer_type(dtype=DataType.FP32), 2),
    ]:
        assert not ir.structural_equal(original, changed)
        assert ir.structural_hash(original) != ir.structural_hash(changed)


def test_nested_buffer_tuple_round_trip():
    type_ = ir.TupleType([_buffer_type(), ir.MultiBufferType(_buffer_type(), 2)])
    restored = _round_trip_type(type_)
    assert isinstance(restored, ir.TupleType)
    assert isinstance(restored.types[0], ir.BufferType)
    assert isinstance(restored.types[1], ir.MultiBufferType)


def test_void_eval_statement_round_trip_and_distinction_from_unknown():
    span = ir.Span.unknown()
    call = ir.Call(ir.Op("buffer.test_write"), [], {}, ir.VoidType(), span)
    statement = ir.EvalStmt(call, span)
    restored = ir.deserialize(ir.serialize(statement))
    ir.assert_structural_equal(statement, restored)
    assert ir.structural_hash(statement) == ir.structural_hash(restored)
    assert not ir.structural_equal(ir.VoidType(), ir.UnknownType())
    assert ir.structural_hash(ir.VoidType()) != ir.structural_hash(ir.UnknownType())


@pytest.mark.parametrize("missing_field", sorted(set(_buffer_fields()) - {"type_kind"}))
def test_buffer_deserialization_rejects_missing_descriptors(missing_field):
    fields = _buffer_fields()
    del fields[missing_field]
    with pytest.raises(ValueError, match=f"missing required field '{missing_field}'"):
        ir.deserialize(_replace_buffer_descriptor(sorted(fields.items())))


@pytest.mark.parametrize("extra_field", ["memref", "base", "addr", "shape"])
def test_buffer_deserialization_rejects_storage_identity_or_legacy_shape(extra_field):
    fields = _buffer_fields() | {extra_field: b"\xc0"}
    with pytest.raises(ValueError, match="unexpected fields"):
        ir.deserialize(_replace_buffer_descriptor(sorted(fields.items())))


def test_buffer_deserialization_rejects_duplicate_descriptor_fields():
    fields = sorted(_buffer_fields().items()) + [("dtype", bytes([DataType.FP32.code()]))]
    with pytest.raises(ValueError, match="duplicate field 'dtype'"):
        ir.deserialize(_replace_buffer_descriptor(fields))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("physical_shape", b"\xc0", "integer array"),
        ("physical_shape", b"\x92\x80\x20", "must be an integer"),
        ("valid_shape", b"\xc0", "integer array"),
        ("valid_shape", b"\x92\xc3\x20", "must be an integer"),
        ("dtype", b"\xc3", "must be an integer"),
        ("memory_space", b"\xc0", "must be an integer"),
        ("blayout", b"\xc3", "must be an integer"),
        ("slayout", b"\xc3", "must be an integer"),
        ("pad", b"\xc3", "must be an integer"),
        ("compact", b"\xc3", "must be an integer"),
        ("fractal", b"\xc3", "must be an integer"),
    ],
)
def test_buffer_deserialization_rejects_wrong_field_types(field, value, message):
    fields = _buffer_fields() | {field: value}
    with pytest.raises(ValueError, match=message):
        ir.deserialize(_replace_buffer_descriptor(sorted(fields.items())))


def test_multi_buffer_deserialization_rejects_non_buffer_element():
    value = ir.Var("slots", ir.MultiBufferType(_buffer_type(), 2), ir.Span.unknown())
    original = _fixed_type_map(sorted(_buffer_fields().items()))
    replacement = _fixed_type_map([("type_kind", b"\xabUnknownType")])
    payload = ir.serialize(value)
    assert payload.count(original) == 1
    with pytest.raises(ValueError, match="element_type must be a BufferType"):
        ir.deserialize(payload.replace(original, replacement))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
