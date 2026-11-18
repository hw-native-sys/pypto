# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Completion descriptors preserve source pipes without duplicating core ownership."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.printer import python_print


def test_sync_token_type_fields_and_structural_comparison():
    token = ir.SyncTokenType(ir.PipeType.MTE2)
    assert isinstance(token, ir.Type)
    assert token.pipe == ir.PipeType.MTE2
    assert not hasattr(token, "core_type")
    same = ir.SyncTokenType(ir.PipeType.MTE2)
    ir.assert_structural_equal(token, same)
    assert ir.structural_hash(token) == ir.structural_hash(same)
    assert not ir.structural_equal(token, ir.SyncTokenType(ir.PipeType.V))
    assert ir.structural_hash(token) != ir.structural_hash(ir.SyncTokenType(ir.PipeType.V))
    with pytest.raises(AttributeError):
        setattr(token, "pipe", ir.PipeType.V)


def test_sync_token_type_validates_semantics_without_hardware_tables():
    with pytest.raises(ValueError, match="concrete PipeType"):
        ir.SyncTokenType(ir.PipeType.ALL)


def test_sync_token_annotation_has_one_concrete_pipeline():
    annotation = cast(ir.Type, pl.SyncToken[pl.PipeType.MTE2])
    ir.assert_structural_equal(annotation, ir.SyncTokenType(ir.PipeType.MTE2))
    # Core ownership is supplied by the enclosing AIC/AIV function, never an
    # optional annotation field whose default could silently disagree with it.
    with pytest.raises(TypeError, match="one PipeType"):
        pl.SyncToken[pl.PipeType.MTE2, "aiv"]
    with pytest.raises(TypeError, match="one PipeType"):
        pl.SyncToken[1]
    with pytest.raises(ValueError, match="concrete PipeType"):
        pl.SyncToken[pl.PipeType.ALL]


def test_sync_token_type_serialization_and_printing():
    token = ir.SyncTokenType(ir.PipeType.MTE2)
    var = ir.Var("token", token, ir.Span.unknown())
    restored = ir.deserialize(ir.serialize(var))
    ir.assert_structural_equal(var, restored, enable_auto_mapping=True)
    assert ir.serialize(var) == ir.serialize(restored)
    assert python_print(token, format=False) == "pl.SyncToken[pl.PipeType.MTE2]"


def _pack_type_fields(fields: dict[str, bytes]) -> bytes:
    # Mutate only a descriptor in an otherwise valid blob, without requiring
    # an optional Python msgpack package just to generate malformed input.
    return bytes([0x80 + len(fields)]) + b"".join(
        bytes([0xA0 + len(key)]) + key.encode() + value for key, value in sorted(fields.items())
    )


@pytest.fixture
def serialized_token() -> tuple[bytes, dict[str, bytes]]:
    fields = {
        "type_kind": b"\xadSyncTokenType",
        "pipe": bytes([int(ir.PipeType.MTE2)]),
    }
    var = ir.Var("token", ir.SyncTokenType(ir.PipeType.MTE2), ir.Span.unknown())
    payload = ir.serialize(var)
    assert payload.count(_pack_type_fields(fields)) == 1
    return payload, fields


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipe", b"\x07"),  # ALL has no concrete completion stream.
        ("pipe", b"\xff"),
        ("pipe", b"\xce\x80\x00\x00\x00"),  # Reject enum narrowing.
        ("pipe", b"\xc3"),
    ],
)
def test_sync_token_type_rejects_invalid_serialized_fields(serialized_token, field, value):
    payload, fields = serialized_token
    original = _pack_type_fields(fields)
    fields[field] = value
    with pytest.raises(ValueError):
        ir.deserialize(payload.replace(original, _pack_type_fields(fields)))


def test_sync_token_type_requires_serialized_pipeline(serialized_token):
    payload, fields = serialized_token
    original = _pack_type_fields(fields)
    del fields["pipe"]
    with pytest.raises(ValueError, match="missing required field"):
        ir.deserialize(payload.replace(original, _pack_type_fields(fields)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
