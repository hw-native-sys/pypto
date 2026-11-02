# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Register descriptors are independent of tile storage and preserve all semantics."""

import pytest
from pypto import DataType, ir
from pypto.ir.printer import python_print


def test_reg_type_fields_and_identity():
    vec = ir.RegType(64, DataType.FP32, ir.RegKind.VecReg)
    assert isinstance(vec, ir.Type)
    assert vec.shape == vec.lanes == 64
    assert vec.dtype == DataType.FP32
    assert vec.kind == ir.RegKind.VecReg
    same = ir.RegType(64, DataType.FP32, ir.RegKind.VecReg)
    assert ir.structural_equal(vec, same)
    assert ir.structural_hash(vec) == ir.structural_hash(same)
    for other in (
        ir.RegType(32, DataType.FP32, ir.RegKind.VecReg),
        ir.RegType(64, DataType.INT32, ir.RegKind.VecReg),
        ir.RegType(64, DataType.BOOL, ir.RegKind.MaskReg),
    ):
        assert not ir.structural_equal(vec, other)
        assert ir.structural_hash(vec) != ir.structural_hash(other)
    with pytest.raises(AttributeError):
        setattr(vec, "shape", 128)


@pytest.mark.parametrize("shape", [0, -1, -(2**63)])
def test_reg_type_rejects_nonpositive_lanes(shape):
    with pytest.raises(ValueError, match="positive"):
        ir.RegType(shape, DataType.FP32, ir.RegKind.VecReg)


@pytest.mark.parametrize("shape", [[64], 1.5, "64", b"64"])
def test_reg_type_rejects_noninteger_shape(shape):
    # Text and bytes must not become lane counts through nanobind integer coercion.
    with pytest.raises(TypeError):
        ir.RegType(shape, DataType.FP32, ir.RegKind.VecReg)


def test_reg_type_rejects_dtype_kind_mismatch():
    with pytest.raises(ValueError, match="BOOL"):
        ir.RegType(64, DataType.FP32, ir.RegKind.MaskReg)
    with pytest.raises(ValueError, match="BOOL"):
        ir.RegType(64, DataType.BOOL, ir.RegKind.VecReg)


@pytest.mark.parametrize("mask", [False, True])
def test_reg_type_serialization(mask):
    dtype = DataType.BOOL if mask else DataType.FP32
    kind = ir.RegKind.MaskReg if mask else ir.RegKind.VecReg
    var = ir.Var("r", ir.RegType(64, dtype, kind), ir.Span.unknown())
    restored = ir.deserialize(ir.serialize(var))
    assert isinstance(restored, ir.Var)
    assert isinstance(restored.type, ir.RegType)
    ir.assert_structural_equal(var, restored, enable_auto_mapping=True)
    assert ir.serialize(restored) == ir.serialize(var)


def _type_map(fields):
    return bytes([0x80 + len(fields)]) + b"".join(
        bytes([0xA0 + len(key)]) + key.encode() + value for key, value in sorted(fields.items())
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", b"\x02"),
        ("kind", b"\xcd\x01\x00"),
        ("kind", b"\xff"),
        ("kind", b"\xc3"),
        ("shape", b"\x00"),
        ("shape", b"\xc3"),
        ("shape", b"\xcf\x80\x00\x00\x00\x00\x00\x00\x00"),
        ("shape", b"\x91\x40"),
        ("dtype", b"\xcd\x01\x34"),
        ("dtype", b"\xcc\xff"),
    ],
)
def test_reg_type_rejects_malformed_serialized_descriptor(field, value):
    fields = {
        "type_kind": b"\xa7RegType",
        "shape": b"\x40",
        "dtype": bytes([DataType.FP32.code()]),
        "kind": b"\x00",
    }
    var = ir.Var("r", ir.RegType(64, DataType.FP32, ir.RegKind.VecReg), ir.Span.unknown())
    payload = ir.serialize(var)
    original = _type_map(fields)
    assert payload.count(original) == 1
    fields[field] = value
    with pytest.raises((ValueError, TypeError)):
        ir.deserialize(payload.replace(original, _type_map(fields)))


def test_reg_type_prints_scalar_logical_lane_count():
    vec = ir.RegType(64, DataType.FP32, ir.RegKind.VecReg)
    mask = ir.RegType(64, DataType.BOOL, ir.RegKind.MaskReg)
    assert python_print(vec, format=False) == "pl.Reg[64, pl.FP32, pl.RegKind.VecReg]"
    assert python_print(mask, format=False) == "pl.Reg[64, pl.BOOL, pl.RegKind.MaskReg]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
