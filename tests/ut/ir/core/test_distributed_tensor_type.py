# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for ``ir.DistributedTensorType`` (N1.2).

The distributed tensor subclass is distinguished from plain :class:`TensorType`
*only* by ``ObjectKind``; structurally identical fields. It exists so cross-rank
op verifiers (added in N6) can reject plain tensors via
``As<DistributedTensorType>``.
"""

import pytest
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    ConstInt,
    DistributedTensorType,
    Span,
    TensorType,
    assert_structural_equal,
    deserialize,
    serialize,
    structural_equal,
)


def _shape(*dims: int) -> list[ConstInt]:
    return [ConstInt(d, DataType.INT64, Span.unknown()) for d in dims]


def test_construct_with_constant_shape():
    """``DistributedTensorType([64], FP32)`` constructs and exposes shape/dtype."""
    dt = DistributedTensorType([64], DataType.FP32)
    assert dt.dtype == DataType.FP32
    assert len(dt.shape) == 1


def test_construct_with_expr_shape():
    """Expr-shape ctor mirrors TensorType's signature."""
    dt = DistributedTensorType(_shape(64, 128), DataType.FP32)
    assert len(dt.shape) == 2


def test_type_name_distinct_from_tensor_type():
    """Precise ObjectKind keeps DistributedTensorType separate from TensorType."""
    dt = DistributedTensorType([64], DataType.FP32)
    plain = TensorType([64], DataType.FP32)
    assert type(dt).__name__ == "DistributedTensorType"
    assert type(plain).__name__ == "TensorType"


def test_inherits_from_tensor_type():
    """C++ inheritance is preserved across the binding so DSL helpers that
    accept ``TensorType`` still work on the distributed subclass."""
    dt = DistributedTensorType([64], DataType.FP32)
    assert isinstance(dt, TensorType)


def test_structural_equal_same_shape_dtype():
    a = DistributedTensorType([64], DataType.FP32)
    b = DistributedTensorType([64], DataType.FP32)
    assert structural_equal(a, b)


def test_structural_not_equal_to_plain_tensor_type():
    """Plain TensorType and DistributedTensorType are distinct types — no
    cross-class structural equality, even with identical shape/dtype."""
    dt = DistributedTensorType([64], DataType.FP32)
    plain = TensorType([64], DataType.FP32)
    assert not structural_equal(dt, plain)


def test_structural_not_equal_different_dtype():
    a = DistributedTensorType([64], DataType.FP32)
    b = DistributedTensorType([64], DataType.FP16)
    assert not structural_equal(a, b)


def test_assert_structural_equal_passes():
    """``assert_structural_equal`` accepts equivalent distributed types."""
    a = DistributedTensorType([64], DataType.FP32)
    b = DistributedTensorType([64], DataType.FP32)
    assert_structural_equal(a, b)


def test_assert_structural_equal_diagnoses_class_mismatch():
    dt = DistributedTensorType([64], DataType.FP32)
    plain = TensorType([64], DataType.FP32)
    with pytest.raises(Exception, match="Type name mismatch"):
        assert_structural_equal(dt, plain)


def test_serialization_roundtrip():
    """The deserializer reconstructs the precise subclass."""
    from pypto.pypto_core.ir import Var  # noqa: PLC0415

    dt = DistributedTensorType([64, 128], DataType.FP32)
    # Wrap the type in a Var (the only way to feed a Type through the IRNode
    # serializer); the deserializer must restore the precise subclass.
    var = Var("v", dt, Span.unknown())
    blob = serialize(var)
    restored = deserialize(blob)
    assert isinstance(restored, Var)
    assert isinstance(restored.type, DistributedTensorType)
    assert structural_equal(dt, restored.type)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
