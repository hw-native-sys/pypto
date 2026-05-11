# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for array operations (array.create / array.get_element / array.update_element).

ArrayType is the on-core fixed-size 1-D homogeneous array. Writes are
SSA-functional via ``array.update_element`` — they return a fresh ArrayType
SSA value rather than mutating in place.
"""

from typing import cast

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir.op import array as ir_array


def _extent(t: ir.ArrayType) -> int:
    """Return ArrayType's extent as a Python int (helper to narrow Expr -> ConstInt)."""
    return cast(ir.ConstInt, t.extent).value


# ----------------------------------------------------------------------------
# ArrayType construction
# ----------------------------------------------------------------------------


def test_array_type_construct_int_extent():
    t = ir.ArrayType(DataType.INT32, 16)
    assert t.dtype == DataType.INT32
    assert t.memory_space == ir.MemorySpace.ScalarLocal
    assert len(t.shape) == 1
    assert _extent(t) == 16


def test_array_type_construct_const_int_extent():
    extent = ir.ConstInt(8, DataType.INDEX, ir.Span.unknown())
    t = ir.ArrayType(DataType.INT64, extent)
    assert t.dtype == DataType.INT64
    assert _extent(t) == 8


def test_array_type_accepts_all_integer_dtypes():
    for dt in (
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
        DataType.BOOL,
    ):
        t = ir.ArrayType(dt, 4)
        assert t.dtype == dt


def test_array_type_rejects_non_integer_dtype():
    for dt in (DataType.FP16, DataType.FP32, DataType.BF16):
        with pytest.raises(ValueError, match="ArrayType element dtype must be"):
            ir.ArrayType(dt, 4)


def test_array_type_rejects_non_positive_extent():
    with pytest.raises(ValueError, match="extent must be positive"):
        ir.ArrayType(DataType.INT32, 0)
    with pytest.raises(ValueError, match="extent must be positive"):
        ir.ArrayType(DataType.INT32, -1)


def test_array_type_rejects_dynamic_extent():
    v = ir.Var("n", ir.ScalarType(DataType.INT32), ir.Span.unknown())
    with pytest.raises(ValueError, match="extent must be a compile-time ConstInt"):
        ir.ArrayType(DataType.INT32, v)


# ----------------------------------------------------------------------------
# array.create / get_element / update_element ops
# ----------------------------------------------------------------------------


def test_array_create_op():
    call = ir_array.create(16, DataType.INT32)
    assert isinstance(call, ir.Call)
    assert call.op.name == "array.create"
    assert isinstance(call.type, ir.ArrayType)
    array_type = cast(ir.ArrayType, call.type)
    assert array_type.dtype == DataType.INT32
    assert _extent(array_type) == 16
    assert array_type.memory_space == ir.MemorySpace.ScalarLocal


def test_array_get_element_op_returns_scalar():
    arr = ir_array.create(8, DataType.INT32)
    idx = ir.ConstInt(3, DataType.INT32, ir.Span.unknown())
    call = ir_array.get_element(arr, idx)
    assert call.op.name == "array.get_element"
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.INT32


def test_array_update_element_op_returns_array():
    arr = ir_array.create(8, DataType.INT32)
    idx = ir.ConstInt(3, DataType.INT32, ir.Span.unknown())
    val = ir.ConstInt(42, DataType.INT32, ir.Span.unknown())
    call = ir_array.update_element(arr, idx, val)
    assert call.op.name == "array.update_element"
    array_type = cast(ir.ArrayType, call.type)
    assert array_type.dtype == DataType.INT32
    assert _extent(array_type) == 8


def test_array_update_element_rejects_dtype_mismatch():
    arr = ir_array.create(8, DataType.INT32)
    idx = ir.ConstInt(0, DataType.INT32, ir.Span.unknown())
    bad_val = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
    with pytest.raises(ValueError, match="value dtype .* must match array dtype"):
        ir_array.update_element(arr, idx, bad_val)


def test_array_get_element_rejects_non_array_first_arg():
    not_arr = ir.ConstInt(0, DataType.INT32, ir.Span.unknown())
    idx = ir.ConstInt(0, DataType.INT32, ir.Span.unknown())
    with pytest.raises(ValueError, match="first argument must be ArrayType"):
        ir_array.get_element(not_arr, idx)


def test_array_create_rejects_dynamic_extent():
    n = ir.Var("n", ir.ScalarType(DataType.INT32), ir.Span.unknown())
    with pytest.raises(ValueError, match="must be a compile-time ConstInt"):
        ir_array.create(n, DataType.INT32)


# ----------------------------------------------------------------------------
# DSL @pl.function end-to-end
# ----------------------------------------------------------------------------


def test_pl_function_with_array_indexing():
    """The DSL parser desugars arr[i] / arr[i]=v into array.get_element / update_element."""

    @pl.function
    def kernel(x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        arr = pl.array.create(8, pl.INT32)
        arr[0] = 7
        arr[3] = 42
        # Side-effecting reads ensure get_element calls survive in the body.
        _ = arr[0]
        _ = arr[3]
        return x

    # Walk the SeqStmts and confirm we see the expected ops.
    op_names = []
    body = kernel.body
    assert isinstance(body, ir.SeqStmts)
    for stmt in body.stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            op_names.append(stmt.value.op.name)
    assert "array.create" in op_names
    assert op_names.count("array.update_element") == 2
    assert op_names.count("array.get_element") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
