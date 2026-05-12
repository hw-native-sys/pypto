# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for distributed ops registered via OpRegistry.

After the MemRef-mirror redesign:

* ``WindowBufferType`` is a singleton (no per-instance fields).
* ``WindowBuffer`` is a :class:`Var` subclass with no ``name``/``dtype``
  fields; it wraps a base ``Var(PtrType)`` plus a per-rank byte size and
  host-staging flags. Constructed by the comm-collection pass.
* ``pld.alloc_window_buffer(size, name=...)`` is pure-allocation and returns
  the singleton :class:`PtrType` (same as ``tile.alloc``).
* ``pld.window(buf, shape, dtype=...)`` consumes a ``Ptr`` and returns
  :class:`DistributedTensorType`; ``window_buffer`` back-reference is
  ``None`` at parse time and filled in by the comm-collection pass later.
"""

import pytest
from pypto import DataType, ir


def _make_shape_tuple(values: list[int], span: ir.Span) -> ir.MakeTuple:
    return ir.MakeTuple([ir.ConstInt(v, DataType.INT64, span) for v in values], span)


# ---------------------------------------------------------------------------
# WindowBufferType singleton
# ---------------------------------------------------------------------------


def test_window_buffer_type_is_singleton():
    """``WindowBufferType.get()`` returns a structurally-equal instance every call."""
    a = ir.WindowBufferType.get()
    b = ir.WindowBufferType.get()
    assert a is b
    assert ir.structural_equal(a, ir.WindowBufferType())


# ---------------------------------------------------------------------------
# pld.alloc_window_buffer op
# ---------------------------------------------------------------------------


def test_alloc_window_buffer_returns_ptr_type():
    """Pure-allocation: alloc returns the singleton PtrType (mirrors tile.alloc)."""
    span = ir.Span.unknown()
    size = ir.ConstInt(1024, DataType.INT64, span)
    call = ir.create_op_call(
        "pld.alloc_window_buffer",
        [size],
        {"name": "buf"},
        span,
    )
    assert isinstance(call.type, ir.PtrType)
    # Op preserves the parser-injected name kwarg for downstream consumers.
    assert call.kwargs["name"] == "buf"
    # No dtype kwarg on the op surface — alloc is dtype-agnostic.
    assert "dtype" not in call.kwargs


def test_alloc_window_buffer_requires_non_empty_name():
    span = ir.Span.unknown()
    size = ir.ConstInt(4, DataType.INT64, span)
    with pytest.raises(Exception, match="non-empty 'name'"):
        ir.create_op_call(
            "pld.alloc_window_buffer",
            [size],
            {"name": ""},
            span,
        )


# ---------------------------------------------------------------------------
# WindowBuffer Var subclass
# ---------------------------------------------------------------------------


def test_window_buffer_is_var_subclass_wrapping_ptr():
    """WindowBuffer is a Var whose type is the singleton WindowBufferType,
    wrapping a base Ptr Var (mirrors MemRef wrapping a base Ptr)."""
    span = ir.Span.unknown()
    base = ir.Var("buf", ir.PtrType(), span)
    size = ir.ConstInt(64, DataType.INT64, span)
    wb = ir.WindowBuffer(base, size, span=span)
    assert isinstance(wb, ir.Var)
    assert isinstance(wb.type, ir.WindowBufferType)
    # name_hint flows from base.name_hint — no separate name field on
    # WindowBuffer (mirrors MemRef).
    assert wb.name_hint == "buf"
    assert wb.base is base
    assert isinstance(wb.size, ir.ConstInt)
    assert wb.size.value == 64
    assert wb.load_from_host is False
    assert wb.store_to_host is False


# ---------------------------------------------------------------------------
# pld.window op
# ---------------------------------------------------------------------------


def test_window_returns_distributed_tensor_with_no_buffer_at_parse_time():
    """``pld.window(ptr, shape, dtype=...)`` returns DistributedTensorType
    with shape + dtype set; ``window_buffer`` is None until the
    comm-collection pass populates it."""
    span = ir.Span.unknown()
    base = ir.Var("buf", ir.PtrType(), span)
    shape = _make_shape_tuple([64], span)
    call = ir.create_op_call("pld.window", [base, shape], {"dtype": DataType.FP16}, span)
    assert isinstance(call.type, ir.DistributedTensorType)
    assert call.type.dtype == DataType.FP16
    assert len(call.type.shape) == 1
    assert isinstance(call.type.shape[0], ir.ConstInt)
    assert call.type.shape[0].value == 64
    # window_buffer back-reference is filled in by the comm-collection pass,
    # not by the op deducer — at parse time it is None.
    assert call.type.window_buffer is None


def test_window_rejects_non_ptr_arg():
    """A Var with a non-PtrType type cannot be passed to ``pld.window``."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)
    bad = ir.Var("x", tensor_type, span)
    shape = _make_shape_tuple([64], span)
    with pytest.raises(Exception, match="Ptr"):
        ir.create_op_call("pld.window", [bad, shape], {"dtype": DataType.FP32}, span)


def test_window_rejects_non_make_tuple_shape():
    span = ir.Span.unknown()
    base = ir.Var("buf", ir.PtrType(), span)
    bad_shape = ir.ConstInt(8, DataType.INT64, span)
    with pytest.raises(Exception, match="shape tuple"):
        ir.create_op_call("pld.window", [base, bad_shape], {"dtype": DataType.FP32}, span)


# ---------------------------------------------------------------------------
# DistributedTensorType.window_buffer back-reference
# ---------------------------------------------------------------------------


def test_distributed_tensor_type_distinguishes_distinct_window_buffers():
    """Same shape + dtype but different window_buffer ⇒ structurally distinct."""
    span = ir.Span.unknown()
    base_a = ir.Var("buf_a", ir.PtrType(), span)
    base_b = ir.Var("buf_b", ir.PtrType(), span)
    wb_a = ir.WindowBuffer(base_a, ir.ConstInt(32, DataType.INT64, span), span=span)
    wb_b = ir.WindowBuffer(base_b, ir.ConstInt(32, DataType.INT64, span), span=span)
    shape = [ir.ConstInt(32, DataType.INT64, span)]
    dt_a = ir.DistributedTensorType(shape, DataType.FP32, wb_a)
    dt_b = ir.DistributedTensorType(shape, DataType.FP32, wb_b)
    assert dt_a.window_buffer is wb_a
    assert dt_b.window_buffer is wb_b
    assert not ir.structural_equal(dt_a, dt_b)


def test_distributed_tensor_type_with_and_without_window_buffer_differ():
    """Param-annotation form (no buffer) and bound form (with buffer) differ."""
    span = ir.Span.unknown()
    base = ir.Var("buf", ir.PtrType(), span)
    wb = ir.WindowBuffer(base, ir.ConstInt(32, DataType.INT64, span), span=span)
    shape = [ir.ConstInt(32, DataType.INT64, span)]
    dt_param = ir.DistributedTensorType(shape, DataType.FP32)
    dt_bound = ir.DistributedTensorType(shape, DataType.FP32, wb)
    assert dt_param.window_buffer is None
    assert dt_bound.window_buffer is wb
    assert not ir.structural_equal(dt_param, dt_bound)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
