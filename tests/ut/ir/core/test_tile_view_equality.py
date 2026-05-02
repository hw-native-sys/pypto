# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for TileView equality operators."""

import pytest
from pypto import DataType, ir


def _make_span():
    return ir.Span.unknown()


def _make_const(value, span=None):
    if span is None:
        span = _make_span()
    return ir.ConstInt(value, DataType.INT64, span)


def _make_view(
    valid_shape=None,
    stride=None,
    start_offset=None,
    blayout=ir.TileLayout.row_major,
    slayout=ir.TileLayout.none_box,
    fractal=512,
    pad=ir.PadValue.null,
):
    """Create a TileView with given parameters, using sensible defaults."""
    span = _make_span()
    if valid_shape is None:
        valid_shape = [_make_const(16, span), _make_const(16, span)]
    if stride is None:
        stride = [_make_const(1, span), _make_const(16, span)]
    if start_offset is None:
        start_offset = _make_const(0, span)
    return ir.TileView(valid_shape, stride, start_offset, blayout, slayout, fractal, pad)


class TestTileViewEquality:
    """Tests for TileView.__eq__ and __ne__."""

    def test_default_views_equal(self):
        """Two default-constructed TileViews are equal."""
        v1 = ir.TileView()
        v2 = ir.TileView()
        assert v1 == v2
        assert not (v1 != v2)

    def test_identical_views_equal(self):
        """TileViews constructed with the same parameters are equal."""
        v1 = _make_view()
        v2 = _make_view()
        assert v1 == v2

    def test_different_valid_shape(self):
        """Views with different valid_shape are not equal."""
        span = _make_span()
        v1 = _make_view(valid_shape=[_make_const(16, span), _make_const(16, span)])
        v2 = _make_view(valid_shape=[_make_const(32, span), _make_const(16, span)])
        assert v1 != v2
        assert not (v1 == v2)

    def test_different_valid_shape_length(self):
        """Views with different-length valid_shape are not equal."""
        span = _make_span()
        v1 = _make_view(valid_shape=[_make_const(16, span)])
        v2 = _make_view(valid_shape=[_make_const(16, span), _make_const(16, span)])
        assert v1 != v2

    def test_different_stride(self):
        """Views with different stride are not equal."""
        span = _make_span()
        v1 = _make_view(stride=[_make_const(1, span), _make_const(16, span)])
        v2 = _make_view(stride=[_make_const(1, span), _make_const(32, span)])
        assert v1 != v2

    def test_different_start_offset(self):
        """Views with different start_offset are not equal."""
        v1 = _make_view(start_offset=_make_const(0))
        v2 = _make_view(start_offset=_make_const(8))
        assert v1 != v2

    def test_different_blayout(self):
        """Views with different blayout are not equal."""
        v1 = _make_view(blayout=ir.TileLayout.row_major)
        v2 = _make_view(blayout=ir.TileLayout.none_box)
        assert v1 != v2

    def test_different_slayout(self):
        """Views with different slayout are not equal."""
        v1 = _make_view(slayout=ir.TileLayout.none_box)
        v2 = _make_view(slayout=ir.TileLayout.row_major)
        assert v1 != v2

    def test_different_fractal(self):
        """Views with different fractal are not equal."""
        v1 = _make_view(fractal=512)
        v2 = _make_view(fractal=256)
        assert v1 != v2

    def test_different_pad(self):
        """Views with different pad are not equal."""
        v1 = _make_view(pad=ir.PadValue.null)
        v2 = _make_view(pad=ir.PadValue.zero)
        assert v1 != v2

    def test_symbolic_same_object_equal(self):
        """Symbolic exprs that are the same object compare equal."""
        span = _make_span()
        sym = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        v1 = _make_view(valid_shape=[sym], stride=[_make_const(1, span)])
        v2 = _make_view(valid_shape=[sym], stride=[_make_const(1, span)])
        assert v1 == v2

    def test_symbolic_different_objects_not_equal(self):
        """Different symbolic expr objects compare not-equal (conservative)."""
        span = _make_span()
        sym1 = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        sym2 = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        v1 = _make_view(valid_shape=[sym1], stride=[_make_const(1, span)])
        v2 = _make_view(valid_shape=[sym2], stride=[_make_const(1, span)])
        assert v1 != v2


class TestTileViewHashEqConsistency:
    """Regression tests for the Python hash/eq contract on TileView."""

    def test_default_views_hash_equally(self):
        v1 = ir.TileView()
        v2 = ir.TileView()
        assert v1 == v2
        assert hash(v1) == hash(v2)

    def test_constint_value_carve_out_in_hash(self):
        # AreExprsEqual treats two distinct ConstInt nodes with the same value
        # as equal. Hash must follow suit.
        span = _make_span()
        v1 = _make_view(valid_shape=[_make_const(16, span), _make_const(16, span)])
        v2 = _make_view(valid_shape=[_make_const(16, span), _make_const(16, span)])
        assert v1 == v2
        assert hash(v1) == hash(v2)
        assert v1 in {v2}

    def test_distinct_var_ptrs_remain_distinct(self):
        # Two distinct Var pointers with the same name compare unequal under
        # AreExprsEqual. Hash collisions are legal under Python's contract, so
        # check the membership behavior callers actually depend on.
        span = _make_span()
        sym1 = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        sym2 = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        v1 = _make_view(valid_shape=[sym1], stride=[_make_const(1, span)])
        v2 = _make_view(valid_shape=[sym2], stride=[_make_const(1, span)])
        assert v1 != v2
        assert v1 not in {v2}

    def test_shared_var_ptr_hashes_equally(self):
        # Same Var instance shared across both views — eq AND hash equal.
        span = _make_span()
        sym = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        v1 = _make_view(valid_shape=[sym], stride=[_make_const(1, span)])
        v2 = _make_view(valid_shape=[sym], stride=[_make_const(1, span)])
        assert v1 == v2
        assert hash(v1) == hash(v2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
