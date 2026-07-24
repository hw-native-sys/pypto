# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the intra-core multi-buffer tile ops.

``tile.alloc_multi`` / ``tile.multi_get`` model ptoas ``pto.alloc_multi_tile`` /
``pto.multi_tile_get``: one allocation reserves ``count`` physical tile slots
sharing a slot shape, and each iteration selects a slot at a runtime index
(typically ``loop_var % count``). These tests cover the IR-level type deduction
and the ``count`` bound validation only; codegen emission is covered end-to-end
elsewhere.
"""

import pytest

from pypto import DataType, ir


def _slot_shape(span):
    return ir.MakeTuple([ir.ConstInt(16, DataType.INT32, span), ir.ConstInt(256, DataType.INT32, span)], span)


def test_alloc_multi_deduces_slot_tile_type():
    """tile.alloc_multi returns the per-slot TileType (not a Ptr)."""
    span = ir.Span.unknown()
    call = ir.create_op_call(
        "tile.alloc_multi",
        [_slot_shape(span)],
        {"dtype": DataType.FP32, "target_memory": ir.MemorySpace.Vec, "count": 2},
        span,
    )
    result = call.type
    assert isinstance(result, ir.TileType)
    assert [e.value for e in result.shape if isinstance(e, ir.ConstInt)] == [16, 256]
    assert result.dtype == DataType.FP32


def test_multi_get_deduces_slot_tile_type():
    """tile.multi_get returns the multi-buffer's per-slot TileType."""
    span = ir.Span.unknown()
    alloc = ir.create_op_call(
        "tile.alloc_multi",
        [_slot_shape(span)],
        {"dtype": DataType.FP32, "target_memory": ir.MemorySpace.Vec, "count": 4},
        span,
    )
    mtb = ir.Var("mtb", alloc.type, span)
    idx = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    get = ir.create_op_call("tile.multi_get", [mtb, idx], {"count": 4}, span)

    result = get.type
    assert isinstance(result, ir.TileType)
    assert [e.value for e in result.shape if isinstance(e, ir.ConstInt)] == [16, 256]
    assert result.dtype == DataType.FP32


@pytest.mark.parametrize("bad_count", [0, 1, 17, 32])
def test_alloc_multi_rejects_out_of_range_count(bad_count):
    """count must be in [2, 16] (mirrors the ptoas multi_tile_buf verifier)."""
    span = ir.Span.unknown()
    with pytest.raises(ValueError, match=r"count.*must be in \[2, 16\]"):
        ir.create_op_call(
            "tile.alloc_multi",
            [_slot_shape(span)],
            {"dtype": DataType.FP32, "target_memory": ir.MemorySpace.Vec, "count": bad_count},
            span,
        )


@pytest.mark.parametrize("good_count", [2, 8, 16])
def test_alloc_multi_accepts_in_range_count(good_count):
    """Boundary and interior depths in [2, 16] are accepted."""
    span = ir.Span.unknown()
    call = ir.create_op_call(
        "tile.alloc_multi",
        [_slot_shape(span)],
        {"dtype": DataType.FP32, "target_memory": ir.MemorySpace.Vec, "count": good_count},
        span,
    )
    assert isinstance(call.type, ir.TileType)


def test_multi_get_requires_tile_source():
    """tile.multi_get's first argument must be a tile (from tile.alloc_multi)."""
    span = ir.Span.unknown()
    not_a_tile = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    idx = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    with pytest.raises(ValueError, match="requires its first argument to be a tile"):
        ir.create_op_call("tile.multi_get", [not_a_tile, idx], {"count": 2}, span)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
