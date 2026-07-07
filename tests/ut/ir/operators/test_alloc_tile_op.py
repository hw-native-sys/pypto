# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the first-class ``alloc_tile`` IR op (issue #1956).

``alloc_tile`` is the explicit tile *handle* onto a region: it references
``(base Ptr, byte_offset, shape)`` and deduces the typed ``TileType`` handle that
carries that ``MemRef``. It is complementary to ``tile.alloc`` (the region → base
``Ptr``). Codegen emits ``pto.alloc_tile`` from it 1:1; a placement/must-alias
pass emits exactly one per must-alias group at a scope that dominates all uses, so
aliasing is structural (shared handle SSA) rather than physical-address based.
"""

import pytest
from pypto import DataType, ir


def _make_alloc_tile(
    *,
    offset: int = 16384,
    shape: tuple[int, ...] = (64, 64),
    dtype: DataType = DataType.FP32,
    memory_space: ir.MemorySpace = ir.MemorySpace.Vec,
) -> ir.Call:
    span = ir.Span.unknown()
    base = ir.Var("mem_vec_0", ir.PtrType.get(), span)
    byte_offset = ir.ConstInt(offset, DataType.INT64, span)
    shape_tuple = ir.MakeTuple([ir.ConstInt(d, DataType.INT64, span) for d in shape], span)
    return ir.create_op_call(
        "alloc_tile",
        [base, byte_offset, shape_tuple],
        {"dtype": dtype, "memory_space": memory_space},
        span,
    )


def test_alloc_tile_is_registered():
    assert ir.is_op_registered("alloc_tile")
    # Registry getter round-trips the canonical name (guards against typos).
    assert ir.get_op("alloc_tile").name == "alloc_tile"


def test_alloc_tile_deduces_tile_handle_type():
    call = _make_alloc_tile()
    t = call.type
    assert isinstance(t, ir.TileType)
    dims = []
    for e in t.shape:
        assert isinstance(e, ir.ConstInt)
        dims.append(e.value)
    assert dims == [64, 64]
    assert t.dtype == DataType.FP32
    assert t.memory_space == ir.MemorySpace.Vec


def test_alloc_tile_result_carries_memref():
    call = _make_alloc_tile(offset=32768)
    t = call.type
    assert isinstance(t, ir.TileType)
    memref = t.memref
    assert memref is not None, "alloc_tile handle must carry a MemRef referencing (base, byte_offset)"
    assert memref.base_.name_hint == "mem_vec_0"
    offset = memref.byte_offset_
    assert isinstance(offset, ir.ConstInt)
    assert offset.value == 32768


def test_alloc_tile_memref_size_tracks_shape_and_dtype():
    # 64 * 64 * 4 bytes (FP32) = 16384 bytes.
    call = _make_alloc_tile(shape=(64, 64), dtype=DataType.FP32)
    t = call.type
    assert isinstance(t, ir.TileType)
    assert t.memref is not None
    assert t.memref.size_ == 64 * 64 * 4


def test_alloc_tile_distinct_from_region_tile_alloc():
    # tile.alloc returns a Ptr (the region); alloc_tile returns a TileType (the
    # handle). The two levels are complementary, not interchangeable.
    span = ir.Span.unknown()
    region = ir.create_op_call(
        "tile.alloc",
        [
            ir.ConstInt(int(ir.MemorySpace.Vec.value), DataType.INT64, span),
            ir.ConstInt(4096, DataType.INT64, span),
        ],
        span,
    )
    assert not isinstance(region.type, ir.TileType)
    assert isinstance(_make_alloc_tile().type, ir.TileType)


def test_alloc_tile_rejects_wrong_arg_count():
    span = ir.Span.unknown()
    base = ir.Var("mem_vec_0", ir.PtrType.get(), span)
    with pytest.raises(ValueError):
        # Missing the shape argument.
        ir.create_op_call(
            "alloc_tile",
            [base, ir.ConstInt(0, DataType.INT64, span)],
            {"dtype": DataType.FP32, "memory_space": ir.MemorySpace.Vec},
            span,
        )


def test_alloc_tile_rejects_non_constant_shape():
    span = ir.Span.unknown()
    base = ir.Var("mem_vec_0", ir.PtrType.get(), span)
    dyn = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    with pytest.raises(ValueError):
        ir.create_op_call(
            "alloc_tile",
            [base, ir.ConstInt(0, DataType.INT64, span), ir.MakeTuple([dyn], span)],
            {"dtype": DataType.FP32, "memory_space": ir.MemorySpace.Vec},
            span,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
