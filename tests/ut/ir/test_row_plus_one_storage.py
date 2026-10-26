# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public RowPlusOne descriptors preserve a layout contract independent of FP4 conversion."""

import pypto.language as pl
import pytest
from pypto import DataType, ir

SPAN = ir.Span.unknown()


def _const(n: int) -> ir.ConstInt:
    return ir.ConstInt(n, DataType.INDEX, SPAN)


def test_row_plus_one_descriptor_roundtrips_without_fp4_conversion():
    # The type describes FP8 storage itself; selecting a producer is a separate
    # optimization policy. A plain parameter needs no packed conversion op.
    view = ir.TileView(
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        compact=ir.CompactMode.row_plus_one,
    )
    value = ir.Var(
        "value",
        ir.TileType([128, 256], DataType.FP8E4M3FN, tile_view=view, memory_space=ir.MemorySpace.Vec),
        SPAN,
    )
    func = ir.Function(
        "kernel", [(value, ir.ParamDirection.In)], [], ir.EvalStmt(value, SPAN), SPAN, ir.FunctionType.AIV
    )
    program = ir.Program([func], "layout", SPAN)
    ir.assert_structural_equal(ir.deserialize(ir.serialize(program)), program)
    ir.assert_structural_equal(pl.parse_program(ir.python_print(program)), program)


@pytest.mark.parametrize("shape", [(127, 256), (128, 255), (0, 256)])
def test_row_plus_one_public_tile_rejects_unaligned_physical_geometry(shape):
    view = ir.TileView(
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        compact=ir.CompactMode.row_plus_one,
    )
    with pytest.raises(ValueError, match="static positive physical rows"):
        ir.TileType(list(shape), DataType.FP8E4M3FN, tile_view=view, memory_space=ir.MemorySpace.Vec)


def test_row_plus_one_public_buffer_descriptor_is_explicitly_unsupported():
    # Buffer IR's byte-count and multi-buffer recipes are dense. Rejecting the
    # descriptor prevents an explicit buffer.alloc from under-reserving padding.
    with pytest.raises(ValueError, match="use Tile IR"):
        ir.BufferType(
            [128, 256],
            DataType.FP8E4M3FN,
            ir.MemorySpace.Vec,
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            compact=ir.CompactMode.row_plus_one,
        )


def test_row_plus_one_public_tile_rejects_multi_slot_memref():
    view = ir.TileView(
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        compact=ir.CompactMode.row_plus_one,
    )
    base = ir.Var("padded_slots", ir.PtrType(), SPAN)
    memory = ir.MemRef(base, 0, 33024, slots=2, slot=_const(0))
    with pytest.raises(ValueError, match="independent Tile allocations"):
        ir.TileType([128, 256], DataType.FP8E4M3FN, memory, view, ir.MemorySpace.Vec)
