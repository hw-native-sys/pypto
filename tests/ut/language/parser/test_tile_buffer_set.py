# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser tests for tuple-like explicit tile buffer sets."""

import pypto.language as pl
from pypto import ir


def test_dynamic_buffer_slot_parses_to_dedicated_ir_operations():
    @pl.function
    def rotate(index: pl.Scalar[pl.INDEX]):
        buffers = pl.create_tile_buffers(2, [16, 128], pl.FP32, pl.Mem.Acc)
        slot = buffers[index % 2]
        pl.tile.release(slot)

    printed = rotate.as_python()
    assert "tile.create_buffer_set" in printed
    assert "tile.buffer_slot" in printed
    assert "tile.release" in printed


def test_tile_buffer_set_annotation_round_trips_through_printer():
    @pl.function
    def select(index: pl.Scalar[pl.INDEX]):
        buffers = pl.create_tile_buffers(3, [16, 128], pl.FP32, pl.Mem.Right)
        _slot = buffers[index]

    printed = ir.python_print(select, format=False)
    reparsed = pl.parse(printed)

    assert ir.python_print(reparsed, format=False) == printed


def test_runtime_wrapper_supports_len_and_dynamic_getitem():
    buffers = pl.create_tile_buffers(2, [16, 128], pl.FP32, pl.Mem.Acc)
    index = ir.Var("index", ir.ScalarType(pl.INDEX), ir.Span.unknown())

    slot = buffers[index]

    assert len(buffers) == 2
    assert isinstance(slot, pl.Tile)
    slot_expr = slot.unwrap()
    assert isinstance(slot_expr, ir.Call)
    assert slot_expr.op.name == "tile.buffer_slot"


def test_destination_form_parses_selected_slot_as_out_operand():
    @pl.function
    def load_into_slot(x: pl.Tensor[[16, 128], pl.FP32], index: pl.Scalar[pl.INDEX]):
        buffers = pl.create_tile_buffers(2, [16, 128], pl.FP32, pl.Mem.Vec)
        slot = buffers[index % 2]
        _value = pl.load(x, [0, 0], [16, 128], out=slot)

    printed = load_into_slot.as_python()
    assert "pl.tile.load_into" in printed
