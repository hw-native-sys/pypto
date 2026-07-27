# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for tile producers that write directly into selected buffer slots."""

import pytest
from pypto import DataType, ir
from pypto.ir.op import tile


def _slot(shape, dtype, memory_space):
    buffers = tile.create_buffer_set(shape, dtype, memory_space, 2)
    return tile.buffer_slot(buffers, 0)


def _matmul_operands():
    lhs = tile.create([16, 32], DataType.BF16, ir.MemorySpace.Left)
    rhs = tile.create([32, 128], DataType.BF16, ir.MemorySpace.Right)
    return lhs, rhs


def test_matmul_out_binds_result_to_selected_acc_slot():
    lhs, rhs = _matmul_operands()
    out = _slot([16, 128], DataType.FP32, ir.MemorySpace.Acc)

    result = tile.matmul(lhs, rhs, out=out)

    assert result.op.name == "tile.matmul_into"
    assert result.args[-1] is out
    assert result.type is out.type


def test_legacy_matmul_keeps_original_op():
    lhs, rhs = _matmul_operands()

    result = tile.matmul(lhs, rhs)

    assert result.op.name == "tile.matmul"


@pytest.mark.parametrize(
    ("out", "message"),
    [
        (_slot([8, 128], DataType.FP32, ir.MemorySpace.Acc), "shape"),
        (_slot([16, 128], DataType.INT32, ir.MemorySpace.Acc), "dtype"),
        (_slot([16, 128], DataType.FP32, ir.MemorySpace.Vec), "memory space"),
    ],
)
def test_matmul_out_rejects_incompatible_destination(out, message):
    lhs, rhs = _matmul_operands()

    with pytest.raises(ValueError, match=message):
        tile.matmul(lhs, rhs, out=out)


@pytest.mark.parametrize(
    ("view", "message"),
    [
        (ir.TileView(valid_shape=[8, 128]), "valid_shape"),
        (
            ir.TileView(blayout=ir.TileLayout.row_major, slayout=ir.TileLayout.row_major),
            "layout",
        ),
    ],
)
def test_matmul_out_rejects_incompatible_view(view, message):
    lhs, rhs = _matmul_operands()
    out = ir.Var(
        "out",
        ir.TileType([16, 128], DataType.FP32, None, view, ir.MemorySpace.Acc),
        ir.Span.unknown(),
    )

    with pytest.raises(ValueError, match=message):
        tile.matmul(lhs, rhs, out=out)


def test_load_extract_move_and_matmul_acc_have_destination_forms():
    span = ir.Span.unknown()
    tensor = ir.Var("tensor", ir.TensorType([16, 128], DataType.FP32), span)
    vec_out = _slot([16, 128], DataType.FP32, ir.MemorySpace.Vec)
    loaded = tile.load(tensor, [0, 0], [16, 128], out=vec_out)
    assert loaded.op.name == "tile.load_into"
    assert loaded.type is vec_out.type

    mat_src = tile.create([16, 128], DataType.BF16, ir.MemorySpace.Mat)
    right_out = _slot([16, 32], DataType.BF16, ir.MemorySpace.Right)
    extracted = tile.extract(mat_src, 0, 0, [16, 32], target_memory=ir.MemorySpace.Right, out=right_out)
    assert extracted.op.name == "tile.extract_into"
    assert extracted.type is right_out.type

    moved = tile.move(
        mat_src,
        ir.MemorySpace.Right,
        out=_slot([16, 128], DataType.BF16, ir.MemorySpace.Right),
    )
    assert moved.op.name == "tile.move_into"

    lhs, rhs = _matmul_operands()
    acc_out = _slot([16, 128], DataType.FP32, ir.MemorySpace.Acc)
    accumulated = tile.matmul_acc(acc_out, lhs, rhs, out=acc_out)
    assert accumulated.op.name == "tile.matmul_acc_into"
    assert accumulated.type is acc_out.type
