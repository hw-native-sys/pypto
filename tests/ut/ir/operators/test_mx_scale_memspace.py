# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend950 / a5sim unit tests for MX LeftScale/RightScale data-path IR.

Covers memory-space layouts, mx_layout load defaults, and reshape inheritance
scoped to MX scale fractal=32 tiles — without requiring tquant/matmul_mx.
"""

from __future__ import annotations

import pytest
from pypto import DataType, backend, ir
from pypto.backend import BackendType
from pypto.ir.op import tile_ops as tile


@pytest.fixture(autouse=True)
def _ascend950():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        yield
    finally:
        backend.reset_for_testing()


def _const_shape(rows: int, cols: int):
    span = ir.Span.unknown()
    return [ir.ConstInt(rows, DataType.INDEX, span), ir.ConstInt(cols, DataType.INDEX, span)]


def _tile_var(shape, dtype, *, memory=None, view=None, name="t"):
    tt = ir.TileType(shape, dtype, None, view, memory)
    return ir.Var(name, tt, ir.Span.unknown())


class TestMxScaleMemSpacesA5:
    def test_leftscale_move_layout_row_major_fractal32(self):
        src = _tile_var(
            _const_shape(16, 8),
            DataType.UINT8,
            memory=ir.MemorySpace.Mat,
            view=ir.TileView(
                blayout=ir.TileLayout.row_major,
                slayout=ir.TileLayout.row_major,
                fractal=32,
            ),
        )
        call = tile.move(src, target_memory=ir.MemorySpace.LeftScale)
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.LeftScale
        assert out.dtype == DataType.FP8E8M0  # ui8 promotes for loc=scaling
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.row_major
        assert view.slayout == ir.TileLayout.row_major
        assert view.fractal == 32

    def test_rightscale_move_layout_col_major_fractal32(self):
        src = _tile_var(
            _const_shape(8, 16),
            DataType.UINT8,
            memory=ir.MemorySpace.Mat,
            view=ir.TileView(
                blayout=ir.TileLayout.col_major,
                slayout=ir.TileLayout.col_major,
                fractal=32,
            ),
        )
        call = tile.move(src, target_memory=ir.MemorySpace.RightScale)
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.RightScale
        assert out.dtype == DataType.FP8E8M0
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.col_major
        assert view.fractal == 32

    def test_mx_layout_load_defaults_to_mat(self):
        tensor = ir.Var(
            "s",
            ir.TensorType(_const_shape(16, 8), DataType.FP8E8M0),
            ir.Span.unknown(),
        )
        call = tile.load(tensor, [0, 0], [16, 8], mx_layout="mx_a_zz")
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.Mat
        view = out.get_effective_tile_view()
        assert view.fractal == 32
        assert view.blayout == ir.TileLayout.row_major
        assert view.slayout == ir.TileLayout.row_major

    def test_mx_b_layout_load_col_major(self):
        tensor = ir.Var(
            "w_s",
            ir.TensorType(_const_shape(8, 16), DataType.FP8E8M0),
            ir.Span.unknown(),
        )
        call = tile.load(tensor, [0, 0], [8, 16], mx_layout="mx_b_nn")
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.Mat
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.col_major
        assert view.fractal == 32

    def test_reshape_inherits_mx_scale_box_only(self):
        mx = _tile_var(
            _const_shape(1, 128),
            DataType.UINT8,
            memory=ir.MemorySpace.Mat,
            view=ir.TileView(
                blayout=ir.TileLayout.row_major,
                slayout=ir.TileLayout.row_major,
                fractal=32,
            ),
        )
        reshaped = tile.reshape(mx, [16, 8])
        assert isinstance(reshaped.type, ir.TileType)
        mx_view = reshaped.type.get_effective_tile_view()
        assert mx_view.fractal == 32
        assert mx_view.blayout == ir.TileLayout.row_major
        assert mx_view.slayout == ir.TileLayout.row_major

        # Non-MX column tile: reshape [16,1] → [1,16] must re-infer row_major.
        col = _tile_var(
            _const_shape(16, 1),
            DataType.FP32,
            memory=ir.MemorySpace.Vec,
            view=ir.TileView(blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.none_box),
        )
        row = tile.reshape(col, [1, 16])
        assert isinstance(row.type, ir.TileType)
        row_view = row.type.get_effective_tile_view()
        assert row_view.blayout == ir.TileLayout.row_major
        assert row_view.slayout == ir.TileLayout.none_box

    def test_move_target_shape_leftscale(self):
        flat = _tile_var(
            _const_shape(1, 128),
            DataType.UINT8,
            memory=ir.MemorySpace.Mat,
            view=ir.TileView(blayout=ir.TileLayout.row_major, slayout=ir.TileLayout.none_box),
        )
        call = tile.move(
            flat,
            target_memory=ir.MemorySpace.LeftScale,
            target_shape=[16, 8],
        )
        out = call.type
        assert isinstance(out, ir.TileType)
        assert isinstance(out.shape[0], ir.ConstInt) and out.shape[0].value == 16
        assert isinstance(out.shape[1], ir.ConstInt) and out.shape[1].value == 8
        assert out.memory_space == ir.MemorySpace.LeftScale
        assert out.dtype == DataType.FP8E8M0
        view = out.get_effective_tile_view()
        assert view.fractal == 32
        assert view.blayout == ir.TileLayout.row_major
