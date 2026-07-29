# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend950 / a5sim unit tests for MX LeftScale/RightScale data-path IR.

Covers memory-space layouts and mx_layout load defaults for host-prequant MX.
"""

from __future__ import annotations

import pytest
from pypto import DataType, backend, ir, passes
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
        call = tile.load(tensor, [0, 0], [16, 8], target_memory=ir.MemorySpace.Mat, mx_layout="mx_a_zz")
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
        call = tile.load(tensor, [0, 0], [8, 16], target_memory=ir.MemorySpace.Mat, mx_layout="mx_b_nn")
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.Mat
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.col_major
        assert view.fractal == 32

    def test_mx_layout_rejects_vec_target(self):
        tensor = ir.Var(
            "s",
            ir.TensorType(_const_shape(16, 8), DataType.FP8E8M0),
            ir.Span.unknown(),
        )
        with pytest.raises(Exception, match="Mat|Vec"):
            tile.load(tensor, [0, 0], [16, 8], target_memory=ir.MemorySpace.Vec, mx_layout="mx_a_zz")

    def test_infer_preserves_mx_fractal_when_target_memory_kwarg_absent(self):
        """Infer must keep fractal=32 when mx_layout load lacks target_memory kwarg.

        Simulates pre-stamp IR: type already Mat+fractal=32, kwargs only mx_layout.
        Consumer move→LeftScale demands Mat; Infer must not rebuild via NZ Mat view.
        """
        span = ir.Span.unknown()
        shape = _const_shape(16, 2)
        tensor = ir.Var("a_s", ir.TensorType(shape, DataType.FP8E8M0), span)
        out = ir.Var("out", ir.TensorType(shape, DataType.FP8E8M0), span)

        zero = ir.ConstInt(0, DataType.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple(shape, span)

        # Canonical create stamps Mat; strip it to exercise Infer's mx_layout path.
        stamped = ir.create_op_call(
            "tile.load",
            [tensor, offsets, shapes, shapes],
            {"mx_layout": "mx_a_zz"},
            span,
        )
        stripped_kwargs = {k: v for k, v in stamped.kwargs.items() if k != "target_memory"}
        assert "target_memory" not in stripped_kwargs
        load_call = ir.Call(stamped.op, list(stamped.args), stripped_kwargs, stamped.type, span)

        tas = ir.Var("tas", load_call.type, span)
        move_call = tile.move(tas, target_memory=ir.MemorySpace.LeftScale)
        las = ir.Var("las", move_call.type, span)

        # Separate Vec load/store so the function has a valid sink; mx load is
        # kept live via the LeftScale move.
        vec_load = tile.load(tensor, [0, 0], [16, 2], target_memory=ir.MemorySpace.Vec)
        vec_tile = ir.Var("vec_tile", vec_load.type, span)
        store_call = tile.store(vec_tile, [0, 0], out)
        result = ir.Var("result", out.type, span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(tas, load_call, span),
                ir.AssignStmt(las, move_call, span),
                ir.AssignStmt(vec_tile, vec_load, span),
                ir.AssignStmt(result, store_call, span),
                ir.ReturnStmt([result], span),
            ],
            span,
        )
        func = ir.Function(
            "mx_infer_preserve",
            [(tensor, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
            [out.type],
            body,
            span,
            ir.FunctionType.InCore,
        )
        prog = ir.Program([func], "mx_infer_preserve", span)
        after = passes.infer_tile_memory_space()(prog)

        load_calls = []

        def _collect(stmt):
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                if stmt.value.op.name == "tile.load":
                    load_calls.append(stmt.value)
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    _collect(s)

        for fn in after.functions.values():
            _collect(fn.body)

        mx_loads = [c for c in load_calls if "mx_layout" in c.kwargs]
        assert len(mx_loads) == 1
        mx_load = mx_loads[0]
        assert mx_load.kwargs.get("target_memory") == ir.MemorySpace.Mat
        view = mx_load.type.get_effective_tile_view()
        assert view.fractal == 32
        assert view.blayout == ir.TileLayout.row_major
        assert view.slayout == ir.TileLayout.row_major

    def test_reshape_reinfers_layout_from_new_shape(self):
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
