# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for source-layout normalization at automatic V2C boundaries."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile_ops as T


def _make_program(blayout, slayout, destination, *, sliced=False, valid_shape=None, shape=(32, 64)):
    span = ir.Span.unknown()
    view = ir.TileView(valid_shape=valid_shape or list(shape), blayout=blayout, slayout=slayout)
    source = ir.Var("source", ir.TileType(list(shape), DataType.BF16, None, view, pl.Mem.Vec), span)
    right = ir.Var("right", ir.TileType([shape[1], 16], DataType.BF16, None, None, pl.Mem.Right), span)
    if destination == pl.Mem.Right:
        right = ir.Var("left", ir.TileType([16, 32], DataType.BF16, None, None, pl.Mem.Left), span)
    out_shape = [16, 64] if destination == pl.Mem.Right else [shape[0], 16]
    out = ir.Var("out", ir.TensorType(out_shape, DataType.FP32), span)
    stmts = []

    def bind(name, call):
        var = ir.Var(name, call.type, span)
        stmts.append(ir.AssignStmt(var, call, span))
        return var

    value = bind("value", T.add(source, source))
    if sliced:
        value = bind("window", T.slice(value, [16, 64], [8, 0]))
    boundary = bind("boundary", T.move(value, target_memory=destination))
    if destination == pl.Mem.Right:
        acc = bind("acc", T.matmul(right, boundary))
    else:
        left = bind("left", T.move(boundary, target_memory=pl.Mem.Left))
        acc = bind("acc", T.matmul(left, right))
    result = bind("result", T.move(acc, target_memory=pl.Mem.Vec))
    stored = bind("stored", T.store(result, [0, 0], out))
    stmts.append(ir.ReturnStmt([stored], span))
    function = ir.Function(
        "kernel",
        [source, right, (out, ir.ParamDirection.Out)],
        [out.type],
        ir.SeqStmts(stmts, span),
        span,
        pl.FunctionType.InCore,
    )
    return ir.Program([function], "Before", span)


def _calls(function):
    calls = []

    class Collector(ir.IRVisitor):
        def visit_call(self, op):
            calls.append(op)
            super().visit_call(op)

    Collector().visit_stmt(function.body)
    return calls


def _layout(expr):
    view = expr.type.get_effective_tile_view()
    return view.blayout, view.slayout


def _shape(expr):
    return [dim.value for dim in expr.type.shape]


@pytest.mark.parametrize("destination", [pl.Mem.Mat, pl.Mem.Left, pl.Mem.Right])
@pytest.mark.parametrize("layout", ["nd", "dn", "nz"])
def test_v2c_push_has_nz_payload_and_original_shape(layout, destination):
    """A DN producer must never feed a converting move directly; NZ needs no copy."""
    row, col, none = pl.TileLayout.row_major, pl.TileLayout.col_major, pl.TileLayout.none_box
    blayout, slayout = {"nd": (row, none), "dn": (col, none), "nz": (col, row)}[layout]
    before = _make_program(blayout, slayout, destination)
    after = passes.expand_mixed_kernel()(before)
    aiv_calls = _calls(after.get_function("kernel_aiv"))
    pushes = [call for call in aiv_calls if call.op.name == ir.get_op("tile.tpush_to_aic").name]
    assert len(pushes) == 1
    pushed = pushes[0].args[0]
    assert _shape(pushed) == [32, 64]
    assert _layout(pushed) == (col, row)
    moves = [call for call in aiv_calls if call.op.name == ir.get_op("tile.move").name]
    transposes = [call for call in aiv_calls if call.op.name == ir.get_op("tile.transpose_view").name]
    assert len(moves) == (0 if layout == "nz" else 1)
    assert len(transposes) == (2 if layout == "dn" else 0)
    if layout != "nz":
        assert _layout(moves[0].args[0]) == (row, none)
        assert _layout(moves[0]) == ((row, col) if layout == "dn" else (col, row))
        assert _shape(moves[0]) == ([64, 32] if layout == "dn" else [32, 64])
    pops = [
        call
        for call in _calls(after.get_function("kernel_aic"))
        if call.op.name == ir.get_op("tile.tpop_from_aiv").name
    ]
    assert len(pops) == 1
    assert _shape(pops[0]) == [32, 64]
    assert _layout(pops[0]) == (col, row)


@pytest.mark.parametrize(
    ("blayout", "slayout"),
    [(pl.TileLayout.row_major, pl.TileLayout.col_major), (pl.TileLayout.row_major, pl.TileLayout.row_major)],
)
def test_v2c_rejects_unsupported_source_layout(blayout, slayout):
    """Unsupported ZN/other payloads fail instead of being relabelled as NZ."""
    before = _make_program(blayout, slayout, pl.Mem.Mat)
    with pytest.raises(ValueError, match="cross-core.*operand must reach the boundary"):
        passes.expand_mixed_kernel()(before)


def test_v2c_accepts_explicit_row_major_column_as_nd():
    """An [M, 1] tile labelled row_major is ND data, not a transposed alias."""
    row, col, none = pl.TileLayout.row_major, pl.TileLayout.col_major, pl.TileLayout.none_box
    before = _make_program(row, none, pl.Mem.Mat, shape=(32, 1))
    after = passes.expand_mixed_kernel()(before)
    aiv_calls = _calls(after.get_function("kernel_aiv"))
    moves = [call for call in aiv_calls if call.op.name == ir.get_op("tile.move").name]
    assert len(moves) == 1
    assert _layout(moves[0].args[0]) == (row, none)
    assert _layout(moves[0]) == (col, row)
    pushes = [call for call in aiv_calls if call.op.name == ir.get_op("tile.tpush_to_aic").name]
    assert len(pushes) == 1
    assert _shape(pushes[0].args[0]) == [32, 1]
    assert _layout(pushes[0].args[0]) == (col, row)


def test_v2c_rejects_dn_slice_with_boundary_diagnostic():
    before = _make_program(pl.TileLayout.col_major, pl.TileLayout.none_box, pl.Mem.Mat, sliced=True)
    with pytest.raises(ValueError, match="cross-core.*operand.*tile.slice window"):
        passes.expand_mixed_kernel()(before)


@pytest.mark.parametrize("k", [128, 384])
def test_transposed_vec_matmul_compiles_with_and_without_l0_tiling(k, tmp_path):
    """Exercise the real lowering pipeline, including AutoTileMatmulL0's Mat move."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[k, 128], pl.BF16],
            b: pl.Tensor[[k, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ):
            loaded = pl.load(a, [0, 0], [k, 128])
            value = pl.maximum(loaded, 0.0)
            lhs = pl.tile.transpose_view(value)
            rhs = pl.load(b, [0, 0], [k, 128], target_memory=pl.Mem.Mat)
            acc = pl.matmul(lhs, rhs)
            pl.store(acc, [0, 0], out)

    expanded = []

    def capture(pass_obj, program):
        if pass_obj.get_name() == "ExpandMixedKernel":
            expanded.append(program)

    with passes.PassContext([passes.CallbackInstrument(after_pass=capture, name="CaptureV2C")]):
        ir.compile(Before, output_dir=str(tmp_path), platform="a5", dump_passes=False, skip_ptoas=True)
    assert len(expanded) == 1
    calls = _calls(expanded[0].get_function("kernel_aiv"))
    moves = [call for call in calls if call.op.name == ir.get_op("tile.move").name]
    assert len(moves) == 1
    assert _layout(moves[0].args[0]) == (pl.TileLayout.row_major, pl.TileLayout.none_box)
    assert _layout(moves[0]) == (pl.TileLayout.row_major, pl.TileLayout.col_major)
    pushes = [call for call in calls if call.op.name == ir.get_op("tile.tpush_to_aic").name]
    assert len(pushes) == 1
    assert _shape(pushes[0].args[0]) == [128, k]
    assert _layout(pushes[0].args[0]) == (pl.TileLayout.col_major, pl.TileLayout.row_major)


def test_dn_normalization_preserves_partial_valid_shape():
    before = _make_program(pl.TileLayout.col_major, pl.TileLayout.none_box, pl.Mem.Mat, valid_shape=[24, 48])
    after = passes.expand_mixed_kernel()(before)
    pushes = [
        call
        for call in _calls(after.get_function("kernel_aiv"))
        if call.op.name == ir.get_op("tile.tpush_to_aic").name
    ]
    assert len(pushes) == 1
    view = pushes[0].args[0].type.get_effective_tile_view()
    assert [dim.value for dim in view.valid_shape] == [24, 48]
    assert _shape(pushes[0].args[0]) == [32, 64]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
