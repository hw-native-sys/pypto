# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the AccCompactValid property verifier.

``mad`` takes M from the L0A operand's *valid* rows and lays the product out in
L0C with an N-fractal stride of ``ceil(M/16)*16`` (pto-isa ``TMatmul.hpp``).
Every L0C reader instead derives that stride from the tile's compile-time
physical ``Rows`` *unless* the tile is compact, in which case it recomputes
``ceil(validRow/16)*16`` (``tstore_common.hpp``). An accumulator that stays
non-compact is therefore read back at a pitch it was never written at, and every
N-fractal above the first is scrambled -- issue #2470 through ``tile.store``,
issue #2510 through the Cube->Vector ``tile.tpush_to_aiv``.

The check sits on ``tile.matmul_acc``, where both halves of the comparison are in
hand: the lhs whose valid rows ``mad`` takes M from, and the accumulator buffer
the result aliases in place. A reader cannot make that comparison -- a
``tile.store`` cannot tell an accumulator ``mad`` wrote from an Acc tile some
``tile.load`` filled at the physical pitch.

The second rule guards the same pitch through aliases: ``tile.set_validshape`` is
metadata-only and keeps ``compact``, so re-declaring the valid rows across a
fractal boundary changes the number every compact reader derives its pitch from
while the bytes stay packed where ``mad`` put them.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import passes

ROWS = 64
COLS = 128
K = 256
VALID_ROWS = 16


def _verify(prog):
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AccCompactValid)
    return passes.PropertyVerifierRegistry.verify(props, prog)


def _matmul_acc_program(acc_rows, lhs_valid_rows, acc_compact):
    """One AIC function accumulating *lhs_valid_rows* rows into an [acc_rows, COLS] Acc."""
    span = ir.Span.unknown()
    rows = ir.ConstInt(acc_rows, pl.INDEX, span)
    cols = ir.ConstInt(COLS, pl.INDEX, span)
    k_dim = ir.ConstInt(K, pl.INDEX, span)
    valid_rows = ir.ConstInt(lhs_valid_rows, pl.INDEX, span)

    acc_type = ir.TileType(
        [rows, cols],
        pl.INT32,
        None,
        ir.TileView(valid_shape=[valid_rows, cols], compact=acc_compact),
        ir.MemorySpace.Acc,
    )
    lhs_type = ir.TileType(
        [rows, k_dim],
        pl.INT8,
        None,
        ir.TileView(valid_shape=[valid_rows, k_dim], compact=ir.CompactMode.normal),
        ir.MemorySpace.Left,
    )
    rhs_type = ir.TileType([k_dim, cols], pl.INT8, None, None, ir.MemorySpace.Right)

    acc = ir.Var("acc", acc_type, span)
    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)
    result = ir.Var("result", acc_type, span)
    call = ir.Call(ir.Op("tile.matmul_acc"), [acc, lhs, rhs], acc_type, span)
    func = ir.Function(
        "acc_chain",
        [(acc, ir.ParamDirection.InOut), (lhs, ir.ParamDirection.In), (rhs, ir.ParamDirection.In)],
        [],
        ir.SeqStmts([ir.AssignStmt(result, call, span)], span),
        span,
        ir.FunctionType.AIC,
    )
    return ir.Program([func], "acc_chain_program", span)


def _vec_tile_program(compact):
    """One AIV function holding a Vec tile that claims a compact mode."""
    span = ir.Span.unknown()
    rows = ir.ConstInt(ROWS, pl.INDEX, span)
    cols = ir.ConstInt(COLS, pl.INDEX, span)
    tile_type = ir.TileType(
        [rows, cols],
        pl.INT32,
        None,
        ir.TileView(valid_shape=[ir.ConstInt(VALID_ROWS, pl.INDEX, span), cols], compact=compact),
        ir.MemorySpace.Vec,
    )
    popped = ir.Var("popped", tile_type, span)
    pop = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, tile_type, span)
    func = ir.Function(
        "vec_consumer",
        [],
        [],
        ir.SeqStmts([ir.AssignStmt(popped, pop, span)], span),
        span,
        ir.FunctionType.AIV,
    )
    return ir.Program([func], "vec_consumer_program", span)


def test_row_narrowed_non_compact_accumulator_is_rejected():
    """The chain behind #2470 / #2510: mad writes at pitch 16, the buffer says 64."""
    diags = _verify(_matmul_acc_program(ROWS, VALID_ROWS, ir.CompactMode.null))
    assert len(diags) == 1
    assert diags[0].rule_name == "AccCompactValid"
    assert "tile.matmul_acc" in diags[0].message
    assert "compact" in diags[0].message


def test_row_narrowed_compact_accumulator_is_accepted():
    """Compact is exactly the flag that makes a reader recompute mad's pitch."""
    assert _verify(_matmul_acc_program(ROWS, VALID_ROWS, ir.CompactMode.normal)) == []


def test_full_height_accumulator_needs_no_compact():
    """With valid rows == physical rows the two pitches coincide."""
    assert _verify(_matmul_acc_program(ROWS, ROWS, ir.CompactMode.null)) == []


def test_single_fractal_block_accumulator_needs_no_compact():
    """``ceil(1/16)*16 == 16``: a [16, N] accumulator packs to its own box.

    This is the gemv shape -- one valid row in a single-fractal-block box -- where
    the compact and non-compact readings are the same pitch, so demanding the flag
    would reject legal IR.
    """
    assert _verify(_matmul_acc_program(16, 1, ir.CompactMode.null)) == []


def _set_validshape_program(src_valid_rows, new_valid_rows, compact=ir.CompactMode.normal):
    """One AIC function that re-declares a compact accumulator's valid rows."""
    span = ir.Span.unknown()
    rows = ir.ConstInt(ROWS, pl.INDEX, span)
    cols = ir.ConstInt(COLS, pl.INDEX, span)
    src_valid = ir.ConstInt(src_valid_rows, pl.INDEX, span)
    new_valid = ir.ConstInt(new_valid_rows, pl.INDEX, span)

    src_type = ir.TileType(
        [rows, cols],
        pl.INT32,
        None,
        ir.TileView(valid_shape=[src_valid, cols], compact=compact),
        ir.MemorySpace.Acc,
    )
    dst_type = ir.TileType(
        [rows, cols],
        pl.INT32,
        None,
        ir.TileView(valid_shape=[new_valid, cols], compact=compact),
        ir.MemorySpace.Acc,
    )
    src = ir.Var("acc", src_type, span)
    dst = ir.Var("acc_narrowed", dst_type, span)
    call = ir.Call(ir.Op("tile.set_validshape"), [src, new_valid, cols], dst_type, span)
    func = ir.Function(
        "acc_alias",
        [(src, ir.ParamDirection.In)],
        [],
        ir.SeqStmts([ir.AssignStmt(dst, call, span)], span),
        span,
        ir.FunctionType.AIC,
    )
    return ir.Program([func], "acc_alias_program", span)


def test_re_narrowing_a_compact_accumulator_across_a_fractal_boundary_is_rejected():
    """``mad`` packed the bytes at ceil(17/16)*16 = 32; the alias would read them at 16.

    ``tile.set_validshape`` is metadata-only and keeps ``compact``, so changing the
    valid rows changes the number every compact reader derives its pitch from
    without repacking anything.
    """
    diags = _verify(_set_validshape_program(17, 16))
    assert len(diags) == 1
    assert diags[0].rule_name == "AccCompactValid"
    assert "re-narrows a compact accumulator" in diags[0].message


def test_re_narrowing_within_one_fractal_block_is_accepted():
    """17 -> 20 valid rows both pack to 32, so no reader changes its stride."""
    assert _verify(_set_validshape_program(17, 20)) == []


def test_narrowing_a_fresh_full_box_accumulator_is_accepted():
    """The AutoTileMatmulL0 seed: tile.create(compact=True) narrowed straight away.

    A buffer nothing has written yet re-interprets nothing, so declaring its valid
    region is not an alias re-pitch.
    """
    assert _verify(_set_validshape_program(ROWS, VALID_ROWS)) == []


def test_re_narrowing_a_non_compact_accumulator_is_accepted():
    """Without the flag every reader uses the physical rows, whatever the extent."""
    assert _verify(_set_validshape_program(17, 16, compact=ir.CompactMode.null)) == []


def test_compact_outside_a_fractal_space_is_rejected():
    """A Vec tile has no fractal pitch, so it must not claim a compact one."""
    diags = _verify(_vec_tile_program(ir.CompactMode.normal))
    assert len(diags) == 1
    assert diags[0].rule_name == "AccCompactValid"
    assert "Vec" in diags[0].message


def test_plain_vec_tile_is_accepted():
    """The same tile without the mode is the ordinary C2V pop result."""
    assert _verify(_vec_tile_program(ir.CompactMode.null)) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
