# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the LowerAutoVectorSplit pass (RFC #1300 convergence).

The pass is the live auto-split lowering path: it converts an AUTO ``pl.split``
mixed InCore function into the explicit ``split_aiv`` form *before*
ExpandMixedKernel. It inserts ``tile.aiv_shard`` at C->V boundaries (and
``tile.aic_gather`` at V->C boundaries), halves only the VECTOR sub-region
(affinity-gated reuse of the shared ``split_axis`` halving machinery), injects
``get_subblock_idx``, and stamps ``split`` + ``split_aiv``. CUBE-affine operands
stay full (the affinity gate).

These tests hand-build a minimal mixed InCore function at the
post-InferTileMemorySpace level (memory spaces already assigned) and run the pass
in isolation with verification disabled.

The per-op vector halving tests (load / slice / reshape / store offset /
singleton / loop tracking / reduce-on-split-axis throw) were migrated here from
``test_split_vector_kernel.py``: those facts are produced by the shared
``split_axis::ProcessStmts`` machinery, which SplitVectorKernel's deleted per-op
halving driver and this pass both call. The new pass routes each VECTOR-affine
leaf statement through that same machinery, so the halving is identical (Stage 1
proved byte-identity); only the entry point changed.
"""

import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile_ops as T

MS = ir.MemorySpace
FP32 = DataType.FP32
_IN = ir.ParamDirection.In
_OUT = ir.ParamDirection.Out


def _tile(shape, view=None, mem=None):
    return ir.TileType(shape, FP32, None, view, mem)


def _tensor(shape):
    return ir.TensorType(shape, FP32)


def _lower(program):
    with passes.PassContext([]):
        return passes.lower_auto_vector_split()(program)


def _incore_program(params, stmts, return_types, *, mode=ir.SplitMode.UP_DOWN, name="split_auto"):
    """Build a single-function mixed InCore Program carrying a function-level split mode.

    ``params`` is a list of ``(Var, ParamDirection)`` pairs; ``stmts`` is the
    flat body (including the terminating ``ReturnStmt``). The function is tagged
    ``FunctionType.InCore`` with ``attrs={"split": mode}`` — exactly what reaches
    LowerAutoVectorSplit in the real pipeline after InferTileMemorySpace.

    A leading cube->vector boundary (``move(cube_seed Mat -> Vec)``) is injected so
    the function is genuinely MIXED: LowerAutoVectorSplit only lowers mixed
    cube<->vector functions — a pure-vector ``pl.split`` function has no boundary
    to converge and is (correctly) left untouched. The boundary result is unused;
    the op-under-test in ``stmts`` is the vector sub-region that gets halved.
    """
    span = ir.Span.unknown()
    cube_seed = ir.Var("cube_seed", _tile([128, 128], mem=MS.Mat), span)
    seed_move = T.move(cube_seed, MS.Vec, span=span)
    assert isinstance(seed_move.type, ir.TileType)
    seed_vec = ir.Var("seed_vec", _tile(seed_move.type.shape, seed_move.type.tile_view, MS.Vec), span)
    func = ir.Function(
        name,
        [(cube_seed, _IN), *params],
        return_types,
        ir.SeqStmts([ir.AssignStmt(seed_vec, seed_move, span), *stmts], span),
        span,
        ir.FunctionType.InCore,
        attrs={"split": mode},
    )
    return ir.Program([func], name, span)


def _func_attrs(program):
    return next(iter(program.functions.values())).attrs


def _build_c2v_mixed_program():
    """Mixed InCore UP_DOWN: cube tile (Mat) --move(C->V)--> Vec, vector add, store.

    The ``move(qk Mat -> Vec)`` is a CUBE_TO_VECTOR boundary; the vector add and
    store form the vector sub-region that must be halved on dim0.
    """
    span = ir.Span.unknown()
    qk = ir.Var("qk", _tile([128, 128], mem=MS.Mat), span)
    out_0 = ir.Var("out_0", ir.TensorType([128, 128], FP32), span)

    move = T.move(qk, MS.Vec, span=span)
    assert isinstance(move.type, ir.TileType)
    popped = ir.Var("popped", _tile(move.type.shape, move.type.tile_view, MS.Vec), span)
    add = T.add(popped, popped, span)
    assert isinstance(add.type, ir.TileType)
    y = ir.Var("y", _tile(add.type.shape, add.type.tile_view, MS.Vec), span)
    store = T.store(y, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(popped, move, span),
            ir.AssignStmt(y, add, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    func = ir.Function(
        "split_auto",
        [(qk, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.UP_DOWN},
    )
    return ir.Program([func], "test_c2v_mixed", span)


def test_c2v_boundary_becomes_aiv_shard_and_vector_region_is_halved():
    program = _build_c2v_mixed_program()
    after = _lower(program)
    text = ir.python_print(after)

    # The C->V tile.move boundary is rewritten to tile.aiv_shard(split=1).
    assert "tile.aiv_shard" in text or "aiv_shard" in text
    assert "tile.move" not in text  # the only move was the boundary; it is gone.

    # The vector sub-region (add + store result) is halved on dim0: 128 -> 64.
    # The shard result and the add result carry the half extent.
    assert "[64, 128]" in text
    # The cube operand (qk parameter) stays FULL.
    assert "[128, 128]" in text

    # get_subblock_idx injected and split_aiv stamped.
    assert "get_subblock_idx" in text
    func = next(iter(after.functions.values()))
    assert func.attrs.get("split_aiv") is True
    assert func.attrs.get("split") is not None


def test_store_offset_is_localized_per_subblock():
    """The vector store offset is localized: [0, 0] -> [0 + subblock_idx * 64, 0]."""
    program = _build_c2v_mixed_program()
    after = _lower(program)
    text = ir.python_print(after)
    # AdjustOffsets adds subblock_idx * half on the split axis (dim0).
    assert "subblock_idx" in text
    assert "* 64" in text


# ---------------------------------------------------------------------------
# Vector sub-region per-op halving (migrated from test_split_vector_kernel.py).
#
# Each builds a mixed InCore function whose vector sub-region contains the op
# under test and asserts the new pass halves it via the shared split_axis
# machinery — the same facts the deleted SplitVectorKernel halving driver
# asserted, now exercised through LowerAutoVectorSplit.
# ---------------------------------------------------------------------------


def test_vector_load_halved_and_offset_localized():
    """UP_DOWN: a VECTOR tile.load halves its result + shape/valid args (128 -> 64)
    and localizes its split-dim offset per subblock."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    prev = ir.Var("prev", load.type, span)
    store = T.store(prev, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(data, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(prev, load, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    after = _lower(program)
    text = ir.python_print(after)
    # Result + shape/valid args halved on dim0; offset localized per subblock.
    assert "prev: pl.Tile[[64, 128]" in text
    assert "[64, 128], [64, 128]" in text  # shape + valid_shape args both halved
    assert "[0 + subblock_idx * 64, 0]" in text  # load offset localized
    assert _func_attrs(after).get("split_aiv") is True


def test_vector_load_halved_left_right():
    """LEFT_RIGHT: the load halves on dim1 (128 -> 64) and localizes the col offset."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    prev = ir.Var("prev", load.type, span)
    store = T.store(prev, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(data, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(prev, load, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
        mode=ir.SplitMode.LEFT_RIGHT,
    )
    text = ir.python_print(_lower(program))
    assert "prev: pl.Tile[[128, 64]" in text
    assert "[0, 0 + subblock_idx * 64]" in text  # col offset localized


def test_vector_slice_halves_shape_and_localizes_offset():
    """UP_DOWN: a tile.slice of a full (unsplit) Vec source halves its static shape
    tuple in lockstep with the result (the qk_pv strided sub-slice fix) and
    localizes its zero-base offset per subblock."""
    span = ir.Span.unknown()
    src = ir.Var("src", _tile([128, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    sl = T.slice(src, [128, 128], [0, 0], span=span)
    sub = ir.Var("sub", sl.type, span)
    store = T.store(sub, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(src, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(sub, sl, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    # Result type AND the static shape tuple arg both halve to [64, 128].
    assert "sub: pl.Tile[[64, 128]" in text
    assert "pl.tile.slice(src, [64, 128], [0 + subblock_idx * 64, 0])" in text


def test_vector_slice_nonzero_base_offset_localizes_additively():
    """UP_DOWN: a strided sub-slice at a non-zero base offset localizes additively —
    the original offset is preserved and subblock_idx*half is added on the split
    axis (the exact qk_pv ``oi[16:32]`` pattern)."""
    span = ir.Span.unknown()
    src = ir.Var("src", _tile([256, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    sl = T.slice(src, [128, 128], [16, 0], span=span)
    sub = ir.Var("sub", sl.type, span)
    store = T.store(sub, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(src, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(sub, sl, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    assert "sub: pl.Tile[[64, 128]" in text
    # Base offset 16 preserved, subblock_idx * 64 added on the split axis.
    assert "pl.tile.slice(src, [64, 128], [16 + subblock_idx * 64, 0])" in text


def test_slice_of_split_tracked_source_halves_shape_keeps_offset():
    """LEFT_RIGHT: a tile.slice whose source is already split-tracked (a halved
    load) halves its static shape tuple but leaves its offset unchanged — the
    source is already in lane-local coordinates."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([16, 128]), span)
    out_0 = ir.Var("out_0", _tensor([16, 128]), span)
    load = T.load(data, [0, 0], [16, 128], target_memory=MS.Vec, span=span)
    prev = ir.Var("prev", load.type, span)
    sl = T.slice(prev, [16, 128], [0, 0], span=span)
    sub = ir.Var("sub", sl.type, span)
    store = T.store(sub, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(data, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(prev, load, span),
            ir.AssignStmt(sub, sl, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
        mode=ir.SplitMode.LEFT_RIGHT,
    )
    text = ir.python_print(_lower(program))
    # Source load halved [16,128] -> [16,64] (tracked); the slice halves its
    # shape tuple in lockstep but keeps the [0, 0] lane-local offset.
    assert "prev: pl.Tile[[16, 64]" in text
    assert "pl.tile.slice(prev, [16, 64], [0, 0])" in text


def test_reshape_of_rank1_load_is_sliced_per_subblock():
    """UP_DOWN: a rank-1 load reshaped to [N, 1] is emitted at full width and
    followed by a per-subblock column slice so each lane reads its own row-half
    (the v2-minimal slice fix; rank-1 loads carry no 2D split axis)."""
    span = ir.Span.unknown()
    scale = ir.Var("scale", _tensor([128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 1]), span)
    load = T.load(scale, [0], [128], target_memory=MS.Vec, span=span)
    scale_row = ir.Var("scale_row", load.type, span)
    reshape = T.reshape(scale_row, [128, 1], span=span)
    scale_2d = ir.Var("scale_2d", reshape.type, span)
    store = T.store(scale_2d, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(scale, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(scale_row, load, span),
            ir.AssignStmt(scale_2d, reshape, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    # Rank-1 load bypassed (stays [128]); reshape stays full [128, 1]; a slice to
    # [64, 1] at the per-subblock row offset is appended.
    assert "scale_row: pl.Tile[[128]" in text
    assert "pl.tile.reshape(scale_row, [128, 1])" in text
    assert "pl.tile.slice(scale_2d, [64, 1], [subblock_idx * 64, 0])" in text


def test_reshape_of_already_split_input_halves_shape_arg():
    """UP_DOWN: a reshape whose input is already split halves its shape ARGUMENT
    too ([256, 1] -> [128, 1]), not just the result type, so memory_reuse sizes
    the output from the halved literal."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([16, 16]), span)
    out_0 = ir.Var("out_0", _tensor([256, 1]), span)
    load = T.load(data, [0, 0], [16, 16], target_memory=MS.Vec, span=span)
    prev = ir.Var("prev", load.type, span)
    reshape = T.reshape(prev, [256, 1], span=span)
    flat = ir.Var("flat", reshape.type, span)
    store = T.store(flat, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(data, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(prev, load, span),
            ir.AssignStmt(flat, reshape, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    # Input load halved [16,16] -> [8,16]; reshape result AND shape arg halve.
    assert "prev: pl.Tile[[8, 16]" in text
    assert "flat: pl.Tile[[128, 1]" in text
    assert "pl.tile.reshape(prev, [128, 1])" in text


def test_reshape_migrates_split_axis_row_to_col_and_back():
    """UP_DOWN: a [N,1]<->[1,N] reshape migrates the split axis, not corrupts it (gh#1864).

    The rms_norm column reshape moves the split data (rows) into the column dim and
    back. Each AIV lane keeps its own half, so the reshape targets must halve the
    MIGRATED dim ([1,8], then [8,1]) -- not stay at the stale full width ([1,16])
    which left lane 1 reading garbage and emitting inf. No per-subblock slice is
    needed (the partition is lane-local through the migration)."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([16, 1]), span)
    out_0 = ir.Var("out_0", _tensor([16, 1]), span)
    load = T.load(data, [0, 0], [16, 1], target_memory=MS.Vec, span=span)
    col = ir.Var("col", load.type, span)
    to_row = T.reshape(col, [1, 16], span=span)
    row = ir.Var("row", to_row.type, span)
    inv = T.recip(row, span=span)
    inv_row = ir.Var("inv_row", inv.type, span)
    to_col = T.reshape(inv_row, [16, 1], span=span)
    back = ir.Var("back", to_col.type, span)
    store = T.store(back, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(data, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(col, load, span),
            ir.AssignStmt(row, to_row, span),
            ir.AssignStmt(inv_row, inv, span),
            ir.AssignStmt(back, to_col, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    assert "col: pl.Tile[[8, 1]" in text  # load halved on the row split dim
    assert "pl.tile.reshape(col, [1, 8])" in text  # row -> col migration (was [1, 16])
    assert "pl.tile.reshape(inv_row, [8, 1])" in text  # col -> row migration back
    assert "pl.tile.slice" not in text  # each lane self-contained; no slice needed
    # Store offset localized on the row dim, one half per subblock.
    assert "subblock_idx * 8" in text


def test_reshape_untrackable_split_axis_rejected():
    """A reshape whose split partition can't map to a clean per-dim halving is rejected.

    The dim-0 split of a [6, 4] tile partitions at flat offset 12 (rows 0-2 vs 3-5).
    Reshaping to [3, 8] would place that boundary mid-row, so no result dim can
    carry the halved split cleanly -- the pass rejects rather than miscompile."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([6, 4]), span)
    out_0 = ir.Var("out_0", _tensor([3, 8]), span)
    load = T.load(data, [0, 0], [6, 4], target_memory=MS.Vec, span=span)
    prev = ir.Var("prev", load.type, span)
    reshape = T.reshape(prev, [3, 8], span=span)
    flat = ir.Var("flat", reshape.type, span)
    store = T.store(flat, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(data, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(prev, load, span),
            ir.AssignStmt(flat, reshape, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    with pytest.raises(ValueError, match="moves the split axis"):
        _lower(program)


def test_singleton_broadcast_tile_preserved():
    """UP_DOWN: a [1, 128] broadcast tile is NOT halved on the singleton split dim."""
    span = ir.Span.unknown()
    src = ir.Var("src", _tile([1, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", _tensor([1, 128]), span)
    add = T.add(src, src, span)
    av = ir.Var("av", add.type, span)
    store = T.store(av, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(src, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(av, add, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    assert "av: pl.Tile[[1, 128]" in text  # preserved, not [0, 128] / halved


def test_loop_iter_arg_keeps_split_tracking():
    """UP_DOWN: a loop iter_arg seeded by a halved load keeps split-aware store
    offsets inside the loop body (tile_vars tracking flows through iter_args)."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    accum = ir.Var("accum", load.type, span)

    # for i in range(2): out_0 = store(accum, [0,0], out_0)
    i_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    out_iter = ir.IterArg("out_it", out_0.type, out_0, span)
    body_store = T.store(accum, [0, 0], out_iter, span=span)
    body_store_var = ir.Var("out_it_next", body_store.type, span)
    loop_ret = ir.Var("out_loop", out_0.type, span)
    for_body = ir.SeqStmts(
        [ir.AssignStmt(body_store_var, body_store, span), ir.YieldStmt([body_store_var], span)],
        span,
    )
    for_stmt = ir.ForStmt(
        i_var,
        ir.ConstInt(0, DataType.INDEX, span),
        ir.ConstInt(2, DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
        [out_iter],
        for_body,
        [loop_ret],
        span,
    )
    program = _incore_program(
        [(data, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(accum, load, span),
            for_stmt,
            ir.ReturnStmt([loop_ret], span),
        ],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    # The loaded accumulator is halved; the in-loop store offset is localized
    # using the same tracked half extent.
    assert "accum: pl.Tile[[64, 128]" in text
    assert text.count("[0 + subblock_idx * 64, 0]") >= 2  # load + in-loop store


def test_reduce_on_split_axis_rejected():
    """A reduce that collapses the split axis (dim0 under UP_DOWN) raises ValueError —
    a partial per-lane reduction is a miscompile."""
    span = ir.Span.unknown()
    src = ir.Var("src", _tile([128, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    reduced = T.sum(src, axis=0, keepdim=True, span=span)
    rv = ir.Var("rv", reduced.type, span)
    store = T.store(rv, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(src, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(rv, reduced, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    with pytest.raises(ValueError, match="reduces on the split axis"):
        _lower(program)


def test_vc_boundary_becomes_aic_gather_and_cube_placement_stays_full():
    """UP_DOWN: a V->C tile.move boundary becomes tile.aic_gather, and the cube
    placement move on the gathered tile stays FULL ([128, 128] Mat) — the cube
    side never sees a halved tile."""
    span = ir.Span.unknown()
    vec = ir.Var("vec", _tile([128, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    move = T.move(vec, MS.Mat, span=span)  # V->C boundary
    gathered = ir.Var("gathered", move.type, span)
    store = T.store(vec, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(vec, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(gathered, move, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    assert "tile.aic_gather" in text
    # The cube placement move keeps the FULL [128, 128] Mat tile.
    assert "gathered: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat]" in text


def test_pure_vector_split_is_left_untouched():
    """A PURE-vector ``pl.split`` function (no cube boundary) is NOT lowered.

    Regression for the CI failure where LowerAutoVectorSplit stamped ``split_aiv``
    on a pure-vector function (an elementwise op split across the AIV lanes);
    ExpandMixedKernel then stripped the ``split`` attr in its non-mixed AIV-convert
    branch, leaving ``split_aiv`` without a split mode and tripping
    SplitVectorKernel. Such functions have no cube<->vector boundary to converge,
    so the pass must leave them exactly as-is (split preserved, no split_aiv, body
    un-halved) — the same un-split behavior they had before the convergence.
    """
    span = ir.Span.unknown()
    # Built directly (NOT via _incore_program, which injects a cube boundary) so
    # the function is genuinely pure-vector: load (Vec) -> store, no cube op.
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    store = T.store(t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    func = ir.Function(
        "pure_vec",
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(t, load, span),
                ir.AssignStmt(out_store, store, span),
                ir.ReturnStmt([out_store], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.UP_DOWN},
    )
    after = _lower(ir.Program([func], "pure_vec", span))

    func_after = next(iter(after.functions.values()))
    assert func_after.attrs.get("split_aiv") is None  # NOT marked
    assert func_after.attrs.get("split") is not None  # split mode preserved
    text = ir.python_print(after)
    assert "get_subblock_idx" not in text  # not lowered
    assert "aiv_shard" not in text  # no boundary inserted
    assert "[64, 128]" not in text  # body not halved


# ---------------------------------------------------------------------------
# Explicit SplitAivScopeStmt region path (RFC #1300 nestable first-class node).
#
# LowerAutoVectorSplit is the SOLE consumer of SplitAivScopeStmt: it injects a
# per-region subblock index, halves ONLY the vector compute INSIDE each region
# (region-local maps so no leak to sibling regions or out-of-region full-width
# ops), validates a per-region transpose hazard, then DROPS the scope wrapper.
# The AUTO whole-function path above is unchanged.
# ---------------------------------------------------------------------------


def _count_split_aiv_scopes(program):
    """Count surviving SplitAivScopeStmt nodes across all functions."""
    count = 0

    class Finder(ir.IRVisitor):
        def visit_split_aiv_scope_stmt(self, op):
            nonlocal count
            count += 1
            super().visit_split_aiv_scope_stmt(op)

    finder = Finder()
    for func in program.functions.values():
        finder.visit_stmt(func.body)
    return count


def _vec_load_region(span, mode, data, out, *, full_shape=(128, 128)):
    """A SplitAivScopeStmt region: aiv_id binding + a Vec load + store.

    Mirrors the parser-produced shape (the body opens with
    ``aiv_id = tile.get_subblock_idx()``). Returns (region_node, out_store_var).
    """
    aiv_id_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_id_call.type, span)
    load = T.load(data, [0, 0], list(full_shape), target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    store = T.store(t, [0, 0], out, span=span)
    out_store = ir.Var("out_store", store.type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(aiv_id, aiv_id_call, span),
            ir.AssignStmt(t, load, span),
            ir.AssignStmt(out_store, store, span),
        ],
        span,
    )
    region = ir.SplitAivScopeStmt(split=mode, body=body, span=span)
    return region, out_store


def _explicit_region_program(stmts, params, return_types, *, name="split_explicit"):
    """A single InCore function whose body carries explicit SplitAivScopeStmt regions."""
    span = ir.Span.unknown()
    func = ir.Function(name, params, return_types, ir.SeqStmts(stmts, span), span, ir.FunctionType.InCore)
    return ir.Program([func], name, span)


def test_explicit_region_erased():
    """Pass 21 consumes the region: no SplitAivScopeStmt survives, and the func is
    stamped split_aiv + split_aiv_region_validated."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    region, out_store = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_0)
    program = _explicit_region_program(
        [region, ir.ReturnStmt([out_store], span)],
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
    )
    after = _lower(program)
    assert _count_split_aiv_scopes(after) == 0  # the scope wrapper is dropped
    fattrs = _func_attrs(after)
    assert fattrs.get("split_aiv") is True
    assert fattrs.get("split_aiv_region_validated") is True
    text = ir.python_print(after)
    assert "split_aiv(" not in text  # the loop form is gone (node erased)
    assert "get_subblock_idx" in text  # region body still derives the lane index


def test_region_injects_subblock_idx():
    """The pass prepends a `subblock_idx = tile.get_subblock_idx()` binding at the
    region head and halves the vector load on the split axis."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    region, out_store = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_0)
    program = _explicit_region_program(
        [region, ir.ReturnStmt([out_store], span)],
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
    )
    text = ir.python_print(_lower(program))
    assert "subblock_idx" in text  # injected per-region index
    assert "t: pl.Tile[[64, 128]" in text  # the in-region load is halved on dim0
    assert "[0 + subblock_idx * 64, 0]" in text  # offset localized per subblock


def test_region_halves_only_inside():
    """Out-of-region vector compute stays FULL-WIDTH; only the in-region load is
    halved (region-local maps do not leak)."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_outer = ir.Var("out_outer", _tensor([128, 128]), span)
    out_inner = ir.Var("out_inner", _tensor([128, 128]), span)

    # Out-of-region vector load + store: must stay FULL [128, 128].
    outer_load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    t_outer = ir.Var("t_outer", outer_load.type, span)
    outer_store = T.store(t_outer, [0, 0], out_outer, span=span)
    outer_store_var = ir.Var("outer_store", outer_store.type, span)

    region, inner_store_var = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_inner)

    program = _explicit_region_program(
        [
            ir.AssignStmt(t_outer, outer_load, span),
            ir.AssignStmt(outer_store_var, outer_store, span),
            region,
            ir.ReturnStmt([outer_store_var, inner_store_var], span),
        ],
        [(data, _IN), (out_outer, _OUT), (out_inner, _OUT)],
        [out_outer.type, out_inner.type],
    )
    text = ir.python_print(_lower(program))
    assert "t_outer: pl.Tile[[128, 128]" in text  # outside region: full width
    assert "t: pl.Tile[[64, 128]" in text  # inside region: halved


def test_multi_mode_two_regions():
    """Two sibling regions with DIFFERENT modes halve independently: UP_DOWN on
    dim0, LEFT_RIGHT on dim1 — no cross-region leak."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_ud = ir.Var("out_ud", _tensor([128, 128]), span)
    out_lr = ir.Var("out_lr", _tensor([128, 128]), span)

    region_ud, store_ud = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_ud)
    region_lr, store_lr = _vec_load_region(span, ir.SplitMode.LEFT_RIGHT, data, out_lr)

    program = _explicit_region_program(
        [region_ud, region_lr, ir.ReturnStmt([store_ud, store_lr], span)],
        [(data, _IN), (out_ud, _OUT), (out_lr, _OUT)],
        [out_ud.type, out_lr.type],
    )
    after = _lower(program)
    assert _count_split_aiv_scopes(after) == 0
    text = ir.python_print(after)
    # Each region halves on its OWN axis: UP_DOWN -> dim0, LEFT_RIGHT -> dim1.
    assert "pl.Tile[[64, 128]" in text  # UP_DOWN region halves dim0
    assert "pl.Tile[[128, 64]" in text  # LEFT_RIGHT region halves dim1
    # The two regions use independent per-region subblock indices (no leak).
    assert "[0 + subblock_idx" in text  # UP_DOWN: row offset localized
    assert "[0, 0 + subblock_idx" in text  # LEFT_RIGHT: col offset localized


def test_transpose_hazard_per_region():
    """A tile.transpose that swaps the split axis inside a region is rejected with
    an actionable ValueError (validated with THAT region's split_dim)."""
    span = ir.Span.unknown()
    src = ir.Var("src", _tile([16, 8], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", _tensor([8, 16]), span)
    aiv_id_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_id_call.type, span)
    tr = T.transpose(src, 0, 1, span=span)  # swaps split dim0 on a non-singleton source
    zt = ir.Var("zt", tr.type, span)
    store = T.store(zt, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    region = ir.SplitAivScopeStmt(
        split=ir.SplitMode.UP_DOWN,
        body=ir.SeqStmts(
            [
                ir.AssignStmt(aiv_id, aiv_id_call, span),
                ir.AssignStmt(zt, tr, span),
                ir.AssignStmt(out_store, store, span),
            ],
            span,
        ),
        span=span,
    )
    program = _explicit_region_program(
        [region, ir.ReturnStmt([out_store], span)],
        [(src, _IN), (out_0, _OUT)],
        [out_0.type],
    )
    with pytest.raises(ValueError, match="swaps the split axis"):
        _lower(program)


def test_auto_path_unchanged():
    """The AUTO whole-function pl.split path is untouched: it still inserts
    aiv_shard, halves the vector region, stamps split_aiv — and crucially does NOT
    take the explicit-region branch (no split_aiv_region_validated marker)."""
    program = _build_c2v_mixed_program()
    after = _lower(program)
    fattrs = _func_attrs(after)
    assert fattrs.get("split_aiv") is True
    assert fattrs.get("split_aiv_region_validated") is None  # AUTO path, not region path
    text = ir.python_print(after)
    assert "aiv_shard" in text
    assert "[64, 128]" in text  # vector sub-region halved
    assert _count_split_aiv_scopes(after) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
