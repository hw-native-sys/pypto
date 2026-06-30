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

The pass runs at the tile level (post-InferTileMemorySpace), so the ``@pl.program``
DSL cannot express its input *or* its lowered output. Each transform-output test
therefore hand-builds both the ``Before`` program and an explicit ``Expected``
lowered program with the same helpers and asserts ``ir.assert_structural_equal``
between the pass result and ``Expected`` (the project's mandated transform-test
style). The two negative tests (reduce-on-split-axis, transpose hazard) keep
``pytest.raises`` because a rejected transform produces no ``After`` IR.

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

# The index type of the per-lane ``tile.get_subblock_idx()`` (Scalar[INDEX]).
_IDX = T.get_subblock_idx(span=ir.Span.unknown()).type


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


# ---------------------------------------------------------------------------
# Expected-IR (lowered-form) construction helpers.
#
# LowerAutoVectorSplit runs at the tile level, so the @pl.program DSL cannot
# author its output. These helpers build the *lowered* Expected the same way the
# Before is hand-built, so each test asserts via ir.assert_structural_equal
# against an explicit Expected program (not a python_print substring grep).
# ---------------------------------------------------------------------------


def _sub_var(name="subblock_idx"):
    """A fresh per-lane index ``Var`` (the ``subblock_idx`` the pass injects)."""
    return ir.Var(name, _IDX, ir.Span.unknown())


def _get_subblock(var, span):
    """``<var> = tile.get_subblock_idx()`` binding."""
    return ir.AssignStmt(var, T.get_subblock_idx(span=span), span)


def _shard_vec(tile, split, half_shape, span):
    """A C->V boundary ``tile.aiv_shard`` returning a HALF *Vec* tile.

    The pass reattaches the move's Vec destination memory onto the deduced half
    type, so the result is Vec — ``T.aiv_shard`` alone would inherit the cube
    input's memory, hence the explicit result type here.
    """
    return ir.Call(
        ir.get_op("tile.aiv_shard"), [tile], {"split": split}, _tile(half_shape, None, MS.Vec), span
    )


def _gather_vec(tile, split, full_shape, span):
    """A V->C boundary ``tile.aic_gather`` returning a doubled *Vec* tile."""
    return ir.Call(
        ir.get_op("tile.aic_gather"), [tile], {"split": split}, _tile(full_shape, None, MS.Vec), span
    )


def _move_call(src, target_mem, result_type, span):
    """``tile.move(src, target_memory=...)`` with an explicit result type.

    The pass keeps the original cube-placement move's full ``[128, 128]`` Mat
    result type while rebinding its input to the (doubled) gather result.
    """
    return ir.Call(ir.get_op("tile.move"), [src], {"target_memory": target_mem}, result_type, span)


def _half_add_type(half_shape, span):
    """The halved elementwise-add result type, carrying the Vec col-major
    ``tile_view`` the pass preserves.

    Derived by moving a half-shaped cube tile to Vec and adding — mirroring how
    the original add result type was built, but at the halved extent so the
    view's ``valid_shape`` halves in lockstep.
    """
    qkh = ir.Var("qkh", _tile(half_shape, mem=MS.Mat), span)
    mvh = T.move(qkh, MS.Vec, span=span)
    assert isinstance(mvh.type, ir.TileType)
    pvh = ir.Var("pvh", _tile(mvh.type.shape, mvh.type.tile_view, MS.Vec), span)
    return T.add(pvh, pvh, span).type


def _expected_incore(params, lowered_stmts, return_types, *, mode, sub, name="split_auto"):
    """Lowered counterpart of ``_incore_program``.

    The injected cube-seed C->V boundary becomes
    ``subblock_idx = tile.get_subblock_idx()`` + ``aiv_shard(cube_seed)``,
    followed by the (already-halved) vector ``lowered_stmts`` the caller builds
    using ``sub``. Stamps ``split`` + ``split_aiv``.
    """
    span = ir.Span.unknown()
    cube_seed = ir.Var("cube_seed", _tile([128, 128], mem=MS.Mat), span)
    half = [64, 128] if mode.value == 1 else [128, 64]
    seed_shard = _shard_vec(cube_seed, mode.value, half, span)
    assert isinstance(seed_shard.type, ir.Type)
    seed_vec = ir.Var("seed_vec", seed_shard.type, span)
    body = ir.SeqStmts(
        [_get_subblock(sub, span), ir.AssignStmt(seed_vec, seed_shard, span), *lowered_stmts],
        span,
    )
    func = ir.Function(
        name,
        [(cube_seed, _IN), *params],
        return_types,
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": mode, "split_aiv": True},
    )
    return ir.Program([func], name, span)


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


def _expected_c2v_mixed_program():
    """Lowered form of ``_build_c2v_mixed_program``: the C->V ``tile.move`` becomes
    ``tile.aiv_shard(split=1)`` (HALF Vec), the vector add result halves to
    ``[64, 128]`` (keeping its col-major view), the store offset is localized per
    subblock, ``subblock_idx`` is injected, and ``split_aiv`` is stamped.
    """
    span = ir.Span.unknown()
    qk = ir.Var("qk", _tile([128, 128], mem=MS.Mat), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    sub = _sub_var()
    shard = _shard_vec(qk, 1, [64, 128], span)
    popped = ir.Var("popped", shard.type, span)
    y_type = _half_add_type([64, 128], span)
    add = ir.Call(ir.get_op("tile.add"), [popped, popped], y_type, span)
    y = ir.Var("y", y_type, span)
    store = T.store(y, [0 + sub * 64, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    body = ir.SeqStmts(
        [
            _get_subblock(sub, span),
            ir.AssignStmt(popped, shard, span),
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
        attrs={"split": ir.SplitMode.UP_DOWN, "split_aiv": True},
    )
    return ir.Program([func], "test_c2v_mixed", span)


def test_c2v_boundary_becomes_aiv_shard_and_vector_region_is_halved():
    """The C->V ``tile.move`` becomes ``tile.aiv_shard(split=1)`` and the vector
    sub-region (add + store result) is halved to ``[64, 128]`` while the cube
    operand ``qk`` stays full; ``subblock_idx`` is injected and ``split_aiv``
    stamped (the full lowered shape is pinned by ``Expected``)."""
    ir.assert_structural_equal(_lower(_build_c2v_mixed_program()), _expected_c2v_mixed_program())


def test_store_offset_is_localized_per_subblock():
    """The vector store offset is localized: ``[0, 0] -> [0 + subblock_idx * 64, 0]``
    (AdjustOffsets adds ``subblock_idx * half`` on the split axis, dim0)."""
    ir.assert_structural_equal(_lower(_build_c2v_mixed_program()), _expected_c2v_mixed_program())


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

    # Expected: load result + shape/valid args halve on dim0; offset localized.
    sub = _sub_var()
    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_load = T.load(e_data, [0 + sub * 64, 0], [64, 128], target_memory=MS.Vec, span=span)
    e_prev = ir.Var("prev", e_load.type, span)
    e_store = T.store(e_prev, [0 + sub * 64, 0], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_data, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_prev, e_load, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.UP_DOWN,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


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

    sub = _sub_var()
    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_load = T.load(e_data, [0, 0 + sub * 64], [128, 64], target_memory=MS.Vec, span=span)
    e_prev = ir.Var("prev", e_load.type, span)
    e_store = T.store(e_prev, [0, 0 + sub * 64], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_data, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_prev, e_load, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.LEFT_RIGHT,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_vector_slice_halves_shape_and_localizes_offset():
    """UP_DOWN: a tile.slice of a full (unsplit) Vec source halves its static shape
    tuple in lockstep with the result (the qk_pv strided sub-slice fix) and
    localizes its zero-base offset per subblock."""
    span = ir.Span.unknown()
    src = ir.Var("src", _tile([128, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    sl = T.slice(src, [128, 128], [0, 0], span=span)
    sub_t = ir.Var("sub", sl.type, span)
    store = T.store(sub_t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(src, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(sub_t, sl, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )

    # Result type AND the static shape tuple arg both halve to [64, 128].
    sub = _sub_var()
    e_src = ir.Var("src", _tile([128, 128], mem=MS.Vec), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_sl = T.slice(e_src, [64, 128], [0 + sub * 64, 0], span=span)
    e_sub_t = ir.Var("sub", e_sl.type, span)
    e_store = T.store(e_sub_t, [0 + sub * 64, 0], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_src, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_sub_t, e_sl, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.UP_DOWN,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_vector_slice_nonzero_base_offset_localizes_additively():
    """UP_DOWN: a strided sub-slice at a non-zero base offset localizes additively —
    the original offset is preserved and subblock_idx*half is added on the split
    axis (the exact qk_pv ``oi[16:32]`` pattern)."""
    span = ir.Span.unknown()
    src = ir.Var("src", _tile([256, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    sl = T.slice(src, [128, 128], [16, 0], span=span)
    sub_t = ir.Var("sub", sl.type, span)
    store = T.store(sub_t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(src, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(sub_t, sl, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
    )

    # Base offset 16 preserved, subblock_idx * 64 added on the split axis.
    sub = _sub_var()
    e_src = ir.Var("src", _tile([256, 128], mem=MS.Vec), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_sl = T.slice(e_src, [64, 128], [16 + sub * 64, 0], span=span)
    e_sub_t = ir.Var("sub", e_sl.type, span)
    e_store = T.store(e_sub_t, [0 + sub * 64, 0], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_src, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_sub_t, e_sl, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.UP_DOWN,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


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
    sub_t = ir.Var("sub", sl.type, span)
    store = T.store(sub_t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    program = _incore_program(
        [(data, _IN), (out_0, _OUT)],
        [
            ir.AssignStmt(prev, load, span),
            ir.AssignStmt(sub_t, sl, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        [out_0.type],
        mode=ir.SplitMode.LEFT_RIGHT,
    )

    # Source load halved [16,128] -> [16,64] (tracked); the slice halves its
    # shape tuple in lockstep but keeps the [0, 0] lane-local offset.
    sub = _sub_var()
    e_data = ir.Var("data", _tensor([16, 128]), span)
    e_out = ir.Var("out_0", _tensor([16, 128]), span)
    e_load = T.load(e_data, [0, 0 + sub * 64], [16, 64], target_memory=MS.Vec, span=span)
    e_prev = ir.Var("prev", e_load.type, span)
    e_sl = T.slice(e_prev, [16, 64], [0, 0], span=span)
    e_sub_t = ir.Var("sub", e_sl.type, span)
    e_store = T.store(e_sub_t, [0, 0 + sub * 64], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_data, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_prev, e_load, span),
            ir.AssignStmt(e_sub_t, e_sl, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.LEFT_RIGHT,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


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

    # Rank-1 load bypassed (stays [128]); reshape stays full [128, 1]; a slice to
    # [64, 1] at the per-subblock row offset is appended; store offset localized.
    sub = _sub_var()
    e_scale = ir.Var("scale", _tensor([128]), span)
    e_out = ir.Var("out_0", _tensor([128, 1]), span)
    e_load = T.load(e_scale, [0], [128], target_memory=MS.Vec, span=span)
    e_scale_row = ir.Var("scale_row", e_load.type, span)
    e_reshape = T.reshape(e_scale_row, [128, 1], span=span)
    e_scale_2d = ir.Var("scale_2d", e_reshape.type, span)
    e_sl = T.slice(e_scale_2d, [64, 1], [sub * 64, 0], span=span)
    e_scale_2d_1 = ir.Var("scale_2d_1", e_sl.type, span)
    e_store = T.store(e_scale_2d_1, [0 + sub * 64, 0], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_scale, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_scale_row, e_load, span),
            ir.AssignStmt(e_scale_2d, e_reshape, span),
            ir.AssignStmt(e_scale_2d_1, e_sl, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.UP_DOWN,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


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

    # Input load halved [16,16] -> [8,16]; reshape result AND shape arg halve.
    sub = _sub_var()
    e_data = ir.Var("data", _tensor([16, 16]), span)
    e_out = ir.Var("out_0", _tensor([256, 1]), span)
    e_load = T.load(e_data, [0 + sub * 8, 0], [8, 16], target_memory=MS.Vec, span=span)
    e_prev = ir.Var("prev", e_load.type, span)
    e_reshape = T.reshape(e_prev, [128, 1], span=span)
    e_flat = ir.Var("flat", e_reshape.type, span)
    e_store = T.store(e_flat, [0 + sub * 128, 0], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_data, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_prev, e_load, span),
            ir.AssignStmt(e_flat, e_reshape, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.UP_DOWN,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


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

    # The singleton split dim is preserved: av stays [1, 128], store offset [0, 0].
    sub = _sub_var()
    e_src = ir.Var("src", _tile([1, 128], mem=MS.Vec), span)
    e_out = ir.Var("out_0", _tensor([1, 128]), span)
    e_add = T.add(e_src, e_src, span)
    e_av = ir.Var("av", e_add.type, span)
    e_store = T.store(e_av, [0, 0], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_src, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_av, e_add, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.UP_DOWN,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


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

    # The loaded accumulator is halved; the in-loop store offset is localized
    # using the same tracked half extent.
    sub = _sub_var()
    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_load = T.load(e_data, [0 + sub * 64, 0], [64, 128], target_memory=MS.Vec, span=span)
    e_accum = ir.Var("accum", e_load.type, span)
    e_i = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    e_out_iter = ir.IterArg("out_it", e_out.type, e_out, span)
    e_body_store = T.store(e_accum, [0 + sub * 64, 0], e_out_iter, span=span)
    e_body_store_var = ir.Var("out_it_next", e_body_store.type, span)
    e_loop_ret = ir.Var("out_loop", e_out.type, span)
    e_for_body = ir.SeqStmts(
        [ir.AssignStmt(e_body_store_var, e_body_store, span), ir.YieldStmt([e_body_store_var], span)],
        span,
    )
    e_for_stmt = ir.ForStmt(
        e_i,
        ir.ConstInt(0, DataType.INDEX, span),
        ir.ConstInt(2, DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
        [e_out_iter],
        e_for_body,
        [e_loop_ret],
        span,
    )
    expected = _expected_incore(
        [(e_data, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_accum, e_load, span),
            e_for_stmt,
            ir.ReturnStmt([e_loop_ret], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.UP_DOWN,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_reduce_on_split_axis_rejected():
    """A reduce that collapses the split axis (dim0 under UP_DOWN) raises ValueError —
    a partial per-lane reduction is a miscompile. NEGATIVE test: a rejected
    transform produces no ``After`` IR, so Before-After-Expected does not apply."""
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

    # V->C move becomes aic_gather (HALF -> FULL, doubled to [256, 128] Vec);
    # the cube placement move keeps the FULL [128, 128] Mat tile.
    sub = _sub_var()
    e_vec = ir.Var("vec", _tile([128, 128], mem=MS.Vec), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_gather = _gather_vec(e_vec, 1, [256, 128], span)
    e_gathered_mat = ir.Var("gathered_mat", e_gather.type, span)
    e_move = _move_call(e_gathered_mat, MS.Mat, _tile([128, 128], None, MS.Mat), span)
    e_gathered = ir.Var("gathered", e_move.type, span)
    e_store = T.store(e_vec, [0, 0], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_incore(
        [(e_vec, _IN), (e_out, _OUT)],
        [
            ir.AssignStmt(e_gathered_mat, e_gather, span),
            ir.AssignStmt(e_gathered, e_move, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [e_out.type],
        mode=ir.SplitMode.UP_DOWN,
        sub=sub,
    )
    ir.assert_structural_equal(_lower(program), expected)


def _pure_vector_program():
    """A PURE-vector ``pl.split`` function (no cube boundary): load (Vec) -> store.

    Built directly (NOT via ``_incore_program``, which injects a cube boundary) so
    the function is genuinely pure-vector — there is no cube<->vector boundary to
    converge, so the pass must leave it exactly as-is.
    """
    span = ir.Span.unknown()
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
    return ir.Program([func], "pure_vec", span)


def test_pure_vector_split_is_left_untouched():
    """A PURE-vector ``pl.split`` function (no cube boundary) is NOT lowered.

    Regression for the CI failure where LowerAutoVectorSplit stamped ``split_aiv``
    on a pure-vector function (an elementwise op split across the AIV lanes);
    ExpandMixedKernel then stripped the ``split`` attr in its non-mixed AIV-convert
    branch, leaving ``split_aiv`` without a split mode and tripping
    SplitVectorKernel. Such functions have no cube<->vector boundary to converge,
    so the pass must leave them exactly as-is (split preserved, no split_aiv, body
    un-halved) — the Expected is the input program, unchanged.
    """
    ir.assert_structural_equal(_lower(_pure_vector_program()), _pure_vector_program())


# ---------------------------------------------------------------------------
# Explicit SplitAivScopeStmt region path (RFC #1300 nestable first-class node).
#
# LowerAutoVectorSplit is the SOLE consumer of SplitAivScopeStmt: it injects a
# per-region subblock index, halves ONLY the vector compute INSIDE each region
# (region-local maps so no leak to sibling regions or out-of-region full-width
# ops), validates a per-region transpose hazard, then DROPS the scope wrapper.
# The AUTO whole-function path above is unchanged.
# ---------------------------------------------------------------------------

# Attrs the region path stamps on the function (no whole-function ``split`` mode —
# each region carries its own ``split_``).
_REGION_ATTRS = {"split_aiv": True, "split_aiv_region_validated": True}


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


def _lowered_vec_load_region(span, mode, data, out, *, full_shape=(128, 128)):
    """Scope-erased lowered form of ``_vec_load_region``.

    The region path prepends an injected ``subblock_idx = get_subblock_idx()``,
    keeps the region's own ``aiv_id`` binding, halves the load on the split axis,
    and localizes the load + store offsets per subblock. Returns
    ``(lowered_stmts, out_store_var)``.
    """
    sub = _sub_var()
    aiv_id = ir.Var("aiv_id", _IDX, span)
    if mode.value == 1:
        half = [full_shape[0] // 2, full_shape[1]]
        off = [0 + sub * (full_shape[0] // 2), 0]
    else:
        half = [full_shape[0], full_shape[1] // 2]
        off = [0, 0 + sub * (full_shape[1] // 2)]
    load = T.load(data, off, half, target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    store = T.store(t, off, out, span=span)
    out_store = ir.Var("out_store", store.type, span)
    stmts: list[ir.Stmt] = [
        _get_subblock(sub, span),
        _get_subblock(aiv_id, span),
        ir.AssignStmt(t, load, span),
        ir.AssignStmt(out_store, store, span),
    ]
    return stmts, out_store


def _explicit_region_program(stmts, params, return_types, *, name="split_explicit"):
    """A single InCore function whose body carries explicit SplitAivScopeStmt regions."""
    span = ir.Span.unknown()
    func = ir.Function(name, params, return_types, ir.SeqStmts(stmts, span), span, ir.FunctionType.InCore)
    return ir.Program([func], name, span)


def _expected_region_program(stmts, params, return_types, *, name="split_explicit", attrs=None):
    """Lowered counterpart of ``_explicit_region_program``: the scope wrapper is
    erased and the function is stamped ``split_aiv`` + ``split_aiv_region_validated``
    (unless ``attrs`` overrides)."""
    span = ir.Span.unknown()
    func = ir.Function(
        name,
        params,
        return_types,
        ir.SeqStmts(stmts, span),
        span,
        ir.FunctionType.InCore,
        attrs=attrs if attrs is not None else dict(_REGION_ATTRS),
    )
    return ir.Program([func], name, span)


def test_explicit_region_erased():
    """Pass 21 consumes the region: no SplitAivScopeStmt survives, and the func is
    stamped split_aiv + split_aiv_region_validated. The region body keeps its own
    ``aiv_id`` and gains the injected ``subblock_idx`` + halved load (Expected)."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    region, out_store = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_0)
    program = _explicit_region_program(
        [region, ir.ReturnStmt([out_store], span)],
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
    )

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    stmts, e_out_store = _lowered_vec_load_region(span, ir.SplitMode.UP_DOWN, e_data, e_out)
    expected = _expected_region_program(
        [*stmts, ir.ReturnStmt([e_out_store], span)],
        [(e_data, _IN), (e_out, _OUT)],
        [e_out.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_region_injects_subblock_idx():
    """The pass prepends a `subblock_idx = tile.get_subblock_idx()` binding at the
    region head and halves the vector load on the split axis (Expected pins both
    the injected index and the halved in-region load)."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    region, out_store = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_0)
    program = _explicit_region_program(
        [region, ir.ReturnStmt([out_store], span)],
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
    )

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    stmts, e_out_store = _lowered_vec_load_region(span, ir.SplitMode.UP_DOWN, e_data, e_out)
    expected = _expected_region_program(
        [*stmts, ir.ReturnStmt([e_out_store], span)],
        [(e_data, _IN), (e_out, _OUT)],
        [e_out.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


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

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out_outer = ir.Var("out_outer", _tensor([128, 128]), span)
    e_out_inner = ir.Var("out_inner", _tensor([128, 128]), span)
    e_outer_load = T.load(e_data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    e_t_outer = ir.Var("t_outer", e_outer_load.type, span)
    e_outer_store = T.store(e_t_outer, [0, 0], e_out_outer, span=span)
    e_outer_store_var = ir.Var("outer_store", e_outer_store.type, span)
    stmts, e_inner_store_var = _lowered_vec_load_region(span, ir.SplitMode.UP_DOWN, e_data, e_out_inner)
    expected = _expected_region_program(
        [
            ir.AssignStmt(e_t_outer, e_outer_load, span),
            ir.AssignStmt(e_outer_store_var, e_outer_store, span),
            *stmts,
            ir.ReturnStmt([e_outer_store_var, e_inner_store_var], span),
        ],
        [(e_data, _IN), (e_out_outer, _OUT), (e_out_inner, _OUT)],
        [e_out_outer.type, e_out_inner.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_multi_mode_two_regions():
    """Two sibling regions with DIFFERENT modes halve independently: UP_DOWN on
    dim0, LEFT_RIGHT on dim1 — no cross-region leak. Each region gets its own
    injected subblock index (Expected has two independent index bindings)."""
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

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out_ud = ir.Var("out_ud", _tensor([128, 128]), span)
    e_out_lr = ir.Var("out_lr", _tensor([128, 128]), span)
    stmts_ud, e_store_ud = _lowered_vec_load_region(span, ir.SplitMode.UP_DOWN, e_data, e_out_ud)
    stmts_lr, e_store_lr = _lowered_vec_load_region(span, ir.SplitMode.LEFT_RIGHT, e_data, e_out_lr)
    expected = _expected_region_program(
        [*stmts_ud, *stmts_lr, ir.ReturnStmt([e_store_ud, e_store_lr], span)],
        [(e_data, _IN), (e_out_ud, _OUT), (e_out_lr, _OUT)],
        [e_out_ud.type, e_out_lr.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_transpose_hazard_per_region():
    """A tile.transpose that swaps the split axis inside a region is rejected with
    an actionable ValueError (validated with THAT region's split_dim). NEGATIVE
    test: a rejected transform produces no ``After`` IR, so Before-After-Expected
    does not apply."""
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


def test_explicit_aiv_shard_region_passed_through_not_double_sharded():
    """A region whose body already carries a user-authored tile.aiv_shard (the
    user sharded the cube tile manually and wrote the vector compute on the
    per-lane half) must be spliced through UNCHANGED: the scope wrapper is
    dropped but the body is NOT re-routed through the affinity-gated halving.

    Regression: re-halving such a body double-sharded the explicit aiv_shard
    (the downstream Acc->Vec move was misread as a fresh C->V boundary and
    rewritten to a second aiv_shard), orphaning a halved Acc memref that never
    got an allocation and crashing PTO codegen. Expected pins exactly ONE
    aiv_shard (the user's, on an Acc tile) and ONE get_subblock_idx (no injected
    subblock_idx), with no SplitAivScopeStmt surviving.
    """
    span = ir.Span.unknown()
    a_left = ir.Var("a_left", _tile([128, 128], mem=MS.Left), span)
    b_right = ir.Var("b_right", _tile([128, 128], mem=MS.Right), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)

    matmul = T.matmul(a_left, b_right, span=span)  # cube Acc tile, full width, OUTSIDE the region
    qk = ir.Var("qk", matmul.type, span)

    aiv_id_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_id_call.type, span)
    shard = T.aiv_shard(qk, split=1, span=span)  # USER's explicit C->V shard -> this lane's half
    qk_h = ir.Var("qk_h", shard.type, span)
    sc = T.muls(qk_h, 2.0, span=span)  # vector compute on the half
    sc_var = ir.Var("sc", sc.type, span)
    store = T.store(sc_var, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    region = ir.SplitAivScopeStmt(
        split=ir.SplitMode.UP_DOWN,
        body=ir.SeqStmts(
            [
                ir.AssignStmt(aiv_id, aiv_id_call, span),
                ir.AssignStmt(qk_h, shard, span),
                ir.AssignStmt(sc_var, sc, span),
                ir.AssignStmt(out_store, store, span),
            ],
            span,
        ),
        span=span,
    )
    program = _explicit_region_program(
        [ir.AssignStmt(qk, matmul, span), region, ir.ReturnStmt([out_store], span)],
        [(a_left, _IN), (b_right, _IN), (out_0, _OUT)],
        [out_0.type],
    )

    # The body is spliced through unchanged (NO re-halving): the user's single
    # aiv_shard (Acc) + single aiv_id binding survive; only the scope wrapper is
    # dropped and the function is stamped split_aiv + split_aiv_region_validated.
    e_a = ir.Var("a_left", _tile([128, 128], mem=MS.Left), span)
    e_b = ir.Var("b_right", _tile([128, 128], mem=MS.Right), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_matmul = T.matmul(e_a, e_b, span=span)
    e_qk = ir.Var("qk", e_matmul.type, span)
    e_aiv_id = ir.Var("aiv_id", _IDX, span)
    e_shard = T.aiv_shard(e_qk, split=1, span=span)
    e_qk_h = ir.Var("qk_h", e_shard.type, span)
    e_sc = T.muls(e_qk_h, 2.0, span=span)
    e_sc_var = ir.Var("sc", e_sc.type, span)
    e_store = T.store(e_sc_var, [0, 0], e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    expected = _expected_region_program(
        [
            ir.AssignStmt(e_qk, e_matmul, span),
            _get_subblock(e_aiv_id, span),
            ir.AssignStmt(e_qk_h, e_shard, span),
            ir.AssignStmt(e_sc_var, e_sc, span),
            ir.AssignStmt(e_out_store, e_store, span),
            ir.ReturnStmt([e_out_store], span),
        ],
        [(e_a, _IN), (e_b, _IN), (e_out, _OUT)],
        [e_out.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_while_nested_region_lowered_and_erased():
    """A SplitAivScopeStmt nested inside a WhileStmt body is lowered + erased:
    LowerExplicitRegions recurses into the while body (mirroring the for/if arms),
    so no SplitAivScopeStmt survives to the codegen guard. Expected pins the
    lowered region inside the rebuilt while body."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    cond = ir.ConstInt(0, DataType.BOOL, span)
    region, out_store = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_0)
    while_stmt = ir.WhileStmt(cond, [], ir.SeqStmts([region], span), [], span)
    program = _explicit_region_program(
        [while_stmt, ir.ReturnStmt([out_store], span)],
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
    )

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_cond = ir.ConstInt(0, DataType.BOOL, span)
    stmts, e_out_store = _lowered_vec_load_region(span, ir.SplitMode.UP_DOWN, e_data, e_out)
    e_while = ir.WhileStmt(e_cond, [], ir.SeqStmts(stmts, span), [], span)
    expected = _expected_region_program(
        [e_while, ir.ReturnStmt([e_out_store], span)],
        [(e_data, _IN), (e_out, _OUT)],
        [e_out.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_empty_region_is_noop():
    """An empty region (e.g. body emptied by DCE) is a no-op: the scope wrapper is
    dropped with nothing spliced in (no crash from the per-lane index injection),
    while out-of-region full-width compute is preserved and the function is still
    stamped split_aiv + split_aiv_region_validated."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)

    outer_load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    t_outer = ir.Var("t_outer", outer_load.type, span)
    outer_store = T.store(t_outer, [0, 0], out_0, span=span)
    outer_store_var = ir.Var("outer_store", outer_store.type, span)
    empty_region = ir.SplitAivScopeStmt(split=ir.SplitMode.UP_DOWN, body=ir.SeqStmts([], span), span=span)

    program = _explicit_region_program(
        [
            ir.AssignStmt(t_outer, outer_load, span),
            ir.AssignStmt(outer_store_var, outer_store, span),
            empty_region,
            ir.ReturnStmt([outer_store_var], span),
        ],
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
    )

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_load = T.load(e_data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    e_t = ir.Var("t_outer", e_load.type, span)
    e_store = T.store(e_t, [0, 0], e_out, span=span)
    e_store_var = ir.Var("outer_store", e_store.type, span)
    expected = _expected_region_program(
        [
            ir.AssignStmt(e_t, e_load, span),
            ir.AssignStmt(e_store_var, e_store, span),
            ir.ReturnStmt([e_store_var], span),
        ],
        [(e_data, _IN), (e_out, _OUT)],
        [e_out.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_sibling_regions_get_distinct_subblock_idx_names():
    """Two sibling regions get DISTINCT injected ``subblock_idx`` names. The pass
    reserves the per-region index against the enclosing function body's names AND
    grows the set after each region, so the second region can't reuse the first's
    name (an empty reservation set made both ``subblock_idx``, breaking SSA).

    ``assert_structural_equal`` ignores Var name hints, so distinctness is asserted
    by walking the lowered IR for the injected ``tile.get_subblock_idx`` bindings.
    """
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_a = ir.Var("out_a", _tensor([128, 128]), span)
    out_b = ir.Var("out_b", _tensor([128, 128]), span)
    region_a, store_a = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_a)
    region_b, store_b = _vec_load_region(span, ir.SplitMode.UP_DOWN, data, out_b)
    program = _explicit_region_program(
        [region_a, region_b, ir.ReturnStmt([store_a, store_b], span)],
        [(data, _IN), (out_a, _OUT), (out_b, _OUT)],
        [out_a.type, out_b.type],
    )
    lowered = _lower(program)

    subblock_op = ir.get_op("tile.get_subblock_idx").name
    injected: list[str] = []

    def walk(node):
        if (
            isinstance(node, ir.AssignStmt)
            and isinstance(node.value, ir.Call)
            and node.value.op.name == subblock_op
            and node.var.name_hint.startswith("subblock_idx")
        ):
            injected.append(node.var.name_hint)
        if isinstance(node, ir.SeqStmts):
            for s in node.stmts:
                walk(s)
        else:
            body = getattr(node, "body", None)
            if body is not None:
                walk(body)

    for func in lowered.functions.values():
        walk(func.body)

    # One injected index per region; the two names must be distinct.
    assert len(injected) == 2, f"expected 2 injected subblock_idx bindings, got {injected}"
    assert len(set(injected)) == 2, f"sibling regions must get distinct names, got {injected}"


def test_mixed_explicit_implicit_region_rejected():
    """A region that MIXES an explicit ``tile.aiv_shard`` with a plain full-width
    vector op (a Vec ``tile.load`` the implicit path would otherwise halve) is
    rejected with an actionable user error: the explicit boundary keeps the region
    in half-width form, so the un-localized full-width op would corrupt both AIV
    lanes. NEGATIVE test: a rejected transform produces no ``After`` IR."""
    span = ir.Span.unknown()
    a_left = ir.Var("a_left", _tile([128, 128], mem=MS.Left), span)
    b_right = ir.Var("b_right", _tile([128, 128], mem=MS.Right), span)
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)

    matmul = T.matmul(a_left, b_right, span=span)
    qk = ir.Var("qk", matmul.type, span)

    aiv_id_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_id_call.type, span)
    shard = T.aiv_shard(qk, split=1, span=span)  # explicit C->V boundary (half)
    qk_h = ir.Var("qk_h", shard.type, span)
    # A full-width Vec load NOT derived from the shard: the implicit affinity gate
    # would halve it, but the explicit passthrough would leave it full-width.
    full_load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    full_t = ir.Var("full_t", full_load.type, span)
    store = T.store(full_t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    region = ir.SplitAivScopeStmt(
        split=ir.SplitMode.UP_DOWN,
        body=ir.SeqStmts(
            [
                ir.AssignStmt(aiv_id, aiv_id_call, span),
                ir.AssignStmt(qk_h, shard, span),
                ir.AssignStmt(full_t, full_load, span),
                ir.AssignStmt(out_store, store, span),
            ],
            span,
        ),
        span=span,
    )
    program = _explicit_region_program(
        [ir.AssignStmt(qk, matmul, span), region, ir.ReturnStmt([out_store], span)],
        [(a_left, _IN), (b_right, _IN), (data, _IN), (out_0, _OUT)],
        [out_0.type],
    )
    with pytest.raises(ValueError, match="mixes explicit"):
        _lower(program)


def test_auto_path_unchanged():
    """The AUTO whole-function pl.split path is untouched: it still inserts
    aiv_shard, halves the vector region, stamps split_aiv — and crucially does NOT
    take the explicit-region branch (no split_aiv_region_validated marker). The
    Expected carries ``{"split", "split_aiv"}`` only, so the absent region marker
    is load-bearing in the structural comparison."""
    ir.assert_structural_equal(_lower(_build_c2v_mixed_program()), _expected_c2v_mixed_program())


def test_while_inside_region_halves_vector_op():
    """A WhileStmt *inside* a region body has its vector ops halved: LowerStmts
    recurses into the while (mirroring its for/if arms), so the load is split on
    the axis and its offset localized rather than left full-width on both lanes.
    Before: region{ aiv_id, while{ load[128,128], store } }.
    Expected: region erased -> subblock_idx + aiv_id + while{ load[64,128] @ localized, store }."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    cond = ir.ConstInt(0, DataType.BOOL, span)
    aiv_id_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_id_call.type, span)
    load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    store = T.store(t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    b_while = ir.WhileStmt(
        cond,
        [],
        ir.SeqStmts([ir.AssignStmt(t, load, span), ir.AssignStmt(out_store, store, span)], span),
        [],
        span,
    )
    region = ir.SplitAivScopeStmt(
        split=ir.SplitMode.UP_DOWN,
        body=ir.SeqStmts([ir.AssignStmt(aiv_id, aiv_id_call, span), b_while], span),
        span=span,
    )
    program = _explicit_region_program(
        [region, ir.ReturnStmt([out_store], span)], [(data, _IN), (out_0, _OUT)], [out_0.type]
    )

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_cond = ir.ConstInt(0, DataType.BOOL, span)
    sub = _sub_var()
    e_aiv = ir.Var("aiv_id", _IDX, span)
    off = [0 + sub * 64, 0]  # UP_DOWN: row offset localized per subblock
    e_load = T.load(e_data, off, [64, 128], target_memory=MS.Vec, span=span)
    e_t = ir.Var("t", e_load.type, span)
    e_store = T.store(e_t, off, e_out, span=span)
    e_out_store = ir.Var("out_store", e_store.type, span)
    e_while = ir.WhileStmt(
        e_cond,
        [],
        ir.SeqStmts([ir.AssignStmt(e_t, e_load, span), ir.AssignStmt(e_out_store, e_store, span)], span),
        [],
        span,
    )
    expected = _expected_region_program(
        [_get_subblock(sub, span), _get_subblock(e_aiv, span), e_while, ir.ReturnStmt([e_out_store], span)],
        [(e_data, _IN), (e_out, _OUT)],
        [e_out.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_mixed_explicit_implicit_region_in_while_rejected():
    """The mixed-explicit validator recurses into a WhileStmt inside the region, so
    a plain full-width vector op buried in a while (not derived from the explicit
    tile.aiv_shard) is still rejected. NEGATIVE test: a rejected transform has no
    ``After`` IR."""
    span = ir.Span.unknown()
    a_left = ir.Var("a_left", _tile([128, 128], mem=MS.Left), span)
    b_right = ir.Var("b_right", _tile([128, 128], mem=MS.Right), span)
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    cond = ir.ConstInt(0, DataType.BOOL, span)
    matmul = T.matmul(a_left, b_right, span=span)
    qk = ir.Var("qk", matmul.type, span)
    aiv_id_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_id_call.type, span)
    shard = T.aiv_shard(qk, split=1, span=span)  # explicit C->V boundary (half)
    qk_h = ir.Var("qk_h", shard.type, span)
    # Full-width Vec load NOT derived from the shard, buried inside a while.
    full_load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    full_t = ir.Var("full_t", full_load.type, span)
    store = T.store(full_t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    inner_while = ir.WhileStmt(
        cond,
        [],
        ir.SeqStmts([ir.AssignStmt(full_t, full_load, span), ir.AssignStmt(out_store, store, span)], span),
        [],
        span,
    )
    region = ir.SplitAivScopeStmt(
        split=ir.SplitMode.UP_DOWN,
        body=ir.SeqStmts(
            [ir.AssignStmt(aiv_id, aiv_id_call, span), ir.AssignStmt(qk_h, shard, span), inner_while], span
        ),
        span=span,
    )
    program = _explicit_region_program(
        [ir.AssignStmt(qk, matmul, span), region, ir.ReturnStmt([out_store], span)],
        [(a_left, _IN), (b_right, _IN), (data, _IN), (out_0, _OUT)],
        [out_0.type],
    )
    with pytest.raises(ValueError, match="mixes explicit"):
        _lower(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
