# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit AIV region lowering and rejection of retired AUTO splitting.

Region bodies are authored per lane. Lowering retains lexical placement and
localizes explicit boundary valid extents, without changing computation shapes
or load/store addresses. Tests keep the print/parse roundtrip instrument enabled.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto import backend as _backend
from pypto.ir.instruments import make_roundtrip_instrument
from pypto.ir.op import tile_ops as T
from pypto.runtime import RunConfig

MS = ir.MemorySpace
FP32 = DataType.FP32
_IN = ir.ParamDirection.In
_OUT = ir.ParamDirection.Out

# The index type of the per-lane ``tile.get_subblock_idx()`` (Scalar[INDEX]).
_IDX = T.get_subblock_idx(span=ir.Span.unknown()).type


def _tile(shape, mem=None):
    return ir.TileType(shape, FP32, None, None, mem)


def _tensor(shape):
    return ir.TensorType(shape, FP32)


class _EraseSplitRegions(ir.IRMutator):
    """Project out region wrappers for the existing per-op shape/offset goldens.

    Region structure is tested separately on the unprojected pass output.
    The pass's roundtrip instrument always sees that unprojected output.
    """

    def visit_split_aiv_scope_stmt(self, op):
        return self.visit_stmt(op.body)

    def visit_seq_stmts(self, op):
        stmts = []
        for stmt in op.stmts:
            rewritten = self.visit_stmt(stmt)
            if isinstance(rewritten, ir.SeqStmts):
                stmts.extend(rewritten.stmts)
            else:
                stmts.append(rewritten)
        return ir.SeqStmts(stmts, op.span)


def _split_region_count(program):
    class Counter(ir.IRVisitor):
        count = 0

        def visit_split_aiv_scope_stmt(self, op):
            self.count += 1
            self.visit_stmt(op.body)

    counter = Counter()
    counter.visit_program(program)
    return counter.count


def _lower(program, *, keep_regions=False):
    """Run the pass with the print->parse roundtrip instrument kept ON.

    The programs here are minimal and hand-shaped rather than pipeline-produced,
    so BEFORE_AND_AFTER *property* verification (which the conftest also installs)
    rejects them up front — the region ``Before`` bodies do not satisfy
    ``IncoreTileOps``. That is why this file overrides the ambient context.

    The roundtrip instrument is deliberately kept, though: it asserts the pass's
    OUTPUT survives print->parse, which is cheap here and is exactly the check
    that a fully suppressed ``PassContext([])`` was hiding — the V->C boundary
    emitted a ``tile.move`` whose result shape contradicted its operand, and no
    test noticed until the DSL conversion tripped over it.
    """
    with passes.PassContext([make_roundtrip_instrument()]):
        result = passes.lower_auto_vector_split()(program)
    return result if keep_regions else _EraseSplitRegions().visit_program(result)


def test_pure_vector_split_is_left_untouched():
    """A PURE-vector ``pl.split`` function (no cube boundary) is NOT lowered.

    Regression for the CI failure where LowerAutoVectorSplit stamped ``split_aiv``
    on a pure-vector function (an elementwise op split across the AIV lanes);
    ExpandMixedKernel then stripped the ``split`` attr in its non-mixed AIV-convert
    path and the kernel lost its split entirely. There is no cube<->vector boundary
    to converge here, so the pass must leave the function exactly as-is.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def pure_vec(
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            t = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(t, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def pure_vec(
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            t = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(t, [0, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


# ---------------------------------------------------------------------------
# Hand-built IR helpers for the explicit SplitAivScopeStmt region path below.
# ---------------------------------------------------------------------------


def _sub_var(name="subblock_idx"):
    """A fresh per-lane index ``Var`` (the ``subblock_idx`` the pass injects)."""
    return ir.Var(name, _IDX, ir.Span.unknown())


def _get_subblock(var, span):
    """``<var> = tile.get_subblock_idx()`` binding."""
    return ir.AssignStmt(var, T.get_subblock_idx(span=span), span)


def _notify(span, sig, peer, notify_op=0):
    """A ``pld.system.notify`` Call — a core-agnostic (SHARED) comm op.

    SHARED is what ExpandMixedKernel duplicates onto both lanes, so this is the
    op whose placement the region stamp exists to decide.
    """
    zero = ir.ConstInt(0, DataType.INDEX, span)
    offsets = ir.MakeTuple([zero, zero], span)
    value = ir.ConstInt(1, DataType.INT32, span)
    return ir.create_op_call("pld.system.notify", [sig, peer, offsets, value], {"op": notify_op}, span)


def _placements(program, op_name):
    """Whether each call is lexically inside a retained AIV region."""
    found = []

    class Membership(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.depth = 0

        def visit_split_aiv_scope_stmt(self, op):
            self.depth += 1
            self.visit_stmt(op.body)
            self.depth -= 1

        def visit_call(self, op):
            assert "core_placement" not in op.attrs
            if isinstance(op.op, ir.Op) and op.op.name == op_name:
                found.append("aiv" if self.depth else None)
            super().visit_call(op)

    visitor = Membership()
    visitor.visit_program(program)
    return found


# Hand-built regions isolate lowering from outlining. Full-pipeline DSL tests
# below cover parser-produced regions, placement, and explicit communication.

# Attrs the region path stamps on the function (no whole-function ``split`` mode —
# each region carries its own ``split_``). Same for every mode, including the
# task-parallel ``None``: the ``split_aiv`` marker alone routes the function to the
# both-lanes split path downstream (never the lane-0-only no-split replay).
_REGION_ATTRS = {"split_aiv": True}


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
    """The authored lane binding, load shape and addresses survive every mode."""
    region, out_store = _vec_load_region(span, mode, data, out, full_shape=full_shape)
    assert isinstance(region.body, ir.SeqStmts)
    return list(region.body.stmts), out_store


def _explicit_region_program(stmts, params, return_types, *, name="split_explicit"):
    """A single InCore function whose body carries explicit SplitAivScopeStmt regions."""
    span = ir.Span.unknown()
    func = ir.Function(name, params, return_types, ir.SeqStmts(stmts, span), span, ir.FunctionType.InCore)
    return ir.Program([func], name, span)


def _expected_region_program(stmts, params, return_types, *, name="split_explicit", attrs=None):
    """Projected body golden: omit the retained wrapper and keep ``split_aiv``."""
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


def test_explicit_region_survives_lowering_and_is_consumed_by_expansion():
    """The structural placement carrier lives exactly through pass 24."""
    span = ir.Span.unknown()
    before = _notify_region_program(span, ir.SplitMode.NONE, in_region=True)
    lowered = _lower(before, keep_regions=True)
    assert "pl.split_aiv" in ir.python_print(lowered)
    assert "core_placement" not in ir.python_print(lowered)
    assert "split_aiv_region_validated" not in ir.python_print(lowered)
    with passes.PassContext([]):
        expanded = passes.expand_mixed_kernel()(lowered)
    assert "pl.split_aiv" not in ir.python_print(expanded)
    aic = [f for f in expanded.functions.values() if f.func_type == ir.FunctionType.AIC]
    aiv = [f for f in expanded.functions.values() if f.func_type == ir.FunctionType.AIV]
    assert aic and aiv
    assert "pld.system.notify" not in ir.python_print(aic[0])
    assert "pld.system.notify" in ir.python_print(aiv[0])


def test_explicit_region_body_is_lowered():
    """The region retains its authored lane index, load shape and store address."""
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


def test_none_region_keeps_tiles_full_and_binds_aiv_id():
    """A task-parallel (NONE) region is passed through FULL-width: the load is NOT
    halved, offsets are NOT localized, NO internal subblock_idx is injected, the
    author's aiv_id binding survives, and the function is stamped split_aiv.
    The wrapper is retained until expansion; _lower removes it for comparison."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    region, out_store = _vec_load_region(span, ir.SplitMode.NONE, data, out_0)
    program = _explicit_region_program(
        [region, ir.ReturnStmt([out_store], span)],
        [(data, _IN), (out_0, _OUT)],
        [out_0.type],
    )

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    stmts, e_out_store = _lowered_vec_load_region(span, ir.SplitMode.NONE, e_data, e_out)
    expected = _expected_region_program(
        [*stmts, ir.ReturnStmt([e_out_store], span)],
        [(e_data, _IN), (e_out, _OUT)],
        [e_out.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def _none_region_gather_stmts(span, data):
    """``aiv_id`` + a Vec load + an explicit ``tile.aic_gather(split=0)``.

    The V->C crossing spelled out in a task-parallel region: split=0 says "cross
    the boundary, do not split", so the gathered type keeps the FULL [128, 128]
    shape instead of doubling it. Returns ``(stmts, gather_var)``.
    """
    aiv_id_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_id_call.type, span)
    load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    gather = T.aic_gather(t, split=int(ir.SplitMode.NONE.value), span=span)
    g = ir.Var("g", gather.type, span)
    stmts: list[ir.Stmt] = [
        ir.AssignStmt(aiv_id, aiv_id_call, span),
        ir.AssignStmt(t, load, span),
        ir.AssignStmt(g, gather, span),
    ]
    return stmts, g


def test_none_region_admits_explicit_boundary_unchanged():
    """A boundary op inside a NONE region is ACCEPTED and passed through verbatim.

    Without a split axis the op keeps the one meaning that still applies — this
    value crosses the AIC/AIV boundary — and its split=0 deduction preserves the
    shape, so there is nothing for this pass to halve, re-join or re-localize.
    The scope wrapper is dropped and the body is spliced in unchanged, exactly as
    for a boundary-free NONE region; ExpandMixedKernel then folds the op into the
    split=0 tpush/tpop pair (see the end-to-end test at the bottom of this file).
    """
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    stmts, g = _none_region_gather_stmts(span, data)
    region = ir.SplitAivScopeStmt(split=ir.SplitMode.NONE, body=ir.SeqStmts(stmts, span), span=span)
    program = _explicit_region_program(
        [region, ir.ReturnStmt([g], span)],
        [(data, _IN), (out_0, _OUT)],
        [g.type],
    )

    e_data = ir.Var("data", _tensor([128, 128]), span)
    e_out = ir.Var("out_0", _tensor([128, 128]), span)
    e_stmts, e_g = _none_region_gather_stmts(span, e_data)
    expected = _expected_region_program(
        [*e_stmts, ir.ReturnStmt([e_g], span)],
        [(e_data, _IN), (e_out, _OUT)],
        [e_g.type],
    )
    ir.assert_structural_equal(_lower(program), expected)
    # The gathered tile is FULL, not doubled: the crossing is not a split.
    g_type = g.type
    assert isinstance(g_type, ir.TileType)
    assert g_type.shape == [128, 128]


def test_region_preserves_authored_lane_index():
    """A region needs no extra subblock binding beyond its authored lane ID."""
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


def test_region_and_surrounding_memory_shapes_are_preserved():
    """Neither in-region nor surrounding memory shapes are implicitly changed."""
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
    """Sibling regions retain their own modes and authored computation."""
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
    # aiv_shard (Acc) + single aiv_id binding survive. The function is stamped
    # split_aiv; _lower removes the retained wrapper for this body-only golden.
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
    retained without injecting a lane index. The _lower helper removes its wrapper
    for comparison; out-of-region compute survives and the function is stamped split_aiv."""
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


def test_sibling_regions_do_not_inject_subblock_indices():
    """Lowering leaves sibling lane bindings alone instead of minting extra indices."""
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

    # Each region already binds its authored lane ID; lowering adds none.
    assert injected == []


def test_mixed_explicit_implicit_region_splices_through():
    """An explicit shard does not cause unrelated full-width computation to be rewritten."""
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
    params = [(a_left, _IN), (b_right, _IN), (data, _IN), (out_0, _OUT)]
    program = _explicit_region_program(
        [ir.AssignStmt(qk, matmul, span), region, ir.ReturnStmt([out_store], span)],
        params,
        [out_0.type],
    )
    expected = _expected_region_program(
        [
            ir.AssignStmt(qk, matmul, span),
            ir.AssignStmt(aiv_id, aiv_id_call, span),
            ir.AssignStmt(qk_h, shard, span),
            ir.AssignStmt(full_t, full_load, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        params,
        [out_0.type],
    )
    # The full-width load survives verbatim: nothing was halved, nothing localized.
    ir.assert_structural_equal(_lower(program), expected)


# ---------------------------------------------------------------------------
# AUTO path + a surviving scope: rejected, not silently skipped.
#
# The halving walks recurse into for / while / if / seq but deliberately not
# into a ScopeStmt, so a mixed AUTO function whose body still holds a scope
# cannot be lowered. `RollupAffinity` used to have no ScopeStmt arm either, so
# such a function rolled up SHARED, `IsMixedCubeVector` read false, and the
# function was passed through COMPLETELY UNCHANGED — no split, no diagnostic.
# The rollup now classifies through the scope, and the pass rejects what it
# cannot lower, mirroring the explicit region path's guard.
# ---------------------------------------------------------------------------


def test_auto_path_scope_bodied_pure_vector_still_passes_through():
    """The guard fires only for MIXED — a pure-vector body is untouched, not rejected.

    Boundary partner to the rejection above, and the reason ``RollupAffinity``
    had to be taught to classify *through* a scope rather than the pass simply
    rejecting every scope-bodied split function: a pure-vector ``pl.split`` has
    no cube/vector boundary to converge, so it is legitimately passed through
    (ExpandMixedKernel strips its split later). Getting this wrong would turn a
    silent skip into a spurious hard error.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def pure_vec(
            self,
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
                out_store = pl.tile.store(t, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def pure_vec(
            self,
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
                out_store = pl.tile.store(t, [0, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_while_inside_region_preserves_vector_op():
    """A while body keeps its authored full-width load and zero store offset."""
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
    e_aiv = ir.Var("aiv_id", _IDX, span)
    off = [0, 0]
    e_load = T.load(e_data, off, [128, 128], target_memory=MS.Vec, span=span)
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
        [_get_subblock(e_aiv, span), e_while, ir.ReturnStmt([e_out_store], span)],
        [(e_data, _IN), (e_out, _OUT)],
        [e_out.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


def test_mixed_explicit_implicit_region_in_while_splices_through():
    """The same full-width op buried in a WhileStmt inside the region is likewise
    spliced through. The scan that used to recurse here ran only to reject; with
    the manual path unvalidated, control flow inside the region is simply carried
    along with the rest of the body."""
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
    params = [(a_left, _IN), (b_right, _IN), (data, _IN), (out_0, _OUT)]
    program = _explicit_region_program(
        [ir.AssignStmt(qk, matmul, span), region, ir.ReturnStmt([out_store], span)],
        params,
        [out_0.type],
    )
    expected = _expected_region_program(
        [
            ir.AssignStmt(qk, matmul, span),
            ir.AssignStmt(aiv_id, aiv_id_call, span),
            ir.AssignStmt(qk_h, shard, span),
            inner_while,
            ir.ReturnStmt([out_store], span),
        ],
        params,
        [out_0.type],
    )
    ir.assert_structural_equal(_lower(program), expected)


# ---------------------------------------------------------------------------
# Explicit-region admissions: values that are NOT derived from tile.aiv_shard but
# are still per-lane by construction. Two classes are admitted — pure generators
# (tile.full/create/ci/random) and address-carrying ops (tile.load/slice/extract)
# whose args reference the region's lane index. The rationale for each, and for
# why a generator is NOT added to half_tiles, lives at ScanSplitBody in
# src/ir/transforms/utils/split_axis_utils.cpp — keep it in one place.
#
# The explicit path splices the region body through UNCHANGED, so a positive
# test's Expected is literally its Before minus the scope wrapper. That identity
# is the property under test, so one helper builds both.
# ---------------------------------------------------------------------------


def _admission_program(span, body_fn, *, wrap, nest_in_loop=False, mode=ir.SplitMode.UP_DOWN):
    """matmul -> explicit region -> store, with ``body_fn`` supplying the middle.

    ``wrap=True`` nests the region statements in a SplitAivScopeStmt (the Before).
    ``wrap=False`` splices them flat and stamps the region attrs (the Expected) —
    the explicit path drops only the wrapper.

    ``body_fn(span, stmts, aiv_id, qk_h, data) -> VarPtr`` appends its statements
    and returns the tile to store. ``nest_in_loop`` puts the body + store inside a
    ForStmt so the lane-scalar dataflow must survive the walk's loop recursion.
    """
    a_left = ir.Var("a_left", _tile([128, 128], mem=MS.Left), span)
    b_right = ir.Var("b_right", _tile([128, 128], mem=MS.Right), span)
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)

    matmul = T.matmul(a_left, b_right, span=span)
    qk = ir.Var("qk", matmul.type, span)
    aiv_id = _sub_var("aiv_id")
    shard = T.aiv_shard(qk, split=1 if mode == ir.SplitMode.UP_DOWN else 2, span=span)
    qk_h = ir.Var("qk_h", shard.type, span)

    inner: list[ir.Stmt] = []
    stored = body_fn(span, inner, aiv_id, qk_h, data)
    store = T.store(stored, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    inner.append(ir.AssignStmt(out_store, store, span))

    region_stmts: list[ir.Stmt] = [_get_subblock(aiv_id, span), ir.AssignStmt(qk_h, shard, span)]
    if nest_in_loop:
        region_stmts.append(
            ir.ForStmt(
                ir.Var("i", _IDX, span),
                ir.ConstInt(0, DataType.INDEX, span),
                ir.ConstInt(2, DataType.INDEX, span),
                ir.ConstInt(1, DataType.INDEX, span),
                [],
                ir.SeqStmts(inner, span),
                [],
                span,
            )
        )
    else:
        region_stmts.extend(inner)

    params = [(a_left, _IN), (b_right, _IN), (data, _IN), (out_0, _OUT)]
    if wrap:
        region = ir.SplitAivScopeStmt(split=mode, body=ir.SeqStmts(region_stmts, span), span=span)
        return _explicit_region_program(
            [ir.AssignStmt(qk, matmul, span), region, ir.ReturnStmt([out_store], span)],
            params,
            [out_0.type],
        )
    return _expected_region_program(
        [ir.AssignStmt(qk, matmul, span), *region_stmts, ir.ReturnStmt([out_store], span)],
        params,
        [out_0.type],
    )


def _half_generator_body(span, stmts, aiv_id, qk_h, data):
    """``zeros = tile.full([64, 128])`` at the per-lane half extent, combined with
    the shard result."""
    zeros = T.full([64, 128], FP32, 0.0, span=span)
    z = ir.Var("zeros", zeros.type, span)
    relu = T.maximum(qk_h, z, span=span)
    r = ir.Var("relu", relu.type, span)
    stmts += [ir.AssignStmt(z, zeros, span), ir.AssignStmt(r, relu, span)]
    return r


def _lane_localized_load_body(span, stmts, aiv_id, qk_h, data):
    """A GM load the author localized with the region's own lane index:
    ``tile.load(data, [aiv_id * 64, 0], [64, 128])``."""
    off = aiv_id * 64
    o = ir.Var("row0", off.type, span)
    load = T.load(data, [o, 0], [64, 128], target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    relu = T.maximum(qk_h, t, span=span)
    r = ir.Var("relu", relu.type, span)
    stmts += [ir.AssignStmt(o, off, span), ir.AssignStmt(t, load, span), ir.AssignStmt(r, relu, span)]
    return r


def _lane_localized_slice_body(span, stmts, aiv_id, qk_h, data):
    """A tile.slice localized via its OFFSET arg (index 2): the source is a
    full-width Vec tile and each lane takes its own [64, 128] window."""
    full = T.full([128, 128], FP32, 1.0, span=span)
    f = ir.Var("full_t", full.type, span)
    off = aiv_id * 64
    o = ir.Var("row0", off.type, span)
    sl = T.slice(f, [64, 128], [o, 0], span=span)
    t = ir.Var("t", sl.type, span)
    relu = T.maximum(qk_h, t, span=span)
    r = ir.Var("relu", relu.type, span)
    stmts += [
        ir.AssignStmt(f, full, span),
        ir.AssignStmt(o, off, span),
        ir.AssignStmt(t, sl, span),
        ir.AssignStmt(r, relu, span),
    ]
    return r


def _lane_localized_extract_body(span, stmts, aiv_id, qk_h, data):
    """A tile.extract localized via its index_row arg (index 1). The Mat source is
    created in-region so it is a defined var (a free var could not be mapped by
    structural comparison); tile.create is a generator, so it stays NEUTRAL and
    the extract is admitted purely on its lane-referencing address arg."""
    src_call = T.create([128, 128], FP32, MS.Mat, span=span)
    src = ir.Var("src_mat", src_call.type, span)
    row = aiv_id * 64
    rv = ir.Var("row0", row.type, span)
    ex = T.extract(src, rv, 0, [64, 128], target_memory=MS.Vec, span=span)
    t = ir.Var("t", ex.type, span)
    relu = T.maximum(qk_h, t, span=span)
    r = ir.Var("relu", relu.type, span)
    stmts += [
        ir.AssignStmt(src, src_call, span),
        ir.AssignStmt(rv, row, span),
        ir.AssignStmt(t, ex, span),
        ir.AssignStmt(r, relu, span),
    ]
    return r


def _lane_localized_gather_row_body(span, stmts, aiv_id, qk_h, data):
    """A tile.gather_row localized via its SRC_OFFSET arg (index 3) — the scattered
    per-lane gather of issue #2244.

    The accumulator is a ``tile.full`` authored at the per-lane HALF extent, so it
    is a NEUTRAL generator (not in half_tiles) and the gather is admitted purely on
    its lane-referencing READ offset: each lane pulls a different GM row into its
    own UB half."""
    acc_call = T.full([64, 128], FP32, 0.0, span=span)
    acc = ir.Var("acc", acc_call.type, span)
    row = aiv_id * 64
    rv = ir.Var("src_row", row.type, span)
    gr = T.gather_row(acc, data, [0, 0], [rv, 0], [1, 128], span=span)
    t = ir.Var("t", gr.type, span)
    relu = T.maximum(qk_h, t, span=span)
    r = ir.Var("relu", relu.type, span)
    stmts += [
        ir.AssignStmt(acc, acc_call, span),
        ir.AssignStmt(rv, row, span),
        ir.AssignStmt(t, gr, span),
        ir.AssignStmt(r, relu, span),
    ]
    return r


def _gather_row_dst_only_localized_body(span, stmts, aiv_id, qk_h, data):
    """A tile.gather_row whose lane reference sits in DST_OFFSET (index 2) while
    SRC_OFFSET is lane-invariant: both lanes read the SAME GM row into different
    slots. Only the READ offset localizes a gather, so this must still be
    reported."""
    acc_call = T.full([64, 128], FP32, 0.0, span=span)
    acc = ir.Var("acc", acc_call.type, span)
    row = aiv_id * 64
    rv = ir.Var("dst_row", row.type, span)
    gr = T.gather_row(acc, data, [rv, 0], [0, 0], [1, 128], span=span)
    t = ir.Var("t", gr.type, span)
    stmts += [ir.AssignStmt(acc, acc_call, span), ir.AssignStmt(rv, row, span), ir.AssignStmt(t, gr, span)]
    return t


def _lane_ref_in_non_address_arg_body(span, stmts, aiv_id, qk_h, data):
    """A tile.load whose OFFSET is [0, 0] — both lanes read the same base rows —
    but which mentions aiv_id in its valid_shape. Scanning every arg instead of
    just the address args would wrongly admit this."""
    valid = aiv_id + 1
    v = ir.Var("valid", valid.type, span)
    load = T.load(data, [0, 0], [64, 128], valid_shape=[v, 128], target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    stmts += [ir.AssignStmt(v, valid, span), ir.AssignStmt(t, load, span)]
    return t


def _full_width_generator_body(span, stmts, aiv_id, qk_h, data):
    """``z = tile.full([128, 128])`` at FULL width, consumed by an op that takes
    nothing else — no shard lineage, no lane reference."""
    zeros = T.full([128, 128], FP32, 0.0, span=span)
    z = ir.Var("zeros", zeros.type, span)
    add = T.add(z, z, span=span)
    y = ir.Var("y", add.type, span)
    stmts += [ir.AssignStmt(z, zeros, span), ir.AssignStmt(y, add, span)]
    return y


def _laundering_body(span, stmts, aiv_id, qk_h, data):
    """``tile.set_validshape(full_width_tile, 1, aiv_id * 64)`` — a lane reference
    on a NON-addressing op, which must not launder the full tile in."""
    zeros = T.full([128, 128], FP32, 0.0, span=span)
    z = ir.Var("zeros", zeros.type, span)
    lane = aiv_id * 64
    ln = ir.Var("lane", lane.type, span)
    sv = T.set_validshape(z, 1, ln, span=span)
    s = ir.Var("sv", sv.type, span)
    add = T.add(s, s, span=span)
    y = ir.Var("y", add.type, span)
    stmts += [
        ir.AssignStmt(z, zeros, span),
        ir.AssignStmt(ln, lane, span),
        ir.AssignStmt(s, sv, span),
        ir.AssignStmt(y, add, span),
    ]
    return y


def _singleton_broadcast_body(span, stmts, aiv_id, qk_h, data):
    """Both lanes read the same broadcast row, then multiply their own shard."""
    load = T.load(data, [0, 0], [1, 128], target_memory=MS.Vec, span=span)
    scale = ir.Var("scale", load.type, span)
    mul = T.col_expand_mul(qk_h, scale, span=span)
    y = ir.Var("scaled", mul.type, span)
    stmts.extend([ir.AssignStmt(scale, load, span), ir.AssignStmt(y, mul, span)])
    return y


def test_explicit_region_admits_singleton_broadcast_load():
    """A singleton split axis is replicated, not an unlocalized full-width load."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _singleton_broadcast_body, wrap=True)),
        _admission_program(span, _singleton_broadcast_body, wrap=False),
    )


def _singleton_column_body(span, stmts, aiv_id, qk_h, data):
    """LEFT_RIGHT keeps each row's multiplier replicated across column shards."""
    load = T.load(data, [0, 0], [128, 1], target_memory=MS.Vec, span=span)
    scale = ir.Var("scale", load.type, span)
    mul = T.row_expand_mul(qk_h, scale, span=span)
    y = ir.Var("scaled", mul.type, span)
    stmts.extend([ir.AssignStmt(scale, load, span), ir.AssignStmt(y, mul, span)])
    return y


def test_explicit_region_admits_singleton_column_load():
    """LEFT_RIGHT admits the singleton column without halving it again."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _singleton_column_body, wrap=True, mode=ir.SplitMode.LEFT_RIGHT)),
        _admission_program(span, _singleton_column_body, wrap=False, mode=ir.SplitMode.LEFT_RIGHT),
    )


def _broadcast_expanded_to_full_width_body(span, stmts, aiv_id, qk_h, data):
    """A replicated row must not make a full-width expansion count as a shard."""
    load = T.load(data, [0, 0], [1, 128], target_memory=MS.Vec, span=span)
    scale = ir.Var("scale", load.type, span)
    full = T.full([128, 128], FP32, 0.0, span=span)
    target = ir.Var("target", full.type, span)
    expand = T.col_expand(target, scale, span=span)
    y = ir.Var("expanded", expand.type, span)
    stmts.extend(
        [
            ir.AssignStmt(scale, load, span),
            ir.AssignStmt(target, full, span),
            ir.AssignStmt(y, expand, span),
        ]
    )
    return y


def test_broadcast_expanded_to_full_width_splices_through():
    """An authored broadcast expansion remains full width in each lane."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _broadcast_expanded_to_full_width_body, wrap=True)),
        _admission_program(span, _broadcast_expanded_to_full_width_body, wrap=False),
    )


def test_region_admits_half_width_generator():
    """A pure generator authored at the per-lane half extent inside an explicit
    region is admitted and spliced through UNCHANGED — the pass rewrites nothing
    on the explicit path, so Expected is the Before minus the scope wrapper."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _half_generator_body, wrap=True)),
        _admission_program(span, _half_generator_body, wrap=False),
    )


def test_region_admits_lane_localized_load():
    """An address-carrying op whose offset references the region's lane index is
    per-lane by construction and is admitted, spliced through unchanged."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _lane_localized_load_body, wrap=True)),
        _admission_program(span, _lane_localized_load_body, wrap=False),
    )


def test_region_admits_lane_localized_load_nested_in_loop():
    """The lane-scalar dataflow survives the walk's LOOP recursion: ``aiv_id`` is
    bound at the region top level but the localized load sits inside a ForStmt, so
    the scan must carry the lane set into the loop body to admit it. This is the
    shape real kernels take (a per-lane load inside a cache-page loop)."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _lane_localized_load_body, wrap=True, nest_in_loop=True)),
        _admission_program(span, _lane_localized_load_body, wrap=False, nest_in_loop=True),
    )


def test_region_admits_lane_localized_slice():
    """tile.slice localized through its OFFSET arg (index 2) is admitted — the
    address-arg indices differ per op, so each addressing op needs its own case."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _lane_localized_slice_body, wrap=True)),
        _admission_program(span, _lane_localized_slice_body, wrap=False),
    )


def test_region_admits_lane_localized_extract():
    """tile.extract localized through its index_row arg (index 1) is admitted."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _lane_localized_extract_body, wrap=True)),
        _admission_program(span, _lane_localized_extract_body, wrap=False),
    )


def test_region_admits_lane_localized_gather_row():
    """tile.gather_row localized through its SRC_OFFSET arg (index 3) is admitted.

    This is the scattered per-lane gather of issue #2244: ``pl.gather_row`` is the
    only op that reads GM at an arbitrary RUNTIME offset, so a paged KV top-k list
    can only be sharded across the two AIV lanes this way. Before the op was added
    to ``AddressArgs`` its address was never consulted, and the pass's own fix (2)
    ("localize it with the region's lane index") was unreachable for it."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _lane_localized_gather_row_body, wrap=True)),
        _admission_program(span, _lane_localized_gather_row_body, wrap=False),
    )


def test_region_admits_lane_localized_gather_row_nested_in_loop():
    """The real shape: the per-row gather sits inside a ``pl.range`` loop while
    ``aiv_id`` is bound at the region head, so the lane-scalar set must survive the
    scan's loop recursion for the gather to be admitted."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _lane_localized_gather_row_body, wrap=True, nest_in_loop=True)),
        _admission_program(span, _lane_localized_gather_row_body, wrap=False, nest_in_loop=True),
    )


def test_region_splices_through_gather_row_localized_only_on_dst():
    """A gather is per-lane only when its READ offset is lane-derived; here the
    lane reference sits in ``dst_offset`` over a lane-invariant ``src_offset``, so
    both lanes fetch the SAME GM row into different slots. Real hazard, but on the
    manual path it is spliced through — the author owns it."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _gather_row_dst_only_localized_body, wrap=True)),
        _admission_program(span, _gather_row_dst_only_localized_body, wrap=False),
    )


def test_region_splices_through_lane_reference_outside_address_args():
    """A ``tile.load`` at offset [0, 0] that mentions aiv_id only in its
    valid_shape has BOTH lanes reading the same base rows. The manual path emits
    it unchanged rather than reporting it."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _lane_ref_in_non_address_arg_body, wrap=True)),
        _admission_program(span, _lane_ref_in_non_address_arg_body, wrap=False),
    )


def test_region_splices_through_consumer_of_full_width_generator():
    """A generator and its consumer retain the authored shape and store address."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _full_width_generator_body, wrap=True)),
        _admission_program(span, _full_width_generator_body, wrap=False),
    )


def test_region_splices_through_lane_reference_on_non_addressing_op():
    """``tile.set_validshape(full_width_tile, 1, aiv_id * 64)`` used to be the
    laundering case the admission scan had to refuse. With no scan on the manual
    path there is nothing to launder: the statement is emitted as written."""
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _laundering_body, wrap=True)),
        _admission_program(span, _laundering_body, wrap=False),
    )


# ---------------------------------------------------------------------------
# Scope-nested regions are rejected, not silently passed through.
#
# Region lowering walks for / while / if / seq but deliberately NOT ScopeStmt: a
# scope carries outlining and name-visibility semantics that region-local
# halving must not reach through. A region behind a scope therefore cannot be
# lowered, and the pass must reject it before claiming AivSplitLoweredValid.
#
# These are the only tests here authored in the ``@pl.program`` DSL, because the
# DSL is what produces the shape: an author-written ``with pl.at(...)`` inside a
# function declared ``pl.FunctionType.InCore``. OutlineIncoreScopes outlines
# scopes only out of Opaque / Orchestration functions, so that scope is still
# standing when this pass runs.
#
# The scope must be author-written. The parser emits a top-level region BARE in
# an InCore function — it has to, so that printing an outlined function and
# reparsing it rebuilds the same IR — so there is no synthesized wrapper to lean
# on. Whether a region is ALLOWED in such a function is a separate placement
# rule, enforced earlier by AivSplitValid check (h).
# ---------------------------------------------------------------------------

_SCOPE_NESTED_MSG = "nested inside a scope"


def test_scope_nested_region_rejected():
    """A region behind a ScopeStmt is rejected with an actionable diagnostic."""

    @pl.program
    class Nested:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            a: pl.Tensor[[128, 128], pl.FP32],
            c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            # Author-written scope. It must be explicit: the parser emits a
            # top-level region BARE inside an InCore function (so that printing
            # an outlined function and reparsing it rebuilds the same IR), so a
            # synthesized wrapper is no longer available to produce this shape.
            with pl.at(level=pl.Level.CORE_GROUP):
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    base = aiv_id * 64
                    c = pl.store(pl.exp(pl.load(a, [base, 0], [64, 128])), [base, 0], c)
            return c

    with pytest.raises(ValueError, match=_SCOPE_NESTED_MSG):
        _lower(Nested)


def test_scope_nested_region_guards_not_bypassed():
    """The guard fires even for a body that would trip a per-region check.

    Before the guard, this body's ``ValidateMixedExplicitRegion`` violation (a
    full-width ``tile.load`` alongside an explicit ``tile.aiv_shard``) went
    completely unchecked because the region was never visited.
    """

    @pl.program
    class NestedMixed:
        @pl.function(type=pl.FunctionType.InCore)
        def f(
            self,
            a_left: pl.Tile[[128, 128], pl.FP32, pl.MemorySpace.Left],
            b_right: pl.Tile[[128, 128], pl.FP32, pl.MemorySpace.Right],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                qk = pl.matmul(a_left, b_right)
                for _ in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    _qk_h = pl.aiv_shard(qk)  # explicit boundary
                    t = pl.load(data, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec)  # full width
                    out_0 = pl.tile.store(t, [0, 0], out_0)
            return out_0

    with pytest.raises(ValueError, match=_SCOPE_NESTED_MSG):
        _lower(NestedMixed)


def test_scope_inside_region_body_is_rejected():
    """The mirror case: a scope nested INSIDE a region body, not around it.

    The region wrapper is retained, but the inner walks (``LowerStmts`` / ``CheckNoCubeTileHalved`` /
    ``ScanSplitBody``) step over a ``ScopeStmt`` rather than entering it,
    and the vector ops inside would be spliced out FULL-WIDTH with both AIV lanes
    computing the whole tile, silently. Hand-built because the DSL cannot reach
    it: ``OutlineIncoreScopes`` lifts a ``with pl.at(...)`` inside a region into
    its own function, and AivSplitValid check (h) rejects authoring a region in
    an InCore function the outliner did not produce. This guards IR that skips
    pass 8 — hand-built, or a deserialized ``.pto``.
    """
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    aiv_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_call.type, span)
    load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    store = T.store(t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    inner = ir.SeqStmts([ir.AssignStmt(t, load, span), ir.AssignStmt(out_store, store, span)], span)
    region = ir.SplitAivScopeStmt(
        split=ir.SplitMode.UP_DOWN,
        count=2,
        body=ir.SeqStmts(
            [
                ir.AssignStmt(aiv_id, aiv_call, span),
                ir.InCoreScopeStmt(split=ir.SplitMode.NONE, name_hint="", body=inner, span=span),
            ],
            span,
        ),
        span=span,
    )
    # ``split_aiv`` set so AivSplitValid check (h) accepts the function — the
    # point is the pass guard, not the placement rule.
    func = ir.Function(
        "k",
        [(data, ir.ParamDirection.In), (out_0, ir.ParamDirection.Out)],
        [out_0.type],
        ir.SeqStmts([region], span),
        span,
        ir.FunctionType.InCore,
        attrs={"split_aiv": True},
    )

    with pytest.raises(ValueError, match=r"a scope survives inside a pl\.split_aiv region body"):
        _lower(ir.Program([func], "p", span))


def test_outlined_region_still_lowers_and_stamps():
    """The canonical Opaque form is unaffected: pass 7 outlines, pass 23 lowers.

    Guards the boundary of the rejection above — the scope must be gone by the
    time this pass runs, and when it is, the region lowers and the function is
    stamped as before.
    """

    @pl.program
    class Canonical:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.FP32],
            c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                base = aiv_id * 64
                c = pl.store(pl.exp(pl.load(a, [base, 0], [64, 128])), [base, 0], c)
            return c

    # ``c`` is a captured ``pl.Out`` tensor that the region rebinds under its own
    # name. OutlineIncoreScopes captures it as an ``InOut`` param (it used to be
    # left free, which is why this test could not reach the roundtrip instrument
    # at all); assert that here, since the outlined program is the input this
    # pass is under test on.
    with passes.PassContext([]):
        outlined = passes.outline_incore_scopes()(Canonical)
    outlined_incore = [f for f in outlined.functions.values() if f.func_type == ir.FunctionType.InCore]
    assert len(outlined_incore) == 1, "OutlineIncoreScopes should have produced one InCore function"
    assert [p.name_hint for p in outlined_incore[0].params] == ["a", "c"], (
        "the rebound pl.Out capture must be a parameter, not a free variable"
    )
    pl.parse_program(ir.python_print(outlined))  # outlined program round-trips

    # Uses ``_lower`` like every other test here, so the roundtrip instrument
    # guards this path too. It used to opt out: a ``mode=NONE`` region made
    # OutlineIncoreScopes stamp ``split=pl.SplitMode.NONE`` on the function, and
    # the parser drops an explicit NONE, so print->parse lost that attr and
    # structural equality reported "Kwargs size mismatch". The outliner no longer
    # stamps a NONE (absence is the canonical encoding of "no split"), so the
    # lowered output — region erased — now round-trips exactly.
    after = _lower(outlined)

    incore = [f for f in after.functions.values() if f.func_type == ir.FunctionType.InCore]
    assert len(incore) == 1, "OutlineIncoreScopes should have produced one InCore function"
    assert "pl.split_aiv" not in ir.python_print(after), "region must be erased"
    assert dict(incore[0].attrs) == _REGION_ATTRS, (
        f"expected exactly {_REGION_ATTRS}, got {dict(incore[0].attrs)}"
    )


# ---------------------------------------------------------------------------
# Retained lexical membership. Calls do not carry placement attributes; actual
# AIC/AIV routing is covered by ExpandMixedKernel and full-pipeline tests.
# ---------------------------------------------------------------------------


def _notify_region_program(span, mode, *, in_region, nest_in_loop=False):
    """matmul (cube, always out-of-region) + a region holding a vector load,
    with the notify placed either inside the region or after it."""
    a_left = ir.Var("a_left", _tile([128, 128], mem=MS.Left), span)
    b_right = ir.Var("b_right", _tile([128, 128], mem=MS.Right), span)
    data = ir.Var("data", _tensor([128, 128]), span)
    sig = ir.Var("sig", ir.DistributedTensorType([4, 4], DataType.INT32), span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)

    matmul = T.matmul(a_left, b_right, span=span)
    qk = ir.Var("qk", matmul.type, span)

    aiv_id = _sub_var("aiv_id")
    load = T.load(data, [0, 0], [128, 128], target_memory=MS.Vec, span=span)
    t = ir.Var("t", load.type, span)
    store = T.store(t, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    notify_stmt = ir.EvalStmt(_notify(span, sig, peer), span)
    if nest_in_loop:
        notify_stmt = ir.ForStmt(
            ir.Var("i", _IDX, span),
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(2, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [],
            ir.SeqStmts([notify_stmt], span),
            [],
            span,
        )

    region_stmts: list[ir.Stmt] = [
        _get_subblock(aiv_id, span),
        ir.AssignStmt(t, load, span),
        ir.AssignStmt(out_store, store, span),
    ]
    if in_region:
        region_stmts.append(notify_stmt)
    region = ir.SplitAivScopeStmt(split=mode, body=ir.SeqStmts(region_stmts, span), span=span)

    top: list[ir.Stmt] = [ir.AssignStmt(qk, matmul, span), region]
    if not in_region:
        top.append(notify_stmt)
    top.append(ir.ReturnStmt([out_store], span))
    return _explicit_region_program(
        top,
        [(a_left, _IN), (b_right, _IN), (data, _IN), (sig, _IN), (peer, _IN), (out_0, _OUT)],
        [out_0.type],
    )


_NOTIFY = ir.get_op("pld.system.notify").name


def test_none_region_retains_comm_op():
    """Task-parallel (NONE) retains its notify inside the region.

    This is the arm the mixed comm kernels use — ``pl.split_aiv(2,
    mode=pl.SplitMode.NONE)`` runs the full body on both AIV lanes — and the one
    the user docs tell authors to reach for.
    """
    span = ir.Span.unknown()
    program = _notify_region_program(span, ir.SplitMode.NONE, in_region=True)

    assert _placements(_lower(program, keep_regions=True), _NOTIFY) == ["aiv"]


def test_data_parallel_region_retains_comm_op():
    """A data-parallel region retains lexical placement of its communication calls."""
    span = ir.Span.unknown()
    program = _notify_region_program(span, ir.SplitMode.UP_DOWN, in_region=True)

    assert _placements(_lower(program, keep_regions=True), _NOTIFY) == ["aiv"]


def test_explicit_boundary_region_retains_comm_op():
    """Explicit boundaries and communication calls retain their lexical region."""
    span = ir.Span.unknown()
    a_left = ir.Var("a_left", _tile([128, 128], mem=MS.Left), span)
    b_right = ir.Var("b_right", _tile([128, 128], mem=MS.Right), span)
    sig = ir.Var("sig", ir.DistributedTensorType([4, 4], DataType.INT32), span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)

    matmul = T.matmul(a_left, b_right, span=span)
    qk = ir.Var("qk", matmul.type, span)
    aiv_id = _sub_var("aiv_id")
    shard = T.aiv_shard(qk, split=1, span=span)
    qk_h = ir.Var("qk_h", shard.type, span)
    sc = T.muls(qk_h, 2.0, span=span)
    sc_var = ir.Var("sc", sc.type, span)
    store = T.store(sc_var, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    region = ir.SplitAivScopeStmt(
        split=ir.SplitMode.UP_DOWN,
        body=ir.SeqStmts(
            [
                _get_subblock(aiv_id, span),
                ir.AssignStmt(qk_h, shard, span),
                ir.AssignStmt(sc_var, sc, span),
                ir.AssignStmt(out_store, store, span),
                ir.EvalStmt(_notify(span, sig, peer), span),
            ],
            span,
        ),
        span=span,
    )
    program = _explicit_region_program(
        [ir.AssignStmt(qk, matmul, span), region, ir.ReturnStmt([out_store], span)],
        [(a_left, _IN), (b_right, _IN), (sig, _IN), (peer, _IN), (out_0, _OUT)],
        [out_0.type],
    )

    after = _lower(program, keep_regions=True)
    assert _placements(after, _NOTIFY) == ["aiv"]
    # The boundary op itself is NOT stamped: it runs on BOTH lanes (tpush on the
    # cube side, tpop on the vector side), so "aiv" would be false of it.
    assert _placements(after, ir.get_op("tile.aiv_shard").name) == ["aiv"]


def test_region_retains_nested_comm_op():
    """The stamp walk descends into compound statements.

    A notify buried in a ForStmt inside the region is as region-placed as one at
    the region's top level; missing it would silently reintroduce the
    duplication bug for any loop-carried signal.
    """
    span = ir.Span.unknown()
    program = _notify_region_program(span, ir.SplitMode.NONE, in_region=True, nest_in_loop=True)

    assert _placements(_lower(program, keep_regions=True), _NOTIFY) == ["aiv"]


def test_outside_comm_op_stays_outside_region():
    """The negative: an out-of-region notify keeps no placement.

    The region is authoritative only for what it contains. Stamping outside it
    would claim a placement the author never made.
    """
    span = ir.Span.unknown()
    program = _notify_region_program(span, ir.SplitMode.NONE, in_region=False)

    assert _placements(_lower(program, keep_regions=True), _NOTIFY) == [None]


def test_region_membership_does_not_add_call_attributes():
    """Intrinsic compute stays in the region without placement attributes."""
    span = ir.Span.unknown()
    after = _lower(_notify_region_program(span, ir.SplitMode.NONE, in_region=True), keep_regions=True)

    assert _placements(after, ir.get_op("tile.load").name) == ["aiv"]
    assert _placements(after, ir.get_op("tile.store").name) == ["aiv"]
    assert _placements(after, ir.get_op("tile.get_subblock_idx").name) == ["aiv"]


# ---------------------------------------------------------------------------
# End to end: an explicit crossing out of a task-parallel (NONE) region.
#
# These run the WHOLE pipeline (``pl.jit(...).lower()``), not just this pass,
# because the fact under test spans three of them: this pass splices the region
# body through with the boundary op intact, ExpandMixedKernel folds that op into
# a tpush/tpop pair, and the split=0 stamp has to survive both. Property
# verification is on for the ride, so these double as the acceptance side of the
# AivSplitValid crossing checks (whose rejection side lives in
# ``test_verify_aiv_split.py``).
# ---------------------------------------------------------------------------

_E2E_M, _E2E_N, _E2E_K = 64, 64, 64


def _e2e_kernel_text(fn):
    """Lower ``fn`` for a2a3 (Ascend910) and return (aic_body_text, aiv_body_text).

    a2a3 rather than this file's ambient Ascend950 because 910B is the backend
    where a no-split crossing is dual-AIV dispatched, i.e. where the lane rule in
    the tests below actually applies. The conftest fixture pins the backend at
    setup and resets at teardown, so re-selecting it here is local to the test.

    Property verification stays ON — it is half of what these tests assert — but
    the ambient print->parse roundtrip instrument is dropped. A ``pl.jit`` kernel
    whose ``pl.at`` body opens a top-level ``pl.split_aiv`` does not survive
    print->parse today (the parser re-wraps a bare top-level region in an InCore
    ScopeStmt, so OutlineIncoreScopes' output reparses as InCoreScopeStmt !=
    SplitAivScopeStmt). That is a pre-existing parser/printer asymmetry, not
    something the crossing introduces: a region with no crossing at all fails the
    same way.
    """
    _backend.reset_for_testing()
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        program = fn.lower(config=RunConfig(platform="a2a3"))

    def body_of(func_type):
        matches = [f for f in program.functions.values() if f.func_type == func_type]
        assert len(matches) == 1, f"expected exactly one {func_type} function, got {len(matches)}"
        return ir.python_print(matches[0])

    return body_of(ir.FunctionType.AIC), body_of(ir.FunctionType.AIV)


def test_e2e_none_region_v2c_crossing_lowers_to_split_zero_transport():
    """``pl.aic_gather`` in a NONE region becomes a split=0 tpush/tpop pair.

    The vector lane pushes the FULL tile (no halving — there is no split axis)
    and the cube lane pops the same full tile. ``split=0`` on both ends is what
    tells the transport this is a no-split crossing.

    LANE RULE (documented, not enforced — docs/en/user/language/04-scopes.md):
    the ISA requires BOTH AIV sub-lanes to take part in a no-split handshake, and
    they share one slot with no per-lane offset, so lane 0's push is the one the
    cube reads. Making lane 1 contribute nothing when the two lanes hold
    different data is the author's job (guard the divergent work, or gather a
    value both lanes agree on); the compiler does not synthesize it here.
    """

    @pl.jit
    def none_v2c(
        a: pl.Tensor[[_E2E_M, _E2E_K], pl.FP16],
        q: pl.Tensor[[_E2E_M, _E2E_K], pl.FP16],
        out: pl.Out[pl.Tensor[[_E2E_M, _E2E_N], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="none_v2c", allow_early_resolve=True):
            for aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):  # noqa: B007 - task-parallel
                v = pl.aic_gather(pl.exp(a))
            out[0:_E2E_M, 0:_E2E_N] = pl.matmul(q, v, out_dtype=pl.FP32)
        return out

    aic, aiv = _e2e_kernel_text(none_v2c)
    assert '"tile.tpush_to_aic"' in aiv and '"split": 0' in aiv
    assert "pl.tile.tpop_from_aiv(split=0)" in aic
    # Full width on both ends: the crossing preserved the shape.
    assert "pl.Tile[[64, 64], pl.FP16" in aiv
    assert "pl.Tile[[64, 64], pl.FP16" in aic
    assert "pl.tile.aic_gather" not in aic and "pl.tile.aic_gather" not in aiv


def test_e2e_none_region_c2v_crossing_lowers_to_split_zero_transport():
    """Mirror of the V->C case: ``pl.aiv_shard`` in a NONE region, cube -> vector.

    ``pl.cross_core_slot`` sizes the c2v ring down; the default 8-slot depth of a
    full-width FP32 tile does not fit UB, which is a property of the shape rather
    than of the crossing.
    """

    @pl.jit
    def none_c2v(
        a: pl.Tensor[[_E2E_M, _E2E_K], pl.FP16],
        b: pl.Tensor[[_E2E_K, _E2E_N], pl.FP16],
        out: pl.Out[pl.Tensor[[_E2E_M, _E2E_N], pl.FP32]],
    ):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="none_c2v",
            allow_early_resolve=True,
            optimizations=[pl.cross_core_slot(slot_num=2)],
        ):
            mm = pl.matmul(a, b, out_dtype=pl.FP32)
            for aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):  # noqa: B007 - task-parallel
                out[0:_E2E_M, 0:_E2E_N] = pl.exp(pl.aiv_shard(mm))
        return out

    aic, aiv = _e2e_kernel_text(none_c2v)
    assert '"tile.tpush_to_aiv"' in aic and '"split": 0' in aic
    assert "pl.tile.tpop_from_aic(split=0)" in aiv
    assert "pl.Tile[[64, 64], pl.FP32" in aiv
    assert "pl.tile.aiv_shard" not in aic and "pl.tile.aiv_shard" not in aiv


def _singleton_arithmetic_body(span, stmts, aiv_id, qk_h, data):
    load = T.load(data, [0, 0], [1, 128], target_memory=MS.Vec, span=span)
    scale = ir.Var("scale", load.type, span)
    add = T.add(scale, scale, span=span)
    doubled = ir.Var("doubled", add.type, span)
    mul = T.col_expand_mul(qk_h, doubled, span=span)
    y = ir.Var("scaled", mul.type, span)
    stmts.extend(
        [
            ir.AssignStmt(scale, load, span),
            ir.AssignStmt(doubled, add, span),
            ir.AssignStmt(y, mul, span),
        ]
    )
    return y


def test_singleton_arithmetic_stays_replicated():
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _singleton_arithmetic_body, wrap=True)),
        _admission_program(span, _singleton_arithmetic_body, wrap=False),
    )


def test_legacy_split_with_private_vec_scratch_requires_explicit_boundaries():
    """Private replicated scratch does not opt a mixed kernel back into AUTO splitting."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            qk: pl.Tile[[16, 64], pl.FP32, pl.Mem.Mat],
            scales: pl.Tensor[[1, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
            scale = pl.tile.load(scales, [0, 0], [1, 64], target_memory=pl.Mem.Vec)
            scratch = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            inverse_scale = pl.tile.rsqrt(scale, tmp=scratch, high_precision=True)
            scaled = pl.tile.mul(popped, inverse_scale)
            return pl.tile.store(scaled, [0, 0], out)

    with pytest.raises(ValueError, match="Implicit AIV splitting is no longer supported"):
        _lower(Before)


def _alias_projection_body(span, stmts, aiv_id, qk_h, data):
    alias = ir.Var("alias", qk_h.type, span)
    pair = ir.MakeTuple([alias, alias], span)
    pair_var = ir.Var("pair", pair.type, span)
    item = ir.TupleGetItemExpr(pair_var, 1, span)
    add = T.add(item, item, span=span)
    result = ir.Var("result", add.type, span)
    stmts.extend(
        [
            ir.AssignStmt(alias, qk_h, span),
            ir.AssignStmt(pair_var, pair, span),
            ir.AssignStmt(result, add, span),
        ]
    )
    return result


def test_manual_alias_and_tuple_projection_preserve_shard_facts():
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _alias_projection_body, wrap=True)),
        _admission_program(span, _alias_projection_body, wrap=False),
    )


@pytest.mark.parametrize("callee_name", ["broadcast_helper", "tile.add"])
def test_manual_singleton_function_call_does_not_use_operator_effects(callee_name):
    """A function name, even one colliding with an op, has no registry entry."""
    span = ir.Span.unknown()

    def body(span, stmts, aiv_id, qk_h, data):
        call = ir.Call(ir.GlobalVar(callee_name), [qk_h], {}, _tile([1, 128], mem=MS.Vec), span)
        result = ir.Var("called", call.type, span)
        stmts.append(ir.AssignStmt(result, call, span))
        return qk_h

    # Isolate admission of a GlobalVar call; the external callee is deliberately
    # absent, so this fixture is not a self-contained printer/parser program.
    # If callee resolution is added, replace it with a real Inline callee.
    with passes.PassContext([]):
        lowered = passes.lower_auto_vector_split()(_admission_program(span, body, wrap=True))
    ir.assert_structural_equal(
        _EraseSplitRegions().visit_program(lowered), _admission_program(span, body, wrap=False)
    )


def _tuple_merge_body(span, stmts, aiv_id, qk_h, data, *, element=0):
    full = T.full([64, 128], DataType.FP32, 1.0, span=span)
    neutral = ir.Var("neutral", full.type, span)
    stmts.append(ir.AssignStmt(neutral, full, span))
    pair = ir.MakeTuple([qk_h, qk_h], span)
    other = ir.MakeTuple([qk_h, neutral], span)
    merged = ir.Var("merged", pair.type, span)
    stmts.append(
        ir.IfStmt(
            ir.ConstInt(1, DataType.BOOL, span),
            ir.YieldStmt([pair], span),
            ir.YieldStmt([other], span),
            [merged],
            span,
        )
    )
    item = ir.TupleGetItemExpr(merged, element, span)
    add = T.add(item, item, span=span)
    result = ir.Var("result", add.type, span)
    stmts.append(ir.AssignStmt(result, add, span))
    return result


def test_manual_tuple_merge_preserves_shard_facts():
    span = ir.Span.unknown()
    ir.assert_structural_equal(
        _lower(_admission_program(span, _tuple_merge_body, wrap=True)),
        _admission_program(span, _tuple_merge_body, wrap=False),
    )


def test_manual_tuple_merge_splices_through_a_half_merged_element():
    """Element 1 is sharded on one branch and neutral on the other, so the merge
    carries no shard fact. On the manual path that is emitted as written."""
    span = ir.Span.unknown()

    def body(span, stmts, aiv_id, qk_h, data):
        return _tuple_merge_body(span, stmts, aiv_id, qk_h, data, element=1)

    ir.assert_structural_equal(
        _lower(_admission_program(span, body, wrap=True)),
        _admission_program(span, body, wrap=False),
    )


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("initial_half,consume_exit", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("tuple_carry", [False, True])
def test_manual_loop_carry_shapes_splice_through(loop_kind, initial_half, consume_exit, tuple_carry):
    """Manual loop-carry shapes are preserved without inferring or propagating sharding."""
    span = ir.Span.unknown()

    def body(span, stmts, aiv_id, qk_h, data):
        full = T.full([64, 128], DataType.FP32, 1.0, span=span)
        neutral = ir.Var("neutral", full.type, span)
        stmts.append(ir.AssignStmt(neutral, full, span))
        initial, backedge = (qk_h, neutral) if initial_half else (neutral, qk_h)
        if tuple_carry:
            initial = ir.MakeTuple([initial, qk_h], span)
            backedge = ir.MakeTuple([backedge, qk_h], span)
        carry = ir.IterArg("carry", initial.type, initial, span)
        returned = ir.Var("returned", initial.type, span)
        carried = ir.TupleGetItemExpr(carry, 0, span) if tuple_carry else carry
        add = T.add(carried, carried, span=span)
        used = ir.Var("used", add.type, span)
        # A neutral initial value is not used in the body; the unsafe admission
        # occurs at the exit when zero iterations return that initial value.
        body_stmts: list[ir.Stmt] = [ir.AssignStmt(used, add, span)] if initial_half else []
        body_stmts.append(ir.YieldStmt([backedge], span))
        loop_body = ir.SeqStmts(body_stmts, span) if initial_half else body_stmts[0]
        if loop_kind == "for":
            loop = ir.ForStmt(
                ir.Var("i", _IDX, span),
                ir.ConstInt(0, DataType.INDEX, span),
                ir.ConstInt(2 if initial_half else 0, DataType.INDEX, span),
                ir.ConstInt(1, DataType.INDEX, span),
                [carry],
                loop_body,
                [returned],
                span,
            )
        else:
            loop = ir.WhileStmt(
                ir.ConstInt(int(initial_half), DataType.BOOL, span), [carry], loop_body, [returned], span
            )
        stmts.append(loop)
        if initial_half or not consume_exit:
            return qk_h
        exited = ir.TupleGetItemExpr(returned, 0, span) if tuple_carry else returned
        result_call = T.add(exited, exited, span=span)
        result = ir.Var("result", result_call.type, span)
        stmts.append(ir.AssignStmt(result, result_call, span))
        return result

    # Every combination now splices through: a lost carry shard fact and a
    # zero-iteration exit that cannot be classified are both authoring concerns,
    # and the manual path does not adjudicate them. The shard-fact machinery is
    # still exercised here — it must not CRASH on any of these shapes — it just
    # no longer decides whether the body compiles.
    ir.assert_structural_equal(
        _lower(_admission_program(span, body, wrap=True)),
        _admission_program(span, body, wrap=False),
    )


def test_nested_none_region_owns_its_transpose_mode():
    """The outer data-parallel region must not validate the child's full tile."""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([16, 16]), span)
    out = ir.Var("out", _tensor([16, 16]), span)
    load = T.load(data, [0, 0], [16, 16], target_memory=MS.Vec, span=span)
    value = ir.Var("value", load.type, span)
    transpose = T.transpose(value, 0, 1, span=span)
    transposed = ir.Var("transposed", transpose.type, span)
    store = T.store(transposed, [0, 0], out, span=span)
    stored = ir.Var("stored", store.type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(value, load, span),
            ir.AssignStmt(transposed, transpose, span),
            ir.AssignStmt(stored, store, span),
        ],
        span,
    )
    inner = ir.SplitAivScopeStmt(split=ir.SplitMode.NONE, body=body, span=span)
    outer = ir.SplitAivScopeStmt(split=ir.SplitMode.UP_DOWN, body=inner, span=span)
    program = _explicit_region_program(
        [outer, ir.ReturnStmt([stored], span)], [(data, _IN), (out, _OUT)], [out.type]
    )
    # Hand-built lowered input: nested source split loops are not DSL syntax.
    with passes.PassContext([make_roundtrip_instrument()]):
        expanded = passes.expand_mixed_kernel()(program)
    assert _split_region_count(expanded) == 0
    assert len(_placements(expanded, ir.get_op("tile.transpose").name)) == 1


# ---------------------------------------------------------------------------
# The whole-function transpose-split hazard.
#
# A tile.transpose that SWAPS the split axis migrates the per-lane data to the
# other dimension while the split still halves the original axis, so the result
# is mis-shaped and miscomputes. The split is a perf decision the user owns, so
# the pass fails loud rather than silently compiling it un-split.
#
# These tests lived in test_expand_mixed_kernel_a2a3.py while ExpandMixedKernel
# owned the check. It moved here because this is the earliest pass that sees
# every function carrying a function-level split mode -- including the
# pure-vector ones this pass declines to lower, which is the only reason the
# check ever had to run at pass 24. The per-REGION analogue is the AivSplitValid
# verifier's check (l), backstopped by ValidateTransposeSplitHazard in this pass
# (see test_transpose_hazard_per_region above).
#
# These use the full @pl.program Before form rather than this file's hand-built
# region fixtures: the subject is a function-level pl.split attr, which the DSL
# expresses directly.
# ---------------------------------------------------------------------------


def _run_to_expand_with_flatten(program: ir.Program) -> ir.Program:
    """The prefix above, carried on through ExpandMixedKernel.

    Only the singleton carve-out needs this: accepting the transpose is pass 23's
    answer, but "the split SURVIVES" is a fact about the expanded AIV function.
    """
    p = passes.convert_to_ssa()(program)
    p = passes.lower_composite_ops()(p)
    p = passes.flatten_tile_nd_to_2d()(p)
    p = passes.infer_tile_memory_space()(p)
    return passes.expand_mixed_kernel()(p)


def _run_to_lower_split_with_flatten(program: ir.Program) -> ir.Program:
    """The tile-pipeline prefix a tile.transpose needs to reach this pass.

    FlattenTileNdTo2D adds the transpose scratch arg that the TileOps2D verifier
    requires; InferTileMemorySpace resolves the spaces the affinity gate reads.
    """
    p = passes.convert_to_ssa()(program)
    p = passes.lower_composite_ops()(p)
    p = passes.flatten_tile_nd_to_2d()(p)
    p = passes.infer_tile_memory_space()(p)
    return passes.lower_auto_vector_split()(p)


def _assert_actionable_split_error(excinfo, mode_name: str) -> None:
    """The error must name the split mode and surface both fix directions so the
    user can act without reading the source: drop the split, or remove the
    transpose."""
    msg = str(excinfo.value)
    assert "swaps the split axis" in msg, msg
    assert mode_name in msg, msg
    assert "pl.SplitMode.NONE" in msg, msg  # direction 1: drop the split
    assert "column slice" in msg, msg  # direction 2: remove the transpose


def test_unsplittable_transpose_raises_actionable_error():
    """A requested UP_DOWN split is rejected with a ValueError when the kernel
    contains a tile.transpose that swaps the split axis.

    tile.transpose swaps axes, so the per-lane split data migrates to the other
    dim while SplitVectorKernel still halves the original split axis — it cannot
    type such a transpose correctly. The split is a perf decision the user owns,
    so the pass fails loud instead of silently compiling it un-split. Here the
    [16, 8] matmul result is transposed under UP_DOWN (split dim 0, non-singleton).
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def t_hazard(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 8], pl.BF16],
            out_0: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
        ) -> pl.Tensor[[8, 16], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 8], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(x_left, y_right)  # [16, 8] (cube result)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            zt = pl.transpose(z_vec, axis1=0, axis2=1)  # source dim0=16 non-singleton -> error
            out_0 = pl.store(zt, [0, 0], out_0)
            return out_0

    with pytest.raises(ValueError) as excinfo:
        _run_to_lower_split_with_flatten(Before)
    assert "Implicit AIV splitting is no longer supported" in str(excinfo.value)


def test_left_right_transpose_also_raises():
    """The error is mode-independent: a LEFT_RIGHT split whose transpose source is
    non-singleton on the split axis (dim 1) is also rejected, because the
    transpose migrates the column split axis just as UP_DOWN migrates rows."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def t_lr(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 16], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 16], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(x_left, y_right)  # [16, 16] (cube result)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            zt = pl.transpose(z_vec, axis1=0, axis2=1)  # source dim1=16 non-singleton -> error
            out_0 = pl.store(zt, [0, 0], out_0)
            return out_0

    with pytest.raises(ValueError) as excinfo:
        _run_to_lower_split_with_flatten(Before)
    assert "Implicit AIV splitting is no longer supported" in str(excinfo.value)


def test_unsplittable_int8_transpose_raises_actionable_error():
    """The error is dtype-independent: an int8 transpose that swaps a
    non-singleton split axis is rejected just like the fp/bf16 cases.

    A bf16 matmul (cube) keeps the kernel mixed; separately, an int8 [16, 32]
    tensor is loaded into Vec and transposed under UP_DOWN (source dim0=16
    non-singleton), so ExpandMixedKernel rejects the split.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def t_hazard_i8(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 8], pl.BF16],
            q: pl.Tensor[[16, 32], pl.INT8],
            out_0: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
            out_1: pl.Out[pl.Tensor[[32, 16], pl.INT8]],
        ) -> pl.Tensor[[16, 8], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 8], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z = pl.matmul(x_left, y_right)  # [16, 8] (cube result, keeps the kernel mixed)
            z_vec = pl.move(z, target_memory=pl.MemorySpace.Vec)
            out_0 = pl.store(z_vec, [0, 0], out_0)
            q_vec = pl.load(q, [0, 0], [16, 32], target_memory=pl.MemorySpace.Vec)
            qt = pl.transpose(q_vec, axis1=0, axis2=1)  # int8 source dim0=16 non-singleton -> error
            out_1 = pl.store(qt, [0, 0], out_1)
            return out_0

    with pytest.raises(ValueError) as excinfo:
        _run_to_lower_split_with_flatten(Before)
    assert "Implicit AIV splitting is no longer supported" in str(excinfo.value)


def test_pure_vector_split_transpose_also_raises():
    """A PURE-VECTOR pl.split kernel is checked too — the coverage the move adds.

    LowerAutoVectorSplit lowers only genuinely mixed (cube<->vector) functions; a
    pure-vector pl.split has no boundary to converge, so the pass passes it
    through untouched. That is exactly why this check used to sit at
    ExpandMixedKernel, which sees every function. Running it over the
    pass-through branch keeps that coverage at pass 23 — without it, this kernel
    would reach SplitVectorKernel with a transpose it cannot type and miscompile
    silently.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def t_pure_vec(
            self,
            q: pl.Tensor[[16, 32], pl.FP32],
            out_0: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ) -> pl.Tensor[[32, 16], pl.FP32]:
            q_vec = pl.load(q, [0, 0], [16, 32], target_memory=pl.MemorySpace.Vec)
            qt = pl.transpose(q_vec, axis1=0, axis2=1)  # source dim0=16 non-singleton -> error
            out_0 = pl.store(qt, [0, 0], out_0)
            return out_0

    with pytest.raises(ValueError) as excinfo:
        _run_to_lower_split_with_flatten(Before)
    _assert_actionable_split_error(excinfo, "UP_DOWN")


@pytest.mark.parametrize("mode", [pl.SplitMode.UP_DOWN, pl.SplitMode.LEFT_RIGHT])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("direct_expand", [False, True])
def test_function_level_auto_split_reports_explicit_migration(mode, nested, direct_expand):
    """Reject AUTO instead of silently retaining full-width work on both lanes."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": mode})
        def main(acc: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc]):
            if nested:
                for _ in pl.range(2):
                    vec = pl.tile.move(acc, target_memory=pl.Mem.Vec)
                    _result = pl.tile.add(vec, vec)
            else:
                vec_else = pl.tile.move(acc, target_memory=pl.Mem.Vec)
                _result_else = pl.tile.add(vec_else, vec_else)

    with pytest.raises(ValueError, match="Implicit AIV splitting is no longer supported") as error:
        if direct_expand:
            passes.expand_mixed_kernel()(Before)
        else:
            _lower(Before)
    assert "pl.aiv_shard" in str(error.value)
    assert "pl.aic_gather" in str(error.value)
    assert "pl.split_aiv" in str(error.value)


def test_explicit_region_guards_an_unassigned_store_of_an_empty_lane():
    """The explicit ``pl.aiv_shard`` walk guards a statement-form store like an assigned one.

    5 of 16 rows never reach lane 1, and a zero-row store is outside the ISA
    contract, so the store must run only where the lane holds data.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def split_region(
            qk: pl.Tile[[16, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                shard = pl.aiv_shard(qk)
                scaled = pl.tile.exp(shard)
                pl.tile.store(scaled, [aiv_id * 8, 0], out_0)
            return out_0

    class StoreGuards(ir.IRVisitor):
        guarded = 0
        unguarded = 0

        def visit_if_stmt(self, op):
            then_body = op.then_body
            if isinstance(then_body, ir.EvalStmt) and isinstance(then_body.expr, ir.Call):
                if then_body.expr.op.name == ir.get_op("tile.store").name:
                    self.guarded += 1
                    return
            super().visit_if_stmt(op)

        def visit_eval_stmt(self, op):
            if isinstance(op.expr, ir.Call) and op.expr.op.name == ir.get_op("tile.store").name:
                self.unguarded += 1
            super().visit_eval_stmt(op)

    with passes.PassContext([]):
        after = passes.lower_auto_vector_split()(Before)
    guards = StoreGuards()
    guards.visit_program(after)
    assert (guards.guarded, guards.unguarded) == (1, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
