# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Before/after tests for folding the explicit split-reshape ops in ExpandMixedKernel.

``tile.aiv_shard`` (C->V, full -> half) and ``tile.aic_gather`` (V->C, half ->
full) are recognised directly by ExpandMixedKernel as op-driven cross-core
boundaries and folded into the same tpush/tpop machinery used for a cross-C/V
``tile.move``. These tests hand-build a minimal InCore ``split_aiv`` function at
the post-InferTileMemorySpace level (memory spaces already assigned), run
``expand_mixed_kernel`` in isolation (verification disabled), and assert the
whole expanded program via ``ir.assert_structural_equal`` against a hand-authored
``Expected`` (the genuine, per-lane expanded form).

The Ascend950 backend (where the V->C direction needs an NZ fractal adapter) is
configured by the directory-level ``conftest.py``.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile_ops as T

MS = ir.MemorySpace
FP32 = DataType.FP32
_IN = ir.ParamDirection.In
_OUT = ir.ParamDirection.Out


def _tile(shape, view=None, mem=None):
    return ir.TileType(shape, FP32, None, view, mem)


def _expand(program):
    """Run ExpandMixedKernel under the ambient ``conftest`` verification context.

    The hand-built post-InferTileMemorySpace programs below satisfy both
    BEFORE_AND_AFTER property verification and the print->parse roundtrip, so
    there is nothing to suppress — the instruments back up the structural
    comparison each test already makes.
    """
    return passes.expand_mixed_kernel()(program)


def _assert_no_free_var(program):
    """Codegen guard: a free/dangling Var prints as a ``__FREE_VAR`` placeholder.

    ExpandMixedKernel splits a kernel into AIC/AIV lanes; a mis-routed value
    leaves a lane referencing an undefined Var, which the printer marks
    ``__FREE_VAR`` and which later crashes PTO emission. This property is not
    structurally expressible, so it is checked separately from the
    before/after structural comparison.
    """
    assert "__FREE_VAR" not in ir.python_print(program)


# ---------------------------------------------------------------------------
# CASE aiv_shard (C->V, full -> half)
# ---------------------------------------------------------------------------


def _build_aiv_shard_program():
    """qk[128,128]Vec --aiv_shard(split=1)--> half[64,128], consumed by a vector add."""
    span = ir.Span.unknown()
    qk = ir.Var("qk", _tile([128, 128], mem=MS.Vec), span)
    out_0 = ir.Var("out_0", ir.TensorType([64, 128], FP32), span)

    shard = T.aiv_shard(qk, split=1, span=span)
    assert isinstance(shard.type, ir.TileType)
    half = ir.Var("half", _tile(shard.type.shape, shard.type.tile_view, MS.Vec), span)
    add = T.add(half, half, span)
    assert isinstance(add.type, ir.TileType)
    y = ir.Var("y", _tile(add.type.shape, add.type.tile_view, MS.Vec), span)
    store = T.store(y, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(half, shard, span),
            ir.AssignStmt(y, add, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    func = ir.Function(
        "split_aiv",
        [(qk, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.UP_DOWN, "split_aiv": True},
    )
    return ir.Program([func], "test_aiv_shard", span), qk


def test_aiv_shard_folds_into_cube_to_vector_boundary():
    program, _ = _build_aiv_shard_program()
    after = _expand(program)

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.AIC,
            level=pl.Level.AIC,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv_aic(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            # AIC: pushes the FULL tile (the unadapted qk parameter), split == 1.
            split_aiv_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(
                name="split_aiv_c2v_slot_buffer", peer_func="split_aiv_aiv"
            )
            pl.system.aic_initialize_pipe(
                split_aiv_c2v_slot_buffer_import,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=65536,
                slot_num=2,
            )
            pl.tile.tpush_to_aiv(qk, split=1)

        @pl.function(
            type=pl.FunctionType.AIV,
            level=pl.Level.AIV,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv_aiv(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            split_aiv_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_c2v_slot_buffer", size=131072, base=-1
            )
            pl.system.aiv_initialize_pipe(
                split_aiv_c2v_slot_buffer, pl.const(0, pl.INT32), dir_mask=1, slot_size=65536, slot_num=2
            )
            # AIV: pops the HALF tile [64,128] in Vec (identity / non-NZ view), split == 1.
            half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=1)
            y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(half, half)
            pl.system.tfree_to_aic(half)
            out_store: pl.Tensor[[64, 128], pl.FP32] = pl.tile.store(y, [0, 0], out_0)
            return out_store

        @pl.function(
            type=pl.FunctionType.Group,
            level=pl.Level.CORE_GROUP,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            self.split_aiv_aic(qk, out_0)
            self.split_aiv_aiv(qk, out_0)
            return out_0

    ir.assert_structural_equal(after, Expected)
    _assert_no_free_var(after)


# ---------------------------------------------------------------------------
# CASE aic_gather (V->C, half -> full)
# ---------------------------------------------------------------------------


def _build_aic_gather_program():
    """half2[64,128]Vec --aic_gather(split=1)--> full[128,128], move->Left, matmul."""
    span = ir.Span.unknown()
    a = ir.Var("a", _tile([64, 128], mem=MS.Vec), span)
    b = ir.Var("b", _tile([128, 128], mem=MS.Right), span)
    out_0 = ir.Var("out_0", ir.TensorType([128, 128], FP32), span)

    add = T.add(a, a, span)
    assert isinstance(add.type, ir.TileType)
    half2 = ir.Var("half2", _tile(add.type.shape, add.type.tile_view, MS.Vec), span)
    gather = T.aic_gather(half2, split=1, span=span)
    assert isinstance(gather.type, ir.TileType)
    full = ir.Var("full", _tile(gather.type.shape, gather.type.tile_view, MS.Vec), span)
    move_left = T.move(full, MS.Left, span=span)
    assert isinstance(move_left.type, ir.TileType)
    full_left = ir.Var("full_left", _tile(move_left.type.shape, move_left.type.tile_view, MS.Left), span)
    matmul = T.matmul(full_left, b, span)
    assert isinstance(matmul.type, ir.TileType)
    z = ir.Var("z", _tile(matmul.type.shape, matmul.type.tile_view, MS.Acc), span)
    move_vec = T.move(z, MS.Vec, span=span)
    assert isinstance(move_vec.type, ir.TileType)
    z_vec = ir.Var("z_vec", _tile(move_vec.type.shape, move_vec.type.tile_view, MS.Vec), span)
    store = T.store(z_vec, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(half2, add, span),
            ir.AssignStmt(full, gather, span),
            ir.AssignStmt(full_left, move_left, span),
            ir.AssignStmt(z, matmul, span),
            ir.AssignStmt(z_vec, move_vec, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    func = ir.Function(
        "split_aiv",
        [(a, _IN), (b, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.UP_DOWN, "split_aiv": True},
    )
    return ir.Program([func], "test_aic_gather", span), half2


# ---------------------------------------------------------------------------
# The pto-isa split CODE the folded transport carries
# ---------------------------------------------------------------------------


def _build_shard_program(box_rows, valid_rows, lane_stride=None):
    """The same C->V shard over a [box_rows, 128] cube tile valid to `valid_rows`.

    Only the split code is under test here, so the body stops at the shard: its
    result is stored straight out, at the per-lane box the deducer produced.
    `lane_stride` is the partition stride LowerAutoVectorSplit stamps when it
    rebalances a ragged boundary; None leaves the default box partition.
    """
    span = ir.Span.unknown()
    view = ir.TileView(
        valid_shape=[
            ir.ConstInt(valid_rows, DataType.INDEX, span),
            ir.ConstInt(128, DataType.INDEX, span),
        ]
    )
    qk = ir.Var("qk", _tile([box_rows, 128], view, MS.Vec), span)
    half_rows = (box_rows + 1) // 2
    out_0 = ir.Var("out_0", ir.TensorType([half_rows, 128], FP32), span)

    shard_kwargs = {} if lane_stride is None else {"lane_stride": lane_stride}
    shard = T.aiv_shard(qk, split=1, span=span, **shard_kwargs)
    assert isinstance(shard.type, ir.TileType)
    half = ir.Var("half", _tile(shard.type.shape, shard.type.tile_view, MS.Vec), span)
    store = T.store(half, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(half, shard, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    func = ir.Function(
        "split_aiv",
        [(qk, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.UP_DOWN, "split_aiv": True},
    )
    return ir.Program([func], "test_shard_split_code", span)


@pytest.mark.parametrize(
    ("box_rows", "valid_rows", "lane_stride", "expected_code", "why"),
    [
        (128, 128, None, 1, "both lanes hold 64 rows -> the even code"),
        (128, 127, None, 3, "lanes hold 64 and 63 -> TILE_UP_DOWN_ODD"),
        (127, 127, None, 3, "an odd BOX: lanes hold 64 and 63 -> TILE_UP_DOWN_ODD"),
        (128, 40, None, 1, "lane 1 is empty, so its band is never read -> the even code"),
        (16, 13, 7, 3, "a rebalanced ragged boundary: lanes hold 7 and 6 -> TILE_UP_DOWN_ODD"),
        (16, 12, 6, 1, "a rebalanced EVEN valid region: lanes hold 6 and 6 -> the even code"),
    ],
)
def test_folded_transport_carries_the_pto_isa_split_code(
    box_rows, valid_rows, lane_stride, expected_code, why
):
    """The transport code states how the two AIV lanes' RUNTIME extents relate.

    pto-isa builds lane 1's band inside the FIFO slot from the popped tile's own
    valid extent — at ``e1`` cells for TILE_UP_DOWN / TILE_LEFT_RIGHT, one past
    it for the _ODD modes. The boundary op only carries the authored mode, so
    ExpandMixedKernel is where the code is chosen (split_axis::ShardSplitCode).
    """
    printed = ir.python_print(_expand(_build_shard_program(box_rows, valid_rows, lane_stride)))

    # The stride rides onto the transport too, so the torch reference runtime
    # cuts the lanes where the compiler did.
    stride_attr = "" if lane_stride is None else f", lane_stride={lane_stride}"
    assert f"pl.tile.tpush_to_aiv(qk, split={expected_code}{stride_attr})" in printed, why
    assert f"pl.tile.tpop_from_aic(split={expected_code}{stride_attr})" in printed, why


def test_folded_transport_carries_the_left_right_odd_code():
    """The dim-1 mirror: an odd COLUMN axis takes TILE_LEFT_RIGHT_ODD (code 4)."""
    span = ir.Span.unknown()
    qk = ir.Var("qk", _tile([128, 15], None, MS.Vec), span)
    out_0 = ir.Var("out_0", ir.TensorType([128, 8], FP32), span)

    shard = T.aiv_shard(qk, split=2, span=span)
    assert isinstance(shard.type, ir.TileType)
    half = ir.Var("half", _tile(shard.type.shape, shard.type.tile_view, MS.Vec), span)
    store = T.store(half, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(half, shard, span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    func = ir.Function(
        "split_aiv",
        [(qk, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.LEFT_RIGHT, "split_aiv": True},
    )
    printed = ir.python_print(_expand(ir.Program([func], "test_lr_split_code", span)))

    assert "pl.tile.tpush_to_aiv(qk, split=4)" in printed, printed
    assert "pl.tile.tpop_from_aic(split=4)" in printed, printed


def test_folded_transport_rejects_lane_extents_pto_isa_cannot_place():
    """Lanes more than one cell apart have no expressible band pair.

    (A RUNTIME split-axis valid extent keeps the even code instead of being
    rejected — see the comment on `ShardSplitCode` and pto-isa#263. That path is
    covered on device by tests/st/runtime/cross_core/test_cross_core_split_parity.py,
    not here: a hand-built program with a dynamic extent does not satisfy this
    file's ambient roundtrip instrument.)

    A 100-of-128 extent leaves lane 0 with 64 rows and lane 1 with 36: neither
    ``e1`` (36) nor ``e1 + 1`` (37) is row 64, so every code would pop the wrong
    rows. Report it, naming the extents that would work.
    """
    with pytest.raises(ValueError, match="leaves the two AIV lanes 64 and 36 cells"):
        _expand(_build_shard_program(128, 100))


def test_folded_transport_stays_on_the_box_partition_without_a_stride():
    """Without the attr the same tile keeps the box partition — and is rejected.

    13 of a 16-row box gives 8 and 5 on the box partition; only the rebalanced
    form above is placeable, so the attr is what distinguishes the two.
    """
    with pytest.raises(ValueError, match="leaves the two AIV lanes 8 and 5 cells"):
        _expand(_build_shard_program(16, 13))


def test_aic_gather_folds_into_vector_to_cube_boundary():
    program, _ = _build_aic_gather_program()
    after = _expand(program)

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.AIC,
            level=pl.Level.AIC,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv_aic(
            self,
            a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec],
            b: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ):
            split_aiv_v2c_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_v2c_slot_buffer", size=131072, base=-1
            )
            split_aiv_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(
                name="split_aiv_c2v_slot_buffer", peer_func="split_aiv_aiv"
            )
            pl.system.aic_initialize_pipe(
                split_aiv_c2v_slot_buffer_import,
                split_aiv_v2c_slot_buffer,
                dir_mask=3,
                slot_size=65536,
                slot_num=2,
            )
            # AIC: V->C pop yields the FULL tile [128,128] in Mat. The Mat default
            # effective tile_view is NZ (col_major), so no explicit view is needed.
            full: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat] = pl.tile.tpop_from_aiv(split=1)
            # The original follow-on move(full -> Left) survives on the AIC lane
            # (it is NOT re-detected as a second cross-core boundary).
            full_left: pl.Tile[[128, 128], pl.FP32, pl.Mem.Left] = pl.tile.move(
                full, target_memory=pl.Mem.Left
            )
            pl.system.tfree_to_aiv(full)
            z: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(full_left, b)
            pl.tile.tpush_to_aiv(z, split=0)

        @pl.function(
            type=pl.FunctionType.AIV,
            level=pl.Level.AIV,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv_aiv(
            self,
            a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec],
            b: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            split_aiv_v2c_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(
                name="split_aiv_v2c_slot_buffer", peer_func="split_aiv_aic"
            )
            split_aiv_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_c2v_slot_buffer", size=131072, base=-1
            )
            pl.system.aiv_initialize_pipe(
                split_aiv_c2v_slot_buffer,
                split_aiv_v2c_slot_buffer_import,
                dir_mask=3,
                slot_size=65536,
                slot_num=2,
            )
            half2: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(a, a)
            # Push-side fractal adapter: move the HALF [64,128] into Vec with an
            # explicit NZ (col_major) layout, then push it to the cube FIFO.
            half2_nz: pl.Tile[
                [64, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
            ] = pl.tile.move(
                half2,
                target_memory=pl.Mem.Vec,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            pl.tile.tpush_to_aic(half2_nz, split=1)
            z_vec: pl.Tile[
                [128, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
            ] = pl.tile.tpop_from_aic(split=0)
            out_store: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(z_vec, [0, 0], out_0)
            pl.system.tfree_to_aic(z_vec)
            return out_store

        @pl.function(
            type=pl.FunctionType.Group,
            level=pl.Level.CORE_GROUP,
            role=pl.Role.SubWorker,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_aiv(
            self,
            a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec],
            b: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            self.split_aiv_aic(a, b, out_0)
            self.split_aiv_aiv(a, b, out_0)
            return out_0

    ir.assert_structural_equal(after, Expected)
    _assert_no_free_var(after)


# ---------------------------------------------------------------------------
# Region placement: core_placement="aiv" keeps a comm op off the cube lane
#
# A mixed kernel is split into an AIC and an AIV function, and a statement whose
# affinity is SHARED is copied onto BOTH. ``pld.system.notify`` is SHARED — its
# TNOTIFY is core-agnostic by ISA, so declaring a core affinity for it would be
# a false claim — and its atomic-add form is NOT idempotent, so the copy makes
# the waiting rank observe the signal twice.
#
# ``pl.split_aiv`` is how the author says "this runs on the vector lane".
# LowerAutoVectorSplit erases the region node, so it leaves the placement behind
# as ``attrs["core_placement"] = "aiv"`` on the calls whose lane the region
# decides, and ClassifyCallAffinity resolves that to VECTOR. These tests pin the
# consuming end: the stamp, and only the stamp, is what keeps the notify off the
# cube lane. The pass ALSO strips the stamp once it has consumed it, so it never
# reaches a downstream pass or a printed dump.
# ---------------------------------------------------------------------------

_PLACEMENT_ATTRS = {"core_placement": "aiv"}


def _notify_call(span, sig, peer, *, placed: bool) -> ir.Call:
    """A ``pld.system.notify`` (AtomicAdd), optionally region-placed on AIV.

    ``placed=False`` is the identical call without the stamp, so a pair of these
    isolates the stamp as the only difference between the two programs.
    """
    zero = ir.ConstInt(0, DataType.INDEX, span)
    offsets = ir.MakeTuple([zero, zero], span)
    value = ir.ConstInt(1, DataType.INT32, span)
    call = ir.create_op_call("pld.system.notify", [sig, peer, offsets, value], {"op": 0}, span)
    if not placed:
        return call
    return ir.Call(call.op, call.args, call.kwargs, _PLACEMENT_ATTRS, call.type, call.span)


def _build_notify_program(*, placed: bool):
    """The aiv_shard kernel above (mixed: cube push + vector half) plus a notify.

    The shard is what makes the function mixed, so ExpandMixedKernel really does
    split it into a pair — the precondition for anything being duplicated.
    """
    span = ir.Span.unknown()
    qk = ir.Var("qk", _tile([128, 128], mem=MS.Vec), span)
    sig = ir.Var("sig", ir.DistributedTensorType([4, 4], DataType.INT32), span)
    peer = ir.Var("peer", ir.ScalarType(DataType.INT32), span)
    out_0 = ir.Var("out_0", ir.TensorType([64, 128], FP32), span)

    shard = T.aiv_shard(qk, split=1, span=span)
    assert isinstance(shard.type, ir.TileType)
    half = ir.Var("half", _tile(shard.type.shape, shard.type.tile_view, MS.Vec), span)
    add = T.add(half, half, span)
    assert isinstance(add.type, ir.TileType)
    y = ir.Var("y", _tile(add.type.shape, add.type.tile_view, MS.Vec), span)
    store = T.store(y, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(half, shard, span),
            ir.AssignStmt(y, add, span),
            ir.EvalStmt(_notify_call(span, sig, peer, placed=placed), span),
            ir.AssignStmt(out_store, store, span),
            ir.ReturnStmt([out_store], span),
        ],
        span,
    )
    func = ir.Function(
        "split_aiv",
        [(qk, _IN), (sig, _IN), (peer, _IN), (out_0, _OUT)],
        [out_0.type],
        body,
        span,
        ir.FunctionType.InCore,
        attrs={"split": ir.SplitMode.UP_DOWN, "split_aiv": True},
    )
    return ir.Program([func], "test_notify_placement", span)


def _count_op_calls(func, op_name: str) -> int:
    """How many ``Call``s to ``op_name`` appear anywhere in ``func``'s body."""
    seen = 0

    def walk(node):
        nonlocal seen
        if node is None:
            return
        if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == op_name:
            seen += 1
        if isinstance(node, ir.SeqStmts):
            for stmt in node.stmts:
                walk(stmt)
            return
        if isinstance(node, ir.AssignStmt):
            walk(node.value)
        if isinstance(node, ir.EvalStmt):
            walk(node.expr)
        walk(getattr(node, "body", None))

    walk(func.body)
    return seen


def _notify_counts_by_lane(after) -> dict[str, int]:
    """notify count per expanded function, keyed by function name."""
    op_name = ir.get_op("pld.system.notify").name
    return {func.name: _count_op_calls(func, op_name) for func in after.functions.values()}


def test_region_placed_notify_lands_on_aiv_lane_only():
    """A region-placed notify is emitted on the AIV lane and NOT on the AIC one.

    This is the fix for the double-signal bug: one notify survives the split, on
    the vector lane the author chose with ``pl.split_aiv``.
    """
    after = _expand(_build_notify_program(placed=True))
    counts = _notify_counts_by_lane(after)

    assert counts["split_aiv_aiv"] == 1
    assert counts["split_aiv_aic"] == 0
    # Exactly one across the whole program — the property that actually matters,
    # stated independently of which lane won.
    assert sum(counts.values()) == 1
    _assert_no_free_var(after)


def test_unplaced_notify_is_duplicated_onto_both_lanes():
    """The negative: WITHOUT the stamp the same notify is copied onto both lanes.

    This is the reported bug, pinned here so the fix above cannot be mistaken
    for something the pass already did. Nothing rejects the unplaced form —
    putting the comm phase in a region is the author's job, documented rather
    than enforced.
    """
    after = _expand(_build_notify_program(placed=False))
    counts = _notify_counts_by_lane(after)

    assert counts["split_aiv_aiv"] == 1
    assert counts["split_aiv_aic"] == 1
    assert sum(counts.values()) == 2


def test_placement_stamp_is_stripped_after_expansion():
    """The stamp's lifetime ends here: no expanded function still carries it.

    ``core_placement`` exists only to bridge the region erasure in pass 20 to
    the affinity roll-up in pass 21. Leaving it behind would put a defunct
    region marker into every later pass dump, the print -> parse round-trip and
    every downstream structural comparison.
    """
    after = _expand(_build_notify_program(placed=True))

    assert "core_placement" not in ir.python_print(after)


def test_placement_stamp_does_not_move_vector_compute():
    """The stamp changes the notify's lane and nothing else.

    Guards against the override being applied too broadly: the stamped and
    unstamped programs must expand identically apart from where the notify
    lands, so the vector add stays on AIV and the cross-core boundary keeps its
    tpush on AIC and tpop on AIV in both.
    """
    placed = _expand(_build_notify_program(placed=True))
    unplaced = _expand(_build_notify_program(placed=False))

    add_name = ir.get_op("tile.add").name
    push_name = ir.get_op("tile.tpush_to_aiv").name
    pop_name = ir.get_op("tile.tpop_from_aic").name
    for after in (placed, unplaced):
        funcs = {func.name: func for func in after.functions.values()}
        assert _count_op_calls(funcs["split_aiv_aiv"], add_name) == 1
        assert _count_op_calls(funcs["split_aiv_aic"], add_name) == 0
        assert _count_op_calls(funcs["split_aiv_aic"], push_name) == 1
        assert _count_op_calls(funcs["split_aiv_aiv"], pop_name) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
