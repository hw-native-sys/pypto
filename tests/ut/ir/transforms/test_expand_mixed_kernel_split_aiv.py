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
import pypto.language.distributed as pld
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


@pl.program
class _AivShardBefore:
    """qk[128,128]Vec --aiv_shard(split=1)--> half[64,128], consumed by a vector add."""

    @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
    def split_aiv(
        self,
        qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
        out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ) -> pl.Tensor[[64, 128], pl.FP32]:
        half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.aiv_shard(qk, split=1)
        y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(half, half)
        out_store = pl.tile.store(y, [0, 0], out_0)
        return out_store


def test_aiv_shard_folds_into_cube_to_vector_boundary():
    after = _expand(_AivShardBefore)

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
        attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
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
        attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
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

    @pl.program
    class Before:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True},
        )
        def split_aiv(
            self,
            qk: pl.Tile[[128, 15], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 8], pl.FP32]],
        ) -> pl.Tensor[[128, 8], pl.FP32]:
            half: pl.Tile[[128, 8], pl.FP32, pl.Mem.Vec] = pl.tile.aiv_shard(qk, split=2)
            out_store = pl.tile.store(half, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC)
        def split_aiv_aic(
            self,
            qk: pl.Tile[[128, 15], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 8], pl.FP32]],
        ):
            pl.func_attr({"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True})
            split_aiv_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(
                name="split_aiv_c2v_slot_buffer", peer_func="split_aiv_aiv"
            )
            pl.system.aic_initialize_pipe(
                split_aiv_c2v_slot_buffer_import,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=7680,
                slot_num=2,
            )
            # Code 4 == TILE_LEFT_RIGHT_ODD: the lanes hold 8 and 7 columns.
            pl.tile.tpush_to_aiv(qk, split=4)

        @pl.function(type=pl.FunctionType.AIV)
        def split_aiv_aiv(
            self,
            qk: pl.Tile[[128, 15], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 8], pl.FP32]],
        ) -> pl.Tensor[[128, 8], pl.FP32]:
            pl.func_attr({"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True})
            split_aiv_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_c2v_slot_buffer", size=15360, base=-1
            )
            pl.system.aiv_initialize_pipe(
                split_aiv_c2v_slot_buffer,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=7680,
                slot_num=2,
            )
            half: pl.Tile[[128, 8], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=4)
            out_store: pl.Tensor[[128, 8], pl.FP32] = pl.tile.store(half, [0, 0], out_0)
            pl.system.tfree_to_aic(half)
            return out_store

        @pl.function(type=pl.FunctionType.Group)
        def split_aiv(
            self,
            qk: pl.Tile[[128, 15], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 8], pl.FP32]],
        ) -> pl.Tensor[[128, 8], pl.FP32]:
            pl.func_attr({"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True})
            self.split_aiv_aic(qk, out_0)
            self.split_aiv_aiv(qk, out_0)
            return out_0

    ir.assert_structural_equal(_expand(Before), Expected)


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

# The two Before programs differ ONLY in the region placement stamp, so the pair
# of structural comparisons below isolates the stamp as the cause of every
# difference between their expanded forms.
#
# Note the two programs must not share a class name: `@pl.program` resolves the
# decorated class from source, so two same-named classes in one scope collapse
# to whichever appears first.


@pl.program
class _PlacedNotifyBefore:
    @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
    def split_aiv(
        self,
        qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
        sig: pld.DistributedTensor[[4, 4], pl.INT32],
        peer: pl.Scalar[pl.INT32],
        out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ) -> pl.Tensor[[64, 128], pl.FP32]:
        half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.aiv_shard(qk, split=1)
        y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(half, half)
        pld.system.notify(sig, peer, [0, 0], 1, op=pld.NotifyOp.AtomicAdd, attrs={"core_placement": "aiv"})
        out_store = pl.tile.store(y, [0, 0], out_0)
        return out_store


@pl.program
class _UnplacedNotifyBefore:
    @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
    def split_aiv(
        self,
        qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
        sig: pld.DistributedTensor[[4, 4], pl.INT32],
        peer: pl.Scalar[pl.INT32],
        out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ) -> pl.Tensor[[64, 128], pl.FP32]:
        half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.aiv_shard(qk, split=1)
        y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(half, half)
        pld.system.notify(sig, peer, [0, 0], 1, op=pld.NotifyOp.AtomicAdd)
        out_store = pl.tile.store(y, [0, 0], out_0)
        return out_store


def test_region_placed_notify_lands_on_aiv_lane_only():
    """A region-placed notify is emitted on the AIV lane and NOT on the AIC one.

    This is the fix for the double-signal bug: one notify survives the split, on
    the vector lane the author chose with ``pl.split_aiv``.

    ``Expected`` also pins the two facts that used to be separate tests: no
    expanded function still carries the ``core_placement`` stamp (its lifetime
    ends at this pass, and a leftover would show up as an attrs mismatch), and
    the stamp moved nothing but the notify — the vector add stays on AIV and the
    cross-core boundary keeps its tpush on AIC and its tpop on AIV, exactly as
    in the unplaced expansion below.
    """

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC)
        def split_aiv_aic(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            sig: pld.DistributedTensor[[4, 4], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            pl.func_attr({"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
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

        @pl.function(type=pl.FunctionType.AIV)
        def split_aiv_aiv(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            sig: pld.DistributedTensor[[4, 4], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            pl.func_attr({"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
            split_aiv_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_c2v_slot_buffer", size=131072, base=-1
            )
            pl.system.aiv_initialize_pipe(
                split_aiv_c2v_slot_buffer,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=65536,
                slot_num=2,
            )
            half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=1)
            y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(half, half)
            pl.system.tfree_to_aic(half)
            pld.system.notify(sig, peer, [0, 0], 1, op=pld.NotifyOp.AtomicAdd)
            out_store = pl.tile.store(y, [0, 0], out_0)
            return out_store

        @pl.function(type=pl.FunctionType.Group)
        def split_aiv(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            sig: pld.DistributedTensor[[4, 4], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            pl.func_attr({"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
            self.split_aiv_aic(qk, sig, peer, out_0)
            self.split_aiv_aiv(qk, sig, peer, out_0)
            return out_0

    after = _expand(_PlacedNotifyBefore)
    ir.assert_structural_equal(after, Expected)
    _assert_no_free_var(after)


def test_unplaced_notify_is_duplicated_onto_both_lanes():
    """The negative: WITHOUT the stamp the same notify is copied onto both lanes.

    This is the reported bug, pinned here so the fix above cannot be mistaken
    for something the pass already did. Nothing rejects the unplaced form —
    putting the comm phase in a region is the author's job, documented rather
    than enforced. Everything else is identical to the placed expansion.
    """

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC)
        def split_aiv_aic(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            sig: pld.DistributedTensor[[4, 4], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            pl.func_attr({"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
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
            # The duplicate: a SHARED call with no region to place it lands here too.
            pld.system.notify(sig, peer, [0, 0], 1, op=pld.NotifyOp.AtomicAdd)

        @pl.function(type=pl.FunctionType.AIV)
        def split_aiv_aiv(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            sig: pld.DistributedTensor[[4, 4], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            pl.func_attr({"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
            split_aiv_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(
                name="split_aiv_c2v_slot_buffer", size=131072, base=-1
            )
            pl.system.aiv_initialize_pipe(
                split_aiv_c2v_slot_buffer,
                pl.const(0, pl.INT32),
                dir_mask=1,
                slot_size=65536,
                slot_num=2,
            )
            half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=1)
            y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(half, half)
            pl.system.tfree_to_aic(half)
            pld.system.notify(sig, peer, [0, 0], 1, op=pld.NotifyOp.AtomicAdd)
            out_store = pl.tile.store(y, [0, 0], out_0)
            return out_store

        @pl.function(type=pl.FunctionType.Group)
        def split_aiv(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            sig: pld.DistributedTensor[[4, 4], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            out_0: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            pl.func_attr({"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
            self.split_aiv_aic(qk, sig, peer, out_0)
            self.split_aiv_aiv(qk, sig, peer, out_0)
            return out_0

    after = _expand(_UnplacedNotifyBefore)
    ir.assert_structural_equal(after, Expected)
    _assert_no_free_var(after)


# ---------------------------------------------------------------------------
# CASE per-lane valid_shape (the boundary type that references the lane index)
# ---------------------------------------------------------------------------


def _build_lane_valid_shard_program():
    """A C->V shard whose per-lane ``valid_shape`` is an expression over ``aiv_id``.

    This is what a ragged split axis looks like by the time it reaches this pass:
    ``LowerAutoVectorSplit`` repairs the deducer's lane-agnostic ceil(V/2) guess
    by writing the lane's true extent into the shard result's
    TileView, as an expression over the region's own
    ``aiv_id = tile.get_subblock_idx()`` binding. So the shard's TYPE — not just
    its operands — carries a reference to a body-local Var, and the tpop this
    pass builds from it inherits that reference.
    """
    span = ir.Span.unknown()
    idx = ir.ScalarType(DataType.INDEX)
    aiv_id = ir.Var("aiv_id", idx, span)

    qk = ir.Var("qk", _tile([128, 128], None, MS.Vec), span)
    out_0 = ir.Var("out_0", ir.TensorType([64, 128], FP32), span)

    shard = T.aiv_shard(qk, split=1, span=span)
    assert isinstance(shard.type, ir.TileType)
    # 64 rows on lane 0, 40 on lane 1 -> `64 - aiv_id * 24`, the shape
    # LocalizeValidDimForSplit produces for a partially-valid split axis.
    lane_rows = ir.Sub(
        ir.ConstInt(64, DataType.INDEX, span),
        ir.Mul(aiv_id, ir.ConstInt(24, DataType.INDEX, span), DataType.INDEX, span),
        DataType.INDEX,
        span,
    )
    lane_view = ir.TileView(valid_shape=[lane_rows, ir.ConstInt(128, DataType.INDEX, span)])
    half = ir.Var("half", _tile(shard.type.shape, lane_view, MS.Vec), span)
    shard = ir.Call(shard.op, shard.args, shard.kwargs, half.type, span)

    store = T.store(half, [0, 0], out_0, span=span)
    out_store = ir.Var("out_store", store.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(aiv_id, T.get_subblock_idx(span=span), span),
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
        attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
    )
    return ir.Program([func], "test_lane_valid_shard", span)


def test_boundary_tpop_type_binds_the_cloned_lane_index():
    """The tpop's per-lane extent must name the AIV lane's OWN ``aiv_id``.

    Each lane body is deep-cloned, so ``aiv_id = tile.get_subblock_idx()`` gets a
    fresh Var. The boundary tpop is built before that clone and carries the
    shard's ``valid_shape`` verbatim, so its type has to be remapped with
    everything else — otherwise it keeps pointing at the PRE-clone Var and the
    AIV body reads a lane index nothing in it defines. That dangling reference is
    invisible to a structural comparison (both Vars print as ``aiv_id``) and
    survives to codegen, so ``UseAfterDef`` is what states it.
    """
    after = _expand(_build_lane_valid_shard_program())

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.UseAfterDef)
    errors = [
        d
        for d in passes.PropertyVerifierRegistry.verify(props, after)
        if d.severity == passes.DiagnosticSeverity.Error
    ]
    assert not errors, [d.message for d in errors]

    printed = ir.python_print(after)
    # A Var referenced only from a type has no binding to print, so the printer
    # declares it as a free symbol instead of failing — the visible symptom.
    assert "pl.dynamic(" not in printed, printed
    # The extent itself must survive the repair, not be dropped to make it bind.
    assert "aiv_id" in printed, printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
