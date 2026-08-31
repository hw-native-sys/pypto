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

Authoring style
---------------
The pass runs at the tile level (post-InferTileMemorySpace), but that does *not*
put it out of reach of the ``@pl.program`` DSL: memory spaces are ordinary
``pl.Mem.*`` annotations, and the lowered boundary ops have a dedicated outlined
surface form (``pl.tile.aiv_shard(qk, split=1)``) that the printer emits for
exactly this already-lowered shape. The AUTO-path tests below therefore use the
project's mandated Before/Expected DSL style. Note that memrefs play no part
here — ``init_mem_ref`` runs ten passes later, so no ``Before`` or ``Expected``
in this file carries one.

One group is deliberately NOT authored in the DSL: the explicit
``SplitAivScopeStmt`` region tests, whose ``Before`` programs are hand-built so
the region reaches the pass bare. That section's own comment explains why no DSL
spelling can deliver one.

The scope-nesting tests at the very bottom are DSL for the opposite reason: the
DSL is what *produces* the shape under test (the parser's ``InCore`` scope
wrapper), so hand-building would not exercise the guard at all.

Negative tests keep ``pytest.raises``: a rejected transform produces no ``After``
IR, so Before/Expected does not apply. Their ``Before`` programs are still DSL.

``_lower`` keeps the print->parse roundtrip instrument ON (see its docstring for
why property verification is not), so the pass's output is asserted
round-trippable on every test.

End-to-end DSL coverage of this authoring form lives in
``tests/st/codegen/torch/test_torch_codegen_cross_core.py``
(``SplitAivShardProgram``), where the numerics are checked against torch.

The per-op vector halving tests (load / slice / reshape / store offset /
singleton / loop tracking / reduce-on-split-axis throw) were migrated here from
``test_split_vector_kernel.py``; generator rejection is also covered here. Those
facts are produced by the shared ``split_axis::ProcessStmts`` machinery, which
SplitVectorKernel's deleted per-op halving driver and this pass both call. The new
pass routes each VECTOR-affine leaf statement through that same machinery, so the
halving is identical (Stage 1 proved byte-identity); only the entry point changed.

A note on the ``cube_seed`` parameter that every AUTO-path ``Before`` carries:
LowerAutoVectorSplit only lowers *mixed* cube<->vector functions, so each program
opens with a ``pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)`` C->V boundary
to make the function genuinely mixed. Its result is unused; the op under test is
the vector sub-region that gets halved. In the lowered ``Expected`` that boundary
becomes ``pl.tile.aiv_shard(cube_seed, split=<mode>)``.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto import backend as _backend
from pypto.ir.instruments import make_roundtrip_instrument
from pypto.ir.op import tile_ops as T
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
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


def _lower(program):
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
        return passes.lower_auto_vector_split()(program)


# ---------------------------------------------------------------------------
# AUTO whole-function ``pl.split`` path — Before / Expected in the DSL.
# ---------------------------------------------------------------------------


def test_c2v_boundary_becomes_aiv_shard_and_vector_region_is_halved():
    """The C->V ``tile.move`` becomes ``tile.aiv_shard(split=1)`` and the vector
    sub-region (add + store result) is halved to ``[64, 128]`` while the cube
    operand ``qk`` stays full; ``subblock_idx`` is injected and ``split_aiv``
    stamped.

    The explicit ``TileView`` on ``y`` is the load-bearing part: the pass carries
    the pre-split Vec col-major view through the halving, which is *not* what
    ``tile.add`` would deduce from ``aiv_shard``'s view-less half result.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
            y = pl.tile.add(popped, popped)
            out_store = pl.tile.store(y, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            popped = pl.tile.aiv_shard(qk, split=1)
            y: pl.Tile[
                [64, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
            ] = pl.tile.add(popped, popped)
            out_store = pl.tile.store(y, [0 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_auto_c2v_boundary_localizes_a_ragged_split_axis():
    """A ragged split axis gets the LANE's extent on the AUTO path too.

    ``ReshapeSplitAxis`` can only ceil-halve the split-axis valid extent, because
    an op's type function does not know the lane. Here ``subblock_idx`` is in
    scope, so ``LocalizeShardValidForLane`` replaces that guess with the truth —
    lane 0 holds 64 of the 127 valid rows, lane 1 the remaining 63 — instead of
    giving BOTH lanes ``ceil(127 / 2) = 64``.

    The valid extent is 127 of a 128-row box, so the two lanes differ by exactly
    one. The boundary op still carries the authored MODE (``split=1``); the
    pto-isa code that expresses "lane 1 is one shorter" (3 =
    ``TILE_UP_DOWN_ODD``) is picked downstream, when ExpandMixedKernel mints the
    tpush / tpop pair — see ``split_axis::ShardSplitCode``.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[127, 128])],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(popped, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[127, 128])],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            popped: pl.Tile[
                [64, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(
                    valid_shape=[pl.min(pl.max(127, subblock_idx * 64) - subblock_idx * 64, 64), 128]
                ),
            ] = pl.tile.aiv_shard(qk, split=1)
            out_store = pl.tile.store(popped, [0 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_store_offset_at_nonzero_base_localizes_additively():
    """AdjustOffsets ADDS ``subblock_idx * half`` on the split axis rather than
    overwriting the offset: a store at base row 16 becomes ``16 + subblock_idx * 64``.

    The zero-base case is pinned by the C->V test above; this one is what
    distinguishes "additive" from "replaced", so the base is deliberately non-zero
    (and ``out_0`` is [256, 128] so the shifted store still fits).
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
            y = pl.tile.add(popped, popped)
            out_store = pl.tile.store(y, [16, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            popped = pl.tile.aiv_shard(qk, split=1)
            y: pl.Tile[
                [64, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major),
            ] = pl.tile.add(popped, popped)
            out_store = pl.tile.store(y, [16 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


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

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(prev, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            prev = pl.tile.load(data, [0 + subblock_idx * 64, 0], [64, 128], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(prev, [0 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_ragged_boundary_is_rebalanced_across_the_two_lanes():
    """A ragged C->V boundary partitions its VALID region, not the box.

    The box partition gives lane 0 all 8 rows of its half and lane 1 the
    remaining 5 — extents pto-isa cannot place, since it puts lane 1's FIFO band
    at its own extent (or one past it). Balancing the 13 valid rows instead
    gives 7 and 6: placeable under ``TILE_UP_DOWN_ODD``, and the work is split
    evenly. The physical box stays the box half (8); only the partition stride
    changes, and it rides to ExpandMixedKernel on the ``lane_stride`` attr.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            qk: pl.Tile[[16, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[13, 128])],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(popped, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            qk: pl.Tile[[16, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[13, 128])],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            popped: pl.Tile[
                [8, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(valid_shape=[pl.min(pl.max(13, subblock_idx * 7) - subblock_idx * 7, 7), 128]),
            ] = pl.tile.aiv_shard(qk, split=1, lane_stride=7)
            out_store = pl.tile.store(popped, [0 + subblock_idx * 7, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_empty_boundary_keeps_the_box_partition():
    """A boundary with nothing valid has no partition to balance.

    ``ceil(0 / 2)`` is 0, and a zero stride is not a legal partition — the
    boundary op's own attr check rejects it. An empty crossing therefore keeps
    the box partition, where both lanes are empty and the even code is exact.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            qk: pl.Tile[[16, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[0, 128])],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(popped, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            qk: pl.Tile[[16, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[0, 128])],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            popped: pl.Tile[
                [8, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(valid_shape=[pl.min(pl.max(0, subblock_idx * 8) - subblock_idx * 8, 8), 128]),
            ] = pl.tile.aiv_shard(qk, split=1)
            out_store = pl.tile.store(popped, [0 + subblock_idx * 8, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_rebalance_is_declined_when_a_value_is_split_independently():
    """A body that also splits its own value keeps the universal box partition.

    The balanced partition covers only the boundary's 13 valid rows, so a
    ``tile.load`` spanning all 16 would lose row 15 and double-cover row 7. Both
    tiles therefore stay on the box partition (offsets ``idx * 8``), and it is
    ExpandMixedKernel that reports the unplaceable lane extents.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            qk: pl.Tile[[16, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[13, 128])],
            data: pl.Tensor[[16, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [16, 128], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(prev, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            qk: pl.Tile[[16, 128], pl.FP32, pl.Mem.Mat, pl.TileView(valid_shape=[13, 128])],
            data: pl.Tensor[[16, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            popped: pl.Tile[  # noqa: F841
                [8, 128],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(valid_shape=[pl.min(pl.max(13, subblock_idx * 8) - subblock_idx * 8, 8), 128]),
            ] = pl.tile.aiv_shard(qk, split=1)
            prev = pl.tile.load(data, [0 + subblock_idx * 8, 0], [8, 128], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(prev, [0 + subblock_idx * 8, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vector_load_on_an_odd_split_axis_takes_the_ceil_half():
    """An ODD split axis halves to the CEIL box, with the lane's own valid extent.

    A `[15, 32]` tile cannot be cut into two equal boxes, so both lanes get the
    ceil half (`8`) and the per-lane valid extent carries the difference: lane 0
    fills all 8 rows, lane 1 only 7. The store offsets follow the box
    (`idx * 8`), so the two lanes' rows stay contiguous in the output.

    This tile never crosses the AIC/AIV boundary, so no pto-isa split code is
    involved — the ceil halving alone makes an odd extent representable.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[15, 32], pl.FP32],
            out_0: pl.Out[pl.Tensor[[15, 32], pl.FP32]],
        ) -> pl.Tensor[[15, 32], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [15, 32], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(prev, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[15, 32], pl.FP32],
            out_0: pl.Out[pl.Tensor[[15, 32], pl.FP32]],
        ) -> pl.Tensor[[15, 32], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            prev: pl.Tile[
                [8, 32],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(valid_shape=[pl.min(pl.max(15, subblock_idx * 8) - subblock_idx * 8, 8), 32]),
            ] = pl.tile.load(
                data,
                [0 + subblock_idx * 8, 0],
                [8, 32],
                [pl.min(pl.max(15, subblock_idx * 8) - subblock_idx * 8, 8), 32],
                target_memory=pl.Mem.Vec,
            )
            out_store = pl.tile.store(prev, [0 + subblock_idx * 8, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vector_load_on_an_odd_split_axis_left_right():
    """The dim-1 mirror: an odd COLUMN axis also halves to the ceil box.

    A `[32, 15]` tile gives both lanes an 8-column box; lane 0 fills all 8 and
    lane 1 only 7, and the column offsets follow the box (`idx * 8`). Pinning
    the LEFT_RIGHT side separately keeps a dim-1 regression from hiding behind
    the UP_DOWN cases.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[32, 15], pl.FP32],
            out_0: pl.Out[pl.Tensor[[32, 15], pl.FP32]],
        ) -> pl.Tensor[[32, 15], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [32, 15], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(prev, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[32, 15], pl.FP32],
            out_0: pl.Out[pl.Tensor[[32, 15], pl.FP32]],
        ) -> pl.Tensor[[32, 15], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=2)  # noqa: F841
            prev: pl.Tile[
                [32, 8],
                pl.FP32,
                pl.Mem.Vec,
                pl.TileView(valid_shape=[32, pl.min(pl.max(15, subblock_idx * 8) - subblock_idx * 8, 8)]),
            ] = pl.tile.load(
                data,
                [0, 0 + subblock_idx * 8],
                [32, 8],
                [32, pl.min(pl.max(15, subblock_idx * 8) - subblock_idx * 8, 8)],
                target_memory=pl.Mem.Vec,
            )
            out_store = pl.tile.store(prev, [0, 0 + subblock_idx * 8], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vector_load_halved_left_right():
    """LEFT_RIGHT: the load halves on dim1 (128 -> 64) and localizes the col offset."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(prev, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=2)  # noqa: F841
            prev = pl.tile.load(data, [0, 0 + subblock_idx * 64], [128, 64], target_memory=pl.Mem.Vec)
            out_store = pl.tile.store(prev, [0, 0 + subblock_idx * 64], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vector_slice_halves_shape_and_localizes_offset():
    """UP_DOWN: a tile.slice of a full (unsplit) Vec source halves its static shape
    tuple in lockstep with the result (the qk_pv strided sub-slice fix) and
    localizes its zero-base offset per subblock."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            src: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            sub = pl.tile.slice(src, [128, 128], [0, 0])
            out_store = pl.tile.store(sub, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            src: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            sub = pl.tile.slice(src, [64, 128], [0 + subblock_idx * 64, 0])
            out_store = pl.tile.store(sub, [0 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vector_slice_nonzero_base_offset_localizes_additively():
    """UP_DOWN: a strided sub-slice at a non-zero base offset localizes additively —
    the original offset is preserved and subblock_idx*half is added on the split
    axis (the exact qk_pv ``oi[16:32]`` pattern)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            src: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            sub = pl.tile.slice(src, [128, 128], [16, 0])
            out_store = pl.tile.store(sub, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            src: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            sub = pl.tile.slice(src, [64, 128], [16 + subblock_idx * 64, 0])
            out_store = pl.tile.store(sub, [0 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_slice_of_split_tracked_source_halves_shape_keeps_offset():
    """LEFT_RIGHT: a tile.slice whose source is already split-tracked (a halved
    load) halves its static shape tuple but leaves its offset unchanged — the
    source is already in lane-local coordinates."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [16, 128], target_memory=pl.Mem.Vec)
            sub = pl.tile.slice(prev, [16, 128], [0, 0])
            out_store = pl.tile.store(sub, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=2)  # noqa: F841
            prev = pl.tile.load(data, [0, 0 + subblock_idx * 64], [16, 64], target_memory=pl.Mem.Vec)
            sub = pl.tile.slice(prev, [16, 64], [0, 0])
            out_store = pl.tile.store(sub, [0, 0 + subblock_idx * 64], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_reshape_of_rank1_load_is_sliced_per_subblock():
    """UP_DOWN: a rank-1 load reshaped to [N, 1] is emitted at full width and
    followed by a per-subblock column slice so each lane reads its own row-half
    (the v2-minimal slice fix; rank-1 loads carry no 2D split axis)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            scale: pl.Tensor[[128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 1], pl.FP32]],
        ) -> pl.Tensor[[128, 1], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            scale_row = pl.tile.load(scale, [0], [128], target_memory=pl.Mem.Vec)
            scale_2d = pl.tile.reshape(scale_row, [128, 1])
            out_store = pl.tile.store(scale_2d, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            scale: pl.Tensor[[128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 1], pl.FP32]],
        ) -> pl.Tensor[[128, 1], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            scale_row = pl.tile.load(scale, [0], [128], target_memory=pl.Mem.Vec)
            scale_2d = pl.tile.reshape(scale_row, [128, 1])
            scale_2d_1 = pl.tile.slice(scale_2d, [64, 1], [subblock_idx * 64, 0])
            out_store = pl.tile.store(scale_2d_1, [0 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_reshape_of_already_split_input_halves_shape_arg():
    """UP_DOWN: a reshape whose input is already split halves its shape ARGUMENT
    too ([256, 1] -> [128, 1]), not just the result type, so memory_reuse sizes
    the output from the halved literal."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[256, 1], pl.FP32]],
        ) -> pl.Tensor[[256, 1], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            flat = pl.tile.reshape(prev, [256, 1])
            out_store = pl.tile.store(flat, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[256, 1], pl.FP32]],
        ) -> pl.Tensor[[256, 1], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            prev = pl.tile.load(data, [0 + subblock_idx * 8, 0], [8, 16], target_memory=pl.Mem.Vec)
            flat = pl.tile.reshape(prev, [128, 1])
            out_store = pl.tile.store(flat, [0 + subblock_idx * 128, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_auto_reinterpret_view_of_split_input_scales_lane_local_shape():
    """UP_DOWN: auto reinterpret keeps the tracked split axis and scales only the contiguous axis."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 32], pl.INT16]],
        ) -> pl.Tensor[[16, 32], pl.INT16]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            bits = pl.tile.reinterpret_view(prev, dtype=pl.INT16)
            out_store = pl.tile.store(bits, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 32], pl.INT16]],
        ) -> pl.Tensor[[16, 32], pl.INT16]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            prev = pl.tile.load(data, [0 + subblock_idx * 8, 0], [8, 16], target_memory=pl.Mem.Vec)
            bits = pl.tile.reinterpret_view(prev, dtype=pl.INT16)
            out_store = pl.tile.store(bits, [0 + subblock_idx * 8, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_auto_equivalent_explicit_reinterpret_shape_is_halved_with_split_input():
    """UP_DOWN: an explicit spelling of the auto shape is accepted and halved with the source."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 32], pl.INT16]],
        ) -> pl.Tensor[[16, 32], pl.INT16]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            bits = pl.tile.reinterpret_view(prev, dtype=pl.INT16, shape=[16, 32])
            out_store = pl.tile.store(bits, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 32], pl.INT16]],
        ) -> pl.Tensor[[16, 32], pl.INT16]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            prev = pl.tile.load(data, [0 + subblock_idx * 8, 0], [8, 16], target_memory=pl.Mem.Vec)
            bits = pl.tile.reinterpret_view(prev, dtype=pl.INT16, shape=[8, 32])
            out_store = pl.tile.store(bits, [0 + subblock_idx * 8, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_arbitrary_explicit_reinterpret_shape_is_rejected_under_split():
    """A byte-equivalent shape that redistributes dimensions has no safe physical split-axis mapping."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[8, 64], pl.INT16]],
        ) -> pl.Tensor[[8, 64], pl.INT16]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            bits = pl.tile.reinterpret_view(prev, dtype=pl.INT16, shape=[8, 64])
            out_store = pl.tile.store(bits, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="must match its auto-inferred shape"):
        _lower(Before)


def test_reinterpret_view_of_full_source_is_sliced_per_subblock():
    """LEFT_RIGHT: an untracked full tile param is reinterpreted, then sliced per lane."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[16, 32], pl.INT16]],
        ) -> pl.Tensor[[16, 32], pl.INT16]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            bits = pl.tile.reinterpret_view(data, dtype=pl.INT16)
            out_store = pl.tile.store(bits, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[16, 32], pl.INT16]],
        ) -> pl.Tensor[[16, 32], pl.INT16]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=2)  # noqa: F841
            bits = pl.tile.reinterpret_view(data, dtype=pl.INT16)
            bits_1 = pl.tile.slice(bits, [16, 16], [0, subblock_idx * 16])
            out_store = pl.tile.store(bits_1, [0, 0 + subblock_idx * 16], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_reshape_migrates_split_axis_row_to_col_and_back():
    """UP_DOWN: a [N,1]<->[1,N] reshape migrates the split axis, not corrupts it (gh#1864).

    The rms_norm column reshape moves the split data (rows) into the column dim and
    back. Each AIV lane keeps its own half, so the reshape targets must halve the
    MIGRATED dim ([1,8], then [8,1]) -- not stay at the stale full width ([1,16])
    which left lane 1 reading garbage and emitting inf. No per-subblock slice is
    needed (the partition is lane-local through the migration)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 1], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            col = pl.tile.load(data, [0, 0], [16, 1], target_memory=pl.Mem.Vec)
            row = pl.tile.reshape(col, [1, 16])
            inv_row = pl.tile.recip(row)
            back = pl.tile.reshape(inv_row, [16, 1])
            out_store = pl.tile.store(back, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 1], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            col = pl.tile.load(data, [0 + subblock_idx * 8, 0], [8, 1], target_memory=pl.Mem.Vec)
            row = pl.tile.reshape(col, [1, 8])
            inv_row = pl.tile.recip(row)
            back = pl.tile.reshape(inv_row, [8, 1])
            out_store = pl.tile.store(back, [0 + subblock_idx * 8, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_reshape_untrackable_split_axis_rejected():
    """A reshape whose split partition can't map to a clean per-dim halving is rejected.

    The dim-0 split of a [6, 4] tile partitions at flat offset 12 (rows 0-2 vs 3-5).
    Reshaping to [3, 8] would place that boundary mid-row, so no result dim can
    carry the halved split cleanly -- the pass rejects rather than miscompile."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[6, 4], pl.FP32],
            out_0: pl.Out[pl.Tensor[[3, 8], pl.FP32]],
        ) -> pl.Tensor[[3, 8], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            prev = pl.tile.load(data, [0, 0], [6, 4], target_memory=pl.Mem.Vec)
            flat = pl.tile.reshape(prev, [3, 8])
            out_store = pl.tile.store(flat, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="moves the split axis"):
        _lower(Before)


def test_singleton_broadcast_tile_preserved():
    """UP_DOWN: a [1, 128] broadcast tile is NOT halved on the singleton split dim."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            src: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
        ) -> pl.Tensor[[1, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            av = pl.tile.add(src, src)
            out_store = pl.tile.store(av, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            src: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
        ) -> pl.Tensor[[1, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()  # noqa: F841
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            av = pl.tile.add(src, src)
            out_store = pl.tile.store(av, [0, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


# ---------------------------------------------------------------------------
# Position-dependent root generators (tile.ci / tile.random).
#
# These were a single ``@pytest.mark.parametrize`` over (op, mode, shape) when
# the programs were hand-built. A DSL ``Before`` cannot be parametrized that way:
# ``pl.Tile[...]`` annotations and op shape arguments are read from the AST, so
# the shapes have to be literals. One test per case instead.
# ---------------------------------------------------------------------------


def test_ci_left_right_auto_halving_rejected():
    """Root generators need lane-specific position state, not just a halved result type."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            value = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            return value

    with pytest.raises(ValueError, match="automatic split-axis halving") as exc_info:
        _lower(Before)
    assert "tile.ci" in str(exc_info.value)


def test_random_up_down_auto_halving_rejected():
    """Root generators need lane-specific position state, not just a halved result type."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[128, 64], pl.UINT32, pl.Mem.Vec]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            value = pl.tile.random(
                pl.const(1, pl.INT32),
                pl.const(2, pl.INT32),
                pl.const(3, pl.INT32),
                pl.const(4, pl.INT32),
                pl.const(5, pl.INT32),
                pl.const(6, pl.INT32),
                [128, 64],
                dtype=pl.UINT32,
                rounds=10,
            )
            return value

    with pytest.raises(ValueError, match="automatic split-axis halving") as exc_info:
        _lower(Before)
    assert "tile.random" in str(exc_info.value)


def test_random_left_right_auto_halving_rejected():
    """Root generators need lane-specific position state, not just a halved result type."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[128, 64], pl.UINT32, pl.Mem.Vec]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            value = pl.tile.random(
                pl.const(1, pl.INT32),
                pl.const(2, pl.INT32),
                pl.const(3, pl.INT32),
                pl.const(4, pl.INT32),
                pl.const(5, pl.INT32),
                pl.const(6, pl.INT32),
                [128, 64],
                dtype=pl.UINT32,
                rounds=10,
            )
            return value

    with pytest.raises(ValueError, match="automatic split-axis halving") as exc_info:
        _lower(Before)
    assert "tile.random" in str(exc_info.value)


def test_ci_up_down_singleton_split_dim_preserved():
    """A singleton split dimension requires no generator-state rewrite."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            value = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            return value

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[1, 64], pl.INT32, pl.Mem.Vec]:
            subblock_idx = pl.tile.get_subblock_idx()  # noqa: F841
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            value = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            return value

    ir.assert_structural_equal(_lower(Before), Expected)


def test_random_up_down_singleton_split_dim_preserved():
    """A singleton split dimension requires no generator-state rewrite."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[1, 64], pl.UINT32, pl.Mem.Vec]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            value = pl.tile.random(
                pl.const(1, pl.INT32),
                pl.const(2, pl.INT32),
                pl.const(3, pl.INT32),
                pl.const(4, pl.INT32),
                pl.const(5, pl.INT32),
                pl.const(6, pl.INT32),
                [1, 64],
                dtype=pl.UINT32,
                rounds=10,
            )
            return value

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[1, 64], pl.UINT32, pl.Mem.Vec]:
            subblock_idx = pl.tile.get_subblock_idx()  # noqa: F841
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            value = pl.tile.random(
                pl.const(1, pl.INT32),
                pl.const(2, pl.INT32),
                pl.const(3, pl.INT32),
                pl.const(4, pl.INT32),
                pl.const(5, pl.INT32),
                pl.const(6, pl.INT32),
                [1, 64],
                dtype=pl.UINT32,
                rounds=10,
            )
            return value

    ir.assert_structural_equal(_lower(Before), Expected)


def test_random_left_right_singleton_split_dim_preserved():
    """A singleton split dimension requires no generator-state rewrite."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[128, 1], pl.UINT32, pl.Mem.Vec]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            value = pl.tile.random(
                pl.const(1, pl.INT32),
                pl.const(2, pl.INT32),
                pl.const(3, pl.INT32),
                pl.const(4, pl.INT32),
                pl.const(5, pl.INT32),
                pl.const(6, pl.INT32),
                [128, 1],
                dtype=pl.UINT32,
                rounds=10,
            )
            return value

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
        ) -> pl.Tile[[128, 1], pl.UINT32, pl.Mem.Vec]:
            subblock_idx = pl.tile.get_subblock_idx()  # noqa: F841
            seed_vec = pl.tile.aiv_shard(cube_seed, split=2)  # noqa: F841
            value = pl.tile.random(
                pl.const(1, pl.INT32),
                pl.const(2, pl.INT32),
                pl.const(3, pl.INT32),
                pl.const(4, pl.INT32),
                pl.const(5, pl.INT32),
                pl.const(6, pl.INT32),
                [128, 1],
                dtype=pl.UINT32,
                rounds=10,
            )
            return value

    ir.assert_structural_equal(_lower(Before), Expected)


def test_loop_iter_arg_keeps_split_tracking():
    """UP_DOWN: a loop iter_arg seeded by a halved load keeps split-aware store
    offsets inside the loop body (tile_vars tracking flows through iter_args)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            accum = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            for i, (out_it,) in pl.range(2, init_values=(out_0,)):  # noqa: B007
                out_it_next = pl.tile.store(accum, [0, 0], out_it)
                out_loop = pl.yield_(out_it_next)
            return out_loop

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            accum = pl.tile.load(data, [0 + subblock_idx * 64, 0], [64, 128], target_memory=pl.Mem.Vec)
            for i, (out_it,) in pl.range(2, init_values=(out_0,)):  # noqa: B007
                out_it_next = pl.tile.store(accum, [0 + subblock_idx * 64, 0], out_it)
                out_loop = pl.yield_(out_it_next)
            return out_loop

    ir.assert_structural_equal(_lower(Before), Expected)


def test_reduce_on_split_axis_rejected():
    """A reduce that collapses the split axis (dim0 under UP_DOWN) raises ValueError —
    a partial per-lane reduction is a miscompile.

    ``col_sum`` is the axis-0 reduction (``pto.tcolsum``), so under UP_DOWN it
    collapses exactly the split axis."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            src: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            rv = pl.tile.col_sum(src)
            out_store = pl.tile.store(rv, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="reduces on the split axis"):
        _lower(Before)


# ---------------------------------------------------------------------------
# V->C boundary.
#
# tile.aic_gather is declared HALF -> FULL, so its operand must be a per-lane
# half the affinity gate produced. The pass enforces that precondition: an
# un-halved vector operand is rejected rather than doubled (doubling would hand
# the cube a 2x tile while its FULL destination type remained unchanged).
# ---------------------------------------------------------------------------


def test_vc_boundary_becomes_aic_gather_without_redundant_mat_move():
    """UP_DOWN: a V->C Mat boundary becomes tile.aic_gather directly.

    The vector value crossing to the cube is a halved load, so the gather
    reassembles [64, 128] -> [128, 128]. The gather already lands in Mat, so a
    second Mat-placement move would become an unsupported Mat -> Mat TMOV.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            v = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            gathered = pl.tile.move(v, target_memory=pl.Mem.Mat)  # noqa: F841 - V->C boundary
            out_store = pl.tile.store(v, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            v = pl.tile.load(data, [0 + subblock_idx * 64, 0], [64, 128], target_memory=pl.Mem.Vec)
            gathered = pl.tile.aic_gather(v, split=1)  # noqa: F841
            out_store = pl.tile.store(v, [0 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vc_boundary_rejects_unhalved_vector_operand():
    """A full-width vector value at a V->C boundary has no half to gather.

    ``vec`` is a Vec parameter used directly, so the affinity gate never halves
    it. Doubling it via tile.aic_gather would produce a [256, 128] operand under
    a cube-placement move still typed [128, 128] — shape-inconsistent IR that
    does not survive print->parse. The pass reports the authoring error instead.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            vec: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            gathered = pl.tile.move(vec, target_memory=pl.Mem.Mat)  # noqa: F841 - V->C boundary
            out_store = pl.tile.store(vec, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="full-width vector operand"):
        _lower(Before)


def test_vector_op_rejects_unsharded_full_width_operand():
    """A vector op whose operand no lane owns a half of is rejected, not halved.

    ``vec`` is a full-width Vec parameter, so nothing inside the region
    partitions it. Halving only the ``tile.add`` RESULT would emit
    ``tile.add([256, 128], [256, 128]) -> [128, 128]`` — a node whose declared
    shape contradicts type inference over its own arguments, which fails
    print->parse and misleads every later consumer that re-derives types from
    operands. This is the elementwise mirror of
    ``test_vc_boundary_rejects_unhalved_vector_operand`` above: the guard that
    already existed one boundary later now also covers ordinary vector ops.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            vec: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            vec_h = pl.tile.add(vec, vec)
            gathered_mat = pl.tile.move(vec_h, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="carries a full-width operand") as exc_info:
        _lower(Before)
    # The diagnostic names the operand and the in-contract spellings, so the
    # author can act on it without reading the pass.
    message = str(exc_info.value)
    assert "'vec'" in message
    assert "aiv_shard" in message


def test_declared_scratch_operand_stays_full_width():
    """A full-width operand the operator DECLARES as scratch is kept.

    ``tile.row_sum(tile, tmp_tile)`` uses ``tmp_tile`` as hardware workspace whose
    contract is "at least as large as the input", never as per-lane data, so it
    declares ``set_lane_invariant_arg(1)``. The buffer comes from ``tile.create``,
    registered ``CoreAffinity::SHARED``, so the affinity gate deliberately passes
    it through full width for both lanes to declare — it is never tracked, and
    matching on the extent alone would reject the whole rms-norm reduction shape
    (gh#1864's runtime parity probe).

    The exemption is a registry declaration on purpose. See
    ``test_sel_full_width_mask_is_rejected`` for why "the result shape is not
    deduced from this operand" is a different property and cannot stand in.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            scratch = pl.tile.create([256, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            sums = pl.tile.row_sum(v, scratch)
            vec_h = pl.tile.row_expand_mul(v, sums)
            gathered_mat = pl.tile.move(vec_h, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            v = pl.tile.load(x, [0 + subblock_idx * 128, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec)
            # Untouched: the scratch is SHARED, so both lanes declare it whole.
            scratch = pl.tile.create([256, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            sums = pl.tile.row_sum(v, scratch)
            vec_h = pl.tile.row_expand_mul(v, sums)
            gathered_mat = pl.tile.aic_gather(vec_h, split=1)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    # _lower keeps the print->parse roundtrip instrument on, so this also pins the
    # invariant the guard protects: the accepted node still re-parses.
    ir.assert_structural_equal(_lower(Before), Expected)


def test_dynamic_full_width_reshape_is_rejected():
    """A full-width reshape source needs a STATIC half extent to be partitioned.

    ``tile.reshape`` over an untracked source is diverted into a full view plus a
    per-lane ``tile.slice``, and that slice can only be materialized from a static
    half extent. A dynamic one used to fall through to plain result halving, which
    is the offsetless-view miscompile the diversion exists to prevent: both lanes
    read the first half, so lane 1 silently reuses lane 0's data. Reject it, as
    the sibling ``tile.reinterpret_view`` already does.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec],
            rows: pl.Scalar[pl.INDEX],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            reshaped = pl.tile.reshape(data, [rows, 16])
            out_store = pl.tile.store(reshaped, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="requires a static split extent") as exc_info:
        _lower(Before)
    message = str(exc_info.value)
    assert "tile.reshape" in message


# ---------------------------------------------------------------------------
# Halving is gated on TWO independently necessary conditions. Neither implies
# the other, so each needs its own coverage:
#   (1) every tile operand is lane-local, split-axis-singleton, or DECLARED
#       scratch  -- violated by tile.sel's full-width mask, which the operator's
#       type deduction never reads;
#   (2) the halved node still type-checks against its own arguments -- violated
#       by tile.transpose_view, whose axis permutation the broadcast
#       right-alignment in (1) is not entitled to describe.
# ---------------------------------------------------------------------------


def test_sel_full_width_mask_is_rejected():
    """A per-element operand outside the result deduction is still lane data.

    ``tile.sel(mask, lhs, rhs, tmp)`` deduces its result from ``BroadcastShapes``
    over lhs/rhs alone and validates only that ``mask`` is a ``TileType``. So with
    lhs/rhs halved and a full-width ``mask``, the halved node re-deduces cleanly —
    condition (2) is satisfied and cannot catch this. Both lanes would then read
    mask rows 0..127 and silently compute the wrong halves.

    Only condition (1) rejects it, and only because ``tile.sel`` declares arg 3
    (``tmp``) lane-invariant and NOT arg 0.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            mask: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            tmp: pl.Tile[[1, 16], pl.UINT32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            w = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.sel(mask, v, w, tmp)
            gathered_mat = pl.tile.move(picked, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="carries a full-width operand") as exc_info:
        _lower(Before)
    message = str(exc_info.value)
    assert "'mask'" in message


def test_sels_full_width_mask_is_rejected():
    """An oversized mask satisfies ``tile.sels``' coverage rule and is still wrong.

    ``tile.sels`` validates that the mask's carrier rows *cover* src's valid rows,
    which a full-width mask over a halved src passes. Coverage is not partition:
    each lane would read the mask from row 0.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            mask: pl.Tile[[256, 128], pl.INT32, pl.Mem.Vec],
            tmp: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.sels(mask, v, tmp, pl.const(0.0, pl.FP32))
            gathered_mat = pl.tile.move(picked, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="carries a full-width operand") as exc_info:
        _lower(Before)
    assert "'mask'" in str(exc_info.value)


def test_declared_scratch_is_exempt_on_the_left_right_split_axis():
    """The declaration, not the extent, is what exempts a scratch operand.

    ``tile.sel``'s ``tmp`` is ``[1, 16]``: singleton on dim 0, so ``UP_DOWN`` would
    skip it for the wrong reason. Under ``LEFT_RIGHT`` the split axis is dim 1,
    where it is full width — and it must still be accepted, while every real
    operand is lane-local via ``tile.load``.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            tmp: pl.Tile[[1, 16], pl.UINT32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            m = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            w = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.sel(m, v, w, tmp)
            gathered_mat = pl.tile.move(picked, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    lowered = _lower(Before)
    # The tmp operand survives at its declared full width, and the sel result is
    # halved on the LEFT_RIGHT axis.
    printed = lowered.as_python()
    assert "pl.tile.sel(" in printed
    assert "[256, 64]" in printed


def test_gather_full_width_table_is_lane_shared():
    """A lookup table addressed by absolute index is shared, not un-sharded.

    ``tile.gather`` computes ``out[i] = src[indices[i]]`` and takes its result
    shape from ``indices``. The index values are absolute into ``src``, so halving
    ``indices`` already gives each lane its own half of the OUTPUT while both read
    the whole table — the correct lowering, not a missing shard.

    Positional correspondence is what the guard is really about, and ``src`` has
    none with the result, so ``tile.gather`` declares it lane-invariant alongside
    its ``tmp``. See ``test_scatter_update_full_width_index_is_rejected`` for the
    dual, which must stay rejected.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            idx: pl.Tensor[[256, 128], pl.INT32],
            scratch: pl.Tensor[[256, 128], pl.INT32],
            src: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            indices = pl.tile.load(idx, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            # tmp is shape-MATCHED to indices on A2/A3, so it is loaded inside the
            # region and halves with them. Only `src` is lane-shared.
            tmp = pl.tile.load(scratch, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.gather(src, indices, tmp)
            gathered_mat = pl.tile.move(picked, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            idx: pl.Tensor[[256, 128], pl.INT32],
            scratch: pl.Tensor[[256, 128], pl.INT32],
            src: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            indices = pl.tile.load(
                idx, [0 + subblock_idx * 128, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec
            )
            tmp = pl.tile.load(
                scratch, [0 + subblock_idx * 128, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec
            )
            # `src` stays whole; indices, tmp and the result are all halved.
            picked = pl.tile.gather(src, indices, tmp)
            gathered_mat = pl.tile.aic_gather(picked, split=1)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_gatherb_full_width_table_is_lane_shared():
    """The byte-offset form of the same shape: `offset` carries the addressing."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            off_t: pl.Tensor[[256, 128], pl.UINT32],
            src: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 1024], pl.FP32]],
        ) -> pl.Tensor[[256, 1024], pl.FP32]:
            offset = pl.tile.load(off_t, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.gatherb(src, offset, output_dtype=pl.FP32)
            seed = pl.tile.move(rhs, target_memory=pl.Mem.Mat)  # noqa: F841
            out_store = pl.tile.store(picked, [0, 0], out_0)
            return out_store

    lowered = _lower(Before)
    printed = lowered.as_python()
    # The table stays [256, 128] while the gathered result halves to [128, 1024].
    assert "pl.tile.gatherb(src, offset" in printed
    assert "[128, 1024]" in printed


def test_gather_partitioned_table_producer_is_rejected():
    """Declaring the table lane-shared permits full width; it does not keep it so.

    Nothing stops the operand's own producer from being halved: a ``tile.load``
    inside the region is halved with the lane offset baked in. The index values
    are absolute into the WHOLE table, so lane 1 reads the wrong rows and an index
    past the halved extent is out of bounds.

    Neither other gate sees this. The operand is tracked, so condition (1) skips
    it; and ``tile.gather`` deduces its result from ``indices`` alone, so
    condition (2) is satisfied. It needs its own check.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            tbl: pl.Tensor[[256, 128], pl.FP32],
            idx: pl.Tensor[[256, 128], pl.INT32],
            tmp: pl.Tile[[256, 128], pl.INT32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            # The lookup table is LOADED inside the region, so the pass halves it.
            src = pl.tile.load(tbl, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            indices = pl.tile.load(idx, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.gather(src, indices, tmp)
            gathered_mat = pl.tile.move(picked, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="ABSOLUTE indices") as exc_info:
        _lower(Before)
    message = str(exc_info.value)
    assert "'src'" in message
    # The remedy names where the table has to come from.
    assert "OUTSIDE the automatically split region" in message


def test_gatherb_partitioned_table_producer_is_rejected():
    """The byte-offset form carries the same absolute-addressing hazard."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            tbl: pl.Tensor[[256, 128], pl.FP32],
            off_t: pl.Tensor[[256, 128], pl.UINT32],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 1024], pl.FP32]],
        ) -> pl.Tensor[[256, 1024], pl.FP32]:
            src = pl.tile.load(tbl, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            offset = pl.tile.load(off_t, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.gatherb(src, offset, output_dtype=pl.FP32)
            seed = pl.tile.move(rhs, target_memory=pl.Mem.Mat)  # noqa: F841
            out_store = pl.tile.store(picked, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="ABSOLUTE indices") as exc_info:
        _lower(Before)
    assert "'src'" in str(exc_info.value)


def test_scratch_with_partitioned_producer_is_still_accepted():
    """Halving is harmless for scratch, so the two declared kinds must differ.

    ``tile.row_sum``'s ``tmp_tile`` contract is a size relation to the input
    ("at least as large as"), and a halved input takes a halved scratch with it.
    Rejecting every partitioned lane-invariant operand would break this, which is
    why ``LaneInvariantArg`` records *why* an operand is lane-invariant rather
    than just that it is.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            s: pl.Tensor[[256, 128], pl.FP32],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            # Scratch produced INSIDE the region, so the pass halves it too.
            scratch = pl.tile.load(s, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            sums = pl.tile.row_sum(v, scratch)
            vec_h = pl.tile.row_expand_mul(v, sums)
            gathered_mat = pl.tile.move(vec_h, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    printed = _lower(Before).as_python()
    # Both the input and its scratch halve together; the reduction follows.
    assert printed.count("[128, 128], pl.FP32, pl.Mem.Vec") >= 2
    assert "pl.tile.row_sum(v, scratch)" in printed


def test_gather_partitioned_table_is_rejected_even_with_a_singleton_result():
    """The absolute-index check cannot hang off the halving path.

    With ``indices`` shaped ``[1, N]`` the gather result is ``[1, N]`` too, so the
    result's split axis is singleton and the halving path returns early — nothing
    about the RESULT is rewritten. The table underneath it is still halved by its
    own producer, and the trailing ``Substitute`` still swaps the halved var in,
    so the absolute indices address the wrong half regardless. The check
    therefore runs before the singleton early-return, not beside the halving.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            tbl: pl.Tensor[[256, 128], pl.FP32],
            idx: pl.Tensor[[1, 128], pl.INT32],
            tmp: pl.Tile[[1, 128], pl.INT32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
        ) -> pl.Tensor[[1, 128], pl.FP32]:
            src = pl.tile.load(tbl, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            indices = pl.tile.load(idx, [0, 0], [1, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.gather(src, indices, tmp)
            seed = pl.tile.move(rhs, target_memory=pl.Mem.Mat)  # noqa: F841
            out_store = pl.tile.store(picked, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="ABSOLUTE indices") as exc_info:
        _lower(Before)
    assert "source operand 'src'" in str(exc_info.value)


def test_tuple_result_op_over_a_partitioned_operand_is_rejected():
    """A tuple result is not halveable, and its operands are substituted anyway.

    ``tile.gather_compare`` returns ``Tuple[dst[rows, out_cols], cdst[1, rows]]``,
    so the generic path — which only understands a single ``TileType`` — skips it
    entirely: no full-width check, no type consistency, no halving. But the
    trailing ``Substitute`` still rewrites its arguments, so the call ends up
    declaring full-width tuple elements over a halved ``[128, 128]`` input. That
    is illegal IR and would misallocate, so it is rejected; per-element split
    mapping for tuple results is not implemented.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            t: pl.Tensor[[256, 128], pl.FP32],
            tmp: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 16], pl.INT32]],
        ) -> pl.Tensor[[256, 16], pl.INT32]:
            src = pl.tile.load(t, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            dst, cdst = pl.tile.gather_compare(src, pl.const(1.0, pl.FP32), tmp, cmp_mode="eq", out_cols=16)
            seed = pl.tile.move(rhs, target_memory=pl.Mem.Mat)  # noqa: F841
            out_store = pl.tile.store(dst, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="returns a tuple of tiles") as exc_info:
        _lower(Before)
    assert "tile.gather_compare" in str(exc_info.value)


def test_scatter_with_every_operand_partitioned_is_rejected():
    """All three operands being tracked does not make absolute indices safe.

    ``tile.scatter`` writes ``dst.flat[indexes[i, j]] = src[i, j]``, where the
    indexes are FLATTENED offsets into the whole ``dst`` (a column write encodes
    ``i * dst_cols + c``). Loading dst, src and indexes all inside the region
    tracks every one of them, so the full-width check skips them all and type
    consistency passes — yet the index values were never rebased, so the halved
    row stride no longer matches and lane 1 writes past its destination.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            d: pl.Tensor[[256, 128], pl.FP32],
            s: pl.Tensor[[256, 128], pl.FP32],
            i: pl.Tensor[[256, 128], pl.INT32],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            dst = pl.tile.load(d, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            src = pl.tile.load(s, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            indexes = pl.tile.load(i, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            res = pl.tile.scatter(dst, src, indexes)
            seed = pl.tile.move(rhs, target_memory=pl.Mem.Mat)  # noqa: F841
            out_store = pl.tile.store(res, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="ABSOLUTE indices") as exc_info:
        _lower(Before)
    assert "destination operand 'dst'" in str(exc_info.value)


def test_shape_matched_scratch_is_not_exempt_at_full_width():
    """A scratch whose shape the deducer pins is rejected with no metadata involved.

    ``tile.row_argmax`` needs a tmp shaped *exactly* like the source — the
    TROWARGMAX kernel reads the column count from the tmp/src extent, so a wider
    tmp walks past the valid columns and can return the wrong index per column.
    Nothing in the registry says so: ``DeduceTileRowReductionType`` runs with
    ``require_exact_tmp_shape``, so re-deducing the halved call throws and the
    type-consistency gate refuses it. The operator answers for itself.

    That is why ``tile.row_sum`` needs a declaration and this one must not have
    one — see ``test_declared_scratch_operand_stays_full_width`` for the sibling
    whose deducer never looks at ``tmp_tile``, and
    ``tests/ut/ir/operators/test_lane_invariant_arg_coverage.py`` for the check
    that keeps the two apart as deducers change.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            tmp: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 1], pl.INT32]],
        ) -> pl.Tensor[[256, 1], pl.INT32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.row_argmax(v, tmp)
            seed = pl.tile.move(rhs, target_memory=pl.Mem.Mat)  # noqa: F841
            out_store = pl.tile.store(picked, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="carries a full-width operand") as exc_info:
        _lower(Before)
    assert "'tmp'" in str(exc_info.value)


def test_scratch_the_deducer_pins_needs_no_declaration():
    """The registry is consulted only where type deduction is silent.

    ``DeduceTileRsqrtType`` requires ``tmp`` to match the input's rank and every
    dimension, so a full-width ``tmp`` beside a halved input is already refused by
    re-deducing the halved call. ``tile.rsqrt`` therefore carries NO
    ``set_lane_invariant_arg`` — one would claim full width is in contract while
    the operator itself rejects that width, and could never be reached.

    Pinning it here keeps the two halves of the model honest: this is the case a
    declaration must *not* cover, and
    ``tests/ut/ir/operators/test_lane_invariant_arg_coverage.py`` fails if one is
    ever added back.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            tmp: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            scaled = pl.tile.rsqrt(v, tmp=tmp)
            gathered_mat = pl.tile.move(scaled, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="carries a full-width operand") as exc_info:
        _lower(Before)
    assert "'tmp'" in str(exc_info.value)


def test_target_dependent_scratch_stays_exempt():
    """Whether an oversized scratch is out of contract can be a TARGET question.

    ``tile.gather``'s index form does not read ``tmp`` at all on A5, so a
    full-width external tmp beside halved indices is legal there; A2/A3 requires
    it to match the indices, and the PTOAS verifier already enforces that where
    the target is known. Rejecting here would break the legal A5 form to
    duplicate a check this pass cannot make accurately, so the exemption stays.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            idx: pl.Tensor[[256, 128], pl.INT32],
            src: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            tmp: pl.Tile[[256, 128], pl.INT32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            indices = pl.tile.load(idx, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            picked = pl.tile.gather(src, indices, tmp)
            gathered_mat = pl.tile.move(picked, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    printed = _lower(Before).as_python()
    assert "pl.tile.gather(src, indices, tmp)" in printed
    # tmp keeps its declared full width; only `indices` and the result are halved.
    assert "tmp: pl.Tile[[256, 128], pl.INT32" in printed
    assert "picked: pl.Tile[[128, 128]" in printed


def test_loop_carried_tile_accumulator_is_tracked():
    """The AUTO path must repair loop carries, or it rejects a legal accumulator.

    The AUTO arm recurses through its own affinity-gated walk rather than
    ``ProcessStmts``, so it never reached the ``ForStmt`` carry repair. The
    ``iter_arg`` then stayed full width and untracked while its init was halved,
    and the ``tile.add`` on the carry was reported as a full-width operand — a
    legal program failing to compile. Existing loop coverage carried only
    ``Tensor`` iter_args, which are never halved and so never showed this.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            accum = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            for i, (acc_it,) in pl.range(2, init_values=(accum,)):  # noqa: B007
                acc_next = pl.tile.add(acc_it, acc_it)
                acc_loop = pl.yield_(acc_next)
            out_store = pl.tile.store(acc_loop, [0, 0], out_0)
            return out_store

    printed = _lower(Before).as_python()
    # The carry and everything derived from it are lane-local.
    assert "pl.tile.add(acc_it, acc_it)" in printed
    assert printed.count("[64, 128]") >= 3
    # The store on the loop-exit var picks up the per-lane offset. Assert it on the
    # store LINE: a bare `"subblock_idx * 64" in printed` also matches the tile.load
    # above, so it passed whether or not the store was offset at all.
    store_line = next(line for line in printed.splitlines() if "pl.tile.store(" in line)
    assert "pl.tile.store(acc_loop, [0 + subblock_idx * 64, 0]" in store_line


def test_slice_drop_dims_maps_the_result_axis_back_to_the_source_axis():
    """shape / offset / valid_shape are indexed in the PRE-DROP rank.

    ``drop_dims`` erases axes from the result after those tuples are built, so
    ``slice([1, 256, 128], drop_dims=[0]) -> [256, 128]`` puts the result's dim 0
    on the source's dim 1. Halving the source's dim 0 instead left the node
    disagreeing with its own arguments, and the type-consistency check then
    rejected a legal split.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[1, 256, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            v = pl.tile.load(data, [0, 0, 0], [1, 256, 128], target_memory=pl.Mem.Vec)
            sl = pl.tile.slice(v, [1, 256, 128], [0, 0, 0], drop_dims=[0])
            out_store = pl.tile.store(sl, [0, 0], out_0)
            return out_store

    printed = _lower(Before).as_python()
    # The halved extent and the lane offset both land on source axis 1, not 0.
    assert "pl.tile.slice(v, [1, 128, 128], [0, 0 + subblock_idx * 128, 0]" in printed


def test_slice_drop_dims_maps_across_two_dropped_axes():
    """Two dropped axes, so the mapping has to walk past both.

    The mapping consumes ``drop_dims`` with a single cursor, relying on
    ``ParseSliceDropDims`` returning the axes ascending. One dropped axis cannot
    tell that cursor apart from one that never advances — both land on the same
    answer — so the shift only shows up once a second axis is dropped:
    ``slice([1, 1, 256, 128], drop_dims=[0, 1]) -> [256, 128]`` puts the result's
    dim 0 on the source's dim 2.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[1, 1, 256, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            v = pl.tile.load(data, [0, 0, 0, 0], [1, 1, 256, 128], target_memory=pl.Mem.Vec)
            sl = pl.tile.slice(v, [1, 1, 256, 128], [0, 0, 0, 0], drop_dims=[0, 1])
            out_store = pl.tile.store(sl, [0, 0], out_0)
            return out_store

    printed = _lower(Before).as_python()
    # Source axis 2 carries the halved extent and the lane offset; 0 and 1 are dropped.
    assert "pl.tile.slice(v, [1, 1, 128, 128], [0, 0, 0 + subblock_idx * 128, 0]" in printed


def test_scatter_update_partitioned_destination_is_rejected():
    """The scatter dual addresses a DESTINATION by absolute index.

    ``tile.scatter_update`` takes its result from ``input`` and writes rows named
    by absolute ``index`` values, so a partitioned ``input`` is written at the
    wrong offsets. The symmetry with ``tile.gather`` is only apparent: gather may
    keep a full-width source, whereas here the destination is what must stay
    whole, and ``index`` / ``src`` remain ordinary per-lane data.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            index: pl.Tile[[256, 128], pl.INT32, pl.Mem.Vec],
            values: pl.Tile[[256, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            base = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            upd = pl.tile.scatter_update(base, index, values, dim=-2)
            gathered_mat = pl.tile.move(upd, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="ABSOLUTE indices") as exc_info:
        _lower(Before)
    message = str(exc_info.value)
    assert "destination operand 'base'" in message
    assert "writes to the wrong half" in message


@pytest.mark.parametrize("ascend_backend", [_backend.BackendType.Ascend950], indirect=True)
def test_tensor_gather_in_a_split_region_is_diagnosed_through_the_pipeline(ascend_backend):
    """The user-level path reaches the guard, not just hand-written tile IR.

    ``pl.gather`` over a tensor lowers to a ``tile.load`` of the table feeding
    ``tile.gather`` (`op_conversion_registry.cpp`), and only then does
    ``LowerAutoVectorSplit`` run — so the table is produced *inside* the split
    region as a matter of course, and this is the ordinary A5 spelling rather
    than a hand-built corner. Driving the real pipeline is what proves the guard
    is reachable the way an author would hit it; the tile-level tests above pass
    the table as a parameter, which is exactly the shape that never gets halved.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def main(
            tbl: pl.Tensor[[256, 128], pl.FP32],
            idx: pl.Tensor[[256, 128], pl.INT32],
            w: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            picked = pl.gather(tbl, dim=-1, index=idx)
            acc = pl.matmul(picked, w)
            out_0[0:256, 0:128] = acc
            return out_0

    manager = PassManager(OptimizationStrategy.Default)
    with pytest.raises(ValueError, match="ABSOLUTE indices") as exc_info:
        with passes.PassContext([]):
            manager.run_passes(Before)
    # The operand named is the tile.load the tensor->tile conversion synthesized.
    assert "tile.gather" in str(exc_info.value)


def test_transpose_view_of_full_width_source_is_rejected():
    """An axis-permuting view does not obey the broadcast right-alignment.

    ``tile.transpose_view`` swaps the trailing two dims, so an untracked
    ``[1, 256]`` source under ``UP_DOWN`` presents a *singleton* on the axis the
    right-alignment maps the result's split dim onto — condition (1) accepts it.
    The halved ``[128, 1]`` result then contradicts the ``[256, 1]`` the operand
    deduces, and only condition (2) catches that. Unlike ``tile.transpose``, this
    is a pure view and is outside ``FindTransposeSplitHazard``.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[256, 1], pl.FP32]],
        ) -> pl.Tensor[[256, 1], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            viewed = pl.tile.transpose_view(data)
            out_store = pl.tile.store(viewed, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="no longer matches type inference") as exc_info:
        _lower(Before)
    assert "tile.transpose_view" in str(exc_info.value)


# The full-width-operand guard matches operands to the result from the RIGHT, the
# alignment BroadcastShapes gives ``tile.add([M, N], [N]) -> [M, N]``. The three
# tests below pin both answers that alignment produces for the same [128] bias,
# plus the same-rank singleton it must not be confused with. Indexing the operand
# with the result's ABSOLUTE split dim gets both directions wrong: it rejects the
# legal shared bias under UP_DOWN, and under LEFT_RIGHT it reads the index as out
# of range and skips the operand that does span the split axis.


def test_rank1_broadcast_operand_off_the_split_axis_is_shared():
    """A [128] bias right-aligns to dim 1, so UP_DOWN leaves it whole.

    The bias has no axis of its own on the split dim, so both AIV lanes read it
    entire while the ``tile.add`` result still halves to [128, 128].
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            bias: pl.Tile[[128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            vec_h = pl.tile.add(v, bias)
            gathered_mat = pl.tile.move(vec_h, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            bias: pl.Tile[[128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            v = pl.tile.load(x, [0 + subblock_idx * 128, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec)
            vec_h = pl.tile.add(v, bias)
            gathered_mat = pl.tile.aic_gather(vec_h, split=1)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_rank1_broadcast_operand_on_the_split_axis_is_rejected():
    """The SAME [128] bias right-aligns onto dim 1, which LEFT_RIGHT splits.

    Nothing partitions the bias, so halving only the add's result would emit
    ``tile.add([128, 64], [128]) -> [128, 64]``. The operand must be reported on
    its OWN axis 0, not skipped as out of range.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            bias: pl.Tile[[128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            vec_h = pl.tile.add(v, bias)
            gathered_mat = pl.tile.move(vec_h, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="carries a full-width operand") as exc_info:
        _lower(Before)
    message = str(exc_info.value)
    assert "'bias'" in message
    # Reported on the OPERAND's own axis, which is 0 here, not the result's 1.
    assert "on its dim 0" in message


def test_same_rank_singleton_operand_is_shared():
    """A same-rank [1, 128] bias is replicated, not partitioned.

    This is the other accepting branch and must not be confused with the
    right-alignment one: the axis exists, it is simply singleton.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            bias: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            v = pl.tile.load(x, [0, 0], [256, 128], target_memory=pl.Mem.Vec)
            vec_h = pl.tile.add(v, bias)
            gathered_mat = pl.tile.move(vec_h, target_memory=pl.Mem.Mat)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            x: pl.Tensor[[256, 128], pl.FP32],
            bias: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
            rhs: pl.Tile[[128, 128], pl.FP32, pl.Mem.Right],
            out_0: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
        ) -> pl.Tensor[[256, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            v = pl.tile.load(x, [0 + subblock_idx * 128, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec)
            vec_h = pl.tile.add(v, bias)
            gathered_mat = pl.tile.aic_gather(vec_h, split=1)
            left = pl.tile.move(gathered_mat, target_memory=pl.Mem.Left)
            acc = pl.tile.matmul(left, rhs)
            out_store = pl.tile.store(acc, [0, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vc_boundary_gathers_on_the_migrated_split_axis():
    """The gather reassembles the OPERAND's split axis, not the function's.

    A ``[16, 1] -> [1, 16]`` reshape migrates the split axis from dim 0 to dim 1
    (the rms_norm column-reshape shape). Gathering the *function* axis would
    double dim 0, turning the lane-local ``[1, 8]`` into ``[2, 8]`` while the
    cube-placement move still expects ``[1, 16]``. Gathering the tracked axis
    yields ``[1, 16]`` — hence ``split=2`` here under an UP_DOWN function.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 1], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            col = pl.tile.load(data, [0, 0], [16, 1], target_memory=pl.Mem.Vec)
            row = pl.tile.reshape(col, [1, 16])
            gathered = pl.tile.move(row, target_memory=pl.Mem.Mat)  # noqa: F841 - V->C boundary
            out_store = pl.tile.store(col, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 1], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            col = pl.tile.load(data, [0 + subblock_idx * 8, 0], [8, 1], target_memory=pl.Mem.Vec)
            row = pl.tile.reshape(col, [1, 8])
            gathered = pl.tile.aic_gather(row, split=2)  # noqa: F841
            out_store = pl.tile.store(col, [0 + subblock_idx * 8, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vc_boundary_gathers_left_right_shard_fed_directly():
    """LEFT_RIGHT: a C->V shard result fed straight into a V->C boundary gathers
    on dim 1 (``split=2``), not dim 0.

    The C->V arm seeds ``tile_vars`` with the shard's own ``split_dim``; if that
    defaulted to 0, the gather would double rows (``[128, 64] -> [256, 64]``)
    instead of columns and trip the shape invariant. ``popped`` is used directly
    as the V->C operand (no intervening compute), so this exercises the shard
    arm's seeding rather than the per-op halving path.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
            back = pl.tile.move(popped, target_memory=pl.Mem.Mat)  # noqa: F841 - V->C boundary
            out_store = pl.tile.store(popped, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.LEFT_RIGHT, "split_aiv": True},
        )
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            popped = pl.tile.aiv_shard(qk, split=2)
            back = pl.tile.aic_gather(popped, split=2)  # noqa: F841
            out_store = pl.tile.store(popped, [0, 0 + subblock_idx * 64], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_vc_boundary_keeps_non_mat_cube_placement():
    """A gather lands in Mat, but a Left destination still needs tile.move."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            value = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            left = pl.tile.move(value, target_memory=pl.Mem.Left)  # noqa: F841
            out_store = pl.tile.store(value, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            value = pl.tile.load(data, [0 + subblock_idx * 64, 0], [64, 128], target_memory=pl.Mem.Vec)
            left_mat = pl.tile.aic_gather(value, split=1)
            left = pl.tile.move(left_mat, target_memory=pl.Mem.Left)  # noqa: F841
            out_store = pl.tile.store(value, [0 + subblock_idx * 64, 0], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


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
    """The ``core_placement`` attr of every call to ``op_name``, in body order.

    ``None`` for an unstamped call, so a list like ``["aiv", None]`` states both
    which calls the region placed and which it left alone.
    """
    found = []

    def walk(node):
        if node is None:
            return
        if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == op_name:
            found.append(node.attrs.get("core_placement"))
        if isinstance(node, ir.SeqStmts):
            for stmt in node.stmts:
                walk(stmt)
            return
        if isinstance(node, ir.AssignStmt):
            walk(node.value)
        if isinstance(node, ir.EvalStmt):
            walk(node.expr)
        walk(getattr(node, "body", None))

    for func in program.functions.values():
        walk(func.body)
    return found


# ---------------------------------------------------------------------------
# Explicit SplitAivScopeStmt region path (RFC #1300 nestable first-class node).
#
# LowerAutoVectorSplit is the SOLE consumer of SplitAivScopeStmt: it injects a
# per-region subblock index, halves ONLY the vector compute INSIDE each region
# (region-local maps so no leak to sibling regions or out-of-region full-width
# ops), validates a per-region transpose hazard, then DROPS the scope wrapper.
# The AUTO whole-function path above is unchanged.
#
# These ``Before`` programs are hand-built rather than DSL-authored (the AUTO
# section above is DSL). The reason is specific and verified: the parser wraps a
# ``for aiv_id in pl.split_aiv(...)`` region in a scope whenever an InCore scope
# is open — which it always is inside a function declared
# ``pl.FunctionType.InCore`` — and OutlineIncoreScopes only outlines scopes out
# of Opaque / Orchestration functions, so the wrapper survives to this pass,
# which rejects a scope-nested region by design. That holds for a region at
# function top level AND for one nested in a loop, so there is no DSL spelling
# that delivers a *bare* region to this pass. Routing through a plain
# ``@pl.function`` + ``outline_incore_scopes()`` does produce one, but renames
# the function (``main`` -> ``main_incore_0``) and rewrites its parameter list,
# so the pass would no longer be tested in isolation — that path is covered
# once, deliberately, by test_outlined_region_still_lowers_and_stamps.
# ---------------------------------------------------------------------------

# Attrs the region path stamps on the function (no whole-function ``split`` mode —
# each region carries its own ``split_``). Same for every mode, including the
# task-parallel ``None``: the ``split_aiv`` marker alone routes the function to the
# both-lanes split path downstream (never the lane-0-only no-split replay).
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

    For UP_DOWN / LEFT_RIGHT (data-parallel): the region path prepends an
    injected ``subblock_idx = get_subblock_idx()``, keeps the region's own
    ``aiv_id`` binding, halves the load on the split axis, and localizes the
    load + store offsets per subblock.

    For NONE (task-parallel): the body is passed through UNCHANGED (scope erased)
    — the author's ``aiv_id`` binding survives, tiles stay FULL, offsets are not
    localized, and NO internal ``subblock_idx`` is injected. Returns
    ``(lowered_stmts, out_store_var)``.
    """
    aiv_id = ir.Var("aiv_id", _IDX, span)
    if mode.value == 0:  # NONE — no halving, no injected subblock_idx, full tiles.
        load = T.load(data, [0, 0], list(full_shape), target_memory=MS.Vec, span=span)
        t = ir.Var("t", load.type, span)
        store = T.store(t, [0, 0], out, span=span)
        out_store = ir.Var("out_store", store.type, span)
        none_stmts: list[ir.Stmt] = [
            _get_subblock(aiv_id, span),
            ir.AssignStmt(t, load, span),
            ir.AssignStmt(out_store, store, span),
        ]
        return none_stmts, out_store
    sub = _sub_var()
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


def test_none_region_keeps_tiles_full_and_binds_aiv_id():
    """A task-parallel (NONE) region is passed through FULL-width: the load is NOT
    halved, offsets are NOT localized, NO internal subblock_idx is injected, the
    author's aiv_id binding survives, the scope wrapper is dropped, and the
    function is stamped split_aiv + split_aiv_region_validated (same as the
    data-parallel region path)."""
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
    """An AUTO ``pl.split`` function must NOT take the explicit-region branch.

    The full AUTO lowering is pinned by the Before/Expected tests at the top of
    this file. What is asserted here is the one fact those do not isolate: the
    region path's ``split_aiv_region_validated`` marker is absent, so a function
    with no ``SplitAivScopeStmt`` provably went through the whole-function arm.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
            y = pl.tile.add(popped, popped)
            out_store = pl.tile.store(y, [0, 0], out_0)
            return out_store

    (func,) = _lower(Before).functions.values()
    attrs = func.attrs
    # The mode survives as its SplitMode int encoding, not the enum object.
    assert attrs.get("split") == ir.SplitMode.UP_DOWN.value
    assert attrs.get("split_aiv") is True
    assert "split_aiv_region_validated" not in attrs, (
        "AUTO whole-function path must not stamp the region-path marker"
    )


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


def test_auto_path_scope_bodied_mixed_is_rejected():
    """A mixed AUTO ``pl.split`` function behind a scope is rejected."""

    @pl.program
    class Scoped:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            self,
            qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                popped = pl.tile.move(qk, target_memory=pl.Mem.Vec)
                y = pl.tile.add(popped, popped)
                out_store = pl.tile.store(y, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match=r"body still contains a scope"):
        _lower(Scoped)


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


# ---------------------------------------------------------------------------
# Explicit-region admissions: values that are NOT derived from tile.aiv_shard but
# are still per-lane by construction. Two classes are admitted — pure generators
# (tile.full/create/ci/random) and address-carrying ops (tile.load/slice/extract)
# whose args reference the region's lane index. The rationale for each, and for
# why a generator is NOT added to half_tiles, lives at ScanRegionHalfWidth in
# src/ir/transforms/lower_auto_vector_split_pass.cpp — keep it in one place.
#
# The explicit path splices the region body through UNCHANGED, so a positive
# test's Expected is literally its Before minus the scope wrapper. That identity
# is the property under test, so one helper builds both.
# ---------------------------------------------------------------------------


def _admission_program(span, body_fn, *, wrap, nest_in_loop=False):
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
    shard = T.aiv_shard(qk, split=1, span=span)  # UP_DOWN => [64, 128] Vec
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
        region = ir.SplitAivScopeStmt(
            split=ir.SplitMode.UP_DOWN, body=ir.SeqStmts(region_stmts, span), span=span
        )
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


def test_region_rejects_gather_row_localized_only_on_dst():
    """A gather is per-lane only when its READ offset is lane-derived. With the
    lane reference in ``dst_offset`` and a lane-invariant ``src_offset``, both
    lanes fetch the SAME GM row — full-width work replicated — so it must still be
    reported. This is what pins ``AddressArgs`` to src_offset alone. NEGATIVE
    test: no ``After`` IR."""
    with pytest.raises(ValueError, match=r"mixes explicit.*tile\.gather_row"):
        _lower(_admission_program(ir.Span.unknown(), _gather_row_dst_only_localized_body, wrap=True))


def test_region_rejects_lane_reference_outside_address_args():
    """A lane reference only localizes when it lands in an op's ADDRESS args. A
    tile.load at offset [0, 0] that mentions aiv_id only in its valid_shape has
    BOTH lanes reading the same base rows, so it must still be reported —
    otherwise its consumers would be trusted as half-width. NEGATIVE test."""
    with pytest.raises(ValueError, match=r"mixes explicit.*tile\.load"):
        _lower(_admission_program(ir.Span.unknown(), _lane_ref_in_non_address_arg_body, wrap=True))


def test_region_rejects_consumer_of_full_width_generator():
    """A generator is admitted for ITSELF only — it does not join the half-width
    dataflow. So a consumer reachable from a full-width generator and from no
    shard is still reported. Without this, ``z = tile.full([128,128]);
    y = tile.add(z, z)`` would be silently accepted and BOTH AIV lanes would
    compute (and store) the full tile. NEGATIVE test: no ``After`` IR."""
    with pytest.raises(ValueError, match=r"mixes explicit.*tile\.add"):
        _lower(_admission_program(ir.Span.unknown(), _full_width_generator_body, wrap=True))


def test_region_rejects_lane_reference_on_non_addressing_op():
    """A lane reference is trusted only on an ADDRESS-carrying op. A lane-derived
    scalar reaching a non-addressing op says nothing about the result's width, so
    ``tile.set_validshape(full_width_tile, 1, aiv_id * 64)`` must not launder a
    full-width tile into the half-width dataflow. NEGATIVE test: no ``After``
    IR. (A full-width load with NO lane reference is covered by
    test_mixed_explicit_implicit_region_rejected above.)"""
    with pytest.raises(ValueError, match=r"mixes explicit.*tile\.set_validshape"):
        _lower(_admission_program(ir.Span.unknown(), _laundering_body, wrap=True))


# ---------------------------------------------------------------------------
# Scope-nested regions are rejected, not silently passed through.
#
# Region lowering walks for / while / if / seq but deliberately NOT ScopeStmt: a
# scope carries outlining and name-visibility semantics that region-local
# halving must not reach through. A region behind a scope therefore cannot be
# lowered, and the pass must say so instead of stamping
# ``split_aiv_region_validated`` on a function whose region guards never ran.
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

    The region itself is consumed here, so the surviving-region guard passes —
    but the inner walks (``LowerStmts`` / ``CheckNoCubeTileHalved`` /
    ``ScanRegionHalfWidth``) step over a ``ScopeStmt`` rather than entering it,
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
    """The canonical Opaque form is unaffected: pass 7 outlines, pass 18 lowers.

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
# Region placement stamp (``attrs["core_placement"] = "aiv"``)
#
# The pass ERASES the SplitAivScopeStmt wrapper, so ExpandMixedKernel — which
# runs next and duplicates every SHARED statement onto both lanes — would
# otherwise have no way to tell that the author placed a statement on the vector
# lane. The stamp is that carrier.
#
# It is written only on calls whose lane the region DECIDES: a call that states
# its own lane (``tile.get_subblock_idx`` declares SHARED by policy) or whose
# memory spec already fixes one (any vector op is VECTOR; the aiv_shard /
# aic_gather boundary is MIXED because it genuinely runs on both lanes) is
# placed without it. So a comm op is stamped and the compute around it is not —
# which is what the ``[..., None]`` expectations below pin.
#
# All three region arms are covered (task-parallel splice, explicit-boundary
# splice, data-parallel halving), plus nesting inside a loop, plus the
# out-of-region negative.
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


def test_none_region_stamps_comm_op_with_aiv_placement():
    """Task-parallel (NONE) splice arm: the in-region notify is stamped.

    This is the arm the mixed comm kernels use — ``pl.split_aiv(2,
    mode=pl.SplitMode.NONE)`` runs the full body on both AIV lanes — and the one
    the user docs tell authors to reach for.
    """
    span = ir.Span.unknown()
    program = _notify_region_program(span, ir.SplitMode.NONE, in_region=True)

    assert _placements(_lower(program), _NOTIFY) == ["aiv"]


def test_data_parallel_region_stamps_comm_op_with_aiv_placement():
    """Data-parallel (UP_DOWN) halving arm: the stamp survives the rewriting.

    The halving machinery replaces calls as it localizes offsets and halves
    shapes, so the stamp is applied to the FINAL statements — a stamp written
    before the rewrite would mark calls that no longer exist.
    """
    span = ir.Span.unknown()
    program = _notify_region_program(span, ir.SplitMode.UP_DOWN, in_region=True)

    assert _placements(_lower(program), _NOTIFY) == ["aiv"]


def test_explicit_boundary_region_stamps_comm_op_with_aiv_placement():
    """Explicit-boundary splice arm: a body the author already half-widthed.

    A user-written ``tile.aiv_shard`` routes the region through the pass-through
    arm instead of the halving one, and that arm must stamp too — the body is no
    less region-placed for being pre-halved. The whole body derives from the
    shard result (the mixed-explicit validator rejects a full-width op here), so
    this cannot reuse the shared builder above.
    """
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

    after = _lower(program)
    assert _placements(after, _NOTIFY) == ["aiv"]
    # The boundary op itself is NOT stamped: it runs on BOTH lanes (tpush on the
    # cube side, tpop on the vector side), so "aiv" would be false of it.
    assert _placements(after, ir.get_op("tile.aiv_shard").name) == [None]


def test_comm_op_nested_in_loop_inside_region_is_stamped():
    """The stamp walk descends into compound statements.

    A notify buried in a ForStmt inside the region is as region-placed as one at
    the region's top level; missing it would silently reintroduce the
    duplication bug for any loop-carried signal.
    """
    span = ir.Span.unknown()
    program = _notify_region_program(span, ir.SplitMode.NONE, in_region=True, nest_in_loop=True)

    assert _placements(_lower(program), _NOTIFY) == ["aiv"]


def test_comm_op_outside_region_is_not_stamped():
    """The negative: an out-of-region notify keeps no placement.

    The region is authoritative only for what it contains. Stamping outside it
    would claim a placement the author never made.
    """
    span = ir.Span.unknown()
    program = _notify_region_program(span, ir.SplitMode.NONE, in_region=False)

    assert _placements(_lower(program), _NOTIFY) == [None]


def test_region_compute_is_not_stamped():
    """Only calls whose lane the region decides are stamped.

    ``tile.load`` is VECTOR from its own memory spec and
    ``tile.get_subblock_idx`` declares SHARED by policy (both lanes need the
    lane index), so neither needs — or gets — a placement. Pins that the stamp
    is a placement decision rather than a region-membership marker sprayed over
    the whole body.
    """
    span = ir.Span.unknown()
    after = _lower(_notify_region_program(span, ir.SplitMode.NONE, in_region=True))

    assert _placements(after, ir.get_op("tile.load").name) == [None]
    assert _placements(after, ir.get_op("tile.store").name) == [None]
    assert _placements(after, ir.get_op("tile.get_subblock_idx").name) == [None]


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
    """Lower ``fn`` for a2a3 (Ascend910B) and return (aic_body_text, aiv_body_text).

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
    assert "pl.tile.tpush_to_aic(" in aiv and "split=0" in aiv
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
    assert "pl.tile.tpush_to_aiv(" in aic and "split=0" in aic
    assert "pl.tile.tpop_from_aic(split=0)" in aiv
    assert "pl.Tile[[64, 64], pl.FP32" in aiv
    assert "pl.tile.aiv_shard" not in aic and "pl.tile.aiv_shard" not in aiv


def test_if_merge_variable_is_retyped_and_tracked():
    """An IfStmt that merges a tile must carry the split through its merge var.

    Both branches yield a value the split halved, so the merge variable is
    lane-local too. Left at its declared full width it disagrees with both
    ``Yield`` values, and — because it is never registered in ``tile_vars`` — a
    following ``tile.store`` gets no lane offset, so both AIV lanes write from
    output row 0 instead of taking a half each.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            flag: pl.Scalar[pl.INT64],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            v = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            if flag > 0:
                doubled = pl.tile.add(v, v)
                merged: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(doubled)
            else:
                merged: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(v)
            out_store = pl.tile.store(merged, [0, 0], out_0)
            return out_store

    printed = _lower(Before).as_python()
    # The merge variable is halved with its branches...
    assert "merged: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec]" in printed
    # ...and the store that consumes it takes this lane's half of the output.
    assert "pl.tile.store(merged, [0 + subblock_idx * 64, 0], out_0)" in printed


def test_if_branches_disagreeing_on_the_merge_are_rejected():
    """One halved branch and one full-width branch have no single merge type.

    ``v`` is loaded inside the region and halved; ``shared`` is a full-width
    parameter nothing partitions. Halving the merge variable would be wrong for
    the else path and leaving it full width wrong for the then path, so neither
    choice is a merge — take the rejection instead of silently giving one AIV
    lane the wrong extent.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            shared: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            flag: pl.Scalar[pl.INT64],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            v = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            if flag > 0:
                merged: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(v)
            else:
                merged: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.yield_(shared)
            out_store = pl.tile.store(merged, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="disagree about merge variable") as exc_info:
        _lower(Before)
    assert "'merged'" in str(exc_info.value)


def test_source_level_no_else_phi_still_merges_after_ssa_conversion():
    """A source-level ``if`` with no ``else`` reaches this pass *with* one.

    ``SSAVerifier::VerifyIfStmt`` rejects an ``IfStmt`` that defines
    ``return_vars_`` without an else branch, and ``ConvertToSSA`` synthesizes the
    else ``Yield`` (from the binding the merge shadows) precisely so the shape
    below is legal by the time any later pass sees it. So the merge repair needs
    no else-less special case — it only has to handle what SSA guarantees.

    Running the real conversion here rather than hand-shaping the ``IfStmt`` is
    the point: it is what proves the guarantee holds for the source form users
    actually write.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            flag: pl.Scalar[pl.INT64],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            merged = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            if flag > 0:
                merged = pl.tile.add(merged, merged)
            out_store = pl.tile.store(merged, [0, 0], out_0)
            return out_store

    in_ssa = passes.convert_to_ssa()(Before)
    converted = in_ssa.get_function("split_auto")
    assert converted is not None
    body = converted.body
    stmts = body.stmts if isinstance(body, ir.SeqStmts) else [body]
    if_stmt = next(s for s in stmts if isinstance(s, ir.IfStmt))
    # The guarantee this test rests on: the conversion supplied the else branch.
    assert if_stmt.return_vars, "expected the rebind to become an IfStmt merge"
    assert if_stmt.else_body is not None, "ConvertToSSA must synthesize the else for a no-else phi"

    # SSA renames, so anchor on the merge's own name rather than the source spelling.
    merge_name = if_stmt.return_vars[0].name_hint
    printed = _lower(in_ssa).as_python()
    assert f"{merge_name}: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec]" in printed
    store_line = next(line for line in printed.splitlines() if "pl.tile.store(" in line)
    assert f"pl.tile.store({merge_name}, [0 + subblock_idx * 64, 0]" in store_line


def test_loop_backedge_yielding_a_full_width_value_is_rejected():
    """The backedge must agree with the carry it feeds.

    ``RepairIterArgs`` / ``RepairReturnVars`` repair the carry's entry and exit,
    but nothing looked at the value the body yields back into it. With a halved
    init and a body that yields a full-width parameter, the carry is retyped to
    ``[64, 128]`` while the yielded value stays ``[128, 128]`` — the gh#2203
    "declared type contradicts the value" defect on the carry path, which no
    operand check sees because it inspects a ``Call``'s arguments and this is a
    ``Yield``.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            full: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            accum = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
            for i, (acc_it,) in pl.range(2, init_values=(accum,)):  # noqa: B007
                acc_loop = pl.yield_(full)
            out_store = pl.tile.store(acc_loop, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="yields") as exc_info:
        _lower(Before)
    assert "acc" in str(exc_info.value)


def test_loop_backedge_yielding_a_lane_local_value_into_a_shared_carry_is_rejected():
    """The mismatch is rejected in both directions.

    Here the carry's init is a full-width parameter, so nothing halved it, while
    the body yields a value the split made lane-local. A per-lane value cannot
    live in a full-width carry any more than the converse, and the check is
    symmetric so neither direction silently survives.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[128, 128], pl.FP32],
            full: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
            out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            for i, (acc_it,) in pl.range(2, init_values=(full,)):  # noqa: B007
                halved = pl.tile.load(data, [0, 0], [128, 128], target_memory=pl.Mem.Vec)
                acc_loop = pl.yield_(halved)
            out_store = pl.tile.store(acc_loop, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="is full width, but the body yields") as exc_info:
        _lower(Before)
    assert "'halved'" in str(exc_info.value)


def test_slice_drop_dims_maps_the_tracked_source_axis_onto_the_result_axis():
    """The split axis lives on the OPERAND; the halving path indexes the RESULT.

    ``drop_dims`` erases axes and then clamps back to 2D by *prepending* unit
    axes, so the tracked source's axis 0 lands on result axis 1 here. Reading the
    result at the source's axis instead found the synthetic unit axis, took the
    singleton early-return, and passed the slice through untouched while its
    source was halved underneath it — a 16-row window over an 8-row source.

    This is the mirror of ``test_slice_drop_dims_maps_the_result_axis_back_to_the_source_axis``:
    that one maps result → source for an *untracked* source, this one maps
    source → result for a *tracked* one. Both directions are needed.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[16, 4], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 4], pl.FP32],
            out_0: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> pl.Tensor[[1, 16], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            v = pl.tile.load(data, [0, 0], [16, 4], target_memory=pl.Mem.Vec)
            sl = pl.tile.slice(v, [16, 1], [0, 0], drop_dims=[1])
            out_store = pl.tile.store(sl, [0, 0], out_0)
            return out_store

    @pl.program
    class Expected:
        @pl.function(
            type=pl.FunctionType.InCore,
            attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
        )
        def split_auto(
            cube_seed: pl.Tile[[16, 4], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 4], pl.FP32],
            out_0: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> pl.Tensor[[1, 16], pl.FP32]:
            subblock_idx = pl.tile.get_subblock_idx()
            seed_vec = pl.tile.aiv_shard(cube_seed, split=1)  # noqa: F841
            v = pl.tile.load(data, [0 + subblock_idx * 8, 0], [8, 4], [8, 4], target_memory=pl.Mem.Vec)
            # The window follows the source onto its own axis: 8 rows, not 16.
            sl = pl.tile.slice(v, [8, 1], [0, 0], [], [1])
            # The store lands on result axis 1, where the split axis now lives.
            out_store = pl.tile.store(sl, [0, 0 + subblock_idx * 8], out_0)
            return out_store

    ir.assert_structural_equal(_lower(Before), Expected)


def test_slice_dropping_the_tracked_split_axis_is_rejected():
    """``drop_dims``' unit requirement is on the WINDOW, not on the source.

    A tracked ``[16, 4]`` source windowed ``[1, 4]`` may drop axis 0, so the axis
    the split partitions can be erased while it is still live. Each lane would
    then take that window from its own half — lane 1 selects source row 8 where
    the program asked for row 0 — and the result carries no axis left to tell the
    lanes apart, so the following store cannot give them separate destinations
    either. Both lanes would write the same place with different data.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def split_auto(
            cube_seed: pl.Tile[[16, 4], pl.FP32, pl.Mem.Mat],
            data: pl.Tensor[[16, 4], pl.FP32],
            out_0: pl.Out[pl.Tensor[[1, 4], pl.FP32]],
        ) -> pl.Tensor[[1, 4], pl.FP32]:
            seed_vec = pl.tile.move(cube_seed, target_memory=pl.Mem.Vec)  # noqa: F841
            v = pl.tile.load(data, [0, 0], [16, 4], target_memory=pl.Mem.Vec)
            sl = pl.tile.slice(v, [1, 4], [0, 0], drop_dims=[0])
            out_store = pl.tile.store(sl, [0, 0], out_0)
            return out_store

    with pytest.raises(ValueError, match="the axis the automatic split partitions") as exc_info:
        _lower(Before)
    assert "drops dim 0" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
