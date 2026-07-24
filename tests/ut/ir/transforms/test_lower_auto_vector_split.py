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
why property verification is not), so the pass's output is asserted round-trippable
on every test but one — ``test_outlined_region_still_lowers_and_stamps``, which
opts out for an upstream reason documented at that test.

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
from pypto.ir.instruments import make_roundtrip_instrument
from pypto.ir.op import tile_ops as T

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
# the cube a 2x tile while the cube-placement move kept its original FULL result
# type, contradicting tile.move's shape-preserving contract).
# ---------------------------------------------------------------------------


def test_vc_boundary_becomes_aic_gather_and_cube_placement_stays_full():
    """UP_DOWN: a V->C tile.move boundary becomes tile.aic_gather, and the cube
    placement move on the gathered tile stays FULL ([128, 128] Mat) — the cube
    side never sees a halved tile.

    The vector value crossing to the cube is a halved load, so the gather
    reassembles [64, 128] -> [128, 128] and the move's kept [128, 128] agrees.
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
            gathered_mat = pl.tile.aic_gather(v, split=1)
            gathered = pl.tile.move(gathered_mat, target_memory=pl.Mem.Mat)  # noqa: F841
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
            gathered_mat = pl.tile.aic_gather(row, split=2)
            gathered = pl.tile.move(gathered_mat, target_memory=pl.Mem.Mat)  # noqa: F841
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
            back_mat = pl.tile.aic_gather(popped, split=2)
            back = pl.tile.move(back_mat, target_memory=pl.Mem.Mat)  # noqa: F841
            out_store = pl.tile.store(popped, [0, 0 + subblock_idx * 64], out_0)
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


def _shard_vec(tile, split, half_shape, span):
    """A C->V boundary ``tile.aiv_shard`` returning a HALF *Vec* tile.

    ``tile.aiv_shard`` declares Vec as its result memory (the consuming vector
    lane), and ``OpRegistry::Create`` fills that onto the space-less deduced half
    type — so the pass itself attaches nothing. This helper still has to state Vec
    explicitly only because it builds the ``ir.Call`` directly, bypassing Create.
    """
    return ir.Call(ir.get_op("tile.aiv_shard"), [tile], {"split": split}, _tile(half_shape, MS.Vec), span)


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


def test_none_region_rejects_aiv_shard():
    """A boundary op (tile.aiv_shard) inside a NONE region is rejected: a
    task-parallel region has no split axis to shard. NEGATIVE — no After IR.
    (The always-on lowering CHECK fires even with verification disabled.)"""
    span = ir.Span.unknown()
    data = ir.Var("data", _tensor([128, 128]), span)
    out_0 = ir.Var("out_0", _tensor([128, 128]), span)
    aiv_id_call = T.get_subblock_idx(span=span)
    aiv_id = ir.Var("aiv_id", aiv_id_call.type, span)
    cube = ir.Var("cube", _tile([128, 128], mem=MS.Mat), span)
    cube_load = T.load(data, [0, 0], [128, 128], target_memory=MS.Mat, span=span)
    shard = _shard_vec(cube, 1, [64, 128], span)
    sh = ir.Var("sh", shard.type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(aiv_id, aiv_id_call, span),
            ir.AssignStmt(cube, cube_load, span),
            ir.AssignStmt(sh, shard, span),
        ],
        span,
    )
    region = ir.SplitAivScopeStmt(split=ir.SplitMode.NONE, body=body, span=span)
    program = _explicit_region_program(
        [region, ir.ReturnStmt([sh], span)],
        [(data, _IN), (out_0, _OUT)],
        [sh.type],
    )
    with pytest.raises(ValueError, match="must not contain tile.aiv_shard"):
        _lower(program)


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


def _lane_ref_in_non_address_arg_body(span, stmts, aiv_id, qk_h, data):
    """A tile.load whose OFFSET is [0, 0] — both lanes read the same base rows —
    but which mentions aiv_id in its valid_shape. Scanning every arg instead of
    just the address args would wrongly admit this."""
    valid = aiv_id + 1
    v = ir.Var("valid", valid.type, span)
    load = T.load(data, [0, 0], [64, 128], valid_shapes=[v, 128], target_memory=MS.Vec, span=span)
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
# DSL is what produces the shape: the parser wraps a top-level
# ``for aiv_id in pl.split_aiv(...)`` in an InCore ScopeStmt, and
# OutlineIncoreScopes only outlines scopes out of Opaque / Orchestration
# functions — so the wrapper survives into a function declared
# ``pl.FunctionType.InCore``.
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
            # The parser wraps this top-level region in an InCore ScopeStmt.
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
            qk = pl.matmul(a_left, b_right)
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):  # noqa: B007
                qk_h = pl.aiv_shard(qk)  # noqa: F841 - explicit boundary
                t = pl.load(data, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec)  # full width
                out_0 = pl.tile.store(t, [0, 0], out_0)
            return out_0

    with pytest.raises(ValueError, match=_SCOPE_NESTED_MSG):
        _lower(NestedMixed)


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

    # This is the one test that cannot use ``_lower``: it must run with the
    # roundtrip instrument OFF. OutlineIncoreScopes emits an InCore function that
    # reads and writes ``c`` without declaring it a parameter
    # (``def main_incore_0(a)`` with a free ``c``), so the outlined program does
    # not survive print->parse ("Undefined variable 'c'") — verified BEFORE this
    # pass runs. That is an OutlineIncoreScopes defect, not one of this pass; the
    # region lowering below is what is under test here.
    with passes.PassContext([]):
        outlined = passes.outline_incore_scopes()(Canonical)
        after = passes.lower_auto_vector_split()(outlined)

    incore = [f for f in after.functions.values() if f.func_type == ir.FunctionType.InCore]
    assert len(incore) == 1, "OutlineIncoreScopes should have produced one InCore function"
    assert "pl.split_aiv" not in ir.python_print(after), "region must be erased"
    for key, value in _REGION_ATTRS.items():
        assert incore[0].attrs.get(key) == value, f"expected {key}={value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
