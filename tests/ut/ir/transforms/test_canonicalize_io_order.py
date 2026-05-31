# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSL-style Before/Expected tests for the CanonicalizeIOOrder pass.

The pass walks every ``SeqStmts`` **inside a ``ForKind.Pipeline`` body** and
reorders its top-level statements along a hardware-unit stage ladder — scalar
compute, tile.load, (producer) tile compute, cross-core push, cross-core pop,
(consumer) tile compute, tile.store — all subject to the SSA dependency graph.
Loops that are not pipelined are left untouched.

Tests that want reorder wrap the outer in ``pl.pipeline(..., stage=1)`` to opt
in. The pass demotes ``ForKind.Pipeline`` → ``ForKind.Sequential`` and strips
any stale ``pipeline_stages`` attr on exit, so the Expected programs use plain
``pl.range`` — matching the post-pass state.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_pass(program: ir.Program) -> ir.Program:
    """Run CanonicalizeIOOrder with structural verification disabled — our
    Before programs use minimal tile IR that doesn't satisfy the full set of
    structural prerequisites the pipeline normally enforces."""
    with passes.PassContext([], passes.VerificationLevel.NONE):
        return passes.canonicalize_io_order()(program)


class TestCanonicalizeIOOrder:
    """Before/Expected pairs verifying the priority-aware topological reorder."""

    def test_symmetric_pingpong_layout(self):
        """[load_0, compute_0, store_0, load_1, compute_1, store_1] →
        [load_0, load_1, compute_0, compute_1, store_0, store_1]."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    tc0: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    pl.tile.store(tc0, [0, 0], out)
                    ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    tc1: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta1, ta1)
                    pl.tile.store(tc1, [64, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    tc0: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    tc1: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta1, ta1)
                    pl.tile.store(tc0, [0, 0], out)
                    pl.tile.store(tc1, [64, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_scalar_offset_lifts_above_independent_load(self):
        """Scalar compute lifts above loads (cat 0 < cat 1) while still preceding
        any load that depends on it. ``off`` floats to the top; ``ta2`` stays
        below ``off``; ``ta`` (independent) follows once ``off`` is emitted."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    ta: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    off: pl.Scalar[pl.INDEX] = 64
                    ta2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [off, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta, ta2)
                    pl.tile.store(tc, [0, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    off: pl.Scalar[pl.INDEX] = 64
                    ta: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    ta2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [off, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta, ta2)
                    pl.tile.store(tc, [0, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_scalar_compute_lifts_to_top_unblocking_loads(self):
        """Per-clone scalar address-arithmetic lifts above all loads, allowing
        sibling clones' loads to cluster at the top.

        Without ``ScalarCompute`` priority, ``off1`` (idx 4) would only emit
        after group 0's compute and store, and ``t1`` would never reach the
        load cluster. With it, both ``off0`` and ``off1`` go first, both loads
        cluster, then both computes, then both stores — the layout that
        ``MemoryReuse`` needs for ping-pong buffering."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[256, 64], pl.FP32], out: pl.Tensor[[256, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    off0: pl.Scalar[pl.INDEX] = i * 64
                    t0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [off0, 0], [64, 64])
                    c0: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(t0, t0)
                    pl.tile.store(c0, [off0, 0], out)
                    off1: pl.Scalar[pl.INDEX] = (i + 1) * 64
                    t1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [off1, 0], [64, 64])
                    c1: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(t1, t1)
                    pl.tile.store(c1, [off1, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[256, 64], pl.FP32], out: pl.Tensor[[256, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    off0: pl.Scalar[pl.INDEX] = i * 64
                    off1: pl.Scalar[pl.INDEX] = (i + 1) * 64
                    t0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [off0, 0], [64, 64])
                    t1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [off1, 0], [64, 64])
                    c0: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(t0, t0)
                    c1: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(t1, t1)
                    pl.tile.store(c0, [off0, 0], out)
                    pl.tile.store(c1, [off1, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_already_ordered_region_is_noop(self):
        """A region already in canonical [load, compute, store] order is unchanged —
        the reorder preserves IR identity when no swap would help."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    ta: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta, ta)
                    pl.tile.store(tc, [0, 0], out)

        After = _run_pass(Before)
        # IR identity preserved — the reorder detects no change is needed.
        assert After is Before

    def test_function_body_outside_pipeline_is_not_reordered(self):
        """Scope check: a function body with interleaved load/store — but no
        enclosing ``ForKind.Pipeline`` — must be left untouched. This is the
        key difference from the pre-refactor behavior, where the reorder ran
        at every SeqStmts including the function body itself."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                # Interleaved load/store at function scope — without an
                # enclosing pipeline loop, the pass does not reorder this.
                ta: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                pl.tile.store(ta, [0, 0], out)
                tb: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                pl.tile.store(tb, [0, 0], out)

        After = _run_pass(Before)
        # No pipeline scope → identity preserved.
        assert After is Before

    def test_no_io_ops_is_noop(self):
        """A region with neither loads nor stores is unchanged."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, a: pl.Tile[[64, 64], pl.FP32], b: pl.Tile[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    _x: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(a, a)
                    _y: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(b, b)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, a: pl.Tile[[64, 64], pl.FP32], b: pl.Tile[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    _x: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(a, a)
                    _y: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(b, b)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_store_and_write_both_sink_to_bottom(self):
        """Both ``tile.store`` and ``tile.write`` are categorized as writes and
        sink to the bottom — interleaved input is clustered into loads-then-writes."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    t1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    pl.tile.store(t1, [0, 0], out)
                    t2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    pl.tile.write(t2, [0, 0], 7.0)  # pyright: ignore[reportArgumentType]

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    t1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    t2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    pl.tile.store(t1, [0, 0], out)
                    pl.tile.write(t2, [0, 0], 7.0)  # pyright: ignore[reportArgumentType]

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_load_and_read_both_lift_to_top(self):
        """``tile.read`` (scalar read from a tile) is categorized as a read and
        lifts to the top alongside ``tile.load`` — both beat compute and store.

        The load must appear first in the source (DSL requires defined-before-use),
        but the read and store can be reordered relative to each other. The pass
        should lift the read above the store."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    # Store placed before read in source — reorder should swap them.
                    pl.tile.store(t, [0, 0], out)
                    _elem: pl.Scalar[pl.FP32] = pl.tile.read(t, [0, 0])

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    # tile.read (Load category) lifts above the store.
                    _elem: pl.Scalar[pl.FP32] = pl.tile.read(t, [0, 0])
                    # tile.store sinks to the bottom.
                    pl.tile.store(t, [0, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_relative_order_preserved_among_independent_loads(self):
        """3 independent loads keep their original relative order after lifting."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[192, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    _ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    _ta2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [128, 0], [64, 64])
                    pl.tile.store(tc, [0, 0], out)

        # ta0, ta1, ta2 are independent → all cluster at the top in original
        # relative order. tc (reads ta0 only) follows. store follows last.
        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[192, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    _ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    _ta2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [128, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    pl.tile.store(tc, [0, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_l1_to_l0_extract_clusters_above_matmul(self):
        """`tile.extract` with Mat source and Left/Right target is the ISA
        TEXTRACT L1→L0 data-movement op. Reordering it into the Load tier lets
        all the extracts in an iteration body cluster ahead of their matmul_acc
        consumers — the precondition for L1→L0 ping-pong on Left/Right buffers
        (analogous to how tile.load clustering enables DDR→Mat ping-pong).

        Mirrors the qwen3_decode q_proj inner K-loop body: two unrolled
        ``extract→extract→matmul_acc`` triplets must rewrite to all four
        extracts followed by both matmul_accs.
        """

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[16, 128], pl.BF16], in_b: pl.Tensor[[128, 256], pl.BF16]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    a_mat: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        in_a, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                    )
                    b_mat: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        in_b, [0, 0], [128, 256], target_memory=pl.Mem.Mat
                    )
                    a0: pl.Tile[[16, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a_mat, 0, 0, [16, 64], target_memory=pl.Mem.Left
                    )
                    b0: pl.Tile[[64, 256], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_mat, 0, 0, [64, 256], target_memory=pl.Mem.Right
                    )
                    acc0: pl.Tile[[16, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a0, b0)
                    a1: pl.Tile[[16, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a_mat, 0, 64, [16, 64], target_memory=pl.Mem.Left
                    )
                    b1: pl.Tile[[64, 256], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_mat, 64, 0, [64, 256], target_memory=pl.Mem.Right
                    )
                    _acc1: pl.Tile[[16, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc0, a1, b1)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[16, 128], pl.BF16], in_b: pl.Tensor[[128, 256], pl.BF16]):
                for i in pl.range(0, 2, 1):
                    a_mat: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        in_a, [0, 0], [16, 128], target_memory=pl.Mem.Mat
                    )
                    b_mat: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        in_b, [0, 0], [128, 256], target_memory=pl.Mem.Mat
                    )
                    a0: pl.Tile[[16, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a_mat, 0, 0, [16, 64], target_memory=pl.Mem.Left
                    )
                    b0: pl.Tile[[64, 256], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_mat, 0, 0, [64, 256], target_memory=pl.Mem.Right
                    )
                    a1: pl.Tile[[16, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a_mat, 0, 64, [16, 64], target_memory=pl.Mem.Left
                    )
                    b1: pl.Tile[[64, 256], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_mat, 64, 0, [64, 256], target_memory=pl.Mem.Right
                    )
                    acc0: pl.Tile[[16, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a0, b0)
                    _acc1: pl.Tile[[16, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc0, a1, b1)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_non_l1_to_l0_extract_stays_in_compute_tier(self):
        """`tile.extract` whose target is **not** Left/Right (here: Acc) must
        stay in the TileCompute tier and not promote to Load.

        The L1→L0 promotion is keyed on src=Mat ∧ target∈{Left,Right} —
        analogous extracts to other memory spaces (Acc spill, Vec view, etc.)
        don't represent the matmul-input prefetch pattern and shouldn't be
        lifted above TileCompute consumers.

        Verified by an extract that depends on the matmul's Acc result: if the
        promotion mis-fired, the topo sort would still land in this order
        (deps force it), so we use a sibling Acc-target extract that is
        independent of the matmul. With the correct categorization the extract
        stays after the matmul (stable order within TileCompute, original
        position preserved).
        """

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[16, 64], pl.BF16], in_b: pl.Tensor[[64, 256], pl.BF16]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    a_mat: pl.Tile[[16, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        in_a, [0, 0], [16, 64], target_memory=pl.Mem.Mat
                    )
                    b_mat: pl.Tile[[64, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        in_b, [0, 0], [64, 256], target_memory=pl.Mem.Mat
                    )
                    a_left: pl.Tile[[16, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a_mat, 0, 0, [16, 64], target_memory=pl.Mem.Left
                    )
                    b_right: pl.Tile[[64, 256], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_mat, 0, 0, [64, 256], target_memory=pl.Mem.Right
                    )
                    _acc: pl.Tile[[16, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a_left, b_right)
                    # Mat→Acc extract — independent of the matmul, but NOT
                    # the L1→L0a/b prefetch pattern, so it must stay in
                    # TileCompute (after the matmul in the original sequence).
                    _spare: pl.Tile[[16, 256], pl.BF16, pl.Mem.Acc] = pl.tile.extract(
                        b_mat, 0, 0, [16, 256], target_memory=pl.Mem.Acc
                    )

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[16, 64], pl.BF16], in_b: pl.Tensor[[64, 256], pl.BF16]):
                for i in pl.range(0, 2, 1):
                    # Loads + L1→L0a/b extracts cluster at the top.
                    a_mat: pl.Tile[[16, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        in_a, [0, 0], [16, 64], target_memory=pl.Mem.Mat
                    )
                    b_mat: pl.Tile[[64, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        in_b, [0, 0], [64, 256], target_memory=pl.Mem.Mat
                    )
                    a_left: pl.Tile[[16, 64], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        a_mat, 0, 0, [16, 64], target_memory=pl.Mem.Left
                    )
                    b_right: pl.Tile[[64, 256], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        b_mat, 0, 0, [64, 256], target_memory=pl.Mem.Right
                    )
                    # The Mat→Acc extract stays in TileCompute and so keeps
                    # its original (post-matmul) position.
                    _acc: pl.Tile[[16, 256], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a_left, b_right)
                    _spare: pl.Tile[[16, 256], pl.BF16, pl.Mem.Acc] = pl.tile.extract(
                        b_mat, 0, 0, [16, 256], target_memory=pl.Mem.Acc
                    )

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_yield_terminator_stays_last_after_store_sinks(self):
        """A trailing ``YieldStmt`` is peeled off the region, the remaining
        non-terminator stmts are priority-sorted, then the terminator is
        re-appended last — so a sunk ``tile.store`` can never end up after the
        yield (which would make the store unreachable).

        Body in source order is ``[load t, store(t), nxt = acc + i, yield(nxt)]``.
        Categories of the 3 non-terminators: ``t``=Load(1), ``store``=Store(3),
        ``nxt``=ScalarCompute(0). Dependency edges within the region:
        ``store`` ← ``t``; ``nxt`` reads loop-level ``acc``/``i`` only, so it has
        no intra-region predecessor. Priority-aware topo sort emits
        ``nxt`` (cat 0) → ``t`` (cat 1) → ``store`` (cat 3); the peeled
        ``yield`` is re-appended last. (pass source: IsTerminator + ReorderRegion
        terminator peeling, lines 137-142, 227-282.)
        """

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(
                self,
                in_a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Tensor[[64, 64], pl.FP32],
                s0: pl.Scalar[pl.INDEX],
            ) -> pl.Scalar[pl.INDEX]:
                for i, (acc,) in pl.pipeline(0, 2, 1, stage=1, init_values=(s0,)):
                    t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    pl.tile.store(t, [0, 0], out)
                    nxt: pl.Scalar[pl.INDEX] = acc + i
                    r = pl.yield_(nxt)
                return r

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self,
                in_a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Tensor[[64, 64], pl.FP32],
                s0: pl.Scalar[pl.INDEX],
            ) -> pl.Scalar[pl.INDEX]:
                for i, (acc,) in pl.range(0, 2, 1, init_values=(s0,)):
                    # ScalarCompute lifts to the top.
                    nxt: pl.Scalar[pl.INDEX] = acc + i
                    # Load follows.
                    t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    # Store sinks — but only among non-terminators.
                    pl.tile.store(t, [0, 0], out)
                    # The terminator stays absolutely last.
                    r = pl.yield_(nxt)
                return r

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_pipeline_reorders_inner_and_demotes_both(self):
        """A pipeline loop nested inside another pipeline loop: the inner body
        (pipeline-depth 2) is reordered, and BOTH loops are demoted to
        ``ForKind::Sequential`` on exit.

        The pass keeps an ``inside_pipeline_depth_`` counter that increments per
        nested pipeline and reorders any SeqStmts while it is non-zero (pass
        source lines 171-203). The demotion runs once per ``ForKind::Pipeline``
        loop, so both the outer and inner loops become ``pl.range``.

        Inner body ``[load t, store(t), load t2, add(t2)]`` reorders to
        loads-clustered-then-compute-then-store:
        ``t`` (Load) and ``t2`` (Load) lift; ``_c`` (TileCompute) settles in the
        middle; ``store(t)`` (Store) sinks. ``t``/``t2`` are independent, so
        their original relative order is preserved.
        """

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    for j in pl.pipeline(0, 2, 1, stage=1):
                        t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                        pl.tile.store(t, [0, 0], out)
                        t2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                        _c: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(t2, t2)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                # Both loops demoted Pipeline → Sequential.
                for i in pl.range(0, 2, 1):
                    for j in pl.range(0, 2, 1):
                        t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                        t2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                        _c: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(t2, t2)
                        pl.tile.store(t, [0, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_inout_discipline_violation_skips_whole_function(self):
        """A function that violates the InOut-use discipline is skipped entirely —
        the pass neither reorders its pipeline body nor demotes the pipeline loop.

        The driver runs ``CollectInOutUseDisciplineDiagnostics`` once per function
        and, when non-empty, re-emits the original function unchanged (pass source
        lines 316-319). Here ``main`` passes ``x`` as ``InOut`` to ``mutate`` and
        then reads the pre-call ``x`` again (``y = pl.add(x, x)``) — a violation.
        Even though the pipeline body below WOULD be reordered (interleaved
        load/store), the whole function is left untouched, so the program is
        returned by identity.
        """

        @pl.program
        class Before:
            @pl.function
            def mutate(self, T: pl.InOut[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
                r: pl.Tensor[[64, 64], pl.FP32] = pl.add(T, T)
                return r

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                in_a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                _t_new: pl.Tensor[[64, 64], pl.FP32] = self.mutate(x)
                y: pl.Tensor[[64, 64], pl.FP32] = pl.add(x, x)  # VIOLATION: reads x after InOut pass
                for i in pl.pipeline(0, 2, 1, stage=1):
                    t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    pl.tile.store(t, [0, 0], out)
                    t2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    _c: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(t2, t2)
                return y

        After = _run_pass(Before)
        # Violation → whole-program identity (no reorder, no Pipeline→Sequential demotion).
        assert After is Before

    def test_submit_inside_pipeline_is_tile_compute_not_io(self):
        """``pl.submit`` inside a ``pl.manual_scope`` within a pipeline body: the
        Submit-valued ``AssignStmt`` must be categorized as ``TileCompute`` (not
        misclassified as Load/Store), while its ``TASK_ID`` tuple projection is a
        ``ScalarType`` assign and so lifts as ``ScalarCompute``. The Submit's
        ``deps_`` edge must be honored by the topo sort.

        ``CategorizeStmt`` only recognizes Load/Store on ``AssignStmt`` whose value
        is a ``Call`` (pass source lines 106-117). A ``Submit`` is a sibling kind,
        not a ``Call``, so it falls through to the LHS-type check: the tuple-typed
        ``_submit_tmp`` LHS is not scalar → ``TileCompute``.

        Body in source order (after DSL lowering) is::

            _submit_tmp = Submit(k1, x)   # TileCompute (Submit, not Call)
            a   = _submit_tmp[0]          # TileCompute (Tensor)
            a_tid = _submit_tmp[1]        # ScalarCompute (TASK_ID scalar)
            _submit_tmp = Submit(k1, x, deps=[a_tid])  # TileCompute
            b   = _submit_tmp[0]          # TileCompute
            b_tid = _submit_tmp[1]        # ScalarCompute

        Hand-derived priority-aware topo sort (deps: a/a_tid ← first submit;
        second submit ← a_tid; b/b_tid ← second submit) emits::

            [submit_0, a_tid, a, submit_1, b_tid, b]

        i.e. each ``*_tid`` (ScalarCompute) lifts above its sibling tensor
        projection, the second submit stays after ``a_tid`` (deps honored), and
        the outer pipeline loop is demoted to Sequential.

        The reorder swaps tuple projections, which the DSL emission order cannot
        express directly, so this asserts the demotion + the by-hand-derived
        statement order via structural inspection (not a snapshot).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    for i in pl.pipeline(0, 2, 1, stage=1):
                        _a, a_tid = pl.submit(self.k1, x)
                        _b, _b_tid = pl.submit(self.k1, x, deps=[a_tid])
                return out

        After = _run_pass(Before)
        assert After is not Before  # the reorder + demotion changed the IR

        # Locate the (now-Sequential) loop's body within the RuntimeScopeStmt.
        fn = After.get_function("main")
        assert fn is not None

        loop = None

        def find_loop(stmt):
            nonlocal loop
            if isinstance(stmt, ir.ForStmt):
                loop = stmt
                return
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    find_loop(s)
            elif isinstance(stmt, ir.RuntimeScopeStmt):
                find_loop(stmt.body)

        find_loop(fn.body)
        assert loop is not None, "expected the pipeline loop in main"
        # The transient Pipeline marker must be demoted to Sequential.
        assert loop.kind == ir.ForKind.Sequential

        body = loop.body
        assert isinstance(body, ir.SeqStmts)
        stmts = list(body.stmts)
        assert len(stmts) == 6

        # Hand-derived order: [submit_0, a_tid, a, submit_1, b_tid, b].
        def is_submit(s):
            return isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Submit)

        def is_proj(s):
            return isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.TupleGetItemExpr)

        def is_scalar_assign(s):
            return is_proj(s) and isinstance(s.var.type, ir.ScalarType)

        # submit_0 stays first (it has no intra-region predecessor).
        assert is_submit(stmts[0])
        # a_tid (ScalarCompute) lifts above a (TileCompute tensor projection).
        assert is_scalar_assign(stmts[1])
        assert is_proj(stmts[2]) and not is_scalar_assign(stmts[2])
        # submit_1 follows a_tid (deps=[a_tid] edge honored).
        assert is_submit(stmts[3])
        # b_tid (ScalarCompute) lifts above b.
        assert is_scalar_assign(stmts[4])
        assert is_proj(stmts[5]) and not is_scalar_assign(stmts[5])


def _run_lower_then_canon(program: ir.Program) -> ir.Program:
    """Replicate (LowerPipelineLoops) then reorder (CanonicalizeIOOrder) — the
    real pipeline order — with structural verification disabled."""
    with passes.PassContext([], passes.VerificationLevel.NONE):
        lowered = passes.lower_pipeline_loops()(program)
        return passes.canonicalize_io_order()(lowered)


def _positions(text: str, marker: str) -> list[int]:
    """Character offsets of every occurrence of ``marker`` in ``text``."""
    out: list[int] = []
    start = 0
    while (i := text.find(marker, start)) != -1:
        out.append(i)
        start = i + 1
    return out


def _pos(text: str, marker: str) -> int:
    """Offset of the single/first occurrence of ``marker`` (asserts presence)."""
    p = _positions(text, marker)
    assert p, f"marker {marker!r} not found in:\n{text}"
    return p[0]


class TestCrossCorePipeline:
    """Cross-core (AIC↔AIV) stage clustering for fused cube/vector pipelines (#1610).

    The pass generalizes its hardware-unit stage ladder across the cross-core
    boundary: producer compute (tier 2) < cross-core push (3) < cross-core pop
    (4) < consumer compute (5). Clustering each stage across the replicated
    clones keeps sibling tiles co-live so MemoryReuse can ping-pong them.
    """

    def test_aic_stages_cluster_for_pingpong(self):
        """AIC body ``QK -> tpush -> tpop -> SV`` (2 clones) regroups into
        ``loads | QK0,QK1 | tpush0,tpush1 | tpop0,tpop1 | SV0,SV1 | stores``."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[32, 64], pl.FP32], out: pl.Tensor[[32, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [0, 0], out)
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(oi1, [16, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[32, 64], pl.FP32], out: pl.Tensor[[32, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [16, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(oi0, [0, 0], out)
                    pl.tile.store(oi1, [16, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_aiv_consumer_side_clusters(self):
        """AIV body ``tpop_from_aic -> softmax -> tpush_to_aic`` (2 clones):
        both receives cluster ahead of the first consumer compute."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    scores0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    sm0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(scores0, scores0)
                    pl.tile.tpush_to_aic(sm0, split=0)
                    scores1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    sm1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(scores1, scores1)
                    pl.tile.tpush_to_aic(sm1, split=0)

        text = ir.python_print(_run_pass(Before))
        # Both tpop receives precede the first consumer compute -> received-score
        # buffers stay co-live (ping-pong on the consumer side too).
        assert _pos(text, "scores0") < _pos(text, "sm0")
        assert _pos(text, "scores1") < _pos(text, "sm0")

    def test_three_clone_cross_core_clusters(self):
        """Clustering generalizes past 2 clones: all QKs, then all tpushes, then
        all tpops, then all SVs."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[48, 64], pl.FP32], out: pl.Tensor[[48, 64], pl.FP32]):
                for i in pl.pipeline(0, 3, 1, stage=1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [0, 0], out)
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(oi1, [16, 0], out)
                    qa2: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [32, 0], [16, 64])
                    rs2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa2, qa2)
                    pl.tile.tpush_to_aiv(rs2, split=0)
                    e2: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e2, e2)
                    pl.tile.store(oi2, [32, 0], out)

        text = ir.python_print(_run_pass(Before))
        push = _positions(text, "tpush_to_aiv")
        pop = _positions(text, "tpop_from_aiv")
        assert len(push) == 3 and len(pop) == 3
        # All three producers (rs*) precede all three pushes; all pushes precede
        # all pops; the last push precedes the first pop (full stage grouping).
        assert max(_pos(text, n) for n in ("rs0", "rs1", "rs2")) < min(push)
        assert max(push) < min(pop)
        # All three pops precede the first consumer compute (oi0).
        assert max(pop) < _pos(text, "oi0")

    def test_sv_init_create_stays_with_consumer(self):
        """A tile.create that only feeds consumer-stage compute is NOT hoisted
        into the producer cluster — it stays after the pushes/pops with its SV."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[32, 64], pl.FP32], out: pl.Tensor[[32, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    sv_init0: pl.Tile[[16, 64], pl.FP32] = pl.tile.create([16, 64], dtype=pl.FP32)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, sv_init0)
                    pl.tile.store(oi0, [0, 0], out)
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    sv_init1: pl.Tile[[16, 64], pl.FP32] = pl.tile.create([16, 64], dtype=pl.FP32)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, sv_init1)
                    pl.tile.store(oi1, [16, 0], out)

        text = ir.python_print(_run_pass(Before))
        # Each SV-init create lands after the last tpush (consumer side), not in
        # the producer cluster ahead of the sends.
        last_push = max(_positions(text, "tpush_to_aiv"))
        assert _pos(text, "sv_init0") > last_push
        assert _pos(text, "sv_init1") > last_push

    def test_tpush_is_not_sunk_like_store(self):
        """tpush is egress but must NOT sink like tile.store: it stays before the
        pop, while the store sinks to the very bottom."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[16, 64], pl.FP32], out: pl.Tensor[[16, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [0, 0], out)

        text = ir.python_print(_run_pass(Before))
        assert _pos(text, "tpush_to_aiv") < _pos(text, "tpop_from_aiv")
        assert _pos(text, "tpop_from_aiv") < _pos(text, "tile.store")

    def test_loads_and_stores_unaffected_by_cross_core(self):
        """Load stays at the top tier and store at the bottom even with the
        cross-core stages present in between."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[32, 64], pl.FP32], out: pl.Tensor[[32, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [0, 0], out)
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(oi1, [16, 0], out)

        text = ir.python_print(_run_pass(Before))
        loads = _positions(text, "pl.tile.load")
        stores = _positions(text, "pl.tile.store")
        push = _positions(text, "tpush_to_aiv")
        pop = _positions(text, "tpop_from_aiv")
        # Every load precedes every cross-core op; every cross-core op precedes
        # every store.
        assert max(loads) < min(push + pop)
        assert max(push + pop) < min(stores)

    def test_consumer_chain_through_intermediate(self):
        """``after_pop`` is transitive: an intermediate compute between the pop
        and the final consumer is also placed in the consumer cluster."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[32, 64], pl.FP32], out: pl.Tensor[[32, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    mid0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(mid0, mid0)
                    pl.tile.store(oi0, [0, 0], out)
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    mid1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(mid1, mid1)
                    pl.tile.store(oi1, [16, 0], out)

        text = ir.python_print(_run_pass(Before))
        # The intermediate (mid*) and final (oi*) consumer computes all follow
        # the producer cluster (rs*) and the pushes.
        last_push = max(_positions(text, "tpush_to_aiv"))
        for name in ("mid0", "oi0", "mid1", "oi1"):
            assert _pos(text, name) > last_push

    def test_tfree_follows_its_consumer(self):
        """tfree (consumer-tier) is emitted after the SV compute that last-uses
        the popped tile."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[16, 64], pl.FP32], out: pl.Tensor[[16, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.system.tfree_to_aiv(e0)
                    pl.tile.store(oi0, [0, 0], out)

        text = ir.python_print(_run_pass(Before))
        assert _pos(text, "oi0") < _pos(text, "tfree_to_aiv")

    def test_fifo_and_causal_order_preserved(self):
        """Per-pipe FIFO (push0<push1, pop0<pop1) and causality (every push
        before every pop) hold after the reorder."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[32, 64], pl.FP32], out: pl.Tensor[[32, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [0, 0], out)
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(oi1, [16, 0], out)

        text = ir.python_print(_run_pass(Before))
        push = _positions(text, "tpush_to_aiv")
        pop = _positions(text, "tpop_from_aiv")
        assert push == sorted(push) and pop == sorted(pop)  # stable per-pipe order
        assert max(push) < min(pop)  # every push before every pop

    def test_no_cross_core_pipeline_unchanged(self):
        """A pure load/compute/store pipeline (no cross-core ops) reorders
        exactly as before — guards the Store-tier renumber."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    tc0: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    pl.tile.store(tc0, [0, 0], out)
                    ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    tc1: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta1, ta1)
                    pl.tile.store(tc1, [64, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    tc0: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    tc1: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta1, ta1)
                    pl.tile.store(tc0, [0, 0], out)
                    pl.tile.store(tc1, [64, 0], out)

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_pipeline_stage2_replicate_then_reorder(self):
        """End-to-end IR flow: a single cross-core body under ``stage=2`` is
        replicated by LowerPipelineLoops then stage-grouped by CanonicalizeIOOrder."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=2):
                    off: pl.Scalar[pl.INDEX] = i * 16
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [off, 0], [16, 64])
                    rs: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.tpush_to_aiv(rs, split=0)
                    e: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e, e)
                    pl.tile.store(oi, [off, 0], out)

        text = ir.python_print(_run_lower_then_canon(Before))
        push = _positions(text, "tpush_to_aiv")
        pop = _positions(text, "tpop_from_aiv")
        # stage=2 replicates the body; the two pushes cluster before the two pops.
        assert len(push) == 2 and len(pop) == 2
        assert max(push) < min(pop)

    def test_consumer_only_l0_move_defers_to_consumer(self):
        """A `tile.move` into L0 (Right) that feeds only the post-pop SV matmul is
        NOT hoisted into the producer cluster — it is demoted to ConsumerCompute and
        sits with its consumer, so MemoryReuse can share one L0 buffer per clone
        instead of partitioning L0 across clones (#1610 follow-up)."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(
                self,
                q: pl.Tensor[[32, 64], pl.FP32],
                vt: pl.Tensor[[128, 64], pl.FP32],
                out: pl.Tensor[[32, 64], pl.FP32],
            ):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    q0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [0, 0], [16, 64])
                    rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(q0, q0)
                    pl.tile.tpush_to_aiv(rs0, split=0)
                    v0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                        vt, [0, 0], [64, 64], target_memory=pl.Mem.Mat
                    )
                    e0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Left] = pl.tile.tpop_from_aiv(split=0)
                    v_right0: pl.Tile[[64, 64], pl.FP32, pl.Mem.Right] = pl.tile.move(
                        v0, target_memory=pl.Mem.Right
                    )
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(e0, v_right0)
                    pl.tile.store(oi0, [0, 0], out)
                    q1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(q1, q1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    v1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                        vt, [0, 0], [64, 64], target_memory=pl.Mem.Mat
                    )
                    e1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Left] = pl.tile.tpop_from_aiv(split=0)
                    v_right1: pl.Tile[[64, 64], pl.FP32, pl.Mem.Right] = pl.tile.move(
                        v1, target_memory=pl.Mem.Right
                    )
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(e1, v_right1)
                    pl.tile.store(oi1, [16, 0], out)

        text = ir.python_print(_run_pass(Before))
        last_push = max(_positions(text, "tpush_to_aiv"))
        last_pop = max(_positions(text, "tpop_from_aiv"))
        # Both V→Right moves are deferred past the pushes AND the pops (consumer
        # tier), i.e. not hoisted into the producer cluster.
        assert _pos(text, "v_right0") > last_push
        assert _pos(text, "v_right1") > last_push
        assert _pos(text, "v_right0") > last_pop
        assert _pos(text, "v_right1") > last_pop
        # Producer compute (the QK stand-ins) still clusters ahead of the pushes.
        assert max(_pos(text, "rs0"), _pos(text, "rs1")) < last_push

    def test_consumer_phase_tpush_not_hoisted_over_trailing_compute(self):
        """A consumer-phase ``tpush_to_aic`` (V2C send, downstream of a ``tpop``)
        must NOT be hoisted ahead of sibling consumer compute — the AIV softmax
        case. Hoisting it (early CrossCorePush tier) shortens the pushed tile's
        live-range and races the async transfer against buffer reuse, stalling the
        AICPU sync on stricter runtimes (#1610). It stays in the consumer tier."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, out: pl.Tensor[[32, 64], pl.FP32]):
                for i in pl.pipeline(0, 2, 1, stage=1):
                    sc0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
                    sent0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(sc0)
                    tail0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(sc0, sc0)
                    pl.tile.tpush_to_aic(sent0, split=0)
                    pl.tile.store(tail0, [0, 0], out)
                    sc1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
                    sent1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(sc1)
                    tail1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(sc1, sc1)
                    pl.tile.tpush_to_aic(sent1, split=0)
                    pl.tile.store(tail1, [16, 0], out)

        text = ir.python_print(_run_pass(Before))
        push = _positions(text, "tpush_to_aic")
        # Each clone's trailing consumer compute (tailN, which precedes its tpush
        # in program order) is NOT jumped over by the consumer-phase tpush.
        assert _pos(text, "tail0") < push[0]
        assert _pos(text, "tail1") < push[1]
        # The receives still cluster (CrossCorePop) ahead of all the sends.
        assert max(_positions(text, "tpop_from_aic")) < min(push)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
