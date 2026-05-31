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
reorders its top-level statements into four priority tiers — scalar compute
first, then tile.load, then tile compute, and finally tile.store — all subject
to the SSA dependency graph. Loops that are not pipelined are left untouched.

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
