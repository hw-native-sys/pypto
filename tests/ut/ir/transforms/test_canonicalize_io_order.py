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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
