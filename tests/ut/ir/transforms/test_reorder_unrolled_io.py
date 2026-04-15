# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSL-style Before/Expected tests for the ReorderUnrolledIO pass.

The pass walks ``ForStmt``s carrying ``attrs_["unroll_replicated"]`` and
reorders their top-level SeqStmts so tile.load floats to the top, tile.store
sinks to the bottom, compute settles in the middle — subject to the SSA
dependency graph.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_pass(program: ir.Program) -> ir.Program:
    """Run ReorderUnrolledIO with structural verification disabled — our
    Before programs use minimal tile IR that doesn't satisfy the full set of
    structural prerequisites the pipeline normally enforces."""
    with passes.PassContext([], passes.VerificationLevel.NONE):
        return passes.reorder_unrolled_io()(program)


class TestReorderUnrolledIO:
    """Before/Expected pairs verifying the priority-aware topological reorder."""

    def test_symmetric_pingpong_layout(self):
        """[load_0, compute_0, store_0, load_1, compute_1, store_1] →
        [load_0, load_1, compute_0, compute_1, store_0, store_1]."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
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
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    tc0: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    tc1: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta1, ta1)
                    pl.tile.store(tc0, [0, 0], out)
                    pl.tile.store(tc1, [64, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_load_blocked_by_compute_stays_put(self):
        """A load whose offset depends on a compute stmt cannot float past it."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    ta: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    off: pl.Scalar[pl.INDEX] = 64
                    ta2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [off, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta, ta2)
                    pl.tile.store(tc, [0, 0], out)

        # `ta` (independent) lifts to the top. `off` must still precede `ta2`
        # (ta2 reads `off`). `ta2` stays below `off`. Computes and stores after.
        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    ta: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    off: pl.Scalar[pl.INDEX] = 64
                    ta2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [off, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta, ta2)
                    pl.tile.store(tc, [0, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_skip_unmarked_for(self):
        """A ForStmt without ``unroll_replicated`` is left untouched."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                # No attrs — pass should ignore.
                for i in pl.range(0, 2, 1):
                    ta: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta, ta)
                    pl.tile.store(tc, [0, 0], out)

        After = _run_pass(Before)
        # IR identity preserved when no marker is present.
        assert After is Before

    def test_no_io_ops_is_noop(self):
        """A marked region with neither loads nor stores is unchanged."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, a: pl.Tile[[64, 64], pl.FP32], b: pl.Tile[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    x: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(a, a)  # noqa: F841
                    y: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(b, b)  # noqa: F841

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, a: pl.Tile[[64, 64], pl.FP32], b: pl.Tile[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    x: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(a, a)  # noqa: F841
                    y: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(b, b)  # noqa: F841

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_store_and_write_both_sink_to_bottom(self):
        """Both ``tile.store`` and ``tile.write`` are categorized as writes and
        sink to the bottom — interleaved input is clustered into loads-then-writes."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    t1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    pl.tile.store(t1, [0, 0], out)
                    t2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])
                    pl.tile.write(t2, [0, 0], 7.0)  # pyright: ignore[reportArgumentType]

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
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
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    # Store placed before read in source — reorder should swap them.
                    pl.tile.store(t, [0, 0], out)
                    elem: pl.Scalar[pl.FP32] = pl.tile.read(t, [0, 0])  # noqa: F841

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    t: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    # tile.read (Load category) lifts above the store.
                    elem: pl.Scalar[pl.FP32] = pl.tile.read(t, [0, 0])  # noqa: F841
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
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])  # noqa: F841
                    ta2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [128, 0], [64, 64])  # noqa: F841
                    pl.tile.store(tc, [0, 0], out)

        # ta0, ta1, ta2 are independent → all cluster at the top in original
        # relative order. tc (reads ta0 only) follows. store follows last.
        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, in_a: pl.Tensor[[192, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.range(0, 2, 1, attrs={"unroll_replicated": 2}):
                    ta0: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [0, 0], [64, 64])
                    ta1: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [64, 0], [64, 64])  # noqa: F841
                    ta2: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(in_a, [128, 0], [64, 64])  # noqa: F841
                    tc: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(ta0, ta0)
                    pl.tile.store(tc, [0, 0], out)

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
