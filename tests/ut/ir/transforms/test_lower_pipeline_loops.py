# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSL-style Before/Expected tests for the LowerPipelineLoops pass.

The pass triggers on any ``ForStmt`` with ``kind_ == ForKind::Pipeline`` and
``attrs_["pipeline_stages"] == F``. Inputs use ``pl.pipeline(N, stage=F)`` —
the user-facing DSL surface — so these tests exercise the full parse →
lower → canonicalize chain against Expected programs written as plain
``pl.range`` (which is what the IR reduces to once the Pipeline marker is
demoted by CanonicalizeIOOrder).
"""

import re

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_pass(program: ir.Program) -> ir.Program:
    """Run ``LowerPipelineLoops`` + ``CanonicalizeIOOrder``. Canonicalize runs
    to demote the transient ``ForKind::Pipeline`` marker back to ``Sequential``
    so the Expected programs (written in plain ``pl.range``) can be compared
    structurally. Canonicalize is a no-op on the scalar-only bodies used in
    these tests (single tier, stable order). Verification stays on so the
    autouse RoundtripInstrument exercises the post-LowerPipelineLoops state."""
    lowered = passes.lower_pipeline_loops()(program)
    return passes.canonicalize_io_order()(lowered)


class TestLowerPipelineMechanics:
    """Before/Expected pairs verifying the cloning + outer-stride rewriting logic."""

    def test_clean_divide_produces_replicated_outer_loop(self):
        """trip=8, factor=4 → outer range(0, 8, 4) with 4 clones, no remainder."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(0, 8, 1, stage=4):
                    _y: pl.Scalar[pl.INDEX] = i
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 8, 4):
                    _y: pl.Scalar[pl.INDEX] = i
                    _y_1: pl.Scalar[pl.INDEX] = i + 1
                    _y_2: pl.Scalar[pl.INDEX] = i + 2
                    _y_3: pl.Scalar[pl.INDEX] = i + 3
                return x

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_with_remainder_appends_tail_branch(self):
        """trip=10, factor=4 → main range(0, 8, 4) with 4 clones + bare 2-clone tail.

        Static path knows rem_iters=2 at compile time — the tail clones are
        flattened directly into the outer scope (no wrapper, no marker attr)."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(0, 10, 1, stage=4):
                    _y: pl.Scalar[pl.INDEX] = i
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 8, 4):
                    _y: pl.Scalar[pl.INDEX] = i
                    _y_1: pl.Scalar[pl.INDEX] = i + 1
                    _y_2: pl.Scalar[pl.INDEX] = i + 2
                    _y_3: pl.Scalar[pl.INDEX] = i + 3
                _y_4: pl.Scalar[pl.INDEX] = 8
                _y_5: pl.Scalar[pl.INDEX] = 9
                return x

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_factor_one_is_noop(self):
        """stage=1 leaves the loop intact (modulo attr cleanup + kind demotion)."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(0, 8, 1, stage=1):
                    _y: pl.Scalar[pl.INDEX] = i
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Same range as Before, attr dropped.
                for i in pl.range(0, 8, 1):
                    _y: pl.Scalar[pl.INDEX] = i
                return x

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_factor_equals_trip_count(self):
        """factor=4, trip=4 → single outer iteration containing 4 clones, no remainder."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(0, 4, 1, stage=4):
                    _y: pl.Scalar[pl.INDEX] = i
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 4, 4):
                    _y: pl.Scalar[pl.INDEX] = i
                    _y_1: pl.Scalar[pl.INDEX] = i + 1
                    _y_2: pl.Scalar[pl.INDEX] = i + 2
                    _y_3: pl.Scalar[pl.INDEX] = i + 3
                return x

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_dynamic_stop_lowers_to_main_plus_cascade(self):
        """Runtime stop → main_end let-binding + main loop + 3-branch modulo cascade."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]):
                for i in pl.pipeline(0, n, 1, stage=4):
                    _y: pl.Scalar[pl.INDEX] = i

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]):
                # Tree shape must match C++ emission:
                # Add(start, Mul(FloorDiv(Sub(stop, start), chunk), chunk))
                unroll_main_end: pl.Scalar[pl.INDEX] = 0 + (n - 0) // 4 * 4
                for i in pl.range(0, unroll_main_end, 4):
                    _y: pl.Scalar[pl.INDEX] = i
                    _y_1: pl.Scalar[pl.INDEX] = i + 1
                    _y_2: pl.Scalar[pl.INDEX] = i + 2
                    _y_3: pl.Scalar[pl.INDEX] = i + 3
                unroll_rem: pl.Scalar[pl.INDEX] = n - unroll_main_end
                if unroll_rem == 1:
                    y_4: pl.Scalar[pl.INDEX] = unroll_main_end  # noqa: F841
                elif unroll_rem == 2:
                    y_5: pl.Scalar[pl.INDEX] = unroll_main_end  # noqa: F841
                    y_6: pl.Scalar[pl.INDEX] = unroll_main_end + 1  # noqa: F841
                elif unroll_rem == 3:
                    y_7: pl.Scalar[pl.INDEX] = unroll_main_end  # noqa: F841
                    y_8: pl.Scalar[pl.INDEX] = unroll_main_end + 1  # noqa: F841
                    y_9: pl.Scalar[pl.INDEX] = unroll_main_end + 2  # noqa: F841

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_dynamic_stop_with_nonunit_step_uses_iteration_count(self):
        """Dynamic bounds with step != 1 must dispatch on iteration count, not index span.

        For ``range(0, n, 2)`` with factor=4, trip_iters = ceil_div(n, 2). The
        main loop runs ``trip_iters // 4`` times with stride ``8``; the tail
        cascades on ``rem_iters = trip_iters - main_iters * 4``, not on
        ``stop - main_end`` (which would be in index units and overshoot)."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]):
                for i in pl.pipeline(0, n, 2, stage=4):
                    _y: pl.Scalar[pl.INDEX] = i

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]):
                # trip_iters = ceil_div(n - 0, 2) = (n - 0 + 1) // 2
                # main_iters = trip_iters // 4
                # main_end   = 0 + main_iters * (4 * 2) = 0 + main_iters * 8
                unroll_main_end: pl.Scalar[pl.INDEX] = 0 + (n - 0 + 1) // 2 // 4 * 8
                for i in pl.range(0, unroll_main_end, 8):
                    _y: pl.Scalar[pl.INDEX] = i
                    _y_1: pl.Scalar[pl.INDEX] = i + 2
                    _y_2: pl.Scalar[pl.INDEX] = i + 4
                    _y_3: pl.Scalar[pl.INDEX] = i + 6
                # rem_iters = trip_iters - main_iters * 4 (iteration units, not index units)
                unroll_rem: pl.Scalar[pl.INDEX] = (n - 0 + 1) // 2 - (n - 0 + 1) // 2 // 4 * 4
                if unroll_rem == 1:
                    y_4: pl.Scalar[pl.INDEX] = unroll_main_end  # noqa: F841
                elif unroll_rem == 2:
                    y_5: pl.Scalar[pl.INDEX] = unroll_main_end  # noqa: F841
                    y_6: pl.Scalar[pl.INDEX] = unroll_main_end + 2  # noqa: F841
                elif unroll_rem == 3:
                    y_7: pl.Scalar[pl.INDEX] = unroll_main_end  # noqa: F841
                    y_8: pl.Scalar[pl.INDEX] = unroll_main_end + 2  # noqa: F841
                    y_9: pl.Scalar[pl.INDEX] = unroll_main_end + 4  # noqa: F841

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_iter_args_clean_divide_threads_state_through_clones(self):
        """Loop-carried scalar threads sequentially through 4 replicated clones.

        Each clone consumes the previous clone's yielded value as its iter_arg
        substitute; the last clone's yield feeds the outer loop's next iteration."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], s0: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                for i, (a,) in pl.pipeline(0, 8, 1, stage=4, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    r = pl.yield_(b)
                return r

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], s0: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                for i, (a,) in pl.range(0, 8, 4, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    b_1: pl.Scalar[pl.INDEX] = b + (i + 1)
                    b_2: pl.Scalar[pl.INDEX] = b_1 + (i + 2)
                    b_3: pl.Scalar[pl.INDEX] = b_2 + (i + 3)
                    r = pl.yield_(b_3)
                return r

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_iter_args_with_remainder_forwards_state_to_tail(self):
        """Main loop's return_var seeds the tail clones' iter-arg uses; the tail's
        final yield binds to the original loop's return_var via an ``AssignStmt``
        so downstream uses remain valid."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], s0: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                for i, (a,) in pl.pipeline(0, 10, 1, stage=4, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    r = pl.yield_(b)
                return r

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], s0: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                for i, (a,) in pl.range(0, 8, 4, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    b_1: pl.Scalar[pl.INDEX] = b + (i + 1)
                    b_2: pl.Scalar[pl.INDEX] = b_1 + (i + 2)
                    b_3: pl.Scalar[pl.INDEX] = b_2 + (i + 3)
                    r_main = pl.yield_(b_3)
                # Tail clones — iter-arg `a` is substituted by r_main directly.
                b_4: pl.Scalar[pl.INDEX] = r_main + 8
                b_5: pl.Scalar[pl.INDEX] = b_4 + 9
                # Bind the original return_var to the tail's final yield.
                r: pl.Scalar[pl.INDEX] = b_5
                return r

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_iter_args_dynamic_cascade_threads_through_every_level(self):
        """Dynamic cascade: every IfStmt carries return_vars matching the iter_arg
        types, every branch ends with a YieldStmt, and the innermost else yields
        the main-loop return_var so ``rem == 0`` is a no-op fall-through."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                s0: pl.Scalar[pl.INDEX],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Scalar[pl.INDEX]:
                for i, (a,) in pl.pipeline(0, n, 1, stage=4, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    r = pl.yield_(b)
                return r

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                s0: pl.Scalar[pl.INDEX],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Scalar[pl.INDEX]:
                unroll_main_end: pl.Scalar[pl.INDEX] = 0 + (n - 0) // 4 * 4
                for i, (a,) in pl.range(0, unroll_main_end, 4, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    b_1: pl.Scalar[pl.INDEX] = b + (i + 1)
                    b_2: pl.Scalar[pl.INDEX] = b_1 + (i + 2)
                    b_3: pl.Scalar[pl.INDEX] = b_2 + (i + 3)
                    r_main = pl.yield_(b_3)
                unroll_rem: pl.Scalar[pl.INDEX] = n - unroll_main_end
                # Each IfStmt level carries its own return_vars and yield — the
                # cascade is nested (not elif/else), because every inner IfStmt
                # is the enclosing one's else body together with a trailing yield
                # that feeds the outer return_var. Each branch body is a bare
                # SeqStmts (no trip-1 ForStmt wrapper); iter-arg uses inside the
                # clones are substituted with r_main directly.
                if unroll_rem == 1:
                    b_4: pl.Scalar[pl.INDEX] = r_main + unroll_main_end
                    r = pl.yield_(b_4)
                else:
                    if unroll_rem == 2:
                        b_5: pl.Scalar[pl.INDEX] = r_main + unroll_main_end
                        b_6: pl.Scalar[pl.INDEX] = b_5 + (unroll_main_end + 1)
                        r_rem2 = pl.yield_(b_6)
                    else:
                        if unroll_rem == 3:
                            b_7: pl.Scalar[pl.INDEX] = r_main + unroll_main_end
                            b_8: pl.Scalar[pl.INDEX] = b_7 + (unroll_main_end + 1)
                            b_9: pl.Scalar[pl.INDEX] = b_8 + (unroll_main_end + 2)
                            r_rem3 = pl.yield_(b_9)
                        else:
                            r_rem3 = pl.yield_(r_main)
                        r_rem2 = pl.yield_(r_rem3)
                    r = pl.yield_(r_rem2)
                return r

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_zero_trip_demotes_to_sequential(self):
        """trip=0 (empty range) → no replication; the loop is demoted straight to
        plain ``pl.range`` with its body untouched.

        Source ``LowerStatic`` (lower_pipeline_loops_pass.cpp:361-363): when
        ``ComputeStaticTripCount(start, stop, step) == 0`` the pass takes the
        ``DemoteToSequential`` branch — kind flips to ``Sequential`` and
        ``pipeline_stages`` is stripped together (preserving the bidirectional
        ``kind == Pipeline ⇔ attr present`` invariant). No clones, no marker;
        the body (here ``_y = i``) is carried through verbatim from the
        recursive ``VisitStmt(op->body_)``. CanonicalizeIOOrder then sees a
        non-Pipeline loop and leaves it alone."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(0, 0, 1, stage=4):
                    _y: pl.Scalar[pl.INDEX] = i
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Same empty range as Before; kind demoted, attr dropped, body kept.
                for i in pl.range(0, 0, 1):
                    _y: pl.Scalar[pl.INDEX] = i
                return x

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_factor_exceeds_trip_is_pure_tail_no_main_loop(self):
        """trip=2, factor=4 → main_iters=0, so NO main loop is emitted; the entire
        remainder (2 clones) is flattened directly into the outer scope.

        Source ``LowerStatic`` (lower_pipeline_loops_pass.cpp:364-401): with
        ``main_iters = trip / factor == 0`` the ``if (main_iters > 0)`` guard
        skips ``BuildMainLoop`` entirely. ``rem_iters = trip % factor == 2`` so
        ``has_tail`` is true and ``BuildTailSeq`` replicates the body at offsets
        ``tail_base + j*step`` for ``j ∈ [0, 2)`` with ``tail_base = 0``. Clone 0
        substitutes ``i → 0`` (OffsetIndex folds to ConstInt 0), clone 1
        substitutes ``i → 1``. With no iter_args there are no binding
        AssignStmts, and the bare SeqStmts has no enclosing Pipeline ForStmt, so
        CanonicalizeIOOrder performs no reorder/demotion."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(0, 2, 1, stage=4):
                    _y: pl.Scalar[pl.INDEX] = i
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # No outer loop at all — just the two tail clones at offsets 0, 1.
                _y: pl.Scalar[pl.INDEX] = 0
                _y_1: pl.Scalar[pl.INDEX] = 1
                return x

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_pure_tail_iter_args_seed_from_init_values(self):
        """trip=3 < factor=4 with iter_args → the tail-only path seeds loop-carried
        state directly from the source loop's ``init_values`` (no main loop to
        forward from), then binds the original return_var with a trailing assign.

        Source ``LowerStatic`` (lower_pipeline_loops_pass.cpp:388-399): since
        ``main_iters == 0``, ``tail_init_values`` comes from
        ``InitValueExprs(op->iter_args_)`` (= ``s0``) rather than a main-loop
        return_var. The 3 tail clones thread state sequentially: clone 0 uses
        ``a → s0`` at offset 0 (``b = s0 + 0``), clone 1 uses ``a → b`` at
        offset 1, clone 2 uses ``a → b_1`` at offset 2. Finally an AssignStmt
        binds the original ``return_var`` ``r`` to the last clone's yield so
        downstream uses stay valid."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], s0: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                for i, (a,) in pl.pipeline(0, 3, 1, stage=4, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    r = pl.yield_(b)
                return r

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], s0: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
                # Tail-only: iter-arg `a` seeds from init value s0 (clone 0 at i=0).
                b: pl.Scalar[pl.INDEX] = s0 + 0
                b_1: pl.Scalar[pl.INDEX] = b + 1
                b_2: pl.Scalar[pl.INDEX] = b_1 + 2
                # Bind original return_var to the tail's final yield.
                r: pl.Scalar[pl.INDEX] = b_2
                return r

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_pipeline_inner_lowered_before_outer_replication(self):
        """Nested ``pl.pipeline`` → the inner pipeline is lowered FIRST, then the
        already-lowered inner loop is replicated by the outer pipeline.

        Source ``LowerPipelineMutator::VisitStmt_`` (lower_pipeline_loops_pass.cpp:182):
        ``auto inner_body = VisitStmt(op->body_)`` recurses before the outer loop
        is replicated, so the body cloned ``factor`` times by the outer loop is
        the inner loop *after* its own lowering. For outer/inner both
        ``trip=2, factor=2`` (clean divide, no remainder), each level becomes a
        main loop of stride 2 holding 2 body clones. After outer replication
        there are TWO inner-pipeline loops (one per outer clone), each with fresh
        SSA def-vars (``clone_def_vars=true``). CanonicalizeIOOrder then demotes
        both Pipeline markers to Sequential (scalar bodies → no reorder)."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(0, 2, 1, stage=2):
                    for j in pl.pipeline(0, 2, 1, stage=2):
                        _y: pl.Scalar[pl.INDEX] = j
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 2, 2):
                    # Outer clone 0: inner pipeline lowered to a 2-clone main loop.
                    for j in pl.range(0, 2, 2):
                        _y: pl.Scalar[pl.INDEX] = j
                        _y_1: pl.Scalar[pl.INDEX] = j + 1
                    # Outer clone 1: a second, fresh-SSA copy of the inner loop.
                    for j2 in pl.range(0, 2, 2):
                        _y2: pl.Scalar[pl.INDEX] = j2
                        _y3: pl.Scalar[pl.INDEX] = j2 + 1
                return x

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)

    def test_dynamic_nonunit_step_with_iter_args_cascades_in_iteration_units(self):
        """Dynamic stop + step=2 + iter_args, factor=4: the iteration-count formulas
        drive both the main loop and the modulo cascade, and loop-carried state
        threads through the main clones and every cascade branch.

        Combines the dynamic-nonunit-step accounting
        (lower_pipeline_loops_pass.cpp:430-466) with iter-arg threading
        (lines 472-522). ``trip_iters = (n - 0 + 1)//2`` (ceil_div by step=2),
        ``main_iters = trip_iters//4``, ``main_end = 0 + main_iters*(4*2)``
        bound to ``unroll_main_end``; the main loop strides by ``F*step = 8``
        with per-clone index offsets ``+0,+2,+4,+6``. ``unroll_rem =
        trip_iters - main_iters*4`` (iteration units, not index units). The
        cascade is a nested if/else (k = factor-1 .. 1, built innermost-first);
        each branch's tail clones seed iter-arg ``a`` from the main-loop
        return_var ``r_main`` and thread sequentially; every IfStmt carries
        return_vars and ends with a YieldStmt, and the innermost else yields
        ``r_main`` for the ``rem == 0`` fall-through."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                s0: pl.Scalar[pl.INDEX],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Scalar[pl.INDEX]:
                for i, (a,) in pl.pipeline(0, n, 2, stage=4, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    r = pl.yield_(b)
                return r

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                s0: pl.Scalar[pl.INDEX],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Scalar[pl.INDEX]:
                # main_end = 0 + ceil_div(n-0, 2)//4 * (4*2) = 0 + (n-0+1)//2//4 * 8
                unroll_main_end: pl.Scalar[pl.INDEX] = 0 + (n - 0 + 1) // 2 // 4 * 8
                for i, (a,) in pl.range(0, unroll_main_end, 8, init_values=(s0,)):
                    b: pl.Scalar[pl.INDEX] = a + i
                    b_1: pl.Scalar[pl.INDEX] = b + (i + 2)
                    b_2: pl.Scalar[pl.INDEX] = b_1 + (i + 4)
                    b_3: pl.Scalar[pl.INDEX] = b_2 + (i + 6)
                    r_main = pl.yield_(b_3)
                # rem in ITERATION units: trip_iters - main_iters*factor.
                unroll_rem: pl.Scalar[pl.INDEX] = (n - 0 + 1) // 2 - (n - 0 + 1) // 2 // 4 * 4
                if unroll_rem == 1:
                    t1: pl.Scalar[pl.INDEX] = r_main + unroll_main_end
                    r = pl.yield_(t1)
                else:
                    if unroll_rem == 2:
                        t2: pl.Scalar[pl.INDEX] = r_main + unroll_main_end
                        t3: pl.Scalar[pl.INDEX] = t2 + (unroll_main_end + 2)
                        r_rem2 = pl.yield_(t3)
                    else:
                        if unroll_rem == 3:
                            t4: pl.Scalar[pl.INDEX] = r_main + unroll_main_end
                            t5: pl.Scalar[pl.INDEX] = t4 + (unroll_main_end + 2)
                            t6: pl.Scalar[pl.INDEX] = t5 + (unroll_main_end + 4)
                            r_rem3 = pl.yield_(t6)
                        else:
                            r_rem3 = pl.yield_(r_main)
                        r_rem2 = pl.yield_(r_rem3)
                    r = pl.yield_(r_rem2)
                return r

        After = _run_pass(Before)
        ir.assert_structural_equal(After, Expected)


class TestAccumulatorMembership:
    """LowerPipelineLoops tags a pipeline stage's *operands* with
    ``pipeline_membership`` so MemoryReuse keeps their stage clones in separate
    buffers (the operand double-buffer). It must NOT tag *cube accumulators*: an
    Acc-space tile written by a ``tile.matmul*`` MAD is written by the single
    serialized cube, which retires one tile's MAD before the next's regardless of
    how many stages overlap, so it is never co-live with the next stage's
    accumulator. Tagging it would make MemoryReuse's capacity-gate request one L0C
    buffer per stage and shed it back (a spurious PH-MR-001, and 2^N requested
    buffers for an accumulator nested N pipeline loops deep). The exception keys on
    the producer op, not on ``Mem.Acc``: a data-movement op that also targets Acc
    (e.g. ``tile.extract(..., target_memory=Acc)``) is a real per-stage buffer and
    stays tagged."""

    def test_pipelined_accumulator_is_not_tagged(self):
        """A ``tile.matmul`` accumulator inside a ``pl.pipeline`` loop stays
        untagged after lowering, while its operands — and a non-cube Acc producer
        (``tile.extract`` into Acc) — get distinct-stage membership. The tagger is
        backend-independent (no backend fixture)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def kernel(
                self,
                a: pl.Tensor[[64, 32], pl.BF16],
                b: pl.Tensor[[32, 32], pl.BF16],
                out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 64, 32, stage=2, init_values=(out,)):
                    am: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        a, [i, 0], [32, 32], target_memory=pl.Mem.Mat
                    )
                    bm: pl.Tile[[32, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                        b, [0, 0], [32, 32], target_memory=pl.Mem.Mat
                    )
                    lft: pl.Tile[[32, 32], pl.BF16, pl.Mem.Left] = pl.tile.move(am, target_memory=pl.Mem.Left)
                    rgt: pl.Tile[[32, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(
                        bm, target_memory=pl.Mem.Right
                    )
                    c: pl.Tile[[32, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lft, rgt)
                    # A non-cube Acc producer: Mat->Acc data movement, NOT a MAD.
                    _spare: pl.Tile[[32, 32], pl.BF16, pl.Mem.Acc] = pl.tile.extract(
                        bm, 0, 0, [32, 32], target_memory=pl.Mem.Acc
                    )
                    nxt: pl.Tensor[[64, 32], pl.FP32] = pl.store(c, [i, 0], acc)
                    r = pl.yield_(nxt)
                return r

        After = passes.lower_pipeline_loops()(Before)
        lines = ir.python_print(After).splitlines()

        # stage=2 replicates the body into 2 clones. Neither clone's Acc matmul
        # carries pipeline_membership.
        acc_matmuls = [ln for ln in lines if "Mem.Acc" in ln and "tile.matmul" in ln]
        assert len(acc_matmuls) == 2, (
            f"expected 2 replicated Acc matmuls, got {len(acc_matmuls)}: {acc_matmuls}"
        )
        for ln in acc_matmuls:
            assert "pipeline_membership" not in ln, (
                f"cube accumulator must not carry pipeline_membership (cube serializes MADs): {ln.strip()}"
            )

        # A non-cube Acc producer (Mat->Acc extract) is data movement, not a
        # serialized MAD, so it stays tagged like any operand.
        acc_extracts = [ln for ln in lines if "Mem.Acc" in ln and "tile.extract" in ln]
        assert len(acc_extracts) == 2, f"expected 2 replicated Acc extracts, got {len(acc_extracts)}"
        for ln in acc_extracts:
            assert "pipeline_membership" in ln, (
                f"a non-cube Acc producer (tile.extract) must keep membership: {ln.strip()}"
            )

        # The operand moves are tagged with distinct-stage membership ("0:0" / "0:1").
        operand_moves = [ln for ln in lines if "tile.move" in ln]
        assert len(operand_moves) == 4, f"expected 4 replicated operand moves, got {len(operand_moves)}"
        stages = sorted(
            m.group(1) for ln in operand_moves if (m := re.search(r'pipeline_membership":\s*"([^"]*)"', ln))
        )
        assert stages == ["0:0", "0:0", "0:1", "0:1"], (
            f"pipelined operands must carry distinct-stage membership, got {stages}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
