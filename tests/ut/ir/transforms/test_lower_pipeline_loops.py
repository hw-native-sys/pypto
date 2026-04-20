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

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_pass(program: ir.Program) -> ir.Program:
    """Run ``LowerPipelineLoops`` + ``CanonicalizeIOOrder`` with structural
    verification disabled — our Before programs are intentionally minimal and
    skip the tile-lowering chain. Canonicalize runs to demote the transient
    ``ForKind::Pipeline`` marker back to ``Sequential`` so the Expected
    programs (written in plain ``pl.range``) can be compared structurally.
    Canonicalize is a no-op on the scalar-only bodies used in these tests
    (single tier, stable order)."""
    with passes.PassContext([], passes.VerificationLevel.NONE):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
