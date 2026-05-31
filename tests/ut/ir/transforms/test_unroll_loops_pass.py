# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for UnrollLoops pass.

Tests use the Before/Expected pattern with @pl.program decorator.
Unrolled IR is piped through convert_to_ssa() so that structural
equality (assert_structural_equal) can be used for all comparisons.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.printer import python_print


def _unroll_and_ssa(program):
    """Apply unroll_loops followed by convert_to_ssa."""
    return passes.convert_to_ssa()(passes.unroll_loops()(program))


class TestBasicUnroll:
    """Tests for basic loop unrolling."""

    def test_simple_unroll(self):
        """Unroll a simple loop with 3 iterations."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(3):
                    x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                x_1: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 1.0)
                x_2: pl.Tensor[[64], pl.FP32] = pl.add(x_1, 1.0)
                return x_2

        After = _unroll_and_ssa(Before)
        ir.assert_structural_equal(After, Expected)

    def test_unroll_with_start_stop_step(self):
        """Unroll with explicit start, stop, step: unroll(0, 6, 2) -> 3 iterations."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(0, 6, 2):
                    x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                x_1: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 1.0)
                x_2: pl.Tensor[[64], pl.FP32] = pl.add(x_1, 1.0)
                return x_2

        After = _unroll_and_ssa(Before)
        ir.assert_structural_equal(After, Expected)

    def test_unroll_loop_var_in_expression(self):
        """Verify loop variable is substituted with constants in expressions."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(3):
                    x = pl.add(x, i)
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 0)
                x_1: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 1)
                x_2: pl.Tensor[[64], pl.FP32] = pl.add(x_1, 2)
                return x_2

        After = _unroll_and_ssa(Before)
        ir.assert_structural_equal(After, Expected)

    def test_single_iteration_unroll(self):
        """Unroll with a single iteration."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(1):
                    x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return x_0

        After = _unroll_and_ssa(Before)
        ir.assert_structural_equal(After, Expected)

    def test_negative_step_unroll(self):
        """Negative step unroll(6, 0, -2) iterates i = 6, 4, 2 (3 iterations).

        Exercises both the ``step < 0`` trip-count / emit branches
        (unroll_loops_pass.cpp:99-101, 124-126) and the ``Neg(ConstInt)``
        negative-literal extraction in GetConstIntValue (lines 58-64): the
        parser represents the ``-2`` step as ``neg(ConstInt(2))``. The loop
        variable is substituted with each descending constant value, so the
        added literals must be 6, 4, 2 in that order.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(6, 0, -2):
                    x = pl.add(x, i)
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 6)
                x_1: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 4)
                x_2: pl.Tensor[[64], pl.FP32] = pl.add(x_1, 2)
                return x_2

        After = _unroll_and_ssa(Before)
        ir.assert_structural_equal(After, Expected)


class TestNestedLoops:
    """Tests for unrolling with nested loops."""

    def test_unroll_inside_regular_loop(self):
        """Unroll loop nested inside a regular pl.range() loop."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for j in pl.range(n):
                    for i in pl.unroll(2):
                        x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for j, (x_iter,) in pl.range(n, init_values=(x,)):
                    x_0: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, 1.0)
                    x_1: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 1.0)
                    x_rv = pl.yield_(x_1)
                return x_rv

        After = _unroll_and_ssa(Before)
        ir.assert_structural_equal(After, Expected)

    def test_regular_loop_not_unrolled(self):
        """Regular (non-unroll) loops should remain unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    x = pl.add(x, 1.0)
                return x

        After = passes.unroll_loops()(Before)
        ir.assert_structural_equal(After, Before)

    def test_chunked_unroll_not_expanded(self):
        """Chunked unroll loops are left intact for SplitChunkedLoops to handle.

        VisitStmt_ skips expansion when ``op->chunk_config_.has_value()``
        (unroll_loops_pass.cpp:164), so a ``pl.unroll(..., chunk=...)`` loop
        survives this pass unchanged — the chunk is later lowered by
        SplitChunkedLoops. The pass returns the original function unmodified,
        so the result is structurally identical to the input (no SSA conversion
        needed since nothing is transformed).
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                    for i in pl.unroll(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                        x = pl.add(x, 1.0)
                return x

        After = passes.unroll_loops()(Before)
        ir.assert_structural_equal(After, Before)


class TestZeroTripLoop:
    """Tests for zero-trip unrolled loops."""

    def test_zero_trip(self):
        """Unroll loop with zero iterations produces empty body."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(0, 0, 1):
                    x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        After = _unroll_and_ssa(Before)
        ir.assert_structural_equal(After, Expected)


class TestUnrollLimits:
    """Tests for the compile-time unroll trip-count limit."""

    def test_trip_count_exceeds_max_raises(self):
        """Trip count above kMaxUnrollIterations (1024) raises ValueError.

        unroll_loops_pass.cpp:102-106 rejects any loop whose computed trip
        count exceeds 1024. ``pl.unroll(2000)`` has trip count 2000 > 1024,
        so running the pass must raise rather than expand 2000 body copies.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(2000):
                    x = pl.add(x, 1.0)
                return x

        with pytest.raises(Exception, match="exceeds maximum allowed"):
            passes.unroll_loops()(Before)


class TestParserValidation:
    """Tests for parser-level validation of pl.unroll()."""

    def test_unroll_with_init_values_rejected(self):
        """pl.unroll() cannot be combined with init_values."""
        with pytest.raises(Exception, match="cannot be combined with init_values"):

            @pl.program
            class _:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i, (acc,) in pl.unroll(3, init_values=(x,)):
                        acc = pl.add(acc, 1.0)
                        acc = pl.yield_(acc)
                    return x


class TestPrinterRoundTrip:
    """Tests for IR printing of unroll loops."""

    def test_unroll_prints_as_pl_unroll(self):
        """ForKind.Unroll should print as pl.unroll() in output."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(3):
                    x = pl.add(x, 1.0)
                return x

        printed = python_print(Prog)
        assert "pl.unroll(" in printed


class TestPipelineFallback:
    """Tests that unexpanded unroll loops survive non-codegen pipeline stages."""

    @pytest.mark.filterwarnings("ignore:.*RoundtripInstrument.*IR not printable:UserWarning")
    def test_unexpanded_unroll_survives_pipeline(self):
        """Skipping UnrollLoops should not crash through SSA/flatten/verifier pipeline."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(3):
                    x = pl.add(x, 1.0)
                return x

        # Run SSA and verifier without UnrollLoops — this validates pipeline
        # robustness before backend codegen.
        result = passes.convert_to_ssa()(Prog)
        result = passes.flatten_call_expr()(result)
        result = passes.run_verifier()(result)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
