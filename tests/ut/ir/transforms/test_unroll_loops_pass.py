# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for UnrollLoops pass.

Pre-SSA tests compare printed IR because unrolled bodies retain original
Var pointers from the loop body, while hand-written Expected code creates
chained def-use patterns. After SSA conversion, structural equality works.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.printer import python_print


def _get_function_body(printed: str) -> str:
    """Extract the function body lines from printed IR (after the def line)."""
    lines = printed.strip().splitlines()
    body_lines = []
    in_body = False
    for line in lines:
        if in_body:
            body_lines.append(line.strip())
        if line.strip().startswith("def main("):
            in_body = True
    return "\n".join(body_lines)


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

        After = passes.unroll_loops()(Before)
        body = _get_function_body(python_print(After))
        assert body.count("pl.tensor.add(x, 1.0)") == 3
        assert "pl.range" not in body

    def test_unroll_with_start_stop_step(self):
        """Unroll with explicit start, stop, step: unroll(0, 6, 2) -> 3 iterations."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(0, 6, 2):
                    x = pl.add(x, 1.0)
                return x

        After = passes.unroll_loops()(Before)
        body = _get_function_body(python_print(After))
        assert body.count("pl.tensor.add(x, 1.0)") == 3
        assert "pl.range" not in body

    def test_unroll_loop_var_in_expression(self):
        """Verify loop variable is substituted with constants in expressions."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(3):
                    x = pl.add(x, i)
                return x

        After = passes.unroll_loops()(Before)
        body = _get_function_body(python_print(After))
        assert "pl.tensor.add(x, 0)" in body
        assert "pl.tensor.add(x, 1)" in body
        assert "pl.tensor.add(x, 2)" in body
        assert "pl.range" not in body

    def test_single_iteration_unroll(self):
        """Unroll with a single iteration."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(1):
                    x = pl.add(x, 1.0)
                return x

        After = passes.unroll_loops()(Before)
        body = _get_function_body(python_print(After))
        assert body.count("pl.tensor.add(x, 1.0)") == 1
        assert "pl.range" not in body


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

        After = passes.unroll_loops()(Before)
        body = _get_function_body(python_print(After))
        # Should have a regular for loop with 2 copies inside
        assert "pl.range(" in body  # The outer loop remains
        assert body.count("pl.tensor.add(x, 1.0)") == 2
        assert "pl.unroll" not in body

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

        After = passes.unroll_loops()(Before)
        printed = python_print(After)
        assert "pl.tensor.add" not in printed
        assert "return x" in printed


class TestEndToEnd:
    """End-to-end tests: unroll followed by SSA conversion."""

    def test_unroll_then_ssa(self):
        """Verify unrolled code correctly converts to SSA."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(3):
                    x = pl.add(x, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                x_1: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 1.0)
                x_2: pl.Tensor[[64], pl.FP32] = pl.add(x_1, 1.0)
                return x_2

        After = passes.unroll_loops()(Before)
        After = passes.convert_to_ssa()(After)
        ir.assert_structural_equal(After, Expected)  # type: ignore[arg-type]

    def test_unroll_with_loop_var_then_ssa(self):
        """Verify loop variable substitution is correct after SSA."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.unroll(3):
                    x = pl.add(x, i)
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                x_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 0)
                x_1: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 1)
                x_2: pl.Tensor[[64], pl.FP32] = pl.add(x_1, 2)
                return x_2

        After = passes.unroll_loops()(Before)
        After = passes.convert_to_ssa()(After)
        ir.assert_structural_equal(After, Expected)  # type: ignore[arg-type]


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
                        acc = pl.add(acc, 1.0)  # noqa: PLW2901
                        acc = pl.yield_(acc)  # noqa: PLW2901
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
