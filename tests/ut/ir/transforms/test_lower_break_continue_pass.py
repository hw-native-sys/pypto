# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for LowerBreakContinue pass.

Tests use printed IR comparison because the pass generates internal variable
names (_alive_0, etc.) that are difficult to reproduce via @pl.program.
"""

import pypto.language as pl
import pytest
from pypto import passes
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


class TestNoOp:
    """Tests that the pass is a no-op when no break/continue is present."""

    def test_no_break_no_continue(self):
        """Loop without break/continue should be unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    x = pl.add(x, 1.0)
                return x

        After = passes.lower_break_continue()(Before)
        assert python_print(After) == python_print(Before)


class TestContinueLowering:
    """Tests for continue statement lowering."""

    def test_continue_with_condition(self):
        """if (cond) continue; rest → if (cond) {} else { rest }"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    if i < 5:
                        continue
                    x = pl.add(x, 1.0)
                return x

        After = passes.lower_break_continue()(Before)
        body = _get_function_body(python_print(After))

        # continue should be removed
        assert "continue" not in body
        # Remaining code should be in the else branch
        assert "else:" in body
        assert "pl.tensor.adds" in body

    def test_bare_continue_discards_remaining(self):
        """Bare continue discards all subsequent statements in the loop body."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    continue
                    x = pl.add(x, 1.0)
                return x

        After = passes.lower_break_continue()(Before)
        body = _get_function_body(python_print(After))

        assert "continue" not in body
        assert "pl.tensor.adds" not in body


class TestBreakLoweringInFor:
    """Tests for break statement lowering in for loops."""

    def test_break_with_condition(self):
        """if (cond) break; rest → alive = alive AND NOT(cond); if (alive) { rest }"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    if i < 5:
                        break
                    x = pl.add(x, 1.0)
                return x

        After = passes.lower_break_continue()(Before)
        body = _get_function_body(python_print(After))

        # break should be removed
        assert "break" not in body
        # Alive flag should be initialized
        assert "_alive_0" in body
        assert "True" in body
        # Alive flag update: alive AND NOT(cond)
        assert "and" in body
        assert "not" in body

    def test_break_after_statements(self):
        """Statements before if-break should be preserved."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    x = pl.add(x, 1.0)
                    if i > 5:
                        break
                return x

        After = passes.lower_break_continue()(Before)
        body = _get_function_body(python_print(After))

        assert "break" not in body
        assert "pl.tensor.adds" in body
        assert "_alive_0" in body

    def test_bare_break(self):
        """Bare break → alive = false, discard remaining."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    break
                    x = pl.add(x, 1.0)
                return x

        After = passes.lower_break_continue()(Before)
        body = _get_function_body(python_print(After))

        assert "break" not in body
        assert "_alive_0" in body
        assert "False" in body
        # Statement after break should be discarded
        assert "pl.tensor.adds" not in body


class TestBreakLoweringInWhile:
    """Tests for break statement lowering in while loops."""

    def test_while_break_with_condition(self):
        """Break in while loop: adds alive flag ANDed into while condition."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                for (x,) in pl.while_(init_values=(x,)):  # noqa: PLR1704
                    pl.cond(x < 10)
                    x = x + 1  # noqa: PLW2901
                    if x > 5:
                        break
                    x_out = pl.yield_(x)  # noqa: F841
                return x

        After = passes.lower_break_continue()(Before)
        body = _get_function_body(python_print(After))

        assert "break" not in body
        assert "_alive_0" in body
        # While condition should include alive flag
        assert "and" in body


class TestCombinedBreakContinue:
    """Tests for loops with both break and continue."""

    def test_continue_and_break_in_same_loop(self):
        """Continue is lowered; break inside nested else is preserved for later passes."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    if i < 3:
                        continue
                    x = pl.add(x, 1.0)
                    if i > 7:
                        break
                return x

        After = passes.lower_break_continue()(Before)
        body = _get_function_body(python_print(After))

        # Continue should be lowered into if/else
        assert "continue" not in body
        assert "else:" in body
        # Alive flag should be initialized (break triggers it)
        assert "_alive_0" in body
        assert "pl.tensor.adds" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
