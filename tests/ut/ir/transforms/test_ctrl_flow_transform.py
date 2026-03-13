# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for CtrlFlowTransform pass.

Pre-SSA tests compare printed IR because the pass creates new Var objects
(break flags, loop vars), making structural equality impractical.
End-to-end tests verify the full pipeline: CtrlFlowTransform -> ConvertToSSA.
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


class TestBreakOnly:
    """Tests for break elimination (ForStmt -> WhileStmt conversion)."""

    def test_break_in_for_loop(self):
        """ForStmt with break should become WhileStmt with break flag."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 5:
                        break
                    x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))

        # Should have while loop with break flag
        assert "while" in body
        assert "__break_0" in body
        # No raw break/continue keywords (excluding __break_0 variable references)
        assert "\n            break\n" not in python_print(After)
        assert "continue" not in body
        # Break flag init and condition
        assert "__break_0: pl.Scalar[pl.BOOL] = False" in body
        assert "not __break_0" in body
        # Break path sets flag to True
        assert "__break_0: pl.Scalar[pl.BOOL] = True" in body
        # iter_adv guarded by break flag
        assert "if not __break_0:" in body

    def test_break_first_stmt(self):
        """Break as the very first statement in the loop body."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 0:
                        break
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))
        assert "while" in body
        assert "__break_0" in body
        assert "\n            break\n" not in python_print(After)


class TestContinueOnly:
    """Tests for continue elimination (if-else restructuring)."""

    def test_continue_in_for_loop(self):
        """ForStmt with continue should restructure into if-else."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 5:
                        continue
                    x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))

        # Should keep ForStmt (no while conversion needed)
        assert "for i in pl.range" in body
        # Continue should be eliminated, replaced with if-else
        assert "continue" not in body
        assert "else:" in body
        # The add should be in the else branch
        assert "pl.tensor.adds(x, 1.0)" in body

    def test_continue_with_multiple_remaining_stmts(self):
        """Continue with multiple statements after it should absorb all into else."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 5:
                        continue
                    x = pl.add(x, 1.0)
                    x = pl.mul(x, 2.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))
        assert "continue" not in body
        assert "pl.tensor.adds" in body
        assert "pl.tensor.muls" in body


class TestBreakAndContinue:
    """Tests for loops containing both break and continue."""

    def test_break_and_continue_same_loop(self):
        """Loop with both break and continue: eliminate continue first, then break."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 10:
                        break
                    x = pl.add(x, 1.0)
                    if i > 5:
                        continue
                    x = pl.mul(x, 2.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))

        # Should convert to while (due to break)
        assert "while" in body
        assert "__break_0" in body
        # Both break and continue should be eliminated
        assert "break" not in body.replace("__break_0", "").replace("not __break_0", "")
        assert "continue" not in body
        # Both operations should be present
        assert "pl.tensor.adds" in body
        assert "pl.tensor.muls" in body


class TestWhileLoops:
    """Tests for break/continue in while loops."""

    def test_while_break(self):
        """WhileStmt with break should augment condition with break flag."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INT64] = 0
                while i < n:
                    if i > 5:
                        break
                    x = pl.add(x, 1.0)
                    i = i + 1
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))

        assert "while" in body
        assert "__break_0" in body
        assert "break" not in body.replace("__break_0", "").replace("not __break_0", "")
        assert "not __break_0" in body

    def test_while_continue(self):
        """WhileStmt with continue should restructure into if-else."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INT64] = 0
                while i < n:
                    i = i + 1
                    if i > 5:
                        continue
                    x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))

        assert "while" in body
        assert "continue" not in body
        assert "else:" in body


class TestIdentity:
    """Tests for loops without break/continue (should be unchanged)."""

    def test_no_break_continue(self):
        """Normal ForStmt without break/continue should be unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Before)

    def test_parallel_loop_unchanged(self):
        """Parallel ForStmt (no break/continue allowed) should be unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.parallel(64):
                    x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Before)


class TestNestedLoops:
    """Tests for nested loops with break/continue."""

    def test_nested_inner_break(self):
        """Only inner loop with break should be transformed."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], m: pl.Scalar[pl.INT64], n: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                for j in pl.range(m):
                    for i in pl.range(n):
                        if i > 5:
                            break
                        x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))

        # Outer loop should remain a for loop
        assert "for j in pl.range" in body
        # Inner loop should become a while
        assert "while" in body
        assert "__break_0" in body
        assert "break" not in body.replace("__break_0", "").replace("not __break_0", "")


class TestEndToEnd:
    """End-to-end tests: CtrlFlowTransform -> ConvertToSSA."""

    def test_break_then_ssa(self):
        """Verify break-transformed code correctly converts to SSA."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 5:
                        break
                    x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        After = passes.convert_to_ssa()(After)
        # Should not crash and should produce valid SSA
        body = _get_function_body(python_print(After))
        assert "pl.while_" in body
        assert "pl.yield_" in body

    def test_continue_then_ssa(self):
        """Verify continue-transformed code correctly converts to SSA."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 5:
                        continue
                    x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        After = passes.convert_to_ssa()(After)
        body = _get_function_body(python_print(After))
        # Should have proper SSA form with yield
        assert "pl.yield_" in body

    def test_break_continue_then_ssa(self):
        """Verify combined break+continue code correctly converts to SSA."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 10:
                        break
                    x = pl.add(x, 1.0)
                    if i > 5:
                        continue
                    x = pl.mul(x, 2.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        After = passes.convert_to_ssa()(After)
        body = _get_function_body(python_print(After))
        assert "pl.while_" in body
        assert "pl.yield_" in body


class TestPassProperties:
    """Tests for pass property declarations."""

    def test_pass_name(self):
        """Verify the pass has the correct name."""
        p = passes.ctrl_flow_transform()
        assert p.get_name() == "CtrlFlowTransform"

    def test_required_properties(self):
        """Verify no required properties (TypeChecked is structural, not per-pass)."""
        p = passes.ctrl_flow_transform()
        required = p.get_required_properties()
        assert required.empty()

    def test_produced_properties(self):
        """Verify produced properties include StructuredCtrlFlow."""
        p = passes.ctrl_flow_transform()
        produced = p.get_produced_properties()
        assert produced.contains(passes.IRProperty.StructuredCtrlFlow)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
