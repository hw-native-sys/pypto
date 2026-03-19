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

import re

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


def _has_bare_keyword(code: str, keyword: str) -> bool:
    """Check if a bare keyword (break/continue) appears as a statement in the code."""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped == keyword:
            return True
    return False


def _require_break_flag_name(body: str) -> str:
    """Extract the auto-named break flag introduced by CtrlFlowTransform."""
    match = re.search(r"\b(break__tmp_v\d+)\b", body)
    assert match is not None
    return match.group(1)


# ===========================================================================
# Pre-SSA tests (non-strict_ssa input)
# ===========================================================================


class TestBreakOnly:
    """Tests for break elimination (ForStmt -> WhileStmt conversion)."""

    def test_break_in_for_loop(self):
        """ForStmt with break should become WhileStmt with break flag."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
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
        break_name = _require_break_flag_name(body)
        # No raw break/continue keywords (excluding break-flag references)
        assert "\n            break\n" not in python_print(After)
        assert "continue" not in body
        # Break flag init and condition
        assert f"{break_name}: pl.Scalar[pl.BOOL] = False" in body
        assert f"not {break_name}" in body
        # Break path sets flag to True
        assert f"{break_name}: pl.Scalar[pl.BOOL] = True" in body
        # iter_adv guarded by break flag
        assert f"if not {break_name}:" in body

    def test_break_first_stmt(self):
        """Break as the very first statement in the loop body."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 0:
                        break
                return x

        After = passes.ctrl_flow_transform()(Before)
        body = _get_function_body(python_print(After))
        assert "while" in body
        _require_break_flag_name(body)
        assert "\n            break\n" not in python_print(After)


class TestContinueOnly:
    """Tests for continue elimination (if-else restructuring)."""

    def test_continue_in_for_loop(self):
        """ForStmt with continue should restructure into if-else."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
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


class TestBreakAndContinue:
    """Tests for loops containing both break and continue."""

    def test_break_and_continue_same_loop(self):
        """Loop with both break and continue: eliminate continue first, then break."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
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
        break_name = _require_break_flag_name(body)
        # Both break and continue should be eliminated
        assert "break" not in body.replace(break_name, "")
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
            @pl.function(type=pl.FunctionType.InCore)
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
        break_name = _require_break_flag_name(body)
        assert "break" not in body.replace(break_name, "")
        assert f"not {break_name}" in body

    def test_while_continue(self):
        """WhileStmt with continue should restructure into if-else."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
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

    def test_while_break_with_ssa_iter_args(self):
        """WhileStmt SSA input with break (Var yield values)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                n: pl.Scalar[pl.INT64] = 0
                for cnt, x_iter in pl.while_(init_values=(n, x_0)):
                    pl.cond(cnt < 10)
                    if cnt > 5:
                        break
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    c2: pl.Scalar[pl.INT64] = cnt + 1
                    cnt, x_iter = pl.yield_(c2, y)  # noqa: PLW2901
                return x_iter

        After = passes.ctrl_flow_transform()(Before)
        printed = python_print(After)
        body = _get_function_body(printed)
        assert "BreakStmt" not in printed
        assert "ContinueStmt" not in printed
        _require_break_flag_name(body)
        assert "add" in body

    def test_while_break_with_ssa_inline_expr(self):
        """WhileStmt SSA input with break (non-Var inline expr in yield).

        Verifies that break yields current iter_args for non-Var expressions,
        not next-iteration advancement expressions like cnt + 1.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                n: pl.Scalar[pl.INT64] = 0
                for cnt, x_iter in pl.while_(init_values=(n, x_0)):
                    pl.cond(cnt < 10)
                    if cnt > 5:
                        break
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    cnt, x_iter = pl.yield_(cnt + 1, y)  # noqa: PLW2901
                return x_iter

        After = passes.ctrl_flow_transform()(Before)
        printed = python_print(After)
        body = _get_function_body(printed)
        assert "BreakStmt" not in printed
        assert "ContinueStmt" not in printed
        _require_break_flag_name(body)


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

    def test_orchestration_skipped(self):
        """Orchestration functions should not be transformed (break/continue are native)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 5:
                        break
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
            @pl.function(type=pl.FunctionType.InCore)
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
        break_name = _require_break_flag_name(body)
        assert "break" not in body.replace(break_name, "")


class TestEndToEnd:
    """End-to-end tests: CtrlFlowTransform -> NormalizeStmtStructure -> ConvertToSSA."""

    def test_break_then_ssa(self):
        """Verify break-transformed code correctly converts to SSA."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
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
            @pl.function(type=pl.FunctionType.InCore)
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


# ===========================================================================
# SSA-form standalone tests (strict_ssa=True)
# ===========================================================================


def test_continue_in_for():
    """Continue in ForStmt restructured to if/else with phi-node yield."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 5:
                    continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")
    # Should still be a ForStmt (no break)
    assert "pl.range(" in printed
    # Phi-node approach: IfStmt with yields feeding a trailing yield
    assert "pl.yield_(" in printed


def test_break_in_for():
    """Break in ForStmt converts to WhileStmt with break flag."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 5:
                    break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


def test_break_and_continue_in_for():
    """ForStmt with both break and continue."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 3:
                    continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i > 7:
                    break
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


def test_no_break_continue_noop():
    """Pass is identity when no break/continue (InCore SSA form)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Before)


def test_continue_multiple_iter_args():
    """Continue with multiple iter_args yields current iter_arg values."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(
            self,
            a_0: pl.Tensor[[64], pl.FP32],
            b_0: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            for i, (a_iter, b_iter) in pl.range(0, 10, 1, init_values=(a_0, b_0)):
                if i < 5:
                    continue
                a_new: pl.Tensor[[64], pl.FP32] = pl.add(a_iter, b_iter)
                b_new: pl.Tensor[[64], pl.FP32] = pl.add(b_iter, a_iter)
                a_iter, b_iter = pl.yield_(a_new, b_new)  # noqa: PLW2901
            return a_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_continue_with_pre_continue_assignment():
    """Continue after assignments — backward resolution yields iter_arg value."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i < 5:
                    continue
                z: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                x_iter = pl.yield_(z)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_break_negative_step():
    """Break in for loop with negative step uses > condition."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(10, 0, -1, init_values=(x_0,)):
                if i < 3:
                    break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")


def test_aic_function_type():
    """Pass processes AIC function type."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC, strict_ssa=True)
        def aic_kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 5:
                    continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_continue_no_iter_args():
    """Continue in loop with no carried state."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(0, 10, 1):
                if i < 5:
                    continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_0, x_0)  # noqa: F841
            return x_0

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_break_no_iter_args():
    """Break in loop with no carried state."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(0, 10, 1):
                if i > 5:
                    break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_0, x_0)  # noqa: F841
            return x_0

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")


def test_multiple_continues_in_body():
    """Two separate if-continue blocks in the same loop body."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 2:
                    continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i > 8:
                    continue
                z: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                x_iter = pl.yield_(z)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_back_to_back_breaks():
    """Two separate if-break blocks in the same loop body."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 8:
                    break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i > 5:
                    break
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


def test_break_then_continue():
    """Break guard first, then continue guard in same body."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 8:
                    break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i < 3:
                    continue
                z: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                x_iter = pl.yield_(z)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


def test_multiple_iter_args_with_break():
    """Break with multiple iter_args — all are carried through WhileStmt."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(
            self,
            a_0: pl.Tensor[[64], pl.FP32],
            b_0: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            for i, (a_iter, b_iter) in pl.range(0, 10, 1, init_values=(a_0, b_0)):
                if i > 5:
                    break
                a_new: pl.Tensor[[64], pl.FP32] = pl.add(a_iter, b_iter)
                b_new: pl.Tensor[[64], pl.FP32] = pl.add(b_iter, a_iter)
                a_iter, b_iter = pl.yield_(a_new, b_new)  # noqa: PLW2901
            return a_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


# ===========================================================================
# Unconditional break/continue
# ===========================================================================


def test_unconditional_break():
    """Bare break as first statement — loop executes 0 iterations effectively."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                break
                x_iter = pl.yield_(x_iter)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")


def test_unconditional_continue():
    """Bare continue as first statement — all iterations are skipped."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                continue
                x_iter = pl.yield_(x_iter)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


# ===========================================================================
# Nested loops
# ===========================================================================


def test_nested_loops_only_inner():
    """Only inner loop with continue is transformed, outer loop unchanged."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(0, 4, 1, init_values=(x_0,)):
                for j, (x_inner,) in pl.range(0, 8, 1, init_values=(x_outer,)):
                    if j < 2:
                        continue
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                    x_inner = pl.yield_(y)  # noqa: PLW2901
                x_outer = pl.yield_(x_inner)  # noqa: PLW2901
            return x_outer

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")
    # Outer loop should still be a ForStmt
    assert "pl.range(4" in printed or "pl.range(0, 4" in printed


def test_both_outer_and_inner_loop_have_break():
    """Outer and inner loop both have break — both converted to WhileStmt."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(0, 4, 1, init_values=(x_0,)):
                for j, (x_inner,) in pl.range(0, 8, 1, init_values=(x_outer,)):
                    if j > 3:
                        break
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                    x_inner = pl.yield_(y)  # noqa: PLW2901
                if i > 2:
                    break
                x_outer = pl.yield_(x_inner)  # noqa: PLW2901
            return x_outer

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")


def test_nested_continue_outer_break_inner():
    """Continue in outer loop, break in inner loop."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(0, 4, 1, init_values=(x_0,)):
                for j, (x_inner,) in pl.range(0, 8, 1, init_values=(x_outer,)):
                    if j > 3:
                        break
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                    x_inner = pl.yield_(y)  # noqa: PLW2901
                if i < 2:
                    continue
                x_outer = pl.yield_(x_inner)  # noqa: PLW2901
            return x_outer

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")
    # Inner loop should be while (has break), outer stays for (only continue)
    assert "pl.while_" in printed
    assert "pl.range(" in printed


def test_nested_continue_both_loops():
    """Continue in both inner and outer loops."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(0, 4, 1, init_values=(x_0,)):
                if i < 1:
                    continue
                for j, (x_inner,) in pl.range(0, 8, 1, init_values=(x_outer,)):
                    if j < 2:
                        continue
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                    x_inner = pl.yield_(y)  # noqa: PLW2901
                x_outer = pl.yield_(x_inner)  # noqa: PLW2901
            return x_outer

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")
    # Both should still be ForStmts (no break)
    assert "pl.while_" not in printed


def test_nested_break_and_continue_inner():
    """Inner loop has both break and continue, outer is clean."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(0, 4, 1, init_values=(x_0,)):
                for j, (x_inner,) in pl.range(0, 8, 1, init_values=(x_outer,)):
                    if j < 2:
                        continue
                    if j > 5:
                        break
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                    x_inner = pl.yield_(y)  # noqa: PLW2901
                x_outer = pl.yield_(x_inner)  # noqa: PLW2901
            return x_outer

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")
    # Inner loop should become while (has break)
    assert "pl.while_" in printed


def test_nested_loop_both_have_break_and_continue():
    """Both inner and outer loops have break and continue."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(0, 4, 1, init_values=(x_0,)):
                if i < 1:
                    continue
                for j, (x_inner,) in pl.range(0, 8, 1, init_values=(x_outer,)):
                    if j < 2:
                        continue
                    if j > 5:
                        break
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                    x_inner = pl.yield_(y)  # noqa: PLW2901
                if i > 2:
                    break
                x_outer = pl.yield_(x_inner)  # noqa: PLW2901
            return x_outer

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


def test_three_level_nesting_break_at_each():
    """Three levels of nested loops, break at each level."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_l1,) in pl.range(0, 3, 1, init_values=(x_0,)):
                for j, (x_l2,) in pl.range(0, 4, 1, init_values=(x_l1,)):
                    for k, (x_l3,) in pl.range(0, 5, 1, init_values=(x_l2,)):
                        if k > 2:
                            break
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x_l3, x_l3)
                        x_l3 = pl.yield_(y)  # noqa: PLW2901
                    if j > 1:
                        break
                    x_l2 = pl.yield_(x_l3)  # noqa: PLW2901
                if i > 0:
                    break
                x_l1 = pl.yield_(x_l2)  # noqa: PLW2901
            return x_l1

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")


# ===========================================================================
# Nested branches (break/continue inside nested ifs)
# ===========================================================================


def test_continue_in_else_branch():
    """Continue in else branch of IfStmt (not then branch)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=False)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 5:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                else:
                    continue
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_break_in_else_branch():
    """Break in else branch of IfStmt."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=False)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 7:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                else:
                    break
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


def test_if_else_continue_then_break_else():
    """Continue in then branch, break in else branch of same IfStmt."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i < 3:
                    continue
                elif i > 7:
                    break
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


def test_normal_if_else_before_continue():
    """If/else without break/continue, followed by a continue guard."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 5:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_0)
                if i < 2:
                    continue
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_deeply_nested_if_with_continue():
    """Continue inside three levels of nested ifs."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 8:
                    if i < 5:
                        if i < 2:
                            continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_deeply_nested_if_with_break():
    """Break inside three levels of nested ifs."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 3:
                    if i > 5:
                        if i > 7:
                            break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


# ===========================================================================
# Multi-function and pipeline integration
# ===========================================================================


def test_multi_function_program():
    """Program with InCore and Orchestration — only InCore transformed."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def incore_kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 5:
                    break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = self.incore_kernel(x)
            return y

    After = passes.ctrl_flow_transform()(Before)
    printed = After.as_python()
    # InCore function should have break lowered
    assert not _has_bare_keyword(printed, "break")


def test_pipeline_integration():
    """Pass works in a partial compilation pipeline."""

    @pl.program
    class Input:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.incore():
                for i in pl.range(10):
                    if i < 5:
                        continue
                    x = pl.add(x, x)
            return x

    after_ssa = passes.convert_to_ssa()(Input)
    after_outline = passes.outline_incore_scopes()(after_ssa)
    after_lower = passes.ctrl_flow_transform()(after_outline)

    printed = after_lower.as_python()
    assert not _has_bare_keyword(printed, "continue")
    assert not _has_bare_keyword(printed, "break")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
