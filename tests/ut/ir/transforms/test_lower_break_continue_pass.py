# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _has_bare_keyword(code: str, keyword: str) -> bool:
    """Check if a bare keyword (break/continue) appears as a statement in the code."""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped == keyword:
            return True
    return False


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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


def test_no_break_continue_noop():
    """Pass is identity when no break/continue."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.lower_break_continue()(Before)
    ir.assert_structural_equal(After, Before)


def test_orchestration_untouched():
    """Non-InCore functions are not transformed."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(0, 10, 1):
                if i > 5:
                    break
            return x_0

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")
    # Outer loop should still be a ForStmt
    assert "pl.range(4" in printed or "pl.range(0, 4" in printed


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    # InCore function should have break lowered
    assert not _has_bare_keyword(printed, "break")


def test_pass_properties():
    """Verify pass has correct properties."""
    p = passes.lower_break_continue()
    assert p.get_name() == "LowerBreakContinue"

    required = p.get_required_properties()
    assert required.contains(passes.IRProperty.SSAForm)
    assert required.contains(passes.IRProperty.SplitIncoreOrch)

    produced = p.get_produced_properties()
    assert produced.contains(passes.IRProperty.SSAForm)
    assert produced.contains(passes.IRProperty.SplitIncoreOrch)


def test_pass_in_pipeline():
    """Verify pass is registered in both Default and CCE strategies."""
    for strategy in [OptimizationStrategy.Default, OptimizationStrategy.CCE]:
        pm = PassManager.get_strategy(strategy)
        names = pm.get_pass_names()
        assert "LowerBreakContinue" in names
        # Must come after InferTileMemorySpace
        rtl_idx = names.index("ResolveTransposeLayout")
        lbc_idx = names.index("LowerBreakContinue")
        assert lbc_idx == rtl_idx + 1


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
    after_lower = passes.lower_break_continue()(after_outline)

    printed = after_lower.as_python()
    assert not _has_bare_keyword(printed, "continue")
    assert not _has_bare_keyword(printed, "break")


# ===========================================================================
# Nested loops
# ===========================================================================


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")
    # Inner loop should be while (has break), outer stays for (only continue)
    assert "pl.while_" in printed
    assert "pl.range(" in printed


def test_nested_break_outer_continue_inner():
    """Break in outer loop, continue in inner loop."""

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
                if i > 2:
                    break
                x_outer = pl.yield_(x_inner)  # noqa: PLW2901
            return x_outer

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")
    # Inner loop should become while (has break)
    assert "pl.while_" in printed


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")


# ===========================================================================
# Nested branches
# ===========================================================================


def test_continue_in_nested_if():
    """Continue inside a nested if (if inside if)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 8:
                    if i < 3:
                        continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_break_in_nested_if():
    """Break inside a nested if."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 3:
                    if i > 7:
                        break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


def test_continue_in_else_branch():
    """Continue in else branch of IfStmt (not then branch)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 5:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                else:
                    continue
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


def test_break_in_else_branch():
    """Break in else branch of IfStmt."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 7:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                else:
                    break
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


# ===========================================================================
# Multiple break/continue patterns
# ===========================================================================


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


def test_continue_then_break():
    """Continue guard first, then break guard in same body."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 2:
                    continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i > 7:
                    break
                x_iter = pl.yield_(y)  # noqa: PLW2901
            return x_iter

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


# ===========================================================================
# Complex nested combinations
# ===========================================================================


def test_nested_loop_inner_continue_outer_break_and_continue():
    """Inner loop has continue, outer loop has both break and continue."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(0, 6, 1, init_values=(x_0,)):
                if i < 1:
                    continue
                for j, (x_inner,) in pl.range(0, 8, 1, init_values=(x_outer,)):
                    if j < 2:
                        continue
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                    x_inner = pl.yield_(y)  # noqa: PLW2901
                if i > 4:
                    break
                x_outer = pl.yield_(x_inner)  # noqa: PLW2901
            return x_outer

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert not _has_bare_keyword(printed, "continue")


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
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

    After = passes.lower_break_continue()(Before)
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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


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

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "break")
    assert "pl.while_" in printed


def test_computation_between_continues():
    """Multiple continues with computation in between each guard."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 20, 1, init_values=(x_0,)):
                a: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i < 5:
                    continue
                b: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                if i < 10:
                    continue
                c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                if i < 15:
                    continue
                x_iter = pl.yield_(c)  # noqa: PLW2901
            return x_iter

    After = passes.lower_break_continue()(Before)
    printed = After.as_python()
    assert not _has_bare_keyword(printed, "continue")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
