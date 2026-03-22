# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for CtrlFlowTransform pass.

Tests compare pass output against Expected IR using ir.assert_structural_equal.
For simple patterns (single-level phi-nodes), Expected is written with @pl.program.
For complex nested phi-node patterns that the DSL parser cannot express, Expected
is constructed using IRBuilder.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.builder import IRBuilder

_SPAN = ir.Span.unknown()
_IDX = ir.DataType.INDEX
_BOOL = ir.DataType.BOOL
_FP32 = ir.DataType.FP32
_INT64 = ir.DataType.INT64

_IDX_T = ir.ScalarType(_IDX)
_BOOL_T = ir.ScalarType(_BOOL)
_INT64_T = ir.ScalarType(_INT64)


def _ci(val: int, dt: ir.DataType = _IDX) -> ir.ConstInt:
    """Create ConstInt with given type."""
    return ir.ConstInt(val, dt, _SPAN)


def _cb(val: bool) -> ir.ConstBool:
    """Create ConstBool."""
    return ir.ConstBool(val, _SPAN)


def _tt(shape: list[int] | None = None, dt: ir.DataType = _FP32) -> ir.TensorType:
    """Create TensorType."""
    return ir.TensorType(shape or [64], dt)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INDEX] = 0
                brk: pl.Scalar[pl.BOOL] = False
                while i < n and not brk:
                    if i > 5:
                        brk: pl.Scalar[pl.BOOL] = True
                        pl.yield_()
                    else:
                        x = pl.add(x, 1.0)
                        pl.yield_()
                    if not brk:
                        i = i + 1
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INDEX] = 0
                brk: pl.Scalar[pl.BOOL] = False
                while i < n and not brk:
                    if i > 0:
                        brk: pl.Scalar[pl.BOOL] = True
                        pl.yield_()
                    else:
                        pl.yield_()
                    if not brk:
                        i = i + 1
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(n):
                    if i > 5:
                        pass
                    else:
                        x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INDEX] = 0
                brk: pl.Scalar[pl.BOOL] = False
                while i < n and not brk:
                    if i > 10:
                        brk: pl.Scalar[pl.BOOL] = True
                        pl.yield_()
                    else:
                        x = pl.add(x, 1.0)
                        if i > 5:
                            pass
                        else:
                            x = pl.mul(x, 2.0)
                        pl.yield_()
                    if not brk:
                        i = i + 1
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INT64] = 0
                brk: pl.Scalar[pl.BOOL] = False
                while i < n and not brk:
                    if i > 5:
                        brk: pl.Scalar[pl.BOOL] = True
                        pl.yield_()
                    else:
                        x = pl.add(x, 1.0)
                        i = i + 1
                        pl.yield_()
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INT64] = 0
                while i < n:
                    i = i + 1
                    if i > 5:
                        pass
                    else:
                        x = pl.add(x, 1.0)
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)

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
                    cnt, x_iter = pl.yield_(c2, y)
                return x_iter

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                n: pl.Scalar[pl.INT64] = 0
                brk: pl.Scalar[pl.BOOL] = False
                for cnt, x_iter in pl.while_(init_values=(n, x_0)):
                    pl.cond(cnt < 10 and not brk)
                    if cnt > 5:
                        brk: pl.Scalar[pl.BOOL] = True
                        cnt_phi, x_phi = pl.yield_(cnt, x_iter)
                    else:
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                        c2: pl.Scalar[pl.INT64] = cnt + 1
                        cnt_phi, x_phi = pl.yield_(c2, y)
                    cnt, x_iter = pl.yield_(cnt_phi, x_phi)
                return x_iter

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)

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
                    cnt, x_iter = pl.yield_(cnt + 1, y)
                return x_iter

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                n: pl.Scalar[pl.INT64] = 0
                brk: pl.Scalar[pl.BOOL] = False
                for cnt, x_iter in pl.while_(init_values=(n, x_0)):
                    pl.cond(cnt < 10 and not brk)
                    if cnt > 5:
                        brk: pl.Scalar[pl.BOOL] = True
                        cnt_phi, x_phi = pl.yield_(cnt, x_iter)
                    else:
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                        cnt_phi, x_phi = pl.yield_(cnt + 1, y)
                    cnt, x_iter = pl.yield_(cnt_phi, x_phi)
                return x_iter

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)


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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self, x: pl.Tensor[[64], pl.FP32], m: pl.Scalar[pl.INT64], n: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                for j in pl.range(m):
                    i: pl.Scalar[pl.INDEX] = 0
                    brk: pl.Scalar[pl.BOOL] = False
                    while i < n and not brk:
                        if i > 5:
                            brk: pl.Scalar[pl.BOOL] = True
                            pl.yield_()
                        else:
                            x = pl.add(x, 1.0)
                            pl.yield_()
                        if not brk:
                            i = i + 1
                return x

        After = passes.ctrl_flow_transform()(Before)
        ir.assert_structural_equal(After, Expected)


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

        # Build Expected with IRBuilder (nested phi-node pattern for break-flag + iter guard)
        ib = IRBuilder()
        tt = _tt()

        with ib.program("Expected") as prog:
            with ib.function("main", type=ir.FunctionType.InCore) as f:
                x_0 = f.param("x_0", tt)
                n_0 = f.param("n_0", _INT64_T)
                f.return_type(tt)

                i_0 = ib.let("i_0", _ci(0))
                brk_0 = ib.let("brk_0", _cb(False))

                cond_placeholder = _cb(True)
                with ib.while_loop(cond_placeholder) as loop:
                    brk_iter = loop.iter_arg("brk_iter", brk_0)
                    i_iter = loop.iter_arg("i_iter", i_0)
                    x_iter = loop.iter_arg("x_iter", x_0)
                    loop.set_condition(
                        ir.And(ir.Lt(i_iter, n_0, _BOOL, _SPAN), ir.Not(brk_iter, _BOOL, _SPAN), _BOOL, _SPAN)
                    )

                    # if i_iter > 5: break; else: x = add(x, 1.0)
                    with ib.if_stmt(ir.Gt(i_iter, _ci(5), _BOOL, _SPAN)) as if_b:
                        if_b.return_var("brk_phi", _BOOL_T)
                        if_b.return_var("x_phi", tt)
                        brk_1 = ib.let("brk_1", _cb(True))
                        ib.emit(ir.YieldStmt([brk_1, x_iter], _SPAN))
                        if_b.else_()
                        x_1 = ib.let(
                            "x_1",
                            ir.Call(
                                ir.Op("tensor.adds"), [x_iter, ir.ConstFloat(1.0, _FP32, _SPAN)], tt, _SPAN
                            ),
                        )
                        ib.emit(ir.YieldStmt([brk_iter, x_1], _SPAN))
                    brk_phi = if_b.output(0)
                    x_phi = if_b.output(1)

                    # if not brk_phi: i = i + 1; else: pass
                    with ib.if_stmt(ir.Not(brk_phi, _BOOL, _SPAN)) as if_inc:
                        if_inc.return_var("i_phi", _IDX_T)
                        i_1 = ib.let("i_1", ir.Add(i_iter, _ci(1), _IDX, _SPAN))
                        ib.emit(ir.YieldStmt([i_1], _SPAN))
                        if_inc.else_()
                        ib.emit(ir.YieldStmt([i_iter], _SPAN))
                    i_phi = if_inc.output(0)

                    loop.return_var("brk_rv")
                    loop.return_var("i_rv")
                    loop.return_var("x_rv")
                    ib.emit(ir.YieldStmt([brk_phi, i_phi, x_phi], _SPAN))

                x_rv = loop.output(2)
                ib.return_stmt(x_rv)
            prog.add_function(f.get_result())

        Expected = prog.get_result()
        ir.assert_structural_equal(After, Expected)

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(
                self, x_0: pl.Tensor[[64], pl.FP32], n_0: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (x_iter,) in pl.range(n_0, init_values=(x_0,)):
                    if i > 5:
                        x_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                    else:
                        x_1: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, 1.0)
                        x_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_1)
                    x_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_phi)
                return x_rv

        ir.assert_structural_equal(After, Expected)


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
                x_iter = pl.yield_(y)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:  # noqa: F841
            for i, (x_iter,) in pl.range(10, init_values=(x_0,)):
                if i < 5:
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                x_iter: pl.Tensor[[64], pl.FP32] = pl.yield_(phi)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


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
                x_iter = pl.yield_(y)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0
            brk: pl.Scalar[pl.BOOL] = False
            for (x_iter,) in pl.while_(init_values=(x_0,)):
                pl.cond(i_idx < 10 and not brk)  # noqa: F841
                if i_idx > 5:
                    brk: pl.Scalar[pl.BOOL] = True
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                x_iter = pl.yield_(phi)
                if not brk:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + 1
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


def _build_break_and_continue_expected() -> ir.Program:
    """Build Expected IR for break_and_continue_in_for using IRBuilder.

    The DSL parser cannot express nested phi-node patterns where an if/else has
    yield-returning variables at multiple nesting levels. IRBuilder allows direct
    construction of the correct IR.
    """
    ib = IRBuilder()
    tt = _tt()

    with ib.program("Expected") as prog:
        with ib.function("kernel", type=ir.FunctionType.InCore) as f:
            x_0 = f.param("x_0", tt)
            f.return_type(tt)

            i_idx = ib.let("i_idx", _ci(0))
            brk = ib.let("brk", _cb(False))

            cond = _cb(True)
            with ib.while_loop(cond) as loop:
                x_iter = loop.iter_arg("x_iter", x_0)
                loop.set_condition(
                    ir.And(ir.Lt(i_idx, _ci(10), _BOOL, _SPAN), ir.Not(brk, _BOOL, _SPAN), _BOOL, _SPAN)
                )

                # outer if: i_idx < 3
                with ib.if_stmt(ir.Lt(i_idx, _ci(3), _BOOL, _SPAN)) as if_outer:
                    if_outer.return_var("phi2", tt)
                    ib.emit(ir.YieldStmt([x_iter], _SPAN))
                    if_outer.else_()
                    y = ib.let("y", ir.Call(ir.Op("tensor.add"), [x_iter, x_iter], tt, _SPAN))
                    # inner if: i_idx > 7
                    with ib.if_stmt(ir.Gt(i_idx, _ci(7), _BOOL, _SPAN)) as if_inner:
                        if_inner.return_var("phi1", tt)
                        ib.assign(brk, _cb(True))
                        ib.emit(ir.YieldStmt([y], _SPAN))
                        if_inner.else_()
                        ib.emit(ir.YieldStmt([y], _SPAN))
                    phi1 = if_inner.output(0)
                    ib.emit(ir.YieldStmt([phi1], _SPAN))
                phi2 = if_outer.output(0)

                loop.return_var("x_rv")
                ib.emit(ir.YieldStmt([phi2], _SPAN))

                with ib.if_stmt(ir.Not(brk, _BOOL, _SPAN)):
                    ib.assign(i_idx, ir.Add(i_idx, _ci(1), _IDX, _SPAN))

            ib.return_stmt(loop.output(0))
        prog.add_function(f.get_result())

    return prog.get_result()


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
                x_iter = pl.yield_(y)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    Expected = _build_break_and_continue_expected()
    ir.assert_structural_equal(After, Expected)


def test_no_break_continue_noop():
    """Pass is identity when no break/continue (InCore SSA form)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)
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
                a_iter, b_iter = pl.yield_(a_new, b_new)
            return a_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(
            self,
            a_0: pl.Tensor[[64], pl.FP32],
            b_0: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:  # noqa: F841
            for i, (a_iter, b_iter) in pl.range(10, init_values=(a_0, b_0)):  # noqa: F841
                if i < 5:
                    a_phi, b_phi = pl.yield_(a_iter, b_iter)
                else:
                    a_new: pl.Tensor[[64], pl.FP32] = pl.add(a_iter, b_iter)
                    b_new: pl.Tensor[[64], pl.FP32] = pl.add(b_iter, a_iter)
                    a_phi, b_phi = pl.yield_(a_new, b_new)
                a_iter, b_iter = pl.yield_(a_phi, b_phi)
            return a_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


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
                x_iter = pl.yield_(z)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)  # noqa: F841
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:  # noqa: F841
            for i, (x_iter,) in pl.range(10, init_values=(x_0,)):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i < 5:
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                else:
                    z: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(z)
                x_iter: pl.Tensor[[64], pl.FP32] = pl.yield_(phi)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


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
                x_iter = pl.yield_(y)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 10
            brk: pl.Scalar[pl.BOOL] = False
            for (x_iter,) in pl.while_(init_values=(x_0,)):
                pl.cond(i_idx > 0 and not brk)
                if i_idx < 3:
                    brk: pl.Scalar[pl.BOOL] = True
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                x_iter = pl.yield_(phi)
                if not brk:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + -1
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


def test_aic_function_type():
    """Pass processes AIC function type."""

    @pl.program  # noqa: F841
    class Before:
        @pl.function(type=pl.FunctionType.AIC, strict_ssa=True)
        def aic_kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i < 5:
                    continue
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                x_iter = pl.yield_(y)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, strict_ssa=True)
        def aic_kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(10, init_values=(x_0,)):
                if i < 5:
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                x_iter: pl.Tensor[[64], pl.FP32] = pl.yield_(phi)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


def test_continue_no_iter_args():
    """Continue in loop with no carried state."""  # noqa: F841

    # noqa: F841
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(0, 10, 1):
                if i < 5:
                    continue
                _y: pl.Tensor[[64], pl.FP32] = pl.add(x_0, x_0)
            return x_0

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(10):
                if i < 5:
                    pass
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_0, x_0)  # noqa: F841
            return x_0

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


def test_break_no_iter_args():
    """Break in loop with no carried state."""

    @pl.program
    class Before:  # noqa: F841
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(0, 10, 1):
                if i > 5:
                    break
                _y: pl.Tensor[[64], pl.FP32] = pl.add(x_0, x_0)
            return x_0

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0
            brk: pl.Scalar[pl.BOOL] = False
            while i_idx < 10 and not brk:
                if i_idx > 5:
                    brk: pl.Scalar[pl.BOOL] = True
                    pl.yield_()
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_0, x_0)  # noqa: F841
                    pl.yield_()
                if not brk:
                    i_idx = i_idx + 1
            return x_0

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


def _build_multiple_continues_expected() -> ir.Program:
    """Build Expected for test_multiple_continues_in_body."""
    ib = IRBuilder()
    tt = _tt()

    with ib.program("Expected") as prog:
        with ib.function("kernel", type=ir.FunctionType.InCore) as f:
            x_0 = f.param("x_0", tt)
            f.return_type(tt)

            i = ib.var("i", _IDX_T)
            with ib.for_loop(i, _ci(0), _ci(10), _ci(1)) as loop:
                x_iter = loop.iter_arg("x_iter", x_0)

                # outer if: i < 2
                with ib.if_stmt(ir.Lt(i, _ci(2), _BOOL, _SPAN)) as if_outer:
                    if_outer.return_var("phi2", tt)
                    ib.emit(ir.YieldStmt([x_iter], _SPAN))
                    if_outer.else_()
                    y = ib.let("y", ir.Call(ir.Op("tensor.add"), [x_iter, x_iter], tt, _SPAN))
                    # inner if: i > 8
                    with ib.if_stmt(ir.Gt(i, _ci(8), _BOOL, _SPAN)) as if_inner:
                        if_inner.return_var("phi1", tt)
                        ib.emit(ir.YieldStmt([x_iter], _SPAN))
                        if_inner.else_()
                        z = ib.let("z", ir.Call(ir.Op("tensor.add"), [y, y], tt, _SPAN))
                        ib.emit(ir.YieldStmt([z], _SPAN))
                    phi1 = if_inner.output(0)
                    ib.emit(ir.YieldStmt([phi1], _SPAN))
                phi2 = if_outer.output(0)

                loop.return_var("x_rv")
                ib.emit(ir.YieldStmt([phi2], _SPAN))

            ib.return_stmt(loop.output(0))
        prog.add_function(f.get_result())

    return prog.get_result()


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
                x_iter = pl.yield_(z)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    Expected = _build_multiple_continues_expected()
    ir.assert_structural_equal(After, Expected)


def _build_back_to_back_breaks_expected() -> ir.Program:
    """Build Expected for test_back_to_back_breaks."""
    ib = IRBuilder()
    tt = _tt()

    with ib.program("Expected") as prog:
        with ib.function("kernel", type=ir.FunctionType.InCore) as f:
            x_0 = f.param("x_0", tt)
            f.return_type(tt)

            i_idx = ib.let("i_idx", _ci(0))
            brk = ib.let("brk", _cb(False))

            cond = _cb(True)
            with ib.while_loop(cond) as loop:
                x_iter = loop.iter_arg("x_iter", x_0)
                loop.set_condition(
                    ir.And(ir.Lt(i_idx, _ci(10), _BOOL, _SPAN), ir.Not(brk, _BOOL, _SPAN), _BOOL, _SPAN)
                )

                # outer if: i_idx > 8
                with ib.if_stmt(ir.Gt(i_idx, _ci(8), _BOOL, _SPAN)) as if_outer:
                    if_outer.return_var("phi2", tt)
                    ib.assign(brk, _cb(True))
                    ib.emit(ir.YieldStmt([x_iter], _SPAN))
                    if_outer.else_()
                    y = ib.let("y", ir.Call(ir.Op("tensor.add"), [x_iter, x_iter], tt, _SPAN))
                    # inner if: i_idx > 5
                    with ib.if_stmt(ir.Gt(i_idx, _ci(5), _BOOL, _SPAN)) as if_inner:
                        if_inner.return_var("phi1", tt)
                        ib.assign(brk, _cb(True))
                        ib.emit(ir.YieldStmt([y], _SPAN))
                        if_inner.else_()
                        ib.emit(ir.YieldStmt([y], _SPAN))
                    phi1 = if_inner.output(0)
                    ib.emit(ir.YieldStmt([phi1], _SPAN))
                phi2 = if_outer.output(0)

                loop.return_var("x_rv")
                ib.emit(ir.YieldStmt([phi2], _SPAN))

                with ib.if_stmt(ir.Not(brk, _BOOL, _SPAN)):
                    ib.assign(i_idx, ir.Add(i_idx, _ci(1), _IDX, _SPAN))

            ib.return_stmt(loop.output(0))
        prog.add_function(f.get_result())

    return prog.get_result()


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
                x_iter = pl.yield_(y)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    Expected = _build_back_to_back_breaks_expected()
    ir.assert_structural_equal(After, Expected)


# noqa: F841
def _build_break_then_continue_expected() -> ir.Program:  # noqa: F841
    """Build Expected for test_break_then_continue."""
    ib = IRBuilder()
    tt = _tt()

    with ib.program("Expected") as prog:
        with ib.function("kernel", type=ir.FunctionType.InCore) as f:
            x_0 = f.param("x_0", tt)
            f.return_type(tt)

            i_idx = ib.let("i_idx", _ci(0))
            brk = ib.let("brk", _cb(False))

            cond = _cb(True)
            with ib.while_loop(cond) as loop:
                x_iter = loop.iter_arg("x_iter", x_0)
                loop.set_condition(
                    ir.And(ir.Lt(i_idx, _ci(10), _BOOL, _SPAN), ir.Not(brk, _BOOL, _SPAN), _BOOL, _SPAN)
                )

                # outer if: i_idx > 8
                with ib.if_stmt(ir.Gt(i_idx, _ci(8), _BOOL, _SPAN)) as if_outer:
                    if_outer.return_var("phi2", tt)
                    ib.assign(brk, _cb(True))
                    ib.emit(ir.YieldStmt([x_iter], _SPAN))
                    if_outer.else_()
                    y = ib.let("y", ir.Call(ir.Op("tensor.add"), [x_iter, x_iter], tt, _SPAN))
                    # inner if: i_idx < 3
                    with ib.if_stmt(ir.Lt(i_idx, _ci(3), _BOOL, _SPAN)) as if_inner:
                        if_inner.return_var("phi1", tt)
                        ib.emit(ir.YieldStmt([x_iter], _SPAN))
                        if_inner.else_()
                        z = ib.let("z", ir.Call(ir.Op("tensor.add"), [y, y], tt, _SPAN))
                        ib.emit(ir.YieldStmt([z], _SPAN))
                    phi1 = if_inner.output(0)  # noqa: F841
                    # NOTE: yields x_iter, not phi1
                    ib.emit(ir.YieldStmt([x_iter], _SPAN))
                phi2 = if_outer.output(0)

                loop.return_var("x_rv")
                ib.emit(ir.YieldStmt([phi2], _SPAN))

                with ib.if_stmt(ir.Not(brk, _BOOL, _SPAN)):
                    ib.assign(i_idx, ir.Add(i_idx, _ci(1), _IDX, _SPAN))

            ib.return_stmt(loop.output(0))
        prog.add_function(f.get_result())

    return prog.get_result()


def test_break_then_continue():
    """Break guard first, then continue guard in same body."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)  # noqa: F841
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:  # noqa: F841
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                if i > 8:
                    break
                y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                if i < 3:
                    continue
                z: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                x_iter = pl.yield_(z)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    Expected = _build_break_then_continue_expected()
    ir.assert_structural_equal(After, Expected)


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
                a_iter, b_iter = pl.yield_(a_new, b_new)
            return a_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(
            self,
            a_0: pl.Tensor[[64], pl.FP32],
            b_0: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0
            brk: pl.Scalar[pl.BOOL] = False
            for a_iter, b_iter in pl.while_(init_values=(a_0, b_0)):
                pl.cond(i_idx < 10 and not brk)
                if i_idx > 5:
                    brk: pl.Scalar[pl.BOOL] = True
                    a_phi, b_phi = pl.yield_(a_iter, b_iter)
                else:
                    a_new: pl.Tensor[[64], pl.FP32] = pl.add(a_iter, b_iter)
                    b_new: pl.Tensor[[64], pl.FP32] = pl.add(b_iter, a_iter)
                    a_phi, b_phi = pl.yield_(a_new, b_new)
                a_iter, b_iter = pl.yield_(a_phi, b_phi)
                if not brk:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + 1
            return a_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


# ===========================================================================
# Unconditional break/continue
# ===========================================================================
# noqa: F841


def test_unconditional_break():
    """Bare break as first statement — loop executes 0 iterations effectively."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                break
                x_iter = pl.yield_(x_iter)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0
            brk: pl.Scalar[pl.BOOL] = False
            for (x_iter,) in pl.while_(init_values=(x_0,)):
                pl.cond(i_idx < 10 and not brk)
                brk: pl.Scalar[pl.BOOL] = True
                x_iter = pl.yield_(x_iter)
                if not brk:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + 1
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


# noqa: F841


def test_unconditional_continue():
    """Bare continue as first statement — all iterations are skipped."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                continue
                x_iter = pl.yield_(x_iter)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(10, init_values=(x_0,)):
                x_iter: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


# ===========================================================================
# Nested loops
# ===========================================================================


def test_nested_loops_only_inner():
    """Only inner loop with continue is transformed, outer loop unchanged."""

    @pl.program  # noqa: F841
    class Before:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(0, 4, 1, init_values=(x_0,)):
                for j, (x_inner,) in pl.range(0, 8, 1, init_values=(x_outer,)):
                    if j < 2:
                        continue
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                    x_inner = pl.yield_(y)
                x_outer = pl.yield_(x_inner)
            return x_outer  # noqa: F841

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(4, init_values=(x_0,)):
                for j, (x_inner,) in pl.range(8, init_values=(x_outer,)):
                    if j < 2:
                        phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)
                    else:
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                        phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                    x_inner: pl.Tensor[[64], pl.FP32] = pl.yield_(phi)
                x_outer: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)
            return x_outer

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


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
                    x_inner = pl.yield_(y)
                if i > 2:
                    break
                x_outer = pl.yield_(x_inner)
            return x_outer

    # noqa: F841
    After = passes.ctrl_flow_transform()(Before)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0
            brk_o: pl.Scalar[pl.BOOL] = False
            for (x_outer,) in pl.while_(init_values=(x_0,)):
                pl.cond(i_idx < 4 and not brk_o)
                j_idx: pl.Scalar[pl.INDEX] = 0
                brk_i: pl.Scalar[pl.BOOL] = False
                for (x_inner,) in pl.while_(init_values=(x_outer,)):
                    pl.cond(j_idx < 8 and not brk_i)
                    if j_idx > 3:
                        brk_i: pl.Scalar[pl.BOOL] = True
                        x_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)
                    else:
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                        x_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                    x_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_phi)  # noqa: F841
                    if not brk_i:
                        j_idx: pl.Scalar[pl.INDEX] = j_idx + 1
                if i_idx > 2:
                    brk_o: pl.Scalar[pl.BOOL] = True
                    o_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                else:
                    o_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                o_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(o_phi)
                if not brk_o:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + 1
            return o_rv

    ir.assert_structural_equal(After, Expected)


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
                    x_inner = pl.yield_(y)
                if i < 2:
                    continue
                x_outer = pl.yield_(x_inner)
            return x_outer

    After = passes.ctrl_flow_transform()(Before)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(4, init_values=(x_0,)):
                j_idx: pl.Scalar[pl.INDEX] = 0
                brk_i: pl.Scalar[pl.BOOL] = False
                for (x_inner,) in pl.while_(init_values=(x_outer,)):
                    pl.cond(j_idx < 8 and not brk_i)  # noqa: F841
                    if j_idx > 3:
                        brk_i: pl.Scalar[pl.BOOL] = True
                        x_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)
                    else:
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                        x_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                    x_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_phi)  # noqa: F841
                    if not brk_i:
                        j_idx: pl.Scalar[pl.INDEX] = j_idx + 1
                if i < 2:
                    o_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                else:
                    o_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                x_outer: pl.Tensor[[64], pl.FP32] = pl.yield_(o_phi)
            return x_outer

    ir.assert_structural_equal(After, Expected)


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
                    x_inner = pl.yield_(y)
                x_outer = pl.yield_(x_inner)
            return x_outer

    After = passes.ctrl_flow_transform()(Before)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(4, init_values=(x_0,)):
                if i < 1:
                    x_outer_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                else:
                    for j, (x_inner,) in pl.range(8, init_values=(x_outer,)):  # noqa: F841
                        if j < 2:
                            x_inner_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)
                        else:
                            y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                            x_inner_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                        x_inner: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner_phi)
                    x_outer_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                x_outer: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer_phi)
            return x_outer

    ir.assert_structural_equal(After, Expected)


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
                    x_inner = pl.yield_(y)
                x_outer = pl.yield_(x_inner)
            return x_outer

    After = passes.ctrl_flow_transform()(Before)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_outer,) in pl.range(4, init_values=(x_0,)):
                j_idx: pl.Scalar[pl.INDEX] = 0
                brk_i: pl.Scalar[pl.BOOL] = False
                for (x_inner,) in pl.while_(init_values=(x_outer,)):
                    pl.cond(j_idx < 8 and not brk_i)
                    if j_idx < 2:
                        x_phi2: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)
                    else:
                        if j_idx > 5:  # noqa: F841
                            brk_i: pl.Scalar[pl.BOOL] = True
                            x_phi1: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)
                        else:
                            y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                            x_phi1: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                        x_phi2: pl.Tensor[[64], pl.FP32] = pl.yield_(x_phi1)
                    x_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(x_phi2)
                    if not brk_i:
                        j_idx: pl.Scalar[pl.INDEX] = j_idx + 1
                x_outer: pl.Tensor[[64], pl.FP32] = pl.yield_(x_rv)
            return x_outer

    ir.assert_structural_equal(After, Expected)


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
                    x_inner = pl.yield_(y)
                if i > 2:
                    break
                x_outer = pl.yield_(x_inner)
            return x_outer

    After = passes.ctrl_flow_transform()(Before)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0
            brk_o: pl.Scalar[pl.BOOL] = False
            for (x_outer,) in pl.while_(init_values=(x_0,)):
                pl.cond(i_idx < 4 and not brk_o)
                if i_idx < 1:
                    o_phi2: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                else:
                    j_idx: pl.Scalar[pl.INDEX] = 0
                    brk_i: pl.Scalar[pl.BOOL] = False
                    for (x_inner,) in pl.while_(init_values=(x_outer,)):
                        pl.cond(j_idx < 8 and not brk_i)
                        if j_idx < 2:
                            i_phi2: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)  # noqa: F841
                        else:
                            if j_idx > 5:
                                brk_i: pl.Scalar[pl.BOOL] = True
                                i_phi1: pl.Tensor[[64], pl.FP32] = pl.yield_(x_inner)
                            else:
                                y: pl.Tensor[[64], pl.FP32] = pl.add(x_inner, x_inner)
                                i_phi1: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                            i_phi2: pl.Tensor[[64], pl.FP32] = pl.yield_(i_phi1)
                        i_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(i_phi2)  # noqa: F841
                        if not brk_i:
                            j_idx: pl.Scalar[pl.INDEX] = j_idx + 1
                    if i_idx > 2:
                        brk_o: pl.Scalar[pl.BOOL] = True
                        o_phi1: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                    else:
                        o_phi1: pl.Tensor[[64], pl.FP32] = pl.yield_(x_outer)
                    o_phi2: pl.Tensor[[64], pl.FP32] = pl.yield_(o_phi1)
                o_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(o_phi2)
                if not brk_o:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + 1
            return o_rv

    ir.assert_structural_equal(After, Expected)


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
                        x_l3 = pl.yield_(y)
                    if j > 1:
                        break
                    x_l2 = pl.yield_(x_l3)
                if i > 0:
                    break
                x_l1 = pl.yield_(x_l2)
            return x_l1

    After = passes.ctrl_flow_transform()(Before)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0
            brk1: pl.Scalar[pl.BOOL] = False
            for (x_l1,) in pl.while_(init_values=(x_0,)):
                pl.cond(i_idx < 3 and not brk1)
                j_idx: pl.Scalar[pl.INDEX] = 0
                brk2: pl.Scalar[pl.BOOL] = False
                for (x_l2,) in pl.while_(init_values=(x_l1,)):
                    pl.cond(j_idx < 4 and not brk2)
                    k_idx: pl.Scalar[pl.INDEX] = 0
                    brk3: pl.Scalar[pl.BOOL] = False
                    for (x_l3,) in pl.while_(init_values=(x_l2,)):
                        pl.cond(k_idx < 5 and not brk3)  # noqa: F841
                        if k_idx > 2:
                            brk3: pl.Scalar[pl.BOOL] = True  # noqa: F841
                            l3_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_l3)
                        else:
                            y: pl.Tensor[[64], pl.FP32] = pl.add(x_l3, x_l3)
                            l3_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                        l3_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(l3_phi)  # noqa: F841
                        if not brk3:
                            k_idx: pl.Scalar[pl.INDEX] = k_idx + 1
                    if j_idx > 1:  # noqa: F841
                        brk2: pl.Scalar[pl.BOOL] = True
                        l2_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_l2)
                    else:
                        l2_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_l2)
                    l2_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(l2_phi)  # noqa: F841
                    if not brk2:
                        j_idx: pl.Scalar[pl.INDEX] = j_idx + 1
                if i_idx > 0:
                    brk1: pl.Scalar[pl.BOOL] = True
                    l1_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_l1)
                else:
                    l1_phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_l1)
                l1_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(l1_phi)
                if not brk1:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + 1
            return l1_rv

    ir.assert_structural_equal(After, Expected)


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
                x_iter = pl.yield_(y)
            return x_iter

    @pl.program
    class Expected:  # noqa: F841
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (x_iter,) in pl.range(10, init_values=(x_0,)):
                if i > 5:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                else:
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                x_iter: pl.Tensor[[64], pl.FP32] = pl.yield_(phi)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


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
                x_iter = pl.yield_(y)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0  # noqa: F841
            brk: pl.Scalar[pl.BOOL] = False
            for (x_iter,) in pl.while_(init_values=(x_0,)):
                pl.cond(i_idx < 10 and not brk)
                if i_idx < 7:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                else:
                    brk: pl.Scalar[pl.BOOL] = True
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                x_iter = pl.yield_(phi)
                if not brk:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + 1
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


def _build_if_else_continue_then_break_else_expected() -> ir.Program:
    """Build Expected for test_if_else_continue_then_break_else."""
    ib = IRBuilder()
    tt = _tt()

    with ib.program("Expected") as prog:
        with ib.function("kernel", type=ir.FunctionType.InCore) as f:
            x_0 = f.param("x_0", tt)
            f.return_type(tt)

            i_idx = ib.let("i_idx", _ci(0))
            brk = ib.let("brk", _cb(False))

            cond = _cb(True)
            with ib.while_loop(cond) as loop:
                x_iter = loop.iter_arg("x_iter", x_0)
                loop.set_condition(
                    ir.And(ir.Lt(i_idx, _ci(10), _BOOL, _SPAN), ir.Not(brk, _BOOL, _SPAN), _BOOL, _SPAN)
                )

                y = ib.let("y", ir.Call(ir.Op("tensor.add"), [x_iter, x_iter], tt, _SPAN))

                # outer if: i_idx < 3
                with ib.if_stmt(ir.Lt(i_idx, _ci(3), _BOOL, _SPAN)) as if_outer:
                    if_outer.return_var("phi2", tt)
                    ib.emit(ir.YieldStmt([y], _SPAN))
                    if_outer.else_()
                    # inner if: i_idx > 7
                    with ib.if_stmt(ir.Gt(i_idx, _ci(7), _BOOL, _SPAN)) as if_inner:
                        if_inner.return_var("phi1", tt)
                        ib.assign(brk, _cb(True))
                        ib.emit(ir.YieldStmt([x_iter], _SPAN))
                        if_inner.else_()
                        ib.emit(ir.YieldStmt([x_iter], _SPAN))
                    phi1 = if_inner.output(0)
                    ib.emit(ir.YieldStmt([phi1], _SPAN))
                phi2 = if_outer.output(0)

                loop.return_var("x_rv")
                ib.emit(ir.YieldStmt([phi2], _SPAN))

                with ib.if_stmt(ir.Not(brk, _BOOL, _SPAN)):
                    ib.assign(i_idx, ir.Add(i_idx, _ci(1), _IDX, _SPAN))

            ib.return_stmt(loop.output(0))
        prog.add_function(f.get_result())

    return prog.get_result()


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
                x_iter = pl.yield_(y)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    Expected = _build_if_else_continue_then_break_else_expected()
    ir.assert_structural_equal(After, Expected)


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
                x_iter = pl.yield_(y)
            return x_iter

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:  # noqa: F841
            for i, (x_iter,) in pl.range(10, init_values=(x_0,)):
                if i < 5:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_0)  # noqa: F841
                if i < 2:
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                else:
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                x_iter: pl.Tensor[[64], pl.FP32] = pl.yield_(phi)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


def _build_deeply_nested_continue_expected() -> ir.Program:
    """Build Expected for test_deeply_nested_if_with_continue."""
    ib = IRBuilder()
    tt = _tt()

    with ib.program("Expected") as prog:
        with ib.function("kernel", type=ir.FunctionType.InCore) as f:
            x_0 = f.param("x_0", tt)
            f.return_type(tt)

            i = ib.var("i", _IDX_T)
            with ib.for_loop(i, _ci(0), _ci(10), _ci(1)) as loop:
                x_iter = loop.iter_arg("x_iter", x_0)

                # if i < 8
                with ib.if_stmt(ir.Lt(i, _ci(8), _BOOL, _SPAN)) as if_l1:
                    if_l1.return_var("phi3", tt)
                    # if i < 5
                    with ib.if_stmt(ir.Lt(i, _ci(5), _BOOL, _SPAN)) as if_l2:
                        if_l2.return_var("phi2", tt)
                        # if i < 2
                        with ib.if_stmt(ir.Lt(i, _ci(2), _BOOL, _SPAN)) as if_l3:
                            if_l3.return_var("phi1", tt)
                            ib.emit(ir.YieldStmt([x_iter], _SPAN))
                            if_l3.else_()
                            ib.emit(ir.YieldStmt([x_iter], _SPAN))
                        phi1 = if_l3.output(0)
                        ib.emit(ir.YieldStmt([phi1], _SPAN))
                        if_l2.else_()
                        ib.emit(ir.YieldStmt([x_iter], _SPAN))
                    phi2 = if_l2.output(0)
                    ib.emit(ir.YieldStmt([phi2], _SPAN))
                    if_l1.else_()
                    ib.emit(ir.YieldStmt([x_iter], _SPAN))
                phi3 = if_l1.output(0)

                ib.let("y", ir.Call(ir.Op("tensor.add"), [x_iter, x_iter], tt, _SPAN))
                loop.return_var("x_rv")
                ib.emit(ir.YieldStmt([phi3], _SPAN))

            ib.return_stmt(loop.output(0))
        prog.add_function(f.get_result())

    return prog.get_result()


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
                x_iter = pl.yield_(y)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    Expected = _build_deeply_nested_continue_expected()
    ir.assert_structural_equal(After, Expected)


def _build_deeply_nested_break_expected() -> ir.Program:
    """Build Expected for test_deeply_nested_if_with_break."""
    ib = IRBuilder()
    tt = _tt()

    with ib.program("Expected") as prog:
        with ib.function("kernel", type=ir.FunctionType.InCore) as f:
            x_0 = f.param("x_0", tt)
            f.return_type(tt)

            i_idx = ib.let("i_idx", _ci(0))
            brk = ib.let("brk", _cb(False))

            cond = _cb(True)
            with ib.while_loop(cond) as loop:
                x_iter = loop.iter_arg("x_iter", x_0)
                loop.set_condition(
                    ir.And(ir.Lt(i_idx, _ci(10), _BOOL, _SPAN), ir.Not(brk, _BOOL, _SPAN), _BOOL, _SPAN)
                )

                # if i_idx > 3
                with ib.if_stmt(ir.Gt(i_idx, _ci(3), _BOOL, _SPAN)) as if_l1:
                    if_l1.return_var("phi4", tt)
                    # if i_idx > 5
                    with ib.if_stmt(ir.Gt(i_idx, _ci(5), _BOOL, _SPAN)) as if_l2:
                        if_l2.return_var("phi3", tt)
                        # if i_idx > 7
                        with ib.if_stmt(ir.Gt(i_idx, _ci(7), _BOOL, _SPAN)) as if_l3:
                            if_l3.return_var("phi2", tt)
                            ib.assign(brk, _cb(True))
                            ib.emit(ir.YieldStmt([x_iter], _SPAN))
                            if_l3.else_()
                            ib.emit(ir.YieldStmt([x_iter], _SPAN))
                        phi2 = if_l3.output(0)
                        ib.emit(ir.YieldStmt([phi2], _SPAN))
                        if_l2.else_()
                        ib.emit(ir.YieldStmt([x_iter], _SPAN))
                    phi3 = if_l2.output(0)
                    ib.emit(ir.YieldStmt([phi3], _SPAN))
                    if_l1.else_()
                    ib.emit(ir.YieldStmt([x_iter], _SPAN))
                phi4 = if_l1.output(0)  # noqa: F841

                # if not brk: y = add(...); yield phi4; else: yield phi4
                with ib.if_stmt(ir.Not(brk, _BOOL, _SPAN)) as if_guard:
                    if_guard.return_var("phi5", tt)
                    y = ib.let("y", ir.Call(ir.Op("tensor.add"), [x_iter, x_iter], tt, _SPAN))  # noqa: F841
                    ib.emit(ir.YieldStmt([phi4], _SPAN))
                    if_guard.else_()
                    ib.emit(ir.YieldStmt([phi4], _SPAN))
                phi5 = if_guard.output(0)

                loop.return_var("x_rv")
                ib.emit(ir.YieldStmt([phi5], _SPAN))

                with ib.if_stmt(ir.Not(brk, _BOOL, _SPAN)):
                    ib.assign(i_idx, ir.Add(i_idx, _ci(1), _IDX, _SPAN))

            ib.return_stmt(loop.output(0))
        prog.add_function(f.get_result())

    return prog.get_result()


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
                x_iter = pl.yield_(y)
            return x_iter

    After = passes.ctrl_flow_transform()(Before)
    Expected = _build_deeply_nested_break_expected()
    ir.assert_structural_equal(After, Expected)


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
                x_iter = pl.yield_(y)
            return x_iter

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = self.incore_kernel(x)
            return y

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def incore_kernel(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i_idx: pl.Scalar[pl.INDEX] = 0
            brk: pl.Scalar[pl.BOOL] = False
            for (x_iter,) in pl.while_(init_values=(x_0,)):
                pl.cond(i_idx < 10 and not brk)  # noqa: F841
                if i_idx > 5:
                    brk: pl.Scalar[pl.BOOL] = True
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(y)
                x_iter = pl.yield_(phi)
                if not brk:
                    i_idx: pl.Scalar[pl.INDEX] = i_idx + 1
            return x_iter

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = self.incore_kernel(x)
            return y

    After = passes.ctrl_flow_transform()(Before)
    ir.assert_structural_equal(After, Expected)


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
    After = passes.ctrl_flow_transform()(after_outline)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
        def main_incore_0(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:  # noqa: F841
            for i, (x_iter,) in pl.range(10, init_values=(x_0,)):
                if i < 5:
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_iter)
                else:
                    x_new: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    phi: pl.Tensor[[64], pl.FP32] = pl.yield_(x_new)
                x_rv: pl.Tensor[[64], pl.FP32] = pl.yield_(phi)
            return x_rv

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            x_rv: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x_0)
            return x_rv

    ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
