# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for FlattenCallExpr pass.

This pass flattens nested call expressions into three-address code by extracting
nested calls into temporary variables.

Tests use the Before/Expected pattern with @pl.program decorator.
Uses assert_structural_equal to compare pass output with expected IR.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.pypto_core import passes as _core_passes


def NormalizeIR(program):
    """Normalize Expected IR structure to match flatten_call_expr pass output.

    This is a test comparison utility, not a second pass under test.
    The flatten_call_expr pass internally applies normalize_stmt_structure
    before call expression flattening. Expected IR from the DSL must go
    through the same structural transformation for assert_structural_equal
    to succeed.
    """
    return passes.normalize_stmt_structure()(program)


class TestFlattenCallInCallArgs:
    """Tests for flattening nested calls in call arguments."""

    def test_single_nested_call_in_args(self):
        """Test flattening a single nested call in arguments: mul(add(x, 1.0), 2.0)"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Nested call: mul(add(x, 1.0), 2.0)
                result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(x, 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Flattened: temp variable for inner call
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                result: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v0, 2.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_multiple_nested_calls_in_args(self):
        """Test flattening multiple nested calls: add(mul(x, 2.0), mul(y, 3.0))"""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                # Multiple nested calls in args
                result: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), pl.mul(y, 3.0))
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                # Both nested calls extracted
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.mul(y, 3.0)
                result: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v0, t__tmp_v1)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_deeply_nested_calls(self):
        """Test deeply nested calls: mul(add(exp(x), 1.0), 2.0)"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Deeply nested: mul(add(exp(x), 1.0), 2.0)
                result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(pl.exp(x), 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # All nested calls extracted in order
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v0, 1.0)
                result: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v1, 2.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenCallInBinaryExpr:
    """Tests for flattening calls in binary expression operands.

    Note: Currently tensor operations don't support scalar binary expressions with calls,
    so these tests verify the pass handles tensor operations correctly.
    """

    def test_call_in_left_operand(self):
        """Test call in left operand: add(mul(x, 2.0), x)"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Nested call in left operand
                result: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), x)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                result: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v0, x)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_call_in_right_operand(self):
        """Test call in right operand: add(x, exp(x))"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Nested call in right operand
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, pl.exp(x))
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, t__tmp_v0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_calls_in_both_operands(self):
        """Test calls in both operands: add(mul(x, 2.0), exp(x))"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Nested calls in both operands
                result: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), pl.exp(x))
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v0, t__tmp_v1)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenCallInUnaryExpr:
    """Tests for flattening calls in unary-like expression operands."""

    def test_call_in_unary_operand(self):
        """Test call in unary-like expression: exp(exp(x))"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Unary-like operation with nested call: exp(exp(x))
                result: pl.Tensor[[64], pl.FP32] = pl.exp(pl.exp(x))
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                result: pl.Tensor[[64], pl.FP32] = pl.exp(t__tmp_v0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenCallInIfCondition:
    """Tests for flattening calls in if statement conditions.

    Note: In the current DSL, if conditions use scalar comparisons, not calls.
    This test ensures the pass handles if statements correctly.
    """

    def test_if_with_nested_calls_in_branches(self):
        """Test nested calls inside if branches"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i in pl.range(10):
                    if i > 5:
                        # Nested call in then branch
                        result = pl.add(pl.mul(result, 2.0), 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i in pl.range(10):
                    if i > 5:
                        t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.mul(result, 2.0)
                        result = pl.add(t__tmp_v0, 1.0)
                return result

        After = passes.flatten_call_expr()(passes.convert_to_ssa()(Before))
        ir.assert_structural_equal(After, NormalizeIR(passes.convert_to_ssa()(Expected)))

    def test_nested_calls_before_and_after_if(self):
        """Test nested calls before and after if statement

        This test verifies that temporary variables are correctly inserted when:
        1. There's a nested call before an if statement
        2. There's a nested call after the if statement
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                y: pl.Tensor[[64], pl.FP32],
                z: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                c: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                # Nested call before if
                x: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(y, 1.0), z)
                for i in pl.range(10):
                    if i > 5:
                        _result: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)
                # Nested call after if
                a: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(b, 2.0), c)
                return pl.add(x, a)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                y: pl.Tensor[[64], pl.FP32],
                z: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                c: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                # Flattened: temp variable for first nested call
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(y, 1.0)
                x: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v0, z)
                for i in pl.range(10):
                    if i > 5:
                        _result: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)
                # Flattened: temp variable for second nested call
                t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.add(b, 2.0)
                a: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v1, c)
                t__tmp_v2: pl.Tensor[[64], pl.FP32] = pl.add(x, a)
                return t__tmp_v2

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_if_with_get_block_idx_in_condition(self):
        """Test get_block_idx call in if condition"""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # get_block_idx() in if condition
                if pl.tile.get_block_idx() < 10:
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.tile.store(tile, offsets=[0, 0], output_tensor=output)
                return output

        @pl.program
        class Expected:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                if t__tmp_v0 < 10:
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.tile.store(tile, offsets=[0, 0], output_tensor=output)
                return output

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenCallInWhileCondition:
    """Tests for flattening calls in WhileStmt conditions.

    The WhileStmt visitor (flatten_call_expr_pass.cpp:302-326) visits the
    condition, then saves any extracted temporaries as ``condition_pending``
    and restores them for the parent ``SeqStmts`` so they land *before* the
    while loop. A nested call in the condition therefore hoists to a sibling
    statement preceding the ``for ... in pl.while_(...)`` node, leaving the
    iter-arg / yield machinery untouched (the visitor only rewrites
    ``condition_`` and ``body_`` via MutableCopy).
    """

    def test_nested_call_in_while_condition(self):
        """`pl.cond(get_block_idx() < 10)` hoists the call before the while loop.

        The condition is a ``Lt`` whose left operand is the ``get_block_idx()``
        Call. ProcessBinaryExpr extracts that operand into ``t__tmp_v0``; the
        WhileStmt visitor surfaces it to the enclosing SeqStmts so it sits as a
        sibling before the loop, and the condition becomes ``t__tmp_v0 < 10``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                n: pl.Scalar[pl.INT64] = 0
                for cnt, x_iter in pl.while_(init_values=(n, x_0)):
                    pl.cond(pl.tile.get_block_idx() < 10)
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    c2: pl.Scalar[pl.INT64] = cnt + 1
                    cnt, x_iter = pl.yield_(c2, y)
                return x_iter

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                n: pl.Scalar[pl.INT64] = 0
                # get_block_idx() hoisted to a sibling before the while loop.
                t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for cnt, x_iter in pl.while_(init_values=(n, x_0)):
                    pl.cond(t__tmp_v0 < 10)
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    c2: pl.Scalar[pl.INT64] = cnt + 1
                    cnt, x_iter = pl.yield_(c2, y)
                return x_iter

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenCallInForRange:
    """Tests for flattening calls in for loop bodies.

    Note: In the current DSL, for loop range expressions use constants/scalars, not calls.
    This test ensures the pass handles for loops correctly.
    """

    def test_nested_calls_in_for_body(self):
        """Test nested calls inside for loop body"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(10):
                    # Nested call in loop body
                    result = pl.mul(pl.add(result, 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(10):
                    t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)
                    result = pl.mul(t__tmp_v0, 2.0)
                return result

        After = passes.flatten_call_expr()(passes.convert_to_ssa()(Before))
        ir.assert_structural_equal(After, NormalizeIR(passes.convert_to_ssa()(Expected)))

    def test_for_with_get_block_idx_in_range(self):
        """Test get_block_idx call in for range expression"""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # get_block_idx() in for range
                for _i in pl.range(pl.tile.get_block_idx()):
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.tile.store(tile, offsets=[0, 0], output_tensor=output)
                return output

        @pl.program
        class Expected:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for _i in pl.range(t__tmp_v0):
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.tile.store(tile, offsets=[0, 0], output_tensor=output)
                return output

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_for_with_calls_in_start_and_step(self):
        """Calls in for-range start and step are both hoisted before the loop.

        The ForStmt visitor (flatten_call_expr_pass.cpp:266-300) visits
        ``start``, ``stop``, ``step`` in that order and extracts any Call into a
        temp. Here ``start = get_block_idx()`` and ``step = get_block_num()``
        are calls (``stop = 10`` is a constant), so two temps are emitted in
        start-then-step order before the loop: ``t__tmp_v0`` for start,
        ``t__tmp_v1`` for step.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for _i in pl.range(pl.tile.get_block_idx(), 10, pl.tile.get_block_num()):
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.tile.store(tile, offsets=[0, 0], output_tensor=output)
                return output

        @pl.program
        class Expected:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                t__tmp_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_num()
                for _i in pl.range(t__tmp_v0, 10, t__tmp_v1):
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.tile.store(tile, offsets=[0, 0], output_tensor=output)
                return output

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenComplexNesting:
    """Tests for complex nesting scenarios."""

    def test_nested_control_flow_with_calls(self):
        """Test nested control flow with multiple call sites"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(5):
                    temp: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(result, 2.0), pl.exp(x))
                    if i > 2:
                        result = temp
                    else:
                        result = pl.add(temp, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(5):
                    t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.mul(result, 2.0)
                    t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                    temp: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v0, t__tmp_v1)
                    if i > 2:
                        result = temp
                    else:
                        result = pl.add(temp, 1.0)
                return result

        After = passes.flatten_call_expr()(passes.convert_to_ssa()(Before))
        ir.assert_structural_equal(After, NormalizeIR(passes.convert_to_ssa()(Expected)))

    def test_multiple_statements_with_nested_calls(self):
        """Test multiple statements with nested calls"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), 1.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(a, 3.0), 4.0)
                c: pl.Tensor[[64], pl.FP32] = pl.add(pl.exp(b), pl.exp(a))
                return c

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                a: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v0, 1.0)
                t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.add(a, 3.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v1, 4.0)
                t__tmp_v2: pl.Tensor[[64], pl.FP32] = pl.exp(b)
                t__tmp_v3: pl.Tensor[[64], pl.FP32] = pl.exp(a)
                c: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v2, t__tmp_v3)
                return c

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenAlreadyFlat:
    """Tests for IR that is already flat (no nested calls)."""

    def test_already_flat_code(self):
        """Test that already flat code is unchanged"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, 2.0)
                c: pl.Tensor[[64], pl.FP32] = pl.exp(b)
                return c

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, 2.0)
                c: pl.Tensor[[64], pl.FP32] = pl.exp(b)
                return c

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_no_calls_at_all(self):
        """Test IR with no operation calls"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenPreservesFuncType:
    """Tests that flatten_call_expr preserves func_type_ on functions."""

    def test_preserve_orchestration_func_type(self):
        """func_type=Orchestration is preserved after flattening."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(x, 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                tmp0: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x, 1.0)
                result: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(tmp0, 2.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_preserve_incore_func_type(self):
        """func_type=InCore is preserved after flattening."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(x, 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                tmp0: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x, 1.0)
                result: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(tmp0, 2.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, Expected)


class TestFlattenPreservesAttrs:
    """Tests that flatten_call_expr preserves Submit.deps when rewriting args.

    A ``pl.submit(..., deps=[tid, ...])`` lowers to an ``ir.Submit`` whose
    typed ``deps_`` field carries the producer TaskIds. FlattenCallExpr
    rebuilds the Submit when it hoists nested arg expressions to fresh
    temporaries; the rebuild must keep ``deps_`` intact, otherwise codegen
    never emits the ``params.set_dependencies(arr, count)`` call and the
    user's explicit dep wiring is silently lost.
    """

    def test_manual_dep_edges_survive_arg_flatten(self):
        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        ctx = _core_passes.PassContext(instruments)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[32], pl.FP32]],
            ) -> pl.Tensor[[32], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.k1, x)
                    # Inline `pl.slice(...)` arg forces FlattenCallExpr to rebuild
                    # the k2 Submit. The deps=[a_tid] kwarg attaches the dep on
                    # Submit::deps_ which must survive the rebuild.
                    _b, _ = pl.submit(self.k2, x, pl.slice(out, [32], [0]), deps=[a_tid])
                return out

        with ctx:
            After = passes.flatten_call_expr()(Prog)
        fn = After.get_function("main")
        assert fn is not None, "main function must exist after flatten_call_expr"

        def find_k2_submit(stmt):
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    r = find_k2_submit(s)
                    if r is not None:
                        return r
            elif isinstance(stmt, ir.RuntimeScopeStmt):
                return find_k2_submit(stmt.body)
            elif isinstance(stmt, ir.AssignStmt):
                v = stmt.value
                if isinstance(v, ir.Submit) and v.op.name == "k2":
                    return v
            return None

        k2_submit = find_k2_submit(fn.body)
        assert k2_submit is not None, "expected the k2 pl.submit in the manual scope"
        assert len(k2_submit.deps) == 1, (
            f"Submit.deps dropped after FlattenCallExpr; got {list(k2_submit.deps)!r}"
        )

    @pytest.mark.xfail(
        reason="suspected bug: FlattenCallExpr does not hoist a nested Call arg of a Submit "
        "(no VisitExpr_(SubmitPtr) override extracts top-level Call args), so the three-address "
        "constraint 'Call arguments cannot be calls' is violated for pl.submit args",
        strict=False,
    )
    def test_nested_call_arg_in_submit_is_flattened(self):
        """A nested Call passed as a Submit arg should be hoisted to a temp.

        The pass doc rule #1 states "Call arguments cannot be calls", and
        ``.claude/rules/pass-submit-awareness.md`` rule #1 requires Submit to be
        walked like Call. The semantically-correct output therefore hoists the
        ``pl.slice(out, [32], [0])`` arg of the ``k2`` submit into ``t__tmp_v0``
        before the submit, exactly as a plain ``Call`` arg would be flattened.

        FlattenCallExpr only overrides ``VisitExpr_(CallPtr)`` and never
        ``VisitExpr_(SubmitPtr)``, so the base mutator's Submit visitor leaves
        the nested ``slice`` Call inline. This Expected encodes the intended
        semantics; the pass currently fails to produce it (suspected bug).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def k2(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[32], pl.FP32]],
            ) -> pl.Tensor[[32], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    _b, _ = pl.submit(self.k2, x, pl.slice(out, [32], [0]))
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def k2(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[32], pl.FP32]],
            ) -> pl.Tensor[[32], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    # Nested slice() Call hoisted to a temp before the submit.
                    t__tmp_v0: pl.Tensor[[32], pl.FP32] = pl.slice(out, [32], [0])
                    _b, _ = pl.submit(self.k2, x, t__tmp_v0)
                return out

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenCallInScopeStmt:
    """Tests for flattening nested calls inside ScopeStmt (pl.incore / pl.at) blocks.

    Verifies that extracted temporaries stay inside the enclosing scope,
    preventing tensor ops from being hoisted outside incore/at blocks.
    Regression tests for issue #941.
    """

    def test_nested_call_inside_at_core_group(self):
        """Test that nested call inside pl.at(level=CORE_GROUP) keeps temp inside the scope."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(x, 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v0, 2.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_multiple_nested_calls_inside_scope(self):
        """Test multiple nested calls inside a scope block preserve ordering."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    a: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), 1.0)
                    b: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(a, 3.0), 4.0)
                return b

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                    a: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v0, 1.0)
                    t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.add(a, 3.0)
                    b: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v1, 4.0)
                return b

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_nested_scope_blocks(self):
        """Test nested scope blocks keep temporaries in the correct inner scope."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(x, 1.0), 2.0)
                with pl.at(level=pl.Level.CORE_GROUP):
                    b: pl.Tensor[[64], pl.FP32] = pl.add(pl.exp(y), 3.0)
                return pl.add(a, b)

        @pl.program
        class Expected:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                a: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v0, 2.0)
                with pl.at(level=pl.Level.CORE_GROUP):
                    t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.exp(y)
                    b: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v1, 3.0)
                t__tmp_v2: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return t__tmp_v2

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_scope_inside_for_loop_with_nested_calls(self):
        """Test scope block inside a for loop with nested calls."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for _i in pl.range(10):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        result = pl.mul(pl.add(result, 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for _i in pl.range(10):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)
                        result = pl.mul(t__tmp_v0, 2.0)
                return result

        After = passes.flatten_call_expr()(passes.convert_to_ssa()(Before))
        ir.assert_structural_equal(After, NormalizeIR(passes.convert_to_ssa()(Expected)))

    def test_deeply_nested_call_inside_scope(self):
        """Test deeply nested calls inside scope: mul(add(exp(x), 1.0), 2.0)"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(pl.exp(x), 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                    t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.add(t__tmp_v0, 1.0)
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v1, 2.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenCallInClusterScope:
    """Tests for flattening nested calls inside a Cluster scope body.

    ``with pl.cluster():`` lowers to a ``ClusterScopeStmt``. The scope visitor
    (flatten_call_expr_pass.cpp:399-404) delegates to the shared
    ``FlattenScopeBody`` helper (lines 364-382), which keeps extracted
    temporaries *inside* the scope body (mirroring the ``pl.at()`` behaviour)
    so execution-context boundaries are preserved. The sibling Spmd scope
    visitor (lines 414-420) reuses the same helper; see the
    spmd-2-statement-body note in the deferred report for why it is not
    exercised here.
    """

    def test_nested_call_inside_cluster_scope(self):
        """Nested call directly in a ``pl.cluster()`` body keeps its temp inside.

        ``mul(add(x, 1.0), 2.0)`` flattens to ``t__tmp_v0 = add(x, 1.0)`` then
        ``result = mul(t__tmp_v0, 2.0)``, both staying inside the cluster scope
        (FlattenScopeBody, lines 364-382 + 399-404).
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(x, 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.cluster():
                    t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v0, 2.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


class TestFlattenCallInReturn:
    """Tests for flattening Call expressions that appear directly inside ReturnStmt.

    Regression tests for issue #1394: `return some_call(...)` (no intermediate
    variable) used to leave a ReturnStmt-wrapped Call untouched, and the
    OrchestrationCodegen's ReturnStmt visitor is a no-op — so the kernel
    dispatch was silently dropped from generated chip orchestration code.
    """

    def test_single_call_in_return(self):
        """`return pl.add(x, 1.0)` is rewritten to bind the call to a temp first."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return pl.add(x, 1.0)

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return t__tmp_v0

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_nested_call_in_return(self):
        """`return pl.mul(pl.add(x, 1.0), 2.0)` extracts both calls into temps."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return pl.mul(pl.add(x, 1.0), 2.0)

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t__tmp_v0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                t__tmp_v1: pl.Tensor[[64], pl.FP32] = pl.mul(t__tmp_v0, 2.0)
                return t__tmp_v1

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
