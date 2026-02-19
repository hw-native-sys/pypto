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
from pypto import ir, passes


def NormalizeIR(program):
    """Normalize IR structure to match flatten_call_expr pass output.

    The pass internally applies normalize_stmt_structure before and
    flatten_single_stmt after the call expression flattening. Expected IR
    from the DSL must go through the same structural transformations for
    assert_structural_equal to succeed.
    """
    return passes.flatten_single_stmt()(passes.normalize_stmt_structure()(program))


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
                _t0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                result: pl.Tensor[[64], pl.FP32] = pl.mul(_t0, 2.0)
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
                _t0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                _t1: pl.Tensor[[64], pl.FP32] = pl.mul(y, 3.0)
                result: pl.Tensor[[64], pl.FP32] = pl.add(_t0, _t1)
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
                _t0: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                _t1: pl.Tensor[[64], pl.FP32] = pl.add(_t0, 1.0)
                result: pl.Tensor[[64], pl.FP32] = pl.mul(_t1, 2.0)
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
                _t0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                result: pl.Tensor[[64], pl.FP32] = pl.add(_t0, x)
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
                _t0: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, _t0)
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
                _t0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                _t1: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(_t0, _t1)
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
                _t0: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                result: pl.Tensor[[64], pl.FP32] = pl.exp(_t0)
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
                        _t0: pl.Tensor[[64], pl.FP32] = pl.mul(result, 2.0)
                        result = pl.add(_t0, 1.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

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
                        result = pl.add(result, 1.0)
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
                _t0: pl.Tensor[[64], pl.FP32] = pl.add(y, 1.0)
                x: pl.Tensor[[64], pl.FP32] = pl.mul(_t0, z)
                for i in pl.range(10):
                    if i > 5:
                        result = pl.add(result, 1.0)
                # Flattened: temp variable for second nested call
                _t1: pl.Tensor[[64], pl.FP32] = pl.add(b, 2.0)
                a: pl.Tensor[[64], pl.FP32] = pl.mul(_t1, c)
                return pl.add(x, a)

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
                if pl.block.get_block_idx() < 10:  # type: ignore[operator]
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.block.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.block.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=output)
                return output

        @pl.program
        class Expected:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                _t0: pl.Scalar[pl.UINT64] = pl.block.get_block_idx()
                if _t0 < 10:  # type: ignore[operator]
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.block.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.block.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=output)
                return output

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
                    _t0: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)
                    result = pl.mul(_t0, 2.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

    def test_for_with_get_block_idx_in_range(self):
        """Test get_block_idx call in for range expression"""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # get_block_idx() in for range
                for i in pl.range(pl.block.get_block_idx()):  # type: ignore[attr-defined,arg-type]  # type: ignore[attr-defined,arg-type]
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.block.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.block.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=output)
                return output

        @pl.program
        class Expected:
            @pl.function
            def main(
                self, a: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                _t0: pl.Scalar[pl.UINT64] = pl.block.get_block_idx()  # type: ignore[attr-defined]
                for i in pl.range(_t0):  # type: ignore[arg-type]
                    tile: pl.Tile[[32, 32], pl.FP32] = pl.block.load(a, offsets=[0, 0], shapes=[32, 32])
                    pl.block.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=output)
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
                    _t0: pl.Tensor[[64], pl.FP32] = pl.mul(result, 2.0)
                    _t1: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                    temp: pl.Tensor[[64], pl.FP32] = pl.add(_t0, _t1)
                    if i > 2:
                        result = temp
                    else:
                        result = pl.add(temp, 1.0)
                return result

        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

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
                _t0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                a: pl.Tensor[[64], pl.FP32] = pl.add(_t0, 1.0)
                _t1: pl.Tensor[[64], pl.FP32] = pl.add(a, 3.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(_t1, 4.0)
                _t2: pl.Tensor[[64], pl.FP32] = pl.exp(b)
                _t3: pl.Tensor[[64], pl.FP32] = pl.exp(a)
                c: pl.Tensor[[64], pl.FP32] = pl.add(_t2, _t3)
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


class TestFlattenWithVerifier:
    """Tests that flattened IR passes verification."""

    def test_flatten_then_verify(self):
        """Test that flattened IR is valid and can be verified"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Nested calls
                result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(pl.exp(x), 1.0), 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                _t0: pl.Tensor[[64], pl.FP32] = pl.exp(x)
                _t1: pl.Tensor[[64], pl.FP32] = pl.add(_t0, 1.0)
                result: pl.Tensor[[64], pl.FP32] = pl.mul(_t1, 2.0)
                return result

        # Flatten the code
        After = passes.flatten_call_expr()(Before)
        ir.assert_structural_equal(After, NormalizeIR(Expected))

        # Verify the flattened code is valid
        verify_pass = passes.run_verifier()
        verified = verify_pass(After)
        assert verified is not None


class TestFlattenPreservesFuncType:
    """Tests that flatten_call_expr preserves func_type_ on functions."""

    def test_preserve_orchestration_func_type(self):
        """Test that func_type is preserved after flattening for Orchestration functions."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(x, 1.0), 2.0)
                return result

        After = passes.flatten_call_expr()(Before)

        after_func = After.get_function("main")
        assert after_func is not None
        assert after_func.func_type == pl.FunctionType.Orchestration

    def test_preserve_incore_func_type(self):
        """Test that func_type is preserved after flattening for InCore functions."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(x, 1.0), 2.0)
                return result

        After = passes.flatten_call_expr()(Before)

        after_func = After.get_function("main")
        assert after_func is not None
        assert after_func.func_type == pl.FunctionType.InCore


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
